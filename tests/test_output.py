#!/usr/bin/env python3
"""
Unit and integration tests for ObjectDetectra AI pipeline components.
Covers:
- Correct video identifiers
- Gzip output vs standard JSONL output
- Duplicate-free Parquet discovery
- Error record handling and partial output preservation
- End-to-end pipeline execution with synthetic Parquet video data
"""

import gzip
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from run import (
    OutputWriter,
    discover_parquet_files,
    format_video_id,
    process_single_video,
    process_parquet_file,
    TARGET_CLASSES,
)


class DummyBox:
    def __init__(self, cls_list):
        import torch
        self.cls = torch.tensor(cls_list, dtype=torch.float32)

    def __len__(self):
        return len(self.cls)


class DummyResult:
    def __init__(self, cls_list):
        self.boxes = DummyBox(cls_list)


class MockYOLOModel:
    """Mock YOLO model returning controlled detections for synthetic testing."""
    def __init__(self):
        self.call_count = 0

    def predict(
        self,
        frames,
        imgsz=640,
        conf=0.4,
        iou=0.7,
        classes=None,
        device=None,
        verbose=False,
    ):
        self.call_count += 1
        results = []
        for _ in frames:
            # Simulate 1 person (cls 0) and 1 laptop (cls 63)
            results.append(DummyResult([0, 63]))
        return results


class TestOutputAndDiscovery(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_video_identifier_formatting(self):
        """
        Specification:
        video is the Parquet file name without its folder, a colon,
        and the row number in that file, counting from 0.
        """
        p = Path("/datasets/finevideo/data/train-00003-of-01357.parquet")
        self.assertEqual(format_video_id(p, 4), "train-00003-of-01357.parquet:4")
        self.assertEqual(format_video_id(p, 0), "train-00003-of-01357.parquet:0")
        self.assertEqual(format_video_id(Path("train-00049-of-01357.parquet"), 12), "train-00049-of-01357.parquet:12")

    def test_duplicate_free_parquet_discovery(self):
        """
        Specification:
        Discover Parquet files without duplicates, deterministically sorted.
        """
        # Create directory structure
        sub_a = self.test_path / "sub_a"
        sub_b = self.test_path / "sub_b"
        sub_a.mkdir()
        sub_b.mkdir()

        f1 = self.test_path / "train-00001.parquet"
        f2 = sub_a / "train-00002.parquet"
        f3 = sub_b / "train-00003.parquet"
        non_parquet = sub_a / "notes.txt"

        for f in [f1, f2, f3, non_parquet]:
            f.write_text("dummy")

        discovered = discover_parquet_files(self.test_path)
        self.assertEqual(len(discovered), 3)
        self.assertEqual(discovered, sorted([f1.resolve(), f2.resolve(), f3.resolve()]))
        self.assertNotIn(non_parquet.resolve(), discovered)

    def test_single_file_discovery(self):
        """Test discover_parquet_files when a single file path is passed."""
        f1 = self.test_path / "single.parquet"
        f1.write_text("dummy")
        discovered = discover_parquet_files(f1)
        self.assertEqual(discovered, [f1.resolve()])

    def test_output_writer_uncompressed_jsonl(self):
        """Verify uncompressed JSONL writing and incremental flushing."""
        out_file = self.test_path / "results.jsonl"
        records = [
            {"video": "file.parquet:0", "second": 0, "objects": ["person"], "people": 1},
            {"video": "file.parquet:0", "second": 1, "objects": [], "people": 0},
        ]

        with OutputWriter(out_file) as writer:
            for r in records:
                writer.write_record(r)
            writer.flush()

        self.assertTrue(out_file.exists())
        with open(out_file, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["video"], "file.parquet:0")
        self.assertEqual(lines[0]["second"], 0)
        self.assertEqual(lines[0]["objects"], ["person"])
        self.assertEqual(lines[0]["people"], 1)

    def test_output_writer_gzip_jsonl(self):
        """Verify gzip compressed output writing and integrity."""
        out_file = self.test_path / "results.jsonl.gz"
        record = {"video": "file.parquet:0", "second": 12, "objects": ["car"], "people": 0}

        with OutputWriter(out_file) as writer:
            writer.write_record(record)
            writer.flush()

        self.assertTrue(out_file.exists())
        with gzip.open(out_file, "rt", encoding="utf-8") as gz_in:
            line = gz_in.readline()
            data = json.loads(line)

        self.assertEqual(data["second"], 12)
        self.assertEqual(data["objects"], ["car"])

    def test_empty_or_missing_mp4_bytes(self):
        """Damaged video with empty or missing bytes must write error record and not crash."""
        out_file = self.test_path / "results.jsonl"
        writer = OutputWriter(out_file)
        writer.__enter__()

        process_single_video(
            video_id="test.parquet:0",
            mp4_bytes=None,
            model=None,
            device="cpu",
            batch_size=16,
            writer=writer,
        )
        writer.__exit__(None, None, None)

        with open(out_file, "r", encoding="utf-8") as fh:
            line = json.loads(fh.readline())
        self.assertEqual(line["video"], "test.parquet:0")
        self.assertIn("error", line)

    def test_mid_video_failure_recovery(self):
        """
        Specification:
        If a video fails partway through, keep the lines you already wrote
        for fully completed seconds and add one error line.
        Incomplete final second must NOT be emitted.
        """
        out_file = self.test_path / "results.jsonl"
        writer = OutputWriter(out_file)
        writer.__enter__()

        # Generator that yields frames for second 0, second 1, and halfway through second 2
        def failing_frame_generator(mp4_bytes):
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            # Second 0 (2 frames - fully completed)
            yield dummy_img, 0.1, 0
            yield dummy_img, 0.6, 0
            # Second 1 (2 frames - fully completed)
            yield dummy_img, 1.2, 1
            yield dummy_img, 1.8, 1
            # Second 2 (2 frames - incomplete, crashes before second 2 finishes)
            yield dummy_img, 2.1, 2
            yield dummy_img, 2.4, 2
            # Crash halfway through second 2
            raise RuntimeError("Corrupt H.264 bitstream halfway through second 2")

        mock_model = MockYOLOModel()

        with patch("run.decode_video_frames", side_effect=failing_frame_generator):
            process_single_video(
                video_id="corrupt.parquet:5",
                mp4_bytes=b"dummy_bytes",
                model=mock_model,
                device="cpu",
                batch_size=2,
                writer=writer,
            )

        writer.__exit__(None, None, None)

        with open(out_file, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh]

        # Should contain second 0, second 1, and 1 error line. Second 2 MUST NOT be emitted!
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0]["second"], 0)
        self.assertEqual(lines[0]["video"], "corrupt.parquet:5")
        self.assertEqual(lines[1]["second"], 1)
        self.assertEqual(lines[1]["video"], "corrupt.parquet:5")
        self.assertIn("error", lines[2])
        self.assertEqual(lines[2]["video"], "corrupt.parquet:5")
        self.assertIn("Corrupt H.264", lines[2]["error"])

        # Explicitly assert that second 2 is absent
        emitted_seconds = [l["second"] for l in lines if "second" in l]
        self.assertNotIn(2, emitted_seconds)
        self.assertEqual(emitted_seconds, [0, 1])

    def test_mid_video_failure_during_second_0(self):
        """
        If a video fails halfway through second 0, no seconds were fully completed.
        Zero second records should be emitted, and exactly one error record should be emitted.
        """
        out_file = self.test_path / "results_sec0_fail.jsonl"
        with OutputWriter(out_file) as writer:
            def failing_at_sec0(mp4_bytes):
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                yield dummy_img, 0.1, 0
                yield dummy_img, 0.4, 0
                raise RuntimeError("Crash halfway through second 0")

            mock_model = MockYOLOModel()
            with patch("run.decode_video_frames", side_effect=failing_at_sec0):
                process_single_video(
                    video_id="corrupt_early.parquet:1",
                    mp4_bytes=b"dummy_bytes",
                    model=mock_model,
                    device="cpu",
                    batch_size=2,
                    writer=writer,
                )

        with open(out_file, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh]

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["video"], "corrupt_early.parquet:1")
        self.assertIn("error", lines[0])
        self.assertIn("Crash halfway through second 0", lines[0]["error"])

    def test_pyav_container_closed_on_exception_and_exit(self):
        """
        Verify decode_video_frames() guarantees container.close() via try/finally
        even when decoding raises an exception.
        """
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.metadata = {}
        mock_stream.time_base = 1.0 / 30.0
        mock_container.streams.video = [mock_stream]

        # Generator for container.decode that yields 1 frame then raises an exception
        def failing_decode(video=0):
            frame = MagicMock()
            frame.time = 0.0
            frame.to_ndarray.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            yield frame
            raise RuntimeError("Decode failed mid-stream")

        mock_container.decode.side_effect = failing_decode
        mock_av.open.return_value = mock_container

        with patch.dict("sys.modules", {"av": mock_av}):
            from run import decode_video_frames
            gen = decode_video_frames(b"fake_mp4_bytes")
            # First item yields fine
            first_item = next(gen)
            self.assertEqual(first_item[2], 0)
            # Second item raises exception
            with self.assertRaises(RuntimeError):
                next(gen)

        # Confirm container.close() was called in finally block
        mock_container.close.assert_called_once()

    def test_synthetic_parquet_pipeline_end_to_end(self):
        """
        End-to-end integration test creating a synthetic Parquet table,
        iterating with PyArrow, decoding frames, running mock inference,
        and validating the serialized JSON Lines output.
        """
        # 1. Create synthetic Parquet file
        parquet_file = self.test_path / "train-00000-of-01357.parquet"
        dummy_mp4_bytes = b"fake_mp4_stream_data"
        table = pa.Table.from_pydict({
            "mp4": [dummy_mp4_bytes, dummy_mp4_bytes],
            "json": ['{"meta": 1}', '{"meta": 2}'],
        })
        pq.write_table(table, parquet_file)

        # 2. Generator that yields 4 frames across 2 seconds (0 and 1)
        def synthetic_frame_generator(mp4_bytes):
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            yield dummy_img, 0.2, 0
            yield dummy_img, 0.8, 0
            yield dummy_img, 1.3, 1
            yield dummy_img, 1.7, 1

        mock_model = MockYOLOModel()
        out_file = self.test_path / "results.jsonl"

        with OutputWriter(out_file) as writer:
            with patch("run.decode_video_frames", side_effect=synthetic_frame_generator):
                count = process_parquet_file(
                    parquet_path=parquet_file,
                    model=mock_model,
                    device="cpu",
                    batch_size=2,
                    writer=writer,
                )

        self.assertEqual(count, 2)  # 2 videos processed

        # 3. Verify output contents
        with open(out_file, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh]

        # Video 0: second 0, second 1. Video 1: second 0, second 1 -> 4 records
        self.assertEqual(len(lines), 4)

        # Check records for video 0
        self.assertEqual(lines[0]["video"], "train-00000-of-01357.parquet:0")
        self.assertEqual(lines[0]["second"], 0)
        self.assertEqual(lines[0]["objects"], ["person", "laptop"])
        self.assertEqual(lines[0]["people"], 1)

        self.assertEqual(lines[1]["video"], "train-00000-of-01357.parquet:0")
        self.assertEqual(lines[1]["second"], 1)

        # Check records for video 1
        self.assertEqual(lines[2]["video"], "train-00000-of-01357.parquet:1")
        self.assertEqual(lines[2]["second"], 0)
        self.assertEqual(lines[3]["video"], "train-00000-of-01357.parquet:1")
        self.assertEqual(lines[3]["second"], 1)

    def test_real_pyav_synthetic_mp4(self):
        """
        When PyAV is installed (e.g. Kaggle / Ubuntu CI), encode and decode
        a real synthetic MP4 container in memory to validate PTS and decoder logic.
        """
        try:
            import av
        except ImportError:
            self.skipTest("PyAV not installed in this environment; skipping real MP4 container test.")

        # Encode a 10-frame synthetic MP4 container in memory
        bio = io.BytesIO()
        out_container = av.open(bio, mode="w", format="mp4")
        stream = out_container.add_stream("h264", rate=5)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"

        for i in range(10):
            frame = av.VideoFrame(64, 64, "rgb24")
            for plane in frame.planes:
                plane.update(b"\x00" * plane.buffer_size)
            frame.pts = i
            for packet in stream.encode(frame):
                out_container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            out_container.mux(packet)
        out_container.close()

        mp4_bytes = bio.getvalue()
        self.assertGreater(len(mp4_bytes), 0)

        # Now decode using run.decode_video_frames
        from run import decode_video_frames
        decoded = list(decode_video_frames(mp4_bytes))
        self.assertEqual(len(decoded), 10)
        # Check first frame relative ts is 0.0
        self.assertEqual(decoded[0][1], 0.0)
        self.assertEqual(decoded[0][2], 0)
        # Check last frame second is 1 (at rate 5, frames 0-4 are second 0, frames 5-9 are second 1)
        self.assertEqual(decoded[-1][2], 1)


if __name__ == "__main__":
    unittest.main()

