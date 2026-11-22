#!/usr/bin/env python3
"""
ObjectDetectra AI - Video Indexing at Speed
Authoritative implementation conforming to the Alethia AI Take-Home Brief.

CLI Entrypoint:
    python run.py --input <folder of .parquet files> --output results.jsonl

Key Invariants:
1. Every decoded frame is processed (temporal sampling and frame skipping strictly prohibited).
2. Presentation timestamps (PTS) relative to the first decoded frame define seconds:
   t_rel = frame.time - first_frame.time, second = floor(t_rel).
3. Contiguous coverage: seconds 0 through floor(t_rel_last) are emitted.
4. An object is present if detected in at least ceil(total_frames / 2) frames of that second.
5. People count is the median across that second's frames, rounded down on halfway values.
6. Target COCO classes: person (0), car (2), dog (16), laptop (63), cell phone (67).
7. Rotation metadata applied prior to inference.
8. Damaged videos never halt the pipeline; completed seconds are preserved and error line emitted.
"""

import argparse
import gzip
import io
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow.parquet as pq

# Target COCO class mappings
TARGET_CLASSES: Dict[int, str] = {
    0: "person",
    2: "car",
    16: "dog",
    63: "laptop",
    67: "cell phone",
}

# Canonical order for structured JSON serialization
CANONICAL_TARGET_ORDER: List[str] = ["person", "car", "dog", "laptop", "cell phone"]
TARGET_CLASS_IDS: List[int] = list(TARGET_CLASSES.keys())


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ObjectDetectra AI - Video Indexing at Speed using YOLO11l"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to folder containing .parquet files (or single .parquet file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output results file (.jsonl or .jsonl.gz)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11l.pt",
        help="Path to official YOLO11l weights (default: yolo11l.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for YOLO inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference ('cuda:0' or 'cpu'). Default: auto-detect",
    )
    return parser.parse_args(args)


# -----------------------------------------------------------------------------
# Aggregation & Mathematical Helpers
# -----------------------------------------------------------------------------

def calculate_median_people(people_counts: List[int]) -> int:
    """
    Calculate the median of person detections across frames in a second.
    Rounds down when the median falls halfway between two numbers.

    Specification:
        - For odd-length lists: middle value.
        - For even-length lists: (middle_left + middle_right) // 2 (exact floor).
    """
    if not people_counts:
        return 0
    s = sorted(people_counts)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


def aggregate_second_records(
    total_frames: int,
    class_counts: Dict[str, int],
    people_counts: List[int],
) -> Tuple[List[str], int]:
    """
    Aggregate per-frame statistics for a single second.

    Returns:
        (objects, people)
        - objects: target classes appearing in at least ceil(total_frames / 2) frames.
        - people: median people count, rounded down on halfway values.
    """
    if total_frames == 0:
        return [], 0

    # Object present if in at least half the frames: count >= ceil(total_frames / 2)
    # Algebraically equivalent to: count * 2 >= total_frames
    present_objects = [
        cls_name
        for cls_name in CANONICAL_TARGET_ORDER
        if class_counts.get(cls_name, 0) * 2 >= total_frames
    ]
    people_median = calculate_median_people(people_counts)
    return present_objects, people_median


def finalize_video_seconds(
    video_id: str,
    seconds_data: Dict[int, Dict[str, Any]],
    last_second: int,
) -> List[Dict[str, Any]]:
    """
    Generate contiguous second records from second 0 through last_second inclusive.
    Empty seconds (or gaps) yield objects=[] and people=0.
    """
    records: List[Dict[str, Any]] = []
    if last_second < 0:
        return records

    for s in range(last_second + 1):
        if s in seconds_data and seconds_data[s]["total_frames"] > 0:
            sec_entry = seconds_data[s]
            objects, people = aggregate_second_records(
                sec_entry["total_frames"],
                sec_entry["class_counts"],
                sec_entry["people_counts"],
            )
        else:
            objects, people = [], 0

        records.append({
            "video": video_id,
            "second": s,
            "objects": objects,
            "people": people,
        })
    return records


# -----------------------------------------------------------------------------
# File & Parquet Utilities
# -----------------------------------------------------------------------------

def discover_parquet_files(input_path: Path) -> List[Path]:
    """
    Discover all Parquet files under input_path without duplicates.
    Deterministic sorting by resolved path.
    """
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() == ".parquet":
            return [input_path]
        return []

    if not input_path.is_dir():
        return []

    discovered = set()
    parquet_files: List[Path] = []
    # Scan recursively
    for candidate in sorted(input_path.rglob("*.parquet")):
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in discovered:
            discovered.add(resolved)
            parquet_files.append(resolved)

    return sorted(parquet_files)


def format_video_id(parquet_path: Path, row_idx: int) -> str:
    """
    Video identifier: Parquet filename without folder, a colon, and 0-based row number.
    Example: 'train-00003-of-01357.parquet:4'
    """
    return f"{parquet_path.name}:{row_idx}"


class OutputWriter:
    """
    Incremental output writer supporting uncompressed JSONL and gzip-compressed JSONL.
    Flushes records incrementally to safeguard partial runs.
    """
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_gzip = output_path.suffix.lower() == ".gz"
        self._fh = None

    def __enter__(self):
        if self.is_gzip:
            self._fh = gzip.open(self.output_path, "wt", encoding="utf-8")
        else:
            self._fh = open(self.output_path, "w", encoding="utf-8")
        return self

    def write_record(self, record: Dict[str, Any]) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# -----------------------------------------------------------------------------
# Video Decoding & Rotation
# -----------------------------------------------------------------------------

def apply_rotation(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate BGR NumPy frame by angle (degrees clockwise) without OpenCV dependency.
    """
    angle = angle % 360
    if angle == 90:
        return np.ascontiguousarray(np.rot90(img, k=3))
    elif angle == 180:
        return np.ascontiguousarray(np.rot90(img, k=2))
    elif angle == 270:
        return np.ascontiguousarray(np.rot90(img, k=1))
    return img


def get_video_rotation(video_stream) -> int:
    """
    Extract container/stream rotation tag if present.
    """
    # 1. Stream metadata
    if hasattr(video_stream, "metadata") and video_stream.metadata:
        rotate_tag = video_stream.metadata.get("rotate")
        if rotate_tag is not None:
            try:
                return int(rotate_tag) % 360
            except ValueError:
                pass

    # 2. Side data displaymatrix
    if hasattr(video_stream, "side_data"):
        try:
            for sd in video_stream.side_data:
                if getattr(sd, "type", None) == "DISPLAYMATRIX":
                    if hasattr(sd, "rotation"):
                        return int(sd.rotation) % 360
        except Exception:
            pass

    return 0


def decode_video_frames(
    mp4_bytes: bytes,
) -> Iterator[Tuple[np.ndarray, float, int]]:
    """
    Decodes all frames from MP4 bytes in memory using PyAV.
    Preserves exact presentation timestamps (PTS) relative to the first decoded frame.
    Guarantees container closure via try/finally upon normal exit, exception, or early generator close.

    Yields:
        (frame_bgr, relative_timestamp, second_idx)
    """
    try:
        import av
    except ImportError:
        raise RuntimeError("PyAV ('av') is required for video decoding but not installed.")

    bio = io.BytesIO(mp4_bytes)
    container = None
    try:
        container = av.open(bio, format="mp4")

        if not container.streams.video:
            raise ValueError("No video stream found in MP4 container.")

        video_stream = container.streams.video[0]
        # Threading optimization for software decoding
        video_stream.thread_type = "AUTO"

        rotation = get_video_rotation(video_stream)
        first_frame_ts: Optional[float] = None

        for frame in container.decode(video=0):
            # Determine exact presentation timestamp
            frame_time = frame.time
            if frame_time is None:
                if frame.pts is not None and video_stream.time_base is not None:
                    frame_time = float(frame.pts * video_stream.time_base)
                else:
                    raise ValueError("Frame missing presentation timestamp (PTS).")

            if first_frame_ts is None:
                first_frame_ts = frame_time

            relative_ts = max(0.0, float(frame_time - first_frame_ts))
            second_idx = math.floor(relative_ts)

            bgr = frame.to_ndarray(format="bgr24")
            if rotation != 0:
                bgr = apply_rotation(bgr, rotation)

            yield bgr, relative_ts, second_idx
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Detection Engine
# -----------------------------------------------------------------------------

def load_yolo_model(weights_path: str, device: Optional[str] = None):
    """
    Load official YOLO11l weights with Ultralytics defaults.
    """
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Ultralytics and PyTorch are required for detection.")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[YOLO] Loading model from '{weights_path}' on device '{device}'...")
    model = YOLO(weights_path)
    return model, device


def run_batch_inference(
    model: Any,
    frames: List[np.ndarray],
    device: str,
) -> List[Tuple[Set[str], int]]:
    """
    Execute batched inference on BGR frames with mandatory take-home parameters:
      - imgsz: 640
      - conf: 0.4
      - iou: 0.7
      - classes: [0, 2, 16, 63, 67]
      - Default inference precision (no forced fp16 until benchmarked)
      - Ultralytics default letterboxing and class-aware NMS

    Returns:
        List of (detected_target_classes_in_frame, person_count_in_frame)
    """
    if not frames:
        return []

    results = model.predict(
        frames,
        imgsz=640,
        conf=0.4,
        iou=0.7,
        classes=TARGET_CLASS_IDS,
        device=device,
        verbose=False,
    )

    frame_outputs: List[Tuple[Set[str], int]] = []
    for res in results:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            frame_outputs.append((set(), 0))
            continue

        cls_ids = [int(c) for c in boxes.cls.tolist()]
        person_count = cls_ids.count(0)
        detected_classes = {
            TARGET_CLASSES[cid] for cid in cls_ids if cid in TARGET_CLASSES
        }
        frame_outputs.append((detected_classes, person_count))

    return frame_outputs


# -----------------------------------------------------------------------------
# Video Processing Pipeline
# -----------------------------------------------------------------------------

def process_single_video(
    video_id: str,
    mp4_bytes: Optional[bytes],
    model: Any,
    device: str,
    batch_size: int,
    writer: OutputWriter,
) -> None:
    """
    Process a single video's MP4 payload end-to-end.
    Maintains robustness:
      - Emits completed seconds.
      - Emits error record on decoder or container failure without halting pipeline.
    """
    if mp4_bytes is None or len(mp4_bytes) == 0:
        writer.write_record({"video": video_id, "error": "empty or missing mp4 bytes"})
        writer.flush()
        return

    seconds_data: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"total_frames": 0, "class_counts": Counter(), "people_counts": []}
    )
    last_second = -1
    highest_emitted_second = -1

    batch_frames: List[np.ndarray] = []
    batch_seconds: List[int] = []

    def _flush_batch():
        nonlocal batch_frames, batch_seconds
        if not batch_frames:
            return
        outputs = run_batch_inference(model, batch_frames, device)
        for sec, (classes, p_count) in zip(batch_seconds, outputs):
            seconds_data[sec]["total_frames"] += 1
            seconds_data[sec]["people_counts"].append(p_count)
            for c in classes:
                seconds_data[sec]["class_counts"][c] += 1
        batch_frames = []
        batch_seconds = []

    try:
        frame_iter = decode_video_frames(mp4_bytes)
        for frame_bgr, rel_ts, sec_idx in frame_iter:
            last_second = max(last_second, sec_idx)
            batch_frames.append(frame_bgr)
            batch_seconds.append(sec_idx)

            if len(batch_frames) >= batch_size:
                _flush_batch()

        # Flush any remaining decoded frames
        _flush_batch()

        if last_second < 0:
            # Container opened but yielded 0 decodable frames
            writer.write_record({"video": video_id, "error": "could not decode video"})
            writer.flush()
            return

        # Video completed successfully: write all seconds 0 .. last_second
        records = finalize_video_seconds(video_id, seconds_data, last_second)
        for r in records:
            writer.write_record(r)
        writer.flush()

    except Exception as exc:
        # Partial failure or decode error mid-video:
        # 1. Process any frames currently sitting in batch
        try:
            _flush_batch()
        except Exception:
            pass

        # 2. Flush only fully completed seconds prior to the failure.
        # An incomplete final second (during which failure occurred) must NOT be emitted.
        # Completed seconds are those strictly less than last_second: [0 .. last_second - 1].
        completed_second_limit = last_second - 1
        if completed_second_limit >= 0:
            records = finalize_video_seconds(video_id, seconds_data, completed_second_limit)
            for r in records:
                writer.write_record(r)

        # 3. Emit required error record
        err_msg = str(exc).strip() or "could not decode video"
        writer.write_record({"video": video_id, "error": err_msg})
        writer.flush()


def process_parquet_file(
    parquet_path: Path,
    model: Any,
    device: str,
    batch_size: int,
    writer: OutputWriter,
) -> int:
    """
    Process all videos in a single Parquet file row by row.
    Uses PyArrow incremental record batch streaming to avoid loading full file into RAM.
    """
    print(f"[PIPELINE] Ingesting: {parquet_path.name}")
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception as e:
        print(f"[ERROR] Failed to open Parquet file {parquet_path.name}: {e}", file=sys.stderr)
        # Emit an error record representing the file-level failure
        writer.write_record({"video": f"{parquet_path.name}:0", "error": f"corrupted parquet file: {e}"})
        writer.flush()
        return 0

    total_rows = pf.metadata.num_rows
    processed_count = 0
    current_row_idx = 0

    # Stream batch by batch reading only the 'mp4' column
    for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
        mp4_col = batch.column("mp4")
        for val in mp4_col:
            video_id = format_video_id(parquet_path, current_row_idx)
            mp4_bytes = val.as_py()
            process_single_video(
                video_id=video_id,
                mp4_bytes=mp4_bytes,
                model=model,
                device=device,
                batch_size=batch_size,
                writer=writer,
            )
            current_row_idx += 1
            processed_count += 1

    return processed_count


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------

def main(args: Optional[List[str]] = None) -> int:
    parsed_args = parse_args(args)
    input_path = Path(parsed_args.input)
    output_path = Path(parsed_args.output)
    weights_path = parsed_args.weights
    batch_size = parsed_args.batch_size
    device = parsed_args.device

    # 1. Discover Parquet files
    parquet_files = discover_parquet_files(input_path)
    if not parquet_files:
        print(f"[FATAL] No .parquet files discovered at: {input_path}", file=sys.stderr)
        return 1

    print(f"[PIPELINE] Discovered {len(parquet_files)} Parquet file(s) without duplicates.")

    # 2. Verify / Load YOLO11l weights
    if not Path(weights_path).exists():
        print(
            f"[FATAL] YOLO11l weights not found at '{weights_path}'.\n"
            f"        Run 'python prepare.py' first to download official weights.",
            file=sys.stderr,
        )
        return 1

    model, resolved_device = load_yolo_model(weights_path, device=device)

    # 3. Process files and stream output
    total_videos_processed = 0
    with OutputWriter(output_path) as writer:
        for p_file in parquet_files:
            count = process_parquet_file(
                parquet_path=p_file,
                model=model,
                device=resolved_device,
                batch_size=batch_size,
                writer=writer,
            )
            total_videos_processed += count

    print(f"[PIPELINE] Completed indexing {total_videos_processed} videos.")
    print(f"[PIPELINE] Output saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
