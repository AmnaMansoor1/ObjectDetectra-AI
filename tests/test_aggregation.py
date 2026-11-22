#!/usr/bin/env python3
"""
Unit tests for ObjectDetectra AI per-second aggregation and mathematical invariants.
Covers:
- Object presence exactly at 50%
- Object presence below 50%
- Odd frame counts
- Even frame counts
- Halfway median rounding down
- Empty seconds
- Contiguous second sequence generation
"""

import unittest
from run import (
    calculate_median_people,
    aggregate_second_records,
    finalize_video_seconds,
    CANONICAL_TARGET_ORDER,
)


class TestAggregationLogic(unittest.TestCase):

    def test_median_empty(self):
        """Empty list should return 0 people."""
        self.assertEqual(calculate_median_people([]), 0)

    def test_median_odd_frame_counts(self):
        """Odd number of frames: median is exact center value."""
        self.assertEqual(calculate_median_people([3]), 3)
        self.assertEqual(calculate_median_people([1, 2, 5]), 2)
        self.assertEqual(calculate_median_people([5, 1, 2]), 2)
        self.assertEqual(calculate_median_people([0, 0, 1]), 0)
        self.assertEqual(calculate_median_people([10, 20, 30, 40, 50]), 30)

    def test_median_even_frame_counts_exact(self):
        """Even frame counts where average is an integer."""
        self.assertEqual(calculate_median_people([2, 4]), 3)
        self.assertEqual(calculate_median_people([1, 1]), 1)
        self.assertEqual(calculate_median_people([2, 2, 4, 4]), 3)

    def test_halfway_median_rounding_down(self):
        """
        Specification: Median rounded down when it falls halfway between two numbers.
        Python's round() uses banker's rounding to even, which rounds 3.5 to 4.
        Our implementation MUST round 3.5 down to 3, 2.5 down to 2, 1.5 down to 1.
        """
        # Halfway 0.5 -> 0
        self.assertEqual(calculate_median_people([0, 1]), 0)
        # Halfway 1.5 -> 1
        self.assertEqual(calculate_median_people([1, 2]), 1)
        # Halfway 2.5 -> 2
        self.assertEqual(calculate_median_people([2, 3]), 2)
        # Halfway 3.5 -> 3 (Banker's round would give 4)
        self.assertEqual(calculate_median_people([3, 4]), 3)
        # Halfway 2.5 from distant spread -> 2
        self.assertEqual(calculate_median_people([0, 5]), 2)
        # 4 frames: middle elements 2, 3 -> halfway 2.5 -> 2
        self.assertEqual(calculate_median_people([1, 2, 3, 10]), 2)
        # 4 frames: middle elements 3, 4 -> halfway 3.5 -> 3
        self.assertEqual(calculate_median_people([1, 3, 4, 8]), 3)

    def test_object_presence_exactly_50_percent(self):
        """
        Specification: An object is present if detected in at least half of that second's frames.
        count * 2 >= total_frames
        """
        # 2 frames, 1 detection = 50% -> Included
        objs, _ = aggregate_second_records(
            total_frames=2,
            class_counts={"person": 1, "car": 0},
            people_counts=[1, 0],
        )
        self.assertIn("person", objs)
        self.assertNotIn("car", objs)

        # 4 frames, 2 detections = 50% -> Included
        objs, _ = aggregate_second_records(
            total_frames=4,
            class_counts={"laptop": 2, "dog": 1},
            people_counts=[0, 0, 0, 0],
        )
        self.assertIn("laptop", objs)
        self.assertNotIn("dog", objs)

        # 6 frames, 3 detections = 50% -> Included
        objs, _ = aggregate_second_records(
            total_frames=6,
            class_counts={"cell phone": 3},
            people_counts=[0] * 6,
        )
        self.assertIn("cell phone", objs)

    def test_object_presence_below_50_percent(self):
        """Objects detected in fewer than 50% of frames must NOT be included."""
        # 4 frames, 1 detection = 25% < 50% -> Excluded
        objs, _ = aggregate_second_records(
            total_frames=4,
            class_counts={"car": 1},
            people_counts=[0, 0, 0, 0],
        )
        self.assertNotIn("car", objs)

        # 3 frames, 1 detection = 33.3% < 50% -> Excluded (needs ceil(3/2) = 2)
        objs, _ = aggregate_second_records(
            total_frames=3,
            class_counts={"person": 1},
            people_counts=[1, 0, 0],
        )
        self.assertNotIn("person", objs)

        # 5 frames, 2 detections = 40% < 50% -> Excluded (needs ceil(5/2) = 3)
        objs, _ = aggregate_second_records(
            total_frames=5,
            class_counts={"dog": 2},
            people_counts=[0] * 5,
        )
        self.assertNotIn("dog", objs)

    def test_object_presence_above_50_percent(self):
        """Objects detected in more than 50% of frames must be included."""
        # 3 frames, 2 detections = 66.7% >= 50% -> Included
        objs, _ = aggregate_second_records(
            total_frames=3,
            class_counts={"person": 2},
            people_counts=[1, 1, 0],
        )
        self.assertIn("person", objs)

        # 5 frames, 3 detections = 60% >= 50% -> Included
        objs, _ = aggregate_second_records(
            total_frames=5,
            class_counts={"laptop": 3},
            people_counts=[0] * 5,
        )
        self.assertIn("laptop", objs)

    def test_empty_second_record(self):
        """Second with 0 total frames must output empty objects and 0 people."""
        objs, people = aggregate_second_records(
            total_frames=0,
            class_counts={},
            people_counts=[],
        )
        self.assertEqual(objs, [])
        self.assertEqual(people, 0)

    def test_finalize_video_seconds_contiguous_coverage(self):
        """
        Specification: Write a line for every second up to and including
        the second that holds the last frame. Missing seconds must be filled.
        """
        video_id = "test.parquet:0"
        # Simulate video where frames exist in second 0 and second 3 (seconds 1 and 2 are gaps)
        seconds_data = {
            0: {
                "total_frames": 2,
                "class_counts": {"person": 2},
                "people_counts": [1, 1],
            },
            3: {
                "total_frames": 2,
                "class_counts": {"car": 2},
                "people_counts": [0, 0],
            },
        }
        records = finalize_video_seconds(video_id, seconds_data, last_second=3)

        self.assertEqual(len(records), 4)  # seconds 0, 1, 2, 3
        # Second 0
        self.assertEqual(records[0]["second"], 0)
        self.assertEqual(records[0]["objects"], ["person"])
        self.assertEqual(records[0]["people"], 1)

        # Second 1 (gap)
        self.assertEqual(records[1]["second"], 1)
        self.assertEqual(records[1]["objects"], [])
        self.assertEqual(records[1]["people"], 0)

        # Second 2 (gap)
        self.assertEqual(records[2]["second"], 2)
        self.assertEqual(records[2]["objects"], [])
        self.assertEqual(records[2]["people"], 0)

        # Second 3
        self.assertEqual(records[3]["second"], 3)
        self.assertEqual(records[3]["objects"], ["car"])
        self.assertEqual(records[3]["people"], 0)

        for r in records:
            self.assertEqual(r["video"], video_id)

    def test_canonical_target_order_preservation(self):
        """Detected objects must preserve canonical order."""
        objs, _ = aggregate_second_records(
            total_frames=2,
            class_counts={"cell phone": 2, "person": 2, "car": 2},
            people_counts=[1, 1],
        )
        self.assertEqual(objs, ["person", "car", "cell phone"])


if __name__ == "__main__":
    unittest.main()
