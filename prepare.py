#!/usr/bin/env python3
"""
ObjectDetectra AI - Environment Preparation Script
Downloads official Ultralytics YOLO11l weights (yolo11l.pt) prior to offline execution.

Evaluation Context:
This script is executed once before the timed run in an environment with internet access.
The subsequent evaluation run (run.py) operates completely offline.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

OFFICIAL_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
)
DEFAULT_WEIGHTS_NAME = "yolo11l.pt"
MIN_EXPECTED_SIZE_BYTES = 40 * 1024 * 1024  # YOLO11l is approximately 52 MB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare environment and download YOLO11l pretrained weights."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS_NAME,
        help="Target filename or path for yolo11l.pt (default: yolo11l.pt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if weights already exist.",
    )
    return parser.parse_args()


def download_weights(target_path: Path, force: bool = False) -> None:
    if target_path.exists() and not force:
        size = target_path.stat().st_size
        if size >= MIN_EXPECTED_SIZE_BYTES:
            print(f"[PREPARE] Weights already present at {target_path} ({size / (1024 * 1024):.1f} MB). Skipping download.")
            return
        else:
            print(f"[PREPARE] Existing weights file {target_path} is incomplete ({size} bytes). Re-downloading...")

    print(f"[PREPARE] Downloading official YOLO11l weights from:")
    print(f"          {OFFICIAL_WEIGHTS_URL}")
    print(f"          Target: {target_path.resolve()}")

    try:
        # Attempt via ultralytics first if installed
        try:
            from ultralytics import YOLO
            print("[PREPARE] Using Ultralytics loader to fetch official yolo11l.pt...")
            model = YOLO(DEFAULT_WEIGHTS_NAME)
            # Ensure file is saved at target_path
            src_file = Path(DEFAULT_WEIGHTS_NAME)
            if src_file.resolve() != target_path.resolve() and src_file.exists():
                import shutil
                shutil.copy2(src_file, target_path)
            print(f"[PREPARE] Successfully downloaded via Ultralytics.")
            return
        except ImportError:
            print("[PREPARE] Ultralytics not yet installed or importable; falling back to direct HTTPS download...")

        # Direct HTTPS download fallback
        def _reporthook(block_num: int, block_size: int, total_size: int):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, (downloaded / total_size) * 100.0)
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r[PREPARE] Progress: {percent:5.1f}% ({mb_down:.1f}/{mb_tot:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(OFFICIAL_WEIGHTS_URL, str(target_path), reporthook=_reporthook)
        print("\n[PREPARE] Download completed.")

        size = target_path.stat().st_size
        if size < MIN_EXPECTED_SIZE_BYTES:
            raise RuntimeError(f"Downloaded weights file is too small ({size} bytes). Download may be corrupted.")

        print(f"[PREPARE] Verified {target_path.name} ({size / (1024 * 1024):.1f} MB). Ready for offline execution.")

    except Exception as e:
        print(f"\n[PREPARE ERROR] Failed to download weights: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    download_weights(weights_path, force=args.force)
    print("[PREPARE] Preparation step finished successfully.")


if __name__ == "__main__":
    main()
