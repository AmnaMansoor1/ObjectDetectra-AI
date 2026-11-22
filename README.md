# ObjectDetectra AI: Video Indexing at Speed

High-throughput, mathematically rigorous video indexing pipeline designed to extract fine-grained object and people statistics from large-scale Parquet datasets using official Ultralytics YOLO11l weights (`yolo11l.pt`).

---

## 1. System Requirements & Environment

The pipeline is engineered and validated for:
- **Evaluation Environment**: Ubuntu 22.04 LTS, Python 3.11, CUDA 12-compatible drivers, NVIDIA RTX 4090 (24 GB VRAM), 32 CPU threads, 64 GB RAM.
- **Execution & Development**: Antigravity (Windows) and Kaggle Linux environment (Python 3.13, 2× NVIDIA Tesla T4 GPUs).

---

## 2. Installation & Setup

### System Packages (Ubuntu 22.04)
The PyAV package bundles FFmpeg libraries, but standard multimedia system dependencies ensure robust decoder initialization:

```bash
sudo apt-get update && sudo apt-get install -y \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config
```

### Python Dependencies
Install the exact pinned dependencies in your Python virtual environment:

```bash
pip install -r requirements.txt
```

---

## 3. Preparation Step (Executed with Internet Access)

Before the evaluation run begins, execute the preparation script to fetch the official YOLO11l weights (`yolo11l.pt`). This step requires internet access and ensures that all subsequent indexing executes completely offline:

```bash
python prepare.py
```

This verifies and caches `yolo11l.pt` in the working directory.

---

## 4. Video Indexing Execution (Offline Evaluation)

The indexing pipeline runs without internet access using the required standard CLI command:

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

### Gzip Compressed Output Support
To write directly to compressed JSON Lines (saving storage and meeting submission artifact requirements):

```bash
python run.py --input <folder of .parquet files> --output results.jsonl.gz
```

### Additional Configurable Parameters
- `--weights <path>`: Path to YOLO11l model weights (default: `yolo11l.pt`).
- `--batch-size <int>`: Batch size for batched GPU forward passes (default: `32`).
- `--device <str>`: Explicit device selection, e.g. `cuda:0` or `cpu` (default: auto-detected).

---

## 5. Architectural & Mathematical Invariants

1. **Exhaustive Frame Processing**: Every decodable frame is processed. Frame skipping, sampling strides, and temporal decimation are prohibited.
2. **Relative Timestamp Alignment**:
   $$\Delta t = t_{\text{frame}} - t_{\text{first\_decoded\_frame}}$$
   $$\text{second} = \lfloor \Delta t \rfloor$$
3. **Contiguous Second Coverage**: Emits records for all contiguous integer seconds $0, 1, \dots, \lfloor \Delta t_{\text{last\_frame}} \rfloor$. Unpopulated seconds contain `objects: []` and `people: 0`.
4. **Target Classes**:
   - `person` (COCO ID 0)
   - `car` (COCO ID 2)
   - `dog` (COCO ID 16)
   - `laptop` (COCO ID 63)
   - `cell phone` (COCO ID 67)
5. **Object Presence Threshold**: An object is recorded as present in a second if detected in at least half of that second's frames ($\text{count} \ge \lceil N_{\text{second}} / 2 \rceil$).
6. **People Count Median (Round-Down)**: Median of people detections across all frames in that second. For even-length counts, halfway medians are explicitly rounded down using integer division:
   $$\text{people} = \left\lfloor \frac{P_{n/2 - 1} + P_{n/2}}{2} \right\rfloor$$
7. **Orientation Normalization**: Container metadata and stream display matrices are inspected. Rotated videos ($90^\circ, 180^\circ, 270^\circ$) are rotated upright prior to inference.
8. **Fault Isolation**: Corrupted or truncated MP4 streams do not interrupt execution. All seconds completed prior to decoding failure are flushed, followed by a structured error record:
   ```json
   {"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
   ```

---

## 6. Running Unit & Integration Tests

Run the test suite locally or in CI:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
