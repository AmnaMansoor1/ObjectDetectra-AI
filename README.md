# ObjectDetectra AI
### Video Indexing at Speed using YOLO11l

A Python-based video indexing pipeline designed to process large-scale video datasets stored in Apache Parquet files.

The system uses the official pretrained **Ultralytics YOLO11l** model to detect five object categories and calculate per-second people counts. It supports GPU-accelerated inference, incremental result writing, and error handling for damaged videos.

## 1. Project Overview

The assignment involves approximately **125 hours of video**, distributed across the first 50 Parquet files of the Hugging Face FineVideo dataset.

For each second of every video, the system:

- Detects `person`, `car`, `dog`, `laptop`, and `cell phone`.
- Determines object presence using the specified 50% frame-occurrence threshold.
- Calculates the median number of people detected per frame.
- Produces structured JSON Lines output.
- Continues processing when individual videos are damaged.

### Model Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11l |
| Weights | `yolo11l.pt` |
| Input image size | 640 |
| Confidence threshold | 0.4 |
| NMS IoU threshold | 0.7 |
| Preprocessing | Ultralytics default letterbox |
| NMS | Class-aware |
| Input colour order | BGR |

## 2. System Requirements

### Evaluation Environment

The employer's evaluation environment is:

| Component | Specification |
|---|---|
| Operating system | Ubuntu 22.04 |
| Python | 3.11 |
| GPU | NVIDIA RTX 4090 |
| GPU memory | 24 GB |
| CPU | 32 threads |
| RAM | 64 GB |
| GPU driver | CUDA 12 compatible |

### Development Environment

The implementation was developed using Google Antigravity on Windows and tested using Kaggle's free GPU environment.

The Kaggle environment provided two NVIDIA Tesla T4 GPUs, approximately 31 GB RAM, and four logical CPU threads.

## 3. Installation

### Ubuntu System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config
```

### Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All Python dependencies required by the application are listed in `requirements.txt`.

## 4. Model Preparation

The official YOLO11l weights must be downloaded before the offline evaluation run.

```bash
python prepare.py
```

This preparation step requires internet access and makes `yolo11l.pt` available for subsequent inference.

## 5. Running the Pipeline

The required command-line interface is:

```bash
python run.py --input <folder> --output results.jsonl
```

For gzip-compressed output:

```bash
python run.py --input <folder> --output results.jsonl.gz
```

### Optional Arguments

| Argument | Description |
|---|---|
| `--weights` | Path to official YOLO11l weights |
| `--batch-size` | Number of frames in an inference batch |
| `--device` | Inference device, e.g. `cuda:0` or `cpu` |

The implementation supports selecting a GPU device and configuring the inference batch size.

## 6. Processing Architecture

The processing pipeline follows these stages:

1. **Parquet ingestion:** PyArrow reads the video bytes from the `mp4` column.
2. **Video decoding:** PyAV decodes the embedded MP4 video into frames.
3. **Timestamp alignment:** Frames are assigned to seconds relative to the first decoded frame.
4. **Orientation handling:** Rotation metadata is considered before inference.
5. **Object detection:** YOLO11l processes batches of frames on the selected device.
6. **Temporal aggregation:** Frame-level detections are combined into per-second object-presence and people-count results.
7. **Serialization:** Results are written incrementally to JSONL or gzip-compressed JSONL.

### Temporal Aggregation

For a second containing \(N\) frames, an object is considered present when detected in at least:

\[
\left\lceil \frac{N}{2} \right\rceil
\]

frames.

The people count is the median of the per-frame person counts, rounded down for halfway values.

The correctness baseline processes every decoded frame. Temporal sampling would require separate validation against the full-frame reference.

## 7. Output Format

Each successful record contains the video identifier, second index, detected objects, and people count.

```json
{"video":"train-00003-of-01357.parquet:4","second":12,"objects":["person","laptop"],"people":2}
```

Seconds without qualifying detections contain:

```json
{"video":"train-00003-of-01357.parquet:4","second":13,"objects":[],"people":0}
```

An unsuccessful video produces an error record:

```json
{"video":"train-00003-of-01357.parquet:5","error":"could not decode video"}
```

The video identifier consists of the Parquet filename and its zero-based row index.

## 8. Error Handling

The pipeline is designed to handle individual video failures without terminating the entire dataset run.

If decoding fails partway through a video, previously completed second records are preserved, and an error record is written.

The system also processes input files incrementally to avoid loading the complete dataset into memory.

## 9. Testing and Validation

Run the unit and integration tests using:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite covers:

- Object presence at the exact 50% threshold
- Odd and even frame counts
- Median rounding behaviour
- Empty detections
- JSONL and gzip output
- Video identifier formatting
- Partial decoding failures
- PyAV-based video decoding

The tests were executed on Kaggle, with 21 tests passing.

Testing on actual FineVideo samples also confirmed nonempty object detections. The employer's hidden-test accuracy thresholds have not been independently verified.



## 10. Performance and Limitations

Initial testing was conducted on NVIDIA Tesla T4 GPUs.

The implementation has demonstrated GPU-based inference and incremental output generation. Full-dataset processing is computationally intensive, and temporary Kaggle session interruptions can result in the loss of unsaved progress.

Performance measurements and the hardware used for processing are documented in `HARDWARE.md`.

**Dataset completion status:** The available submission results cover only the videos successfully processed before the deadline. The full 50-file dataset run was not completed, and the included output must not be interpreted as covering all 125 hours.

## 11. Repository Structure

```text
ObjectDetectra-AI/
├── run.py
├── prepare.py
├── requirements.txt
├── README.md
├── HARDWARE.md
├── WRITEUP.md
├── results.jsonl.gz
├── tests/
│   ├── test_aggregation.py
│   └── test_output.py
└── ai_chats/
    ├── links.txt
    └── antigravity_chat.jsonl
```

The `WRITEUP.md` file contains the candidate's independently written responses to the assignment questions.

The `ai_chats/` directory contains AI interaction references and exported chat history.

## 12. Evaluation Criteria

The employer evaluates the submitted script using unseen FineVideo videos and reference outputs generated through full-frame YOLO11l inference.

| Metric | Required threshold |
|---|---|
| Object-presence F1 | ≥ 0.95 |
| Exact people count | ≥ 85% |
| People count within ±1 | ≥ 95% |
| Maximum evaluation runtime | 6 hours |

Accuracy requirements must be met before execution speed is considered.

---

**Project:** ObjectDetectra AI  
**Assignment:** Video Indexing at Speed  
**Author:** Amna Mansoor  
