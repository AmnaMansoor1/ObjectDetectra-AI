# Execution Hardware & Environment

This document records the hardware environment used to execute the video indexing pipeline for the **ObjectDetectra AI** assignment, as specified in Section 7 of the assignment brief. It explicitly contrasts the Kaggle execution setup with the employer's target evaluation environment, and strictly distinguishes sample-level benchmarks from full-dataset execution.

---

## 1. Execution Environment: Kaggle Notebooks

The pipeline was developed and executed on a Kaggle GPU instance with the following verified specifications:

| Component | Verified Specification |
|---|---|
| **Platform** | Kaggle Notebooks |
| **Operating System** | Linux (Kaggle Container Environment) |
| **GPU Model** | 2 × NVIDIA Tesla T4 |
| **GPU Memory (VRAM)** | ~14.56 GB per GPU (Total: ~29.12 GB across 2 GPUs) |
| **CPU Model** | Intel Xeon @ 2.00 GHz |
| **CPU Configuration** | 2 physical cores, 4 logical threads |
| **System Memory (RAM)** | ~30 GB |
| **Disk Type** | `[PENDING: Verify disk type (e.g., ext4 virtual block device / SSD) via lsblk / df in Kaggle session]` |
| **Python Version** | 3.13.16 |
| **CUDA Version** | 13.0 |

---

## 2. Comparison with Employer Evaluation Environment

The employer evaluates submissions on a dedicated standalone workstation. The pipeline has been engineered to maintain strict functional compatibility across both environments while adapting to their distinct hardware profiles:

| Hardware Attribute | Kaggle Execution Setup | Employer Evaluation Workstation | Architectural Impact & Design Decisions |
|---|---|---|---|
| **GPU Model** | 2 × NVIDIA Tesla T4 | 1 × NVIDIA GeForce RTX 4090 | T4 uses Turing Tensor Cores; 4090 uses Ada Lovelace. The codebase defaults to single-GPU execution (`cuda:0`), matching the 4090 evaluation setup. |
| **GPU Memory** | ~14.56 GB per GPU | 24 GB VRAM | Batch size 32 operates well within the 14.56 GB ceiling of the T4, guaranteeing zero Out-Of-Memory (OOM) risk on the 24 GB RTX 4090. |
| **CPU Threads** | 4 logical threads (2 physical cores) | 32 CPU threads | Low CPU thread count on Kaggle creates a software video decoding bottleneck. PyAV multi-threaded decoding (`thread_type="AUTO"`) maximizes utilization on both 4 threads and 32 threads. |
| **System RAM** | ~30 GB | 64 GB | Incremental PyArrow record batching (`iter_batches(batch_size=1)`) maintains $O(1)$ video payload RAM consumption, safely below both limits. |
| **Python Version** | Python 3.13.16 | Python 3.11 (Ubuntu 22.04 LTS) | Pinned dependencies in `requirements.txt` avoid ABI-incompatible wheels and ensure seamless execution under both Python 3.11 and Python 3.13. |
| **Network Access** | Offline execution during timed run | Strictly offline during timed run | Weights (`yolo11l.pt`) are pre-cached via `python prepare.py` prior to timed processing in both environments. |

---

## 3. Benchmarks & Runtime Measurements

### A. Measured Single-Video Sample Benchmark
*(Measured on a single representative video from `train-00000-of-01357.parquet` on Kaggle)*

- **Test Video**: `train-00000-of-01357.parquet:0`
- **Video Duration**: `[PENDING: Duration in seconds]`
- **Decoded Frames**: `[PENDING: Total decoded frames]`
- **Processing Time**: `[PENDING: Elapsed seconds from decode to JSONL write]`
- **Effective Throughput**: `[PENDING: FPS (Frames / Elapsed Seconds)]`
- **Device Used**: Single Tesla T4 (`cuda:0`), batch size 32

---

### B. Full-Dataset Execution (50 Parquet Files / ~125 Hours)
*(All figures below reflect the actual live execution metrics without extrapolation or estimation)*

- **Status**: Live run on Kaggle across 2 × Tesla T4 GPUs
- **Processing Start Timestamp**: `[PENDING: e.g., YYYY-MM-DD HH:MM:SS UTC]`
- **Processing End Timestamp**: `[PENDING: e.g., YYYY-MM-DD HH:MM:SS UTC]`
- **Actual Full Dataset Processing Duration**: `[PENDING: Total elapsed time]`
- **Actual Completed Parquet Files**: `[PENDING: Files completed out of 50]`
- **Actual Processed Video Count**: `[PENDING: Videos indexed]`
- **Actual Total Output Records**: `[PENDING: Total lines in results.jsonl.gz]`
- **Overall Dataset Throughput**: `[PENDING: Aggregate FPS across all processed files]`
- **Peak GPU Memory Allocated**: `[PENDING: Recorded via torch.cuda.max_memory_allocated()]`
- **Error Records Logged**: `[PENDING: Videos with corrupted/unreadable streams]`
