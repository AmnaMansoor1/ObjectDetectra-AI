# ObjectDetectra AI - Candidate Prompts Summary

This document lists all prompts submitted by the candidate during development in chronological order.

> **Note on Full Chat History**:
> Section 7 of the assignment brief states:
> *"Your full AI chat history for this task, from every AI tool you used... Please don't edit or shorten the chats, apart from replacing passwords and API tokens with [REMOVED]. A missing or edited history counts against you."*
> To guarantee full compliance, the complete, unedited dialogue history (prompts + assistant responses) is preserved in [`antigravity_chat.md`](antigravity_chat.md) and [`antigravity_chat.jsonl`](antigravity_chat.jsonl).

---

## Prompt 1

```text
I am developing an AI Engineering take-home assignment titled "Video Indexing at Speed" using the official Ultralytics YOLO11l object detection model.

I want you to act as a Senior Computer Vision Engineer and Technical Architect.

## 1. Project objective

Develop a Python-based video indexing pipeline that processes approximately 125 hours of video stored across the first 50 Parquet files of the Hugging Face FineVideo dataset.

For every second of every video, the system must:
- Detect the presence of five COCO object classes: person, car, dog, laptop, and cell phone.
- Calculate the median number of people detected across the frames in that second.
- Generate structured JSON Lines output.
- Continue processing when individual videos are damaged or cannot be decoded.

## 2. Model requirements

The implementation must use the official pretrained YOLO11l weights (`yolo11l.pt`).

Required configuration:
- Input size: 640 × 640.
- Confidence threshold: 0.4.
- IoU threshold: 0.7.
- COCO class IDs: person (0), car (2), dog (16), laptop (63), cell phone (67).
- Preserve Ultralytics default letterbox preprocessing and class-aware NMS.
- Use BGR frame inputs.
- Apply video rotation metadata when present.

Do not train a custom model or substitute another YOLO model.

## 3. Critical processing requirements

- Process every decoded frame. Frame skipping and temporal sampling are prohibited.
- Assign frames to seconds using timestamps relative to the first decoded frame.
- Include an object in a second if it is detected in at least half of that second's frames.
- Calculate the median of per-frame people counts and round halfway values down.
- Generate one record for every second through the second containing the last frame.
- Preserve already completed seconds if a later processing error occurs.
- Record an error line for an unsuccessful video.

## 4. Input and output

Input: Parquet files containing MP4 video bytes in the `mp4` column.

Output: JSON Lines with one record per video-second, or one error record for an unsuccessful video.

Required execution command:

`python run.py --input <folder of .parquet files> --output results.jsonl`

The submission must also support producing the required compressed artifact `results.jsonl.gz`.

## 5. Development and execution environments

I will use:
- Antigravity on my Windows laptop for code development.
- Kaggle for free GPU-based execution.
- GitHub for repository hosting.

My current Kaggle environment has:
- 2 NVIDIA Tesla T4 GPUs.
- 14.56 GB GPU memory per GPU.
- Approximately 31 GB system RAM.
- 2 physical CPU cores and 4 logical threads.
- Approximately 19.5 GB available in `/kaggle/working`.
- Python 3.13.16.
- PyTorch 2.11.0+cu130.
- CUDA 13.0.

The employer's evaluation environment is Ubuntu 22.04, Python 3.11, CUDA 12-compatible drivers, an RTX 4090 with 24 GB VRAM, 32 CPU threads and 64 GB RAM.

The implementation must remain compatible with the employer's evaluation environment.

## 6. Engineering priorities

Prioritize the following, in order:

1. Functional correctness against the assignment specification.
2. Reliable handling of large Parquet datasets and damaged videos.
3. Memory-efficient streaming and incremental output writing.
4. GPU inference throughput and efficient batching.
5. Reproducibility and maintainability.

Do not compromise detection semantics for speed.

## 7. Your first task: analyze, do not implement

Read the complete assignment PDF available in this workspace.

Before generating code or modifying any files, provide the following:

A. Requirements analysis:
Create a table containing each major functional requirement, its technical interpretation, and how it will be validated.

B. Proposed architecture:
Explain the processing pipeline from Parquet ingestion through video decoding, YOLO inference, per-second aggregation and JSONL serialization.

C. Performance analysis:
Identify likely bottlenecks and distinguish safe optimizations from prohibited shortcuts.

D. Technical risks:
Identify potential problems involving variable frame rates, video timestamps, rotation metadata, memory management, GPU utilization, Parquet reading, damaged videos and output recovery.

E. Implementation plan:
Recommend a minimal, testable development sequence, starting with one small video before scaling to the full dataset.

F. Dependency strategy:
Recommend the smallest practical set of Python dependencies, taking account of both Kaggle Python 3.13 and the employer's Python 3.11 environment.

## 8. Constraints

Do not write the implementation yet.
Do not download the dataset or model weights.
Do not install packages.
Do not generate fabricated benchmark numbers or accuracy metrics.
Do not use frame sampling.
Do not write my WRITEUP.md answers. Those must be independently authored by me.

Be precise, explain important engineering decisions, and refer to the relevant sections of the provided assignment wherever possible.

Wait for my approval before proceeding to implementation.
```

---

## Prompt 2

```text
The architecture analysis is complete. Proceed to implement the first correct, testable version of ObjectDetectra AI.

Use the original assignment PDF as the authoritative specification and follow the architecture you proposed, subject to the corrections below.

## 1. Implementation deliverables

Create the following files:

- run.py
- requirements.txt
- prepare.py
- README.md
- tests/test_aggregation.py
- tests/test_output.py

Do not create unnecessary modules or complex infrastructure.

## 2. Detection implementation

Use:
- Official YOLO11l weights: yolo11l.pt
- Input size: 640
- Confidence threshold: 0.4
- IoU threshold: 0.7
- BGR images
- Ultralytics default preprocessing, letterboxing and class-aware NMS.

Initially use the default inference precision. Do not assume FP16 is equivalent to FP32. Any later precision optimization must be benchmarked and validated.

Filter the final detections to these five COCO classes:
0 person
2 car
16 dog
63 laptop
67 cell phone

Do not use tracking.

Do not pass unnecessary non-default inference settings. In particular, avoid changing other model defaults.

## 3. Frame processing

Every decoded frame must be processed.

Use PyAV to decode MP4 bytes in memory and preserve presentation timestamps.

Calculate:
relative_timestamp = frame_timestamp - first_decoded_frame_timestamp

second = floor(relative_timestamp)

Do not use FPS-based approximations when actual frame timestamps are available.

Do not implement frame skipping, sampling, vid_stride or reduced-resolution inference.

Use configurable batched inference to manage GPU memory.

## 4. Aggregation

Implement per-second aggregation exactly as defined:

- Count the total frames belonging to each second.
- Count frames in which each target class appears at least once.
- Include an object when its frame count is at least ceil(total_frames / 2).
- Calculate the median of the number of person detections in every frame.
- For even-sized lists, use integer division to round halfway medians down.
- Generate all seconds through the second containing the final decoded frame.
- Empty seconds must contain objects=[] and people=0.

## 5. Parquet processing

Use PyArrow's ParquetFile and incremental record-batch iteration.

Read only the mp4 column.

Process one video at a time initially.

Create video identifiers using the Parquet filename and zero-based row index.

Discover Parquet files without duplicates.

## 6. Failure handling and output

- A damaged video must not stop other videos.
- Preserve already completed second records after a mid-video failure.
- Write an error record for the failed video.
- Flush output incrementally.
- Support both JSONL and JSONL.GZ output correctly.
- Handle missing or invalid MP4 data.

## 7. GPU and performance strategy

Start with one GPU device, cuda:0.

Implement straightforward batched inference.

Do not implement dual-GPU multiprocessing, TensorRT or complex concurrent pipelines in this first version.

We will benchmark and optimize after validating correctness.

The final script must work in Python 3.11 on the employer's Ubuntu environment and also be testable in Kaggle's Python 3.13 environment.

## 8. Testing requirements

Create tests for:

- Object presence exactly at 50%.
- Object presence below 50%.
- Odd frame counts.
- Even frame counts.
- Halfway median rounding.
- Empty seconds.
- Correct video identifiers.
- Gzip output.
- Duplicate-free Parquet discovery.

Create a short synthetic MP4 test so we can validate the pipeline without waiting for the complete FineVideo dataset.

Use genuine YOLO predictions in production. Synthetic data may only be used for controlled tests.

## 9. Documentation

Document the exact installation and preparation commands.

Explain how to download yolo11l.pt before offline execution.

Keep the required command unchanged:

python run.py --input <folder of .parquet files> --output results.jsonl

Also document gzip support.

Do not invent execution times, hardware specifications, accuracy metrics or successful full-dataset processing.

Do not write WRITEUP.md answers because those must be independently written by me.

## 10. Execution instructions

Implement the files and run the unit tests.

Do not download the full dataset or start full processing.

At the end, report:
1. Files created.
2. Tests passed and failed.
3. Remaining correctness risks.
4. Exact commands to run the first sample test in Kaggle.

Do not silently omit any assignment requirement.
```

---

## Prompt 3

```text
Perform a targeted correctness review of the existing run.py. Do not rewrite the entire codebase.

I identified two issues requiring attention:

1. PARTIAL-SECOND HANDLING

In process_single_video(), the exception handler currently finalizes seconds through last_second. If decoding fails halfway through a second, this may emit an incomplete second as though it were complete.

Correct the failure handling so that:
- Fully completed seconds are preserved.
- An incomplete final second is not emitted as a successful record.
- Exactly one error record is appended.
- Processing continues with the next video.
- Successful videos still emit every second through their final frame.

Add a regression test where decoding fails halfway through second 2, and verify that seconds 0 and 1 are retained but incomplete second 2 is not emitted.

2. PYAV RESOURCE CLEANUP

Modify decode_video_frames() to guarantee that the PyAV container is closed using try/finally, including when decoding raises an exception or the generator is closed early.

ADDITIONAL CHECKS

- Confirm no frames are skipped.
- Confirm all inference batches preserve the original frame order.
- Confirm every decoded frame contributes to exactly one second.
- Confirm final aggregation handles an incomplete inference batch at normal end-of-video.
- Do not change model settings or aggregation formulas.

Run the complete unit test suite again and report the changes.

Do not download the full dataset or run the full workload.
```

---

## Prompt 4

```text
Create HARDWARE.md for my ObjectDetectra AI assignment.

Use only these verified environment details:

- Platform: Kaggle Notebooks
- GPU: 2 × NVIDIA Tesla T4
- GPU memory: approximately 14.56 GiB per GPU
- CPU: Intel Xeon @ 2.00 GHz
- CPU cores: 2 physical cores, 4 logical threads
- RAM: approximately 30 GiB
- Python: 3.13.16
- CUDA: 13.0
- Operating environment: Kaggle Linux notebook

Do not invent the disk type, full-dataset runtime, processing throughput or number of completed videos.

Add clearly marked pending fields for:
- Actual full dataset processing duration
- Actual number of completed Parquet files
- Actual number of processed videos
- Actual total output records

The document must honestly distinguish the Kaggle environment from the employer's RTX 4090 evaluation environment.
```

---

## Prompt 5

```text
so far i have this log , 
GPU 0
Latest log:
[PIPELINE] Discovered 25 Parquet file(s) without duplicates.
[YOLO] Loading model from '/tmp/ObjectDetectra-AI/yolo11l.pt' on device 'cuda:0'...
[PIPELINE] Ingesting: train-00000-of-01357.parquet
Output size: 5212 bytes

GPU 1
Latest log:
[PIPELINE] Discovered 25 Parquet file(s) without duplicates.
[YOLO] Loading model from '/tmp/ObjectDetectra-AI/yolo11l.pt' on device 'cuda:1'...
[PIPELINE] Ingesting: train-00001-of-01357.parquet
Output size: 4254 bytes is it possible that we can complete processing of all videos before times run out?
```

---

## Prompt 6

```text
Act as a Senior AI Engineering Code Reviewer performing a final technical submission audit for ObjectDetectra AI.

The implementation has been developed and tested using YOLO11l on Kaggle with two Tesla T4 GPUs.

The original assignment PDF is the authoritative specification.

## 1. Source-code audit

Review the existing implementation against the complete assignment.

Verify:
- Every decoded frame is processed.
- PTS-based timestamps determine second assignment.
- All five required COCO classes are correct.
- YOLO11l, 640 resolution, confidence 0.4 and IoU 0.7 are preserved.
- Per-second object presence uses the required majority threshold.
- People counts use median with halfway rounding down.
- Rotation metadata is handled.
- Gzip output is valid.
- Failed videos generate error records.
- Completed results are preserved after video failures.

Identify any unresolved correctness risks.

Do not introduce frame skipping or alter model settings.

## 2. Documentation audit

Review README.md, requirements.txt and prepare.py.

Ensure:
- Installation instructions are reproducible.
- Model weights are downloaded before offline execution.
- The required CLI command is documented.
- Gzip output is explained.
- Python 3.11 compatibility is considered.
- Kaggle-specific setup is separated from the employer's evaluation environment.

## 3. Hardware documentation

Create or update HARDWARE.md using only verified facts:

- Platform: Kaggle Notebooks.
- GPU: 2 NVIDIA Tesla T4 GPUs.
- GPU memory: approximately 14.56 GB each.
- CPU: Intel Xeon @ 2.00 GHz.
- CPU configuration: 2 physical cores, 4 logical threads.
- RAM: approximately 30 GB.
- Python: 3.13.16.
- CUDA: 13.0.
- Environment: Kaggle Linux.

Clearly distinguish the measured one-video runtime from any full-dataset runtime.

Do not invent processing times, disk types, completed video counts or throughput figures.

## 4. Submission checklist

Verify the required files:
- run.py
- requirements.txt
- README.md
- results.jsonl.gz
- HARDWARE.md
- WRITEUP.md
- ai_chats/links.txt

Report missing files and incomplete items.

Check for accidental tokens, secrets, datasets or model weights in the Git repository.

## 5. Strict restrictions

Do not create, write, rewrite, polish or review WRITEUP.md answers.

Do not invent full-dataset benchmark results.

Do not modify the running Kaggle workers or their outputs.

Do not change run.py unless you identify a critical correctness violation. Report such issues separately first.

Provide a concise technical audit with PASS, FAIL and NEEDS VERIFICATION classifications.
```

---

## Prompt 7

```text
in ai chat links add this antigravity chat link
```

---

## Prompt 8

```text
you have added local uri but recruiters would not be able to see it so plz make history accessible to them as well
```

---

## Prompt 9

```text
but they have asked only prompts , you have added whole history
```

---

