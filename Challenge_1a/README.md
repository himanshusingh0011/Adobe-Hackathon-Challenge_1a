# Challenge 1A — Title & Outline Extraction

**Adobe India Hackathon 2025** • Teacher–Student YOLOv8 + Document Heuristics

---

## TL;DR

We built a **two‑stage, teacher→student pipeline** for fast and accurate title/heading detection on PDFs, followed by a **deterministic heuristic engine** that promotes detected headings into a clean document outline (**H1/H2/H3/H4/…**).

* **Teacher:** `yolov8x-doclaynet.pt` finetuned **50 epochs** on **69k+** DocLayNet pages.
* **Student:** `yolov8m-doclaynet.pt` distilled/finetuned to **2 classes** (*Title*, *Section\_Header*) for speed and stability, trained **60 epochs**.
* **Metrics (Student best):** **mAP50 ≈ 0.904**, **mAP50–95 ≈ 0.64**.
* **Runtime:** Lightweight ONNX inference with multi-worker CPU execution; heuristic pass is linear in the number of detected headings.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Approach Overview](#approach-overview)
3. [Dataset & Labels](#dataset--labels)
4. [Modeling: Teacher → Student](#modeling-teacher--student)
5. [Export & Inference](#export--inference)
6. [Heading Heuristics (H1–Hn)](#heading-heuristics-h1hn)
7. [Output Schema](#output-schema)
8. [Project Structure](#project-structure)
9. [Setup & Installation](#setup--installation)
10. [Running the Pipeline](#running-the-pipeline)
11. [Evaluation & Benchmarks](#evaluation--benchmarks)
12. [Design Choices & Rationale](#design-choices--rationale)
13. [Limitations & Edge Cases](#limitations--edge-cases)
14. [Roadmap](#roadmap)
15. [Acknowledgments & License](#acknowledgments--license)

---

## Problem Statement

Given a directory of PDF files, **extract**:

* The **document title**.
* A hierarchical **outline** with heading levels (**H1/H2/H3/H4/…**).

Challenge constraints emphasize **accuracy**, **speed**, and **robustness** to diverse layouts (multi‑column, mixed fonts, scans vs. born‑digital, noisy headings, etc.).

---

## Approach Overview

Our solution combines **deep CV detection** with **deterministic text/layout heuristics**:

1. **Visual Detection (YOLOv8):**
   Detect **Title** and **Section\_Header** regions directly on rendered PDF pages. The detector is layout‑aware and robust to noise and multi‑column structure.

2. **Text Association:**
   For each detected region, extract the underlying text lines (via PDF text spans) intersecting the region.

3. **Heuristic Promotion:**
   Convert the flat list of section headers into a **hierarchical outline** (H1→Hn) using a set of rules based on **font/size cues, geometric consistency, numbering patterns, and structural constraints** (e.g., *H2 cannot exist without a preceding H1*).

This hybrid design leverages the **generalization ability of the detector** with the **explainability and repeatability** of heuristics.

---

## Dataset & Labels

* **Source:** DocLayNet (≈ **69,000+** labeled pages).
* **Classes used:**

  * `Title`
  * `Section_Header`
* **Preprocessing:**

  * Balanced sampling across document types to improve robustness.
  * Standard train/val/test split with page‑shuffled sampling to avoid document leakage.

> We intentionally **collapsed** the label space to the two classes above for the **student**, simplifying the task and boosting speed/precision on headings.

---

## Modeling: Teacher → Student

### Teacher (High‑Capacity)

* **Base:** `yolov8x-doclaynet.pt`
* **Epochs:** **50**
* **Goal:** Achieve strong, stable detection across diverse page styles to serve as a **signal upper bound** and **distillation source**.

### Student (Task‑Specialized)

* **Base:** `yolov8m-doclaynet.pt`
* **Epochs:** **60**
* **Classes:** `Title`, `Section_Header` only
* **Why:**

  * Reduces output dimensionality → **faster inference** and **less confusion** among near‑synonymous layout roles.
  * Better precision on headings, improved latency under CPU‑only constraints.

### Distillation / Fine‑Tuning Notes

* We leverage the teacher’s best checkpoint as a **pseudo‑labeler** / reference during student training.
* The student learns **task‑specific priors** (e.g., heading geometry) while maintaining generalization on varied formats.

---

## Export & Inference

* **Export:** Best student checkpoint is exported to **ONNX** for portable, CPU‑friendly inference.
* **Rendering:** Pages rendered with PyMuPDF at a controlled DPI; images are **letterboxed** to a fixed input size for YOLO.
* **Post‑processing:**

  * Confidence thresholding + NMS
  * Coordinate de‑letterboxing back to PDF space
  * **Text extraction** from intersecting spans (font name/size captured when available)

**Concurrency:** The script supports **multi‑worker** file processing and **intra‑page** threading to utilize available CPU cores while respecting I/O.

---

## Heading Heuristics (H1–Hn)

After detection, we promote **Section\_Header** instances to a hierarchical outline using **deterministic rules**:

1. **Normalization & Ordering**

   * Deduplicate overlapping boxes; prefer larger area and higher confidence.
   * Sort headings **top→bottom**, then **left→right** within rows.

2. **Structural Guards**

   * **Title** is unique (first strong candidate on page 1; backed by font size).
   * **H2 requires a preceding H1**, **H3 requires H2**, etc.
   * Page‑header/footer exclusion via **y‑position bands** and **repeated patterns** across pages.

3. **Font & Geometry Cues**

   * **Relative font size clusters** per document (k‑means/quantile bins) to create stable, document‑local size tiers.
   * Bold/SmallCaps/ALL‑CAPS and **line height** contribute positive signals.
   * **Left margin** and **indentation** clustering to distinguish peer vs. child levels.

4. **Lexical Patterns**

   * Common numbering: `1.`, `1.1`, `I.`, `A.`, `Chapter 3`, etc.
   * **Colon/dot‑leader** endings and trailing space normalization.
   * Heading text is cleaned, preserving intended leading spaces when explicitly present.

5. **Continuity & Recovery**

   * Cross‑page consistency checks (avoid abrupt level jumps).
   * If a level is missing, conservative fallback promotes to the nearest valid parent.

These rules yield a **stable, human‑readable outline** that mirrors the logical structure of the document, even in multi‑column or stylized layouts.

---

## Output Schema

For each input PDF, we emit a single JSON:

```json
{
  "file_name": "example.pdf",
  "title": "Document Title",
  "title_bbox": [x0, y0, x1, y1],
  "outline": [
    { "level": "H1", "text": "1. Introduction", "page": 2, "bbox": [x0, y0, x1, y1] },
    { "level": "H2", "text": "1.1 Background",  "page": 2, "bbox": [x0, y0, x1, y1] },
    { "level": "H2", "text": "1.2 Scope",       "page": 3, "bbox": [x0, y0, x1, y1] },
    { "level": "H1", "text": "2. Methods",      "page": 4, "bbox": [x0, y0, x1, y1] }
  ]
}
```

> **Note:** BBoxes are page‑space coordinates. Leading spaces are preserved when they exist intentionally in the source.

---

## Project Structure

```
Challenge_1a/
├─ models/
│  ├─ teacher/                # yolov8x-doclaynet teacher artifacts (optional)
│  ├─ student/                # yolov8m-doclaynet student artifacts
│  │  ├─ best.onnx            # exported ONNX model (inference)
│  │  └─ best.pt              # PyTorch checkpoint (optional)
├─ input/                     # PDFs to process (read-only)
├─ output/                    # JSON results (one per PDF)
├─ process_pdfs.py            # end-to-end inference + heuristics
├─ requirements.txt
├─ Dockerfile
└─ README.md                  # this file
```

---

## Setup & Installation

### Prerequisites

* **Python** 3.9+
* **Windows/Linux/macOS**
* (Optional) **Docker** for containerized runs

### Python Environment

```bash
# Create & activate a venv (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` (core)

```
PyMuPDF==1.23.0
numpy==1.24.3
opencv-python==4.9.0.80
onnxruntime==1.18.0
scikit-learn==1.3.0
regex==2023.6.3
tqdm==4.66.1
```

> If your PDFs are image‑only scans and you want OCR fallbacks, add `pytesseract` and ensure Tesseract is installed system‑wide.

### Docker (Optional)

```dockerfile
# Dockerfile (excerpt)
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY process_pdfs.py .
RUN mkdir -p /app/input /app/output /app/models
CMD ["python", "process_pdfs.py"]
```

Build:

```bash
docker build -t pdf-outliner:latest .
```

Run (Linux/macOS bash):

```bash
docker run --rm \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/models:/app/models:ro" \
  --network none \
  pdf-outliner:latest \
  --input_dir /app/input \
  --output_dir /app/output \
  --model /app/models/best.onnx \
  --workers 8 --threads 8
```

Run (Windows PowerShell):

```powershell
docker run --rm `
  -v "${PWD}\input:/app/input:ro" `
  -v "${PWD}\output:/app/output" `
  -v "${PWD}\models:/app/models:ro" `
  --network none `
  pdf-outliner:latest `
  --input_dir /app/input `
  --output_dir /app/output `
  --model /app/models/best.onnx `
  --workers 8 --threads 8
```

---

## Running the Pipeline

### CLI

```bash
python process_pdfs.py \
  --input_dir ./input \
  --output_dir ./output \
  --model ./models/best.onnx \
  --workers 8 \
  --threads 8 \
  --dpi 180 \
  --img_size 640 \
  --conf_thres 0.25 \
  --iou_thres 0.45
```

**Key flags**

* `--workers` : number of PDFs processed in parallel.
* `--threads` : intra‑PDF parallelism (e.g., page‑level).
* `--dpi`     : render DPI for page images (balance quality/speed).
* `--img_size`: YOLO input size (e.g., 512 or 640).
* `--conf_thres`, `--iou_thres`: detection filtering/NMS.

> Use **read‑only** bind mount for `/input` and write‑only for `/output` to maintain reproducibility.

---

## Evaluation & Benchmarks

* **Student best checkpoint** (2‑class):

  * **mAP50 ≈ 0.904**
  * **mAP50–95 ≈ 0.64**

**Notes:**

* These figures were obtained on our held‑out validation subset following the training configuration described above.
* Real‑world performance depends on document domain, render DPI, and the strictness of the heuristic constraints.

---

## Design Choices & Rationale

1. **Teacher→Student:**
   A large teacher stabilizes learning across edge layouts. A slim student, restricted to `Title` and `Section_Header`, **improves latency** and **reduces label confusion**.

2. **2‑Class Simplification:**
   Collapsing non‑heading roles prevents over‑detection noise and eases the downstream heuristic’s job.

3. **ONNX Inference:**
   Platform‑agnostic, **CPU‑friendly**, and easy to containerize. Works well with multi‑worker execution.

4. **Deterministic Heuristics:**
   Transparent, explainable rules let us enforce **document‑level consistency**, such as *no H2 without H1*, margin clustering, and numbering‑aware leveling.

---

## Limitations & Edge Cases

* **Stylized layouts:** Extreme decorative headings or heavy graphics can reduce detector confidence.
* **Mixed languages / scripts:** Heuristic cues (e.g., casing) may be less informative; numbering styles vary.
* **Scan‑only PDFs:** Require OCR to recover text; heading detection still works visually, but text fidelity depends on OCR.
* **Nonlinear structures:** Magazines/newsletters with floating decks and sidebars may need extra roles (e.g., `subsection_deck`) for perfect nesting.

---

## Roadmap

* Add **layout‑aware language features** (e.g., small LM with positional prompts) to improve level disambiguation when font cues are weak.
* Explore **self‑training** on unlabeled corpora with the teacher to broaden domain coverage.
* Optional **graph‑based** post‑processing to reconcile multi‑column flows.
* Provide a **web demo** and **batch API** for easy adoption.

---

## Acknowledgments & License

* **DocLayNet** contributors for the dataset and annotations.
* **Ultralytics YOLOv8** for the detection framework.

This repository is for the **Adobe India Hackathon 2025 — Challenge 1A** submission.
See `LICENSE` for terms (or specify licensing here as appropriate).

---

## Appendix

### Example Command Cheatsheet

```bash
# Baseline run (8 workers / 8 threads)
python process_pdfs.py --input_dir ./input --output_dir ./output \
  --model ./models/best.onnx --workers 8 --threads 8

# Faster I/O on big PDFs: lower DPI, smaller img size
python process_pdfs.py --dpi 150 --img_size 512 ...

# Stricter precision: raise conf_thres
python process_pdfs.py --conf_thres 0.35 ...
```

### Troubleshooting

* **“invalid reference format” (Docker):** Avoid spaces in image name/tag; use `pdf-outliner:latest`.
* **ONNXRuntime providers:** If GPU is unavailable, you’ll see `['CPUExecutionProvider']`—that’s expected and supported.
* **Empty output:** Ensure the `--model` path is correct and readable; verify input PDFs are not encrypted and that `--dpi` is ≥150.

---

*If you have questions or want to reproduce the training runs (teacher/student configs, data preparation, and export scripts), open an issue—we’re happy to share additional details.* 🙌
