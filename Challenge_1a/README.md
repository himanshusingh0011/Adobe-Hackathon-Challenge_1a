Adobe India Hackathon 2025 — Teacher–Student YOLOv8 + Document Heuristics
=========================================================================

## TL;DR

We built a **two‑stage teacher→student pipeline** for fast, accurate detection of **Title** and **Section headers** on PDFs, followed by a **deterministic heuristic engine** that constructs a clean outline (**exported as Title + H1/H2/H3** per challenge spec).

* **Teacher:** `yolov8x-doclaynet.pt` finetuned **50 epochs** on **69k+** DocLayNet pages.
* **Student:** `yolov8m-doclaynet.pt` distilled/finetuned to **2 classes** (*Title*, *Section\_Header*) for speed and stability, **60 epochs**.
* **Metrics (Student best):** **mAP50 ≈ 0.904**, **mAP50–95 ≈ 0.64**.
* **Runtime:** ONNXRuntime (CPU‑only, offline) with multi‑worker execution; heuristics are linear in headers detected.
* **Output (required):** `title` + `outline` with **H1/H2/H3** and page numbers.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Approach Overview](#approach-overview)
3. [Dataset & Labels](#dataset--labels)
4. [Modeling: Teacher → Student](#modeling-teacher--student)
5. [Export & Inference](#export--inference)
6. [Heading Heuristics (H1–H3 export)](#heading-heuristics-h1h3-export)
7. [Multilingual Support (JA/DE/EN/ES)](#multilingual-support-jadeed-en-es)
8. [Output Schema (Challenge‑compliant)](#output-schema-challengecompliant)
9. [Project Structure](#project-structure)
10. [Setup & Installation (Docker‑only required)](#setup--installation-dockeronly-required)
11. [Running the Pipeline (Docker)](#running-the-pipeline-docker)
12. [Evaluation & Benchmarks](#evaluation--benchmarks)
13. [Design Choices & Rationale](#design-choices--rationale)
14. [Limitations & Edge Cases](#limitations--edge-cases)
15. [Roadmap](#roadmap)
16. [Acknowledgments & License](#acknowledgments--license)
17. [Appendix: Local Dev (Optional)](#appendix-local-dev-optional)

---

## Problem Statement

From each input PDF, extract:

* A single **document title**.
* A hierarchical outline with **H1/H2/H3** (text + page).

The solution must be **Docker‑only**, **CPU‑only**, **offline**, and process typical PDFs quickly and robustly across diverse layouts.

---

## Approach Overview

We combine **visual detection** with **text/layout heuristics**:

1. **Visual Detection (YOLOv8):**
   Detect **Title** and **Section\_Header** regions on rendered pages (layout‑aware, resilient to multi‑column, mixed fonts, and noise).

2. **Text Association:**
   For each detection, harvest underlying PDF text spans and basic typography (font name/size when available).

3. **Heuristic Promotion → Outline:**
   Promote detected headers to a **well‑formed hierarchy** using **font‑size tiers, indentation/margins, numbering patterns,** and **structural guards** (e.g., no **H2** before the first **H1**).
   Internally we can infer deeper levels, but **we export only H1/H2/H3** to match the challenge format.

---

## Dataset & Labels

* **Source:** DocLayNet (\~**69k+** labeled pages).
* **Classes used:** `Title`, `Section_Header` (student model).
* **Preprocessing:** Balanced sampling across document types; page‑shuffled train/val/test to avoid leakage.

> We intentionally **collapsed** to two layout roles for the student to reduce label confusion and improve speed/precision on headings.

---

## Modeling: Teacher → Student

### Teacher (high capacity)

* **Base:** `yolov8x-doclaynet.pt`
* **Epochs:** 50
* **Role:** Provide strong priors and supervision for student fine‑tuning/distillation.

### Student (task‑specialized)

* **Base:** `yolov8m-doclaynet.pt`
* **Epochs:** 60
* **Classes:** `Title`, `Section_Header`
* **Why:** Smaller head + fewer classes ⇒ **faster CPU inference** and **cleaner heading detections**.

### Distillation / Fine‑Tuning

* Use the teacher’s best checkpoint as a pseudo‑label/reference.
* Student learns task‑specific priors (heading geometry, aspect, placement) while maintaining robustness.

---

## Export & Inference

* **Export:** Student best checkpoint → **ONNX** (≤ 200 MB) for portable CPU inference.
* **Rendering:** PyMuPDF at tuned DPI; letterbox to YOLO input; reverse‑map coords back to PDF space.
* **Post‑processing:** confidence filtering + NMS; text extraction from intersecting spans.
* **Concurrency:** Multi‑worker (across PDFs) and page‑level threading with conservative defaults for an 8‑CPU judge environment.

---

## Heading Heuristics (H1–H3 export)

After detection, we transform **Section\_Header** instances into a stable, 3‑level outline:

1. **Normalization & Ordering**
   Deduplicate overlaps (by confidence/area); sort top‑to‑bottom, then left‑to‑right.

2. **Structural Guards**

   * Unique **Title** (preferred on first page; backed by font size and geometry).
   * **H2 requires H1**, **H3 requires H2**; prevent level jumps.
   * Exclude running headers/footers via vertical bands and repeated patterns.

3. **Font & Geometry Cues**

   * **Document‑local font‑size tiers** (k‑means or quantiles).
   * Bold/ALL‑CAPS (language‑aware; see multilingual section).
   * Left margin & indentation clustering to separate peers vs. children.

4. **Lexical Patterns**

   * Enumerations: `1.`, `1.1`, `I.`, `A.`, `Chapter 3`, etc.
   * Handle colon/dot‑leaders; normalize spacing while preserving intentional leading spaces.

> **Export policy:** we always **cap the JSON to H1/H2/H3** even if deeper structure (H4+) is internally recognized.

---

## Multilingual Support (JA/DE/EN/ES)

Our detector is **language‑agnostic** (visual regions), and text extraction relies on PDF text objects (no internet). We tuned heuristics for **Japanese, German, English, and Spanish**:

* **Common foundation**

  * **Font‑size tiers** and **geometry** dominate level inference (robust across scripts).
  * Numbering patterns using **Arabic/Roman numerals** and alphabetic lists (A/B/C) work in all four languages.
  * Unicode‑aware text normalization (NFKC) for consistent comparators.

* **English & Spanish (Latin)**

  * Support accented uppercase/lowercase (e.g., **Á, É, Í, Ó, Ú, Ñ**).
  * ALL‑CAPS and Title‑Case signals are meaningful and used with font‑size.
  * Spanish headings often include punctuation (e.g., “Introducción: …”); colon handling preserved.

* **German (Latin, diacritics & nouns capitalized)**

  * Umlauts and ß (**Ä/Ö/Ü/ä/ö/ü/ß**) preserved; capitalization patterns are informative but not over‑weighted (since all nouns may be capitalized).
  * Hyphenation handling for long compounds (join soft‑hyphen points during span stitching).

* **Japanese (CJK)**

  * Casing is not applicable; we rely more on **font‑size tiers**, **weight**, **indent**, and **enumeration markers** (e.g., `１`, `１．`, `一`, `第一章`) including **full‑width** numerals and punctuation.
  * Heuristics detect **CJK script presence** and down‑weight Latin‑centric cues, preventing false demotions.
  * When available, PDF‑embedded font metadata (e.g., Mincho/Gothic) helps discriminate headings vs. body text.

> **Scanned PDFs:** Visual detection still finds heading boxes, but text content requires OCR. We ship OCR **disabled by default** to keep size/time constraints; multilingual OCR can be enabled locally (see Appendix) but is not required for judging.

---

## Output Schema (Challenge‑compliant)

One **JSON per input PDF**, written to `./output/<filename>.json`:

```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "1. Introduction", "page": 1 },
    { "level": "H2", "text": "1.1 Background",  "page": 2 },
    { "level": "H3", "text": "Key Definitions", "page": 3 }
  ]
}
```

* **Exactly** Title + H1/H2/H3 with **page numbers**.
* We **do not** include bounding boxes in the exported JSON (kept in optional debug artifacts only).

---

## Project Structure

```
Challenge_1a/
├─ models/
│  └─ best.onnx               # student model (≤ 200 MB) baked into the image
├─ input/                     # PDFs to process (mounted at runtime)
├─ output/                    # JSON results (mounted at runtime)
├─ process_pdfs.py            # detection + heuristics + export (auto-run in container)
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

*(Optional: when DEBUG=true, we can emit per‑page debug images and bbox JSON to a separate folder; not used by judges.)*

---

## Setup & Installation (Docker‑only required)

* **Docker (mandatory):** Judges will build/run on **linux/amd64**, **CPU‑only**, **offline**.
* All dependencies and the **model** are **inside the image**; runtime makes **no network calls**.

**Dockerfile (excerpt)**

```dockerfile
# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models
COPY models/best.onnx /app/models/best.onnx

COPY process_pdfs.py /app/process_pdfs.py

ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV NUMEXPR_MAX_THREADS=1

CMD ["python", "/app/process_pdfs.py"]
```

**Build (judge flow):**

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

---

## Running the Pipeline (Docker)

**Linux/macOS:**

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  --network none \
  mysolutionname:somerandomidentifier
```

**Windows PowerShell:**

```powershell
docker run --rm `
  -v "${PWD}\input:/app/input" `
  -v "${PWD}\output:/app/output" `
  --network none `
  mysolutionname:somerandomidentifier
```

**Behavior:**

* On start, the container **auto‑processes** all PDFs in `/app/input`.
* Writes `./output/<filename>.json` for each PDF.
* Worker/thread/DPI/img‑size defaults are tuned **inside the container** for the 8‑CPU judge environment—no flags required.

---

## Evaluation & Benchmarks

* **Student (2‑class) best:**

  * **mAP50 ≈ 0.904**
  * **mAP50–95 ≈ 0.64**

**Notes:**

* Benchmarked on a held‑out validation split.
* Real‑world performance varies with document domain, render DPI, and heuristic strictness; multilingual behavior is primarily driven by layout cues, with language‑specific rules where helpful.

---

## Design Choices & Rationale

1. **Teacher→Student:** strong priors from a large model; lean student for speed and stability.
2. **2‑Class simplification:** less confusion among layout roles → cleaner signals for heuristics.
3. **ONNX (CPU‑only):** portable, deterministic, and Docker‑friendly.
4. **Deterministic heuristics:** reproducible, explainable, and easy to adapt per language family.

---

## Limitations & Edge Cases

* **Highly decorative/stylized headings** may reduce detector confidence.
* **Mixed scripts** on a single line (e.g., EN + JA) can weaken lexical patterns; geometry/size still dominate.
* **Scan‑only PDFs:** OCR optional; disabled in judge image to respect size/time constraints.
* **Magazines/newsletters:** floating decks/sidebars may need additional roles for perfect nesting.

---

## Roadmap

* Lightweight **language‑aware signals** (script detection is in; add compact, offline language ID for tie‑breaks).
* **Self‑training** with teacher to broaden domain coverage.
* **Graph‑based** post‑processing for multi‑column reconciliation.
* Optional **web demo** and **batch API** outside the challenge.

---

## Acknowledgments & License

* Thanks to **DocLayNet** and **Ultralytics YOLOv8** communities.
* This repository is for **Adobe India Hackathon 2025 — Challenge 1A**.
* See `LICENSE` for terms.

---

## Appendix: Local Dev (Optional)

Local Python is **not used by judges** and is provided only for development.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
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

**Optional OCR (local only):**
Add `pytesseract` and system Tesseract with language packs (`eng`, `spa`, `deu`, `jpn`) if you want multilingual OCR for scanned PDFs during development. Keep it disabled for the judge image.

---

**Consistency checklist (done):**

* Export format: **Title + H1/H2/H3 (no bboxes)**.
* Docker is **mandatory**; build/run commands match judge flow.
* CPU‑only, offline, model **≤ 200 MB** baked in.
* Multilingual section added: **Japanese, German, English, Spanish** with concrete heuristics behavior.

---

*If you have questions or want to reproduce the training runs (teacher/student configs, data preparation, and export scripts), open an issue—we’re happy to share additional details.* 🙌
