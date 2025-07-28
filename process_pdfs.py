#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import fitz                # pip install PyMuPDF
import numpy as np         # pip install numpy
import onnxruntime as ort  # pip install onnxruntime
import cv2                 # pip install opencv-python
import pytesseract         # pip install pytesseract
from pytesseract import Output
import re
import os
from sklearn.cluster import KMeans  # pip install scikit-learn

IOU_THRES = 0.45

def letterbox(img: np.ndarray, size: int, pad_val: int = 114):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), pad_val, dtype=np.uint8)
    dy, dx  = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas, scale, dx, dy

def xywh2xyxy(xywh: np.ndarray):
    x, y, w, h = xywh.T
    return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres=IOU_THRES):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        inter = (np.minimum(boxes[i:i+1,2:], boxes[idxs[1:],2:]) -
                 np.maximum(boxes[i:i+1,:2], boxes[idxs[1:],:2])).clip(0)
        area1 = np.prod(boxes[i,2:] - boxes[i,:2])
        area2 = np.prod(boxes[idxs[1:],2:] - boxes[idxs[1:],:2], axis=1)
        ious  = inter.prod(1) / (area1 + area2 - inter.prod(1) + 1e-7)
        idxs  = idxs[1:][ious < iou_thres]
    return keep

def robust_postprocess(pred: np.ndarray, conf_thres: float):
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 2 and pred.shape[0] == 6 and pred.shape[1] != 6:
        pred = pred.T
    elif pred.ndim > 2:
        pred = pred.reshape(-1, pred.shape[-1])
    if pred.ndim != 2 or pred.shape[1] < 6:
        return np.empty((0,6), dtype=np.float32)

    boxes  = pred[:,:4]
    scores = pred[:,4:]
    if scores.shape[1] == 1:
        confs   = scores[:,0]
        cls_ids = np.zeros_like(confs, dtype=int)
    else:
        cls_ids = np.argmax(scores, axis=1)
        confs   = np.max(scores, axis=1)

    mask = confs >= conf_thres
    if not mask.any():
        return np.empty((0,6), dtype=np.float32)

    boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]
    xyxy = xywh2xyxy(boxes)

    out = []
    for c in np.unique(cls_ids):
        ix = np.where(cls_ids == c)[0]
        for i in nms(xyxy[ix], confs[ix], IOU_THRES):
            x0, y0, x1, y1 = xyxy[ix[i]]
            out.append([x0, y0, x1, y1, confs[ix[i]], int(c)])
    return np.array(out, dtype=np.float32) if out else np.empty((0,6), dtype=np.float32)

def pix_to_np(pix: fitz.Pixmap):
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if arr.shape[2] == 4:
        arr = arr[:,:,:3]
    return arr[:,:,::-1]

def detect_document_title(items):
    # if nothing, bail
    if not items:
        return "Untitled Document", set()

    # compute median OCR font size over all items
    all_fs = [it.get('font_size', 0.0) for it in items]
    median_fs = np.median(all_fs)

    # find first Title (or fallback to first Section-header)
    title_idxs = [i for i,it in enumerate(items) if it['label']=='Title']
    if title_idxs:
        first = title_idxs[0]
    else:
        sec_idxs = [i for i,it in enumerate(items) if it['label']=='Section-header']
        if not sec_idxs:
            return "Untitled Document", set()
        first = min(sec_idxs, key=lambda i: (items[i]['page'], items[i]['bbox'][1]))

    # prepend any Section-headers immediately above as part of the doc title
    used, texts = [], []
    for i, it in enumerate(items[:first]):
        if it['label']=='Section-header':
            texts.append(it['text'])
            used.append(i)
    texts.append(items[first]['text']); used.append(first)
    doc_title = ' '.join(texts).strip()

    # relabel any subsequent Title whose font_size > median_fs
    for ti in title_idxs[1:]:
        if items[ti].get('font_size', 0.0) > median_fs:
            items[ti]['label'] = 'Section-header'

    return doc_title, set(used)


def assign_heading_levels(items):
    import numpy as np
    import re

    # 1) Numeric prefixes: "1.", "2.1", etc.
    for it in items:
        m = re.match(r'^\s*(\d+(?:\.\d+)*)', it['text'])
        if m:
            dots = m.group(1).count('.')
            it['level'] = f"H{min(dots+1, 4)}"
        else:
            it['level'] = None

    # 2) Fallback: use font_size percentiles
    unlabeled = [it for it in items if it['level'] is None]
    if unlabeled:
        # gather all font sizes
        fs = np.array([it.get('font_size', 0.0) for it in unlabeled], dtype=float)
        # compute quartiles
        p25, p50, p75 = np.percentile(fs, [25, 50, 75])

        # assign levels by where each font_size falls
        for it in unlabeled:
            f = it.get('font_size', 0.0)
            if f >= p75:
                lvl = "H1"
            elif f >= p50:
                lvl = "H2"
            elif f >= p25:
                lvl = "H3"
            else:
                lvl = "H4"
            it['level'] = lvl

    return items




def build_outline(items, used_idxs, has_numbered):
    """
    Build the JSON outline.  If the document has no numbered headings at all,
    only keep level 'H1'; otherwise keep everything.
    """
    outline = []
    for idx, it in sorted(
        enumerate(items),
        key=lambda x: (x[1]['page'], x[1]['bbox'][1])
    ):
        # skip the document title itself and any items we flagged as part of the title
        if idx in used_idxs or it['label'] == 'Title':
            continue

        # if there were no numeric prefixes anywhere, drop anything that's not H1
        if not has_numbered and it['level'] != 'H1':
            continue

        outline.append({
            'level': it['level'],
            'text':  it['text'],
            'page':  it['page']
        })
    return outline

def configure_tesseract():
    """Configure Tesseract OCR paths for Windows"""
    if os.name == 'nt':  # Windows
        # Common Tesseract installation paths
        possible_paths = [
            r"C:\Users\agrim\scoop\persist\tesseract\tessdata",
            r"C:\Program Files\Tesseract-OCR\tessdata",
            r"C:\Users\{}\scoop\persist\tesseract\tessdata".format(os.getenv('USERNAME')),
            r"C:\tools\tesseract\tessdata"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['TESSDATA_PREFIX'] = path
                print(f"Set TESSDATA_PREFIX to: {path}")
                return path
        
        print("Warning: Could not find Tesseract tessdata directory")
        return None
    return None

def process_page(page_data):
    """Process a single page – now with font_size from OCR."""
    page_num, pix, sess, sz, conf_thres, print_lock, tessdata_prefix = page_data

    # set TESSDATA_PREFIX if needed
    if tessdata_prefix:
        os.environ['TESSDATA_PREFIX'] = tessdata_prefix

    img = pix_to_np(pix)
    pad, scale, dx, dy = letterbox(img, sz)
    inp = pad.transpose(2,0,1)[None].astype(np.float32) / 255.0
    pred = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    dets = robust_postprocess(pred, conf_thres)

    with print_lock:
        print(f" page {page_num+1}: {len(dets)} boxes")

    page_items = []
    for x0, y0, x1, y1, conf, cls_id in dets:
        x0o = int((x0 - dx) / scale)
        y0o = int((y0 - dy) / scale)
        x1o = int((x1 - dx) / scale)
        y1o = int((y1 - dy) / scale)
        roi = img[y0o:y1o, x0o:x1o]

        # OCR with data output for font-size
        try:
            ocr = pytesseract.image_to_data(
                roi, config="--psm 6", output_type=Output.DICT
            )
            # reconstruct text
            words = [w for w in ocr['text'] if w.strip()]
            txt = " ".join(words)
            # median of word-box heights
            hs = [h for h in ocr['height'] if h>0]
            font_size = float(np.median(hs)) if hs else float(y1o-y0o)
        except Exception as e:
            with print_lock:
                print(f"OCR error on page {page_num+1}: {e}")
            txt = ""
            font_size = float(y1o - y0o)

        page_items.append({
            'page': page_num,
            'label': 'Title' if cls_id==0 else 'Section-header',
            'bbox': [float(x0o), float(y0o), float(x1o), float(y1o)],
            'conf': float(conf),
            'text': txt,
            'font_size': font_size,
            'level': None
        })

    return page_num, page_items

def make_json(model_path, input_dir, output_dir, conf_thres, dpi, threads, max_workers=8):
    # Configure Tesseract OCR
    tessdata_prefix = configure_tesseract()
    
    # Create ONNX session
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    shp = sess.get_inputs()[0].shape
    sz = int(shp[2] or 640)
    assert sz == int(shp[3] or 640)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(threads, 4)
    
    print_lock = Lock()

    for pdf_path in sorted(input_dir.glob("*.pdf")):
        print(f"[PDF] {pdf_path.name}")
        doc = fitz.open(pdf_path)
        
        # prepare page data
        page_data_list = []
        for p in range(len(doc)):
            pix = doc[p].get_pixmap(dpi=dpi)
            page_data_list.append((p, pix, sess, sz, conf_thres, print_lock, tessdata_prefix))
        
        # process pages in parallel
        items = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(process_page, page_data): page_data[0]
                for page_data in page_data_list
            }
            page_results = {}
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_num_result, page_items = future.result()
                    page_results[page_num_result] = page_items
                except Exception as exc:
                    print(f"Page {page_num} generated an exception: {exc}")
                    page_results[page_num] = []
        
        # collect and sort
        for page_num in sorted(page_results.keys()):
            items.extend(page_results[page_num])
        doc.close()
        items.sort(key=lambda it: (it['page'], it['bbox'][1]))
        
        # detect title & used indices
        title, used_idxs = detect_document_title(items)
        # assign H1–H4
        items = assign_heading_levels(items)
        
        # NEW: check for any numeric prefixes in the text
        has_numbered = any(
            re.match(r'^\s*\d+(?:\.\d+)*', it['text'])
            for it in items
        )
        # NEW: pass that flag into build_outline
        outline = build_outline(items, used_idxs, has_numbered)
        
        # write JSON
        result = {'title': title, 'outline': outline}
        out = output_dir / f"{pdf_path.stem}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"→ wrote {out.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect headings in PDFs and output title+outline JSON (with parallel page processing)"
    )
    parser.add_argument("--model", type=Path, required=True, help="path to .onnx")
    parser.add_argument("--input_dir", type=Path, required=True, help="folder of PDFs")
    parser.add_argument("--output_dir", type=Path, required=True, help="where to save JSON")
    parser.add_argument("--conf_thres", type=float, default=0.15, help="confidence threshold")
    parser.add_argument("--dpi", type=int, default=150, help="render DPI")
    parser.add_argument("--threads", type=int, default=8, help="onnxruntime threads")
    parser.add_argument("--max_workers", type=int, default=None, help="max parallel page workers (default: 8)")
    args = parser.parse_args()
    make_json(args.model, args.input_dir, args.output_dir, args.conf_thres, args.dpi, args.threads, args.max_workers)