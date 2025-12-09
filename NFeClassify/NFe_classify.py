# tradein_classify.py
"""
Script enxuto para detecção do quadrado/binário em DANFE (entrada/saída).
Destinado a ser importado e chamado a partir de um Jupyter notebook que já
possui um dicionário `llm_results` com a estrutura:
  llm_results = {
    "43067": {
      "2COMPLETO_page_1.pdf": {...},
      "2COMPLETO_page_2.pdf": {...},
      ...
    },
    ...
  }

Função principal para notebook:
  from tradein_classify import process_root_with_llm
  results = process_root_with_llm(llm_results, ROOT, OUTPUT)

O script não grava debug (nem imagens de debug). Salva crops em:
  OUTPUT/crops/<inquiry>/
E salva um arquivo JSON por inquiry em OUTPUT/<inquiry>.json
"""

import os, json, uuid, traceback
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import pytesseract
from pydantic import BaseModel

# ---- CONFIG ----
DPI = 400
LAYOUT_FALLBACK = (0.45, 0.08, 0.15, 0.15)

# OCR config for single-char recognition
TESS_LANG = "eng"
TESS_SYM_CONF = "--psm 10 -c tessedit_char_whitelist=01"

# Iterative refinement params
REFINE_MAX_ITERS = 3
REFINE_MARGIN_REL = 0.12
REFINE_MIN_AREA_RATIO = 0.004
REFINE_MAX_AREA_RATIO = 0.6
REFINE_TARGET_AREA_RATIO = 0.55
REFINE_MIN_DIM_UPSCALE = 64

DEFAULT_INNER_RATIO = 0.7

# --------------------
class NFResult(BaseModel):
    file: str
    page: int
    entrada_saida: Optional[bool] = None
    raw_text: Optional[str] = None
    occupancy: Optional[float] = None
    method: Optional[str] = None
    crop_path: Optional[str] = None
    notes: Optional[str] = None

# --------------------
# small utilities
# --------------------
def save_crop(img: Image.Image, out_dir: Path, prefix: str, inquiry_id: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_inq{inquiry_id}_{uuid.uuid4().hex[:8]}.png"
    path = out_dir / fname
    img.save(str(path))
    return str(path)

def crop_from_layout(page_img: Image.Image, layout_tuple):
    x_rel,y_rel,w_rel,h_rel = layout_tuple
    W,H = page_img.width, page_img.height
    l = int(W * x_rel); t = int(H * y_rel)
    r = int(min(W, W * (x_rel + w_rel))); b = int(min(H, H * (y_rel + h_rel)))
    l = max(0, min(l, W-1)); t = max(0, min(t, H-1))
    r = max(l+1, min(r, W)); b = max(t+1, min(b, H))
    return page_img.crop((l,t,r,b))

def preprocess_for_occupancy(pil_img: Image.Image):
    gray = pil_img.convert("L")
    gray = ImageOps.autocontrast(gray)
    arr = np.array(gray)
    if arr.size == 0:
        return arr
    _, th = cv2.threshold(arr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)
    return th_inv

def occupancy_of_arr(arr) -> float:
    if arr.size == 0:
        return 0.0
    dark = (arr > 0).sum()
    total = arr.size
    return float(dark)/float(total) if total>0 else 0.0

# --------------------
# OCR + morphology (unchanged approach)
# --------------------
def tight_inner_crop(square: Image.Image, inner_ratio: float = DEFAULT_INNER_RATIO) -> Image.Image:
    W, H = square.size
    l = int(W * (0.5 - inner_ratio/2))
    r = int(W * (0.5 + inner_ratio/2))
    t = int(H * (0.5 - inner_ratio/2))
    b = int(H * (0.5 + inner_ratio/2))
    l = max(0, l); t = max(0, t); r = min(W, r); b = min(H, b)
    if r <= l or b <= t:
        return square
    return square.crop((l,t,r,b))

def classify_symbol_morphology(square_inner: Image.Image) -> Optional[bool]:
    gray = square_inner.convert("L")
    arr = np.array(ImageOps.autocontrast(gray))
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    c = contours[0]
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    if h == 0 or w == 0:
        return None
    aspect = h / float(w)
    solidity = float(area) / (w*h + 1e-9)
    if aspect > 2.5 and solidity < 0.6:
        return True
    if aspect < 2.2 and solidity > 0.45:
        return False
    return None

def ocr_on_inner_multistage(inner_img: Image.Image, stage_scales=((3,4,5), (4,5,6), (6,8))) -> Tuple[Optional[bool], str, Optional[int], Optional[int]]:
    for stage_idx, scales in enumerate(stage_scales, start=1):
        for scale in scales:
            img = inner_img.convert("L")
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            up = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
            try:
                cfg = TESS_SYM_CONF
                txt = pytesseract.image_to_string(up, lang=TESS_LANG, config=cfg)
                txt = (txt or "").strip().replace("\n","").replace(" ","")
                if not txt:
                    continue
                if "1" in txt:
                    return True, txt, stage_idx, scale
                if "0" in txt:
                    return False, txt, stage_idx, scale
                mapped = txt.translate(str.maketrans({"l":"1","I":"1","|":"1","O":"0","o":"0"}))
                if "1" in mapped:
                    return True, txt, stage_idx, scale
                if "0" in mapped:
                    return False, txt, stage_idx, scale
            except Exception:
                continue
    return None, "", None, None

def extract_symbol_from_square_with_3stage(square_img: Image.Image) -> Tuple[Optional[bool], dict]:
    debug: Dict[str, Any] = {}
    inner = tight_inner_crop(square_img, inner_ratio=0.45)
    debug['inner_size'] = inner.size

    val, txt, stage, scale = ocr_on_inner_multistage(inner, stage_scales=((3,4,5),))
    debug['stageA_txt'] = txt
    debug['stageA_scale'] = scale
    if val is not None:
        debug['method'] = 'ocr_stageA'
        debug['stage'] = stage
        debug['used_inner_ratio'] = 0.45
        return val, debug

    larger = tight_inner_crop(square_img, inner_ratio=0.6)
    debug['larger_size'] = larger.size
    val, txt, stage, scale = ocr_on_inner_multistage(larger, stage_scales=((4,5,6),))
    debug['stageB_txt'] = txt
    debug['stageB_scale'] = scale
    if val is not None:
        debug['method'] = 'ocr_stageB'
        debug['stage'] = stage
        debug['used_inner_ratio'] = 0.6
        return val, debug

    almost_full = tight_inner_crop(square_img, inner_ratio=0.85)
    debug['almost_full_size'] = almost_full.size
    val, txt, stage, scale = ocr_on_inner_multistage(almost_full, stage_scales=((6,8),))
    debug['stageC_txt'] = txt
    debug['stageC_scale'] = scale
    if val is not None:
        debug['method'] = 'ocr_stageC'
        debug['stage'] = stage
        debug['used_inner_ratio'] = 0.85
        return val, debug

    m = classify_symbol_morphology(inner)
    debug['morph'] = None if m is None else ("1" if m else "0")
    if m is not None:
        debug['method'] = 'morphology'
        debug['used_inner_ratio'] = 0.45
        return m, debug

    m2 = classify_symbol_morphology(larger)
    debug['morph_larger'] = None if m2 is None else ("1" if m2 else "0")
    if m2 is not None:
        debug['method'] = 'morphology_larger'
        debug['used_inner_ratio'] = 0.6
        return m2, debug

    debug['method'] = 'unreadable'
    return None, debug

# --------------------
# Convert PIL <-> cv2
# --------------------
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_rgb)

# --------------------
# detection configs (relaxed for rectangles)
# --------------------
MIN_BBOX_DIM_PX = 14
MIN_INNER_SYMBOL_AREA = 0.01
ASPECT_MIN_SOFT, ASPECT_MAX_SOFT = 0.4, 2.5
ASPECT_MIN_STRICT, ASPECT_MAX_STRICT = 0.7, 1.6
AREA_RATIO_MIN, AREA_RATIO_MAX = 0.008, 0.35
WHITE_INNER_MIN = 0.28
INNER_DARK_MIN, INNER_DARK_MAX = 0.01, 0.65
INNER_BORDER_REL = 0.12
ARC_LEN_MIN = 28
ASPECT_PENALTY_SCALE = 0.6

# rectangle/border filter
RECTANGULARITY_MIN_STRONG = 0.12
RECTANGULARITY_MIN_WEAK   = 0.06
BORDER_EDGE_RATIO_MIN     = 0.035
BORDER_BAND_REL           = 0.10

# --------------------
# inner analysis / border metrics
# --------------------
def analyze_inner_region(pil_img_warp: Image.Image):
    res = {}
    warp = pil_img_warp.convert("L")
    w, h = warp.size
    arr = np.array(warp)
    white_ratio = float((arr > 200).sum()) / (arr.size + 1e-9)
    res['white_ratio'] = white_ratio

    br = int(min(w,h) * INNER_BORDER_REL)
    inner_l = br; inner_t = br; inner_r = max(1, w - br); inner_b = max(1, h - br)
    if inner_r <= inner_l or inner_b <= inner_t:
        inner_l = int(w*0.2); inner_r = int(w*0.8); inner_t = int(h*0.2); inner_b = int(h*0.8)

    inner = warp.crop((inner_l, inner_t, inner_r, inner_b))
    iarr = np.array(inner)

    _, th = cv2.threshold(iarr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)

    inner_dark_ratio = float((th_inv > 0).sum()) / (th_inv.size + 1e-9)
    res['inner_dark_ratio'] = inner_dark_ratio

    contours, _ = cv2.findContours(th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res['inner_contours_count'] = len(contours)
    if len(contours) == 0:
        res.update({
            'largest_inner_area_ratio': 0.0,
            'largest_inner_aspect': 0.0,
            'largest_inner_solidity': 0.0
        })
        return res

    c = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[0]
    area = cv2.contourArea(c)
    x,y,wc,hc = cv2.boundingRect(c)
    largest_area_ratio = float(area) / (th_inv.size + 1e-9)
    aspect_inner = float(hc) / (wc + 1e-9)
    solidity = float(area) / (wc*hc + 1e-9)

    res.update({
        'largest_inner_area_ratio': largest_area_ratio,
        'largest_inner_aspect': aspect_inner,
        'largest_inner_solidity': solidity,
    })
    return res

def compute_border_metrics(pil_warp: Image.Image):
    warp = pil_warp.convert("L")
    arr = np.array(warp).astype(np.float32)
    h, w = arr.shape[:2]
    band = max(1, int(min(w,h) * BORDER_BAND_REL))

    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mmax = np.percentile(mag, 98) if mag.size>0 else 0.0
    mag_norm = mag / (mmax + 1e-9) if mmax > 0 else mag

    border_mask = np.zeros_like(mag_norm, dtype=np.uint8)
    border_mask[:band, :] = 1; border_mask[-band:, :] = 1
    border_mask[:, :band] = 1; border_mask[:, -band:] = 1

    inner_margin = max(1, min(h,w)//6)
    inner_mask = np.ones_like(mag_norm, dtype=np.uint8)
    inner_mask[:inner_margin, :] = 0; inner_mask[-inner_margin:, :] = 0
    inner_mask[:, :inner_margin] = 0; inner_mask[:, -inner_margin:] = 0

    border_vals = mag_norm[border_mask==1]
    inner_vals = mag_norm[inner_mask==1]

    border_mean = float(np.mean(border_vals)) if border_vals.size>0 else 0.0
    inner_mean = float(np.mean(inner_vals)) if inner_vals.size>0 else 0.0

    thr = max(0.12, np.percentile(mag_norm, 85)) if mag_norm.size>0 else 0.12
    border_edges = (mag_norm[border_mask==1] > thr).sum()
    border_pixels = float((border_mask==1).sum()) + 1e-9
    border_edge_ratio = float(border_edges) / border_pixels

    border_strength = (border_mean / (inner_mean + 1e-9)) if inner_mean > 1e-9 else border_mean * 10.0

    return float(border_edge_ratio), float(border_strength)

def compute_candidate_score(area_ratio, center_dist, rectangularity, inner_metrics, aspect):
    score_area = (area_ratio - AREA_RATIO_MIN) / (AREA_RATIO_MAX - AREA_RATIO_MIN + 1e-9)
    score_area = max(0.0, min(1.0, score_area))
    score_center = max(0.0, 1.0 - center_dist)
    score_rect = max(0.0, min(1.0, rectangularity))

    white_ratio = inner_metrics.get('white_ratio', 0.0)
    inner_dark = inner_metrics.get('inner_dark_ratio', 0.0)
    inner_cnts = inner_metrics.get('inner_contours_count', 0)
    largest_area = inner_metrics.get('largest_inner_area_ratio', 0.0)
    aspect_inner = inner_metrics.get('largest_inner_aspect', 0.0)
    solidity = inner_metrics.get('largest_inner_solidity', 0.0)

    score_white = min(1.0, white_ratio / (WHITE_INNER_MIN + 1e-9))
    score_dark = 0.0
    if INNER_DARK_MIN <= inner_dark <= INNER_DARK_MAX:
        target = (INNER_DARK_MIN + INNER_DARK_MAX) / 2.0
        score_dark = 1.0 - abs(inner_dark - target) / (target + 1e-9)

    score_cnt = min(1.0, inner_cnts / 2.0)
    score_larea = min(1.0, largest_area * 100.0)
    score_aspect_inner = 1.0 if 0.7 <= aspect_inner <= 2.8 else max(0.0, 1.0 - abs(aspect_inner - 1.0)/3.0)
    score_solidity = min(1.0, solidity / 0.5)

    if ASPECT_MIN_STRICT <= aspect <= ASPECT_MAX_STRICT:
        aspect_bonus = 1.0
    elif ASPECT_MIN_SOFT <= aspect <= ASPECT_MAX_SOFT:
        dist = abs(aspect - 1.0)
        norm = min(1.0, dist / (max(1.0, ASPECT_MAX_SOFT - 1.0)))
        aspect_bonus = max(0.0, 1.0 - ASPECT_PENALTY_SCALE * norm)
    else:
        aspect_bonus = 0.0

    w_ext = 0.30
    w_inner = 0.70

    ext_score = 0.5*score_area + 0.3*score_center + 0.2*score_rect
    inner_score = (
        0.40*score_white +
        0.25*score_dark +
        0.15*score_cnt +
        0.10*score_larea +
        0.05*score_aspect_inner +
        0.05*score_solidity
    )

    combined = w_ext * ext_score + w_inner * inner_score
    final = combined * aspect_bonus

    return float(final)

# --------- detect_square_local ----------
def detect_square_local(pil_crop: Image.Image, min_area_ratio=REFINE_MIN_AREA_RATIO, max_area_ratio=REFINE_MAX_AREA_RATIO) -> Optional[Dict[str,Any]]:
    W, H = pil_crop.width, pil_crop.height
    page_area = float(W * H)
    cv_img = pil_to_cv2(pil_crop)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    upscale_factor = 1
    if min(W,H) < REFINE_MIN_DIM_UPSCALE:
        upscale_factor = 2
        gray = cv2.resize(gray, (gray.shape[1]*upscale_factor, gray.shape[0]*upscale_factor), interpolation=cv2.INTER_LINEAR)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v)); upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for c in contours:
        if cv2.arcLength(c, True) < ARC_LEN_MIN:
            continue

        area = cv2.contourArea(c)
        if upscale_factor != 1:
            area = area / (upscale_factor*upscale_factor)
        if area <= 0:
            continue

        x,y,w,h = cv2.boundingRect(c)
        if upscale_factor != 1:
            x = int(x / upscale_factor); y = int(y / upscale_factor)
            w = int(w / upscale_factor); h = int(h / upscale_factor)

        if w < MIN_BBOX_DIM_PX or h < MIN_BBOX_DIM_PX:
            continue

        area_ratio = (w*h) / page_area
        if not (AREA_RATIO_MIN <= area_ratio <= AREA_RATIO_MAX):
            continue

        aspect = w / float(h)
        if not (ASPECT_MIN_SOFT <= aspect <= ASPECT_MAX_SOFT):
            continue

        cx = x + w/2; cy = y + h/2
        center_dist = np.hypot(cx - W/2, cy - H/2) / np.hypot(W/2, H/2)
        rectangularity = area / (w*h + 1e-9)

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            approx_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
        else:
            approx_pts = approx.reshape(-1,2).astype(np.float32)

        def order_pts(pts):
            s = pts.sum(axis=1); diff = np.diff(pts, axis=1).reshape(-1)
            tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
            return np.array([tl, tr, br, bl], dtype=np.float32)

        rect = order_pts(approx_pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
        maxW = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
        maxH = int(max(heightA, heightB))
        if maxW <= 0 or maxH <= 0:
            continue

        dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
        if upscale_factor != 1:
            rect = rect / float(upscale_factor)
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(cv_img, M, (maxW, maxH))
        pil_warp = cv2_to_pil(warp)

        inner_metrics = analyze_inner_region(pil_warp)
        border_edge_ratio, border_strength = compute_border_metrics(pil_warp)
        inner_metrics['border_edge_ratio'] = border_edge_ratio
        inner_metrics['border_strength'] = border_strength

        if inner_metrics['white_ratio'] < WHITE_INNER_MIN:
            continue
        if not (INNER_DARK_MIN <= inner_metrics['inner_dark_ratio'] <= INNER_DARK_MAX):
            continue

        if rectangularity < RECTANGULARITY_MIN_WEAK and inner_metrics.get('border_edge_ratio',0.0) < BORDER_EDGE_RATIO_MIN:
            continue

        final_score = compute_candidate_score(area_ratio, center_dist, rectangularity, inner_metrics, aspect)

        candidates.append((final_score, pil_warp, {
            "bbox": (x,y,w,h),
            "area_ratio": area_ratio,
            "center_dist": center_dist,
            "rectangularity": rectangularity,
            "inner_metrics": inner_metrics,
            "aspect": aspect
        }))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_warp, best_meta = candidates[0]

    return {
        "warped": best_warp,
        "score": float(best_score),
        "meta": best_meta
    }

# --------------------
# refine iterativo (sem debug pesado)
# --------------------
def refine_crop_iteratively(initial_crop: Image.Image, output_crops_dir: Path, max_iters: int = REFINE_MAX_ITERS, margin_rel: float = REFINE_MARGIN_REL) -> Tuple[Image.Image, dict]:
    debug: Dict[str, Any] = {"iterations": []}
    current = initial_crop.copy()
    best_crop = current
    best_score = -999.0

    for it in range(1, max_iters+1):
        det = detect_square_local(current)
        if det is None:
            debug["iterations"].append({"iter": it, "found": False})
            # morphological fallback
            gray = np.array(current.convert("L"))
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = cv2.bitwise_not(th)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
            closed_pil = Image.fromarray(cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB))
            det2 = detect_square_local(closed_pil)
            if det2 is None:
                debug["iterations"].append({"iter": it, "found_after_morph": False})
                break
            else:
                det = det2
                debug["iterations"].append({"iter": it, "found_after_morph": True})
        else:
            debug["iterations"].append({"iter": it, "found": True})

        warped: Image.Image = det["warped"]
        w_ = warped.width; h_ = warped.height
        margin = int(max(w_, h_) * margin_rel)
        padded = Image.new("RGB", (w_ + 2*margin, h_ + 2*margin), (255,255,255))
        padded.paste(warped, (margin, margin))
        current = padded

        area_ratio = det.get("meta", {}).get("area_ratio", 0.0)
        score = det.get("score", 0.0)
        debug["iterations"][-1].update({
            "area_ratio": area_ratio,
            "score": score,
            "warped_size": (warped.width, warped.height),
            "padded_size": (padded.width, padded.height)
        })

        if score > best_score:
            best_score = score
            best_crop = current.copy()

        bbox_area = warped.width * warped.height
        padded_area = padded.width * padded.height
        percent = bbox_area / float(padded_area) if padded_area>0 else 0.0
        debug["iterations"][-1]["bbox_area_pct_of_padded"] = percent

        prev_sizes = [itd.get('warped_size') for itd in debug['iterations'][:-1] if itd.get('warped_size')]
        if prev_sizes:
            prev_w, prev_h = prev_sizes[-1]
            cur_w, cur_h = warped.width, warped.height
            pct_w = abs(cur_w - prev_w) / (prev_w + 1e-9)
            pct_h = abs(cur_h - prev_h) / (prev_h + 1e-9)
            if pct_w < 0.06 and pct_h < 0.06:
                debug["stagnated_on_iter"] = it
                debug["converged"] = False
                debug["reason"] = "stagnation_size"
                return best_crop, debug

        if percent >= REFINE_TARGET_AREA_RATIO:
            debug["converged"] = True
            debug["converged_iter"] = it
            return current, debug

    debug["converged"] = False
    debug["best_score"] = best_score
    debug["best_crop_size"] = best_crop.size
    return best_crop, debug

# --------------------
# Process one PDF (page 1 by default)
# --------------------
def process_single_pdf(pdf_path: Path, output_dir: Path, inquiry_id: str) -> NFResult:
    out = NFResult(file=str(pdf_path), page=1)
    try:
        pages = convert_from_path(str(pdf_path), dpi=DPI, first_page=1, last_page=1)
        if not pages:
            out.notes = "no_pages"
            return out
        page_img = pages[0]

        # initial crop: docling omitted here; fallback to layout
        crop = crop_from_layout(page_img, LAYOUT_FALLBACK)

        # save raw crop inside inquiry crops dir
        crop_dir = output_dir / "crops" / inquiry_id
        raw_crop_path = save_crop(crop, crop_dir, prefix=f"{inquiry_id}_{pdf_path.stem}_raw", inquiry_id=inquiry_id)
        out.crop_path = raw_crop_path

        # refine iteratively
        refined_crop, refine_debug = refine_crop_iteratively(crop, crop_dir, max_iters=REFINE_MAX_ITERS, margin_rel=REFINE_MARGIN_REL)
        refined_path = save_crop(refined_crop, crop_dir, prefix=f"{inquiry_id}_{pdf_path.stem}_refined", inquiry_id=inquiry_id)

        inner = tight_inner_crop(refined_crop, inner_ratio=0.7)
        inner_path = save_crop(inner, crop_dir, prefix=f"{inquiry_id}_{pdf_path.stem}_inner", inquiry_id=inquiry_id)

        arr = preprocess_for_occupancy(inner)
        out.occupancy = occupancy_of_arr(arr)

        val, debug = extract_symbol_from_square_with_3stage(refined_crop)
        out.raw_text = None
        if val is not None:
            out.entrada_saida = val
            out.method = debug.get('method','ocr_3stage_refined')
            out.raw_text = "1" if val else "0"
            out.notes = "refine_debug:" + json.dumps(refine_debug, ensure_ascii=False) + " | ocr_debug:" + json.dumps(debug, ensure_ascii=False)
        else:
            out.entrada_saida = None
            out.method = "unreadable"
            out.notes = "refine_debug:" + json.dumps(refine_debug, ensure_ascii=False) + " | ocr_debug:" + json.dumps(debug, ensure_ascii=False)

        out.crop_path = inner_path

    except Exception as e:
        out.notes = f"error:{e}"
        out.notes = out.notes + " | " + traceback.format_exc().splitlines()[-1]
    return out

# --------------------
# High-level: process dictionary from notebook
# --------------------
def process_root_with_llm(filtered_llm, root_path, out_path, class_filter: str = "0") -> Dict[str, Any]:
    """
    Processa apenas os PDFs que, no dicionário llm_results, têm metadata['class'] == class_filter.
    root_path: pasta raiz com subpastas por inquiry
    out_path: pasta de saída
    class_filter: string do valor de 'class' a processar (padrão '0')
    Retorna: dict com resultados por inquiry (também grava OUTPUT/<inquiry>.json)
    """
    root = Path(root_path)
    output = Path(out_path)
    output.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for inquiry_id, files in filtered_llm.items():
        inquiry_folder = root / str(inquiry_id)
        inquiry_results = []
        for fname, meta in files.items():
            # se o arquivo no llm_results não tem class igual ao filtro, pular
            file_class = (meta.get("class") if isinstance(meta, dict) else None)
            if str(file_class) != str(class_filter):
                # opcional: registrar que foi pulado
                r = NFResult(file=str(inquiry_folder / fname), page=0, notes=f"skipped_class={file_class}")
                inquiry_results.append(r.model_dump())
                continue

            # localizar o pdf na pasta (nome exato ou com variações)
            pdf_path = inquiry_folder / fname
            if not pdf_path.exists():
                alt = list(inquiry_folder.glob(f"{Path(fname).stem}*.pdf"))
                pdf_path = alt[0] if alt else None

            if pdf_path is None or not pdf_path.exists():
                r = NFResult(file=str(inquiry_folder / fname), page=0, notes="missing_pdf")
                inquiry_results.append(r.model_dump())
                continue

            # processa apenas arquivos com class == class_filter
            r = process_single_pdf(pdf_path, output, inquiry_id=str(inquiry_id))
            inquiry_results.append(r.model_dump())

        # grava json por inquiry (mesmo contendo skipped/missing entries)
        out_file = output / f"{inquiry_id}.json"
        with open(out_file, "w", encoding="utf8") as fh:
            json.dump(inquiry_results, fh, ensure_ascii=False, indent=2)
        all_results[inquiry_id] = inquiry_results

    return all_results

# optional command-line usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify NF-e entradas/saidas from PDFs (minimal mode).")
    parser.add_argument("--root", required=True, help="Root folder with inquiries subfolders")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--llm_json", required=False, help="(optional) path to JSON with llm_results structure")
    args = parser.parse_args()

    if args.llm_json:
        with open(args.llm_json, "r", encoding="utf8") as fh:
            llm = json.load(fh)
        res = process_root_with_llm(llm, args.root, args.output)
        print("Done. inquiries:", list(res.keys()))
    else:
        print("This script is meant to be imported and called from a notebook with llm_results dict.")
