import os
import re
import cv2
import numpy as np

DEFAULT_OCR_MIN_CONF_PCT = float(os.environ.get("PLATE_OCR_MIN_CONF_PCT", "45"))
DEFAULT_PLATE_REC_SCORE_THRESH = float(os.environ.get("PLATE_REC_SCORE_THRESH", "0.12"))


def _preprocess_plate_crop(img: np.ndarray) -> np.ndarray:
    """Upscale small crops and boost contrast for OCR."""
    if img is None or img.size == 0:
        return img
    if not isinstance(img, np.ndarray):
        return img
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    if h < 1 or w < 1:
        return img

    min_side = min(h, w)
    if min_side < 56:
        scale = 56.0 / float(min_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _extract_lines_from_ocr_result(ocr_result) -> list[tuple[str, float]]:
    """
    PaddleOCR 2.x: [ [ [box, (text, score)], ... ] ]
    PaddleOCR 3.x / PaddleX: [ OCRResult dict with rec_texts, rec_scores ]
    """
    lines: list[tuple[str, float]] = []
    if not ocr_result:
        return lines

    first = ocr_result[0]

    # New API: dict-like result per image
    if isinstance(first, dict) or (hasattr(first, "get") and hasattr(first, "keys")):
        try:
            rec_texts = first["rec_texts"] if "rec_texts" in first else first.get("rec_texts", [])
        except Exception:
            rec_texts = getattr(first, "rec_texts", None) or []
        try:
            rec_scores = first["rec_scores"] if "rec_scores" in first else first.get("rec_scores", [])
        except Exception:
            rec_scores = getattr(first, "rec_scores", None) or []

        if not rec_texts:
            return lines

        if not rec_scores:
            rec_scores = [1.0] * len(rec_texts)
        for t, s in zip(rec_texts, rec_scores):
            if isinstance(t, tuple):
                t = t[0]
            if t is None:
                continue
            try:
                sc = float(s)
            except (TypeError, ValueError):
                sc = 0.0
            lines.append((str(t).strip(), sc))
        return lines

    # Legacy: list of [box, (text, score)]
    if isinstance(first, (list, tuple)):
        for line in first:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            content = line[1]
            if isinstance(content, (list, tuple)) and len(content) >= 2:
                text = content[0]
                score = content[1]
            elif isinstance(content, str):
                text = content
                score = 0.0
            else:
                continue
            lines.append((str(text).strip(), float(score)))

    return lines


def _pick_best_plate(text_scores: list[tuple[str, float]], min_conf_pct: float) -> tuple[str | None, float | None]:
    """Choose best line; prefer high confidence and plate-like length."""
    if not text_scores:
        return None, None

    combined = re.sub(r"[^a-zA-Z0-9]", "", "".join(t for t, _ in text_scores))

    best_text = None
    best_s = -1.0
    for text, s in text_scores:
        clean = re.sub(r"[^a-zA-Z0-9]", "", text)
        if len(clean) < 3:
            continue
        if (s * 100.0) >= min_conf_pct and s >= best_s:
            best_text = clean
            best_s = s

    if best_text:
        return best_text, best_s

    # Fallback: highest score line with enough characters
    for text, s in sorted(text_scores, key=lambda x: -x[1]):
        clean = re.sub(r"[^a-zA-Z0-9]", "", text)
        if len(clean) >= 4:
            return clean, s

    if len(combined) >= 4:
        return combined, 0.5

    return None, None


def predict_number_plate(img, ocr, min_conf_pct: float | None = None):
    if img is None or (isinstance(img, np.ndarray) and img.size == 0):
        return None, None

    img = _preprocess_plate_crop(img)
    thresh = DEFAULT_OCR_MIN_CONF_PCT if min_conf_pct is None else min_conf_pct
    rec_thresh = DEFAULT_PLATE_REC_SCORE_THRESH

    # Prefer predict() with low rec threshold so weak plate reads are kept
    try:
        ocr_result = ocr.predict(
            img,
            text_rec_score_thresh=rec_thresh,
        )
    except TypeError:
        ocr_result = ocr.ocr(img)
    except Exception:
        ocr_result = ocr.ocr(img)

    if not ocr_result:
        return None, None

    lines = _extract_lines_from_ocr_result(ocr_result)
    if not lines:
        return None, None

    text, score = _pick_best_plate(lines, thresh)
    if text:
        return text, score
    return None, None
