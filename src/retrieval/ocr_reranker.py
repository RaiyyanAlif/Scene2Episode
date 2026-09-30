import os
import re
import json
import argparse
from collections import defaultdict

import cv2
import pytesseract


# ============================================================
# CONFIG
# ============================================================

DEFAULT_TOP_K = 15

# OCR preprocessing variants
OCR_SCALE = 2.0

# Tesseract configuration
TESSERACT_CONFIG = "--oem 3 --psm 6"


# ============================================================
# TESSERACT PATH
# ============================================================

# Windows installation path
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# ============================================================
# OCR
# ============================================================

def preprocess_image(image):
    """
    Prepare frame for OCR.

    Returns several versions because text can appear
    differently depending on contrast/background.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale
    scaled = cv2.resize(
        gray,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )

    # Normal grayscale
    normal = scaled

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        scaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Otsu threshold
    _, otsu = cv2.threshold(
        scaled,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return [
        normal,
        adaptive,
        otsu,
    ]


def clean_text(text):
    """
    Normalize OCR output.
    """

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 .,'!?():;'\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def ocr_frame(image_path):
    """
    Run OCR using multiple preprocessing variants.
    """

    if not os.path.exists(image_path):
        return ""

    image = cv2.imread(image_path)

    if image is None:
        return ""

    variants = preprocess_image(image)

    texts = []

    for variant in variants:
        try:
            text = pytesseract.image_to_string(
                variant,
                config=TESSERACT_CONFIG,
            )

            text = clean_text(text)

            if text:
                texts.append(text)

        except Exception as exc:
            print(f"OCR error: {exc}")

    # Remove duplicates while preserving order
    unique = []

    for text in texts:
        if text not in unique:
            unique.append(text)

    return " | ".join(unique)


# ============================================================
# TEXT SIMILARITY
# ============================================================

def normalize_tokens(text):
    """
    Convert OCR text into useful tokens.
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-z0-9]{2,}",
        text,
    )

    return set(tokens)


def text_similarity(query_text, candidate_text):
    """
    Jaccard similarity between OCR token sets.
    """

    if not query_text or not candidate_text:
        return 0.0

    q = normalize_tokens(query_text)
    c = normalize_tokens(candidate_text)

    if not q or not c:
        return 0.0

    intersection = len(q & c)
    union = len(q | c)

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD INDEX / METADATA
# ============================================================

def load_metadata(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def episode_key(item):
    return (
        item["media_id"],
        item.get("season"),
        item.get("episode"),
    )


# ============================================================
# OCR CACHE
# ============================================================

def load_ocr_cache(path):
    if not os.path.exists(path):
        return {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception:
        return {}


def save_ocr_cache(cache, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# OCR CANDIDATE ANALYSIS
# ============================================================

def get_candidate_ocr(
    metadata,
    candidate_indices,
    cache,
):
    """
    OCR candidate frames.

    Returns:
        {
            frame_index: OCR text
        }
    """

    results = {}

    for idx in candidate_indices:

        item = metadata[idx]

        path = item["image_path"]

        if path in cache:
            text = cache[path]

        else:
            text = ocr_frame(path)
            cache[path] = text

        results[idx] = text

    return results


# ============================================================
# EPISODE OCR SCORE
# ============================================================

def score_episode_ocr(
    query_ocr,
    candidate_indices,
    metadata,
    cache,
):
    """
    Calculate OCR evidence for one candidate episode.
    """

    if not query_ocr:
        return 0.0, 0

    candidate_texts = []

    for idx in candidate_indices:

        item = metadata[idx]

        path = item["image_path"]

        if path in cache:
            text = cache[path]

        else:
            text = ocr_frame(path)
            cache[path] = text

        if text:
            candidate_texts.append(text)

    if not candidate_texts:
        return 0.0, 0

    similarities = [
        text_similarity(
            query_ocr,
            text,
        )
        for text in candidate_texts
    ]

    similarities.sort(reverse=True)

    # Strongest matching OCR evidence matters most
    top_scores = similarities[:3]

    score = sum(top_scores) / len(top_scores)

    return score, len(candidate_texts)


# ============================================================
# DEMO TEST
# ============================================================

def test_ocr_on_frame(image_path):

    print("=" * 80)
    print("SCENE2EPISODE OCR TEST")
    print("=" * 80)

    print()
    print("Frame:")
    print(image_path)

    print()
    print("Running OCR...")

    text = ocr_frame(image_path)

    print()
    print("OCR RESULT")
    print("-" * 80)

    if text:
        print(text)
    else:
        print("[No readable text detected]")

    print("-" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        help="Run OCR on a single image",
    )

    parser.add_argument(
        "--cache",
        type=str,
        default="metadata/ocr_cache.json",
    )

    args = parser.parse_args()

    if args.image:
        test_ocr_on_frame(
            args.image,
        )
        return

    print("=" * 80)
    print("SCENE2EPISODE OCR MODULE")
    print("=" * 80)

    print()
    print("Tesseract:")
    print(pytesseract.get_tesseract_version())

    print()
    print("OCR module ready.")


if __name__ == "__main__":
    main()