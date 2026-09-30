import argparse
import os
import re

import cv2
import pytesseract


# ============================================================
# CONFIG
# ============================================================

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# ============================================================
# OCR
# ============================================================

def extract_ocr(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Upscale for small subtitles/text.
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    results = []

    # --------------------------------------------------------
    # Variant 1: normal grayscale
    # --------------------------------------------------------

    text = pytesseract.image_to_string(
        gray,
        config="--oem 3 --psm 6",
    )

    text = clean_text(text)

    if text:
        results.append(text)

    # --------------------------------------------------------
    # Variant 2: adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    text = pytesseract.image_to_string(
        adaptive,
        config="--oem 3 --psm 6",
    )

    text = clean_text(text)

    if text:
        results.append(text)

    # --------------------------------------------------------
    # Variant 3: Otsu threshold
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    text = pytesseract.image_to_string(
        otsu,
        config="--oem 3 --psm 6",
    )

    text = clean_text(text)

    if text:
        results.append(text)

    return results


# ============================================================
# MERGE OCR RESULTS
# ============================================================

def merge_results(results):

    if not results:
        return ""

    unique = []

    for text in results:

        normalized = text.lower()

        duplicate = False

        for existing in unique:

            if normalized == existing.lower():

                duplicate = True
                break

        if not duplicate:

            unique.append(text)

    return "\n".join(unique)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Scene2Episode OCR analyzer"
    )

    parser.add_argument(
        "image",
        help="Path to query image",
    )

    args = parser.parse_args()

    image_path = os.path.abspath(
        args.image
    )

    if not os.path.exists(image_path):

        print(
            f"ERROR: Image not found:\n"
            f"{image_path}"
        )

        return

    print()
    print("=" * 80)
    print("SCENE2EPISODE OCR ANALYZER")
    print("=" * 80)

    print(
        f"Image: {image_path}"
    )

    print()
    print("Running OCR...")

    results = extract_ocr(
        image_path
    )

    merged = merge_results(
        results
    )

    print()
    print("-" * 80)
    print("OCR RESULT")
    print("-" * 80)

    if merged:

        print(merged)

    else:

        print(
            "No readable text detected."
        )

    print("-" * 80)
    print()


if __name__ == "__main__":
    main()