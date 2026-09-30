import os
import re
import json
from collections import defaultdict

import cv2
import faiss
import numpy as np
import pytesseract


# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = "indexes/scene2episode.index"
METADATA_PATH = "indexes/scene2episode_metadata.json"
OCR_CACHE_PATH = "metadata/ocr_cache.json"

TOP_K = 50

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# ============================================================
# LOAD
# ============================================================

def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_cache():

    if not os.path.exists(OCR_CACHE_PATH):
        return {}

    with open(
        OCR_CACHE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# HELPERS
# ============================================================

def episode_key(item):

    return (
        item["media_id"],
        item.get("season"),
        item.get("episode"),
    )


def episode_name(item):

    if item.get("season") is None:
        return item["title"]

    return (
        f"{item['title']} "
        f"S{item['season']:02d} "
        f"E{item['episode']:02d}"
    )


def clean_text(text):

    text = text.replace(
        "\n",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^A-Za-z0-9 .,'!?():;\-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_tokens(text):

    return set(
        re.findall(
            r"[a-z0-9]{2,}",
            text.lower(),
        )
    )


def text_similarity(a, b):

    if not a or not b:
        return 0.0

    a_tokens = normalize_tokens(a)
    b_tokens = normalize_tokens(b)

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(
        a_tokens & b_tokens
    )

    union = len(
        a_tokens | b_tokens
    )

    return intersection / union


# ============================================================
# OCR
# ============================================================

def preprocess_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    scaled = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    adaptive = cv2.adaptiveThreshold(
        scaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    _, otsu = cv2.threshold(
        scaled,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return [
        scaled,
        adaptive,
        otsu,
    ]


def run_ocr(image_path):

    if not os.path.exists(image_path):
        return ""

    image = cv2.imread(image_path)

    if image is None:
        return ""

    texts = []

    for variant in preprocess_image(image):

        try:

            text = pytesseract.image_to_string(
                variant,
                config="--oem 3 --psm 6",
            )

            text = clean_text(text)

            if text:
                texts.append(text)

        except Exception as exc:

            print(
                f"OCR error: {exc}"
            )

    unique = []

    for text in texts:

        if text not in unique:
            unique.append(text)

    return " | ".join(unique)


def get_ocr(
    image_path,
    cache,
):

    if image_path in cache:
        return cache[image_path]

    text = run_ocr(image_path)

    cache[image_path] = text

    return text


# ============================================================
# QUERY
# ============================================================

def find_query_index(
    metadata,
    media_id,
    timestamp,
):

    best_idx = None
    best_diff = float("inf")

    for i, item in enumerate(metadata):

        if item["media_id"] != media_id:
            continue

        diff = abs(
            item["timestamp_seconds"]
            - timestamp
        )

        if diff < best_diff:

            best_diff = diff
            best_idx = i

    return best_idx


# ============================================================
# RETRIEVE ACTUAL CANDIDATES
# ============================================================

def retrieve_candidates(
    index,
    query_index,
):

    query_vector = index.reconstruct(
        query_index
    )

    query_vector = (
        query_vector
        .reshape(1, -1)
        .astype("float32")
    )

    scores, indices = index.search(
        query_vector,
        TOP_K + 1,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        if idx == query_index:
            continue

        results.append(
            (
                int(idx),
                float(score),
            )
        )

        if len(results) >= TOP_K:
            break

    return results


# ============================================================
# TEMPORAL QUERY FRAMES
# ============================================================

def find_nearest_frame(
    metadata,
    query_item,
    target_time,
):

    best_idx = None
    best_diff = float("inf")

    for i, item in enumerate(metadata):

        if item["media_id"] != query_item["media_id"]:
            continue

        if item.get("season") != query_item.get("season"):
            continue

        if item.get("episode") != query_item.get("episode"):
            continue

        diff = abs(
            item["timestamp_seconds"]
            - target_time
        )

        if diff < best_diff:

            best_diff = diff
            best_idx = i

    return best_idx


def get_query_clip(
    metadata,
    query_index,
):

    query_item = metadata[
        query_index
    ]

    offsets = [
        -4,
        -2,
        0,
        2,
        4,
    ]

    indices = []

    for offset in offsets:

        idx = find_nearest_frame(
            metadata,
            query_item,
            query_item[
                "timestamp_seconds"
            ] + offset,
        )

        if idx is not None:
            indices.append(idx)

    return list(
        dict.fromkeys(indices)
    )


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():

    print("=" * 110)
    print(
        "SCENE2EPISODE "
        "ACTUAL CANDIDATE OCR DIAGNOSTIC"
    )
    print("=" * 110)

    print()
    print("Loading FAISS...")

    index = faiss.read_index(
        INDEX_PATH
    )

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        "Loading metadata..."
    )

    metadata = load_metadata()

    print(
        f"Metadata: {len(metadata)}"
    )

    print(
        "Loading OCR cache..."
    )

    cache = load_cache()

    print(
        f"Cached OCR: {len(cache)}"
    )

    # ========================================================
    # S3E1 FAILURE
    # ========================================================

    query_index = find_query_index(
        metadata,
        "stranger_things_s03_e01",
        2990,
    )

    if query_index is None:

        print(
            "ERROR: query frame not found."
        )

        return

    query_item = metadata[
        query_index
    ]

    print()
    print("=" * 110)
    print(
        "QUERY"
    )
    print("=" * 110)

    print(
        f"Episode : "
        f"{episode_name(query_item)}"
    )

    print(
        f"Time    : "
        f"{query_item['timestamp_formatted']}"
    )

    print(
        f"Frame   : "
        f"{query_item['frame_id']}"
    )

    # ========================================================
    # QUERY CLIP OCR
    # ========================================================

    query_clip = get_query_clip(
        metadata,
        query_index,
    )

    query_texts = []

    print()
    print(
        "QUERY CLIP OCR"
    )

    print("-" * 110)

    for idx in query_clip:

        item = metadata[idx]

        text = get_ocr(
            item["image_path"],
            cache,
        )

        query_texts.append(
            text
        )

        print()
        print(
            f"{item['timestamp_formatted']} "
            f"| {item['frame_id']}"
        )

        print(
            text if text
            else "[NO OCR]"
        )

    # ========================================================
    # ACTUAL FAISS CANDIDATES
    # ========================================================

    candidates = retrieve_candidates(
        index,
        query_index,
    )

    print()
    print("=" * 110)
    print(
        "ACTUAL TOP FAISS CANDIDATES"
    )
    print("=" * 110)

    # Group by episode
    grouped = defaultdict(list)

    for idx, similarity in candidates:

        grouped[
            episode_key(
                metadata[idx]
            )
        ].append(
            (
                idx,
                similarity,
            )
        )

    # Sort episodes by best similarity
    episode_groups = []

    for ep, values in grouped.items():

        values.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        episode_groups.append(
            (
                ep,
                values,
            )
        )

    episode_groups.sort(
        key=lambda x: x[1][0][1],
        reverse=True,
    )

    # ========================================================
    # PRINT TOP EPISODES
    # ========================================================

    for ep_rank, (
        ep,
        values,
    ) in enumerate(
        episode_groups[:10],
        start=1,
    ):

        first_idx = values[0][0]

        print()
        print(
            "=" * 110
        )

        print(
            f"EPISODE RANK {ep_rank}: "
            f"{episode_name(metadata[first_idx])}"
        )

        print(
            "=" * 110
        )

        for frame_rank, (
            candidate_idx,
            clip_score,
        ) in enumerate(
            values[:5],
            start=1,
        ):

            item = metadata[
                candidate_idx
            ]

            candidate_text = get_ocr(
                item["image_path"],
                cache,
            )

            # Best OCR similarity against query clip
            ocr_scores = []

            for query_text in query_texts:

                if query_text and candidate_text:

                    ocr_scores.append(
                        text_similarity(
                            query_text,
                            candidate_text,
                        )
                    )

            best_ocr = (
                max(ocr_scores)
                if ocr_scores
                else 0.0
            )

            print()
            print(
                f"Candidate #{frame_rank}"
            )

            print(
                f"CLIP similarity : "
                f"{clip_score:.4f}"
            )

            print(
                f"Best OCR sim    : "
                f"{best_ocr:.4f}"
            )

            print(
                f"Time            : "
                f"{item['timestamp_formatted']}"
            )

            print(
                f"Frame           : "
                f"{item['frame_id']}"
            )

            print(
                f"Image           : "
                f"{item['image_path']}"
            )

            print(
                "OCR:"
            )

            print(
                candidate_text
                if candidate_text
                else "[NO OCR]"
            )

    # ========================================================
    # SAVE CACHE
    # ========================================================

    print()
    print("=" * 110)

    with open(
        OCR_CACHE_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"OCR cache saved: "
        f"{len(cache)} entries"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()