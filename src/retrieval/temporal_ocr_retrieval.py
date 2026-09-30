import os
import re
import json
import argparse
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

FRAME_INTERVAL = 2.0

CLIP_OFFSETS = [
    -4,
    -2,
    0,
    2,
    4,
]

DEFAULT_TOP_K = 50

# Final score weights
CLIP_WEIGHT = 0.80
OCR_WEIGHT = 0.20

# OCR settings
OCR_SCALE = 2.0
TESSERACT_CONFIG = "--oem 3 --psm 6"

TESSERACT_EXE = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_EXE
    )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# EPISODE HELPERS
# ============================================================

def episode_key(item):

    return (
        item["media_id"],
        item.get("season"),
        item.get("episode"),
    )


def episode_name(item):

    media_id = item["media_id"]

    season = item.get("season")
    episode = item.get("episode")

    if season is None:

        return item["title"]

    return (
        f"{item['title']} "
        f"S{season:02d} "
        f"E{episode:02d}"
    )


def build_lookup(metadata):

    """
    Map:

        (media_id, season, episode, frame_id)
            -> metadata index
    """

    lookup = {}

    for i, item in enumerate(metadata):

        key = (
            item["media_id"],
            item.get("season"),
            item.get("episode"),
            item["frame_id"],
        )

        lookup[key] = i

    return lookup


def find_frame_index(
    item,
    lookup,
    offset_seconds,
):

    """
    Find nearest indexed frame to
    requested temporal offset.
    """

    target_time = (
        item["timestamp_seconds"]
        + offset_seconds
    )

    if target_time < 0:
        target_time = 0

    target_frame_number = (
        int(
            round(
                target_time
                / FRAME_INTERVAL
            )
        )
        + 1
    )

    frame_id = (
        f"{target_frame_number:06d}"
    )

    key = (
        item["media_id"],
        item.get("season"),
        item.get("episode"),
        frame_id,
    )

    return lookup.get(key)


def get_clip_indices(
    query_index,
    metadata,
    lookup,
):

    """
    Get the 5 indexed frames surrounding
    the query frame.
    """

    query_item = metadata[
        query_index
    ]

    indices = []

    for offset in CLIP_OFFSETS:

        idx = find_frame_index(
            query_item,
            lookup,
            offset,
        )

        if idx is not None:

            indices.append(idx)

    # Remove duplicates
    return list(
        dict.fromkeys(indices)
    )


# ============================================================
# OCR
# ============================================================

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


def preprocess_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    scaled = cv2.resize(
        gray,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
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


def ocr_frame(image_path):

    if not os.path.exists(
        image_path
    ):

        return ""

    image = cv2.imread(
        image_path
    )

    if image is None:

        return ""

    variants = preprocess_image(
        image
    )

    texts = []

    for variant in variants:

        try:

            text = (
                pytesseract.image_to_string(
                    variant,
                    config=TESSERACT_CONFIG,
                )
            )

            text = clean_text(
                text
            )

            if text:

                texts.append(text)

        except Exception as exc:

            print(
                f"OCR error: {exc}"
            )

    # Remove duplicate outputs
    unique = []

    for text in texts:

        if text not in unique:

            unique.append(text)

    return " | ".join(unique)


# ============================================================
# OCR CACHE
# ============================================================

def load_ocr_cache():

    if not os.path.exists(
        OCR_CACHE_PATH
    ):

        return {}

    try:

        with open(
            OCR_CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_ocr_cache(cache):

    directory = os.path.dirname(
        OCR_CACHE_PATH
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )

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


def get_ocr(
    image_path,
    cache,
):

    if image_path in cache:

        return cache[
            image_path
        ]

    text = ocr_frame(
        image_path
    )

    cache[
        image_path
    ] = text

    return text


# ============================================================
# OCR TOKEN SIMILARITY
# ============================================================

def normalize_tokens(text):

    text = text.lower()

    return set(
        re.findall(
            r"[a-z0-9]{2,}",
            text,
        )
    )


def text_similarity(
    query_text,
    candidate_text,
):

    if not query_text:
        return 0.0

    if not candidate_text:
        return 0.0

    q = normalize_tokens(
        query_text
    )

    c = normalize_tokens(
        candidate_text
    )

    if not q or not c:

        return 0.0

    intersection = len(
        q & c
    )

    union = len(
        q | c
    )

    if union == 0:

        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# FRAME RETRIEVAL
# ============================================================

def retrieve_frame(
    index,
    query_vector,
    query_index,
    top_k,
):

    vector = (
        query_vector
        .reshape(1, -1)
        .astype("float32")
    )

    scores, indices = index.search(
        vector,
        top_k + 1,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:

            continue

        # Exclude exact query frame
        if idx == query_index:

            continue

        results.append(
            (
                int(idx),
                float(score),
            )
        )

        if len(results) >= top_k:

            break

    return results


# ============================================================
# TEMPORAL CLIP RETRIEVAL
# ============================================================

def retrieve_clip(
    index,
    metadata,
    lookup,
    query_index,
    top_k,
):

    clip_indices = get_clip_indices(
        query_index,
        metadata,
        lookup,
    )

    episode_evidence = defaultdict(
        list
    )

    for clip_idx in clip_indices:

        # Reconstruct indexed CLIP vector
        query_vector = index.reconstruct(
            clip_idx
        )

        results = retrieve_frame(
            index,
            query_vector,
            clip_idx,
            top_k,
        )

        for (
            candidate_idx,
            similarity,
        ) in results:

            ep = episode_key(
                metadata[
                    candidate_idx
                ]
            )

            episode_evidence[
                ep
            ].append(
                {
                    "query_frame": clip_idx,
                    "candidate_frame": candidate_idx,
                    "similarity": similarity,
                }
            )

    return (
        clip_indices,
        episode_evidence,
    )


# ============================================================
# CLIP EPISODE SCORE
# ============================================================

def calculate_clip_score(
    evidence,
    number_of_clip_frames,
):

    if not evidence:

        return 0.0

    similarities = sorted(
        [
            x["similarity"]
            for x in evidence
        ],
        reverse=True,
    )

    top3 = similarities[:3]

    top3_mean = np.mean(
        top3
    )

    mean_similarity = np.mean(
        similarities
    )

    query_frames = set(
        x["query_frame"]
        for x in evidence
    )

    coverage = (
        len(query_frames)
        / number_of_clip_frames
    )

    rank_score = np.mean(
        similarities
    )

    score = (
        0.35 * top3_mean
        + 0.30 * mean_similarity
        + 0.20 * coverage
        + 0.15 * rank_score
    )

    return float(score)


# ============================================================
# OCR EPISODE SCORE
# ============================================================

def calculate_ocr_score(
    query_clip_indices,
    candidate_evidence,
    metadata,
    cache,
):

    # --------------------------------------------------------
    # OCR QUERY CLIP
    # --------------------------------------------------------

    query_texts = []

    for idx in query_clip_indices:

        path = metadata[idx][
            "image_path"
        ]

        text = get_ocr(
            path,
            cache,
        )

        if text:

            query_texts.append(
                text
            )

    if not query_texts:

        return (
            0.0,
            0,
        )

    # --------------------------------------------------------
    # OCR CANDIDATE FRAMES
    # --------------------------------------------------------

    candidate_indices = set(
        x["candidate_frame"]
        for x in candidate_evidence
    )

    candidate_scores = []

    for candidate_idx in candidate_indices:

        path = metadata[
            candidate_idx
        ][
            "image_path"
        ]

        candidate_text = get_ocr(
            path,
            cache,
        )

        if not candidate_text:

            continue

        similarities = []

        for query_text in query_texts:

            sim = text_similarity(
                query_text,
                candidate_text,
            )

            similarities.append(
                sim
            )

        if similarities:

            candidate_scores.append(
                max(similarities)
            )

    if not candidate_scores:

        return (
            0.0,
            0,
        )

    candidate_scores.sort(
        reverse=True
    )

    top_scores = (
        candidate_scores[:3]
    )

    score = float(
        np.mean(top_scores)
    )

    return (
        score,
        len(candidate_scores),
    )


# ============================================================
# TEMPORAL + OCR RETRIEVAL
# ============================================================

def retrieve_temporal_ocr(
    index,
    metadata,
    lookup,
    query_index,
    top_k,
    cache,
):

    (
        clip_indices,
        episode_evidence,
    ) = retrieve_clip(
        index,
        metadata,
        lookup,
        query_index,
        top_k,
    )

    results = []

    for (
        ep,
        evidence,
    ) in episode_evidence.items():

        clip_score = (
            calculate_clip_score(
                evidence,
                len(clip_indices),
            )
        )

        (
            ocr_score,
            ocr_support,
        ) = calculate_ocr_score(
            clip_indices,
            evidence,
            metadata,
            cache,
        )

        final_score = (
            CLIP_WEIGHT * clip_score
            + OCR_WEIGHT * ocr_score
        )

        results.append(
            {
                "episode": ep,
                "clip_score": clip_score,
                "ocr_score": ocr_score,
                "final_score": final_score,
                "support": len(evidence),
                "ocr_support": ocr_support,
                "evidence": evidence,
            }
        )

    results.sort(
        key=lambda x: x[
            "final_score"
        ],
        reverse=True,
    )

    return (
        clip_indices,
        results,
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    metadata,
    query_index,
    clip_indices,
    results,
    ground_truth,
):

    query_item = metadata[
        query_index
    ]

    print()

    print(
        "=" * 115
    )

    print(
        "GROUND TRUTH: "
        f"{episode_name(query_item)} "
        f"@ "
        f"{query_item['timestamp_formatted']}"
    )

    print(
        f"Clip offsets: "
        f"{CLIP_OFFSETS}"
    )

    print(
        "=" * 115
    )

    print(
        f"{'Rank':<6}"
        f"{'Episode':<34}"
        f"{'Final':<10}"
        f"{'CLIP':<10}"
        f"{'OCR':<10}"
        f"{'Support':<9}"
        f"{'OCR sup':<9}"
    )

    print(
        "-" * 115
    )

    for rank, result in enumerate(
        results[:15],
        start=1,
    ):

        if not result[
            "evidence"
        ]:

            continue

        candidate_idx = (
            result[
                "evidence"
            ][0][
                "candidate_frame"
            ]
        )

        item = metadata[
            candidate_idx
        ]

        name = episode_name(
            item
        )

        print(
            f"{rank:<6}"
            f"{name:<34}"
            f"{result['final_score']:<10.4f}"
            f"{result['clip_score']:<10.4f}"
            f"{result['ocr_score']:<10.4f}"
            f"{result['support']:<9}"
            f"{result['ocr_support']:<9}"
        )

    correct = False

    if results:

        predicted = results[0][
            "episode"
        ]

        correct = (
            predicted
            == ground_truth
        )

    print()

    if correct:

        print(
            ">>> RESULT: "
            "TEMPORAL + OCR CORRECT"
        )

    else:

        print(
            ">>> RESULT: "
            "TEMPORAL + OCR STILL INCORRECT"
        )


# ============================================================
# FIND QUERY FRAME
# ============================================================

def find_query_index(
    metadata,
    media_id,
    timestamp,
):

    best_idx = None

    best_diff = float(
        "inf"
    )

    for i, item in enumerate(
        metadata
    ):

        if item[
            "media_id"
        ] != media_id:

            continue

        diff = abs(
            item[
                "timestamp_seconds"
            ]
            - timestamp
        )

        if diff < best_diff:

            best_diff = diff
            best_idx = i

    return best_idx


# ============================================================
# KNOWN FAILURES
# ============================================================

KNOWN_FAILURES = [

    (
        "lucifer_s01_e13",
        898,
    ),

    (
        "stranger_things_s03_e01",
        2990,
    ),

    (
        "lucifer_s01_e03",
        490,
    ),

    (
        "stranger_things_s02_e01",
        1840,
    ),

    (
        "stranger_things_s03_e04",
        288,
    ),

    (
        "lucifer_s01_e09",
        640,
    ),

    (
        "stranger_things_s03_e06",
        2570,
    ),
]


# ============================================================
# TEST KNOWN FAILURES
# ============================================================

def test_known_failures(
    index,
    metadata,
    lookup,
    top_k,
    cache,
):

    print()

    print(
        "=" * 115
    )

    print(
        "TEMPORAL + OCR RETRIEVAL — "
        "KNOWN FAILURES"
    )

    print(
        "=" * 115
    )

    corrected = 0

    for (
        media_id,
        timestamp,
    ) in KNOWN_FAILURES:

        query_index = (
            find_query_index(
                metadata,
                media_id,
                timestamp,
            )
        )

        if query_index is None:

            print(
                f"Could not find "
                f"{media_id} "
                f"@ {timestamp}"
            )

            continue

        ground_truth = episode_key(
            metadata[
                query_index
            ]
        )

        (
            clip_indices,
            results,
        ) = retrieve_temporal_ocr(
            index,
            metadata,
            lookup,
            query_index,
            top_k,
            cache,
        )

        print_results(
            metadata,
            query_index,
            clip_indices,
            results,
            ground_truth,
        )

        if results:

            if (
                results[0][
                    "episode"
                ]
                == ground_truth
            ):

                corrected += 1

    print()

    print(
        "=" * 115
    )

    print(
        f"Known failures corrected: "
        f"{corrected}/7"
    )

    print(
        "=" * 115
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    print(
        "=" * 115
    )

    print(
        "SCENE2EPISODE "
        "TEMPORAL + OCR RETRIEVAL"
    )

    print(
        "=" * 115
    )

    print(
        f"Top-K per frame: "
        f"{args.top_k}"
    )

    print(
        f"Clip offsets   : "
        f"{CLIP_OFFSETS}"
    )

    print(
        f"Score weights   : "
        f"CLIP={CLIP_WEIGHT}, "
        f"OCR={OCR_WEIGHT}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print()

    print(
        "Loading FAISS index..."
    )

    index = faiss.read_index(
        INDEX_PATH
    )

    print(
        f"Vectors: "
        f"{index.ntotal}"
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    print(
        "Loading metadata..."
    )

    metadata = load_metadata(
        METADATA_PATH
    )

    print(
        f"Metadata: "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # LOOKUP
    # --------------------------------------------------------

    print(
        "Building frame lookup..."
    )

    lookup = build_lookup(
        metadata
    )

    print(
        f"Lookup entries: "
        f"{len(lookup)}"
    )

    # --------------------------------------------------------
    # OCR CACHE
    # --------------------------------------------------------

    print()

    print(
        "Loading OCR cache..."
    )

    cache = load_ocr_cache()

    print(
        f"Cached OCR entries: "
        f"{len(cache)}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_known_failures(
        index,
        metadata,
        lookup,
        args.top_k,
        cache,
    )

    # --------------------------------------------------------
    # SAVE CACHE
    # --------------------------------------------------------

    print()

    print(
        "Saving OCR cache..."
    )

    save_ocr_cache(
        cache
    )

    print(
        f"OCR cache entries: "
        f"{len(cache)}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()