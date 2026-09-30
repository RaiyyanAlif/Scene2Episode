# src/evaluation/evaluate_credit_aware.py

import os
import sys
import json
import random
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import pytesseract


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "indexes" / "scene2episode.index"
METADATA_PATH = PROJECT_ROOT / "indexes" / "scene2episode_metadata.json"
OCR_CACHE_PATH = PROJECT_ROOT / "metadata" / "ocr_cache.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

SAMPLE_COUNT = 50
RANDOM_SEED = 42

TOP_K_PER_FRAME = 50

# Query temporal clip
OFFSETS = [-4, -2, 0, 2, 4]

# IMPORTANT:
# Exclude the entire query clip from retrieval.
#
# Since frames are sampled every 2 seconds and the query clip uses
# -4,-2,0,+2,+4, we use +/-6 seconds to safely exclude the entire
# query neighborhood.
QUERY_EXCLUSION_SECONDS = 6.0


# Normal scene
NORMAL_CLIP_WEIGHT = 0.80
NORMAL_OCR_WEIGHT = 0.20

# Credit/title-card/end-credit scene
CREDIT_CLIP_WEIGHT = 0.30
CREDIT_OCR_WEIGHT = 0.70

CREDIT_THRESHOLD = 0.35


# =============================================================================
# CREDIT KEYWORDS
# =============================================================================

CREDIT_KEYWORDS = [
    "MAIN CAST",
    "ADDITIONAL VOICES",
    "CAST",
    "VOICE",
    "VOICES",
    "MAGYAR HANGOK",
    "TOVABBI MAGYAR HANGOK",
    "MAGYAR VALTOZAT",
    "VERSIONE ITALIANA",
    "ELENCO",
    "COMPOSITORS",
    "ANIMATORS",
    "PRODUCTION",
    "DIRECTOR",
    "WRITTEN BY",
    "EDITED BY",
    "SPECIAL EFFECTS",
    "VISUAL EFFECTS",
    "EXECUTIVE PRODUCER",
    "PRODUCER",
    "SCREENPLAY",
    "STORY",
    "MUSIC",
    "DIRECTED BY",
    "CREATED BY",
    "STARRING",
    "WITH",
]


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(temp_path, path)


def normalize_text(text):
    if not text:
        return ""

    text = text.upper()

    # Keep letters/numbers/spaces
    text = re.sub(r"[^A-Z0-9À-ÖØ-Ý ]+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_ocr_text(text):
    if not text:
        return ""

    text = normalize_text(text)

    # Remove very short garbage tokens
    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if len(token) >= 2
    ]

    return " ".join(tokens)


# =============================================================================
# OCR
# =============================================================================

def ocr_image(image_path):
    """
    Run OCR using multiple preprocessing modes.
    Returns the best/combined OCR text.
    """

    try:
        image = Image.open(image_path).convert("RGB")

        import cv2

        image_np = np.array(image)

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

        variants = []

        # ---------------------------------------------------------------------
        # Variant 1: grayscale + upscale
        # ---------------------------------------------------------------------

        upscaled = cv2.resize(
            gray,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )

        variants.append(upscaled)

        # ---------------------------------------------------------------------
        # Variant 2: adaptive threshold
        # ---------------------------------------------------------------------

        adaptive = cv2.adaptiveThreshold(
            upscaled,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        variants.append(adaptive)

        # ---------------------------------------------------------------------
        # Variant 3: Otsu threshold
        # ---------------------------------------------------------------------

        _, otsu = cv2.threshold(
            upscaled,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        variants.append(otsu)

        results = []

        for variant in variants:

            try:
                text = pytesseract.image_to_string(
                    variant,
                    config="--oem 3 --psm 6"
                )

                text = clean_ocr_text(text)

                if text:
                    results.append(text)

            except Exception:
                pass

        if not results:
            return ""

        # Combine unique OCR results
        unique = []

        for text in results:
            if text not in unique:
                unique.append(text)

        return " ".join(unique)

    except Exception:
        return ""


def get_ocr(image_path, ocr_cache):
    """
    Get OCR from cache or run OCR.
    """

    key = str(image_path)

    if key in ocr_cache:
        value = ocr_cache[key]

        if isinstance(value, str):
            return value

        return ""

    text = ocr_image(image_path)

    ocr_cache[key] = text

    return text


# =============================================================================
# OCR SIMILARITY
# =============================================================================

def token_set(text):
    if not text:
        return set()

    return set(
        token
        for token in normalize_text(text).split()
        if len(token) >= 2
    )


def text_similarity(text_a, text_b):
    """
    Token Jaccard similarity.
    """

    a = token_set(text_a)
    b = token_set(text_b)

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union


# =============================================================================
# CREDIT DETECTION
# =============================================================================

def credit_score(text):
    """
    Calculate how strongly OCR text looks like a credit/title-card scene.
    """

    if not text:
        return 0.0

    normalized = normalize_text(text)

    matched = 0

    for keyword in CREDIT_KEYWORDS:

        if keyword in normalized:
            matched += 1

    if matched == 0:
        return 0.0

    # Strong signal if multiple credit keywords appear.
    #
    # 1 keyword  -> ~0.53
    # 2 keywords -> ~0.77
    # 3+          -> 1.0
    #
    # This keeps the same general behavior as the previous evaluator.

    if matched == 1:
        return 0.5333

    if matched == 2:
        return 0.7667

    return 1.0


# =============================================================================
# QUERY CLIP
# =============================================================================

def get_query_frames(query_meta, processor, model, device):
    """
    Load the 5 query frames around the target timestamp.
    """

    query_time = float(
        query_meta["timestamp_seconds"]
    )

    query_frames = []

    for offset in OFFSETS:

        target_time = query_time + offset

        if target_time < 0:
            continue

        frame_index = int(
            round(target_time / 2.0)
        ) + 1

        frame_id = f"{frame_index:06d}"

        # Construct frame path from original query path
        original_path = Path(
            query_meta["image_path"]
        )

        frame_path = original_path.parent / f"{frame_id}.jpg"

        if not frame_path.exists():
            continue

        query_frames.append(
            {
                "offset": offset,
                "timestamp": target_time,
                "path": frame_path,
            }
        )

    if not query_frames:
        return [], None

    images = []

    valid_frames = []

    for item in query_frames:

        try:
            image = Image.open(
                item["path"]
            ).convert("RGB")

            images.append(image)
            valid_frames.append(item)

        except Exception:
            pass

    if not images:
        return [], None

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = vision_outputs.pooler_output

        image_features = model.visual_projection(
            pooled_output
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

    features = image_features.cpu().numpy().astype(
        "float32"
    )

    for i, item in enumerate(valid_frames):
        item["feature"] = features[i]

    return valid_frames, features


# =============================================================================
# QUERY CLIP EXCLUSION
# =============================================================================

def is_query_clip_frame(candidate, query_meta):
    """
    IMPORTANT EVALUATION FIX.

    Returns True when a candidate belongs to the SAME media item
    and is inside the query temporal window.

    Example:

        Query = 2990 sec

        Excluded:
        2984
        2986
        2988
        2990
        2992
        2994
        2996

    This prevents the evaluator from retrieving another frame from
    the exact query clip.
    """

    if candidate["media_id"] != query_meta["media_id"]:
        return False

    candidate_time = float(
        candidate["timestamp_seconds"]
    )

    query_time = float(
        query_meta["timestamp_seconds"]
    )

    difference = abs(
        candidate_time - query_time
    )

    return difference <= QUERY_EXCLUSION_SECONDS


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():

    print("=" * 80)
    print("SCENE2EPISODE CREDIT-AWARE EVALUATION")
    print("=" * 80)

    print(f"Samples     : {SAMPLE_COUNT}")
    print(f"Random seed : {RANDOM_SEED}")
    print()

    # =========================================================================
    # LOAD FAISS
    # =========================================================================

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Index vectors: {index.ntotal}"
    )

    # =========================================================================
    # LOAD METADATA
    # =========================================================================

    print("Loading metadata...")

    metadata = load_json(
        METADATA_PATH
    )

    print(
        f"Metadata entries: {len(metadata)}"
    )

    # =========================================================================
    # LOAD OCR CACHE
    # =========================================================================

    print("Loading OCR cache...")

    if OCR_CACHE_PATH.exists():

        ocr_cache = load_json(
            OCR_CACHE_PATH
        )

    else:

        ocr_cache = {}

    print(
        f"Cached OCR: {len(ocr_cache)}"
    )

    # =========================================================================
    # LOAD CLIP
    # =========================================================================

    print("Loading CLIP model...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device      : {device}")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print("Model loaded.")
    print()

    # =========================================================================
    # BUILD FRAME LOOKUP
    # =========================================================================

    print("Building frame lookup...")

    frame_lookup = {}

    for i, item in enumerate(metadata):

        frame_lookup[
            str(item["image_path"])
        ] = i

    print()

    # =========================================================================
    # SELECT SAME BENCHMARK SAMPLES
    # =========================================================================

    random.seed(RANDOM_SEED)

    sample_count = min(
        SAMPLE_COUNT,
        len(metadata)
    )

    sample_indices = random.sample(
        range(len(metadata)),
        sample_count
    )

    print(
        f"Evaluation samples: {sample_count}"
    )

    print()

    # =========================================================================
    # METRICS
    # =========================================================================

    top1_episode_correct = 0
    top5_episode_correct = 0

    top1_media_correct = 0
    top5_media_correct = 0

    timestamp_errors = []

    failures = []

    credit_queries = 0

    # =========================================================================
    # EVALUATION LOOP
    # =========================================================================

    for sample_number, query_index in enumerate(
        sample_indices,
        start=1
    ):

        query_meta = metadata[
            query_index
        ]

        media_id = query_meta[
            "media_id"
        ]

        query_title = query_meta[
            "title"
        ]

        query_season = query_meta.get(
            "season"
        )

        query_episode = query_meta.get(
            "episode"
        )

        query_time = float(
            query_meta[
                "timestamp_seconds"
            ]
        )

        # ---------------------------------------------------------------------
        # LOAD QUERY FRAMES
        # ---------------------------------------------------------------------

        try:

            query_frames, query_features = (
                get_query_frames(
                    query_meta,
                    processor,
                    model,
                    device
                )
            )

            if not query_frames:
                print(
                    f"[{sample_number:02d}/{sample_count}] "
                    f"{media_id} @ {query_time:.0f}s | "
                    f"ERROR: query frames unavailable"
                )

                continue

        except Exception as e:

            print(
                f"[{sample_number:02d}/{sample_count}] "
                f"{media_id} @ {query_time:.0f}s | "
                f"ERROR: {e}"
            )

            continue

        # ---------------------------------------------------------------------
        # OCR QUERY FRAMES
        # ---------------------------------------------------------------------

        query_ocr = []

        for frame in query_frames:

            text = get_ocr(
                frame["path"],
                ocr_cache
            )

            query_ocr.append(
                {
                    "text": text,
                    "credit": credit_score(text),
                    "frame": frame,
                }
            )

        # ---------------------------------------------------------------------
        # DETERMINE CREDIT MODE
        # ---------------------------------------------------------------------

        query_credit_score = max(
            (
                item["credit"]
                for item in query_ocr
            ),
            default=0.0
        )

        is_credit_query = (
            query_credit_score
            >= CREDIT_THRESHOLD
        )

        if is_credit_query:
            credit_queries += 1

        # ---------------------------------------------------------------------
        # COLLECT CANDIDATES
        # ---------------------------------------------------------------------

        candidate_indices = set()

        candidate_clip_scores = {}

        for frame_number, frame in enumerate(
            query_frames
        ):

            feature = frame[
                "feature"
            ].reshape(1, -1)

            distances, indices = index.search(
                feature,
                TOP_K_PER_FRAME
            )

            for similarity, idx in zip(
                distances[0],
                indices[0]
            ):

                idx = int(idx)

                if idx < 0:
                    continue

                candidate = metadata[
                    idx
                ]

                # =============================================================
                # IMPORTANT:
                # EXCLUDE THE WHOLE QUERY CLIP
                # =============================================================

                if is_query_clip_frame(
                    candidate,
                    query_meta
                ):
                    continue

                candidate_indices.add(
                    idx
                )

                similarity = float(
                    similarity
                )

                # Keep the strongest CLIP similarity
                # if candidate appears in multiple query frames.
                if (
                    idx not in candidate_clip_scores
                    or similarity
                    > candidate_clip_scores[idx]
                ):

                    candidate_clip_scores[
                        idx
                    ] = similarity

        # ---------------------------------------------------------------------
        # SCORE CANDIDATES
        # ---------------------------------------------------------------------

        episode_candidates = defaultdict(list)

        for idx in candidate_indices:

            candidate = metadata[
                idx
            ]

            clip_similarity = candidate_clip_scores.get(
                idx,
                0.0
            )

            candidate_path = Path(
                candidate[
                    "image_path"
                ]
            )

            # OCR candidate
            candidate_text = get_ocr(
                candidate_path,
                ocr_cache
            )

            # -------------------------------------------------------------
            # OCR SIMILARITY AGAINST QUERY CLIP
            # -------------------------------------------------------------

            ocr_scores = []

            for query_item in query_ocr:

                similarity = text_similarity(
                    query_item["text"],
                    candidate_text
                )

                ocr_scores.append(
                    similarity
                )

            if ocr_scores:

                ocr_scores.sort(
                    reverse=True
                )

                # Use strongest OCR evidence.
                #
                # If multiple query frames have useful text,
                # average up to the strongest 3.
                top_ocr = ocr_scores[
                    :3
                ]

                ocr_similarity = float(
                    np.mean(top_ocr)
                )

            else:

                ocr_similarity = 0.0

            # -------------------------------------------------------------
            # CREDIT-AWARE WEIGHTS
            # -------------------------------------------------------------

            if is_credit_query:

                final_score = (
                    CREDIT_CLIP_WEIGHT
                    * clip_similarity
                    +
                    CREDIT_OCR_WEIGHT
                    * ocr_similarity
                )

                # Penalize candidates that have no useful OCR
                # when query itself is clearly a credit scene.
                if not candidate_text:

                    final_score *= 0.90

            else:

                final_score = (
                    NORMAL_CLIP_WEIGHT
                    * clip_similarity
                    +
                    NORMAL_OCR_WEIGHT
                    * ocr_similarity
                )

            # -------------------------------------------------------------
            # STORE CANDIDATE
            # -------------------------------------------------------------

            candidate_info = {
                "index": idx,
                "media_id": candidate["media_id"],
                "title": candidate["title"],
                "type": candidate.get("type"),
                "season": candidate.get("season"),
                "episode": candidate.get("episode"),
                "episode_title": candidate.get(
                    "episode_title"
                ),
                "frame_id": candidate[
                    "frame_id"
                ],
                "timestamp_seconds": float(
                    candidate[
                        "timestamp_seconds"
                    ]
                ),
                "timestamp_formatted": candidate.get(
                    "timestamp_formatted"
                ),
                "image_path": candidate[
                    "image_path"
                ],
                "clip_similarity": clip_similarity,
                "ocr_similarity": ocr_similarity,
                "final_score": final_score,
                "ocr_text": candidate_text,
            }

            # -------------------------------------------------------------
            # GROUP BY EPISODE / MEDIA
            # -------------------------------------------------------------

            episode_key = (
                candidate["media_id"],
                candidate.get("season"),
                candidate.get("episode")
            )

            episode_candidates[
                episode_key
            ].append(
                candidate_info
            )

        # =========================================================================
        # RANK EPISODES
        # =========================================================================

        ranked_episodes = []

        for episode_key, candidates in (
            episode_candidates.items()
        ):

            # Strongest candidate for this episode
            candidates.sort(
                key=lambda x: x["final_score"],
                reverse=True
            )

            best = candidates[0]

            ranked_episodes.append(
                {
                    "episode_key": episode_key,
                    "best": best,
                    "candidates": candidates,
                }
            )

        ranked_episodes.sort(
            key=lambda x: x["best"]["final_score"],
            reverse=True
        )

        # =========================================================================
        # PREDICTION
        # =========================================================================

        if not ranked_episodes:

            print(
                f"[{sample_number:02d}/{sample_count}] "
                f"{media_id} @ {query_time:.0f}s | "
                f"NO CANDIDATES"
            )

            continue

        predicted = ranked_episodes[
            0
        ]

        predicted_best = predicted[
            "best"
        ]

        predicted_key = predicted[
            "episode_key"
        ]

        predicted_media_id = (
            predicted_key[0]
        )

        predicted_season = (
            predicted_key[1]
        )

        predicted_episode = (
            predicted_key[2]
        )

        # =========================================================================
        # GROUND TRUTH KEYS
        # =========================================================================

        true_episode_key = (
            query_meta["media_id"],
            query_meta.get("season"),
            query_meta.get("episode")
        )

        # =========================================================================
        # TOP-1 EPISODE
        # =========================================================================

        episode_correct = (
            predicted_key
            == true_episode_key
        )

        if episode_correct:
            top1_episode_correct += 1

        # =========================================================================
        # TOP-5 EPISODE
        # =========================================================================

        top5_keys = [
            item["episode_key"]
            for item in ranked_episodes[:5]
        ]

        if true_episode_key in top5_keys:
            top5_episode_correct += 1

        # =========================================================================
        # MEDIA ACCURACY
        # =========================================================================

        if predicted_media_id == media_id:
            top1_media_correct += 1

        top5_media_ids = [
            item["episode_key"][0]
            for item in ranked_episodes[:5]
        ]

        if media_id in top5_media_ids:
            top5_media_correct += 1

        # =========================================================================
        # TIMESTAMP ERROR
        # =========================================================================

        predicted_time = float(
            predicted_best[
                "timestamp_seconds"
            ]
        )

        timestamp_error = abs(
            predicted_time
            - query_time
        )

        timestamp_errors.append(
            timestamp_error
        )

        # =========================================================================
        # FAILURE DETAILS
        # =========================================================================

        if not episode_correct:

            failures.append(
                {
                    "sample": sample_number,
                    "media_id": media_id,
                    "season": query_season,
                    "episode": query_episode,
                    "query_time": query_time,
                    "predicted_media_id": predicted_media_id,
                    "predicted_season": predicted_season,
                    "predicted_episode": predicted_episode,
                    "predicted_time": predicted_time,
                    "timestamp_error": timestamp_error,
                    "score": predicted_best[
                        "final_score"
                    ],
                    "clip": predicted_best[
                        "clip_similarity"
                    ],
                    "ocr": predicted_best[
                        "ocr_similarity"
                    ],
                    "credit": is_credit_query,
                    "credit_score": query_credit_score,
                }
            )

        # =========================================================================
        # PROGRESS
        # =========================================================================

        print(
            f"[{sample_number:02d}/{sample_count}] "
            f"{query_title} "
            f"S{query_season if query_season is not None else 'None'} "
            f"E{query_episode if query_episode is not None else 'None'} "
            f"@ {query_time:.0f}s | "
            f"credit={is_credit_query} | OK"
        )

        # =========================================================================
        # PERIODICALLY SAVE OCR CACHE
        # =========================================================================

        if sample_number % 5 == 0:

            try:

                save_json(
                    OCR_CACHE_PATH,
                    ocr_cache
                )

            except Exception as e:

                print(
                    f"Warning: could not save OCR cache: {e}"
                )

    # =========================================================================
    # FINAL SAVE
    # =========================================================================

    try:

        save_json(
            OCR_CACHE_PATH,
            ocr_cache
        )

    except Exception as e:

        print(
            f"Warning: could not save OCR cache: {e}"
        )

    # =========================================================================
    # RESULTS
    # =========================================================================

    evaluated_count = len(
        timestamp_errors
    )

    if evaluated_count == 0:

        print()
        print(
            "No valid evaluation results."
        )

        return

    top1_episode_accuracy = (
        top1_episode_correct
        / evaluated_count
        * 100
    )

    top5_episode_accuracy = (
        top5_episode_correct
        / evaluated_count
        * 100
    )

    top1_media_accuracy = (
        top1_media_correct
        / evaluated_count
        * 100
    )

    top5_media_accuracy = (
        top5_media_correct
        / evaluated_count
        * 100
    )

    mean_timestamp_error = float(
        np.mean(timestamp_errors)
    )

    median_timestamp_error = float(
        np.median(timestamp_errors)
    )

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print()
    print("=" * 80)
    print("CREDIT-AWARE EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"Samples                    : {evaluated_count}"
    )

    print(
        f"Credit queries             : {credit_queries}"
    )

    print()

    print(
        f"Top-1 episode accuracy    : "
        f"{top1_episode_accuracy:.2f}%"
    )

    print(
        f"Top-5 episode accuracy    : "
        f"{top5_episode_accuracy:.2f}%"
    )

    print(
        f"Top-1 media accuracy      : "
        f"{top1_media_accuracy:.2f}%"
    )

    print(
        f"Top-5 media accuracy      : "
        f"{top5_media_accuracy:.2f}%"
    )

    print()

    print(
        f"Mean timestamp error      : "
        f"{mean_timestamp_error:.2f} sec"
    )

    print(
        f"Median timestamp error    : "
        f"{median_timestamp_error:.2f} sec"
    )

    print()

    print(
        f"Top-1 failures             : "
        f"{len(failures)}"
    )

    # =========================================================================
    # PRINT FAILURE DETAILS
    # =========================================================================

    if failures:

        print()
        print("-" * 80)
        print("TOP-1 FAILURES")
        print("-" * 80)

        for failure in failures:

            print()

            print(
                f"Sample {failure['sample']}"
            )

            print(
                f"  Ground truth : "
                f"{failure['media_id']} "
                f"S{failure['season']} "
                f"E{failure['episode']} "
                f"@ {failure['query_time']:.0f}s"
            )

            print(
                f"  Prediction   : "
                f"{failure['predicted_media_id']} "
                f"S{failure['predicted_season']} "
                f"E{failure['predicted_episode']} "
                f"@ {failure['predicted_time']:.0f}s"
            )

            print(
                f"  Score        : "
                f"{failure['score']:.4f}"
            )

            print(
                f"  CLIP         : "
                f"{failure['clip']:.4f}"
            )

            print(
                f"  OCR          : "
                f"{failure['ocr']:.4f}"
            )

            print(
                f"  Credit mode  : "
                f"{failure['credit']}"
            )

            print(
                f"  Timestamp err: "
                f"{failure['timestamp_error']:.2f}s"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()