import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import faiss

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"

if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))


from episode_temporal_reranker_v2 import EpisodeTemporalRerankerV2
from episode_sequence_reranker import EpisodeSequenceReranker
import ocr_reranker


# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "scene2episode.index"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "scene2episode_metadata.json"
)

OCR_CACHE_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "ocr_cache.json"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

OFFSETS = [-4, -2, 0, 2, 4]

TOP_K = 50

QUERY_EXCLUSION_SECONDS = 6.0

DEFAULT_SAMPLES = 50

DEFAULT_SEED = 42


# ============================================================
# CREDIT KEYWORDS
# ============================================================

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
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_timestamp(seconds):
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def episode_key(meta):
    return (
        meta.get("media_id"),
        meta.get("season"),
        meta.get("episode"),
    )


def episode_label(meta):
    title = meta.get(
        "title",
        meta.get("media_id", "UNKNOWN")
    )

    season = meta.get("season")
    episode = meta.get("episode")

    return f"{title} S{season} E{episode}"


# ============================================================
# OCR CACHE
# ============================================================

def save_ocr_cache(cache):
    """
    Safely save OCR cache.
    """

    OCR_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = Path(
        str(OCR_CACHE_PATH) + ".new"
    )

    try:
        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                cache,
                f,
                ensure_ascii=False,
                indent=2
            )

        # os.replace handles replacement atomically
        # on the same filesystem.
        import os

        os.replace(
            str(temp_path),
            str(OCR_CACHE_PATH)
        )

    except PermissionError as e:

        print(
            f"Warning: OCR cache save failed: {e}"
        )

        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

    except Exception as e:

        print(
            f"Warning: OCR cache save failed: {e}"
        )


# ============================================================
# OCR
# ============================================================

def get_ocr_text(
    image_path,
    ocr_cache
):
    """
    Uses the actual API from your ocr_reranker.py:
        ocr_frame()
    """

    image_path = str(image_path)

    if image_path in ocr_cache:
        return ocr_cache[image_path]

    try:

        text = ocr_reranker.ocr_frame(
            image_path
        )

        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

    except Exception:

        text = ""

    ocr_cache[image_path] = text

    return text


def calculate_ocr_similarity(
    query_text,
    candidate_text
):
    if not query_text:
        return 0.0

    if not candidate_text:
        return 0.0

    try:
        return float(
            ocr_reranker.text_similarity(
                query_text,
                candidate_text
            )
        )

    except Exception:
        return 0.0


# ============================================================
# CREDIT DETECTION
# ============================================================

def calculate_credit_score(text):
    if not text:
        return 0.0

    upper = text.upper()

    hits = 0

    for keyword in CREDIT_KEYWORDS:
        if keyword in upper:
            hits += 1

    if hits == 0:
        return 0.0

    return min(
        1.0,
        hits / 3.0
    )


def detect_credit_query(query_texts):
    if not query_texts:
        return False

    scores = [
        calculate_credit_score(text)
        for text in query_texts
    ]

    return max(scores) >= 0.35


# ============================================================
# LEAKAGE PREVENTION
# ============================================================

def is_query_clip_frame(
    candidate,
    query_meta
):
    """
    Exclude candidate frames from the same media item
    within +/- 6 seconds of the query.
    """

    if (
        candidate.get("media_id")
        != query_meta.get("media_id")
    ):
        return False

    candidate_time = float(
        candidate.get(
            "timestamp_seconds",
            0.0
        )
    )

    query_time = float(
        query_meta.get(
            "timestamp_seconds",
            0.0
        )
    )

    return (
        abs(
            candidate_time - query_time
        )
        <= QUERY_EXCLUSION_SECONDS
    )


# ============================================================
# CLIP
# ============================================================

def load_clip():

    print("Loading CLIP model...")

    device = (
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

    model = model.to(device)
    model.eval()

    print("Model loaded.")

    return processor, model, device


@torch.no_grad()
def encode_image(
    image_path,
    processor,
    model,
    device
):

    image = (
        Image.open(image_path)
        .convert("RGB")
    )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    vision_outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    pooled_output = (
        vision_outputs.pooler_output
    )

    image_features = (
        model.visual_projection(
            pooled_output
        )
    )

    image_features = (
        image_features
        /
        image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    return (
        image_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# FIND QUERY FRAMES
# ============================================================

def find_nearest_frame(
    query_meta,
    offset,
    metadata_by_media
):

    media_id = query_meta["media_id"]

    frames = metadata_by_media.get(
        media_id,
        []
    )

    if not frames:
        return None

    target_time = (
        float(
            query_meta[
                "timestamp_seconds"
            ]
        )
        + offset
    )

    return min(
        frames,
        key=lambda frame: abs(
            float(
                frame[
                    "timestamp_seconds"
                ]
            )
            - target_time
        )
    )


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_candidates(
    query_meta,
    query_frame_index,
    query_frame_meta,
    query_embedding,
    index,
    metadata,
    ocr_cache
):

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    query_image = (
        query_frame_meta["image_path"]
    )

    query_ocr = get_ocr_text(
        query_image,
        ocr_cache
    )

    candidates = []

    for similarity, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        candidate_meta = metadata[idx]

        # --------------------------------------------
        # Leakage exclusion
        # --------------------------------------------

        if is_query_clip_frame(
            candidate_meta,
            query_meta
        ):
            continue

        candidate_image = (
            candidate_meta["image_path"]
        )

        candidate_ocr = get_ocr_text(
            candidate_image,
            ocr_cache
        )

        ocr_score = calculate_ocr_similarity(
            query_ocr,
            candidate_ocr
        )

        candidate = dict(
            candidate_meta
        )

        candidate["similarity"] = float(
            similarity
        )

        candidate["clip_similarity"] = float(
            similarity
        )

        candidate["ocr"] = float(
            ocr_score
        )

        # Required by SequenceMatcher.
        candidate["query_frame_index"] = (
            int(query_frame_index)
        )

        candidate["query_frame_timestamp"] = (
            float(
                query_frame_meta[
                    "timestamp_seconds"
                ]
            )
        )

        candidates.append(candidate)

    return candidates, query_ocr


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        "SCENE2EPISODE "
        "TEMPORAL + SEQUENCE RERANKER EVALUATION"
    )
    print("=" * 80)

    print(
        f"Samples     : {args.samples}"
    )

    print(
        f"Random seed : {args.seed}"
    )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    print()
    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Index vectors: {index.ntotal}"
    )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    print("Loading metadata...")

    metadata = load_json(
        METADATA_PATH
    )

    print(
        f"Metadata entries: {len(metadata)}"
    )

    # ========================================================
    # LOAD OCR CACHE
    # ========================================================

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

    # ========================================================
    # GROUP MEDIA FRAMES
    # ========================================================

    metadata_by_media = defaultdict(list)

    for item in metadata:

        metadata_by_media[
            item["media_id"]
        ].append(item)

    for media_id in metadata_by_media:

        metadata_by_media[
            media_id
        ].sort(
            key=lambda item:
            float(
                item[
                    "timestamp_seconds"
                ]
            )
        )

    # ========================================================
    # LOAD CLIP
    # ========================================================

    processor, model, device = load_clip()

    # ========================================================
    # LOAD RERANKERS
    # ========================================================

    temporal_reranker = (
        EpisodeTemporalRerankerV2()
    )

    sequence_reranker = (
        EpisodeSequenceReranker(
            base_weight=0.75,
            sequence_weight=0.25
        )
    )

    # ========================================================
    # SAMPLE
    # ========================================================

    random.seed(args.seed)

    sample_count = min(
        args.samples,
        len(metadata)
    )

    sample_indices = random.sample(
        range(len(metadata)),
        sample_count
    )

    print()
    print(
        f"Evaluation samples: {sample_count}"
    )

    # ========================================================
    # METRICS
    # ========================================================

    top1_episode = 0
    top5_episode = 0

    top1_media = 0
    top5_media = 0

    timestamp_errors = []

    failures = []

    credit_query_count = 0

    # ========================================================
    # EVALUATION
    # ========================================================

    for sample_number, metadata_index in enumerate(
        sample_indices,
        start=1
    ):

        query_meta = metadata[
            metadata_index
        ]

        query_episode = episode_key(
            query_meta
        )

        query_media_id = (
            query_meta["media_id"]
        )

        query_time = float(
            query_meta[
                "timestamp_seconds"
            ]
        )

        # ----------------------------------------------------
        # Build query clip
        # ----------------------------------------------------

        query_frames = []

        for offset in OFFSETS:

            frame = find_nearest_frame(
                query_meta,
                offset,
                metadata_by_media
            )

            if frame is not None:
                query_frames.append(frame)

        # ----------------------------------------------------
        # Retrieve candidates for all 5 frames
        # ----------------------------------------------------

        all_candidates = []

        query_ocr_texts = []

        for query_frame_index, frame_meta in enumerate(
            query_frames
        ):

            embedding = encode_image(
                frame_meta["image_path"],
                processor,
                model,
                device
            )

            candidates, query_ocr = (
                retrieve_candidates(
                    query_meta=query_meta,
                    query_frame_index=query_frame_index,
                    query_frame_meta=frame_meta,
                    query_embedding=embedding,
                    index=index,
                    metadata=metadata,
                    ocr_cache=ocr_cache
                )
            )

            all_candidates.extend(
                candidates
            )

            query_ocr_texts.append(
                query_ocr
            )

        # ----------------------------------------------------
        # Credit detection
        # ----------------------------------------------------

        credit_mode = detect_credit_query(
            query_ocr_texts
        )

        if credit_mode:
            credit_query_count += 1

        # ----------------------------------------------------
        # V3 TEMPORAL + OCR
        # ----------------------------------------------------

        temporal_ranked = (
            temporal_reranker.rank(
                all_candidates,
                total_query_frames=len(
                    query_frames
                ),
                credit_mode=credit_mode
            )
        )

        # ----------------------------------------------------
        # SEQUENCE RERANKER
        # ----------------------------------------------------

        # Neutral query pattern.
        #
        # The sequence matcher primarily evaluates
        # candidate temporal consistency.
        query_similarity_pattern = [
            1.0
            for _ in query_frames
        ]

        sequence_ranked = (
            sequence_reranker.rank(
                all_candidates,
                query_similarities=
                    query_similarity_pattern,
                total_query_frames=
                    len(query_frames)
            )
        )

        # ----------------------------------------------------
        # Sequence lookup
        # ----------------------------------------------------

        sequence_lookup = {}

        for item in sequence_ranked:

            sequence_lookup[
                item["episode_key"]
            ] = item

        # ----------------------------------------------------
        # Combine temporal + sequence
        # ----------------------------------------------------

        combined = []

        for temporal_item in temporal_ranked:

            key = (
                temporal_item[
                    "episode_key"
                ]
            )

            sequence_item = (
                sequence_lookup.get(key)
            )

            if sequence_item is None:

                sequence_score = 0.0
                sequence_timestamp = None

            else:

                sequence_score = float(
                    sequence_item[
                        "sequence_score"
                    ]
                )

                sequence_timestamp = (
                    sequence_item.get(
                        "timestamp_seconds"
                    )
                )

            temporal_score = float(
                temporal_item["score"]
            )

            # ------------------------------------------------
            # Adaptive weights
            # ------------------------------------------------

            if credit_mode:

                temporal_weight = 0.70
                sequence_weight = 0.30

            else:

                temporal_weight = 0.80
                sequence_weight = 0.20

            final_score = (
                temporal_weight
                * temporal_score
                +
                sequence_weight
                * sequence_score
            )

            combined.append(
                {
                    "episode_key": key,
                    "score": final_score,
                    "temporal_score": temporal_score,
                    "sequence_score": sequence_score,
                    "sequence_timestamp":
                        sequence_timestamp,

                    "best_clip":
                        temporal_item.get(
                            "best_clip",
                            0.0
                        ),

                    "top3_mean":
                        temporal_item.get(
                            "top3_mean",
                            0.0
                        ),

                    "mean_clip":
                        temporal_item.get(
                            "mean_clip",
                            0.0
                        ),

                    "coverage":
                        temporal_item.get(
                            "coverage",
                            0.0
                        ),

                    "temporal":
                        temporal_item.get(
                            "temporal",
                            0.0
                        ),

                    "ocr":
                        temporal_item.get(
                            "ocr",
                            0.0
                        ),

                    "credit_mode":
                        credit_mode
                }
            )

        combined.sort(
            key=lambda item:
            item["score"],
            reverse=True
        )

        if not combined:

            print(
                f"[{sample_number:02d}/"
                f"{sample_count}] "
                f"{episode_label(query_meta)} "
                f"@ {int(query_time)}s "
                f"| credit={credit_mode} "
                f"| NO RESULT"
            )

            continue

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = combined[0]

        predicted_key = (
            prediction["episode_key"]
        )

        # ----------------------------------------------------
        # TOP 5
        # ----------------------------------------------------

        top5 = combined[:5]

        top5_keys = [
            item["episode_key"]
            for item in top5
        ]

        # ----------------------------------------------------
        # Episode accuracy
        # ----------------------------------------------------

        correct = (
            predicted_key
            == query_episode
        )

        if correct:
            top1_episode += 1
        else:
            failures.append(
                {
                    "sample": sample_number,
                    "query": query_meta,
                    "prediction": prediction,
                    "top5": top5
                }
            )

        if query_episode in top5_keys:
            top5_episode += 1

        # ----------------------------------------------------
        # Media accuracy
        # ----------------------------------------------------

        predicted_media = (
            predicted_key[0]
        )

        if predicted_media == query_media_id:
            top1_media += 1

        top5_media_ids = {
            key[0]
            for key in top5_keys
        }

        if query_media_id in top5_media_ids:
            top5_media += 1

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        predicted_timestamp = (
            prediction[
                "sequence_timestamp"
            ]
        )

        if predicted_timestamp is None:

            # Fallback to strongest CLIP candidate
            # from predicted episode.

            predicted_candidates = [
                candidate
                for candidate in all_candidates
                if episode_key(candidate)
                == predicted_key
            ]

            if predicted_candidates:

                best_candidate = max(
                    predicted_candidates,
                    key=lambda candidate:
                    float(
                        candidate.get(
                            "similarity",
                            0.0
                        )
                    )
                )

                predicted_timestamp = float(
                    best_candidate[
                        "timestamp_seconds"
                    ]
                )

            else:

                predicted_timestamp = query_time

        predicted_timestamp = float(
            predicted_timestamp
        )

        timestamp_error = abs(
            predicted_timestamp
            - query_time
        )

        timestamp_errors.append(
            timestamp_error
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        status = (
            "OK"
            if correct
            else "FAIL"
        )

        print(
            f"[{sample_number:02d}/"
            f"{sample_count}] "
            f"{episode_label(query_meta)} "
            f"@ {int(query_time)}s "
            f"| credit={credit_mode} "
            f"| {status}"
        )

        # ----------------------------------------------------
        # Save OCR cache every 5 samples
        # ----------------------------------------------------

        if sample_number % 5 == 0:

            save_ocr_cache(
                ocr_cache
            )

    # ========================================================
    # FINAL CACHE SAVE
    # ========================================================

    save_ocr_cache(
        ocr_cache
    )

    # ========================================================
    # METRICS
    # ========================================================

    top1_episode_accuracy = (
        top1_episode
        / sample_count
        * 100
    )

    top5_episode_accuracy = (
        top5_episode
        / sample_count
        * 100
    )

    top1_media_accuracy = (
        top1_media
        / sample_count
        * 100
    )

    top5_media_accuracy = (
        top5_media
        / sample_count
        * 100
    )

    if timestamp_errors:

        mean_timestamp_error = float(
            np.mean(timestamp_errors)
        )

        median_timestamp_error = float(
            np.median(timestamp_errors)
        )

    else:

        mean_timestamp_error = 0.0
        median_timestamp_error = 0.0

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TEMPORAL + SEQUENCE RERANKER RESULTS"
    )
    print("=" * 80)

    print(
        f"Samples                    : "
        f"{sample_count}"
    )

    print(
        f"Credit queries             : "
        f"{credit_query_count}"
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

    # ========================================================
    # FAILURE DETAILS
    # ========================================================

    if failures:

        print()
        print("-" * 80)
        print("TOP-1 FAILURES")
        print("-" * 80)

        for failure in failures:

            query = failure["query"]

            prediction = (
                failure["prediction"]
            )

            query_key = episode_key(
                query
            )

            predicted_key = (
                prediction[
                    "episode_key"
                ]
            )

            query_time = float(
                query[
                    "timestamp_seconds"
                ]
            )

            predicted_timestamp = (
                prediction[
                    "sequence_timestamp"
                ]
            )

            print()

            print(
                f"Sample {failure['sample']}"
            )

            print(
                f"  Ground truth : "
                f"{query_key[0]} "
                f"S{query_key[1]} "
                f"E{query_key[2]} "
                f"@ {int(query_time)}s"
            )

            print(
                f"  Prediction   : "
                f"{predicted_key[0]} "
                f"S{predicted_key[1]} "
                f"E{predicted_key[2]}"
            )

            print(
                f"  Final score  : "
                f"{prediction['score']:.4f}"
            )

            print(
                f"  Temporal     : "
                f"{prediction['temporal_score']:.4f}"
            )

            print(
                f"  Sequence     : "
                f"{prediction['sequence_score']:.4f}"
            )

            print(
                f"  Best CLIP    : "
                f"{prediction['best_clip']:.4f}"
            )

            print(
                f"  Top-3 mean   : "
                f"{prediction['top3_mean']:.4f}"
            )

            print(
                f"  Mean CLIP    : "
                f"{prediction['mean_clip']:.4f}"
            )

            print(
                f"  Coverage     : "
                f"{prediction['coverage']:.4f}"
            )

            print(
                f"  Temporal     : "
                f"{prediction['temporal']:.4f}"
            )

            print(
                f"  OCR          : "
                f"{prediction['ocr']:.4f}"
            )

            print(
                f"  Credit mode  : "
                f"{prediction['credit_mode']}"
            )

            if predicted_timestamp is not None:

                predicted_timestamp = float(
                    predicted_timestamp
                )

                timestamp_error = abs(
                    predicted_timestamp
                    - query_time
                )

                print(
                    f"  Predicted time: "
                    f"{format_timestamp(predicted_timestamp)}"
                )

                print(
                    f"  Timestamp err: "
                    f"{timestamp_error:.2f}s"
                )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()