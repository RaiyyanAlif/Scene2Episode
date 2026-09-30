import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
RETRIEVAL_DIR = SRC_DIR / "retrieval"

sys.path.insert(0, str(RETRIEVAL_DIR))


# ============================================================
# IMPORT EXISTING MODULES
# ============================================================

from ocr_reranker import (
    ocr_frame,
    text_similarity,
)

from episode_temporal_reranker_v2 import (
    EpisodeTemporalRerankerV2,
)

from fingerprint_reranker import (
    FingerprintReranker,
)


# ============================================================
# PATHS
# ============================================================

INDEX_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "scene2episode.index"
)

INDEX_METADATA_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "scene2episode_metadata.json"
)

OCR_CACHE_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "ocr_cache.json"
)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_COUNT = 50
RANDOM_SEED = 42

TOP_K = 50

QUERY_OFFSETS = [
    -4,
    -2,
    0,
    2,
    4,
]

QUERY_EXCLUSION_SECONDS = 6.0

# Weight of the existing V3 temporal+OCR score
TEMPORAL_WEIGHT = 0.80

# Weight of the new fingerprint score
FINGERPRINT_WEIGHT = 0.20


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json_safe(
    data,
    path,
):

    temp_path = path.with_suffix(
        ".tmp"
    )

    try:

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        temp_path.replace(
            path
        )

    except PermissionError:

        print(
            "WARNING: Could not save OCR cache "
            "because the file is currently locked."
        )


def episode_key(meta):

    return (
        meta.get("media_id"),
        meta.get("season"),
        meta.get("episode"),
    )


def format_timestamp(seconds):

    seconds = max(
        0,
        float(seconds)
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600)
        // 60
    )

    secs = int(
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def is_query_clip_frame(
    candidate,
    query_meta,
):

    if (
        candidate["media_id"]
        != query_meta["media_id"]
    ):

        return False

    candidate_time = float(
        candidate[
            "timestamp_seconds"
        ]
    )

    query_time = float(
        query_meta[
            "timestamp_seconds"
        ]
    )

    return (
        abs(
            candidate_time
            - query_time
        )
        <= QUERY_EXCLUSION_SECONDS
    )


# ============================================================
# OCR
# ============================================================

def load_ocr_cache():

    if not OCR_CACHE_PATH.exists():

        return {}

    try:

        return load_json(
            OCR_CACHE_PATH
        )

    except Exception:

        return {}


def get_ocr(
    image_path,
    cache,
):

    key = str(
        image_path
    )

    if key in cache:

        return cache[key]

    try:

        text = ocr_frame(
            image_path
        )

    except Exception:

        text = ""

    cache[key] = text

    return text


def get_query_ocr(
    query_frames,
    cache,
):

    texts = []

    for frame in query_frames:

        text = get_ocr(
            frame["image_path"],
            cache
        )

        texts.append(
            text
        )

    return texts


def credit_score(text):

    if not text:
        return 0.0

    text = text.upper()

    keywords = [
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

    score = 0.0

    for keyword in keywords:

        if keyword in text:

            score += 0.1

    return min(
        1.0,
        score
    )


def is_credit_query(
    query_texts,
):

    scores = [
        credit_score(text)
        for text in query_texts
    ]

    return max(
        scores,
        default=0.0
    ) >= 0.30


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINGERPRINT-INTEGRATED RETRIEVAL EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print()
    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    metadata = load_json(
        INDEX_METADATA_PATH
    )

    print(
        f"FAISS vectors : "
        f"{index.ntotal}"
    )

    print(
        f"Metadata      : "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Load rerankers
    # --------------------------------------------------------

    print()
    print("Loading rerankers...")

    temporal_reranker = (
        EpisodeTemporalRerankerV2()
    )

    fingerprint_reranker = (
        FingerprintReranker()
    )

    # --------------------------------------------------------
    # OCR cache
    # --------------------------------------------------------

    ocr_cache = load_ocr_cache()

    print(
        f"OCR cache entries : "
        f"{len(ocr_cache)}"
    )

    # --------------------------------------------------------
    # Candidate grouping
    # --------------------------------------------------------

    all_indices_by_episode = (
        defaultdict(list)
    )

    for i, meta in enumerate(
        metadata
    ):

        all_indices_by_episode[
            episode_key(meta)
        ].append(i)

    # --------------------------------------------------------
    # Random evaluation samples
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    sample_indices = random.sample(
        range(
            len(metadata)
        ),
        SAMPLE_COUNT
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    top1_episode_correct = 0
    top5_episode_correct = 0

    top1_media_correct = 0
    top5_media_correct = 0

    timestamp_errors = []

    failures = []

    credit_queries = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for sample_number, query_index in enumerate(
        sample_indices,
        start=1
    ):

        query_meta = metadata[
            query_index
        ]

        query_episode = episode_key(
            query_meta
        )

        query_media = query_meta[
            "media_id"
        ]

        query_time = float(
            query_meta[
                "timestamp_seconds"
            ]
        )

        # ----------------------------------------------------
        # Build 5-frame temporal query
        # ----------------------------------------------------

        query_frames = []

        for offset in QUERY_OFFSETS:

            target_time = (
                query_time
                + offset
            )

            # Same media only
            candidates = (
                all_indices_by_episode[
                    query_episode
                ]
            )

            if not candidates:

                continue

            best_index = min(
                candidates,
                key=lambda i:
                abs(
                    float(
                        metadata[i][
                            "timestamp_seconds"
                        ]
                    )
                    - target_time
                )
            )

            query_frames.append(
                metadata[
                    best_index
                ]
            )

        if not query_frames:

            continue

        # ----------------------------------------------------
        # OCR query
        # ----------------------------------------------------

        query_ocr_texts = get_query_ocr(
            query_frames,
            ocr_cache
        )

        credit_mode = is_credit_query(
            query_ocr_texts
        )

        if credit_mode:

            credit_queries += 1

        # ----------------------------------------------------
        # Embed query frames
        # ----------------------------------------------------

        query_embeddings = []

        for frame in query_frames:

            image_path = frame[
                "image_path"
            ]

            # Find exact metadata index
            frame_index = None

            for idx in candidates:

                if metadata[idx][
                    "frame_id"
                ] == frame[
                    "frame_id"
                ]:

                    frame_index = idx

                    break

            if frame_index is None:
                continue

            vector = index.reconstruct(
                frame_index
            )

            vector = np.asarray(
                vector,
                dtype=np.float32
            )

            norm = np.linalg.norm(
                vector
            )

            if norm > 0:

                vector = (
                    vector
                    / norm
                )

            query_embeddings.append(
                vector
            )

        if not query_embeddings:

            continue

        # ----------------------------------------------------
        # FAISS retrieval
        # ----------------------------------------------------

        all_candidates = []

        for q_frame_index, vector in enumerate(
            query_embeddings
        ):

            vector_2d = (
                vector.reshape(
                    1,
                    -1
                )
            )

            distances, indices = (
                index.search(
                    vector_2d,
                    TOP_K
                )
            )

            for rank in range(
                len(indices[0])
            ):

                candidate_index = int(
                    indices[0][rank]
                )

                candidate_meta = metadata[
                    candidate_index
                ]

                # --------------------------------------------
                # Exclude the entire query temporal window
                # --------------------------------------------

                if is_query_clip_frame(
                    candidate_meta,
                    query_meta
                ):

                    continue

                all_candidates.append(
                    {
                        "index":
                            candidate_index,

                        "query_frame_index":
                            q_frame_index,

                        "similarity":
                            float(
                                distances[
                                    0
                                ][
                                    rank
                                ]
                            ),

                        **candidate_meta,
                    }
                )

        # ----------------------------------------------------
        # Existing V3 temporal + OCR reranking
        # ----------------------------------------------------

        temporal_ranked = (
            temporal_reranker.rank(
                all_candidates,
                total_query_frames=len(
                    query_embeddings
                ),
                credit_mode=credit_mode,
            )
        )

        # ----------------------------------------------------
        # Candidate episodes
        # ----------------------------------------------------

        candidate_episodes = [
            result[
                "episode_key"
            ]
            for result in temporal_ranked
        ]

        # Remove duplicates while preserving order
        candidate_episodes = list(
            dict.fromkeys(
                candidate_episodes
            )
        )

        # ----------------------------------------------------
        # Fingerprint reranking
        # ----------------------------------------------------

        fingerprint_ranked = (
            fingerprint_reranker.rank(
                query_embeddings,
                candidate_episodes
            )
        )

        fingerprint_lookup = {
            result[
                "episode_key"
            ]: result
            for result in fingerprint_ranked
        }

        # ----------------------------------------------------
        # Combine V3 + fingerprint
        # ----------------------------------------------------

        combined = []

        for temporal_result in temporal_ranked:

            key = temporal_result[
                "episode_key"
            ]

            fingerprint_result = (
                fingerprint_lookup.get(
                    key
                )
            )

            if fingerprint_result is None:

                fingerprint_score = 0.0

            else:

                fingerprint_score = (
                    fingerprint_result[
                        "score"
                    ]
                )

            final_score = (
                TEMPORAL_WEIGHT
                * temporal_result[
                    "score"
                ]
                +
                FINGERPRINT_WEIGHT
                * fingerprint_score
            )

            combined.append(
                {
                    "episode_key":
                        key,

                    "score":
                        final_score,

                    "temporal_score":
                        temporal_result[
                            "score"
                        ],

                    "fingerprint_score":
                        fingerprint_score,

                    "fingerprint_result":
                        fingerprint_result,
                }
            )

        combined.sort(
            key=lambda x:
            x["score"],
            reverse=True
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        if not combined:

            continue

        top1 = combined[0][
            "episode_key"
        ]

        top5 = [
            x["episode_key"]
            for x in combined[:5]
        ]

        # ----------------------------------------------------
        # Episode accuracy
        # ----------------------------------------------------

        if top1 == query_episode:

            top1_episode_correct += 1

        else:

            failures.append(
                {
                    "query":
                        query_meta,

                    "predicted":
                        top1,

                    "top1_score":
                        combined[0][
                            "score"
                        ],

                    "temporal_score":
                        combined[0][
                            "temporal_score"
                        ],

                    "fingerprint_score":
                        combined[0][
                            "fingerprint_score"
                        ],
                }
            )

        if query_episode in top5:

            top5_episode_correct += 1

        # ----------------------------------------------------
        # Media accuracy
        # ----------------------------------------------------

        top1_media = top1[0]

        if top1_media == query_media:

            top1_media_correct += 1

        top5_media_ids = [
            key[0]
            for key in top5
        ]

        if query_media in top5_media_ids:

            top5_media_correct += 1

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        fingerprint_result = (
            combined[0][
                "fingerprint_result"
            ]
        )

        if (
            top1 == query_episode
            and fingerprint_result is not None
        ):

            predicted_time = (
                fingerprint_result[
                    "timestamp_seconds"
                ]
            )

        else:

            # Use strongest temporal candidate
            temporal_result = (
                temporal_ranked[0]
            )

            predicted_time = (
                temporal_result.get(
                    "timestamp_seconds",
                    query_time
                )
            )

        timestamp_error = abs(
            predicted_time
            - query_time
        )

        timestamp_errors.append(
            timestamp_error
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"[{sample_number:02d}/"
            f"{SAMPLE_COUNT}] "
            f"{query_meta['media_id']} "
            f"{format_timestamp(query_time)} "
            f"-> "
            f"{top1} "
            f"| "
            f"{'OK' if top1 == query_episode else 'FAIL'} "
            f"| "
            f"FP={combined[0]['fingerprint_score']:.3f}"
        )

        # Save cache periodically
        if (
            sample_number % 5 == 0
        ):

            save_json_safe(
                ocr_cache,
                OCR_CACHE_PATH
            )

    # --------------------------------------------------------
    # Final cache save
    # --------------------------------------------------------

    save_json_safe(
        ocr_cache,
        OCR_CACHE_PATH
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = SAMPLE_COUNT

    top1_episode_accuracy = (
        100.0
        * top1_episode_correct
        / total
    )

    top5_episode_accuracy = (
        100.0
        * top5_episode_correct
        / total
    )

    top1_media_accuracy = (
        100.0
        * top1_media_correct
        / total
    )

    top5_media_accuracy = (
        100.0
        * top5_media_correct
        / total
    )

    if timestamp_errors:

        mean_timestamp_error = float(
            np.mean(
                timestamp_errors
            )
        )

        median_timestamp_error = float(
            np.median(
                timestamp_errors
            )
        )

    else:

        mean_timestamp_error = 0.0
        median_timestamp_error = 0.0

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINGERPRINT BENCHMARK RESULTS")
    print("=" * 80)

    print()
    print(
        f"Samples                 : "
        f"{SAMPLE_COUNT}"
    )

    print(
        f"Credit queries          : "
        f"{credit_queries}"
    )

    print()
    print(
        f"Top-1 episode accuracy  : "
        f"{top1_episode_accuracy:.2f}%"
    )

    print(
        f"Top-5 episode accuracy  : "
        f"{top5_episode_accuracy:.2f}%"
    )

    print(
        f"Top-1 media accuracy    : "
        f"{top1_media_accuracy:.2f}%"
    )

    print(
        f"Top-5 media accuracy    : "
        f"{top5_media_accuracy:.2f}%"
    )

    print()
    print(
        f"Mean timestamp error    : "
        f"{mean_timestamp_error:.2f} sec"
    )

    print(
        f"Median timestamp error  : "
        f"{median_timestamp_error:.2f} sec"
    )

    print()
    print(
        f"Top-1 failures          : "
        f"{len(failures)}"
    )

    # --------------------------------------------------------
    # Failure details
    # --------------------------------------------------------

    if failures:

        print()
        print("-" * 80)
        print("FAILURES")
        print("-" * 80)

        for i, failure in enumerate(
            failures,
            start=1
        ):

            query = failure[
                "query"
            ]

            print()
            print(
                f"{i}. "
                f"{query['media_id']} "
                f"{format_timestamp(query['timestamp_seconds'])}"
            )

            print(
                f"   Predicted: "
                f"{failure['predicted']}"
            )

            print(
                f"   Combined: "
                f"{failure['top1_score']:.4f}"
            )

            print(
                f"   Temporal: "
                f"{failure['temporal_score']:.4f}"
            )

            print(
                f"   Fingerprint: "
                f"{failure['fingerprint_score']:.4f}"
            )


if __name__ == "__main__":

    main()