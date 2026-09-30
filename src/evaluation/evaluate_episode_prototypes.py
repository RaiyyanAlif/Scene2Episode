import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
RETRIEVAL_DIR = SRC_DIR / "retrieval"

sys.path.insert(0, str(RETRIEVAL_DIR))


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

PROTOTYPE_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "episode_prototypes.npz"
)

PROTOTYPE_METADATA_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "episode_prototypes_metadata.json"
)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_COUNT = 50
RANDOM_SEED = 42

QUERY_OFFSETS = [
    -4,
    -2,
    0,
    2,
    4,
]


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


def episode_key(meta):

    return (
        meta.get("media_id"),
        meta.get("season"),
        meta.get("episode"),
    )


def normalize(vector):

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def format_timestamp(seconds):

    seconds = max(
        0,
        float(seconds)
    )

    h = int(
        seconds // 3600
    )

    m = int(
        (seconds % 3600) // 60
    )

    s = int(
        seconds % 60
    )

    return (
        f"{h:02d}:"
        f"{m:02d}:"
        f"{s:02d}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EPISODE PROTOTYPE STANDALONE EVALUATION")
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
        f"Vectors  : {index.ntotal}"
    )

    print(
        f"Metadata : {len(metadata)}"
    )

    # --------------------------------------------------------
    # Load prototypes
    # --------------------------------------------------------

    print()
    print("Loading episode prototypes...")

    prototype_data = np.load(
        PROTOTYPE_PATH
    )

    prototypes = (
        prototype_data[
            "embeddings"
        ]
        .astype(np.float32)
    )

    prototype_metadata = load_json(
        PROTOTYPE_METADATA_PATH
    )

    print(
        f"Prototypes : "
        f"{len(prototypes)}"
    )

    print(
        f"Dimension  : "
        f"{prototypes.shape[1]}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    norms = np.linalg.norm(
        prototypes,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    prototypes = (
        prototypes / norms
    )

    # --------------------------------------------------------
    # Group prototypes by episode
    # --------------------------------------------------------

    episode_prototypes = (
        defaultdict(list)
    )

    for i, meta in enumerate(
        prototype_metadata
    ):

        key = episode_key(
            meta
        )

        episode_prototypes[
            key
        ].append(i)

    print(
        f"Episodes : "
        f"{len(episode_prototypes)}"
    )

    # --------------------------------------------------------
    # Group original frames by episode
    # --------------------------------------------------------

    episode_frames = (
        defaultdict(list)
    )

    for i, meta in enumerate(
        metadata
    ):

        episode_frames[
            episode_key(meta)
        ].append(i)

    for key in episode_frames:

        episode_frames[key].sort(
            key=lambda i:
            float(
                metadata[i][
                    "timestamp_seconds"
                ]
            )
        )

    # --------------------------------------------------------
    # Random samples
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

    top1_correct = 0
    top5_correct = 0

    top1_media_correct = 0
    top5_media_correct = 0

    failures = []

    timestamp_errors = []

    # ========================================================
    # EVALUATION
    # ========================================================

    for sample_number, query_index in enumerate(
        sample_indices,
        start=1
    ):

        query_meta = metadata[
            query_index
        ]

        true_episode = episode_key(
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
        # Build 5-frame query
        # ----------------------------------------------------

        query_frame_indices = []

        true_frames = episode_frames[
            true_episode
        ]

        for offset in QUERY_OFFSETS:

            target_time = (
                query_time
                + offset
            )

            nearest = min(
                true_frames,
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

            query_frame_indices.append(
                nearest
            )

        # ----------------------------------------------------
        # Retrieve query embeddings
        # ----------------------------------------------------

        query_embeddings = []

        for frame_index in query_frame_indices:

            vector = index.reconstruct(
                frame_index
            )

            vector = normalize(
                vector
            )

            query_embeddings.append(
                vector
            )

        query_embeddings = np.vstack(
            query_embeddings
        )

        # ----------------------------------------------------
        # Score every episode
        # ----------------------------------------------------

        episode_scores = []

        for ep_key, proto_indices in (
            episode_prototypes.items()
        ):

            episode_vectors = prototypes[
                proto_indices
            ]

            # ------------------------------------------------
            # Query → episode prototype similarities
            # ------------------------------------------------

            similarity_matrix = (
                query_embeddings
                @ episode_vectors.T
            )

            # For each query frame, take its best
            # prototype match.
            best_per_query = (
                np.max(
                    similarity_matrix,
                    axis=1
                )
            )

            # ------------------------------------------------
            # Aggregate
            # ------------------------------------------------

            best_score = float(
                np.max(
                    best_per_query
                )
            )

            mean_score = float(
                np.mean(
                    best_per_query
                )
            )

            top3_count = min(
                3,
                len(best_per_query)
            )

            top3_mean = float(
                np.mean(
                    np.sort(
                        best_per_query
                    )[-top3_count:]
                )
            )

            # Consistency:
            # how many query frames have a strong
            # prototype match?
            coverage = float(
                np.mean(
                    best_per_query
                    >= 0.70
                )
            )

            # ------------------------------------------------
            # Final prototype score
            # ------------------------------------------------

            score = (
                0.35 * best_score
                + 0.35 * top3_mean
                + 0.20 * mean_score
                + 0.10 * coverage
            )

            # Timestamp estimate:
            # use the prototype belonging to the strongest
            # query-frame match.
            strongest_query = int(
                np.argmax(
                    best_per_query
                )
            )

            strongest_proto = int(
                np.argmax(
                    similarity_matrix[
                        strongest_query
                    ]
                )
            )

            global_proto_index = (
                proto_indices[
                    strongest_proto
                ]
            )

            prototype_time = (
                prototype_metadata[
                    global_proto_index
                ].get(
                    "timestamp_seconds",
                    None
                )
            )

            episode_scores.append(
                {
                    "episode_key":
                        ep_key,

                    "score":
                        float(score),

                    "best":
                        best_score,

                    "top3":
                        top3_mean,

                    "mean":
                        mean_score,

                    "coverage":
                        coverage,

                    "timestamp":
                        prototype_time,
                }
            )

        # ----------------------------------------------------
        # Rank
        # ----------------------------------------------------

        episode_scores.sort(
            key=lambda x:
            x["score"],
            reverse=True
        )

        top1 = episode_scores[0][
            "episode_key"
        ]

        top5 = [
            x["episode_key"]
            for x in episode_scores[:5]
        ]

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        if top1 == true_episode:

            top1_correct += 1

        else:

            failures.append(
                {
                    "query":
                        query_meta,

                    "predicted":
                        top1,

                    "score":
                        episode_scores[0][
                            "score"
                        ],

                    "true_score":
                        next(
                            (
                                x["score"]
                                for x in episode_scores
                                if x[
                                    "episode_key"
                                ]
                                == true_episode
                            ),
                            None
                        ),
                }
            )

        if true_episode in top5:

            top5_correct += 1

        # ----------------------------------------------------
        # Media
        # ----------------------------------------------------

        if top1[0] == query_media:

            top1_media_correct += 1

        if query_media in [
            x[0]
            for x in top5
        ]:

            top5_media_correct += 1

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        predicted_timestamp = (
            episode_scores[0][
                "timestamp"
            ]
        )

        if predicted_timestamp is not None:

            timestamp_errors.append(
                abs(
                    float(
                        predicted_timestamp
                    )
                    - query_time
                )
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        status = (
            "OK"
            if top1 == true_episode
            else "FAIL"
        )

        print(
            f"[{sample_number:02d}/"
            f"{SAMPLE_COUNT}] "
            f"{query_meta['media_id']} "
            f"{format_timestamp(query_time)} "
            f"-> "
            f"{top1} "
            f"| {status} "
            f"| score="
            f"{episode_scores[0]['score']:.3f}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("EPISODE PROTOTYPE RESULTS")
    print("=" * 80)

    print()
    print(
        f"Samples                 : "
        f"{SAMPLE_COUNT}"
    )

    print(
        f"Top-1 episode accuracy  : "
        f"{100 * top1_correct / SAMPLE_COUNT:.2f}%"
    )

    print(
        f"Top-5 episode accuracy  : "
        f"{100 * top5_correct / SAMPLE_COUNT:.2f}%"
    )

    print(
        f"Top-1 media accuracy    : "
        f"{100 * top1_media_correct / SAMPLE_COUNT:.2f}%"
    )

    print(
        f"Top-5 media accuracy    : "
        f"{100 * top5_media_correct / SAMPLE_COUNT:.2f}%"
    )

    if timestamp_errors:

        print()
        print(
            f"Mean timestamp error    : "
            f"{np.mean(timestamp_errors):.2f} sec"
        )

        print(
            f"Median timestamp error  : "
            f"{np.median(timestamp_errors):.2f} sec"
        )

    print()
    print(
        f"Top-1 failures          : "
        f"{len(failures)}"
    )

    # --------------------------------------------------------
    # Failures
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
                f"   Predicted : "
                f"{failure['predicted']}"
            )

            print(
                f"   Pred score: "
                f"{failure['score']:.4f}"
            )

            print(
                f"   True score: "
                f"{failure['true_score']:.4f}"
            )


if __name__ == "__main__":

    main()