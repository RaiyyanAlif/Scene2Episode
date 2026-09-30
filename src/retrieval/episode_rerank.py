import argparse
import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "indexes" / "scene2episode.index"
METADATA_PATH = PROJECT_ROOT / "indexes" / "scene2episode_metadata.json"


# ============================================================
# CONFIG
# ============================================================

FRAME_INTERVAL = 2.0

# Search many frames so an episode gets multiple chances
# to provide evidence.
RETRIEVAL_K = 100

# Temporal neighborhood around each frame.
WINDOW_SECONDS = 10

OFFSETS = list(
    range(
        -WINDOW_SECONDS,
        WINDOW_SECONDS + 1,
        int(FRAME_INTERVAL)
    )
)


# ============================================================
# HELPERS
# ============================================================

def episode_key(item):
    return (
        item.get("title"),
        item.get("season"),
        item.get("episode")
    )


def media_key(item):
    return item["media_id"]


def frame_number(item):
    return int(item["frame_id"])


def timestamp(item):
    return float(item["timestamp_seconds"])


def build_frame_lookup(metadata):

    lookup = {}

    for index, item in enumerate(metadata):

        key = (
            item["media_id"],
            frame_number(item)
        )

        lookup[key] = index

    return lookup


def get_neighbor_indices(
    metadata,
    lookup,
    center_index
):

    center = metadata[center_index]

    media_id = center["media_id"]
    center_frame = frame_number(center)

    neighbors = []

    for offset in OFFSETS:

        frame_offset = int(
            offset / FRAME_INTERVAL
        )

        target_frame = (
            center_frame +
            frame_offset
        )

        key = (
            media_id,
            target_frame
        )

        neighbors.append(
            lookup.get(key)
        )

    return neighbors


# ============================================================
# TEMPORAL SCORE
# ============================================================

def temporal_score(
    index,
    metadata,
    lookup,
    query_index,
    candidate_index
):

    query_neighbors = get_neighbor_indices(
        metadata,
        lookup,
        query_index
    )

    candidate_neighbors = get_neighbor_indices(
        metadata,
        lookup,
        candidate_index
    )

    similarities = []

    for q_idx, c_idx in zip(
        query_neighbors,
        candidate_neighbors
    ):

        if q_idx is None or c_idx is None:
            continue

        q_vector = index.reconstruct(
            int(q_idx)
        )

        c_vector = index.reconstruct(
            int(c_idx)
        )

        similarity = float(
            np.dot(
                q_vector,
                c_vector
            )
        )

        similarities.append(
            similarity
        )

    if not similarities:
        return 0.0

    return float(
        np.median(similarities)
    )


# ============================================================
# EPISODE SCORE
# ============================================================

def calculate_episode_score(
    candidates
):

    """
    candidates contains all retrieved frames belonging
    to one episode.

    We combine:

        1. Best frame similarity
        2. Mean of top-5 frame similarities
        3. Best temporal consistency
        4. Mean temporal consistency of top frames
        5. Number of supporting frames

    The goal is to reward an episode that has multiple
    consistent pieces of evidence.
    """

    if not candidates:
        return None

    # --------------------------------------------------------
    # Sort by frame similarity
    # --------------------------------------------------------

    by_frame = sorted(
        candidates,
        key=lambda x: x["frame_similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # Best frame
    # --------------------------------------------------------

    best_frame = by_frame[0]["frame_similarity"]

    # --------------------------------------------------------
    # Top-5 frame mean
    # --------------------------------------------------------

    top5 = by_frame[:5]

    top5_mean = float(
        np.mean(
            [
                x["frame_similarity"]
                for x in top5
            ]
        )
    )

    # --------------------------------------------------------
    # Temporal scores
    # --------------------------------------------------------

    temporal_values = [
        x["temporal_score"]
        for x in candidates
        if x["temporal_score"] > 0
    ]

    if temporal_values:

        best_temporal = max(
            temporal_values
        )

        top_temporal = sorted(
            temporal_values,
            reverse=True
        )[:5]

        temporal_mean = float(
            np.mean(top_temporal)
        )

    else:

        best_temporal = 0.0
        temporal_mean = 0.0

    # --------------------------------------------------------
    # Evidence count
    # --------------------------------------------------------

    evidence_count = len(candidates)

    # Saturating evidence bonus.
    #
    # 1 frame  -> 0.00
    # 2 frames -> small bonus
    # ...
    # 10+      -> maximum bonus
    #

    evidence_bonus = min(
        evidence_count,
        10
    ) / 10.0

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------
    #
    # Initial experimental weighting:
    #
    # 25% best frame
    # 25% top-5 mean
    # 25% best temporal
    # 15% temporal mean
    # 10% evidence
    #

    score = (
        0.25 * best_frame +
        0.25 * top5_mean +
        0.25 * best_temporal +
        0.15 * temporal_mean +
        0.10 * evidence_bonus
    )

    return {
        "score": float(score),
        "best_frame": float(best_frame),
        "top5_mean": float(top5_mean),
        "best_temporal": float(best_temporal),
        "temporal_mean": float(temporal_mean),
        "evidence_count": evidence_count,
        "best_candidate": by_frame[0]
    }


# ============================================================
# RETRIEVE + RERANK
# ============================================================

def rerank_query(
    index,
    metadata,
    lookup,
    query_index,
    retrieval_k=RETRIEVAL_K
):

    # --------------------------------------------------------
    # Query embedding
    #
    # We reconstruct it directly because the query is one
    # of our indexed frames.
    # --------------------------------------------------------

    query_vector = index.reconstruct(
        int(query_index)
    ).reshape(1, -1)

    # --------------------------------------------------------
    # Initial FAISS retrieval
    # --------------------------------------------------------

    similarities, indices = index.search(
        query_vector,
        retrieval_k + 1
    )

    frame_candidates = []

    for similarity, candidate_index in zip(
        similarities[0],
        indices[0]
    ):

        candidate_index = int(
            candidate_index
        )

        if candidate_index < 0:
            continue

        # Remove exact query frame
        if candidate_index == query_index:
            continue

        frame_candidates.append(
            {
                "index": candidate_index,
                "frame_similarity": float(
                    similarity
                ),
                "metadata": metadata[
                    candidate_index
                ]
            }
        )

    # --------------------------------------------------------
    # Temporal verification
    # --------------------------------------------------------

    for candidate in frame_candidates:

        candidate["temporal_score"] = (
            temporal_score(
                index,
                metadata,
                lookup,
                query_index,
                candidate["index"]
            )
        )

    # --------------------------------------------------------
    # Group evidence by episode
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for candidate in frame_candidates:

        key = episode_key(
            candidate["metadata"]
        )

        grouped[key].append(
            candidate
        )

    # --------------------------------------------------------
    # Score every episode
    # --------------------------------------------------------

    episode_results = []

    for key, candidates in grouped.items():

        score_data = calculate_episode_score(
            candidates
        )

        if score_data is None:
            continue

        episode_results.append(
            {
                "episode_key": key,
                **score_data
            }
        )

    # --------------------------------------------------------
    # Sort episodes
    # --------------------------------------------------------

    episode_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return episode_results


# ============================================================
# FIND KNOWN FAILURE QUERY
# ============================================================

def find_query_index(
    metadata,
    title,
    season,
    episode,
    timestamp_value
):

    best_index = None
    best_difference = float("inf")

    for index, item in enumerate(metadata):

        if item["title"] != title:
            continue

        if item.get("season") != season:
            continue

        if item.get("episode") != episode:
            continue

        difference = abs(
            timestamp(item) -
            timestamp_value
        )

        if difference < best_difference:

            best_difference = difference
            best_index = index

    return best_index


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query_meta,
    results
):

    print()
    print("=" * 115)

    print(
        "QUERY"
    )

    print(
        f"{query_meta['title']} "
        f"S{query_meta.get('season')} "
        f"E{query_meta.get('episode')} "
        f"@ {query_meta['timestamp_formatted']}"
    )

    print("=" * 115)

    print(
        f"{'Rank':<6}"
        f"{'Episode':<30}"
        f"{'Score':<10}"
        f"{'Best':<10}"
        f"{'Top5':<10}"
        f"{'Temp':<10}"
        f"{'TMean':<10}"
        f"{'Evidence':<10}"
        f"{'Best Time':<14}"
    )

    print("-" * 115)

    for rank, result in enumerate(
        results,
        start=1
    ):

        best = result[
            "best_candidate"
        ]

        meta = best["metadata"]

        episode = (
            f"{meta['title']} "
            f"S{meta.get('season')} "
            f"E{meta.get('episode')}"
        )

        print(
            f"{rank:<6}"
            f"{episode:<30}"
            f"{result['score']:<10.4f}"
            f"{result['best_frame']:<10.4f}"
            f"{result['top5_mean']:<10.4f}"
            f"{result['best_temporal']:<10.4f}"
            f"{result['temporal_mean']:<10.4f}"
            f"{result['evidence_count']:<10}"
            f"{meta['timestamp_formatted']:<14}"
        )


# ============================================================
# TEST THE 7 KNOWN FAILURES
# ============================================================

def test_known_failures(
    index,
    metadata,
    lookup
):

    known_failures = [

        (
            "Lucifer",
            1,
            13,
            896.0
        ),

        (
            "Stranger_Things",
            3,
            1,
            2990.0
        ),

        (
            "Lucifer",
            1,
            3,
            490.0
        ),

        (
            "Stranger_Things",
            2,
            1,
            1840.0
        ),

        (
            "Stranger_Things",
            3,
            4,
            288.0
        ),

        (
            "Lucifer",
            1,
            9,
            640.0
        ),

        (
            "Stranger_Things",
            3,
            6,
            2570.0
        ),
    ]

    corrected = 0

    print()
    print("=" * 115)
    print("EPISODE-LEVEL RERANKING — KNOWN FAILURES")
    print("=" * 115)

    for (
        title,
        season,
        episode,
        query_timestamp
    ) in known_failures:

        query_index = find_query_index(
            metadata,
            title,
            season,
            episode,
            query_timestamp
        )

        if query_index is None:

            print(
                f"\n[ERROR] Could not find "
                f"{title} S{season} E{episode}"
            )

            continue

        query_meta = metadata[
            query_index
        ]

        results = rerank_query(
            index,
            metadata,
            lookup,
            query_index
        )

        print_results(
            query_meta,
            results
        )

        # ----------------------------------------------------
        # Check top-1
        # ----------------------------------------------------

        if results:

            predicted = results[0][
                "best_candidate"
            ]["metadata"]

            correct = (
                episode_key(predicted)
                ==
                episode_key(query_meta)
            )

            if correct:

                corrected += 1

                print(
                    "\n>>> RESULT: "
                    "EPISODE RERANKING CORRECTED TOP-1"
                )

            else:

                print(
                    "\n>>> RESULT: "
                    "STILL INCORRECT"
                )

    print()
    print("=" * 115)

    print(
        f"Known failures corrected: "
        f"{corrected}/7"
    )

    print("=" * 115)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-failures",
        action="store_true"
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=100
    )

    args = parser.parse_args()

    print("=" * 115)
    print("SCENE2EPISODE EPISODE-LEVEL RERANKING")
    print("=" * 115)

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print(
        "\nLoading FAISS index..."
    )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Vectors: {index.ntotal}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print(
        "Loading metadata..."
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    print(
        f"Metadata: {len(metadata)}"
    )

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    print(
        "Building frame lookup..."
    )

    lookup = build_frame_lookup(
        metadata
    )

    print(
        f"Frame lookup entries: {len(lookup)}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    if args.test_failures:

        global RETRIEVAL_K

        RETRIEVAL_K = args.retrieval_k

        test_known_failures(
            index,
            metadata,
            lookup
        )

    else:

        print()
        print(
            "Use:"
        )

        print(
            "python "
            ".\\src\\retrieval\\episode_rerank.py "
            "--test-failures"
        )

    print()


if __name__ == "__main__":
    main()