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

# Existing frame extraction interval.
# Your dataset uses one frame every 2 seconds.
FRAME_INTERVAL = 2.0

# Simulated short clip:
#
#       -4s  -2s   0s  +2s  +4s
#
# This gives us 5 frames from a 9-second temporal window.
CLIP_OFFSETS = [-4, -2, 0, 2, 4]

# Number of FAISS candidates retrieved for EACH query frame.
DEFAULT_TOP_K = 50


# ============================================================
# METADATA HELPERS
# ============================================================

def episode_key(item):
    """
    Identify the exact episode/media unit.
    """

    return (
        item.get("title"),
        item.get("season"),
        item.get("episode"),
    )


def build_lookup(metadata):
    """
    Create:

        (media_id, frame_number) -> global FAISS index
    """

    lookup = {}

    for index, item in enumerate(metadata):

        key = (
            item["media_id"],
            int(item["frame_id"]),
        )

        lookup[key] = index

    return lookup


def find_frame_index(
    metadata,
    title,
    season,
    episode,
    target_timestamp,
):
    """
    Find the frame closest to a requested timestamp.
    """

    best_index = None
    best_difference = float("inf")

    for index, item in enumerate(metadata):

        if item["title"] != title:
            continue

        if item.get("season") != season:
            continue

        if item.get("episode") != episode:
            continue

        current_timestamp = float(
            item["timestamp_seconds"]
        )

        difference = abs(
            current_timestamp -
            target_timestamp
        )

        if difference < best_difference:

            best_difference = difference
            best_index = index

    return best_index


def get_clip_indices(
    metadata,
    lookup,
    center_index,
):
    """
    Get the existing indexed frames corresponding to:

        t-4
        t-2
        t
        t+2
        t+4
    """

    center = metadata[center_index]

    media_id = center["media_id"]

    center_frame = int(
        center["frame_id"]
    )

    clip_indices = []

    for offset in CLIP_OFFSETS:

        frame_offset = int(
            offset / FRAME_INTERVAL
        )

        target_frame = (
            center_frame +
            frame_offset
        )

        key = (
            media_id,
            target_frame,
        )

        frame_index = lookup.get(key)

        if frame_index is not None:

            clip_indices.append(
                (
                    offset,
                    frame_index,
                )
            )

    return clip_indices


# ============================================================
# SINGLE FRAME RETRIEVAL
# ============================================================

def retrieve_frame(
    index,
    metadata,
    query_index,
    top_k,
):
    """
    Retrieve visually similar frames for one query frame.

    The exact query frame is removed from the results.
    """

    query_vector = index.reconstruct(
        int(query_index)
    ).reshape(1, -1)

    similarities, indices = index.search(
        query_vector,
        top_k + 1,
    )

    results = []

    for similarity, candidate_index in zip(
        similarities[0],
        indices[0],
    ):

        candidate_index = int(
            candidate_index
        )

        if candidate_index < 0:
            continue

        # Never allow the exact query frame
        # to become the answer.
        if candidate_index == query_index:
            continue

        results.append(
            {
                "index": candidate_index,
                "similarity": float(similarity),
                "metadata": metadata[
                    candidate_index
                ],
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# MULTI-FRAME CLIP RETRIEVAL
# ============================================================

def retrieve_clip(
    index,
    metadata,
    lookup,
    center_index,
    top_k,
):
    """
    Retrieve candidates independently for each frame
    in the simulated short clip.

    Then aggregate evidence by episode.
    """

    clip_indices = get_clip_indices(
        metadata,
        lookup,
        center_index,
    )

    episode_evidence = defaultdict(list)

    frame_results = []

    # --------------------------------------------------------
    # Retrieve every frame in the clip
    # --------------------------------------------------------

    for offset, query_index in clip_indices:

        results = retrieve_frame(
            index,
            metadata,
            query_index,
            top_k,
        )

        frame_results.append(
            {
                "offset": offset,
                "query_index": query_index,
                "results": results,
            }
        )

        # ----------------------------------------------------
        # Add each result as evidence for its episode
        # ----------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1,
        ):

            item = result["metadata"]

            key = episode_key(item)

            episode_evidence[key].append(
                {
                    "offset": offset,
                    "rank": rank,
                    "similarity": result[
                        "similarity"
                    ],
                    "index": result["index"],
                    "metadata": item,
                }
            )

    # --------------------------------------------------------
    # Score each episode
    # --------------------------------------------------------

    episode_results = []

    total_clip_frames = len(
        clip_indices
    )

    for key, evidence in episode_evidence.items():

        # ----------------------------------------------------
        # For each query-frame offset, keep the strongest
        # matching frame from this episode.
        # ----------------------------------------------------

        per_frame_best = {}

        for evidence_item in evidence:

            offset = evidence_item[
                "offset"
            ]

            if (
                offset not in per_frame_best
                or
                evidence_item["similarity"]
                >
                per_frame_best[offset][
                    "similarity"
                ]
            ):

                per_frame_best[offset] = (
                    evidence_item
                )

        frame_scores = [
            item["similarity"]
            for item in per_frame_best.values()
        ]

        if not frame_scores:
            continue

        # ----------------------------------------------------
        # Mean similarity across supported clip frames
        # ----------------------------------------------------

        mean_similarity = float(
            np.mean(frame_scores)
        )

        # ----------------------------------------------------
        # Strongest 3 frame matches
        # ----------------------------------------------------

        strongest_scores = sorted(
            frame_scores,
            reverse=True,
        )[:3]

        top3_mean = float(
            np.mean(strongest_scores)
        )

        # ----------------------------------------------------
        # Temporal coverage
        #
        # Example:
        #
        # 5 / 5 frames supported = 1.0
        # 4 / 5 = 0.8
        # 1 / 5 = 0.2
        # ----------------------------------------------------

        support_count = len(
            per_frame_best
        )

        coverage = (
            support_count /
            total_clip_frames
            if total_clip_frames > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Rank-weighted evidence
        #
        # Higher-ranked FAISS results contribute more.
        # ----------------------------------------------------

        rank_scores = []

        for evidence_item in evidence:

            rank = evidence_item["rank"]

            rank_weight = 1.0 / np.log2(
                rank + 1
            )

            rank_scores.append(
                evidence_item["similarity"]
                *
                rank_weight
            )

        rank_score = float(
            max(rank_scores)
        )

        # ----------------------------------------------------
        # Best supporting frame
        # ----------------------------------------------------

        best_evidence = max(
            evidence,
            key=lambda item: item["similarity"],
        )

        # ----------------------------------------------------
        # Final experimental score
        #
        # 35% strongest 3-frame evidence
        # 30% overall clip mean
        # 20% temporal coverage
        # 15% rank-weighted evidence
        # ----------------------------------------------------

        score = (
            0.35 * top3_mean
            +
            0.30 * mean_similarity
            +
            0.20 * coverage
            +
            0.15 * rank_score
        )

        episode_results.append(
            {
                "episode_key": key,
                "score": float(score),
                "top3_mean": top3_mean,
                "mean_similarity": mean_similarity,
                "coverage": float(coverage),
                "support_count": support_count,
                "rank_score": rank_score,
                "best_evidence": best_evidence,
            }
        )

    # --------------------------------------------------------
    # Highest episode score first
    # --------------------------------------------------------

    episode_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return (
        clip_indices,
        frame_results,
        episode_results,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query_meta,
    clip_indices,
    results,
):
    """
    Print top episode candidates.
    """

    print()
    print("=" * 115)

    print(
        f"GROUND TRUTH: "
        f"{query_meta['title']} "
        f"S{query_meta.get('season')} "
        f"E{query_meta.get('episode')} "
        f"@ {query_meta['timestamp_formatted']}"
    )

    print(
        "Clip offsets:",
        [
            offset
            for offset, _ in clip_indices
        ],
    )

    print("=" * 115)

    print(
        f"{'Rank':<6}"
        f"{'Episode':<32}"
        f"{'Score':<10}"
        f"{'Top3':<10}"
        f"{'Mean':<10}"
        f"{'Coverage':<12}"
        f"{'Support':<10}"
        f"{'Best Time':<14}"
    )

    print("-" * 115)

    for rank, result in enumerate(
        results[:15],
        start=1,
    ):

        best = result[
            "best_evidence"
        ]

        meta = best["metadata"]

        episode = (
            f"{meta['title']} "
            f"S{meta.get('season')} "
            f"E{meta.get('episode')}"
        )

        best_time = meta[
            "timestamp_formatted"
        ]

        print(
            f"{rank:<6}"
            f"{episode:<32}"
            f"{result['score']:<10.4f}"
            f"{result['top3_mean']:<10.4f}"
            f"{result['mean_similarity']:<10.4f}"
            f"{result['coverage']:<12.2f}"
            f"{result['support_count']:<10}"
            f"{best_time:<14}"
        )


# ============================================================
# KNOWN FAILURES
# ============================================================

def test_failures(
    index,
    metadata,
    lookup,
    top_k,
):
    """
    Test the exact seven failures from our
    previous 50-frame evaluation.
    """

    failures = [

        (
            "Lucifer",
            1,
            13,
            896.0,
        ),

        (
            "Stranger_Things",
            3,
            1,
            2990.0,
        ),

        (
            "Lucifer",
            1,
            3,
            490.0,
        ),

        (
            "Stranger_Things",
            2,
            1,
            1840.0,
        ),

        (
            "Stranger_Things",
            3,
            4,
            288.0,
        ),

        (
            "Lucifer",
            1,
            9,
            640.0,
        ),

        (
            "Stranger_Things",
            3,
            6,
            2570.0,
        ),
    ]

    corrected = 0

    print()
    print("=" * 115)
    print(
        "TEMPORAL CLIP RETRIEVAL — KNOWN FAILURES"
    )
    print("=" * 115)

    for (
        title,
        season,
        episode,
        target_timestamp,
    ) in failures:

        center_index = find_frame_index(
            metadata,
            title,
            season,
            episode,
            target_timestamp,
        )

        if center_index is None:

            print(
                f"\n[ERROR] Could not find "
                f"{title} S{season} E{episode}"
            )

            continue

        query_meta = metadata[
            center_index
        ]

        (
            clip_indices,
            frame_results,
            episode_results,
        ) = retrieve_clip(
            index,
            metadata,
            lookup,
            center_index,
            top_k,
        )

        print_results(
            query_meta,
            clip_indices,
            episode_results,
        )

        # ----------------------------------------------------
        # Check top-1 episode
        # ----------------------------------------------------

        if episode_results:

            predicted = episode_results[0]

            predicted_key = predicted[
                "episode_key"
            ]

            actual_key = episode_key(
                query_meta
            )

            if predicted_key == actual_key:

                corrected += 1

                print(
                    "\n>>> RESULT: "
                    "TEMPORAL CLIP RETRIEVAL CORRECT"
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

    parser = argparse.ArgumentParser(
        description=(
            "Scene2Episode temporal "
            "multi-frame retrieval"
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "FAISS candidates retrieved "
            "for each clip frame"
        ),
    )

    args = parser.parse_args()

    print("=" * 115)
    print(
        "SCENE2EPISODE TEMPORAL CLIP RETRIEVAL"
    )
    print("=" * 115)

    print(
        f"Top-K per frame: {args.top_k}"
    )

    print(
        f"Clip offsets   : {CLIP_OFFSETS}"
    )

    # --------------------------------------------------------
    # Load FAISS
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
    # Load metadata
    # --------------------------------------------------------

    print(
        "Loading metadata..."
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    print(
        f"Metadata: {len(metadata)}"
    )

    # --------------------------------------------------------
    # Build lookup
    # --------------------------------------------------------

    print(
        "Building frame lookup..."
    )

    lookup = build_lookup(
        metadata
    )

    print(
        f"Lookup entries: {len(lookup)}"
    )

    # --------------------------------------------------------
    # Test known failures
    # --------------------------------------------------------

    test_failures(
        index,
        metadata,
        lookup,
        args.top_k,
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()