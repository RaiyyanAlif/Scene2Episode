import argparse
import json
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

# Frames are sampled every 2 seconds.
FRAME_INTERVAL = 2.0

# Compare +/- 10 seconds around the candidate.
WINDOW_SECONDS = 10

# Therefore:
# -10,-8,-6,-4,-2,0,+2,+4,+6,+8,+10
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


def build_frame_lookup(metadata):

    """
    Creates:

        (media_id, frame_number) -> global FAISS index
    """

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

    for offset_seconds in OFFSETS:

        frame_offset = int(
            offset_seconds / FRAME_INTERVAL
        )

        target_frame = (
            center_frame +
            frame_offset
        )

        key = (
            media_id,
            target_frame
        )

        neighbor_index = lookup.get(key)

        if neighbor_index is None:

            neighbors.append(None)

        else:

            neighbors.append(neighbor_index)

    return neighbors


def reconstruct_embeddings(
    index,
    indices
):

    valid = [
        idx
        for idx in indices
        if idx is not None
    ]

    if not valid:

        return None

    vectors = np.array(
        [
            index.reconstruct(int(idx))
            for idx in valid
        ],
        dtype="float32"
    )

    # Normalize
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vectors = vectors / np.maximum(
        norms,
        1e-12
    )

    return vectors


# ============================================================
# TEMPORAL SCORE
# ============================================================

def temporal_similarity(
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

    # --------------------------------------------------------
    # Compare corresponding temporal positions.
    # --------------------------------------------------------

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

        # Both vectors are already normalized because
        # they were normalized before being inserted into FAISS.
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

    # --------------------------------------------------------
    # Median is intentionally used instead of mean.
    #
    # One bad frame should not destroy the entire
    # temporal score.
    # --------------------------------------------------------

    return float(
        np.median(similarities)
    )


# ============================================================
# COMBINED SCORE
# ============================================================

def combined_score(
    frame_similarity,
    temporal_score
):

    """
    Combine:

        single-frame similarity
        temporal consistency

    Initial weighting:

        40% frame
        60% temporal

    Temporal consistency gets more weight because that is
    exactly what we're testing.
    """

    return (
        0.40 * frame_similarity +
        0.60 * temporal_score
    )


# ============================================================
# VERIFY QUERY
# ============================================================

def verify_query(
    index,
    metadata,
    lookup,
    query_index,
    top_k=10
):

    # --------------------------------------------------------
    # Query vector
    # --------------------------------------------------------

    query_vector = index.reconstruct(
        int(query_index)
    ).reshape(1, -1)

    # --------------------------------------------------------
    # Retrieve initial candidates.
    #
    # +1 because the query frame itself is inside FAISS.
    # --------------------------------------------------------

    similarities, indices = index.search(
        query_vector,
        top_k + 1
    )

    candidates = []

    for similarity, candidate_index in zip(
        similarities[0],
        indices[0]
    ):

        candidate_index = int(
            candidate_index
        )

        if candidate_index < 0:
            continue

        # Exclude exact query frame.
        if candidate_index == query_index:
            continue

        frame_similarity = float(
            similarity
        )

        temporal_score = temporal_similarity(
            index,
            metadata,
            lookup,
            query_index,
            candidate_index
        )

        final_score = combined_score(
            frame_similarity,
            temporal_score
        )

        candidates.append(
            {
                "index": candidate_index,
                "frame_similarity":
                    frame_similarity,
                "temporal_score":
                    temporal_score,
                "combined_score":
                    final_score,
                "metadata":
                    metadata[candidate_index]
            }
        )

        if len(candidates) >= top_k:
            break

    # --------------------------------------------------------
    # Sort using temporal-aware score.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return candidates


# ============================================================
# PRINT RESULT
# ============================================================

def print_results(
    query_meta,
    candidates
):

    print()
    print("=" * 90)

    print(
        "QUERY"
    )

    print(
        f"{query_meta['title']} "
        f"S{query_meta.get('season')} "
        f"E{query_meta.get('episode')} "
        f"@ {query_meta['timestamp_formatted']}"
    )

    print("=" * 90)

    print(
        f"{'Rank':<6}"
        f"{'Episode':<28}"
        f"{'Time':<14}"
        f"{'Frame':<12}"
        f"{'FrameSim':<12}"
        f"{'TempScore':<12}"
        f"{'Final':<10}"
    )

    print("-" * 90)

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        meta = candidate["metadata"]

        episode = (
            f"{meta['title']} "
            f"S{meta.get('season')} "
            f"E{meta.get('episode')}"
        )

        print(
            f"{rank:<6}"
            f"{episode:<28}"
            f"{meta['timestamp_formatted']:<14}"
            f"{meta['frame_id']:<12}"
            f"{candidate['frame_similarity']:<12.4f}"
            f"{candidate['temporal_score']:<12.4f}"
            f"{candidate['combined_score']:<10.4f}"
        )


# ============================================================
# TEST KNOWN FAILURES
# ============================================================

def test_known_failures(
    index,
    metadata,
    lookup
):

    """
    Test the exact 7 failures from our previous evaluation.
    """

    known_failures = [
        ("Lucifer", 1, 13, 896.0),
        ("Stranger_Things", 3, 1, 2990.0),
        ("Lucifer", 1, 3, 490.0),
        ("Stranger_Things", 2, 1, 1840.0),
        ("Stranger_Things", 3, 4, 288.0),
        ("Lucifer", 1, 9, 640.0),
        ("Stranger_Things", 3, 6, 2570.0),
    ]

    print()
    print("=" * 90)
    print("TESTING KNOWN FAILURES")
    print("=" * 90)

    for title, season, episode, timestamp in known_failures:

        query_index = None

        # ----------------------------------------------------
        # Find closest frame to requested timestamp.
        # ----------------------------------------------------

        best_difference = float("inf")

        for i, item in enumerate(metadata):

            if item["title"] != title:
                continue

            if item.get("season") != season:
                continue

            if item.get("episode") != episode:
                continue

            difference = abs(
                float(
                    item["timestamp_seconds"]
                ) - timestamp
            )

            if difference < best_difference:

                best_difference = difference
                query_index = i

        if query_index is None:

            print(
                f"[ERROR] Could not find "
                f"{title} S{season} E{episode}"
            )

            continue

        query_meta = metadata[
            query_index
        ]

        candidates = verify_query(
            index,
            metadata,
            lookup,
            query_index,
            top_k=10
        )

        print_results(
            query_meta,
            candidates
        )

        # ----------------------------------------------------
        # Report whether temporal verification corrected it.
        # ----------------------------------------------------

        if candidates:

            prediction = candidates[0]["metadata"]

            correct = (
                episode_key(prediction)
                ==
                episode_key(query_meta)
            )

            if correct:

                print(
                    "\nRESULT: TEMPORAL VERIFICATION "
                    "CORRECTED THE TOP-1"
                )

            else:

                print(
                    "\nRESULT: STILL INCORRECT"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-k",
        type=int,
        default=10
    )

    parser.add_argument(
        "--test-failures",
        action="store_true"
    )

    args = parser.parse_args()

    print("=" * 90)
    print("SCENE2EPISODE TEMPORAL VERIFICATION")
    print("=" * 90)

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
        encoding="utf-8"
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

    lookup = build_frame_lookup(
        metadata
    )

    print(
        f"Frame lookup entries: {len(lookup)}"
    )

    # --------------------------------------------------------
    # Test known failures
    # --------------------------------------------------------

    if args.test_failures:

        test_known_failures(
            index,
            metadata,
            lookup
        )

    else:

        print()
        print(
            "Use --test-failures to test "
            "the 7 known failure cases."
        )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()