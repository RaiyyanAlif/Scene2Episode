import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINGERPRINT_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "episode_fingerprints.npz"
)

FINGERPRINT_METADATA_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "episode_fingerprints_metadata.json"
)


# ============================================================
# CONFIG
# ============================================================

# How many neighboring fingerprint points to inspect
# around the best matching fingerprint.
NEIGHBOR_RADIUS = 2


# ============================================================
# HELPERS
# ============================================================

def episode_key(meta):
    """
    Unique episode identifier.

    Movies / animated:
        (media_id, None, None)

    Series:
        (media_id, season, episode)
    """

    return (
        meta.get("media_id"),
        meta.get("season"),
        meta.get("episode"),
    )


def cosine_similarity(query, vectors):
    """
    Compute cosine similarity between one normalized
    query vector and many normalized vectors.
    """

    query = np.asarray(
        query,
        dtype=np.float32
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    query_norm = np.linalg.norm(
        query
    )

    if query_norm == 0:
        return np.zeros(
            len(vectors),
            dtype=np.float32
        )

    query = query / query_norm

    return vectors @ query


# ============================================================
# FINGERPRINT RERANKER
# ============================================================

class FingerprintReranker:

    def __init__(
        self,
        fingerprint_path=FINGERPRINT_PATH,
        metadata_path=FINGERPRINT_METADATA_PATH,
        neighbor_radius=NEIGHBOR_RADIUS,
    ):

        self.fingerprint_path = Path(
            fingerprint_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self.neighbor_radius = (
            neighbor_radius
        )

        # ----------------------------------------------------
        # Load vectors
        # ----------------------------------------------------

        data = np.load(
            self.fingerprint_path
        )

        self.embeddings = (
            data["embeddings"]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

        if len(self.embeddings) != len(
            self.metadata
        ):

            raise ValueError(
                "Fingerprint embedding count "
                "does not match metadata count."
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        norms[
            norms == 0
        ] = 1.0

        self.embeddings = (
            self.embeddings / norms
        )

        # ----------------------------------------------------
        # Group fingerprints by episode
        # ----------------------------------------------------

        self.episode_indices = (
            defaultdict(list)
        )

        for index, meta in enumerate(
            self.metadata
        ):

            key = episode_key(
                meta
            )

            self.episode_indices[
                key
            ].append(index)

        # Make sure every episode is chronological.
        for key in self.episode_indices:

            self.episode_indices[key].sort(
                key=lambda i:
                float(
                    self.metadata[i][
                        "timestamp_seconds"
                    ]
                )
            )

        print(
            "FingerprintReranker loaded."
        )

        print(
            f"  Fingerprints : "
            f"{len(self.embeddings)}"
        )

        print(
            f"  Episodes     : "
            f"{len(self.episode_indices)}"
        )

    # ========================================================
    # FIND BEST FINGERPRINT
    # ========================================================

    def best_match_for_query(
        self,
        query_embedding,
        indices,
    ):
        """
        Find the strongest fingerprint match inside
        one episode.
        """

        if not indices:

            return None

        vectors = self.embeddings[
            indices
        ]

        similarities = cosine_similarity(
            query_embedding,
            vectors
        )

        best_local = int(
            np.argmax(
                similarities
            )
        )

        best_global = indices[
            best_local
        ]

        return {
            "index": best_global,

            "similarity": float(
                similarities[
                    best_local
                ]
            ),

            "timestamp_seconds": float(
                self.metadata[
                    best_global
                ][
                    "timestamp_seconds"
                ]
            ),
        }

    # ========================================================
    # LOCAL TEMPORAL WINDOW
    # ========================================================

    def local_temporal_score(
        self,
        query_embedding,
        indices,
        best_position,
    ):
        """
        Inspect neighboring fingerprint points around
        the strongest match.

        This checks whether the surrounding timeline also
        contains visually related material.
        """

        start = max(
            0,
            best_position
            - self.neighbor_radius
        )

        end = min(
            len(indices),
            best_position
            + self.neighbor_radius
            + 1
        )

        local_indices = indices[
            start:end
        ]

        vectors = self.embeddings[
            local_indices
        ]

        similarities = cosine_similarity(
            query_embedding,
            vectors
        )

        # Strongest local match
        best_score = float(
            np.max(
                similarities
            )
        )

        # Mean local similarity
        mean_score = float(
            np.mean(
                similarities
            )
        )

        # Number of reasonably strong neighbors
        strong_count = int(
            np.sum(
                similarities >= 0.70
            )
        )

        coverage = (
            strong_count
            / len(similarities)
        )

        # ----------------------------------------------------
        # Combined fingerprint score
        # ----------------------------------------------------

        score = (
            0.60 * best_score
            + 0.25 * mean_score
            + 0.15 * coverage
        )

        return {
            "score": float(score),

            "best_score": best_score,

            "mean_score": mean_score,

            "coverage": float(
                coverage
            ),

            "local_count": len(
                local_indices
            ),
        }

    # ========================================================
    # SCORE ONE EPISODE
    # ========================================================

    def score_episode(
        self,
        query_embeddings,
        episode,
    ):
        """
        Score an episode against a multi-frame query.

        Each query frame independently finds its best
        fingerprint location inside the episode.

        The strongest evidence is then combined with
        temporal neighborhood evidence.
        """

        indices = self.episode_indices.get(
            episode,
            []
        )

        if not indices:
            return None

        frame_results = []

        for query_index, query_embedding in enumerate(
            query_embeddings
        ):

            best = self.best_match_for_query(
                query_embedding,
                indices
            )

            if best is None:
                continue

            # Position inside the episode fingerprint list
            best_position = indices.index(
                best["index"]
            )

            local = self.local_temporal_score(
                query_embedding,
                indices,
                best_position
            )

            frame_results.append(
                {
                    "query_index":
                        query_index,

                    "fingerprint_index":
                        best["index"],

                    "timestamp_seconds":
                        best[
                            "timestamp_seconds"
                        ],

                    "best_similarity":
                        best[
                            "similarity"
                        ],

                    "local_score":
                        local[
                            "score"
                        ],

                    "local_best":
                        local[
                            "best_score"
                        ],

                    "local_mean":
                        local[
                            "mean_score"
                        ],

                    "coverage":
                        local[
                            "coverage"
                        ],
                }
            )

        if not frame_results:
            return None

        # ----------------------------------------------------
        # Aggregate query-frame evidence
        # ----------------------------------------------------

        similarities = np.array(
            [
                x["best_similarity"]
                for x in frame_results
            ],
            dtype=np.float32
        )

        local_scores = np.array(
            [
                x["local_score"]
                for x in frame_results
            ],
            dtype=np.float32
        )

        # Best frame evidence
        best_similarity = float(
            np.max(
                similarities
            )
        )

        # Average of strongest 3
        top_count = min(
            3,
            len(similarities)
        )

        top3_mean = float(
            np.mean(
                np.sort(
                    similarities
                )[-top_count:]
            )
        )

        # Overall mean
        mean_similarity = float(
            np.mean(
                similarities
            )
        )

        # Local temporal evidence
        best_local = float(
            np.max(
                local_scores
            )
        )

        mean_local = float(
            np.mean(
                local_scores
            )
        )

        # ----------------------------------------------------
        # Final fingerprint score
        # ----------------------------------------------------

        score = (
            0.35 * best_similarity
            + 0.25 * top3_mean
            + 0.15 * mean_similarity
            + 0.15 * best_local
            + 0.10 * mean_local
        )

        # Strongest temporal location
        strongest = max(
            frame_results,
            key=lambda x:
            x["local_score"]
        )

        return {
            "episode_key":
                episode,

            "score":
                float(score),

            "best_similarity":
                best_similarity,

            "top3_mean":
                top3_mean,

            "mean_similarity":
                mean_similarity,

            "best_local":
                best_local,

            "mean_local":
                mean_local,

            "timestamp_seconds":
                strongest[
                    "timestamp_seconds"
                ],

            "frame_results":
                frame_results,
        }

    # ========================================================
    # RANK EPISODES
    # ========================================================

    def rank(
        self,
        query_embeddings,
        candidate_episodes=None,
    ):
        """
        Rank candidate episodes.

        If candidate_episodes is None, all episodes
        are evaluated.
        """

        if candidate_episodes is None:

            candidate_episodes = (
                list(
                    self.episode_indices.keys()
                )
            )

        results = []

        for episode in candidate_episodes:

            result = self.score_episode(
                query_embeddings,
                episode
            )

            if result is not None:

                results.append(
                    result
                )

        results.sort(
            key=lambda x:
            x["score"],
            reverse=True
        )

        return results


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print()
    print("=" * 70)
    print("FINGERPRINT RERANKER SELF-TEST")
    print("=" * 70)

    reranker = FingerprintReranker()

    # Use the first fingerprint as a synthetic query.
    query = (
        reranker.embeddings[0]
    )

    first_episode = (
        reranker.metadata[0]
    )

    key = episode_key(
        first_episode
    )

    results = reranker.rank(
        query_embeddings=[
            query
        ],
        candidate_episodes=[
            key
        ]
    )

    if results:

        result = results[0]

        print()
        print(
            "Episode:",
            result["episode_key"]
        )

        print(
            "Score:",
            round(
                result["score"],
                4
            )
        )

        print(
            "Best similarity:",
            round(
                result[
                    "best_similarity"
                ],
                4
            )
        )

        print(
            "Best local:",
            round(
                result[
                    "best_local"
                ],
                4
            )
        )

        print(
            "Timestamp:",
            result[
                "timestamp_seconds"
            ]
        )

    print()
    print(
        "Fingerprint reranker loaded successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    self_test()