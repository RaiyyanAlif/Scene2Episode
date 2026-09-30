# src/retrieval/episode_temporal_reranker_v2.py

import numpy as np
from collections import defaultdict


class EpisodeTemporalRerankerV2:
    """
    Credit-aware temporal episode reranker.

    Normal scenes:
        CLIP + temporal evidence dominate.

    Credit/title-card scenes:
        OCR becomes significantly more important.

    IMPORTANT:
        Candidate leakage exclusion must happen BEFORE
        candidates reach this reranker.
    """

    def __init__(
        self,

        # -----------------------------------------------------------------
        # NORMAL SCENE
        # -----------------------------------------------------------------

        normal_clip_weight=0.45,
        normal_top3_weight=0.20,
        normal_mean_weight=0.10,
        normal_coverage_weight=0.10,
        normal_temporal_weight=0.10,
        normal_ocr_weight=0.05,

        # -----------------------------------------------------------------
        # CREDIT SCENE
        # -----------------------------------------------------------------

        credit_clip_weight=0.25,
        credit_top3_weight=0.10,
        credit_mean_weight=0.05,
        credit_coverage_weight=0.05,
        credit_temporal_weight=0.20,
        credit_ocr_weight=0.35,

        # Credit OCR threshold
        credit_threshold=0.35,
    ):

        self.normal_weights = {
            "clip": normal_clip_weight,
            "top3": normal_top3_weight,
            "mean": normal_mean_weight,
            "coverage": normal_coverage_weight,
            "temporal": normal_temporal_weight,
            "ocr": normal_ocr_weight,
        }

        self.credit_weights = {
            "clip": credit_clip_weight,
            "top3": credit_top3_weight,
            "mean": credit_mean_weight,
            "coverage": credit_coverage_weight,
            "temporal": credit_temporal_weight,
            "ocr": credit_ocr_weight,
        }

        self.credit_threshold = (
            credit_threshold
        )

    # =========================================================================
    # BEST FRAME
    # =========================================================================

    @staticmethod
    def best_frame(candidates):

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda x: float(
                x.get(
                    "clip_similarity",
                    0.0
                )
            )
        )

    # =========================================================================
    # TOP-3 CLIP
    # =========================================================================

    @staticmethod
    def top3_mean(candidates):

        if not candidates:
            return 0.0

        scores = sorted(
            [
                float(
                    x.get(
                        "clip_similarity",
                        0.0
                    )
                )
                for x in candidates
            ],
            reverse=True
        )

        return float(
            np.mean(
                scores[:3]
            )
        )

    # =========================================================================
    # MEAN CLIP
    # =========================================================================

    @staticmethod
    def mean_similarity(candidates):

        if not candidates:
            return 0.0

        scores = [
            float(
                x.get(
                    "clip_similarity",
                    0.0
                )
            )
            for x in candidates
        ]

        return float(
            np.mean(scores)
        )

    # =========================================================================
    # QUERY FRAME COVERAGE
    # =========================================================================

    @staticmethod
    def query_coverage(
        candidates,
        total_query_frames
    ):

        if not candidates:
            return 0.0

        query_ids = set()

        for candidate in candidates:

            if (
                "query_frame_id"
                in candidate
            ):

                query_ids.add(
                    candidate[
                        "query_frame_id"
                    ]
                )

            elif (
                "query_offset"
                in candidate
            ):

                query_ids.add(
                    candidate[
                        "query_offset"
                    ]
                )

        if not query_ids:
            return 0.0

        return min(
            1.0,
            len(query_ids)
            /
            max(
                1,
                total_query_frames
            )
        )

    # =========================================================================
    # TEMPORAL CONSISTENCY
    # =========================================================================

    @staticmethod
    def temporal_consistency(
        candidates
    ):

        if len(candidates) < 2:
            return 0.0

        scores = np.array(
            [
                float(
                    x.get(
                        "clip_similarity",
                        0.0
                    )
                )
                for x in candidates
            ],
            dtype=np.float32
        )

        # Keep reasonably strong candidates
        threshold = np.percentile(
            scores,
            60
        )

        strong = [
            candidate
            for candidate in candidates
            if float(
                candidate.get(
                    "clip_similarity",
                    0.0
                )
            ) >= threshold
        ]

        if len(strong) < 2:
            return 0.0

        timestamps = np.array(
            [
                float(
                    candidate[
                        "timestamp_seconds"
                    ]
                )
                for candidate in strong
            ],
            dtype=np.float32
        )

        timestamps.sort()

        # -------------------------------------------------------------
        # Densest local cluster
        # -------------------------------------------------------------

        best_cluster = 1

        for i in range(
            len(timestamps)
        ):

            count = 1

            for j in range(
                i + 1,
                len(timestamps)
            ):

                if (
                    timestamps[j]
                    -
                    timestamps[i]
                    <= 12.0
                ):

                    count += 1

                else:

                    break

            best_cluster = max(
                best_cluster,
                count
            )

        cluster_score = min(
            1.0,
            best_cluster / 4.0
        )

        # -------------------------------------------------------------
        # Temporal spread
        # -------------------------------------------------------------

        if len(timestamps) >= 2:

            spread = float(
                np.percentile(
                    timestamps,
                    75
                )
                -
                np.percentile(
                    timestamps,
                    25
                )
            )

            spread_score = max(
                0.0,
                1.0
                -
                spread / 60.0
            )

        else:

            spread_score = 0.0

        return float(
            0.7 * cluster_score
            +
            0.3 * spread_score
        )

    # =========================================================================
    # OCR
    # =========================================================================

    @staticmethod
    def mean_ocr(candidates):

        if not candidates:
            return 0.0

        values = [
            float(
                candidate.get(
                    "ocr_similarity",
                    0.0
                )
            )
            for candidate in candidates
        ]

        return float(
            np.mean(values)
        )

    # =========================================================================
    # BEST OCR
    # =========================================================================

    @staticmethod
    def best_ocr(candidates):

        if not candidates:
            return 0.0

        return max(
            float(
                candidate.get(
                    "ocr_similarity",
                    0.0
                )
            )
            for candidate in candidates
        )

    # =========================================================================
    # OCR CONSISTENCY
    # =========================================================================

    @staticmethod
    def ocr_consistency(candidates):

        if not candidates:
            return 0.0

        values = sorted(
            [
                float(
                    candidate.get(
                        "ocr_similarity",
                        0.0
                    )
                )
                for candidate in candidates
            ],
            reverse=True
        )

        if not values:
            return 0.0

        # Strong OCR evidence from multiple frames
        top = values[:3]

        return float(
            np.mean(top)
        )

    # =========================================================================
    # CREDIT DETECTION
    # =========================================================================

    def is_credit_scene(
        self,
        query_credit_score
    ):

        return (
            float(
                query_credit_score
            )
            >= self.credit_threshold
        )

    # =========================================================================
    # SCORE EPISODE
    # =========================================================================

    def score_episode(
        self,
        candidates,
        total_query_frames,
        credit_mode=False
    ):

        if not candidates:
            return None

        best = self.best_frame(
            candidates
        )

        best_clip = float(
            best.get(
                "clip_similarity",
                0.0
            )
        )

        top3 = self.top3_mean(
            candidates
        )

        mean_clip = self.mean_similarity(
            candidates
        )

        coverage = self.query_coverage(
            candidates,
            total_query_frames
        )

        temporal = self.temporal_consistency(
            candidates
        )

        mean_ocr = self.mean_ocr(
            candidates
        )

        best_ocr = self.best_ocr(
            candidates
        )

        ocr_consistency = (
            self.ocr_consistency(
                candidates
            )
        )

        # -------------------------------------------------------------
        # Select weights
        # -------------------------------------------------------------

        if credit_mode:

            weights = self.credit_weights

        else:

            weights = self.normal_weights

        # -------------------------------------------------------------
        # Base score
        # -------------------------------------------------------------

        final_score = (

            weights["clip"]
            * best_clip

            +

            weights["top3"]
            * top3

            +

            weights["mean"]
            * mean_clip

            +

            weights["coverage"]
            * coverage

            +

            weights["temporal"]
            * temporal

            +

            weights["ocr"]
            * ocr_consistency
        )

        # -------------------------------------------------------------
        # CREDIT-SPECIFIC BOOST
        # -------------------------------------------------------------

        if credit_mode:

            # Strong OCR match should matter significantly.
            if best_ocr >= 0.75:

                final_score += 0.08

            elif best_ocr >= 0.50:

                final_score += 0.04

            # If candidate has no OCR at all during a credit query,
            # slightly penalize it.
            elif best_ocr <= 0.01:

                final_score *= 0.92

        return {
            "episode_key": (
                best["media_id"],
                best.get("season"),
                best.get("episode"),
            ),

            "score": float(
                final_score
            ),

            "best_clip": best_clip,

            "top3_mean": top3,

            "mean_clip": mean_clip,

            "coverage": coverage,

            "temporal": temporal,

            "ocr": ocr_consistency,

            "best_ocr": best_ocr,

            "credit_mode": credit_mode,

            "best_candidate": best,

            "candidate_count": len(
                candidates
            ),
        }

    # =========================================================================
    # RANK EPISODES
    # =========================================================================

    def rank(
        self,
        candidates,
        total_query_frames,
        credit_mode=False
    ):

        grouped = defaultdict(list)

        for candidate in candidates:

            key = (
                candidate[
                    "media_id"
                ],
                candidate.get(
                    "season"
                ),
                candidate.get(
                    "episode"
                ),
            )

            grouped[key].append(
                candidate
            )

        ranked = []

        for (
            episode_key,
            episode_candidates
        ) in grouped.items():

            result = self.score_episode(
                episode_candidates,
                total_query_frames,
                credit_mode=credit_mode
            )

            if result is not None:

                ranked.append(
                    result
                )

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked


# =============================================================================
# SELF TEST
# =============================================================================

if __name__ == "__main__":

    print(
        "Credit-Aware Temporal Reranker V3 loaded."
    )

    reranker = (
        EpisodeTemporalRerankerV2()
    )

    # -------------------------------------------------------------
    # Normal scene test
    # -------------------------------------------------------------

    normal_candidates = [

        {
            "media_id": "test",
            "season": 1,
            "episode": 1,
            "timestamp_seconds": 100,
            "clip_similarity": 0.90,
            "ocr_similarity": 0.10,
            "query_frame_id": 0,
        },

        {
            "media_id": "test",
            "season": 1,
            "episode": 1,
            "timestamp_seconds": 104,
            "clip_similarity": 0.88,
            "ocr_similarity": 0.05,
            "query_frame_id": 1,
        },

        {
            "media_id": "test",
            "season": 1,
            "episode": 1,
            "timestamp_seconds": 108,
            "clip_similarity": 0.87,
            "ocr_similarity": 0.00,
            "query_frame_id": 2,
        },
    ]

    normal_result = reranker.rank(
        normal_candidates,
        total_query_frames=5,
        credit_mode=False
    )

    print()
    print(
        "Normal scene:"
    )

    for result in normal_result:

        print(
            result
        )

    # -------------------------------------------------------------
    # Credit scene test
    # -------------------------------------------------------------

    credit_candidates = [

        {
            "media_id": "credit_test",
            "season": 1,
            "episode": 1,
            "timestamp_seconds": 500,
            "clip_similarity": 0.85,
            "ocr_similarity": 1.00,
            "query_frame_id": 0,
        },

        {
            "media_id": "credit_test",
            "season": 1,
            "episode": 1,
            "timestamp_seconds": 504,
            "clip_similarity": 0.84,
            "ocr_similarity": 0.90,
            "query_frame_id": 1,
        },

        {
            "media_id": "credit_test",
            "season": 1,
            "episode": 1,
            "timestamp_seconds": 508,
            "clip_similarity": 0.83,
            "ocr_similarity": 0.80,
            "query_frame_id": 2,
        },
    ]

    credit_result = reranker.rank(
        credit_candidates,
        total_query_frames=5,
        credit_mode=True
    )

    print()
    print(
        "Credit scene:"
    )

    for result in credit_result:

        print(
            result
        )

    print()
    print(
        "Self-test complete."
    )