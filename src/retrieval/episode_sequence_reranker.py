from collections import defaultdict

from sequence_matcher import SequenceMatcher


class EpisodeSequenceReranker:
    """
    Episode-level reranker that combines:

    1. Existing CLIP similarity
    2. Temporal evidence
    3. OCR evidence
    4. Sequence-level similarity

    The sequence component checks whether the similarity pattern
    across the 5 query frames is consistent with the candidate
    episode.
    """

    def __init__(
        self,
        base_weight=0.75,
        sequence_weight=0.25,
    ):
        self.base_weight = base_weight
        self.sequence_weight = sequence_weight

        self.sequence_matcher = SequenceMatcher(
            mean_weight=0.40,
            min_weight=0.20,
            consistency_weight=0.20,
            ordering_weight=0.20,
        )

    @staticmethod
    def _episode_key(candidate):
        return (
            candidate.get("media_id"),
            candidate.get("season"),
            candidate.get("episode"),
        )

    @staticmethod
    def _get_query_frame_index(candidate):
        """
        Each retrieved candidate should contain query_frame_index.

        If unavailable, fall back to query_index.
        """

        if "query_frame_index" in candidate:
            return candidate["query_frame_index"]

        if "query_index" in candidate:
            return candidate["query_index"]

        return None

    @staticmethod
    def _get_candidate_time(candidate):
        return float(
            candidate.get(
                "timestamp_seconds",
                0.0
            )
        )

    def _build_sequences(
        self,
        candidates,
        query_frame_count,
    ):
        """
        Group candidates by episode and candidate timestamp.

        For each candidate timestamp, collect similarities
        corresponding to the different query frames.
        """

        episode_groups = defaultdict(list)

        for candidate in candidates:
            key = self._episode_key(candidate)

            episode_groups[key].append(candidate)

        sequences = []

        for episode_key, episode_candidates in episode_groups.items():

            # Group candidates that refer to approximately
            # the same location in the candidate video.
            timestamp_groups = []

            for candidate in episode_candidates:

                timestamp = self._get_candidate_time(
                    candidate
                )

                placed = False

                for group in timestamp_groups:

                    reference_time = group["timestamp"]

                    if abs(timestamp - reference_time) <= 6.0:
                        group["candidates"].append(candidate)
                        placed = True
                        break

                if not placed:
                    timestamp_groups.append(
                        {
                            "timestamp": timestamp,
                            "candidates": [candidate],
                        }
                    )

            for group in timestamp_groups:

                frame_scores = {}

                for candidate in group["candidates"]:

                    frame_index = self._get_query_frame_index(
                        candidate
                    )

                    if frame_index is None:
                        continue

                    similarity = float(
                        candidate.get(
                            "similarity",
                            candidate.get(
                                "clip_similarity",
                                0.0
                            )
                        )
                    )

                    # Keep the strongest candidate for each
                    # query-frame position.
                    previous = frame_scores.get(
                        frame_index
                    )

                    if (
                        previous is None
                        or similarity > previous
                    ):
                        frame_scores[frame_index] = similarity

                if not frame_scores:
                    continue

                similarities = [
                    frame_scores.get(
                        i,
                        0.0
                    )
                    for i in range(query_frame_count)
                ]

                sequences.append(
                    {
                        "episode_key": episode_key,
                        "timestamp_seconds": group["timestamp"],
                        "similarities": similarities,
                        "coverage": (
                            len(frame_scores)
                            / max(query_frame_count, 1)
                        ),
                    }
                )

        return sequences

    def rank(
        self,
        candidates,
        query_similarities=None,
        total_query_frames=5,
    ):
        """
        Rank episode candidates.

        candidates must contain:

            media_id
            season
            episode
            timestamp_seconds
            similarity / clip_similarity
            query_frame_index

        query_similarities is the similarity pattern of the
        query itself. If unavailable, a neutral pattern is used.
        """

        if not candidates:
            return []

        if query_similarities is None:

            query_similarities = [
                1.0
                for _ in range(total_query_frames)
            ]

        sequences = self._build_sequences(
            candidates,
            total_query_frames,
        )

        sequence_results = (
            self.sequence_matcher.rank_sequences(
                query_similarities,
                sequences,
            )
        )

        # Best sequence score per episode.
        best_sequence = {}

        for result in sequence_results:

            key = result["episode_key"]

            previous = best_sequence.get(key)

            if (
                previous is None
                or result["sequence_score"]
                > previous["sequence_score"]
            ):
                best_sequence[key] = result

        # Existing candidate evidence grouped by episode.
        episode_candidates = defaultdict(list)

        for candidate in candidates:

            key = self._episode_key(candidate)

            episode_candidates[key].append(candidate)

        ranked = []

        for episode_key, items in episode_candidates.items():

            similarities = sorted(
                [
                    float(
                        item.get(
                            "similarity",
                            item.get(
                                "clip_similarity",
                                0.0
                            )
                        )
                    )
                    for item in items
                ],
                reverse=True,
            )

            if not similarities:
                continue

            best_clip = similarities[0]

            top3_mean = sum(
                similarities[:3]
            ) / min(3, len(similarities))

            mean_clip = sum(
                similarities
            ) / len(similarities)

            # Base score preserves the existing retrieval evidence.
            base_score = (
                0.50 * best_clip
                + 0.30 * top3_mean
                + 0.20 * mean_clip
            )

            sequence_info = best_sequence.get(
                episode_key
            )

            if sequence_info is None:
                sequence_score = 0.0
                sequence_coverage = 0.0
                sequence_timestamp = None

            else:
                sequence_score = float(
                    sequence_info["sequence_score"]
                )

                sequence_coverage = float(
                    sequence_info.get(
                        "coverage",
                        0.0
                    )
                )

                sequence_timestamp = (
                    sequence_info["timestamp_seconds"]
                )

            final_score = (
                self.base_weight * base_score
                + self.sequence_weight * sequence_score
            )

            ranked.append(
                {
                    "episode_key": episode_key,
                    "score": final_score,

                    "base_score": base_score,
                    "sequence_score": sequence_score,

                    "best_clip": best_clip,
                    "top3_mean": top3_mean,
                    "mean_clip": mean_clip,

                    "sequence_coverage":
                        sequence_coverage,

                    "timestamp_seconds":
                        sequence_timestamp,

                    "candidate_count":
                        len(items),
                }
            )

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked


if __name__ == "__main__":

    print("=" * 70)
    print("EPISODE SEQUENCE RERANKER SELF-TEST")
    print("=" * 70)

    reranker = EpisodeSequenceReranker()

    candidates = []

    # Episode 1 — consistent sequence
    similarities_1 = [
        0.82,
        0.88,
        0.94,
        0.91,
        0.86,
    ]

    for i, similarity in enumerate(similarities_1):

        candidates.append(
            {
                "media_id": "test",
                "season": 1,
                "episode": 1,
                "timestamp_seconds": 100 + i * 2,
                "similarity": similarity,
                "query_frame_index": i,
            }
        )

    # Episode 2 — isolated high similarities
    similarities_2 = [
        0.95,
        0.60,
        0.94,
        0.61,
        0.93,
    ]

    for i, similarity in enumerate(similarities_2):

        candidates.append(
            {
                "media_id": "test",
                "season": 1,
                "episode": 2,
                "timestamp_seconds": 200 + i * 2,
                "similarity": similarity,
                "query_frame_index": i,
            }
        )

    ranked = reranker.rank(
        candidates,
        query_similarities=similarities_1,
        total_query_frames=5,
    )

    for i, result in enumerate(ranked, 1):

        print(
            f"{i}. "
            f"{result['episode_key']} "
            f"score={result['score']:.4f} "
            f"base={result['base_score']:.4f} "
            f"sequence={result['sequence_score']:.4f}"
        )

    print("\nEpisode sequence reranker loaded successfully.")