import math
from collections import defaultdict


class SequenceMatcher:
    """
    Matches a query clip against candidate frames by considering
    the similarity pattern across multiple temporal frames.

    Query sequence:
        t-4s, t-2s, t, t+2s, t+4s

    Instead of treating each frame independently, we measure:
        1. Mean similarity
        2. Minimum similarity
        3. Similarity consistency
        4. Temporal ordering consistency
    """

    def __init__(
        self,
        mean_weight=0.40,
        min_weight=0.20,
        consistency_weight=0.20,
        ordering_weight=0.20,
    ):
        self.mean_weight = mean_weight
        self.min_weight = min_weight
        self.consistency_weight = consistency_weight
        self.ordering_weight = ordering_weight

    @staticmethod
    def _normalize(values):
        if not values:
            return []

        minimum = min(values)
        maximum = max(values)

        if maximum - minimum < 1e-8:
            return [1.0] * len(values)

        return [
            (v - minimum) / (maximum - minimum)
            for v in values
        ]

    @staticmethod
    def _consistency(similarities):
        """
        Measures whether the similarities are reasonably stable.

        A genuine temporal match should generally have
        strong similarity across neighboring frames rather
        than one isolated high-similarity frame.
        """

        if len(similarities) <= 1:
            return 0.0

        mean = sum(similarities) / len(similarities)

        variance = sum(
            (x - mean) ** 2
            for x in similarities
        ) / len(similarities)

        std = math.sqrt(variance)

        # Similarity values are approximately [0, 1].
        # Lower variation => higher consistency.
        score = 1.0 - min(std / 0.20, 1.0)

        return max(0.0, min(1.0, score))

    @staticmethod
    def _ordering_score(query_sims, candidate_sims):
        """
        Compare the relative similarity trajectory.

        We don't require exact values. We compare whether
        increases/decreases happen in approximately the same
        temporal direction.
        """

        if len(query_sims) < 3 or len(candidate_sims) < 3:
            return 0.0

        query_diff = [
            query_sims[i + 1] - query_sims[i]
            for i in range(len(query_sims) - 1)
        ]

        candidate_diff = [
            candidate_sims[i + 1] - candidate_sims[i]
            for i in range(len(candidate_sims) - 1)
        ]

        matches = 0

        for q, c in zip(query_diff, candidate_diff):
            # Ignore tiny changes.
            q_sign = 0 if abs(q) < 0.01 else (1 if q > 0 else -1)
            c_sign = 0 if abs(c) < 0.01 else (1 if c > 0 else -1)

            if q_sign == c_sign:
                matches += 1

        return matches / len(query_diff)

    def score(
        self,
        query_similarities,
        candidate_similarities,
    ):
        """
        Calculate sequence similarity.

        Both lists should correspond to the same temporal offsets.
        """

        if not query_similarities:
            return 0.0

        if not candidate_similarities:
            return 0.0

        n = min(
            len(query_similarities),
            len(candidate_similarities),
        )

        query_similarities = query_similarities[:n]
        candidate_similarities = candidate_similarities[:n]

        mean_score = sum(candidate_similarities) / n

        min_score = min(candidate_similarities)

        consistency_score = self._consistency(
            candidate_similarities
        )

        ordering_score = self._ordering_score(
            query_similarities,
            candidate_similarities,
        )

        final_score = (
            self.mean_weight * mean_score
            + self.min_weight * min_score
            + self.consistency_weight * consistency_score
            + self.ordering_weight * ordering_score
        )

        return final_score

    def rank_sequences(
        self,
        query_similarities,
        candidate_sequences,
    ):
        """
        candidate_sequences:

        [
            {
                "episode_key": (...),
                "timestamp_seconds": ...,
                "similarities": [...]
            },
            ...
        ]
        """

        results = []

        for candidate in candidate_sequences:

            similarities = candidate.get(
                "similarities",
                []
            )

            score = self.score(
                query_similarities,
                similarities,
            )

            result = dict(candidate)
            result["sequence_score"] = score

            results.append(result)

        results.sort(
            key=lambda x: x["sequence_score"],
            reverse=True,
        )

        return results


if __name__ == "__main__":

    matcher = SequenceMatcher()

    query = [
        0.82,
        0.88,
        0.94,
        0.91,
        0.86,
    ]

    candidates = [
        {
            "episode_key": ("test", 1, 1),
            "timestamp_seconds": 100,
            "similarities": [
                0.80,
                0.87,
                0.93,
                0.90,
                0.85,
            ],
        },
        {
            "episode_key": ("test", 1, 2),
            "timestamp_seconds": 200,
            "similarities": [
                0.60,
                0.95,
                0.72,
                0.94,
                0.61,
            ],
        },
    ]

    ranked = matcher.rank_sequences(
        query,
        candidates,
    )

    print("=" * 70)
    print("SEQUENCE MATCHER SELF-TEST")
    print("=" * 70)

    for i, result in enumerate(ranked, 1):
        print(
            f"{i}. "
            f"{result['episode_key']} "
            f"score={result['sequence_score']:.4f}"
        )

    print("\nSequence matcher loaded successfully.")