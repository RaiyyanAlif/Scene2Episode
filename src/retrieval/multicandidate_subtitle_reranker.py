# ============================================================
# Scene2Episode
# Multi-Candidate Temporal Subtitle Reranker
# ============================================================

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBTITLE_INDEX_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "subtitle_index.json"
)

WINDOW_SECONDS = 15.0


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = str(text).upper()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Keep letters/numbers/basic punctuation
    text = re.sub(r"[^A-Z0-9À-ÿ'\- ]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    text = normalize_text(text)

    if not text:
        return set()

    return set(text.split())


def text_similarity(a, b):
    """
    Similarity between two subtitle texts.

    65% character sequence similarity
    35% token Jaccard similarity
    """

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a or not tokens_b:
        token_score = 0.0
    else:
        intersection = len(
            tokens_a & tokens_b
        )

        union = len(
            tokens_a | tokens_b
        )

        token_score = (
            intersection / union
            if union
            else 0.0
        )

    score = (
        0.65 * sequence_score
        +
        0.35 * token_score
    )

    return float(score)


# ============================================================
# RERANKER
# ============================================================

class MultiCandidateSubtitleReranker:

    def __init__(
        self,
        subtitle_index_path=None,
        window_seconds=WINDOW_SECONDS
    ):

        if subtitle_index_path is None:
            subtitle_index_path = (
                SUBTITLE_INDEX_PATH
            )

        self.subtitle_index_path = Path(
            subtitle_index_path
        )

        self.window_seconds = float(
            window_seconds
        )

        self.subtitle_data = (
            self._load_subtitle_index()
        )

        self.entries_by_media = defaultdict(
            list
        )

        self._build_entries()

        # Cache:
        # (media_id, rounded_timestamp)
        # -> subtitle window text
        self.window_cache = {}

    # ========================================================
    # LOAD INDEX
    # ========================================================

    def _load_subtitle_index(self):

        with open(
            self.subtitle_index_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    # ========================================================
    # NORMALIZE ENTRY
    # ========================================================

    def _normalize_entry(self, entry):

        if not isinstance(entry, dict):
            return None

        start = (
            entry.get("start")
            if entry.get("start") is not None
            else entry.get("start_seconds")
        )

        end = (
            entry.get("end")
            if entry.get("end") is not None
            else entry.get("end_seconds")
        )

        text = (
            entry.get("text")
            if entry.get("text") is not None
            else entry.get("content")
        )

        if text is None:
            text = entry.get("subtitle")

        if start is None or end is None:
            return None

        try:
            start = float(start)
            end = float(end)
        except (
            TypeError,
            ValueError
        ):
            return None

        if text is None:
            text = ""

        return {
            "start": start,
            "end": end,
            "text": str(text)
        }

    # ========================================================
    # EXTRACT ENTRIES
    # ========================================================

    def _extract_entries(self, value):

        # Direct list
        if isinstance(value, list):

            output = []

            for item in value:

                normalized = (
                    self._normalize_entry(item)
                )

                if normalized is not None:
                    output.append(
                        normalized
                    )

            return output

        # Dictionary
        if isinstance(value, dict):

            # Common container keys
            for key in (
                "entries",
                "subtitles",
                "segments",
                "lines",
                "items",
                "data"
            ):

                if key in value:

                    extracted = (
                        self._extract_entries(
                            value[key]
                        )
                    )

                    if extracted:
                        return extracted

            # Maybe this dict itself is a subtitle entry
            normalized = (
                self._normalize_entry(value)
            )

            if normalized is not None:
                return [normalized]

            # Maybe nested episode/media mapping
            output = []

            for nested_value in value.values():

                extracted = (
                    self._extract_entries(
                        nested_value
                    )
                )

                if extracted:
                    output.extend(
                        extracted
                    )

            return output

        return []

    # ========================================================
    # BUILD ENTRY INDEX
    # ========================================================

    def _build_entries(self):

        data = self.subtitle_data

        # ----------------------------------------------------
        # Most likely format:
        #
        # {
        #     "lucifer_s01_e01": [...],
        #     "lucifer_s01_e05": [...]
        # }
        # ----------------------------------------------------

        if isinstance(data, dict):

            for media_id, value in data.items():

                entries = (
                    self._extract_entries(
                        value
                    )
                )

                if entries:

                    self.entries_by_media[
                        str(media_id)
                    ].extend(
                        entries
                    )

        # ----------------------------------------------------
        # Alternative format:
        #
        # [
        #   {
        #       "media_id": "...",
        #       "entries": [...]
        #   }
        # ]
        # ----------------------------------------------------

        elif isinstance(data, list):

            for item in data:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                media_id = (
                    item.get("media_id")
                    or item.get("id")
                    or item.get("episode_id")
                )

                if media_id is None:
                    continue

                entries = (
                    self._extract_entries(
                        item
                    )
                )

                if entries:

                    self.entries_by_media[
                        str(media_id)
                    ].extend(
                        entries
                    )

        # Sort every episode by timestamp

        for media_id in (
            self.entries_by_media
        ):

            self.entries_by_media[
                media_id
            ].sort(
                key=lambda x: x["start"]
            )

    # ========================================================
    # SAFE TIMESTAMP
    # ========================================================

    @staticmethod
    def _safe_timestamp(value):

        """
        Convert a timestamp to float.

        This deliberately rejects tuples/lists/dicts
        so an episode key can never accidentally be
        interpreted as a timestamp.
        """

        if isinstance(
            value,
            (tuple, list, dict)
        ):

            raise TypeError(
                "Timestamp must be numeric, "
                f"got {type(value).__name__}: {value}"
            )

        return float(value)

    # ========================================================
    # GET WINDOW
    # ========================================================

    def get_window(
        self,
        media_id,
        timestamp
    ):

        media_id = str(media_id)

        timestamp = (
            self._safe_timestamp(
                timestamp
            )
        )

        cache_key = (
            media_id,
            round(timestamp, 3)
        )

        if cache_key in self.window_cache:

            return self.window_cache[
                cache_key
            ]

        entries = self.entries_by_media.get(
            media_id,
            []
        )

        if not entries:

            self.window_cache[
                cache_key
            ] = ""

            return ""

        start_time = (
            timestamp
            -
            self.window_seconds
        )

        end_time = (
            timestamp
            +
            self.window_seconds
        )

        selected = []

        for entry in entries:

            entry_start = float(
                entry["start"]
            )

            entry_end = float(
                entry["end"]
            )

            # Overlap test
            if (
                entry_end >= start_time
                and
                entry_start <= end_time
            ):

                selected.append(
                    entry["text"]
                )

        result = " ".join(
            selected
        )

        result = normalize_text(
            result
        )

        self.window_cache[
            cache_key
        ] = result

        return result

    # ========================================================
    # CANDIDATE METADATA
    # ========================================================

    @staticmethod
    def _candidate_metadata(
        candidate
    ):

        if (
            isinstance(candidate, dict)
            and
            "metadata" in candidate
            and
            isinstance(
                candidate["metadata"],
                dict
            )
        ):

            return candidate["metadata"]

        return candidate

    # ========================================================
    # CANDIDATE MEDIA ID
    # ========================================================

    def _candidate_media_id(
        self,
        candidate,
        episode_key=None
    ):

        metadata = (
            self._candidate_metadata(
                candidate
            )
        )

        media_id = metadata.get(
            "media_id"
        )

        if media_id is not None:
            return str(media_id)

        # Fallback only
        if (
            isinstance(
                episode_key,
                tuple
            )
            and len(episode_key) >= 1
        ):

            return str(
                episode_key[0]
            )

        if episode_key is not None:
            return str(
                episode_key
            )

        return None

    # ========================================================
    # CANDIDATE TIMESTAMP
    # ========================================================

    def _candidate_timestamp(
        self,
        candidate
    ):

        metadata = (
            self._candidate_metadata(
                candidate
            )
        )

        value = metadata.get(
            "timestamp_seconds"
        )

        if value is None:

            value = metadata.get(
                "timestamp"
            )

        if value is None:

            raise KeyError(
                "Candidate metadata does not "
                "contain timestamp_seconds."
            )

        return self._safe_timestamp(
            value
        )

    # ========================================================
    # SCORE ONE CANDIDATE
    # ========================================================

    def score_candidate(
        self,
        query_window,
        episode_key,
        candidate
    ):

        # IMPORTANT:
        # episode_key is NOT a timestamp.
        #
        # Candidate timestamp always comes
        # from candidate metadata.

        candidate_media_id = (
            self._candidate_media_id(
                candidate,
                episode_key
            )
        )

        if candidate_media_id is None:
            return 0.0

        candidate_timestamp = (
            self._candidate_timestamp(
                candidate
            )
        )

        candidate_window = (
            self.get_window(
                candidate_media_id,
                candidate_timestamp
            )
        )

        if not query_window:
            return 0.0

        if not candidate_window:
            return 0.0

        return float(
            text_similarity(
                query_window,
                candidate_window
            )
        )

    # ========================================================
    # SCORE EPISODE
    # ========================================================

    def score_episode(
        self,
        query_window,
        episode_key,
        candidates
    ):

        """
        Score ALL retrieved candidate frames
        belonging to one episode.

        Correct call:

            score_episode(
                query_window,
                episode_key,
                candidates
            )
        """

        if not candidates:

            return {
                "episode_key": episode_key,
                "score": 0.0,
                "best_score": 0.0,
                "top3_mean": 0.0,
                "evidence_score": 0.0,
                "evidence_count": 0,
                "best_candidate": None
            }

        scores = []

        for candidate in candidates:

            score = (
                self.score_candidate(
                    query_window,
                    episode_key,
                    candidate
                )
            )

            scores.append(
                (
                    float(score),
                    candidate
                )
            )

        if not scores:

            return {
                "episode_key": episode_key,
                "score": 0.0,
                "best_score": 0.0,
                "top3_mean": 0.0,
                "evidence_score": 0.0,
                "evidence_count": 0,
                "best_candidate": None
            }

        scores.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # ----------------------------------------------------
        # Best subtitle match
        # ----------------------------------------------------

        best_score = scores[0][0]

        best_candidate = (
            scores[0][1]
        )

        # ----------------------------------------------------
        # Top-3 mean
        # ----------------------------------------------------

        top3_scores = [
            item[0]
            for item in scores[:3]
        ]

        top3_mean = (
            sum(top3_scores)
            /
            len(top3_scores)
        )

        # ----------------------------------------------------
        # Evidence count
        # ----------------------------------------------------

        strong_count = sum(
            1
            for score, _ in scores
            if score >= 0.30
        )

        evidence_score = min(
            1.0,
            strong_count / 3.0
        )

        # ----------------------------------------------------
        # Final subtitle score
        # ----------------------------------------------------

        final_score = (
            0.60 * best_score
            +
            0.25 * top3_mean
            +
            0.15 * evidence_score
        )

        return {
            "episode_key": episode_key,
            "score": float(
                final_score
            ),
            "best_score": float(
                best_score
            ),
            "top3_mean": float(
                top3_mean
            ),
            "evidence_score": float(
                evidence_score
            ),
            "evidence_count": int(
                strong_count
            ),
            "best_candidate": (
                best_candidate
            )
        }

    # ========================================================
    # CACHE INFO
    # ========================================================

    def cache_size(self):

        return len(
            self.window_cache
        )


# ============================================================
# SELF TEST
# ============================================================

def main():

    print("=" * 75)
    print(
        "MULTI-CANDIDATE SUBTITLE RERANKER"
    )
    print("=" * 75)

    reranker = (
        MultiCandidateSubtitleReranker()
    )

    print(
        f"Subtitle episodes: "
        f"{len(reranker.entries_by_media)}"
    )

    total_entries = sum(
        len(entries)
        for entries
        in reranker.entries_by_media.values()
    )

    print(
        f"Subtitle entries: "
        f"{total_entries}"
    )

    # --------------------------------------------------------
    # Known Lucifer subtitle example
    # --------------------------------------------------------

    media_id = "lucifer_s01_e01"
    timestamp = 388.933

    query_window = (
        reranker.get_window(
            media_id,
            timestamp
        )
    )

    print()
    print(
        f"Query media: {media_id}"
    )

    print(
        f"Query timestamp: "
        f"{timestamp:.3f}"
    )

    print(
        f"Query window length: "
        f"{len(query_window)}"
    )

    if query_window:

        print()
        print(
            "Query subtitle window:"
        )

        print(
            query_window[:500]
        )

    # --------------------------------------------------------
    # Fake candidate for self-test
    # --------------------------------------------------------

    candidate = {
        "metadata": {
            "media_id": media_id,
            "season": 1,
            "episode": 1,
            "timestamp_seconds": timestamp
        },
        "clip_similarity": 1.0
    }

    episode_key = (
        media_id,
        1,
        1
    )

    result = (
        reranker.score_episode(
            query_window,
            episode_key,
            [candidate]
        )
    )

    print()
    print(
        f"Best subtitle score: "
        f"{result['best_score']:.4f}"
    )

    print(
        f"Top-3 mean: "
        f"{result['top3_mean']:.4f}"
    )

    print(
        f"Evidence count: "
        f"{result['evidence_count']}"
    )

    print(
        f"Final episode score: "
        f"{result['score']:.4f}"
    )

    if result["score"] > 0:

        print()
        print(
            "SELF-TEST: PASS"
        )

    else:

        print()
        print(
            "SELF-TEST: FAIL"
        )


if __name__ == "__main__":
    main()