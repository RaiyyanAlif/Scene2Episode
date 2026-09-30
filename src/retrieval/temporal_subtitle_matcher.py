import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBTITLE_INDEX_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "subtitle_index.json"
)


# ============================================================
# CONFIG
# ============================================================

# Dialogue window around the query timestamp.
WINDOW_SECONDS = 15.0


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):

    return set(
        normalize_text(text).split()
    )


def text_similarity(a, b):

    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    a_tokens = tokenize(a_norm)
    b_tokens = tokenize(b_norm)

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = (
        a_tokens & b_tokens
    )

    union = (
        a_tokens | b_tokens
    )

    token_score = (
        len(intersection)
        / len(union)
    )

    # Character n-gram-ish similarity using
    # common contiguous word pairs.
    a_words = a_norm.split()
    b_words = b_norm.split()

    a_bigrams = set(
        zip(
            a_words,
            a_words[1:]
        )
    )

    b_bigrams = set(
        zip(
            b_words,
            b_words[1:]
        )
    )

    if a_bigrams or b_bigrams:

        bigram_union = (
            a_bigrams | b_bigrams
        )

        bigram_intersection = (
            a_bigrams & b_bigrams
        )

        bigram_score = (
            len(bigram_intersection)
            / len(bigram_union)
        )

    else:

        bigram_score = 0.0

    return (
        0.70 * token_score
        + 0.30 * bigram_score
    )


# ============================================================
# TEMPORAL SUBTITLE MATCHER
# ============================================================

class TemporalSubtitleMatcher:

    def __init__(
        self,
        index_path=SUBTITLE_INDEX_PATH,
        window_seconds=WINDOW_SECONDS,
    ):

        with open(
            index_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.data = json.load(f)

        self.window_seconds = (
            float(window_seconds)
        )

        print(
            "TemporalSubtitleMatcher loaded."
        )

        print(
            f"  Episodes: "
            f"{len(self.data)}"
        )

    # ========================================================
    # GET DIALOGUE WINDOW
    # ========================================================

    def get_window(
        self,
        media_id,
        timestamp,
    ):

        episode = self.data.get(
            media_id
        )

        if episode is None:

            return []

        start_time = (
            float(timestamp)
            - self.window_seconds
        )

        end_time = (
            float(timestamp)
            + self.window_seconds
        )

        entries = []

        for entry in episode[
            "entries"
        ]:

            if (
                entry["end"]
                >= start_time
                and
                entry["start"]
                <= end_time
            ):

                entries.append(
                    entry
                )

        return entries

    # ========================================================
    # WINDOW TO TEXT
    # ========================================================

    def window_text(
        self,
        media_id,
        timestamp,
    ):

        entries = self.get_window(
            media_id,
            timestamp
        )

        return " ".join(
            entry["text"]
            for entry in entries
        )

    # ========================================================
    # RANK BY QUERY TEXT
    # ========================================================

    def rank_text(
        self,
        query_text,
    ):

        results = []

        for media_id, episode in (
            self.data.items()
        ):

            best_score = 0.0
            best_entry = None

            # ------------------------------------------------
            # First find the strongest individual line.
            # ------------------------------------------------

            for entry in episode[
                "entries"
            ]:

                score = text_similarity(
                    query_text,
                    entry["text"]
                )

                if score > best_score:

                    best_score = score
                    best_entry = entry

            # ------------------------------------------------
            # Then build a temporal window around
            # the strongest line.
            # ------------------------------------------------

            if best_entry is not None:

                center_time = (
                    best_entry["start"]
                )

                window_text = (
                    self.window_text(
                        media_id,
                        center_time
                    )
                )

                window_score = (
                    text_similarity(
                        query_text,
                        window_text
                    )
                )

            else:

                center_time = None
                window_text = ""
                window_score = 0.0

            # ------------------------------------------------
            # Combined score
            # ------------------------------------------------

            final_score = (
                0.40 * best_score
                + 0.60 * window_score
            )

            results.append(
                {
                    "media_id":
                        media_id,

                    "season":
                        episode[
                            "season"
                        ],

                    "episode":
                        episode[
                            "episode"
                        ],

                    "score":
                        float(
                            final_score
                        ),

                    "line_score":
                        float(
                            best_score
                        ),

                    "window_score":
                        float(
                            window_score
                        ),

                    "timestamp":
                        center_time,

                    "matched_text":
                        window_text,
                }
            )

        results.sort(
            key=lambda x:
            x["score"],
            reverse=True
        )

        return results

    # ========================================================
    # RANK BY TIMESTAMP + QUERY TEXT
    # ========================================================

    def rank_timestamp(
        self,
        query_text,
        candidate_media_ids=None,
        query_timestamp=None,
    ):

        if candidate_media_ids is None:

            candidate_media_ids = (
                list(
                    self.data.keys()
                )
            )

        results = []

        for media_id in candidate_media_ids:

            episode = self.data.get(
                media_id
            )

            if episode is None:
                continue

            # ------------------------------------------------
            # If a query timestamp is available, compare
            # against the dialogue around that timestamp.
            # ------------------------------------------------

            if query_timestamp is not None:

                query_window = (
                    self.window_text(
                        media_id,
                        query_timestamp
                    )
                )

                if query_window:

                    score = (
                        text_similarity(
                            query_text,
                            query_window
                        )
                    )

                    results.append(
                        {
                            "media_id":
                                media_id,

                            "season":
                                episode[
                                    "season"
                                ],

                            "episode":
                                episode[
                                    "episode"
                                ],

                            "score":
                                float(score),

                            "timestamp":
                                query_timestamp,

                            "matched_text":
                                query_window,
                        }
                    )

                    continue

            # ------------------------------------------------
            # Fallback: global search
            # ------------------------------------------------

            best_score = 0.0
            best_entry = None

            for entry in episode[
                "entries"
            ]:

                score = text_similarity(
                    query_text,
                    entry["text"]
                )

                if score > best_score:

                    best_score = score
                    best_entry = entry

            results.append(
                {
                    "media_id":
                        media_id,

                    "season":
                        episode[
                            "season"
                        ],

                    "episode":
                        episode[
                            "episode"
                        ],

                    "score":
                        float(
                            best_score
                        ),

                    "timestamp":
                        (
                            best_entry["start"]
                            if best_entry
                            else None
                        ),

                    "matched_text":
                        (
                            best_entry["text"]
                            if best_entry
                            else ""
                        ),
                }
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
    print("TEMPORAL SUBTITLE MATCHER SELF-TEST")
    print("=" * 70)

    matcher = TemporalSubtitleMatcher()

    # --------------------------------------------------------
    # Use a real subtitle from E01
    # --------------------------------------------------------

    e01 = matcher.data[
        "lucifer_s01_e01"
    ]

    sample = e01[
        "entries"
    ][100]

    query_text = sample[
        "text"
    ]

    query_time = sample[
        "start"
    ]

    print()
    print(
        "Query timestamp:",
        query_time
    )

    print(
        "Query text:",
        query_text
    )

    # --------------------------------------------------------
    # Global temporal search
    # --------------------------------------------------------

    results = matcher.rank_text(
        query_text
    )

    print()
    print(
        "Top matches:"
    )

    for i, result in enumerate(
        results[:5],
        start=1
    ):

        print(
            f"{i}. "
            f"{result['media_id']} "
            f"score="
            f"{result['score']:.4f} "
            f"line="
            f"{result['line_score']:.4f} "
            f"window="
            f"{result['window_score']:.4f}"
        )

    print()
    print(
        "Temporal subtitle matcher "
        "loaded successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    self_test()