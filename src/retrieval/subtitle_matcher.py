import json
import re
from pathlib import Path
from difflib import SequenceMatcher


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

WINDOW_SECONDS = 4.0


# ============================================================
# HELPERS
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


def text_similarity(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    # Exact normalized match
    if a == b:
        return 1.0

    # Sequence similarity
    sequence_score = (
        SequenceMatcher(
            None,
            a,
            b
        ).ratio()
    )

    # Token overlap
    a_tokens = set(
        a.split()
    )

    b_tokens = set(
        b.split()
    )

    if not a_tokens or not b_tokens:

        token_score = 0.0

    else:

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

    return (
        0.65 * sequence_score
        + 0.35 * token_score
    )


# ============================================================
# MATCHER
# ============================================================

class SubtitleMatcher:

    def __init__(
        self,
        index_path=SUBTITLE_INDEX_PATH,
    ):

        with open(
            index_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.data = json.load(f)

        print(
            "SubtitleMatcher loaded."
        )

        print(
            f"  Episodes: "
            f"{len(self.data)}"
        )

        print(
            f"  Entries: "
            f"{sum(len(x['entries']) for x in self.data.values())}"
        )

    # ========================================================
    # GET TEXT AROUND TIMESTAMP
    # ========================================================

    def text_at_time(
        self,
        media_id,
        timestamp,
        window=WINDOW_SECONDS,
    ):

        episode = self.data.get(
            media_id
        )

        if episode is None:
            return ""

        start_time = (
            timestamp - window
        )

        end_time = (
            timestamp + window
        )

        texts = []

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

                texts.append(
                    entry["text"]
                )

        return " ".join(
            texts
        )

    # ========================================================
    # MATCH QUERY TEXT
    # ========================================================

    def rank(
        self,
        query_text,
    ):

        results = []

        for media_id, episode in (
            self.data.items()
        ):

            # Compare against all subtitle entries.
            best_score = 0.0
            best_text = ""
            best_time = None

            for entry in episode[
                "entries"
            ]:

                score = text_similarity(
                    query_text,
                    entry["text"]
                )

                if score > best_score:

                    best_score = score

                    best_text = (
                        entry["text"]
                    )

                    best_time = (
                        entry["start"]
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
                            best_score
                        ),

                    "matched_text":
                        best_text,

                    "timestamp":
                        best_time,
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
    print("SUBTITLE MATCHER SELF-TEST")
    print("=" * 70)

    matcher = SubtitleMatcher()

    # Use a real subtitle entry from E01.
    e01 = matcher.data[
        "lucifer_s01_e01"
    ]

    if not e01["entries"]:

        print(
            "No subtitle entries found."
        )

        return

    sample = e01[
        "entries"
    ][100]

    query_text = sample[
        "text"
    ]

    print()
    print(
        "Query text:"
    )

    print(
        query_text
    )

    print()
    print(
        "Expected episode:"
        " lucifer_s01_e01"
    )

    results = matcher.rank(
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
            f"{result['score']:.4f}"
        )

        print(
            f"   "
            f"{result['matched_text']}"
        )

    print()
    print(
        "Subtitle matcher loaded successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    self_test()