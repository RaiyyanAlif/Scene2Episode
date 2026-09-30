import json
import random
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

SAMPLE_COUNT = 100
RANDOM_SEED = 42


# ============================================================
# TEXT NORMALIZATION
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

    if a == b:
        return 1.0

    sequence_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

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
# LOAD DATA
# ============================================================

def load_subtitles():

    with open(
        SUBTITLE_INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SUBTITLE MATCHER BENCHMARK")
    print("=" * 80)

    data = load_subtitles()

    print()
    print(
        f"Episodes indexed : "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # Build flat list of subtitle entries
    # --------------------------------------------------------

    entries = []

    for media_id, episode in data.items():

        for entry in episode[
            "entries"
        ]:

            text = entry[
                "text"
            ].strip()

            if not text:
                continue

            # Ignore extremely short subtitles because
            # words like "yes", "no", "okay", etc. are
            # not useful episode-identifying evidence.
            if len(
                normalize_text(text)
            ) < 8:

                continue

            entries.append(
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

                    "start":
                        entry[
                            "start"
                        ],

                    "end":
                        entry[
                            "end"
                        ],

                    "text":
                        text,
                }
            )

    print(
        f"Usable subtitle entries: "
        f"{len(entries)}"
    )

    if len(entries) < SAMPLE_COUNT:

        print(
            "ERROR: Not enough subtitle entries."
        )

        return

    # --------------------------------------------------------
    # Random sampling
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    samples = random.sample(
        entries,
        SAMPLE_COUNT
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    reciprocal_ranks = []

    failures = []

    # ========================================================
    # EVALUATION
    # ========================================================

    for sample_number, query in enumerate(
        samples,
        start=1
    ):

        true_episode = query[
            "media_id"
        ]

        query_text = query[
            "text"
        ]

        # ----------------------------------------------------
        # Score every episode
        # ----------------------------------------------------

        episode_scores = []

        for media_id, episode in data.items():

            best_score = 0.0
            best_text = ""
            best_time = None

            for entry in episode[
                "entries"
            ]:

                score = text_similarity(
                    query_text,
                    entry[
                        "text"
                    ]
                )

                if score > best_score:

                    best_score = score

                    best_text = entry[
                        "text"
                    ]

                    best_time = entry[
                        "start"
                    ]

            episode_scores.append(
                {
                    "media_id":
                        media_id,

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

        # ----------------------------------------------------
        # Rank
        # ----------------------------------------------------

        episode_scores.sort(
            key=lambda x:
            x["score"],
            reverse=True
        )

        ranked_ids = [
            result[
                "media_id"
            ]
            for result in episode_scores
        ]

        rank = (
            ranked_ids.index(
                true_episode
            )
            + 1
        )

        reciprocal_ranks.append(
            1.0 / rank
        )

        if rank == 1:

            top1_correct += 1

        if rank <= 3:

            top3_correct += 1

        if rank <= 5:

            top5_correct += 1

        if rank != 1:

            failures.append(
                {
                    "query":
                        query,

                    "rank":
                        rank,

                    "predicted":
                        episode_scores[0][
                            "media_id"
                        ],

                    "top_score":
                        episode_scores[0][
                            "score"
                        ],

                    "true_score":
                        next(
                            x["score"]
                            for x in episode_scores
                            if x[
                                "media_id"
                            ]
                            == true_episode
                        ),
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        status = (
            "OK"
            if rank == 1
            else f"rank={rank}"
        )

        print(
            f"[{sample_number:03d}/"
            f"{SAMPLE_COUNT}] "
            f"{true_episode} "
            f"-> "
            f"{episode_scores[0]['media_id']} "
            f"| {status}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("SUBTITLE BENCHMARK RESULTS")
    print("=" * 80)

    print()
    print(
        f"Samples              : "
        f"{SAMPLE_COUNT}"
    )

    print(
        f"Episodes             : "
        f"{len(data)}"
    )

    print()
    print(
        f"Top-1 accuracy       : "
        f"{100 * top1_correct / SAMPLE_COUNT:.2f}%"
    )

    print(
        f"Top-3 accuracy       : "
        f"{100 * top3_correct / SAMPLE_COUNT:.2f}%"
    )

    print(
        f"Top-5 accuracy       : "
        f"{100 * top5_correct / SAMPLE_COUNT:.2f}%"
    )

    print(
        f"MRR                  : "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}"
    )

    print(
        f"Top-1 failures       : "
        f"{len(failures)}"
    )

    # --------------------------------------------------------
    # Failure examples
    # --------------------------------------------------------

    if failures:

        print()
        print("-" * 80)
        print("FAILURE EXAMPLES")
        print("-" * 80)

        for i, failure in enumerate(
            failures[:15],
            start=1
        ):

            query = failure[
                "query"
            ]

            print()
            print(
                f"{i}. "
                f"{failure['predicted']} "
                f"(rank {failure['rank']})"
            )

            print(
                f"   True: "
                f"{query['media_id']}"
            )

            print(
                f"   Query: "
                f"{query['text']}"
            )

            print(
                f"   Top score: "
                f"{failure['top_score']:.4f}"
            )

            print(
                f"   True score: "
                f"{failure['true_score']:.4f}"
            )


if __name__ == "__main__":

    main()