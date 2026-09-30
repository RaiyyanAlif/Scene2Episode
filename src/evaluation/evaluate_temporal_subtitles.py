import json
import random
import sys
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBTITLE_INDEX = (
    PROJECT_ROOT
    / "metadata"
    / "subtitle_index.json"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.retrieval.temporal_subtitle_matcher import (
    TemporalSubtitleMatcher,
    text_similarity,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
SAMPLE_COUNT = 100

TOP_K_VALUES = [1, 3, 5]


# ============================================================
# HELPERS
# ============================================================

def episode_key(media_id):

    return media_id


def reciprocal_rank(
    results,
    correct_media_id,
):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if (
            result["media_id"]
            == correct_media_id
        ):

            return 1.0 / rank

    return 0.0


# ============================================================
# LOAD SUBTITLES
# ============================================================

def load_samples():

    with open(
        SUBTITLE_INDEX,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    samples = []

    for media_id, episode in data.items():

        for entry_index, entry in enumerate(
            episode["entries"]
        ):

            text = entry["text"].strip()

            if not text:
                continue

            samples.append(
                {
                    "media_id":
                        media_id,

                    "entry_index":
                        entry_index,

                    "start":
                        float(
                            entry["start"]
                        ),

                    "end":
                        float(
                            entry["end"]
                        ),

                    "text":
                        text,
                }
            )

    return data, samples


# ============================================================
# BUILD QUERY WINDOW
# ============================================================

def build_query_window(
    data,
    sample,
    window_seconds=15.0,
):

    episode = data[
        sample["media_id"]
    ]

    center = sample["start"]

    start_time = (
        center
        - window_seconds
    )

    end_time = (
        center
        + window_seconds
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

            text = entry["text"].strip()

            if text:

                texts.append(text)

    return " ".join(texts)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 72)
    print("FAIR TEMPORAL SUBTITLE RETRIEVAL BENCHMARK")
    print("=" * 72)

    data, all_samples = load_samples()

    print()
    print(
        f"Total usable subtitle entries: "
        f"{len(all_samples)}"
    )

    random.seed(SEED)

    sample_count = min(
        SAMPLE_COUNT,
        len(all_samples)
    )

    samples = random.sample(
        all_samples,
        sample_count
    )

    print(
        f"Evaluation samples: "
        f"{sample_count}"
    )

    print(
        f"Random seed: {SEED}"
    )

    matcher = TemporalSubtitleMatcher()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    correct_at = {
        k: 0
        for k in TOP_K_VALUES
    }

    rr_total = 0.0

    failures = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for i, sample in enumerate(
        samples,
        start=1
    ):

        query_media = sample[
            "media_id"
        ]

        query_text = sample[
            "text"
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        # Build the query from the temporal context.
        # ----------------------------------------------------

        query_window = build_query_window(
            data,
            sample
        )

        # ----------------------------------------------------
        # Fair evaluation:
        #
        # We search each episode independently, but for the
        # query episode we REMOVE the exact subtitle entry
        # and nearby temporal window.
        # ----------------------------------------------------

        results = []

        for media_id, episode in (
            data.items()
        ):

            best_score = 0.0
            best_timestamp = None
            best_text = ""

            for entry_index, entry in enumerate(
                episode["entries"]
            ):

                # --------------------------------------------
                # Prevent query leakage.
                # --------------------------------------------

                if (
                    media_id
                    == query_media
                ):

                    distance = abs(
                        float(
                            entry["start"]
                        )
                        -
                        float(
                            sample["start"]
                        )
                    )

                    if distance <= 15.0:

                        continue

                # --------------------------------------------
                # Candidate temporal window
                # --------------------------------------------

                candidate_start = (
                    float(entry["start"])
                )

                candidate_window = (
                    matcher.window_text(
                        media_id,
                        candidate_start
                    )
                )

                if not candidate_window:
                    continue
                score = text_similarity(
                    query_window,
                    candidate_window
                )

                if score > best_score:

                    best_score = score

                    best_timestamp = (
                        candidate_start
                    )

                    best_text = (
                        candidate_window
                    )

            results.append(
                {
                    "media_id":
                        media_id,

                    "score":
                        best_score,

                    "timestamp":
                        best_timestamp,

                    "matched_text":
                        best_text,
                }
            )

        results.sort(
            key=lambda x:
            x["score"],
            reverse=True
        )

        rank = None

        for r, result in enumerate(
            results,
            start=1
        ):

            if (
                result["media_id"]
                == query_media
            ):

                rank = r
                break

        if rank is not None:

            rr_total += (
                1.0 / rank
            )

        for k in TOP_K_VALUES:

            if (
                rank is not None
                and rank <= k
            ):

                correct_at[k] += 1

        # ----------------------------------------------------
        # Record failures
        # ----------------------------------------------------

        if (
            rank is None
            or rank > 1
        ):

            top = results[0]

            failures.append(
                {
                    "query":
                        query_media,

                    "time":
                        sample["start"],

                    "text":
                        query_text,

                    "rank":
                        rank,

                    "predicted":
                        top[
                            "media_id"
                        ],

                    "score":
                        top[
                            "score"
                        ],
                }
            )

        if i % 10 == 0:

            print(
                f"Processed "
                f"{i}/{sample_count}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    for k in TOP_K_VALUES:

        accuracy = (
            correct_at[k]
            / sample_count
            * 100
        )

        print(
            f"Top-{k}: "
            f"{accuracy:.2f}%"
        )

    mrr = (
        rr_total
        / sample_count
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    # ========================================================
    # FAILURES
    # ========================================================

    print()
    print(
        f"Top-1 failures: "
        f"{len(failures)}"
    )

    print()

    for i, failure in enumerate(
        failures,
        start=1
    ):

        print(
            f"{i}. "
            f"{failure['query']} "
            f"@ "
            f"{failure['time']:.1f}s"
        )

        print(
            f"   predicted: "
            f"{failure['predicted']}"
        )

        print(
            f"   correct rank: "
            f"{failure['rank']}"
        )

        print(
            f"   score: "
            f"{failure['score']:.4f}"
        )

        print(
            f"   text: "
            f"{failure['text']}"
        )

        print()

    print("=" * 72)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()