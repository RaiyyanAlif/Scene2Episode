import os
import json
import re

import cv2
import pytesseract


# ============================================================
# CONFIG
# ============================================================

METADATA_PATH = "indexes/scene2episode_metadata.json"
OCR_CACHE_PATH = "metadata/ocr_cache.json"

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# ============================================================
# LOAD
# ============================================================

def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_cache():

    if not os.path.exists(
        OCR_CACHE_PATH
    ):
        return {}

    with open(
        OCR_CACHE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):

    text = text.replace(
        "\n",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^A-Za-z0-9 .,'!?():;\-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def episode_name(item):

    if item.get("season") is None:
        return item["title"]

    return (
        f"{item['title']} "
        f"S{item['season']:02d} "
        f"E{item['episode']:02d}"
    )


def normalize_tokens(text):

    return set(
        re.findall(
            r"[a-z0-9]{2,}",
            text.lower(),
        )
    )


def similarity(a, b):

    a_tokens = normalize_tokens(a)
    b_tokens = normalize_tokens(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return (
        len(a_tokens & b_tokens)
        / len(a_tokens | b_tokens)
    )


# ============================================================
# FIND FRAMES
# ============================================================

def find_frames(
    metadata,
    media_id,
    season,
    episode,
    center_time,
    radius=6,
):

    frames = []

    for item in metadata:

        if item["media_id"] != media_id:
            continue

        if item.get("season") != season:
            continue

        if item.get("episode") != episode:
            continue

        diff = abs(
            item["timestamp_seconds"]
            - center_time
        )

        if diff <= radius:

            frames.append(
                item
            )

    frames.sort(
        key=lambda x:
        x["timestamp_seconds"]
    )

    return frames


# ============================================================
# MAIN
# ============================================================

def main():

    metadata = load_metadata()
    cache = load_cache()

    print("=" * 100)
    print("SCENE2EPISODE OCR DIAGNOSTIC")
    print("=" * 100)

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    query_frames = find_frames(
        metadata,
        "stranger_things_s03_e01",
        3,
        1,
        2990,
        radius=6,
    )

    # --------------------------------------------------------
    # COMPETING EPISODES
    # --------------------------------------------------------

    competitors = {
        "S3E1": (
            "stranger_things_s03_e01",
            3,
            1,
        ),
        "S3E2": (
            "stranger_things_s03_e02",
            3,
            2,
        ),
        "S3E3": (
            "stranger_things_s03_e03",
            3,
            3,
        ),
        "S3E4": (
            "stranger_things_s03_e04",
            3,
            4,
        ),
        "S3E5": (
            "stranger_things_s03_e05",
            3,
            5,
        ),
    }

    candidate_frames = {}

    for name, (
        media_id,
        season,
        episode,
    ) in competitors.items():

        candidate_frames[name] = find_frames(
            metadata,
            media_id,
            season,
            episode,
            2990,
            radius=10,
        )

    # --------------------------------------------------------
    # PRINT QUERY OCR
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("QUERY: STRANGER THINGS S3E1 @ 00:49:50")
    print("=" * 100)

    query_texts = []

    for item in query_frames:

        path = item["image_path"]

        text = cache.get(
            path,
            "",
        )

        query_texts.append(text)

        print()
        print(
            f"[{item['timestamp_formatted']}]"
        )
        print(
            f"Frame: {item['frame_id']}"
        )

        if text:
            print(
                f"OCR: {text}"
            )
        else:
            print(
                "OCR: [NO TEXT]"
            )

    # --------------------------------------------------------
    # COMPETITORS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("COMPETING EPISODES")
    print("=" * 100)

    for name, frames in candidate_frames.items():

        print()
        print(
            "=" * 100
        )

        print(
            f"{name}"
        )

        print(
            "=" * 100
        )

        episode_texts = []

        for item in frames:

            path = item[
                "image_path"
            ]

            text = cache.get(
                path,
                "",
            )

            if not text:
                continue

            episode_texts.append(
                text
            )

            print()
            print(
                f"[{item['timestamp_formatted']}]"
            )

            print(
                f"Frame: "
                f"{item['frame_id']}"
            )

            print(
                f"OCR: "
                f"{text}"
            )

        # ----------------------------------------------------
        # SIMILARITY
        # ----------------------------------------------------

        print()

        if not episode_texts:

            print(
                "No OCR text available."
            )

            continue

        scores = []

        for query_text in query_texts:

            if not query_text:
                continue

            for candidate_text in episode_texts:

                score = similarity(
                    query_text,
                    candidate_text,
                )

                scores.append(
                    score
                )

        if scores:

            scores.sort(
                reverse=True
            )

            print(
                "Best OCR similarities:"
            )

            for score in scores[:10]:

                print(
                    f"  {score:.4f}"
                )


if __name__ == "__main__":
    main()