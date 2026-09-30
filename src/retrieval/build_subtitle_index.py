import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "metadata"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "subtitle_index.json"
)


# ============================================================
# HELPERS
# ============================================================

def parse_timestamp(value):
    """
    Convert SRT timestamp:

        00:12:34,567

    into seconds.
    """

    value = value.strip()

    match = re.match(
        r"(\d+):(\d+):(\d+),(\d+)",
        value
    )

    if not match:

        raise ValueError(
            f"Invalid timestamp: {value}"
        )

    hours = int(
        match.group(1)
    )

    minutes = int(
        match.group(2)
    )

    seconds = int(
        match.group(3)
    )

    milliseconds = int(
        match.group(4)
    )

    return (
        hours * 3600
        + minutes * 60
        + seconds
        + milliseconds / 1000.0
    )


def clean_text(text):

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # Remove ASS/formatting-like tags
    text = re.sub(
        r"\{[^}]+\}",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def parse_srt(path):

    text = path.read_text(
        encoding="utf-8-sig",
        errors="replace"
    )

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    blocks = re.split(
        r"\n\s*\n",
        text
    )

    entries = []

    for block in blocks:

        lines = [
            line.strip()
            for line in block.split("\n")
            if line.strip()
        ]

        if len(lines) < 2:
            continue

        # Find timestamp line
        timestamp_index = None

        for i, line in enumerate(
            lines
        ):

            if "-->" in line:

                timestamp_index = i
                break

        if timestamp_index is None:
            continue

        timestamp_line = lines[
            timestamp_index
        ]

        parts = timestamp_line.split(
            "-->"
        )

        if len(parts) != 2:
            continue

        try:

            start_time = parse_timestamp(
                parts[0]
            )

            end_time = parse_timestamp(
                parts[1].split()[0]
            )

        except ValueError:

            continue

        # Everything after timestamp is subtitle text
        subtitle_lines = lines[
            timestamp_index + 1:
        ]

        subtitle_text = clean_text(
            " ".join(
                subtitle_lines
            )
        )

        if not subtitle_text:
            continue

        entries.append(
            {
                "start": start_time,
                "end": end_time,
                "text": subtitle_text,
            }
        )

    return entries


# ============================================================
# EPISODE DETECTION
# ============================================================

def detect_episode(path):

    name = path.name.lower()

    match = re.search(
        r"s(\d{2})e(\d{2})",
        name
    )

    if not match:
        return None

    season = int(
        match.group(1)
    )

    episode = int(
        match.group(2)
    )

    return season, episode


def find_subtitle_files():

    return sorted(
        RAW_DIR.rglob("*.srt")
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BUILDING SUBTITLE INDEX")
    print("=" * 80)

    subtitle_files = (
        find_subtitle_files()
    )

    print()
    print(
        f"Subtitle files found: "
        f"{len(subtitle_files)}"
    )

    subtitle_index = {}

    total_entries = 0

    for path in subtitle_files:

        episode_info = detect_episode(
            path
        )

        if episode_info is None:

            print(
                f"Skipping unknown file: "
                f"{path.name}"
            )

            continue

        season, episode = (
            episode_info
        )

        entries = parse_srt(
            path
        )

        # ----------------------------------------------------
        # Media ID
        # ----------------------------------------------------

        media_id = (
            f"lucifer_s01_e{episode:02d}"
        )

        key = (
            f"{media_id}"
        )

        subtitle_index[key] = {
            "media_id": media_id,
            "season": season,
            "episode": episode,
            "subtitle_file": str(path),
            "entries": entries,
        }

        total_entries += len(
            entries
        )

        print()
        print(
            f"{media_id}"
        )

        print(
            f"  File    : "
            f"{path.name}"
        )

        print(
            f"  Entries : "
            f"{len(entries)}"
        )

        if entries:

            print(
                f"  Duration: "
                f"{entries[-1]['end']:.1f}s"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            subtitle_index,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SUBTITLE INDEX COMPLETE")
    print("=" * 80)

    print(
        f"Episodes indexed : "
        f"{len(subtitle_index)}"
    )

    print(
        f"Subtitle entries  : "
        f"{total_entries}"
    )

    print(
        f"Output             : "
        f"{OUTPUT_PATH}"
    )

    print()
    print("DONE")


if __name__ == "__main__":

    main()