import json
import re
import subprocess
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Scene2Episode")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "metadata"

OUTPUT_FILE = METADATA_DIR / "media.json"


VIDEO_EXTENSIONS = {
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
}


# ============================================================
# FFPROBE
# ============================================================

def get_video_info(video_path: Path):
    """
    Extract video metadata using FFprobe.
    """

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of", "json",
        str(video_path)
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)

    except subprocess.CalledProcessError as e:

        print(f"[ERROR] FFprobe failed:")
        print(f"        {video_path}")
        print(e.stderr)

        return None

    except json.JSONDecodeError:

        print(f"[ERROR] Could not parse FFprobe output:")
        print(f"        {video_path}")

        return None

    # --------------------------------------------------------
    # Stream information
    # --------------------------------------------------------

    streams = data.get("streams", [])

    if not streams:
        print(f"[WARNING] No video stream found: {video_path}")
        return None

    stream = streams[0]

    width = stream.get("width")
    height = stream.get("height")

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = None

    frame_rate = stream.get("r_frame_rate")

    if frame_rate and frame_rate != "0/0":

        try:
            numerator, denominator = map(
                int,
                frame_rate.split("/")
            )

            if denominator != 0:
                fps = numerator / denominator

        except (ValueError, ZeroDivisionError):
            fps = None

    # --------------------------------------------------------
    # Frame count
    # --------------------------------------------------------

    frame_count = stream.get("nb_frames")

    if frame_count is not None:

        try:
            frame_count = int(frame_count)

        except ValueError:
            frame_count = None

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = None

    format_info = data.get("format", {})

    if format_info.get("duration"):

        try:
            duration = float(format_info["duration"])

        except ValueError:
            duration = None

    # --------------------------------------------------------
    # Estimate frame count if unavailable
    # --------------------------------------------------------

    if frame_count is None and duration and fps:

        frame_count = round(duration * fps)

    return {
        "duration_seconds": (
            round(duration, 3)
            if duration is not None
            else None
        ),
        "fps": (
            round(fps, 3)
            if fps is not None
            else None
        ),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


# ============================================================
# EPISODE INFORMATION
# ============================================================

def extract_episode_info(filename: str):
    """
    Extract season and episode number.

    Supports:

        S01E01
        S01e01
        S02 E05
    """

    pattern = r"[Ss](\d{1,2})\s*[Ee](\d{1,2})"

    match = re.search(pattern, filename)

    if not match:
        return None, None

    season = int(match.group(1))
    episode = int(match.group(2))

    return season, episode


# ============================================================
# SUBTITLE SEARCH
# ============================================================

def find_subtitle(video_path: Path):

    subtitle = video_path.with_suffix(".srt")

    if subtitle.exists():
        return str(subtitle)

    return None


# ============================================================
# SAFE ID
# ============================================================

def make_safe_id(text: str):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


# ============================================================
# SCAN MOVIES
# ============================================================

def scan_movies(category):

    category_dir = RAW_DIR / category

    results = []

    if not category_dir.exists():
        return results

    for title_dir in sorted(category_dir.iterdir()):

        if not title_dir.is_dir():
            continue

        title = title_dir.name

        for file in sorted(title_dir.iterdir()):

            if file.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            print(f"[MOVIE] {file.name}")

            info = get_video_info(file)

            if info is None:
                print(
                    f"         [WARNING] "
                    f"Could not read video metadata."
                )

            item = {
                "id": make_safe_id(title),

                "title": title,

                "type": (
                    "animated"
                    if category == "animated"
                    else "movie"
                ),

                "season": None,

                "episode": None,

                "episode_title": None,

                "video_path": str(file),

                "subtitle_path": find_subtitle(file),

                "video_info": info,
            }

            results.append(item)

    return results


# ============================================================
# SCAN SERIES
# ============================================================

def scan_series():

    series_root = RAW_DIR / "series"

    results = []

    if not series_root.exists():
        return results

    for series_dir in sorted(series_root.iterdir()):

        if not series_dir.is_dir():
            continue

        series_name = series_dir.name

        print(f"\n[SERIES] {series_name}")

        for file in sorted(series_dir.rglob("*")):

            if file.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            season, episode = extract_episode_info(
                file.name
            )

            if season is None:

                print(
                    f"  [WARNING] "
                    f"Could not detect SxxExx: "
                    f"{file.name}"
                )

                continue

            print(
                f"  S{season:02d}E{episode:02d} "
                f"-> {file.name}"
            )

            info = get_video_info(file)

            if info is None:

                print(
                    f"         [WARNING] "
                    f"Could not read video metadata."
                )

            series_id = make_safe_id(series_name)

            item = {

                "id": (
                    f"{series_id}"
                    f"_s{season:02d}"
                    f"_e{episode:02d}"
                ),

                "title": series_name,

                "type": "series",

                "season": season,

                "episode": episode,

                "episode_title": None,

                "video_path": str(file),

                "subtitle_path": find_subtitle(file),

                "video_info": info,
            }

            results.append(item)

    return results


# ============================================================
# MAIN SCANNER
# ============================================================

def scan_all():

    print("=" * 60)

    print(
        "        SCENE2EPISODE MEDIA SCANNER"
    )

    print("=" * 60)

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_media = []

    # --------------------------------------------------------
    # Movies
    # --------------------------------------------------------

    all_media.extend(
        scan_movies("movies")
    )

    # --------------------------------------------------------
    # Animated movies
    # --------------------------------------------------------

    all_media.extend(
        scan_movies("animated")
    )

    # --------------------------------------------------------
    # Series
    # --------------------------------------------------------

    all_media.extend(
        scan_series()
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    all_media.sort(
        key=lambda x: (
            x["type"],
            x["title"],
            x["season"] or 0,
            x["episode"] or 0,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_media,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"TOTAL MEDIA ITEMS: {len(all_media)}"
    )

    print(
        f"Metadata saved to:\n{OUTPUT_FILE}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    scan_all()