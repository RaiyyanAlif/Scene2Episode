import argparse
import json
import shutil
import subprocess
from pathlib import Path


# ============================================================
# PROJECT CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Scene2Episode")

METADATA_FILE = PROJECT_ROOT / "metadata" / "media.json"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
FRAME_METADATA_DIR = PROJECT_ROOT / "metadata" / "frames"

# V1:
# Extract 1 frame every 2 seconds
INTERVAL_SECONDS = 2


# ============================================================
# LOAD MEDIA METADATA
# ============================================================

def load_media():

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def get_frame_directory(item):

    if item["type"] == "series":

        return (
            FRAMES_DIR
            / "series"
            / item["title"]
            / f"S{item['season']:02d}"
            / f"E{item['episode']:02d}"
        )

    return (
        FRAMES_DIR
        / item["type"]
        / item["title"]
    )


# ============================================================
# FRAME METADATA DIRECTORY
# ============================================================

def get_metadata_file(item):

    FRAME_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return FRAME_METADATA_DIR / f"{item['id']}.json"


# ============================================================
# FORMAT TIMESTAMP
# ============================================================

def format_timestamp(seconds):

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    milliseconds = int(
        round((seconds - int(seconds)) * 1000)
    )

    if milliseconds == 1000:
        milliseconds = 0
        secs += 1

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{milliseconds:03d}"
    )


# ============================================================
# BUILD FRAME METADATA
# ============================================================

def build_frame_metadata(item, frames):

    frame_records = []

    for index, frame_path in enumerate(frames):

        timestamp = index * INTERVAL_SECONDS

        frame_records.append(
            {
                "frame_id": frame_path.stem,

                "timestamp_seconds": timestamp,

                "timestamp_formatted":
                    format_timestamp(timestamp),

                "image_path":
                    str(frame_path),

                "media_id":
                    item["id"],

                "title":
                    item["title"],

                "type":
                    item["type"],

                "season":
                    item["season"],

                "episode":
                    item["episode"],

                "episode_title":
                    item["episode_title"]
            }
        )

    metadata = {

        "media_id": item["id"],

        "title": item["title"],

        "type": item["type"],

        "season": item["season"],

        "episode": item["episode"],

        "episode_title":
            item["episode_title"],

        "source_video":
            item["video_path"],

        "sampling_interval_seconds":
            INTERVAL_SECONDS,

        "frame_count":
            len(frame_records),

        "frames":
            frame_records
    }

    output_file = get_metadata_file(item)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    return output_file


# ============================================================
# EXTRACT FRAMES
# ============================================================

def extract_frames(item):

    video_path = Path(
        item["video_path"]
    )

    output_dir = get_frame_directory(item)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    duration = item["video_info"]["duration_seconds"]

    print()
    print("=" * 60)
    print("SCENE2EPISODE FRAME EXTRACTION")
    print("=" * 60)

    print(f"Title    : {item['title']}")
    print(f"ID       : {item['id']}")
    print(f"Duration : {duration:.2f} seconds")
    print(f"Interval : {INTERVAL_SECONDS} seconds")

    print(f"Output   : {output_dir}")

    print("=" * 60)

    # --------------------------------------------------------
    # Find FFmpeg
    # --------------------------------------------------------

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:

        print()
        print("[ERROR] FFmpeg not found in PATH.")
        print()
        print("Run:")
        print("    ffmpeg -version")
        print()

        return False

    # --------------------------------------------------------
    # Check whether frames already exist
    # --------------------------------------------------------

    existing_frames = sorted(
        output_dir.glob("*.jpg")
    )

    if existing_frames:

        print()
        print(
            f"[INFO] Found {len(existing_frames)} "
            f"existing frames."
        )

        print(
            "[INFO] Skipping extraction."
        )

        metadata_file = build_frame_metadata(
            item,
            existing_frames
        )

        print()
        print(
            f"[OK] Metadata rebuilt:"
        )

        print(metadata_file)

        return True

    # --------------------------------------------------------
    # Expected frame count
    # --------------------------------------------------------

    expected_frames = int(
        duration / INTERVAL_SECONDS
    )

    print()
    print(
        f"Expected frames: ~{expected_frames}"
    )

    # --------------------------------------------------------
    # FFmpeg output
    # --------------------------------------------------------

    output_pattern = str(
        output_dir / "%06d.jpg"
    )

    command = [

        ffmpeg_path,

        "-hide_banner",

        "-loglevel",
        "warning",

        "-i",
        str(video_path),

        "-vf",
        f"fps=1/{INTERVAL_SECONDS}",

        "-q:v",
        "2",

        output_pattern
    ]

    print()
    print("Starting FFmpeg...")
    print()

    try:

        subprocess.run(
            command,
            check=True
        )

    except subprocess.CalledProcessError:

        print()
        print(
            "[ERROR] FFmpeg extraction failed."
        )

        return False

    # --------------------------------------------------------
    # Read extracted frames
    # --------------------------------------------------------

    frames = sorted(
        output_dir.glob("*.jpg")
    )

    if not frames:

        print()
        print(
            "[ERROR] No frames were generated."
        )

        return False

    # --------------------------------------------------------
    # Build metadata
    # --------------------------------------------------------

    metadata_file = build_frame_metadata(
        item,
        frames
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Frames extracted : {len(frames)}"
    )

    print(
        f"Frame metadata   : {metadata_file}"
    )

    print(
        f"Frames directory : {output_dir}"
    )

    print("=" * 60)

    return True


# ============================================================
# FIND MEDIA BY ID
# ============================================================

def find_media_by_id(
    media,
    media_id
):

    for item in media:

        if item["id"] == media_id:

            return item

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Scene2Episode Frame Extractor"
    )

    parser.add_argument(
        "--id",
        type=str,
        help=
        "Extract frames for one media item"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=
        "Extract frames for all media items"
    )

    args = parser.parse_args()

    if not args.id and not args.all:

        parser.error(
            "Use either --id <media_id> or --all"
        )

    media = load_media()

    # --------------------------------------------------------
    # Single item
    # --------------------------------------------------------

    if args.id:

        item = find_media_by_id(
            media,
            args.id
        )

        if item is None:

            print(
                f"[ERROR] Media ID not found: "
                f"{args.id}"
            )

            print()
            print("Available IDs:")

            for m in media:

                print(
                    f"  {m['id']}"
                )

            return

        extract_frames(item)

    # --------------------------------------------------------
    # All items
    # --------------------------------------------------------

    elif args.all:

        for index, item in enumerate(
            media,
            start=1
        ):

            print()
            print(
                f"[MEDIA {index}/{len(media)}]"
            )

            extract_frames(item)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()