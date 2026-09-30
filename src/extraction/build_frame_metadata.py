import json
from pathlib import Path


PROJECT_ROOT = Path(r"D:\Scene2Episode")

MEDIA_METADATA = PROJECT_ROOT / "metadata" / "media.json"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUTPUT_DIR = PROJECT_ROOT / "metadata" / "frames"

INTERVAL_SECONDS = 2


def load_media():
    with open(MEDIA_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def get_frame_directory(item):
    if item["type"] == "series":
        return (
            FRAMES_DIR
            / "series"
            / item["title"]
            / f"S{item['season']:02d}"
            / f"E{item['episode']:02d}"
        )

    return FRAMES_DIR / item["type"] / item["title"]


def build_metadata(item):

    frame_dir = get_frame_directory(item)

    if not frame_dir.exists():
        print(f"[SKIP] No frames found: {item['id']}")
        return

    frames = sorted(frame_dir.glob("*.jpg"))

    if not frames:
        print(f"[SKIP] No JPG frames found: {item['id']}")
        return

    records = []

    for index, frame_path in enumerate(frames):

        timestamp = index * INTERVAL_SECONDS

        record = {
            "frame_id": frame_path.stem,
            "timestamp_seconds": timestamp,
            "timestamp_formatted": format_timestamp(timestamp),
            "image_path": str(frame_path),
            "media_id": item["id"],
            "title": item["title"],
            "type": item["type"],
            "season": item["season"],
            "episode": item["episode"],
            "episode_title": item["episode_title"]
        }

        records.append(record)

    metadata = {
        "media_id": item["id"],
        "title": item["title"],
        "type": item["type"],
        "season": item["season"],
        "episode": item["episode"],
        "sampling_interval_seconds": INTERVAL_SECONDS,
        "frame_count": len(records),
        "frames": records
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{item['id']}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"[OK] {item['id']}")
    print(f"     Frames : {len(records)}")
    print(f"     Saved  : {output_file}")


def format_timestamp(seconds):

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():

    media = load_media()

    # For now, only build metadata for media
    # whose frames already exist.

    for item in media:
        build_metadata(item)


if __name__ == "__main__":
    main()