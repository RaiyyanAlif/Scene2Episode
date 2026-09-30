import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEDIA_METADATA = PROJECT_ROOT / "metadata" / "media.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
FRAME_METADATA_DIR = PROJECT_ROOT / "metadata" / "frames"


def main():

    with open(MEDIA_METADATA, "r", encoding="utf-8") as f:
        media_items = json.load(f)

    print("=" * 75)
    print("SCENE2EPISODE EMBEDDING INTEGRITY CHECK")
    print("=" * 75)

    total_embeddings = 0
    problems = []

    for media in media_items:

        media_id = media["id"]

        embedding_file = (
            EMBEDDINGS_DIR / f"{media_id}.npy"
        )

        metadata_file = (
            EMBEDDINGS_DIR / f"{media_id}_metadata.json"
        )

        frame_file = (
            FRAME_METADATA_DIR / f"{media_id}.json"
        )

        # ----------------------------------------------------
        # FILE EXISTENCE
        # ----------------------------------------------------

        missing = []

        if not embedding_file.exists():
            missing.append("embedding")

        if not metadata_file.exists():
            missing.append("embedding metadata")

        if not frame_file.exists():
            missing.append("frame metadata")

        if missing:

            problems.append(
                f"{media_id}: missing {', '.join(missing)}"
            )

            print(
                f"[FAIL] {media_id} -> "
                f"missing {', '.join(missing)}"
            )

            continue

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        try:

            embeddings = np.load(
                embedding_file,
                mmap_mode="r"
            )

            with open(
                metadata_file,
                "r",
                encoding="utf-8"
            ) as f:
                embedding_metadata = json.load(f)

            with open(
                frame_file,
                "r",
                encoding="utf-8"
            ) as f:
                frame_metadata = json.load(f)

            if isinstance(frame_metadata, dict):
                frame_records = frame_metadata.get(
                    "frames",
                    []
                )
            else:
                frame_records = frame_metadata

        except Exception as e:

            problems.append(
                f"{media_id}: load error: {e}"
            )

            print(
                f"[FAIL] {media_id} -> {e}"
            )

            continue

        # ----------------------------------------------------
        # CHECK SHAPE
        # ----------------------------------------------------

        shape_ok = (
            len(embeddings.shape) == 2
            and embeddings.shape[1] == 512
        )

        embedding_count = embeddings.shape[0]

        metadata_count = len(
            embedding_metadata
        )

        frame_count = len(
            frame_records
        )

        counts_ok = (
            embedding_count
            == metadata_count
            == frame_count
        )

        if not shape_ok:

            problems.append(
                f"{media_id}: bad shape "
                f"{embeddings.shape}"
            )

        if not counts_ok:

            problems.append(
                f"{media_id}: count mismatch "
                f"embeddings={embedding_count}, "
                f"metadata={metadata_count}, "
                f"frames={frame_count}"
            )

        if shape_ok and counts_ok:

            print(
                f"[OK]   {media_id:<25} "
                f"{embedding_count:>6} frames "
                f"× {embeddings.shape[1]}D"
            )

            total_embeddings += embedding_count

        else:

            print(
                f"[FAIL] {media_id:<25} "
                f"embeddings={embedding_count}, "
                f"metadata={metadata_count}, "
                f"frames={frame_count}, "
                f"shape={embeddings.shape}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("INTEGRITY CHECK COMPLETE")
    print("=" * 75)

    print(
        f"Media items checked : {len(media_items)}"
    )

    print(
        f"Total embeddings    : {total_embeddings:,}"
    )

    print(
        f"Problems found      : {len(problems)}"
    )

    if problems:

        print()
        print("PROBLEMS:")

        for problem in problems:
            print(f" - {problem}")

        print()
        print("DO NOT BUILD THE GLOBAL INDEX YET.")

    else:

        print()
        print("✅ ALL EMBEDDINGS ARE VALID")
        print("✅ ALL COUNTS MATCH")
        print("✅ ALL EMBEDDINGS ARE 512-D")
        print()
        print("Ready for GLOBAL FAISS index.")

    print("=" * 75)


if __name__ == "__main__":
    main()