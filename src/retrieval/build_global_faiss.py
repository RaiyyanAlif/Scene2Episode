import json
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEDIA_METADATA = PROJECT_ROOT / "metadata" / "media.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
INDEX_DIR = PROJECT_ROOT / "indexes"

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GLOBAL_INDEX_FILE = (
    INDEX_DIR / "scene2episode.index"
)

GLOBAL_METADATA_FILE = (
    INDEX_DIR / "scene2episode_metadata.json"
)


# ============================================================
# LOAD MEDIA
# ============================================================

def load_media_metadata():

    with open(
        MEDIA_METADATA,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# BUILD GLOBAL INDEX
# ============================================================

def main():

    print("=" * 75)
    print("SCENE2EPISODE GLOBAL FAISS BUILDER")
    print("=" * 75)

    media_items = load_media_metadata()

    print(
        f"Media items: {len(media_items)}"
    )

    print()

    all_embeddings = []
    all_metadata = []

    total_vectors = 0

    # --------------------------------------------------------
    # LOAD EACH MEDIA ITEM
    # --------------------------------------------------------

    for media in media_items:

        media_id = media["id"]

        embedding_file = (
            EMBEDDINGS_DIR /
            f"{media_id}.npy"
        )

        metadata_file = (
            EMBEDDINGS_DIR /
            f"{media_id}_metadata.json"
        )

        print(
            f"Loading: {media_id}"
        )

        if not embedding_file.exists():

            raise FileNotFoundError(
                f"Missing embeddings:\n"
                f"{embedding_file}"
            )

        if not metadata_file.exists():

            raise FileNotFoundError(
                f"Missing metadata:\n"
                f"{metadata_file}"
            )

        # ----------------------------------------------------
        # LOAD EMBEDDINGS
        # ----------------------------------------------------

        embeddings = np.load(
            embedding_file
        ).astype(np.float32)

        # ----------------------------------------------------
        # LOAD FRAME METADATA
        # ----------------------------------------------------

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as f:

            frame_metadata = json.load(f)

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if len(embeddings) != len(frame_metadata):

            raise ValueError(
                f"Count mismatch for {media_id}: "
                f"{len(embeddings)} embeddings vs "
                f"{len(frame_metadata)} metadata"
            )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        faiss.normalize_L2(
            embeddings
        )

        all_embeddings.append(
            embeddings
        )

        # ----------------------------------------------------
        # BUILD GLOBAL METADATA
        # ----------------------------------------------------

        for frame in frame_metadata:

            record = {
                "media_id": media_id,
                "title": media.get(
                    "title"
                ),
                "type": media.get(
                    "type"
                ),
                "season": media.get(
                    "season"
                ),
                "episode": media.get(
                    "episode"
                ),
                "episode_title": media.get(
                    "episode_title"
                ),
                "frame_id": frame.get(
                    "frame_id"
                ),
                "timestamp_seconds": frame.get(
                    "timestamp_seconds"
                ),
                "timestamp_formatted": frame.get(
                    "timestamp_formatted"
                ),
                "image_path": frame.get(
                    "image_path"
                )
            }

            all_metadata.append(
                record
            )

        total_vectors += len(
            embeddings
        )

        print(
            f"  → {len(embeddings):,} vectors"
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    print()
    print("Combining embeddings...")

    combined_embeddings = np.vstack(
        all_embeddings
    ).astype(np.float32)

    print(
        f"Combined shape: "
        f"{combined_embeddings.shape}"
    )

    print(
        f"Total vectors: "
        f"{total_vectors:,}"
    )

    # --------------------------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------------------------

    if len(combined_embeddings) != len(
        all_metadata
    ):

        raise ValueError(
            "Global embedding/metadata count mismatch!"
        )

    # --------------------------------------------------------
    # BUILD FAISS
    # --------------------------------------------------------

    print()
    print("Building FAISS index...")

    dimension = (
        combined_embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        combined_embeddings
    )

    print(
        f"Vectors indexed: "
        f"{index.ntotal:,}"
    )

    # --------------------------------------------------------
    # SAVE INDEX
    # --------------------------------------------------------

    print()
    print("Saving FAISS index...")

    faiss.write_index(
        index,
        str(GLOBAL_INDEX_FILE)
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    print(
        "Saving global metadata..."
    )

    with open(
        GLOBAL_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("GLOBAL FAISS INDEX COMPLETE")
    print("=" * 75)

    print(
        f"Media items       : {len(media_items)}"
    )

    print(
        f"Total vectors     : {index.ntotal:,}"
    )

    print(
        f"Vector dimension  : {dimension}"
    )

    print(
        f"Index type        : IndexFlatIP"
    )

    print(
        f"Index saved       : "
        f"{GLOBAL_INDEX_FILE}"
    )

    print(
        f"Metadata saved    : "
        f"{GLOBAL_METADATA_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()