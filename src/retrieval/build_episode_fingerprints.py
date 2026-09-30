import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_DIR = (
    PROJECT_ROOT / "embeddings"
)

INDEX_DIR = (
    PROJECT_ROOT / "indexes"
)

OUTPUT_PATH = (
    INDEX_DIR / "episode_fingerprints.npz"
)

METADATA_PATH = (
    INDEX_DIR / "episode_fingerprints_metadata.json"
)


# ============================================================
# CONFIG
# ============================================================

# One fingerprint every 30 seconds.
FINGERPRINT_INTERVAL = 30.0


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def find_embedding_files():

    files = sorted(
        EMBEDDINGS_DIR.rglob("*.npy")
    )

    return files


def episode_key(meta):

    return (
        meta.get("media_id"),
        meta.get("season"),
        meta.get("episode"),
    )


# ============================================================
# LOAD ALL EMBEDDINGS
# ============================================================

def main():

    print("=" * 80)
    print("BUILDING EPISODE TEMPORAL FINGERPRINT INDEX")
    print("=" * 80)

    print()
    print(
        f"Embedding directory:"
        f" {EMBEDDINGS_DIR}"
    )

    embedding_files = find_embedding_files()

    print(
        f"Embedding files found: "
        f"{len(embedding_files)}"
    )

    if not embedding_files:

        print(
            "ERROR: No embedding files found."
        )

        return

    # --------------------------------------------------------
    # Load global metadata
    # --------------------------------------------------------

    global_metadata_path = (
        INDEX_DIR
        / "scene2episode_metadata.json"
    )

    metadata = load_json(
        global_metadata_path
    )

    print(
        f"Metadata entries: "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Map image/frame path -> embedding
    # --------------------------------------------------------

    embedding_lookup = {}

    print()
    print("Loading embeddings...")

    for i, embedding_file in enumerate(
        embedding_files,
        start=1
    ):

        try:

            data = np.load(
                embedding_file
            )

            # Support either:
            #   [N, 512]
            # or
            #   [512]

            if data.ndim == 1:

                data = data.reshape(
                    1,
                    -1
                )

            # The embedding filename normally corresponds
            # to the media item. We match using metadata
            # instead of assuming a naming convention.

            media_id = (
                embedding_file.stem
            )

            embedding_lookup[
                media_id
            ] = data.astype(
                np.float32
            )

        except Exception as e:

            print(
                f"Warning: failed to load "
                f"{embedding_file}: {e}"
            )

        if (
            i % 10 == 0
            or i == len(embedding_files)
        ):

            print(
                f"  Loaded "
                f"{i}/{len(embedding_files)}"
            )

    # --------------------------------------------------------
    # Group global metadata by episode
    # --------------------------------------------------------

    episode_frames = defaultdict(list)

    for item in metadata:

        key = episode_key(
            item
        )

        episode_frames[key].append(
            item
        )

    # --------------------------------------------------------
    # Sort each episode chronologically
    # --------------------------------------------------------

    for key in episode_frames:

        episode_frames[key].sort(
            key=lambda x:
            float(
                x[
                    "timestamp_seconds"
                ]
            )
        )

    # --------------------------------------------------------
    # Build fingerprints
    # --------------------------------------------------------

    fingerprint_vectors = []

    fingerprint_metadata = []

    total_episodes = len(
        episode_frames
    )

    print()
    print(
        f"Episodes found: "
        f"{total_episodes}"
    )

    processed = 0

    for key, frames in episode_frames.items():

        processed += 1

        if not frames:
            continue

        # ----------------------------------------------------
        # Determine timeline
        # ----------------------------------------------------

        max_time = max(
            float(
                frame[
                    "timestamp_seconds"
                ]
            )
            for frame in frames
        )

        target_times = np.arange(
            0.0,
            max_time + 0.1,
            FINGERPRINT_INTERVAL
        )

        # ----------------------------------------------------
        # For every 30-second point, find nearest frame
        # ----------------------------------------------------

        timestamps = np.array(
            [
                float(
                    frame[
                        "timestamp_seconds"
                    ]
                )
                for frame in frames
            ],
            dtype=np.float32
        )

        for target_time in target_times:

            nearest_index = int(
                np.argmin(
                    np.abs(
                        timestamps
                        - target_time
                    )
                )
            )

            frame_meta = frames[
                nearest_index
            ]

            media_id = frame_meta[
                "media_id"
            ]

            frame_id = frame_meta[
                "frame_id"
            ]

            # ------------------------------------------------
            # Find corresponding embedding
            # ------------------------------------------------

            embedding = None

            media_embeddings = (
                embedding_lookup.get(
                    media_id
                )
            )

            if media_embeddings is not None:

                # Embeddings are stored in chronological
                # frame order. Convert frame_id to index.
                try:

                    frame_index = (
                        int(frame_id)
                        - 1
                    )

                    if (
                        0
                        <= frame_index
                        < len(
                            media_embeddings
                        )
                    ):

                        embedding = (
                            media_embeddings[
                                frame_index
                            ]
                        )

                except Exception:
                    embedding = None

            if embedding is None:
                continue

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            norm = np.linalg.norm(
                embedding
            )

            if norm > 0:

                embedding = (
                    embedding
                    / norm
                )

            fingerprint_vectors.append(
                embedding.astype(
                    np.float32
                )
            )

            fingerprint_metadata.append(
                {
                    "media_id":
                        frame_meta[
                            "media_id"
                        ],

                    "title":
                        frame_meta.get(
                            "title"
                        ),

                    "type":
                        frame_meta.get(
                            "type"
                        ),

                    "season":
                        frame_meta.get(
                            "season"
                        ),

                    "episode":
                        frame_meta.get(
                            "episode"
                        ),

                    "episode_title":
                        frame_meta.get(
                            "episode_title"
                        ),

                    "timestamp_seconds":
                        float(
                            frame_meta[
                                "timestamp_seconds"
                            ]
                        ),

                    "timestamp_formatted":
                        frame_meta.get(
                            "timestamp_formatted"
                        ),

                    "frame_id":
                        frame_meta[
                            "frame_id"
                        ],

                    "target_time":
                        float(
                            target_time
                        ),

                    "image_path":
                        frame_meta[
                            "image_path"
                        ],
                }
            )

        if (
            processed % 10 == 0
            or processed == total_episodes
        ):

            print(
                f"  Processed "
                f"{processed}/"
                f"{total_episodes} episodes | "
                f"fingerprints: "
                f"{len(fingerprint_vectors)}"
            )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not fingerprint_vectors:

        print()
        print(
            "ERROR: No fingerprint vectors created."
        )

        return

    vectors = np.vstack(
        fingerprint_vectors
    ).astype(
        np.float32
    )

    # Normalize all vectors again.
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[
        norms == 0
    ] = 1.0

    vectors = (
        vectors
        / norms
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        embeddings=vectors
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            fingerprint_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINGERPRINT INDEX COMPLETE")
    print("=" * 80)

    print(
        f"Fingerprint vectors : "
        f"{len(vectors)}"
    )

    print(
        f"Embedding dimension : "
        f"{vectors.shape[1]}"
    )

    print(
        f"Interval            : "
        f"{FINGERPRINT_INTERVAL}s"
    )

    print(
        f"Vector file         : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Metadata file       : "
        f"{METADATA_PATH}"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()