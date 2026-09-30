import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = PROJECT_ROOT / "indexes"

FAISS_METADATA_PATH = (
    INDEX_DIR / "scene2episode_metadata.json"
)

FAISS_INDEX_PATH = (
    INDEX_DIR / "scene2episode.index"
)

OUTPUT_PATH = (
    INDEX_DIR / "episode_prototypes.npz"
)

OUTPUT_METADATA_PATH = (
    INDEX_DIR / "episode_prototypes_metadata.json"
)


# ============================================================
# CONFIG
# ============================================================

# Number of visual prototypes per episode.
PROTOTYPES_PER_EPISODE = 8

RANDOM_STATE = 42

BATCH_SIZE = 256


# ============================================================
# HELPERS
# ============================================================

def episode_key(meta):

    return (
        meta.get("media_id"),
        meta.get("season"),
        meta.get("episode"),
    )


def load_metadata():

    with open(
        FAISS_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_all_embeddings():

    import faiss

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Dimension: {index.d}"
    )

    print(
        "Reconstructing embeddings..."
    )

    vectors = index.reconstruct_n(
        0,
        index.ntotal
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    # Normalize
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    vectors = (
        vectors / norms
    )

    return vectors


# ============================================================
# BUILD PROTOTYPES
# ============================================================

def main():

    print("=" * 80)
    print("BUILDING EPISODE VISUAL PROTOTYPES")
    print("=" * 80)

    metadata = load_metadata()

    embeddings = load_all_embeddings()

    if len(metadata) != len(embeddings):

        raise ValueError(
            "Metadata and embedding counts do not match."
        )

    print(
        f"Metadata entries: {len(metadata)}"
    )

    # --------------------------------------------------------
    # Group frame indices by episode
    # --------------------------------------------------------

    episode_indices = {}

    for index, meta in enumerate(
        metadata
    ):

        key = episode_key(
            meta
        )

        if key not in episode_indices:

            episode_indices[key] = []

        episode_indices[key].append(
            index
        )

    print(
        f"Episodes found: "
        f"{len(episode_indices)}"
    )

    # --------------------------------------------------------
    # Build prototypes
    # --------------------------------------------------------

    all_prototypes = []
    prototype_metadata = []

    for episode_number, (
        key,
        indices
    ) in enumerate(
        episode_indices.items(),
        start=1
    ):

        episode_vectors = embeddings[
            indices
        ]

        # ----------------------------------------------------
        # Number of clusters
        # ----------------------------------------------------

        n_clusters = min(
            PROTOTYPES_PER_EPISODE,
            len(episode_vectors)
        )

        # ----------------------------------------------------
        # KMeans
        # ----------------------------------------------------

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            batch_size=BATCH_SIZE,
            n_init=3,
        )

        kmeans.fit(
            episode_vectors
        )

        centers = (
            kmeans.cluster_centers_
            .astype(np.float32)
        )

        # Normalize centers
        norms = np.linalg.norm(
            centers,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1.0

        centers = (
            centers / norms
        )

        # ----------------------------------------------------
        # Save prototypes
        # ----------------------------------------------------

        for prototype_index, center in enumerate(
            centers
        ):

            all_prototypes.append(
                center
            )

            prototype_metadata.append(
                {
                    "media_id":
                        key[0],

                    "season":
                        key[1],

                    "episode":
                        key[2],

                    "prototype_index":
                        prototype_index,

                    "cluster_size":
                        int(
                            np.sum(
                                kmeans.labels_
                                == prototype_index
                            )
                        ),
                }
            )

        print(
            f"[{episode_number:02d}/"
            f"{len(episode_indices)}] "
            f"{key} "
            f"-> "
            f"{n_clusters} prototypes"
        )

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    prototypes = np.vstack(
        all_prototypes
    ).astype(
        np.float32
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
        embeddings=prototypes
    )

    with open(
        OUTPUT_METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            prototype_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EPISODE PROTOTYPES COMPLETE")
    print("=" * 80)

    print(
        f"Episodes           : "
        f"{len(episode_indices)}"
    )

    print(
        f"Prototypes/episode : "
        f"{PROTOTYPES_PER_EPISODE}"
    )

    print(
        f"Total prototypes   : "
        f"{len(prototypes)}"
    )

    print(
        f"Dimension           : "
        f"{prototypes.shape[1]}"
    )

    print(
        f"Vector file         : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Metadata file       : "
        f"{OUTPUT_METADATA_PATH}"
    )

    print()
    print("DONE")


if __name__ == "__main__":

    main()