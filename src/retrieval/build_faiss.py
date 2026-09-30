import argparse
from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
INDEX_DIR = PROJECT_ROOT / "indexes"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build_index(media_id):

    embedding_file = EMBEDDINGS_DIR / f"{media_id}.npy"

    if not embedding_file.exists():
        raise FileNotFoundError(
            f"Embedding file not found:\n{embedding_file}"
        )

    print("=" * 70)
    print("SCENE2EPISODE FAISS INDEX BUILDER")
    print("=" * 70)

    print(f"Media      : {media_id}")
    print(f"Embeddings : {embedding_file}")
    print()

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    embeddings = np.load(embedding_file)

    embeddings = embeddings.astype(np.float32)

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    # CLIP embeddings were already normalized,
    # but normalize again for safety.
    faiss.normalize_L2(embeddings)

    # --------------------------------------------------------
    # BUILD INDEX
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    index_file = INDEX_DIR / f"{media_id}.index"

    faiss.write_index(index, str(index_file))

    print()
    print("=" * 70)
    print("FAISS INDEX COMPLETE")
    print("=" * 70)

    print(f"Vectors indexed : {index.ntotal}")
    print(f"Dimension        : {dimension}")
    print(f"Index type       : IndexFlatIP")
    print(f"Saved to         : {index_file}")
    print("=" * 70)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        required=True,
        help="Media ID"
    )

    args = parser.parse_args()

    build_index(args.id)


if __name__ == "__main__":
    main()