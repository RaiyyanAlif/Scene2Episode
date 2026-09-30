import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = PROJECT_ROOT / "indexes"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLIP EMBEDDING
# ============================================================

def get_image_embedding(image_path, processor, model):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = vision_outputs.pooler_output

        image_features = model.visual_projection(
            pooled_output
        )

        image_features = image_features / (
            image_features.norm(
                dim=-1,
                keepdim=True
            ) + 1e-12
        )

    return image_features.cpu().numpy().astype(
        np.float32
    )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(media_id):

    metadata_file = (
        EMBEDDINGS_DIR /
        f"{media_id}_metadata.json"
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{metadata_file}"
        )

    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ============================================================
# SEARCH
# ============================================================

def search(media_id, image_path, top_k):

    index_file = (
        INDEX_DIR /
        f"{media_id}.index"
    )

    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n{index_file}"
        )

    print("=" * 70)
    print("SCENE2EPISODE IMAGE SEARCH")
    print("=" * 70)

    print(f"Media : {media_id}")
    print(f"Query : {image_path}")
    print(f"Device: {DEVICE}")
    print()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading CLIP model...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded.")
    print()

    # --------------------------------------------------------
    # LOAD INDEX
    # --------------------------------------------------------

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(index_file)
    )

    print(
        f"Index loaded: {index.ntotal} vectors"
    )

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    metadata = load_metadata(media_id)

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    print("Generating query embedding...")

    query_embedding = get_image_embedding(
        image_path,
        processor,
        model
    )

    # Safety normalization
    faiss.normalize_L2(query_embedding)

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    print()
    print("=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0 or idx >= len(metadata):
            continue

        result = metadata[idx]

        print()
        print(f"#{rank}")
        print(f"Similarity : {score:.4f}")
        print(
            f"Frame      : "
            f"{result.get('frame_id', 'N/A')}"
        )
        print(
            f"Timestamp  : "
            f"{result.get('timestamp_formatted', 'N/A')}"
        )
        print(
            f"Seconds    : "
            f"{result.get('timestamp_seconds', 'N/A')}"
        )
        print(
            f"Image      : "
            f"{result.get('image_path', 'N/A')}"
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        required=True,
        help="Media ID"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Query image path"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results"
    )

    args = parser.parse_args()

    search(
        args.id,
        args.image,
        args.top_k
    )


if __name__ == "__main__":
    main()