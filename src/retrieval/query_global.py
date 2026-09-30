import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = (
    PROJECT_ROOT /
    "indexes" /
    "scene2episode.index"
)

METADATA_FILE = (
    PROJECT_ROOT /
    "indexes" /
    "scene2episode_metadata.json"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_embedding(image_path, processor, model):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(DEVICE)

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = (
            vision_outputs.pooler_output
        )

        image_features = (
            model.visual_projection(
                pooled_output
            )
        )

        image_features = (
            image_features /
            (
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
                + 1e-12
            )
        )

    return (
        image_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def load_metadata():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Query image"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10
    )

    args = parser.parse_args()

    print("=" * 75)
    print("SCENE2EPISODE GLOBAL IMAGE SEARCH")
    print("=" * 75)

    print(
        f"Query : {args.image}"
    )

    print(
        f"Device: {DEVICE}"
    )

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

    # --------------------------------------------------------
    # LOAD FAISS
    # --------------------------------------------------------

    print("Loading global FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"Vectors available: "
        f"{index.ntotal:,}"
    )

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    print()
    print("Generating query embedding...")

    query_embedding = get_embedding(
        args.image,
        processor,
        model
    )

    faiss.normalize_L2(
        query_embedding
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        args.top_k
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("GLOBAL SEARCH RESULTS")
    print("=" * 75)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        result = metadata[idx]

        print()
        print(f"#{rank}")
        print(
            f"Similarity : {score:.4f}"
        )
        print(
            f"Title      : "
            f"{result.get('title')}"
        )
        print(
            f"Type       : "
            f"{result.get('type')}"
        )
        print(
            f"Season     : "
            f"{result.get('season')}"
        )
        print(
            f"Episode    : "
            f"{result.get('episode')}"
        )
        print(
            f"Episode title: "
            f"{result.get('episode_title')}"
        )
        print(
            f"Frame      : "
            f"{result.get('frame_id')}"
        )
        print(
            f"Timestamp  : "
            f"{result.get('timestamp_formatted')}"
        )
        print(
            f"Seconds    : "
            f"{result.get('timestamp_seconds')}"
        )
        print(
            f"Image      : "
            f"{result.get('image_path')}"
        )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()