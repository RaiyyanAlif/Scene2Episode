import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Scene2Episode")

FRAME_METADATA_DIR = PROJECT_ROOT / "metadata" / "frames"

MAX_IMAGES = 100

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cpu"


# ============================================================
# LOAD FRAME PATHS
# ============================================================

def load_frame_paths(max_images):

    image_paths = []

    metadata_files = sorted(
        FRAME_METADATA_DIR.glob("*.json")
    )

    for metadata_file in metadata_files:

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        for frame in metadata["frames"]:

            image_path = Path(
                frame["image_path"]
            )

            if image_path.exists():

                image_paths.append(
                    image_path
                )

            if len(image_paths) >= max_images:

                return image_paths

    return image_paths


# ============================================================
# GET IMAGE EMBEDDING
# ============================================================

def get_image_embedding(
    model,
    processor,
    image
):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        # Run the CLIP vision encoder
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Pooled visual representation
        pooled_output = vision_outputs.pooler_output

        # Project into CLIP embedding space
        image_features = model.visual_projection(
            pooled_output
        )

        # Normalize for cosine similarity
        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

    return image_features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SCENE2EPISODE EMBEDDING BENCHMARK")
    print("=" * 60)

    print(f"Model : {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print()

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    image_paths = load_frame_paths(
        MAX_IMAGES
    )

    print(
        f"Images selected: {len(image_paths)}"
    )

    if not image_paths:

        print(
            "[ERROR] No images found."
        )

        return

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    print()
    print("Loading CLIP model...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded.")
    print()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("Running warm-up...")

    first_image = Image.open(
        image_paths[0]
    ).convert("RGB")

    _ = get_image_embedding(
        model,
        processor,
        first_image
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("Starting benchmark...")
    print()

    embeddings = []

    start_time = time.perf_counter()

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        embedding = get_image_embedding(
            model,
            processor,
            image
        )

        embeddings.append(
            embedding.cpu()
        )

        if index % 10 == 0:

            print(
                f"Processed {index}/{len(image_paths)}"
            )

    end_time = time.perf_counter()

    # --------------------------------------------------------
    # Combine embeddings
    # --------------------------------------------------------

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    elapsed = (
        end_time - start_time
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    print(
        f"Images processed : "
        f"{len(image_paths)}"
    )

    print(
        f"Embedding shape  : "
        f"{tuple(embeddings.shape)}"
    )

    print(
        f"Time taken       : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Average/image    : "
        f"{elapsed / len(image_paths):.3f} seconds"
    )

    print(
        f"Images/minute    : "
        f"{len(image_paths) / elapsed * 60:.1f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()