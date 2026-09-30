import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEDIA_METADATA = PROJECT_ROOT / "metadata" / "media.json"
FRAME_METADATA_DIR = PROJECT_ROOT / "metadata" / "frames"
OUTPUT_DIR = PROJECT_ROOT / "embeddings"

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# METADATA
# ============================================================

def load_media_metadata():
    with open(MEDIA_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def find_media(media_id, media_items):

    for item in media_items:
        if item["id"] == media_id:
            return item

    raise ValueError(
        f"Media ID not found: {media_id}"
    )


def load_frame_metadata(media_id):

    metadata_file = (
        FRAME_METADATA_DIR /
        f"{media_id}.json"
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Frame metadata not found:\n{metadata_file}"
        )

    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "frames" in data:
        return data["frames"]

    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unexpected frame metadata format:\n"
        f"{metadata_file}"
    )


# ============================================================
# IMAGE
# ============================================================

def load_image(path):

    return Image.open(path).convert("RGB")


# ============================================================
# EMBEDDING
# ============================================================

def generate_embeddings(
    frame_records,
    processor,
    model
):

    all_embeddings = []

    total = len(frame_records)

    start_time = time.time()

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch_records = frame_records[
            start:start + BATCH_SIZE
        ]

        images = [
            load_image(record["image_path"])
            for record in batch_records
        ]

        inputs = processor(
            images=images,
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

            # Normalize CLIP embeddings
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

        embeddings = (
            image_features
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        all_embeddings.append(
            embeddings
        )

        processed = min(
            start + len(batch_records),
            total
        )

        elapsed = time.time() - start_time

        speed = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Processed {processed}/{total} "
            f"| {speed:.1f} images/sec"
        )

    return np.vstack(all_embeddings)


# ============================================================
# PROCESS ONE MEDIA ITEM
# ============================================================

def process_media(
    media_id,
    processor,
    model,
    skip_existing=True
):

    embedding_file = (
        OUTPUT_DIR /
        f"{media_id}.npy"
    )

    metadata_file = (
        OUTPUT_DIR /
        f"{media_id}_metadata.json"
    )

    # --------------------------------------------------------
    # RESUME / SKIP
    # --------------------------------------------------------

    if (
        skip_existing
        and embedding_file.exists()
        and metadata_file.exists()
    ):

        print()
        print(
            f"[SKIP] {media_id} "
            f"(embeddings already exist)"
        )

        return False

    print()
    print("=" * 70)
    print(f"PROCESSING: {media_id}")
    print("=" * 70)

    frame_records = load_frame_metadata(
        media_id
    )

    print(
        f"Frames: {len(frame_records)}"
    )

    start_time = time.time()

    embeddings = generate_embeddings(
        frame_records,
        processor,
        model
    )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        embedding_file,
        embeddings
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            frame_records,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Embedding shape : "
        f"{embeddings.shape}"
    )

    print(
        f"Time taken      : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Images/minute   : "
        f"{len(frame_records) / elapsed * 60:.1f}"
    )

    print(
        f"Saved           : "
        f"{embedding_file}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--id",
        help="Process one media item"
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Process all media items"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD MEDIA
    # --------------------------------------------------------

    media_items = load_media_metadata()

    if args.id:

        media_ids = [
            args.id
        ]

    else:

        media_ids = [
            item["id"]
            for item in media_items
        ]

    print("=" * 70)
    print("SCENE2EPISODE EMBEDDING BUILDER")
    print("=" * 70)

    print(
        f"Model : {MODEL_NAME}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Items : {len(media_ids)}"
    )

    print()

    # --------------------------------------------------------
    # LOAD MODEL ONCE
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
    # PROCESS
    # --------------------------------------------------------

    total_start = time.time()

    processed_count = 0
    skipped_count = 0

    for media_id in media_ids:

        result = process_media(
            media_id,
            processor,
            model,
            skip_existing=True
        )

        if result:
            processed_count += 1
        else:
            skipped_count += 1

    total_elapsed = (
        time.time() - total_start
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL EMBEDDING JOBS COMPLETE")
    print("=" * 70)

    print(
        f"Processed : {processed_count}"
    )

    print(
        f"Skipped   : {skipped_count}"
    )

    print(
        f"Total time: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()