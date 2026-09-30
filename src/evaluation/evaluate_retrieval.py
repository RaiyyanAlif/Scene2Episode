import argparse
import json
import random
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "indexes" / "scene2episode.index"
METADATA_PATH = PROJECT_ROOT / "indexes" / "scene2episode_metadata.json"

MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================================
# IMAGE
# ============================================================

def load_image(image_path):
    return Image.open(image_path).convert("RGB")


# ============================================================
# CLIP
# ============================================================

def load_clip():

    print("Loading CLIP model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return processor, model, device


def get_embedding(
    image,
    processor,
    model,
    device
):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = vision_outputs.pooler_output

        image_features = model.visual_projection(
            pooled_output
        )

        # Normalize for cosine similarity / inner product
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

    return image_features.cpu().numpy().astype(
        "float32"
    )


# ============================================================
# METADATA HELPERS
# ============================================================

def episode_key(item):
    """
    Identifies the exact episode/media unit.

    For movies/animated movies:
        season = None
        episode = None

    For series:
        title + season + episode
    """

    return (
        item.get("title"),
        item.get("season"),
        item.get("episode")
    )


def media_key(item):
    """
    Global metadata uses 'media_id'.
    """

    return item["media_id"]


def timestamp(item):
    """
    Global metadata uses 'timestamp_seconds'.
    """

    return float(
        item["timestamp_seconds"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Fair evaluation of Scene2Episode retrieval"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of random query frames"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieval results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("=" * 75)
    print("SCENE2EPISODE FAIR RETRIEVAL EVALUATION")
    print("=" * 75)

    print(f"Samples     : {args.samples}")
    print(f"Top-K       : {args.top_k}")
    print(f"Random seed : {args.seed}")
    print()

    # ========================================================
    # LOAD FAISS
    # ========================================================

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Index vectors: {index.ntotal}"
    )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    print("Loading metadata...")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    print(
        f"Metadata entries: {len(metadata)}"
    )

    # ========================================================
    # VERIFY INDEX / METADATA
    # ========================================================

    if index.ntotal != len(metadata):

        raise RuntimeError(
            "FAISS index and metadata size mismatch!\n"
            f"FAISS vectors : {index.ntotal}\n"
            f"Metadata      : {len(metadata)}"
        )

    # ========================================================
    # LOAD CLIP
    # ========================================================

    processor, model, device = load_clip()

    print(
        f"Device      : {device}"
    )

    print(
        "Model loaded."
    )

    print()

    # ========================================================
    # RANDOM SAMPLE
    # ========================================================

    random.seed(
        args.seed
    )

    sample_count = min(
        args.samples,
        len(metadata)
    )

    sample_indices = random.sample(
        range(len(metadata)),
        sample_count
    )

    # ========================================================
    # METRICS
    # ========================================================

    top1_episode_correct = 0
    top5_episode_correct = 0

    top1_media_correct = 0
    top5_media_correct = 0

    timestamp_errors = []

    failures = []

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    for count, query_index in enumerate(
        sample_indices,
        start=1
    ):

        query_meta = metadata[
            query_index
        ]

        # ----------------------------------------------------
        # Query image
        # ----------------------------------------------------

        image_path = Path(
            query_meta["image_path"]
        )

        if not image_path.exists():

            print(
                f"[WARNING] Image not found:"
                f" {image_path}"
            )

            continue

        image = load_image(
            image_path
        )

        # ----------------------------------------------------
        # Generate embedding
        # ----------------------------------------------------

        query_embedding = get_embedding(
            image,
            processor,
            model,
            device
        )

        # ----------------------------------------------------
        # Search FAISS
        #
        # Ask for one extra result because the exact query
        # frame is already present in the database.
        # ----------------------------------------------------

        search_k = args.top_k + 1

        similarities, indices = index.search(
            query_embedding,
            search_k
        )

        results = []

        for similarity, idx in zip(
            similarities[0],
            indices[0]
        ):

            if idx < 0:
                continue

            # ------------------------------------------------
            # EXCLUDE EXACT QUERY FRAME
            # ------------------------------------------------

            if idx == query_index:
                continue

            result_meta = metadata[
                int(idx)
            ]

            results.append(
                {
                    "index": int(idx),
                    "similarity": float(
                        similarity
                    ),
                    "metadata": result_meta
                }
            )

            if len(results) >= args.top_k:
                break

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        true_episode = episode_key(
            query_meta
        )

        true_media = media_key(
            query_meta
        )

        true_timestamp = timestamp(
            query_meta
        )

        # ====================================================
        # TOP-1
        # ====================================================

        if results:

            top1_result = results[0]

            top1_meta = top1_result[
                "metadata"
            ]

            predicted_episode = episode_key(
                top1_meta
            )

            predicted_media = media_key(
                top1_meta
            )

            # Episode accuracy
            if predicted_episode == true_episode:

                top1_episode_correct += 1

            else:

                failures.append(
                    {
                        "query": query_meta,
                        "prediction": top1_meta,
                        "similarity":
                            top1_result["similarity"]
                    }
                )

            # Media accuracy
            if predicted_media == true_media:

                top1_media_correct += 1

        # ====================================================
        # TOP-5
        # ====================================================

        top5 = results[:5]

        top5_episodes = {
            episode_key(
                result["metadata"]
            )
            for result in top5
        }

        top5_media = {
            media_key(
                result["metadata"]
            )
            for result in top5
        }

        if true_episode in top5_episodes:

            top5_episode_correct += 1

        if true_media in top5_media:

            top5_media_correct += 1

        # ====================================================
        # TIMESTAMP ERROR
        #
        # Find the strongest retrieved frame from the
        # correct episode.
        # ====================================================

        correct_episode_results = [
            result
            for result in results
            if episode_key(
                result["metadata"]
            ) == true_episode
        ]

        if correct_episode_results:

            best_correct = max(
                correct_episode_results,
                key=lambda x: x["similarity"]
            )

            predicted_timestamp = timestamp(
                best_correct["metadata"]
            )

            error = abs(
                predicted_timestamp -
                true_timestamp
            )

            timestamp_errors.append(
                error
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if count % 10 == 0:

            print(
                f"Evaluated "
                f"{count}/{sample_count}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print("=" * 75)
    print("FAIR EVALUATION RESULTS")
    print("=" * 75)

    print(
        f"Samples                    : "
        f"{sample_count}"
    )

    print(
        f"Top-1 episode accuracy    : "
        f"{top1_episode_correct / sample_count * 100:.2f}%"
    )

    print(
        f"Top-5 episode accuracy    : "
        f"{top5_episode_correct / sample_count * 100:.2f}%"
    )

    print(
        f"Top-1 media accuracy      : "
        f"{top1_media_correct / sample_count * 100:.2f}%"
    )

    print(
        f"Top-5 media accuracy      : "
        f"{top5_media_correct / sample_count * 100:.2f}%"
    )

    # ========================================================
    # TIMESTAMP METRICS
    # ========================================================

    if timestamp_errors:

        print(
            f"Mean timestamp error      : "
            f"{np.mean(timestamp_errors):.2f} sec"
        )

        print(
            f"Median timestamp error    : "
            f"{np.median(timestamp_errors):.2f} sec"
        )

    else:

        print(
            "Mean timestamp error      : N/A"
        )

        print(
            "Median timestamp error    : N/A"
        )

    print()

    print(
        f"Top-1 failures             : "
        f"{len(failures)}"
    )

    # ========================================================
    # FAILURE DETAILS
    # ========================================================

    if failures:

        print()

        print("-" * 75)
        print("TOP-1 FAILURES")
        print("-" * 75)

        for i, failure in enumerate(
            failures[:10],
            start=1
        ):

            query = failure["query"]
            prediction = failure["prediction"]

            similarity = failure[
                "similarity"
            ]

            print()
            print(
                f"Failure #{i}"
            )

            print(
                f"Query : "
                f"{query.get('title')} "
                f"S{query.get('season')} "
                f"E{query.get('episode')} "
                f"@ "
                f"{timestamp(query):.2f}s "
                f"({query.get('timestamp_formatted')})"
            )

            print(
                f"Got   : "
                f"{prediction.get('title')} "
                f"S{prediction.get('season')} "
                f"E{prediction.get('episode')} "
                f"@ "
                f"{timestamp(prediction):.2f}s "
                f"({prediction.get('timestamp_formatted')})"
            )

            print(
                f"Similarity: "
                f"{similarity:.4f}"
            )

    print()

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()