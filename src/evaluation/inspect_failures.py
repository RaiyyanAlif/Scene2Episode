import argparse
import json
import random
from pathlib import Path

import faiss
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "indexes" / "scene2episode.index"
METADATA_PATH = PROJECT_ROOT / "indexes" / "scene2episode_metadata.json"

OUTPUT_DIR = PROJECT_ROOT / "evaluation_reports" / "failures"

MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================================
# CLIP
# ============================================================

def load_clip():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CLIP...")

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

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

    return image_features.cpu().numpy().astype(
        "float32"
    )


# ============================================================
# HELPERS
# ============================================================

def episode_key(item):

    return (
        item.get("title"),
        item.get("season"),
        item.get("episode")
    )


def timestamp(item):

    return float(
        item["timestamp_seconds"]
    )


def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_font(size=28):

    try:

        return ImageFont.truetype(
            "arial.ttf",
            size
        )

    except:

        return ImageFont.load_default()


# ============================================================
# CREATE COMPARISON IMAGE
# ============================================================

def create_comparison(
    query_meta,
    result_meta,
    similarity,
    output_path
):

    query_image = Image.open(
        query_meta["image_path"]
    ).convert("RGB")

    result_image = Image.open(
        result_meta["image_path"]
    ).convert("RGB")

    # Resize both images to same height
    target_height = 500

    query_ratio = (
        query_image.width /
        query_image.height
    )

    result_ratio = (
        result_image.width /
        result_image.height
    )

    query_width = int(
        target_height * query_ratio
    )

    result_width = int(
        target_height * result_ratio
    )

    query_image = query_image.resize(
        (query_width, target_height)
    )

    result_image = result_image.resize(
        (result_width, target_height)
    )

    # Header
    header_height = 180

    total_width = (
        query_width +
        result_width
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            target_height + header_height
        ),
        "white"
    )

    # Paste images
    canvas.paste(
        query_image,
        (0, header_height)
    )

    canvas.paste(
        result_image,
        (query_width, header_height)
    )

    draw = ImageDraw.Draw(canvas)

    font = load_font(24)
    small_font = load_font(20)

    # --------------------------------------------------------
    # Query information
    # --------------------------------------------------------

    query_title = (
        f"GROUND TRUTH\n"
        f"{query_meta.get('title')} "
        f"S{query_meta.get('season')} "
        f"E{query_meta.get('episode')}"
    )

    query_time = (
        f"Time: "
        f"{format_time(timestamp(query_meta))}"
    )

    draw.multiline_text(
        (15, 15),
        query_title,
        fill="black",
        font=font,
        spacing=5
    )

    draw.text(
        (15, 125),
        query_time,
        fill="black",
        font=small_font
    )

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    result_title = (
        f"PREDICTION\n"
        f"{result_meta.get('title')} "
        f"S{result_meta.get('season')} "
        f"E{result_meta.get('episode')}"
    )

    result_time = (
        f"Time: "
        f"{format_time(timestamp(result_meta))}"
    )

    draw.multiline_text(
        (query_width + 15, 15),
        result_title,
        fill="black",
        font=font,
        spacing=5
    )

    draw.text(
        (query_width + 15, 125),
        result_time,
        fill="black",
        font=small_font
    )

    # Similarity
    draw.text(
        (
            query_width +
            result_width -
            230,
            125
        ),
        f"Similarity: {similarity:.4f}",
        fill="black",
        font=small_font
    )

    canvas.save(
        output_path,
        quality=95
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=50
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    print("=" * 75)
    print("SCENE2EPISODE FAILURE INSPECTION")
    print("=" * 75)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    processor, model, device = load_clip()

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Select same samples as evaluation
    # --------------------------------------------------------

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

    failures = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for count, query_index in enumerate(
        sample_indices,
        start=1
    ):

        query_meta = metadata[
            query_index
        ]

        image_path = Path(
            query_meta["image_path"]
        )

        if not image_path.exists():

            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        embedding = get_embedding(
            image,
            processor,
            model,
            device
        )

        # Search extra result to remove self-match
        similarities, indices = index.search(
            embedding,
            args.top_k + 1
        )

        results = []

        for similarity, idx in zip(
            similarities[0],
            indices[0]
        ):

            if idx < 0:
                continue

            if idx == query_index:
                continue

            results.append(
                (
                    float(similarity),
                    int(idx)
                )
            )

            if len(results) >= args.top_k:
                break

        if not results:
            continue

        similarity, prediction_index = results[0]

        prediction_meta = metadata[
            prediction_index
        ]

        # ----------------------------------------------------
        # Check whether top-1 is wrong
        # ----------------------------------------------------

        if episode_key(
            prediction_meta
        ) != episode_key(
            query_meta
        ):

            failures.append(
                {
                    "query_index": query_index,
                    "prediction_index": prediction_index,
                    "similarity": similarity,
                    "query": query_meta,
                    "prediction": prediction_meta
                }
            )

        if count % 10 == 0:

            print(
                f"Processed {count}/{sample_count}"
            )

    # --------------------------------------------------------
    # Generate reports
    # --------------------------------------------------------

    print()
    print(
        f"Failures found: {len(failures)}"
    )

    print(
        f"Saving reports to:"
    )

    print(
        OUTPUT_DIR
    )

    report = []

    for i, failure in enumerate(
        failures,
        start=1
    ):

        query = failure["query"]
        prediction = failure["prediction"]
        similarity = failure["similarity"]

        filename = (
            f"failure_{i:02d}.jpg"
        )

        output_path = (
            OUTPUT_DIR /
            filename
        )

        create_comparison(
            query,
            prediction,
            similarity,
            output_path
        )

        report.append(
            {
                "failure": i,
                "image": str(output_path),
                "ground_truth": {
                    "title": query.get("title"),
                    "season": query.get("season"),
                    "episode": query.get("episode"),
                    "timestamp": query.get(
                        "timestamp_formatted"
                    ),
                    "image_path": query.get(
                        "image_path"
                    )
                },
                "prediction": {
                    "title": prediction.get("title"),
                    "season": prediction.get("season"),
                    "episode": prediction.get("episode"),
                    "timestamp": prediction.get(
                        "timestamp_formatted"
                    ),
                    "similarity": similarity,
                    "image_path": prediction.get(
                        "image_path"
                    )
                }
            }
        )

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR /
        "failure_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Generated comparison images:"
    )

    for item in report:

        print(
            f"  {item['image']}"
        )

    print()
    print(
        f"JSON report:"
    )

    print(
        report_path
    )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()