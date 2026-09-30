import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.result_formatter import print_top_results


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "scene2episode.index"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "scene2episode_metadata.json"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_TOP_K_FRAMES = 50
DEFAULT_TOP_K_EPISODES = 5


# ============================================================
# CLIP EMBEDDER
# ============================================================

class CLIPEmbedder:

    def __init__(self, model_name=MODEL_NAME):

        print("Loading CLIP model...")

        self.device = torch.device("cpu")

        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            local_files_only=False,
        )

        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=False,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")
        print(
            f"CPU threads: "
            f"{torch.get_num_threads()}"
        )

    # ========================================================
    # SINGLE IMAGE
    # ========================================================

    @torch.no_grad()
    def encode_image(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        vision_outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = (
            vision_outputs.pooler_output
        )

        image_features = (
            self.model.visual_projection(
                pooled_output
            )
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        return (
            image_features
            .cpu()
            .numpy()
            .astype("float32")
        )

    # ========================================================
    # MULTIPLE IMAGES
    # Used by video inference
    # ========================================================

    @torch.no_grad()
    def encode_images(self, image_paths):

        if not image_paths:

            return np.empty(
                (0, 512),
                dtype=np.float32,
            )

        images = []

        for image_path in image_paths:

            image = Image.open(
                image_path
            ).convert("RGB")

            images.append(image)

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        vision_outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = (
            vision_outputs.pooler_output
        )

        image_features = (
            self.model.visual_projection(
                pooled_output
            )
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        return (
            image_features
            .cpu()
            .numpy()
            .astype("float32")
        )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    print("Loading metadata...")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    print(
        f"Metadata entries: "
        f"{len(metadata)}"
    )

    return metadata


# ============================================================
# EPISODE KEY
# ============================================================

def episode_key(item):

    return (
        item["media_id"],
        item.get("season"),
        item.get("episode"),
    )


# ============================================================
# GROUP RESULTS BY EPISODE
# ============================================================

def group_results_by_episode(
    frame_results,
):

    groups = {}

    for result in frame_results:

        key = episode_key(
            result["metadata"]
        )

        if key not in groups:

            groups[key] = []

        groups[key].append(
            result
        )

    return groups


# ============================================================
# EPISODE SCORING
# ============================================================

def score_episode(
    candidates,
):

    similarities = sorted(
        [
            float(
                candidate["similarity"]
            )
            for candidate in candidates
        ],
        reverse=True,
    )

    if not similarities:

        return 0.0

    best = similarities[0]

    top3 = np.mean(
        similarities[:3]
    )

    top5 = np.mean(
        similarities[:5]
    )

    return (
        0.60 * best
        + 0.25 * top3
        + 0.15 * top5
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query_embedding,
    index,
    metadata,
    top_k_frames,
    top_k_episodes,
):

    scores, indices = index.search(
        query_embedding,
        top_k_frames,
    )

    frame_results = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):

        if index_id < 0:

            continue

        item = metadata[index_id]

        frame_results.append(
            {
                "similarity": float(score),
                "metadata": item,
            }
        )

    # --------------------------------------------------------
    # Group by episode
    # --------------------------------------------------------

    episode_groups = (
        group_results_by_episode(
            frame_results
        )
    )

    episode_results = []

    for key, candidates in (
        episode_groups.items()
    ):

        candidates.sort(
            key=lambda x: x["similarity"],
            reverse=True,
        )

        best_candidate = candidates[0]

        score = score_episode(
            candidates
        )

        episode_results.append(
            {
                "episode_key": key,
                "score": float(score),
                "best_similarity": float(
                    best_candidate[
                        "similarity"
                    ]
                ),
                "best_candidate": (
                    best_candidate
                ),
                "evidence_count": len(
                    candidates
                ),
            }
        )

    # --------------------------------------------------------
    # Sort highest score first
    # --------------------------------------------------------

    episode_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return episode_results[
        :top_k_episodes
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Scene2Episode image inference"
        )
    )

    parser.add_argument(
        "image",
        help="Path to query image",
    )

    parser.add_argument(
        "--top-k-frames",
        type=int,
        default=DEFAULT_TOP_K_FRAMES,
        help=(
            "Number of FAISS frame "
            "candidates"
        ),
    )

    parser.add_argument(
        "--top-k-episodes",
        type=int,
        default=DEFAULT_TOP_K_EPISODES,
        help=(
            "Number of episode results "
            "to show"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    image_path = Path(
        args.image
    ).resolve()

    if not image_path.exists():

        print(
            f"ERROR: Image not found:\n"
            f"{image_path}"
        )

        return

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Index vectors: "
        f"{index.ntotal}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    if index.ntotal != len(metadata):

        raise RuntimeError(
            "FAISS index and metadata "
            "sizes do not match."
        )

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    embedder = CLIPEmbedder()

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    print()
    print(
        "Encoding query image..."
    )

    query_embedding = (
        embedder.encode_image(
            image_path
        )
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    print(
        "Searching Scene2Episode..."
    )

    results = retrieve(
        query_embedding=query_embedding,
        index=index,
        metadata=metadata,
        top_k_frames=args.top_k_frames,
        top_k_episodes=args.top_k_episodes,
    )

    # --------------------------------------------------------
    # Display formatted result
    # --------------------------------------------------------

    print()

    print(
        f"Query image: "
        f"{image_path}"
    )

    print_top_results(
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
