import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = PROJECT_ROOT / "indexes" / "scene2episode.index"
METADATA_PATH = PROJECT_ROOT / "indexes" / "scene2episode_metadata.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_NUM_FRAMES = 5
DEFAULT_TOP_K_FRAMES = 20
DEFAULT_TOP_K_EPISODES = 5


# ============================================================
# CLIP
# ============================================================

class CLIPEmbedder:

    def __init__(self):

        print("Loading CLIP model...")

        self.device = torch.device("cpu")

        self.processor = CLIPProcessor.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        self.model = CLIPModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")
        print(f"CPU threads: {torch.get_num_threads()}")

    @torch.no_grad()
    def encode_images(self, image_paths):

        images = []

        for path in image_paths:

            image = Image.open(path).convert("RGB")
            images.append(image)

        if not images:

            return np.empty(
                (0, 512),
                dtype=np.float32,
            )

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

        pooled_output = vision_outputs.pooler_output

        image_features = self.model.visual_projection(
            pooled_output
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        return image_features.cpu().numpy().astype(
            "float32"
        )


# ============================================================
# METADATA
# ============================================================

def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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
# FORMAT TIME
# ============================================================

def format_timestamp(seconds):

    seconds = float(seconds)

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:05.2f}"
    )


# ============================================================
# EXTRACT VIDEO FRAMES
# ============================================================

def extract_clip_frames(
    video_path,
    output_dir,
    num_frames,
):

    print()
    print("Extracting video frames...")

    # 1 second duration estimate.
    # FFmpeg will overwrite the temporary files.
    #
    # We use evenly spaced timestamps from the first
    # 3 seconds. If the video is shorter, FFmpeg simply
    # returns the available frames.

    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:

        duration_output = subprocess.check_output(
            duration_cmd,
            text=True,
        ).strip()

        duration = float(duration_output)

    except Exception as exc:

        raise RuntimeError(
            f"Could not determine video duration: {exc}"
        )

    if duration <= 0:

        raise RuntimeError(
            "Video duration is zero."
        )

    # Use the first 3 seconds, or the whole clip
    # if it is shorter than 3 seconds.

    usable_duration = min(
        duration,
        3.0,
    )

    if num_frames == 1:

        timestamps = [
            usable_duration / 2
        ]

    else:

        timestamps = np.linspace(
            0,
            max(
                0.0,
                usable_duration - 0.05,
            ),
            num_frames,
        )

    image_paths = []

    for i, timestamp in enumerate(
        timestamps
    ):

        output_path = (
            output_dir
            / f"frame_{i:02d}.jpg"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        if output_path.exists():

            image_paths.append(
                (
                    output_path,
                    float(timestamp),
                )
            )

    print(
        f"Extracted {len(image_paths)} frames"
    )

    return image_paths


# ============================================================
# SEARCH ONE FRAME
# ============================================================

def search_frame(
    embedding,
    index,
    metadata,
    top_k,
):

    scores, indices = index.search(
        embedding.reshape(1, -1),
        top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        results.append(
            {
                "similarity": float(score),
                "metadata": metadata[idx],
            }
        )

    return results


# ============================================================
# AGGREGATE VIDEO EVIDENCE
# ============================================================

def aggregate_results(
    frame_results,
    top_k_episodes,
):

    episodes = defaultdict(list)

    for frame_id, results in enumerate(
        frame_results
    ):

        for result in results:

            key = episode_key(
                result["metadata"]
            )

            episodes[key].append(
                {
                    "frame_id": frame_id,
                    "similarity": result[
                        "similarity"
                    ],
                    "metadata": result[
                        "metadata"
                    ],
                }
            )

    episode_results = []

    for key, candidates in episodes.items():

        candidates.sort(
            key=lambda x: x["similarity"],
            reverse=True,
        )

        similarities = [
            item["similarity"]
            for item in candidates
        ]

        # Strongest evidence.
        best = similarities[0]

        # Average of strongest 3 matches.
        top3 = np.mean(
            similarities[:3]
        )

        # How many different query frames
        # found this episode?
        covered_frames = len(
            set(
                item["frame_id"]
                for item in candidates
            )
        )

        coverage = (
            covered_frames
            / len(frame_results)
        )

        # Final video-level score.
        score = (
            0.50 * best
            + 0.30 * top3
            + 0.20 * coverage
        )

        # Timestamp:
        # use the strongest candidate's
        # timestamp as the initial estimate.
        best_candidate = candidates[0]

        episode_results.append(
            {
                "episode_key": key,
                "score": float(score),
                "best_similarity": float(best),
                "top3_mean": float(top3),
                "coverage": float(coverage),
                "best_candidate": best_candidate,
                "evidence_count": len(candidates),
            }
        )

    episode_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return episode_results[
        :top_k_episodes
    ]


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    video_path,
    clip_frames,
    results,
):

    print()
    print("=" * 80)
    print("SCENE2EPISODE VIDEO RESULT")
    print("=" * 80)

    print(
        f"Input clip : {video_path}"
    )

    print(
        f"Frames used: {len(clip_frames)}"
    )

    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):

        candidate = result[
            "best_candidate"
        ]

        metadata = candidate[
            "metadata"
        ]

        print("-" * 80)

        print(
            f"RANK {rank}"
        )

        print(
            f"Title       : "
            f"{metadata.get('title')}"
        )

        if metadata.get(
            "season"
        ) is not None:

            print(
                f"Season      : "
                f"{metadata.get('season')}"
            )

        if metadata.get(
            "episode"
        ) is not None:

            print(
                f"Episode     : "
                f"{metadata.get('episode')}"
            )

        if metadata.get(
            "episode_title"
        ):

            print(
                f"Episode title: "
                f"{metadata.get('episode_title')}"
            )
        timestamp = format_timestamp(
            metadata.get(
                "timestamp_seconds",
                0,
            )
        )

        print(
            f"Timestamp   : {timestamp}"
        )

        print(
            f"Similarity  : "
            f"{result['best_similarity']:.4f}"
        )

        print(
            f"Video score : "
            f"{result['score']:.4f}"
        )

        print(
            f"Coverage    : "
            f"{result['coverage'] * 100:.1f}%"
        )

        print(
            f"Evidence    : "
            f"{result['evidence_count']} matches"
        )

    print("-" * 80)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Scene2Episode video inference"
        )
    )

    parser.add_argument(
        "video",
        help="Path to a short video clip",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames to sample",
    )

    parser.add_argument(
        "--top-k-frames",
        type=int,
        default=DEFAULT_TOP_K_FRAMES,
    )

    parser.add_argument(
        "--top-k-episodes",
        type=int,
        default=DEFAULT_TOP_K_EPISODES,
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    ).resolve()

    if not video_path.exists():

        print(
            f"ERROR: Video not found:\n"
            f"{video_path}"
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
        f"Index vectors: {index.ntotal}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("Loading metadata...")

    metadata = load_metadata()

    print(
        f"Metadata entries: {len(metadata)}"
    )

    if index.ntotal != len(metadata):

        raise RuntimeError(
            "FAISS index and metadata "
            "sizes do not match."
        )

    # --------------------------------------------------------
    # Temporary frame directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="scene2episode_"
    ) as temp_dir:

        temp_dir = Path(temp_dir)

        clip_frames = extract_clip_frames(
            video_path,
            temp_dir,
            args.frames,
        )

        if not clip_frames:

            raise RuntimeError(
                "No frames could be extracted."
            )

        image_paths = [
            item[0]
            for item in clip_frames
        ]

        # ----------------------------------------------------
        # CLIP
        # ----------------------------------------------------

        embedder = CLIPEmbedder()

        print()
        print(
            "Encoding clip frames..."
        )

        embeddings = (
            embedder.encode_images(
                image_paths
            )
        )

        # ----------------------------------------------------
        # Search every frame
        # ----------------------------------------------------

        print(
            "Searching FAISS..."
        )

        frame_results = []

        for frame_id, embedding in enumerate(
            embeddings
        ):

            results = search_frame(
                embedding,
                index,
                metadata,
                args.top_k_frames,
            )

            frame_results.append(
                results
            )

            print(
                f"Frame {frame_id + 1}/"
                f"{len(embeddings)}: "
                f"{len(results)} matches"
            )

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        results = aggregate_results(
            frame_results,
            args.top_k_episodes,
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print_results(
            video_path,
            clip_frames,
            results,
        )


if __name__ == "__main__":
    main()