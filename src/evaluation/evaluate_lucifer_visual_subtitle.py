import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import faiss
import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
)

from src.retrieval.episode_temporal_reranker_v2 import (
    EpisodeTemporalRerankerV2,
)

from src.retrieval.multicandidate_subtitle_reranker import (
    MultiCandidateSubtitleReranker,
)


# =============================================================
# PATHS
# =============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

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


# =============================================================
# SETTINGS
# =============================================================

MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)

CLIP_OFFSETS = [
    -4,
    -2,
    0,
    2,
    4,
]

TOP_K_PER_FRAME = 50

QUERY_EXCLUSION_SECONDS = 6.0

VISUAL_WEIGHT = 0.80

SUBTITLE_WEIGHT = 0.20

DEFAULT_SAMPLES = 50

DEFAULT_SEED = 42


# =============================================================
# SUBTITLE-COVERED LUCIFER EPISODES
# =============================================================

SUBTITLE_EPISODES = {
    "lucifer_s01_e01",
    "lucifer_s01_e05",
    "lucifer_s01_e06",
    "lucifer_s01_e08",
    "lucifer_s01_e09",
    "lucifer_s01_e10",
}


# =============================================================
# CLIP
# =============================================================

def load_clip():

    print(
        "Loading CLIP model..."
    )

    processor = (
        CLIPProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        CLIPModel.from_pretrained(
            MODEL_NAME
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        processor,
        model,
        device,
    )


# =============================================================
# IMAGE
# =============================================================

def load_image(
    image_path
):

    return (
        Image.open(
            image_path
        )
        .convert("RGB")
    )


# =============================================================
# CLIP EMBEDDING
# =============================================================

def get_embedding(
    image,
    processor,
    model,
    device,
):

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.no_grad():

        vision_outputs = (
            model.vision_model(
                pixel_values=
                    inputs[
                        "pixel_values"
                    ]
            )
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
            image_features
            /
            image_features.norm(
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


# =============================================================
# EPISODE KEY
# =============================================================

def episode_key(
    item
):

    return (
        item["media_id"],
        item.get("season"),
        item.get("episode"),
    )


# =============================================================
# LEAKAGE CHECK
# =============================================================

def is_query_clip_frame(
    candidate,
    query_meta,
):

    if (
        candidate["media_id"]
        != query_meta["media_id"]
    ):

        return False

    candidate_time = float(
        candidate[
            "timestamp_seconds"
        ]
    )

    query_time = float(
        query_meta[
            "timestamp_seconds"
        ]
    )

    return (
        abs(
            candidate_time
            - query_time
        )
        <= QUERY_EXCLUSION_SECONDS
    )


# =============================================================
# BUILD FRAME LOOKUP
# =============================================================

def build_frame_lookup(
    metadata
):

    lookup = defaultdict(list)

    for item in metadata:

        media_id = (
            item["media_id"]
        )

        timestamp = float(
            item[
                "timestamp_seconds"
            ]
        )

        lookup[
            media_id
        ].append(
            (
                timestamp,
                item,
            )
        )

    for media_id in lookup:

        lookup[
            media_id
        ].sort(
            key=lambda x: x[0]
        )

    return lookup


# =============================================================
# FIND CLOSEST FRAME
# =============================================================

def closest_frame(
    frames,
    target_time,
):

    if not frames:

        return None

    best_item = None
    best_distance = float(
        "inf"
    )

    for timestamp, item in frames:

        distance = abs(
            timestamp
            - target_time
        )

        if distance < best_distance:

            best_distance = (
                distance
            )

            best_item = item

    return best_item


# =============================================================
# RETRIEVE CANDIDATES
# =============================================================

def retrieve_candidates(
    query_meta,
    metadata,
    frame_lookup,
    index,
    processor,
    model,
    device,
):

    query_time = float(
        query_meta[
            "timestamp_seconds"
        ]
    )

    query_media_id = (
        query_meta["media_id"]
    )

    candidates = []

    # ---------------------------------------------------------
    # Five-frame temporal query
    # ---------------------------------------------------------

    for query_frame_id, offset in (
        enumerate(
            CLIP_OFFSETS
        )
    ):

        target_time = (
            query_time
            + offset
        )

        if target_time < 0:

            continue

        query_frame = (
            closest_frame(
                frame_lookup[
                    query_media_id
                ],
                target_time,
            )
        )

        if query_frame is None:

            continue

        image_path = Path(
            query_frame[
                "image_path"
            ]
        )

        if not image_path.exists():

            continue

        image = load_image(
            image_path
        )

        query_embedding = (
            get_embedding(
                image=image,
                processor=processor,
                model=model,
                device=device,
            )
        )

        similarities, indices = (
            index.search(
                query_embedding,
                TOP_K_PER_FRAME,
            )
        )

        # -----------------------------------------------------
        # Add retrieved candidates
        # -----------------------------------------------------

        for rank, (
            similarity,
            index_position,
        ) in enumerate(
            zip(
                similarities[0],
                indices[0],
            ),
            start=1,
        ):

            if index_position < 0:

                continue

            candidate = dict(
                metadata[
                    int(index_position)
                ]
            )

            # -------------------------------------------------
            # IMPORTANT:
            # Exclude entire query temporal window.
            # -------------------------------------------------

            if is_query_clip_frame(
                candidate,
                query_meta,
            ):

                continue

            candidate[
                "clip_similarity"
            ] = float(
                similarity
            )

            candidate[
                "retrieval_rank"
            ] = int(
                rank
            )

            candidate[
                "query_frame_id"
            ] = int(
                query_frame_id
            )

            candidate[
                "query_offset"
            ] = float(
                offset
            )

            candidates.append(
                candidate
            )

    return candidates


# =============================================================
# GROUP CANDIDATES BY EPISODE
# =============================================================

def group_candidates(
    candidates
):

    grouped = defaultdict(
        list
    )

    for candidate in candidates:

        key = episode_key(
            candidate
        )

        grouped[key].append(
            candidate
        )

    return dict(
        grouped
    )


# =============================================================
# EVALUATE ONE QUERY
# =============================================================

def evaluate_query(
    query_meta,
    metadata,
    frame_lookup,
    index,
    processor,
    model,
    device,
    visual_reranker,
    subtitle_reranker,
):

    candidates = (
        retrieve_candidates(
            query_meta=query_meta,
            metadata=metadata,
            frame_lookup=
                frame_lookup,
            index=index,
            processor=processor,
            model=model,
            device=device,
        )
    )

    grouped = (
        group_candidates(
            candidates
        )
    )

    # =========================================================
    # VISUAL V3
    # =========================================================

    visual_results = []

    total_query_frames = len(
        CLIP_OFFSETS
    )

    for key, episode_candidates in (
        grouped.items()
    ):

        result = (
            visual_reranker.score_episode(
                candidates=
                    episode_candidates,
                total_query_frames=
                    total_query_frames,
                credit_mode=False,
            )
        )

        result[
            "episode_key"
        ] = key

        visual_results.append(
            result
        )

    visual_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # =========================================================
    # VISUAL + MULTI-CANDIDATE SUBTITLE
    # =========================================================

    combined_results = []

    query_media_id = (
        query_meta["media_id"]
    )

    query_timestamp = float(
        query_meta[
            "timestamp_seconds"
        ]
    )

    for visual_result in (
        visual_results
    ):

        key = (
            visual_result[
                "episode_key"
            ]
        )

        episode_candidates = (
            grouped.get(
                key,
                []
            )
        )

        subtitle_result = (
            subtitle_reranker.score_episode(
                query_media_id=
                    query_media_id,
                query_timestamp=
                    query_timestamp,
                candidates=
                    episode_candidates,
            )
        )

        visual_score = float(
            visual_result[
                "score"
            ]
        )

        subtitle_score = float(
            subtitle_result[
                "score"
            ]
        )

        combined_score = (
            VISUAL_WEIGHT
            * visual_score
            +
            SUBTITLE_WEIGHT
            * subtitle_score
        )

        combined_results.append(
            {
                "episode_key":
                    key,

                "visual_score":
                    visual_score,

                "subtitle_score":
                    subtitle_score,

                "combined_score":
                    float(
                        combined_score
                    ),

                "visual_result":
                    visual_result,

                "subtitle_result":
                    subtitle_result,
            }
        )

    combined_results.sort(
        key=lambda x:
            x["combined_score"],
        reverse=True,
    )

    return (
        visual_results,
        combined_results,
    )


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lucifer Visual V3 vs "
            "Multi-Candidate Subtitle"
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    print()
    print("=" * 75)
    print(
        "LUCIFER — VISUAL V3 vs "
        "MULTI-CANDIDATE SUBTITLE"
    )
    print("=" * 75)
    print()

    print(
        "Subtitle-covered Lucifer episodes:"
    )

    for media_id in sorted(
        SUBTITLE_EPISODES
    ):

        print(
            f"  {media_id}"
        )

    print()

    # =========================================================
    # LOAD FAISS
    # =========================================================

    print(
        "Loading FAISS index..."
    )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Vectors: "
        f"{index.ntotal}"
    )

    # =========================================================
    # LOAD METADATA
    # =========================================================

    print(
        "Loading metadata..."
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    print(
        f"Metadata: "
        f"{len(metadata)}"
    )

    if (
        index.ntotal
        != len(metadata)
    ):

        raise RuntimeError(
            "FAISS index and metadata "
            "size mismatch."
        )

    # =========================================================
    # BUILD FRAME LOOKUP
    # =========================================================

    print(
        "Building frame lookup..."
    )

    frame_lookup = (
        build_frame_lookup(
            metadata
        )
    )

    # =========================================================
    # FILTER LUCIFER
    # =========================================================

    lucifer_metadata = [
        item
        for item in metadata
        if item["media_id"]
        in SUBTITLE_EPISODES
    ]

    print(
        f"Evaluation frames: "
        f"{len(lucifer_metadata)}"
    )

    # =========================================================
    # RANDOM SAMPLE
    # =========================================================

    random.seed(
        args.seed
    )

    sample_count = min(
        args.samples,
        len(lucifer_metadata),
    )

    samples = random.sample(
        lucifer_metadata,
        sample_count,
    )

    print(
        f"Random seed: "
        f"{args.seed}"
    )

    print(
        f"Evaluation samples: "
        f"{sample_count}"
    )

    # =========================================================
    # LOAD CLIP
    # =========================================================

    processor, model, device = (
        load_clip()
    )

    print(
        f"Device: "
        f"{device}"
    )

    # =========================================================
    # LOAD RERANKERS
    # =========================================================

    print(
        "Loading visual reranker..."
    )

    visual_reranker = (
        EpisodeTemporalRerankerV2()
    )

    print(
        "Loading multi-candidate "
        "subtitle reranker..."
    )

    subtitle_reranker = (
        MultiCandidateSubtitleReranker()
    )

    print(
        f"Subtitle entries loaded: "
        f"{sum(len(v) for v in subtitle_reranker.entries_by_media.values())}"
    )

    # =========================================================
    # METRICS
    # =========================================================

    visual_top1 = 0
    visual_top3 = 0
    visual_top5 = 0

    combined_top1 = 0
    combined_top3 = 0
    combined_top5 = 0

    visual_failures = []
    combined_failures = []

    # =========================================================
    # EVALUATION LOOP
    # =========================================================

    for count, query_meta in enumerate(
        samples,
        start=1,
    ):

        (
            visual_results,
            combined_results,
        ) = evaluate_query(
            query_meta=query_meta,
            metadata=metadata,
            frame_lookup=
                frame_lookup,
            index=index,
            processor=processor,
            model=model,
            device=device,
            visual_reranker=
                visual_reranker,
            subtitle_reranker=
                subtitle_reranker,
        )

        ground_truth = (
            episode_key(
                query_meta
            )
        )

        # =====================================================
        # VISUAL RANK
        # =====================================================

        visual_keys = [
            result[
                "episode_key"
            ]
            for result
            in visual_results
        ]

        if (
            ground_truth
            in visual_keys
        ):

            visual_rank = (
                visual_keys.index(
                    ground_truth
                )
                + 1
            )

        else:

            visual_rank = None

        if visual_rank == 1:

            visual_top1 += 1

        if (
            visual_rank is not None
            and visual_rank <= 3
        ):

            visual_top3 += 1

        if (
            visual_rank is not None
            and visual_rank <= 5
        ):

            visual_top5 += 1

        # =====================================================
        # COMBINED RANK
        # =====================================================

        combined_keys = [
            result[
                "episode_key"
            ]
            for result
            in combined_results
        ]

        if (
            ground_truth
            in combined_keys
        ):

            combined_rank = (
                combined_keys.index(
                    ground_truth
                )
                + 1
            )

        else:

            combined_rank = None

        if combined_rank == 1:

            combined_top1 += 1

        if (
            combined_rank is not None
            and combined_rank <= 3
        ):

            combined_top3 += 1

        if (
            combined_rank is not None
            and combined_rank <= 5
        ):

            combined_top5 += 1

        # =====================================================
        # STORE VISUAL FAILURE
        # =====================================================

        if visual_rank != 1:

            predicted = (
                visual_results[0]
                if visual_results
                else None
            )

            visual_failures.append(
                {
                    "query":
                        query_meta,
                    "rank":
                        visual_rank,
                    "predicted":
                        predicted,
                }
            )

        # =====================================================
        # STORE COMBINED FAILURE
        # =====================================================

        if combined_rank != 1:

            predicted = (
                combined_results[0]
                if combined_results
                else None
            )

            combined_failures.append(
                {
                    "query":
                        query_meta,
                    "rank":
                        combined_rank,
                    "predicted":
                        predicted,
                }
            )

        if (
            count % 10
            == 0
        ):

            print(
                f"Processed "
                f"{count}/{sample_count}"
            )

    # =========================================================
    # RESULTS
    # =========================================================

    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)
    print()

    print(
        "VISUAL V3"
    )

    print(
        "-" * 75
    )

    print(
        f"Top-1: "
        f"{100 * visual_top1 / sample_count:.2f}%"
    )

    print(
        f"Top-3: "
        f"{100 * visual_top3 / sample_count:.2f}%"
    )

    print(
        f"Top-5: "
        f"{100 * visual_top5 / sample_count:.2f}%"
    )

    print()

    print(
        "VISUAL V3 + "
        "MULTI-CANDIDATE SUBTITLE"
    )

    print(
        "-" * 75
    )

    print(
        f"Top-1: "
        f"{100 * combined_top1 / sample_count:.2f}%"
    )

    print(
        f"Top-3: "
        f"{100 * combined_top3 / sample_count:.2f}%"
    )

    print(
        f"Top-5: "
        f"{100 * combined_top5 / sample_count:.2f}%"
    )

    print()

    print(
        f"Visual failures: "
        f"{len(visual_failures)}"
    )

    print(
        f"Combined failures: "
        f"{len(combined_failures)}"
    )

    # =========================================================
    # COMBINED FAILURES
    # =========================================================

    print()
    print("=" * 75)
    print(
        "COMBINED FAILURES"
    )
    print("=" * 75)

    if not combined_failures:

        print()
        print(
            "No Top-1 failures."
        )

    else:

        for number, failure in enumerate(
            combined_failures,
            start=1,
        ):

            query = failure[
                "query"
            ]

            predicted = failure[
                "predicted"
            ]

            print()

            print(
                f"{number}. "
                f"{query['media_id']} "
                f"@ "
                f"{query['timestamp_seconds']:.1f}s"
            )

            print(
                f"   Correct rank: "
                f"{failure['rank']}"
            )

            if predicted is None:

                print(
                    "   Predicted: None"
                )

                continue

            print(
                f"   Predicted: "
                f"{predicted['episode_key']}"
            )

            print(
                f"   V3: "
                f"{predicted['visual_score']:.4f}"
            )

            print(
                f"   Multi-subtitle: "
                f"{predicted['subtitle_score']:.4f}"
            )

            print(
                f"   Combined: "
                f"{predicted['combined_score']:.4f}"
            )

            subtitle_result = (
                predicted[
                    "subtitle_result"
                ]
            )

            print(
                f"   Subtitle best: "
                f"{subtitle_result['best_score']:.4f}"
            )

            print(
                f"   Subtitle top3: "
                f"{subtitle_result['top3_mean']:.4f}"
            )

            print(
                f"   Subtitle evidence: "
                f"{subtitle_result['evidence']}"
            )

    print()
    print("=" * 75)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":

    main()