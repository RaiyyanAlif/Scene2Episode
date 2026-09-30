import argparse
import json
import os
import random
import sys
from collections import defaultdict

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.retrieval.episode_temporal_reranker_v2 import EpisodeTemporalRerankerV2
from src.retrieval.multicandidate_subtitle_reranker import (
    MultiCandidateSubtitleReranker,
)


# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = "indexes/scene2episode.index"
METADATA_PATH = "indexes/scene2episode_metadata.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_SAMPLES = 50
DEFAULT_TOP_K = 50
DEFAULT_SEED = 42

QUERY_OFFSETS = [-4, -2, 0, 2, 4]

# Exclude the entire temporal neighborhood of the query
# from retrieval to prevent leakage.
QUERY_EXCLUSION_SECONDS = 6.0

VISUAL_WEIGHT = 0.80
SUBTITLE_WEIGHT = 0.20

SUBTITLE_WINDOW_SECONDS = 15.0


# ============================================================
# UTILITIES
# ============================================================

def canonical_episode_key(item):
    """
    Canonical episode identifier.

    IMPORTANT:
    This matches EpisodeTemporalRerankerV2.
    """
    return (
        item["media_id"],
        item.get("season"),
        item.get("episode"),
    )


def media_key(item):
    return item["media_id"]


def safe_float(value, default=None):
    try:
        if value is None:
            return default

        if isinstance(value, (tuple, list, dict)):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def normalize_candidate(candidate):
    """
    Convert a candidate into the exact flat format expected by
    EpisodeTemporalRerankerV2.

    Some retrieval code stores metadata like:

        {
            "similarity": ...,
            "metadata": {...}
        }

    while V3 expects:

        {
            "clip_similarity": ...,
            "timestamp_seconds": ...,
            "media_id": ...,
            "season": ...,
            "episode": ...
        }

    This function safely handles both.
    """

    if candidate is None:
        return None

    # Already flat
    if "timestamp_seconds" in candidate and "media_id" in candidate:
        result = dict(candidate)

        if "clip_similarity" not in result:
            if "similarity" in result:
                result["clip_similarity"] = float(
                    result["similarity"]
                )
            elif "score" in result:
                result["clip_similarity"] = float(
                    result["score"]
                )
            else:
                result["clip_similarity"] = 0.0

        return result

    # Nested metadata format
    metadata = candidate.get("metadata")

    if not isinstance(metadata, dict):
        return None

    result = dict(metadata)

    if "clip_similarity" in candidate:
        result["clip_similarity"] = float(
            candidate["clip_similarity"]
        )
    elif "similarity" in candidate:
        result["clip_similarity"] = float(
            candidate["similarity"]
        )
    elif "score" in candidate:
        result["clip_similarity"] = float(
            candidate["score"]
        )
    else:
        result["clip_similarity"] = 0.0

    # Preserve useful retrieval information
    for key in (
        "rank",
        "query_frame_id",
        "query_offset",
        "distance",
    ):
        if key in candidate:
            result[key] = candidate[key]

    return result


def normalize_candidates(candidates):
    """
    Normalize a complete candidate list.
    """
    output = []

    for candidate in candidates:
        normalized = normalize_candidate(candidate)

        if normalized is not None:
            output.append(normalized)

    return output


def is_query_clip_frame(candidate, query_meta):
    """
    Leakage prevention.

    Removes all frames from the same media item that are within
    +/- QUERY_EXCLUSION_SECONDS of the query timestamp.
    """

    if candidate["media_id"] != query_meta["media_id"]:
        return False

    candidate_time = safe_float(
        candidate.get("timestamp_seconds")
    )

    query_time = safe_float(
        query_meta.get("timestamp_seconds")
    )

    if candidate_time is None or query_time is None:
        return False

    return (
        abs(candidate_time - query_time)
        <= QUERY_EXCLUSION_SECONDS
    )


def format_time(seconds):
    if seconds is None:
        return "N/A"

    seconds = float(seconds)

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    return f"{h:02d}:{m:02d}:{s:02d}"


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CLIP
# ============================================================

class CLIPEmbedder:

    def __init__(self, model_name):
        print("Loading CLIP model...")

        self.device = torch.device("cpu")

        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )

        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        # Use all available CPU threads.
        torch.set_num_threads(
            max(1, torch.get_num_threads())
        )

        print(f"Device: {self.device}")
        print(
            f"CPU threads: {torch.get_num_threads()}"
        )

    @torch.no_grad()
    def encode_images(self, image_paths):
        """
        Batch CLIP image encoding.

        This is much faster than calling CLIP separately
        for every image.
        """

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

        return image_features.cpu().numpy().astype(
            "float32"
        )

    @torch.no_grad()
    def encode_image(self, image_path):
        """
        Compatibility wrapper for single images.
        """

        embeddings = self.encode_images(
            [image_path]
        )

        return embeddings[0]


# ============================================================
# QUERY EMBEDDINGS
# ============================================================

def get_query_frame_path(query_meta):
    return query_meta["image_path"]


def build_query_embeddings(
    query_meta,
    frame_lookup,
    embedder,
):
    """
    Build the temporal query clip using ONE batched
    CLIP inference call.

    Offsets:
        -4, -2, 0, +2, +4 seconds
    """

    query_time = float(
        query_meta["timestamp_seconds"]
    )

    media_id = query_meta["media_id"]

    media_frames = frame_lookup.get(
        media_id,
        []
    )

    if not media_frames:
        return []

    selected = []
    selected_ids = set()

    for offset in QUERY_OFFSETS:

        target_time = query_time + offset

        if target_time < 0:
            continue

        closest = min(
            media_frames,
            key=lambda item: abs(
                float(item["timestamp_seconds"])
                - target_time
            ),
        )

        frame_id = closest["frame_id"]

        if frame_id not in selected_ids:
            selected.append(
                {
                    "frame_meta": closest,
                    "offset": (
                        float(
                            closest[
                                "timestamp_seconds"
                            ]
                        )
                        - query_time
                    ),
                }
            )

            selected_ids.add(frame_id)

    if not selected:
        return []

    # --------------------------------------------------------
    # BATCH ALL QUERY FRAMES
    # --------------------------------------------------------

    image_paths = [
        item["frame_meta"]["image_path"]
        for item in selected
    ]

    embeddings = embedder.encode_images(
        image_paths
    )

    # --------------------------------------------------------
    # Build result objects
    # --------------------------------------------------------

    results = []

    for item, embedding in zip(
        selected,
        embeddings,
    ):

        results.append(
            {
                "frame_meta": item["frame_meta"],
                "embedding": embedding,
                "offset": item["offset"],
            }
        )

    return results


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_candidates(
    index,
    metadata,
    query_embeddings,
    top_k,
    query_meta,
):
    """
    Retrieve candidates independently for every query frame.

    Then normalize candidate dictionaries so V3 receives the
    correct flat candidate structure.
    """

    all_candidates = []

    for query_idx, query_item in enumerate(
        query_embeddings
    ):

        embedding = query_item["embedding"]

        vector = np.asarray(
            embedding,
            dtype="float32"
        ).reshape(1, -1)

        scores, indices = index.search(
            vector,
            top_k
        )

        query_frame_id = query_item[
            "frame_meta"
        ]["frame_id"]

        query_offset = query_item["offset"]

        for rank, (
            score,
            idx
        ) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            if idx < 0:
                continue

            candidate_meta = metadata[idx]

            # ------------------------------------------------
            # Leakage prevention
            # ------------------------------------------------

            if is_query_clip_frame(
                candidate_meta,
                query_meta
            ):
                continue

            # ------------------------------------------------
            # Build FLAT candidate dictionary
            # ------------------------------------------------

            candidate = dict(candidate_meta)

            candidate["clip_similarity"] = float(
                score
            )

            candidate["query_frame_id"] = query_frame_id
            candidate["query_offset"] = query_offset
            candidate["rank"] = rank

            all_candidates.append(candidate)

    return all_candidates


# ============================================================
# GROUP BY EPISODE
# ============================================================

def group_candidates_by_episode(candidates):

    grouped = defaultdict(list)

    for candidate in candidates:

        ep_key = canonical_episode_key(
            candidate
        )

        grouped[ep_key].append(candidate)

    return grouped


# ============================================================
# SUBTITLE QUERY
# ============================================================

def build_query_subtitle_window(
    subtitle_reranker,
    query_meta,
):
    """
    Obtain the subtitle text around the query timestamp.
    """

    media_id = query_meta["media_id"]

    timestamp = safe_float(
        query_meta.get("timestamp_seconds")
    )

    if timestamp is None:
        return ""

    try:
        return subtitle_reranker.get_window(
            media_id,
            timestamp,
        )

    except Exception:
        return ""


# ============================================================
# VISUAL RANKING
# ============================================================

def rank_visual_candidates(
    visual_reranker,
    grouped_candidates,
    total_query_frames,
):
    """
    Run EpisodeTemporalRerankerV2 for every episode.
    """

    ranked = []

    for ep_key, candidates in grouped_candidates.items():

        # IMPORTANT:
        # candidates are already normalized to the flat format.
        score_info = visual_reranker.score_episode(
            candidates,
            total_query_frames,
        )

        if score_info is None:
            continue

        score_info["episode_key"] = ep_key

        ranked.append(score_info)

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked


# ============================================================
# SUBTITLE RANKING
# ============================================================

def rank_subtitle_candidates(
    subtitle_reranker,
    query_window,
    grouped_candidates,
):
    """
    Run MultiCandidateSubtitleReranker over every episode.
    """

    ranked = []

    if not query_window:
        return ranked

    for ep_key, candidates in grouped_candidates.items():

        try:

            result = subtitle_reranker.score_episode(
                query_window,
                ep_key,
                candidates,
            )

            if result is None:
                continue

            # Ensure episode key exists in result
            result["episode_key"] = ep_key

            ranked.append(result)

        except Exception as e:

            print(
                f"Warning: subtitle scoring failed "
                f"for {ep_key}: {e}"
            )

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked


# ============================================================
# COMBINE VISUAL + SUBTITLE
# ============================================================

def combine_scores(
    visual_ranked,
    subtitle_ranked,
    subtitle_enabled,
):
    """
    Combine visual V3 and multi-candidate subtitle scores.

    If subtitle is unavailable for the query, keep the visual
    score unchanged.

    If subtitle is available, use:

        0.80 * visual
        0.20 * subtitle
    """

    subtitle_lookup = {
        item["episode_key"]: item["score"]
        for item in subtitle_ranked
    }

    combined = []

    for visual_item in visual_ranked:

        ep_key = visual_item["episode_key"]

        visual_score = float(
            visual_item["score"]
        )

        if not subtitle_enabled:
            subtitle_score = 0.0
            final_score = visual_score

        else:
            subtitle_score = float(
                subtitle_lookup.get(
                    ep_key,
                    0.0
                )
            )

            final_score = (
                VISUAL_WEIGHT * visual_score
                + SUBTITLE_WEIGHT * subtitle_score
            )

        result = dict(visual_item)

        result["visual_score"] = visual_score
        result["subtitle_score"] = subtitle_score
        result["combined_score"] = final_score

        combined.append(result)

    combined.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return combined


# ============================================================
# METRICS
# ============================================================

def evaluate_ranked(
    ranked,
    ground_truth,
):
    """
    Return:
        rank
        top1
        top3
        top5
    """

    ranked_keys = [
        item["episode_key"]
        for item in ranked
    ]

    try:
        rank = ranked_keys.index(
            ground_truth
        ) + 1

    except ValueError:
        rank = None

    return {
        "rank": rank,
        "top1": rank == 1,
        "top3": rank is not None and rank <= 3,
        "top5": rank is not None and rank <= 5,
    }


# ============================================================
# MAIN
# ============================================================
def apply_credit_override(
    visual_ranked,
    subtitle_ranked,
    query_window,
):
    """
    Credit-scene correction.

    If the query contains strong credit-related OCR text,
    prioritize an episode when its multi-candidate subtitle/OCR
    evidence is extremely strong.

    Normal scenes are left unchanged.
    """

    if not query_window:
        return visual_ranked

    credit_keywords = [
        "MAIN CAST",
        "ADDITIONAL VOICES",
        "CAST",
        "VOICE",
        "VOICES",
        "MAGYAR HANGOK",
        "TOVABBI MAGYAR HANGOK",
        "MAGYAR VALTOZAT",
        "VERSIONE ITALIANA",
        "ELENCO",
        "COMPOSITORS",
        "ANIMATORS",
        "PRODUCTION",
        "DIRECTOR",
        "WRITTEN BY",
        "EDITED BY",
        "SPECIAL EFFECTS",
        "VISUAL EFFECTS",
    ]

    query_upper = query_window.upper()

    is_credit = any(
        keyword in query_upper
        for keyword in credit_keywords
    )

    if not is_credit:
        return visual_ranked

    # --------------------------------------------------------
    # Credit scene detected
    # --------------------------------------------------------

    corrected = []

    for item in visual_ranked:

        result = dict(item)

        # Multi-candidate subtitle/OCR score
        subtitle_score = float(
            item.get(
                "subtitle_score",
                item.get("score", 0.0),
            )
        )

        # Very strong OCR evidence
        if subtitle_score >= 0.90:

            result["credit_override"] = True

            # Put extremely strong OCR evidence above
            # ordinary CLIP similarity.
            result["credit_score"] = (
                0.10 * float(
                    item.get(
                        "score",
                        0.0,
                    )
                )
                + 0.90 * subtitle_score
            )

        else:

            result["credit_override"] = False

            result["credit_score"] = (
                0.70 * float(
                    item.get(
                        "score",
                        0.0,
                    )
                )
                + 0.30 * subtitle_score
            )

        corrected.append(result)

    # If at least one candidate has near-perfect OCR,
    # use credit_score ordering.
    strong_credit = [
        item
        for item in corrected
        if item["credit_override"]
    ]

    if strong_credit:

        corrected.sort(
            key=lambda x: x["credit_score"],
            reverse=True,
        )

        return corrected

    # Otherwise retain normal visual ranking.
    corrected.sort(
        key=lambda x: x.get(
            "score",
            0.0,
        ),
        reverse=True,
    )

    return corrected


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
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
        "VISUAL V3 + MULTI-CANDIDATE "
        "TEMPORAL SUBTITLE BENCHMARK"
    )
    print("=" * 75)
    print()

    # ========================================================
    # LOAD FAISS
    # ========================================================

    print("Loading FAISS index...")

    index = faiss.read_index(
        INDEX_PATH
    )

    print(
        f"Index vectors: {index.ntotal}"
    )

    metadata = load_metadata(
        METADATA_PATH
    )

    print(
        f"Metadata entries: {len(metadata)}"
    )

    # ========================================================
    # FRAME LOOKUP
    # ========================================================

    print("Building frame lookup...")

    frame_lookup = defaultdict(list)

    for item in metadata:
        frame_lookup[
            item["media_id"]
        ].append(item)

    # Sort by timestamp
    for media_id in frame_lookup:
        frame_lookup[media_id].sort(
            key=lambda x: float(
                x["timestamp_seconds"]
            )
        )

    # ========================================================
    # SUBTITLE RERANKER
    # ========================================================

    subtitle_reranker = (
        MultiCandidateSubtitleReranker()
    )

    print(
        f"Subtitle episodes: "
        f"{len(subtitle_reranker.entries_by_media)}"
    )

    # ========================================================
    # CLIP
    # ========================================================

    embedder = CLIPEmbedder(
        MODEL_NAME
    )

    # ========================================================
    # V3
    # ========================================================

    visual_reranker = (
        EpisodeTemporalRerankerV2()
    )

    # ========================================================
    # SAMPLE QUERIES
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
        sample_count,
    )

    print(
        f"Samples: {sample_count}"
    )

    print(
        f"Random seed: {args.seed}"
    )

    print()

    # ========================================================
    # METRIC STORAGE
    # ========================================================

    visual_top1 = 0
    visual_top3 = 0
    visual_top5 = 0

    combined_top1 = 0
    combined_top3 = 0
    combined_top5 = 0

    visual_failures = []
    combined_failures = []

    subtitle_enabled_count = 0

    timestamp_errors = []

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    for sample_number, sample_index in enumerate(
        sample_indices,
        start=1
    ):

        query_meta = metadata[
            sample_index
        ]

        ground_truth = canonical_episode_key(
            query_meta
        )

        print(
            f"[{sample_number:02d}/{sample_count}] "
            f"{query_meta['media_id']} "
            f"@ "
            f"{format_time(query_meta['timestamp_seconds'])}"
        )

        # ----------------------------------------------------
        # Build query clip
        # ----------------------------------------------------

        query_embeddings = build_query_embeddings(
            query_meta,
            frame_lookup,
            embedder,
        )

        if not query_embeddings:
            print(
                "  WARNING: no query embeddings"
            )
            continue

        # ----------------------------------------------------
        # Retrieve visual candidates
        # ----------------------------------------------------

        raw_candidates = retrieve_candidates(
            index=index,
            metadata=metadata,
            query_embeddings=query_embeddings,
            top_k=args.top_k,
            query_meta=query_meta,
        )

        # ----------------------------------------------------
        # Normalize candidate format
        # ----------------------------------------------------

        candidates = normalize_candidates(
            raw_candidates
        )

        # ----------------------------------------------------
        # Group by episode
        # ----------------------------------------------------

        grouped_candidates = (
            group_candidates_by_episode(
                candidates
            )
        )

        total_query_frames = len(
            query_embeddings
        )

        # ----------------------------------------------------
        # VISUAL V3
        # ----------------------------------------------------

        visual_ranked = rank_visual_candidates(
            visual_reranker,
            grouped_candidates,
            total_query_frames,
        )

        visual_metrics = evaluate_ranked(
            visual_ranked,
            ground_truth,
        )

        if visual_metrics["top1"]:
            visual_top1 += 1

        if visual_metrics["top3"]:
            visual_top3 += 1

        if visual_metrics["top5"]:
            visual_top5 += 1

        if not visual_metrics["top1"]:

            visual_failures.append(
                {
                    "query": query_meta,
                    "rank": visual_metrics["rank"],
                    "predicted": (
                        visual_ranked[0]["episode_key"]
                        if visual_ranked
                        else None
                    ),
                }
            )

        # ----------------------------------------------------
        # QUERY SUBTITLE WINDOW
        # ----------------------------------------------------

        query_window = (
            build_query_subtitle_window(
                subtitle_reranker,
                query_meta,
            )
        )

        subtitle_enabled = bool(
            query_window.strip()
        )

        if subtitle_enabled:
            subtitle_enabled_count += 1

        # ----------------------------------------------------
        # MULTI-CANDIDATE SUBTITLE
        # ----------------------------------------------------

        subtitle_ranked = []

        if subtitle_enabled:

            subtitle_ranked = (
                rank_subtitle_candidates(
                    subtitle_reranker,
                    query_window,
                    grouped_candidates,
                )
            )

        # ----------------------------------------------------
        # COMBINED
        # ----------------------------------------------------

        combined_ranked = combine_scores(
            visual_ranked=visual_ranked,
            subtitle_ranked=subtitle_ranked,
            subtitle_enabled=subtitle_enabled,
        )
        combined_ranked = apply_credit_override(
            visual_ranked=combined_ranked,
            subtitle_ranked=subtitle_ranked,
            query_window=query_window,
        )

        combined_metrics = evaluate_ranked(
            combined_ranked,
            ground_truth,
        )

        if combined_metrics["top1"]:
            combined_top1 += 1

        if combined_metrics["top3"]:
            combined_top3 += 1

        if combined_metrics["top5"]:
            combined_top5 += 1

        if not combined_metrics["top1"]:

            combined_failures.append(
                {
                    "query": query_meta,
                    "rank": combined_metrics["rank"],
                    "predicted": (
                        combined_ranked[0]["episode_key"]
                        if combined_ranked
                        else None
                    ),
                }
            )

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        if combined_ranked:

            best = combined_ranked[0]

            best_candidate = best.get(
                "best_candidate"
            )

            if best_candidate:

                predicted_time = safe_float(
                    best_candidate.get(
                        "timestamp_seconds"
                    )
                )

                actual_time = safe_float(
                    query_meta.get(
                        "timestamp_seconds"
                    )
                )

                if (
                    predicted_time is not None
                    and actual_time is not None
                ):

                    timestamp_errors.append(
                        abs(
                            predicted_time
                            - actual_time
                        )
                    )

        # ----------------------------------------------------
        # PRINT SAMPLE RESULT
        # ----------------------------------------------------

        visual_prediction = (
            visual_ranked[0]["episode_key"]
            if visual_ranked
            else None
        )

        combined_prediction = (
            combined_ranked[0]["episode_key"]
            if combined_ranked
            else None
        )

        print(
            f"  GT       : {ground_truth}"
        )

        print(
            f"  Visual   : {visual_prediction}"
        )

        print(
            f"  Combined : {combined_prediction}"
        )

        if subtitle_enabled:
            print(
                "  Subtitle : ENABLED"
            )

        else:
            print(
                "  Subtitle : unavailable"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    n = sample_count

    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)

    print()
    print("VISUAL V3")
    print(
        f"Top-1 episode : "
        f"{100 * visual_top1 / n:.2f}%"
    )

    print(
        f"Top-3 episode : "
        f"{100 * visual_top3 / n:.2f}%"
    )

    print(
        f"Top-5 episode : "
        f"{100 * visual_top5 / n:.2f}%"
    )

    print()
    print("VISUAL V3 + MULTI-CANDIDATE SUBTITLE")

    print(
        f"Top-1 episode : "
        f"{100 * combined_top1 / n:.2f}%"
    )

    print(
        f"Top-3 episode : "
        f"{100 * combined_top3 / n:.2f}%"
    )

    print(
        f"Top-5 episode : "
        f"{100 * combined_top5 / n:.2f}%"
    )

    print()
    print(
        f"Subtitle-enabled queries: "
        f"{subtitle_enabled_count}/{n}"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if timestamp_errors:

        mean_error = float(
            np.mean(timestamp_errors)
        )

        median_error = float(
            np.median(timestamp_errors)
        )

        print(
            f"Mean timestamp error: "
            f"{mean_error:.2f} sec"
        )

        print(
            f"Median timestamp error: "
            f"{median_error:.2f} sec"
        )

    else:

        print(
            "Mean timestamp error: N/A"
        )

        print(
            "Median timestamp error: N/A"
        )

    # --------------------------------------------------------
    # Failures
    # --------------------------------------------------------

    print()
    print(
        f"Visual failures: "
        f"{len(visual_failures)}"
    )

    print(
        f"Combined failures: "
        f"{len(combined_failures)}"
    )

    # ========================================================
    # FAILURE DETAILS
    # ========================================================

    if combined_failures:

        print()
        print("=" * 75)
        print("COMBINED TOP-1 FAILURES")
        print("=" * 75)

        for i, failure in enumerate(
            combined_failures,
            start=1,
        ):

            query = failure["query"]

            print()
            print(
                f"{i}. "
                f"{query['media_id']} "
                f"@ "
                f"{format_time(query['timestamp_seconds'])}"
            )

            print(
                f"   Correct : "
                f"{canonical_episode_key(query)}"
            )

            print(
                f"   Predicted: "
                f"{failure['predicted']}"
            )

            print(
                f"   Correct rank: "
                f"{failure['rank']}"
            )

    print()
    print("=" * 75)
    print("BENCHMARK COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()