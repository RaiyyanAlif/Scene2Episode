import os
import tempfile
from pathlib import Path

import streamlit as st

from src.inference import (
    CLIPEmbedder,
    load_metadata,
    retrieve,
)
from src.video_inference import (
    extract_clip_frames,
    search_frame,
    aggregate_results,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Scene2Episode",
    page_icon="🎬",
    layout="centered",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

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


# ============================================================
# LOAD MODEL / INDEX
# ============================================================

@st.cache_resource
def load_engine():

    import faiss

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    metadata = load_metadata()

    embedder = CLIPEmbedder()

    return (
        index,
        metadata,
        embedder,
    )


# ============================================================
# RESULT HELPERS
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


def get_title(metadata):

    return metadata.get(
        "title",
        metadata.get(
            "media_id",
            "Unknown",
        ),
    )


def get_episode_name(metadata):

    title = get_title(metadata)

    season = metadata.get("season")
    episode = metadata.get("episode")

    if (
        season is not None
        and episode is not None
    ):

        return (
            f"{title} "
            f"S{int(season):02d} "
            f"E{int(episode):02d}"
        )

    return title


# ============================================================
# IMAGE INFERENCE
# ============================================================

def run_image_inference(
    image_path,
    index,
    metadata,
    embedder,
):

    embedding = embedder.encode_image(
        image_path
    )

    results = retrieve(
        query_embedding=embedding,
        index=index,
        metadata=metadata,
        top_k_frames=50,
        top_k_episodes=5,
    )

    return results


# ============================================================
# VIDEO INFERENCE
# ============================================================

def run_video_inference(
    video_path,
    index,
    metadata,
    embedder,
):

    with tempfile.TemporaryDirectory(
        prefix="scene2episode_"
    ) as temp_dir:

        temp_dir = Path(temp_dir)

        clip_frames = extract_clip_frames(
            video_path,
            temp_dir,
            num_frames=5,
        )

        if not clip_frames:

            return []

        image_paths = [
            item[0]
            for item in clip_frames
        ]

        embeddings = embedder.encode_images(
            image_paths
        )

        frame_results = []

        for embedding in embeddings:

            results = search_frame(
                embedding,
                index,
                metadata,
                top_k=20,
            )

            frame_results.append(
                results
            )

        results = aggregate_results(
            frame_results,
            top_k_episodes=5,
        )

        return results


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_results(results):

    if not results:

        st.error(
            "No matching scene found."
        )

        return

    best = results[0]

    candidate = best[
        "best_candidate"
    ]

    metadata = candidate[
        "metadata"
    ]

    title = get_title(
        metadata
    )

    season = metadata.get(
        "season"
    )

    episode = metadata.get(
        "episode"
    )

    timestamp = format_timestamp(
        metadata.get(
            "timestamp_seconds",
            0,
        )
    )

    score = float(
        best.get(
            "score",
            best.get(
                "best_similarity",
                0,
            ),
        )
    )

    # --------------------------------------------------------
    # Main result
    # --------------------------------------------------------

    st.success(
        "Scene identified!"
    )

    st.markdown(
        f"""
        ## 🎬 {title}
        """
    )

    if (
        season is not None
        and episode is not None
    ):

        st.markdown(
            f"### Season {int(season)} • "
            f"Episode {int(episode)}"
        )

    st.metric(
        "Timestamp",
        timestamp,
    )

    st.metric(
        "Confidence",
        f"{score * 100:.2f}%",
    )

    # --------------------------------------------------------
    # Episode title
    # --------------------------------------------------------

    episode_title = metadata.get(
        "episode_title"
    )

    if episode_title:

        st.write(
            f"**Episode:** {episode_title}"
        )

    # --------------------------------------------------------
    # Alternatives
    # --------------------------------------------------------

    if len(results) > 1:

        st.divider()

        st.subheader(
            "Other possible matches"
        )

        for rank, result in enumerate(
            results[1:],
            start=2,
        ):

            candidate = result[
                "best_candidate"
            ]

            metadata = candidate[
                "metadata"
            ]

            name = get_episode_name(
                metadata
            )

            timestamp = format_timestamp(
                metadata.get(
                    "timestamp_seconds",
                    0,
                )
            )

            score = float(
                result.get(
                    "score",
                    result.get(
                        "best_similarity",
                        0,
                    ),
                )
            )

            st.write(
                f"**{rank}. {name}** — "
                f"{timestamp} — "
                f"{score * 100:.2f}%"
            )


# ============================================================
# UI
# ============================================================

st.title(
    "🎬 Scene2Episode"
)

st.caption(
    "Identify a movie or TV episode from a "
    "single scene."
)

st.divider()


# ============================================================
# INPUT MODE
# ============================================================

input_type = st.radio(
    "Choose input type",
    [
        "🖼️ Image",
        "🎥 Video",
    ],
    horizontal=True,
)


# ============================================================
# UPLOAD
# ============================================================

if input_type == "🖼️ Image":

    uploaded_file = st.file_uploader(
        "Upload a screenshot",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

else:

    uploaded_file = st.file_uploader(
        "Upload a short video clip",
        type=[
            "mp4",
            "mkv",
            "avi",
            "mov",
            "webm",
        ],
    )


# ============================================================
# RUN
# ============================================================

if uploaded_file is not None:

    if input_type == "🖼️ Image":

        st.image(
            uploaded_file,
            caption="Query image",
            use_container_width=True,
        )

    else:

        st.video(
            uploaded_file
        )

    st.divider()

    if st.button(
        "🔍 Identify Scene",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Analyzing scene..."
        ):

            try:

                index, metadata, embedder = (
                    load_engine()
                )

                suffix = Path(
                    uploaded_file.name
                ).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = Path(
                        temp_file.name
                    )

                try:

                    if (
                        input_type
                        == "🖼️ Image"
                    ):

                        results = (
                            run_image_inference(
                                temp_path,
                                index,
                                metadata,
                                embedder,
                            )
                        )

                    else:

                        results = (
                            run_video_inference(
                                temp_path,
                                index,
                                metadata,
                                embedder,
                            )
                        )

                    display_results(
                        results
                    )

                finally:

                    if temp_path.exists():

                        temp_path.unlink()

            except Exception as exc:

                st.error(
                    "Something went wrong."
                )

                st.exception(
                    exc
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Scene2Episode • CLIP + FAISS "
    "visual retrieval"
)