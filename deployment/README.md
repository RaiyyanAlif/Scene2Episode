# Scene2Episode Deployment

## Requirements

- Python 3.10+
- FFmpeg
- FFprobe
- Internet access on first run to download the CLIP model

## Install Python Dependencies

    pip install -r requirements.txt

## Install FFmpeg on Debian/Ubuntu

    bash scripts/install_ffmpeg.sh

## Run the Application

    streamlit run app.py

## Technologies

- CLIP (openai/clip-vit-base-patch32) for visual embeddings
- FAISS for scene retrieval
- FFmpeg/FFprobe for short-video frame extraction
- Streamlit for the web interface

## Required Runtime Files

    deployment/
    |-- app.py
    |-- requirements.txt
    |-- indexes/
    |   |-- scene2episode.index
    |   |-- scene2episode_metadata.json
    |-- scripts/
    |   |-- install_ffmpeg.sh
    |-- src/
        |-- inference.py
        |-- video_inference.py
        |-- result_formatter.py
        |-- ocr_inference.py

## Runtime Notes

The deployed application requires the FAISS index and metadata files.

Raw videos, extracted frames, and training embeddings are not required by the deployed inference application.

The CLIP model is downloaded automatically from Hugging Face on first use if it is not already cached.
