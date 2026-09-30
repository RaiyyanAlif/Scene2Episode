# 🎬 Scene2Episode

> **Identify a movie or TV episode from a single scene — using only visual information.**

Scene2Episode is an AI-powered visual retrieval system that can identify an unknown movie or television episode from a **single image** or a **short 1–3 second video clip**.

The system uses **CLIP image embeddings + FAISS similarity search** to compare the input scene against a pre-indexed video-frame database and retrieve the most visually similar scenes.

---

## ✨ Features

- 🖼️ Identify movies and TV episodes from a single image
- 🎥 Identify scenes from short 1–3 second video clips
- 📺 Identify:
  - Movie title
  - TV series
  - Season
  - Episode
- ⏱️ Estimate the approximate timestamp of the matching scene
- 🔎 Return alternative matching episodes
- ⚡ Fast vector similarity search using FAISS
- 🤖 CLIP-based visual understanding
- 🖥️ Streamlit web interface
- 💻 CPU-compatible
- 📊 Retrieval evaluation and benchmarking tools

---

## 🧠 How It Works

The overall pipeline is:

```text
                 ┌──────────────────────┐
                 │   Image / Video      │
                 │       Input          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Frame Extraction   │
                 │  (for video input)   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    CLIP Encoder      │
                 │  Visual Embedding    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     FAISS Index      │
                 │   Similarity Search  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Episode Aggregation  │
                 │   & Re-ranking       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Final Result     │
                 │                      │
                 │ Title                │
                 │ Season / Episode     │
                 │ Timestamp            │
                 │ Confidence           │
                 └──────────────────────┘
```

---

## 🔬 Core Technology

### CLIP

Scene2Episode uses:

```text
openai/clip-vit-base-patch32
```

CLIP converts video frames into **512-dimensional visual embeddings**.

Each embedding represents the visual characteristics of a scene.

### FAISS

The generated embeddings are stored in a FAISS vector index.

The current index uses:

```text
IndexFlatIP
```

with:

```text
Dimension: 512
Vectors:   54,523
```

The embeddings are normalized before similarity search, allowing inner-product similarity to act as cosine similarity.

---

## 📂 Dataset

The current dataset contains **34 video items** consisting of movies, animated movies, and TV episodes.

### 🎬 Movies

- Predestination
- Chemical Heart

### 🧸 Animated

- Tangled
- Kung Fu Panda 4

### 📺 TV Series

#### Lucifer

- Season 1
- Episodes 1–13

#### Stranger Things

- Season 2
- Episodes 1–9
- Season 3
- Episodes 1–8

---

## 🎞️ Frame Extraction

Frames are extracted from the source videos at approximately:

```text
1 frame / 2 seconds
```

The extracted frames are stored under:

```text
data/frames/
```

Example:

```text
data/frames/
└── series/
    └── Lucifer/
        └── S01/
            └── E01/
                ├── 000001.jpg
                ├── 000002.jpg
                ├── 000003.jpg
                └── ...
```

Frame metadata is stored under:

```text
metadata/frames/
```

---

## 📊 Current Dataset Statistics

```text
Video items       : 34
Total frames      : 54,523
Embedding size    : 512 dimensions
FAISS vectors     : 54,523
Embedding errors  : 0
```

All indexed embeddings have been verified successfully.

---

# 📈 Retrieval Evaluation

A fair retrieval evaluation was performed using random query frames while excluding the exact query frame from retrieval.

Current benchmark:

```text
Top-1 Episode Accuracy : 94%
Top-3 Episode Accuracy : 94%
Top-5 Episode Accuracy : 96%

Top-1 Media Accuracy   : 94%
Top-3 Media Accuracy   : 94%
Top-5 Media Accuracy   : 96%

Median Timestamp Error : 24 seconds
```

The evaluation excludes the exact query frame to avoid artificially inflating the accuracy through self-matching.

The current V1 system is therefore frozen at:

> **94% Top-1 episode accuracy and 96% Top-5 episode accuracy on the current benchmark.**

Some difficult scenes remain challenging, particularly visually similar scenes within the same television series.

---

# 🧩 Project Structure

```text
Scene2Episode/
│
├── app.py
├── config.yaml
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│       ├── animated/
│       ├── movies/
│       └── series/
│
├── embeddings/
│
├── indexes/
│   ├── scene2episode.index
│   └── scene2episode_metadata.json
│
├── metadata/
│
└── src/
    │
    ├── inference.py
    ├── video_inference.py
    ├── ocr_inference.py
    ├── result_formatter.py
    │
    ├── embedding/
    │   ├── benchmark_embeddings.py
    │   └── build_embeddings.py
    │
    ├── extraction/
    │   ├── build_frame_metadata.py
    │   └── frame_extractor.py
    │
    ├── ingestion/
    │   └── scanner.py
    │
    ├── retrieval/
    │   ├── build_episode_fingerprints.py
    │   ├── build_episode_prototypes.py
    │   ├── build_faiss.py
    │   ├── build_global_faiss.py
    │   ├── build_subtitle_index.py
    │   ├── candidate_ocr_diagnostic.py
    │   ├── credit_aware_retrieval.py
    │   ├── episode_rerank.py
    │   ├── episode_sequence_reranker.py
    │   ├── episode_temporal_reranker_v2.py
    │   ├── fingerprint_reranker.py
    │   ├── multicandidate_subtitle_reranker.py
    │   ├── ocr_diagnostic.py
    │   ├── ocr_reranker.py
    │   ├── query_global.py
    │   ├── query_image.py
    │   ├── sequence_matcher.py
    │   ├── subtitle_matcher.py
    │   ├── temporal_clip_retrieval.py
    │   ├── temporal_ocr_retrieval.py
    │   ├── temporal_subtitle_matcher.py
    │   ├── temporal_verify.py
    │   └── verify_embeddings.py
    │
    └── evaluation/
        ├── evaluate_credit_aware.py
        ├── evaluate_episode_prototypes.py
        ├── evaluate_fingerprint.py
        ├── evaluate_lucifer_visual_subtitle.py
        ├── evaluate_retrieval.py
        ├── evaluate_subtitle_matcher.py
        ├── evaluate_temporal_subtitles.py
        ├── evaluate_temporal_v2.py
        ├── evaluate_visual_plus_subtitle.py
        └── inspect_failures.py
```

> `__pycache__` directories are intentionally omitted because they contain Python-generated cache files.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Scene2Episode
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Install FFmpeg

FFmpeg is required for video frame extraction.

Verify the installation:

```powershell
ffmpeg -version
```

---

# 🚀 Running Scene2Episode

## 🖼️ Image Input

Run:

```powershell
python src/inference.py path\to\image.jpg
```

The system searches the indexed video-frame database and returns the most likely matching scene.

Example:

```text
╔══════════════════════════════════════════════╗
║              SCENE2EPISODE                  ║
╠══════════════════════════════════════════════╣
║ Title      : Stranger_Things                ║
║ Season     : 3                              ║
║ Episode    : 8                              ║
║ Timestamp  : 00:36:14.00                   ║
║ Confidence : 87.90% (High)                 ║
╚══════════════════════════════════════════════╝
```

---

# 🎥 Video Input

Scene2Episode can process a short video clip.

Run:

```powershell
python src/video_inference.py path\to\clip.mp4
```

The video pipeline is:

```text
Short Video
     │
     ▼
Extract Multiple Frames
     │
     ▼
CLIP Embeddings
     │
     ▼
FAISS Search
     │
     ▼
Episode Evidence Aggregation
     │
     ▼
Final Episode Prediction
```

Multiple frames provide additional visual evidence compared with relying on a single frame.

---

# 🖥️ Streamlit Web Application

Scene2Episode includes a web interface built with Streamlit.

Start the application:

```powershell
streamlit run app.py
```

The interface supports both image and video input.

### Image

```text
Upload Image
     ↓
Scene2Episode
     ↓
Visual Retrieval
     ↓
Movie / Series / Episode / Timestamp
```

### Video

```text
Upload 1–3 Second Clip
     ↓
Frame Extraction
     ↓
Visual Retrieval
     ↓
Episode Prediction
```

---

# 🏗️ System Architecture

```text
                     INPUT
                       │
              ┌────────┴────────┐
              │                 │
            IMAGE             VIDEO
              │                 │
              │          Frame Extraction
              │                 │
              └────────┬────────┘
                       │
                       ▼
                CLIP Embedding
                       │
                       ▼
                 512-D Vector
                       │
                       ▼
                FAISS Search
                       │
                       ▼
              Top-K Frame Matches
                       │
                       ▼
             Episode-level Grouping
                       │
                       ▼
              Candidate Re-ranking
                       │
                       ▼
                 FINAL RESULT
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Title         Episode       Timestamp
```

---

# 🔎 Retrieval Strategy

For image retrieval, the system searches the FAISS index and obtains visually similar frames.

Retrieved frames belonging to the same episode are grouped together.

The episode score is calculated using:

```text
Episode Score =
    0.60 × Best Similarity
  + 0.25 × Top-3 Similarity
  + 0.15 × Top-5 Similarity
```

This allows the system to consider multiple pieces of evidence rather than relying only on a single retrieved frame.

---

# 🎬 Video Retrieval Strategy

For video input, multiple frames are extracted from the clip.

Each frame is independently embedded and searched against the FAISS index.

The results are then aggregated using episode-level evidence:

```text
Episode Score =
    0.50 × Best Similarity
  + 0.30 × Top-3 Evidence
  + 0.20 × Coverage
```

This helps determine which episode consistently matches the visual content across the input clip.

---

# 🧪 Evaluation

The project includes several evaluation and experimental retrieval scripts.

The evaluation pipeline measures:

- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- Media accuracy
- Timestamp error
- Median timestamp error
- Failure cases

The primary retrieval evaluation avoids exact self-matches so that the model cannot obtain an artificially perfect score by retrieving the query frame itself.

---

# ⚠️ Known Limitations

Scene2Episode is currently a **visual retrieval system**.

Therefore, visually similar scenes can sometimes cause confusion.

Examples include:

- Similar scenes between different episodes
- Recurring locations
- Similar character shots
- Opening and ending credits
- Similar title-card frames
- Scenes containing visually similar objects

The current V1 system intentionally focuses on the visual retrieval pipeline.

Future versions may incorporate additional modalities such as:

```text
OCR
Audio
Subtitles
Speech
Multimodal reasoning
Temporal scene understanding
```

---

# 🛣️ Roadmap

## V1 — Current

- [x] Video ingestion
- [x] Frame extraction
- [x] CLIP embeddings
- [x] FAISS indexing
- [x] Image retrieval
- [x] Video retrieval
- [x] Episode aggregation
- [x] Timestamp estimation
- [x] Evaluation pipeline
- [x] Streamlit interface

## V2 — Future

- [ ] Improved temporal verification
- [ ] OCR-assisted retrieval
- [ ] Subtitle and audio integration
- [ ] Multimodal scene understanding
- [ ] Larger movie/TV database
- [ ] Improved timestamp precision
- [ ] Mobile application
- [ ] Cloud/API deployment

---

# 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| PyTorch | Deep learning |
| Transformers | CLIP model |
| OpenAI CLIP | Visual embeddings |
| FAISS | Vector similarity search |
| NumPy | Numerical processing |
| OpenCV | Image/video processing |
| FFmpeg | Video processing |
| Streamlit | Web interface |
| Tesseract OCR | Experimental OCR support |

---

# 💻 Hardware

The current system has been tested using CPU-based inference.

```text
GPU: Not required
Device: CPU
CPU threads: 10
```

The system can therefore run without an NVIDIA CUDA GPU, although a GPU can significantly improve embedding-generation speed for larger datasets.

---

# 📌 Example Use Case

Suppose you remember a scene from a TV show but don't remember:

- The show's name
- The season
- The episode
- The exact timestamp

You can simply provide a screenshot:

```text
Screenshot
    ↓
Scene2Episode
    ↓
Visual Search
    ↓
Stranger Things
Season 3
Episode 8
00:36:14
```

This is the core idea behind Scene2Episode.

---

# 👨‍💻 Author

**Khalid Hasan**

B.Sc. Engineering in Information and Communication Technology  
Mawlana Bhashani Science and Technology University (MBSTU)

### Interests

- Artificial Intelligence
- Computer Vision
- Machine Learning
- Robotics
- Software Development
- Research

---

# ⭐ Project

**Scene2Episode**

> *Find the story behind the scene.*