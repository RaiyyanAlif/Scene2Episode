import os
import re
import json
import pickle
import faiss
import numpy as np
import cv2
import pytesseract

from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = "indexes/scene2episode.index"
METADATA_PATH = "indexes/scene2episode_metadata.json"
OCR_CACHE_PATH = "metadata/ocr_cache.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K = 100

# Query clip
OFFSETS = [-4, -2, 0, 2, 4]

# Normal retrieval weights
NORMAL_CLIP_WEIGHT = 0.80
NORMAL_OCR_WEIGHT = 0.20

# Credit-aware weights
CREDIT_CLIP_WEIGHT = 0.30
CREDIT_OCR_WEIGHT = 0.70

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# ============================================================
# LOADERS
# ============================================================

def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ocr_cache():
    if not os.path.exists(OCR_CACHE_PATH):
        return {}

    with open(OCR_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ocr_cache(cache):
    os.makedirs(os.path.dirname(OCR_CACHE_PATH), exist_ok=True)

    with open(OCR_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ============================================================
# OCR
# ============================================================

def normalize_text(text):
    text = text.upper()

    text = re.sub(r"[^A-Z0-9À-ÖØ-Ý\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def ocr_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # upscale
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    variants = []

    variants.append(gray)

    # adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants.append(adaptive)

    # OTSU
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(otsu)

    texts = []

    for img in variants:

        text = pytesseract.image_to_string(
            img,
            config="--oem 3 --psm 6"
        )

        text = normalize_text(text)

        if text:
            texts.append(text)

    if not texts:
        return ""

    # longest OCR output usually contains the most information
    return max(texts, key=len)


def get_ocr(image_path, cache):
    key = os.path.abspath(image_path)

    if key in cache:
        return cache[key]

    text = ocr_image(image_path)

    cache[key] = text

    return text


# ============================================================
# TEXT SIMILARITY
# ============================================================

def tokenize(text):
    if not text:
        return set()

    return set(text.upper().split())


def text_similarity(a, b):

    A = tokenize(a)
    B = tokenize(b)

    if not A or not B:
        return 0.0

    intersection = len(A & B)
    union = len(A | B)

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# CREDIT DETECTION
# ============================================================

CREDIT_KEYWORDS = [
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


def credit_score(text):

    if not text:
        return 0.0

    text_upper = text.upper()

    hits = 0

    for keyword in CREDIT_KEYWORDS:

        if keyword in text_upper:
            hits += 1

    # Number of uppercase/name-like tokens also helps.
    words = text_upper.split()

    long_word_count = sum(
        1
        for word in words
        if len(word) >= 4
    )

    score = 0.0

    score += min(hits / 3.0, 1.0) * 0.70

    if long_word_count >= 10:
        score += 0.30

    return min(score, 1.0)


def is_credit_frame(text):

    return credit_score(text) >= 0.35


# ============================================================
# CLIP
# ============================================================

print("=" * 100)
print("SCENE2EPISODE CREDIT-AWARE RETRIEVAL")
print("=" * 100)

print("\nLoading FAISS...")

index = faiss.read_index(INDEX_PATH)

print("Vectors:", index.ntotal)

print("\nLoading metadata...")

metadata = load_metadata()

print("Metadata:", len(metadata))

print("\nLoading OCR cache...")

ocr_cache = load_ocr_cache()

print("Cached OCR:", len(ocr_cache))

print("\nLoading CLIP...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.eval()

print("CLIP loaded.")


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def image_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    pooled = outputs.pooler_output

    features = model.visual_projection(
        pooled
    )

    features = features / features.norm(
        dim=-1,
        keepdim=True
    )

    return features.detach().numpy().astype(
        "float32"
    )


# ============================================================
# HELPERS
# ============================================================

def format_time(seconds):

    seconds = float(seconds)

    h = int(seconds // 3600)

    m = int((seconds % 3600) // 60)

    s = seconds % 60

    return f"{h:02d}:{m:05.2f}"


def find_nearest_frame(media_id, timestamp):

    best_idx = None
    best_diff = float("inf")

    for i, item in enumerate(metadata):

        if item["media_id"] != media_id:
            continue

        diff = abs(
            float(item["timestamp_seconds"])
            - timestamp
        )

        if diff < best_diff:

            best_diff = diff
            best_idx = i

    return best_idx


# ============================================================
# QUERY
# ============================================================

QUERY_MEDIA_ID = "stranger_things_s03_e01"

QUERY_TIME = 2990.0

print("\n" + "=" * 100)
print("QUERY")
print("=" * 100)

query_center_idx = find_nearest_frame(
    QUERY_MEDIA_ID,
    QUERY_TIME
)

if query_center_idx is None:

    raise RuntimeError(
        "Query frame not found."
    )

query_center = metadata[
    query_center_idx
]

print(
    "Episode :",
    query_center["title"],
    "S",
    query_center["season"],
    "E",
    query_center["episode"]
)

print(
    "Time    :",
    query_center["timestamp_formatted"]
)

print(
    "Frame   :",
    query_center["frame_id"]
)


# ============================================================
# BUILD QUERY CLIP
# ============================================================

query_indices = []

for offset in OFFSETS:

    idx = find_nearest_frame(
        QUERY_MEDIA_ID,
        QUERY_TIME + offset
    )

    if idx is not None:

        query_indices.append(idx)


print("\nQuery frames:")

query_texts = []

query_embeddings = []

for idx in query_indices:

    item = metadata[idx]

    text = get_ocr(
        item["image_path"],
        ocr_cache
    )

    emb = image_embedding(
        item["image_path"]
    )

    query_texts.append(text)
    query_embeddings.append(emb)

    print(
        item["timestamp_formatted"],
        "|",
        item["frame_id"],
        "| credit:",
        round(credit_score(text), 3)
    )

save_ocr_cache(ocr_cache)


# ============================================================
# QUERY CREDIT SIGNAL
# ============================================================

query_credit_scores = [
    credit_score(t)
    for t in query_texts
]

query_credit_score = max(
    query_credit_scores
)

query_is_credit = (
    query_credit_score >= 0.35
)

print(
    "\nQuery credit score:",
    round(query_credit_score, 4)
)

print(
    "Credit-aware mode:",
    query_is_credit
)


# ============================================================
# MULTI-FRAME FAISS RETRIEVAL
# ============================================================

all_candidates = {}

for emb in query_embeddings:

    distances, indices = index.search(
        emb,
        TOP_K
    )

    for similarity, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = metadata[idx]

        key = idx

        if key not in all_candidates:

            all_candidates[key] = {
                "idx": idx,
                "clip_scores": []
            }

        all_candidates[key][
            "clip_scores"
        ].append(
            float(similarity)
        )


# ============================================================
# SCORE CANDIDATES
# ============================================================

episode_results = {}


for candidate in all_candidates.values():

    idx = candidate["idx"]

    item = metadata[idx]

    clip_scores = candidate[
        "clip_scores"
    ]

    best_clip = max(
        clip_scores
    )

    mean_clip = np.mean(
        clip_scores
    )

    # OCR
    candidate_text = get_ocr(
        item["image_path"],
        ocr_cache
    )

    save_ocr_cache(ocr_cache)

    candidate_credit = credit_score(
        candidate_text
    )

    # Compare candidate OCR against
    # ALL query OCR frames.
    ocr_scores = []

    for query_text in query_texts:

        score = text_similarity(
            query_text,
            candidate_text
        )

        ocr_scores.append(score)

    best_ocr = max(
        ocr_scores
    )

    mean_ocr = np.mean(
        ocr_scores
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    if query_is_credit:

        final_score = (
            CREDIT_CLIP_WEIGHT * best_clip
            +
            CREDIT_OCR_WEIGHT * best_ocr
        )

        # Candidate should also look like a credit frame.
        if candidate_credit < 0.25:

            final_score *= 0.90

    else:

        final_score = (
            NORMAL_CLIP_WEIGHT * best_clip
            +
            NORMAL_OCR_WEIGHT * best_ocr
        )

    episode_key = (
        item["media_id"]
    )

    if episode_key not in episode_results:

        episode_results[episode_key] = {
            "media_id": item["media_id"],
            "title": item["title"],
            "season": item["season"],
            "episode": item["episode"],
            "best_score": -1,
            "best_item": None,
            "best_clip": 0,
            "best_ocr": 0,
            "best_credit": 0,
            "count": 0
        }

    result = episode_results[
        episode_key
    ]

    result["count"] += 1

    if final_score > result["best_score"]:

        result["best_score"] = final_score

        result["best_item"] = item

        result["best_clip"] = best_clip

        result["best_ocr"] = best_ocr

        result["best_credit"] = candidate_credit


# ============================================================
# SAVE CACHE
# ============================================================

save_ocr_cache(
    ocr_cache
)


# ============================================================
# SORT
# ============================================================

ranked = sorted(
    episode_results.values(),
    key=lambda x: x["best_score"],
    reverse=True
)


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 100)
print("CREDIT-AWARE EPISODE RANKING")
print("=" * 100)

for rank, result in enumerate(
    ranked[:15],
    start=1
):

    item = result["best_item"]

    print("\n" + "-" * 100)

    print(
        f"RANK {rank}: "
        f"{result['title']} "
        f"S{result['season']:02d} "
        f"E{result['episode']:02d}"
    )

    print(
        "Final score :",
        round(
            result["best_score"],
            4
        )
    )

    print(
        "Best CLIP   :",
        round(
            result["best_clip"],
            4
        )
    )

    print(
        "Best OCR    :",
        round(
            result["best_ocr"],
            4
        )
    )

    print(
        "Credit score:",
        round(
            result["best_credit"],
            4
        )
    )

    print(
        "Evidence    :",
        result["count"]
    )

    if item:

        print(
            "Timestamp   :",
            item["timestamp_formatted"]
        )

        print(
            "Frame       :",
            item["frame_id"]
        )

        print(
            "Image       :",
            item["image_path"]
        )

print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)