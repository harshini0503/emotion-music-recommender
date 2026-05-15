import io
from pathlib import Path
from typing import Dict, Any, List
import cv2

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import timm
import joblib
import xgboost as xgb

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware


# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

FER_WEIGHTS = MODELS_DIR / "fer_model.pt"
MER_XGB = MODELS_DIR / "mer_xgb.json"
MER_META = MODELS_DIR / "mer_meta.joblib"
SPOTIFY_CSV = DATA_DIR / "spotify-new.csv"


# -------------------------
# Labels / mapping
# -------------------------
FER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_TO_MOOD = {
    "happy": "Happy",
    "surprise": "Energetic",
    "neutral": "Calm",
    "sad": "Sad",
    "fear": "Calm",
    "angry": "Energetic",
    "disgust": "Sad",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -------------------------
# Load music model
# -------------------------
meta = joblib.load(MER_META)
MER_FEATURES: List[str] = meta["features"]
MER_CLASSES: List[str] = meta["classes"]

booster = xgb.Booster()
booster.load_model(str(MER_XGB))

songs_df = pd.read_csv(SPOTIFY_CSV)
songs_df.columns = [c.strip().lower() for c in songs_df.columns]

rename_map = {
    "track_name": "name",
    "artist(s)_name": "artist",
    "bpm": "tempo",
    "danceability_%": "danceability",
    "valence_%": "valence",
    "energy_%": "energy",
    "acousticness_%": "acousticness",
    "instrumentalness_%": "instrumentalness",
    "liveness_%": "liveness",
    "speechiness_%": "speechiness",
}

songs_df = songs_df.rename(columns=rename_map)


def _maybe_div100(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns and df[c].dropna().max() > 1.5:
            df[c] = df[c] / 100.0


_maybe_div100(
    songs_df,
    [
        "danceability",
        "valence",
        "energy",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
    ],
)


# -------------------------
# Load FER model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

FER_ARCH = "convnext_tiny"
FER_IMG_SIZE = 128

fer_model = timm.create_model(
    FER_ARCH,
    pretrained=False,
    num_classes=len(FER_CLASSES),
)

if FER_WEIGHTS.exists():
    state = torch.load(FER_WEIGHTS, map_location=device)
    fer_model.load_state_dict(state)
    print(f"✅ Loaded FER weights from {FER_WEIGHTS}")
else:
    print("⚠️ FER weights not found yet. Place models/fer_model.pt")

fer_model.to(device)
fer_model.eval()


# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(pil_img: Image.Image, size: int = 128) -> torch.Tensor:
    img_rgb = np.array(pil_img.convert("RGB"))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        pad = int(0.20 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_rgb.shape[1], x + w + pad)
        y2 = min(img_rgb.shape[0], y + h + pad)

        img_rgb = img_rgb[y1:y2, x1:x2]

    img = Image.fromarray(img_rgb).resize((size, size))

    x = np.array(img).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))

    return torch.from_numpy(x).unsqueeze(0).to(device)



# -------------------------
# Emotion prediction
# -------------------------
def predict_emotion(pil_img: Image.Image) -> Dict[str, Any]:
    if not FER_WEIGHTS.exists():
        return {
            "emotion": "neutral",
            "probs": {c: (1.0 if c == "neutral" else 0.0) for c in FER_CLASSES},
        }

    x = preprocess_image(pil_img, FER_IMG_SIZE)

    with torch.no_grad():
        logits = fer_model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    raw_emotion = FER_CLASSES[idx]
    confidence = float(np.max(probs))

    # Lower threshold for demo. If model is very unsure, fallback to neutral.
    emotion = raw_emotion if confidence >= 0.35 else "neutral"

    return {
        "emotion": emotion,
        "raw_emotion": raw_emotion,
        "confidence": confidence,
        "probs": {FER_CLASSES[i]: float(probs[i]) for i in range(len(FER_CLASSES))},
    }

# -------------------------
# Spotify link helper
# -------------------------
def spotify_search_url(track: str, artist: str) -> str:

    import urllib.parse

    q = f"{track} {artist}".strip()

    return "https://open.spotify.com/search/" + urllib.parse.quote(q)


# -------------------------
# Recommendation engine
# -------------------------
def recommend_songs(target_mood: str, top_k: int = 10) -> List[Dict[str, Any]]:

    df = songs_df.copy()

    key_map = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    if "key" in df.columns and df["key"].dtype == object:
        df["key"] = df["key"].astype(str).str.strip().map(key_map)

    if "mode" in df.columns and df["mode"].dtype == object:
        df["mode"] = df["mode"].astype(str).str.strip().map({"Major": 1, "Minor": 0})

    feats = df[MER_FEATURES].dropna()

    dmat = xgb.DMatrix(feats.values.astype(np.float32), feature_names=MER_FEATURES)

    probs = booster.predict(dmat)

    mood_to_idx = {c: i for i, c in enumerate(MER_CLASSES)}

    target_idx = mood_to_idx.get(target_mood, int(np.argmax(probs.mean(axis=0))))

    scores = probs[:, target_idx]

    out = df.loc[feats.index, ["name", "artist"]].copy()
    out["score"] = scores

    out = out.sort_values("score", ascending=False)

    N = min(200, len(out))
    cand = out.head(N)

    recs = []
    artist_count = {}

    for _, r in cand.iterrows():

        track = str(r.get("name", ""))
        artist = str(r.get("artist", ""))

        if not track or not artist:
            continue

        artist_count.setdefault(artist, 0)

        if artist_count[artist] >= 2:
            continue

        artist_count[artist] += 1

        recs.append(
            {
                "track": track,
                "artist": artist,
                "score": float(r["score"]),
                "spotify_url": spotify_search_url(track, artist),
            }
        )

        if len(recs) >= top_k:
            break

    return recs


# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="Emotion-Based Music Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Backend is running. Use /docs or POST /predict."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...), mood_override: str = Form(default="")):

    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes))

    fer = predict_emotion(pil_img)

    emotion = fer["emotion"]

    mood = EMOTION_TO_MOOD.get(emotion, "Calm")

    if mood_override and mood_override.strip():
        mood = mood_override.strip()

    recs = recommend_songs(mood, top_k=10)

    return {
        "emotion": emotion,
        "emotion_probs": fer["probs"],
        "target_mood": mood,
        "recommendations": recs,
    }