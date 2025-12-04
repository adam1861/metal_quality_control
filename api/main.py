from __future__ import annotations

import io
from pathlib import Path

import hashlib
import secrets
import sqlite3
import time
from contextlib import contextmanager

import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from PIL import Image

from training.inference import (
    CLASS_COLOR_MAP,
    CLASS_ID_TO_NAME,
    load_model,
    mask_to_color,
    pil_to_base64,
    predict_metal_nut_defects,
)

IMAGE_SIZE = (256, 256)
WEIGHTS_PATH = Path("models/best_unet_metalnut_multiclass.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTH_DB = Path("data/auth.db")

app = FastAPI(title="Metal Nut Defect Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@contextmanager
def get_db():
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, int(time.time())),
        )
    return token


def get_user_by_token(token: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT users.id, users.email FROM tokens JOIN users ON tokens.user_id = users.id WHERE tokens.token = ?",
            (token,),
        ).fetchone()
    if not row:
        return None
    return {"id": row[0], "email": row[1]}


class AuthRequest(BaseModel):
    email: EmailStr
    password: str


async def get_current_user(authorization: str | None = None):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ", 1)[1].strip()
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


@app.on_event("startup")
async def load_model_at_startup():
    global model
    init_db()
    try:
        model = load_model(WEIGHTS_PATH, DEVICE, num_classes=5)
        print(f"Model loaded from {WEIGHTS_PATH} on {DEVICE}")
    except FileNotFoundError:
        model = None
        print(
            f"Warning: weights not found at {WEIGHTS_PATH}. "
            "Train the model and place the weights to enable predictions."
        )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(DEVICE)}


@app.post("/auth/signup")
async def signup(payload: AuthRequest):
    with get_db() as conn:
        exists = conn.execute("SELECT 1 FROM users WHERE email = ?", (payload.email,)).fetchone()
        if exists:
            raise HTTPException(status_code=400, detail="User already exists")
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (payload.email, hash_password(payload.password), int(time.time())),
        )
        user_id = conn.execute("SELECT id FROM users WHERE email = ?", (payload.email,)).fetchone()[0]
    token = create_token(user_id)
    return {"token": token, "email": payload.email}


@app.post("/auth/login")
async def login(payload: AuthRequest):
    with get_db() as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE email = ?", (payload.email,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user_id, password_hash = row
    if hash_password(payload.password) != password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user_id)
    return {"token": token, "email": payload.email}


@app.get("/auth/me")
async def me(user=Depends(get_current_user)):
    return {"email": user["email"]}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and save weights first.")

    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    mask, original_image, overlay_image, per_class_counts, defect_ratio = predict_metal_nut_defects(
        model=model,
        image_input=pil_image,
        device=DEVICE,
        image_size=IMAGE_SIZE,
    )

    total_pixels = float(mask.size)
    class_pixel_percentages = {
        CLASS_ID_TO_NAME[idx]: per_class_counts[idx] / total_pixels if total_pixels else 0.0
        for idx in CLASS_ID_TO_NAME
        if idx != 0
    }

    response = {
        "is_defective": bool(defect_ratio > 0.0),
        "defect_ratio": defect_ratio,
        "class_pixel_percentages": class_pixel_percentages,
        "image": {"width": original_image.width, "height": original_image.height},
        "mask_encoded": pil_to_base64(mask_to_color(mask, CLASS_COLOR_MAP)),
        "overlay_image_encoded": pil_to_base64(overlay_image),
    }
    # Pick the defect type with the highest percentage (excluding background).
    if class_pixel_percentages:
        top_cls = max(class_pixel_percentages.items(), key=lambda kv: kv[1])
        response["dominant_defect"] = top_cls[0] if top_cls[1] > 0 else "none"
        response["dominant_defect_ratio"] = top_cls[1]
    else:
        response["dominant_defect"] = "none"
        response["dominant_defect_ratio"] = 0.0
    return response
