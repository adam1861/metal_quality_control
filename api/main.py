from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from training.inference import (
    CLASS_COLOR_MAP,
    CLASS_ID_TO_NAME,
    load_model,
    mask_to_color,
    pil_to_base64,
    predict_metal_nut_defects,
    segment_nut_mask,
)

IMAGE_SIZE = (256, 256)
WEIGHTS_PATH = Path("models/best_unet_metalnut_multiclass.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REJECT_THRESHOLD_ON_NUT = 0.30

app = FastAPI(title="Metal Nut Defect Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
async def load_model_at_startup():
    global model
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

    nut_mask = segment_nut_mask(original_image)
    nut_pixels = int(nut_mask.sum())
    if nut_pixels == 0:
        nut_mask = np.ones_like(mask, dtype=bool)
        nut_pixels = int(nut_mask.sum())

    defect_mask = mask != 0
    defect_pixels_image = int(defect_mask.sum())
    defect_pixels_on_nut = int(np.logical_and(defect_mask, nut_mask).sum())
    defect_ratio_on_nut = defect_pixels_on_nut / float(nut_pixels) if nut_pixels else 0.0

    total_pixels = float(mask.size)
    class_pixel_percentages_image = {
        CLASS_ID_TO_NAME[idx]: per_class_counts[idx] / total_pixels if total_pixels else 0.0
        for idx in CLASS_ID_TO_NAME
        if idx != 0
    }

    class_pixel_percentages_on_nut = {
        CLASS_ID_TO_NAME[idx]: (int(np.logical_and(mask == idx, nut_mask).sum()) / float(nut_pixels) if nut_pixels else 0.0)
        for idx in CLASS_ID_TO_NAME
        if idx != 0
    }

    response = {
        "is_defective": bool(defect_pixels_on_nut > 0),
        # Main metric for QC: defect coverage on the nut surface only.
        "defect_ratio": defect_ratio_on_nut,
        "defect_ratio_on_nut": defect_ratio_on_nut,
        # Optional metric: defect coverage across the whole image (includes background).
        "defect_ratio_image": float(defect_ratio),
        "class_pixel_percentages": class_pixel_percentages_on_nut,
        "class_pixel_percentages_on_nut": class_pixel_percentages_on_nut,
        "class_pixel_percentages_image": class_pixel_percentages_image,
        "nut_pixel_count": nut_pixels,
        "defect_pixel_count_on_nut": defect_pixels_on_nut,
        "defect_pixel_count_image": defect_pixels_image,
        "reject_threshold_on_nut": REJECT_THRESHOLD_ON_NUT,
        "quality_decision": "REJECT" if defect_ratio_on_nut > REJECT_THRESHOLD_ON_NUT else "ACCEPT",
        "image": {"width": original_image.width, "height": original_image.height},
        "mask_encoded": pil_to_base64(mask_to_color(mask, CLASS_COLOR_MAP)),
        "overlay_image_encoded": pil_to_base64(overlay_image),
    }
    # Pick the defect type with the highest percentage (excluding background).
    if class_pixel_percentages_on_nut:
        top_cls = max(class_pixel_percentages_on_nut.items(), key=lambda kv: kv[1])
        response["dominant_defect"] = top_cls[0] if top_cls[1] > 0 else "none"
        response["dominant_defect_ratio"] = top_cls[1]
    else:
        response["dominant_defect"] = "none"
        response["dominant_defect_ratio"] = 0.0
    return response
