from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image

from training.data_utils import preprocess_image
from training.model import UNet

CLASS_ID_TO_NAME = {
    0: "background",
    1: "bent",
    2: "color",
    3: "flip",
    4: "scratch",
}

CLASS_COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 200, 0),
    3: (0, 120, 255),
    4: (255, 215, 0),
}


def load_model(weights_path: Path, device: torch.device, num_classes: int = 5) -> UNet:
    model = UNet(in_channels=3, num_classes=num_classes, base_channels=32)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _load_image(image_input: Union[str, Path, Image.Image, bytes]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    raise TypeError("Unsupported image input type.")


def mask_to_color(mask: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]) -> Image.Image:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        color_mask[mask == cls_id] = color
    return Image.fromarray(color_mask)


def pil_to_base64(image: Image.Image, format: str = "PNG", add_prefix: bool = True) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if add_prefix:
        return f"data:image/{format.lower()};base64,{encoded}"
    return encoded


def predict_metal_nut_defects(
    model: UNet,
    image_input: Union[str, Path, Image.Image, bytes],
    device: torch.device,
    image_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, Image.Image, Image.Image, Dict[int, int], float]:
    """Run inference on a single image and return predictions and visuals."""
    original_image = _load_image(image_input)
    tensor = preprocess_image(original_image, image_size=image_size, normalize=True).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize mask back to original image size for visualization/stats.
    resized_mask = np.array(
        Image.fromarray(preds, mode="L").resize(original_image.size, resample=Image.NEAREST),
        dtype=np.uint8,
    )

    per_class_pixel_counts = {cls: int((resized_mask == cls).sum()) for cls in CLASS_ID_TO_NAME.keys()}
    defect_pixels = int((resized_mask != 0).sum())
    defect_ratio = defect_pixels / float(resized_mask.size)

    overlay_layer = np.zeros((resized_mask.shape[0], resized_mask.shape[1], 4), dtype=np.uint8)
    for cls_id, color in CLASS_COLOR_MAP.items():
        if cls_id == 0:
            continue
        overlay_layer[resized_mask == cls_id] = (*color, 120)
    overlay_image = Image.alpha_composite(
        original_image.convert("RGBA"), Image.fromarray(overlay_layer, mode="RGBA")
    )

    return resized_mask, original_image, overlay_image, per_class_pixel_counts, defect_ratio
