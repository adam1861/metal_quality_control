from __future__ import annotations

import base64
import io
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFilter

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

_FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT_NEIGHBORS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu threshold for a uint8 grayscale image."""
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(gray.size)
    if total <= 0:
        return 0

    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_b = 0.0
    w_b = 0.0
    best_thresh = 0
    best_var = -1.0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0.0:
            continue
        w_f = total - w_b
        if w_f == 0.0:
            break
        sum_b += float(t) * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    return int(best_thresh)


def _largest_component(
    mask: np.ndarray, *, connectivity: int = 8, prefer_non_border: bool = True
) -> tuple[np.ndarray, bool]:
    """Return the largest connected component in a boolean mask.

    Returns (component_mask, touches_border).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    h, w = mask.shape
    if h == 0 or w == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool), False

    neighbors = _EIGHT_NEIGHBORS if connectivity == 8 else _FOUR_NEIGHBORS
    visited = np.zeros((h, w), dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    best_touches_border = True

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            touches_border = False

            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                    touches_border = True

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))

            if not best_pixels:
                best_pixels = pixels
                best_touches_border = touches_border
                continue

            if prefer_non_border:
                if best_touches_border and not touches_border:
                    best_pixels = pixels
                    best_touches_border = touches_border
                    continue
                if (not best_touches_border) and touches_border:
                    continue

            if len(pixels) > len(best_pixels):
                best_pixels = pixels
                best_touches_border = touches_border

    out = np.zeros_like(mask, dtype=bool)
    if best_pixels:
        ys, xs = zip(*best_pixels)
        out[np.array(ys, dtype=np.int32), np.array(xs, dtype=np.int32)] = True
    return out, best_touches_border


def _fill_small_holes(mask: np.ndarray, *, max_hole_area: int) -> np.ndarray:
    """Fill enclosed holes up to `max_hole_area` pixels.

    This avoids filling the nut's large center hole while closing small thresholding holes.
    """
    if max_hole_area <= 0:
        return mask

    mask = mask.astype(bool)
    h, w = mask.shape
    if h == 0 or w == 0:
        return mask

    background = ~mask
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    # Seed with border background pixels.
    for x in range(w):
        if background[0, x]:
            exterior[0, x] = True
            q.append((0, x))
        if background[h - 1, x]:
            exterior[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if background[y, 0]:
            exterior[y, 0] = True
            q.append((y, 0))
        if background[y, w - 1]:
            exterior[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        cy, cx = q.popleft()
        for dy, dx in _FOUR_NEIGHBORS:
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if exterior[ny, nx] or not background[ny, nx]:
                continue
            exterior[ny, nx] = True
            q.append((ny, nx))

    holes = background & ~exterior
    if not holes.any():
        return mask

    visited = np.zeros((h, w), dtype=bool)
    out = mask.copy()

    for y in range(h):
        for x in range(w):
            if not holes[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []

            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                for dy, dx in _FOUR_NEIGHBORS:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not holes[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))

            if len(pixels) <= max_hole_area:
                ys, xs = zip(*pixels)
                out[np.array(ys, dtype=np.int32), np.array(xs, dtype=np.int32)] = True

    return out


def segment_nut_mask(
    image: Image.Image,
    *,
    blur_radius: float = 1.0,
    fill_holes_max_area_ratio: float = 0.01,
) -> np.ndarray:
    """Segment the nut (object of interest) from the dark background.

    Returns a boolean mask (True = nut pixel, False = background).
    """
    gray_img = image.convert("L")
    if blur_radius and blur_radius > 0:
        gray_img = gray_img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    gray = np.array(gray_img, dtype=np.uint8)
    thr = _otsu_threshold(gray)

    # Candidate A: pixels brighter than threshold.
    cand_a, a_touches_border = _largest_component(gray > thr, connectivity=8, prefer_non_border=True)
    # Candidate B: inverted threshold (handles cases where background is brighter).
    cand_b, b_touches_border = _largest_component(gray <= thr, connectivity=8, prefer_non_border=True)

    def _score(candidate: np.ndarray, touches_border: bool) -> tuple[int, int]:
        area = int(candidate.sum())
        # Prefer non-border components strongly.
        border_bonus = 1_000_000 if not touches_border else 0
        return border_bonus + area, area

    best = cand_a
    if cand_b.any() and (_score(cand_b, b_touches_border) > _score(cand_a, a_touches_border)):
        best = cand_b

    if not best.any():
        return np.zeros_like(gray, dtype=bool)

    # If the "best" touches the border, it's likely background. Keep it anyway as a fallback.
    max_hole_area = int(best.size * float(fill_holes_max_area_ratio))
    best = _fill_small_holes(best, max_hole_area=max_hole_area)
    return best.astype(bool)


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
