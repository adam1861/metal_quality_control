from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _list_images(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def preprocess_image(
    image: Image.Image,
    image_size: tuple[int, int] | None = (256, 256),
    normalize: bool = True,
) -> torch.Tensor:
    """Resize and normalize an image the same way as during training."""
    if image_size:
        image = TF.resize(image, image_size)
    tensor = TF.to_tensor(image)
    if normalize:
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


class MetalNutSegmentationDataset(Dataset):
    """Loads image/mask pairs for the metal_nut segmentation task."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: tuple[int, int] | None = (256, 256),
        normalize: bool = True,
        augment: bool = False,
        seed: int | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.rng = random.Random(seed)
        self.image_paths = _list_images(self.images_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name} at {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            if self.rng.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if self.rng.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Rotate by right angles to preserve pixel labels cleanly.
            angle = self.rng.choice([0, 90, 180, 270])
            if angle:
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if self.image_size:
            image = TF.resize(image, self.image_size)
            mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)

        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor, str(img_path.name)
