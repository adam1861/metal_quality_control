from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import stat

DEFECT_CLASS_MAP: Dict[str, int] = {
    "color": 1,
    "scratch": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MVTec metal_nut for multi-class segmentation."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/metal_nut"),
        help="Path to the raw MVTec metal_nut folder (with train/test/ground_truth).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/metal_nut"),
        help="Output directory for processed images and masks.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction of test images per class assigned to the train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of test images per class assigned to the val split (rest go to test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete the existing processed directory before writing.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> Dict[str, Dict[str, Path]]:
    splits = {}
    for split in ("train", "val", "test"):
        images_dir = base / "images" / split
        masks_dir = base / "masks" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = {"images": images_dir, "masks": masks_dir}
    return splits


def find_ground_truth_mask(gt_root: Path, defect_type: str, image_stem: str) -> Path:
    gt_dir = gt_root / defect_type
    candidates = []
    exts = ["png", "bmp", "jpg", "jpeg"]
    for ext in exts:
        candidates.append(gt_dir / f"{image_stem}_mask.{ext}")
        candidates.append(gt_dir / f"{image_stem}.{ext}")
    for cand in candidates:
        if cand.exists():
            return cand
    matches = list(gt_dir.glob(f"{image_stem}*"))
    if not matches:
        raise FileNotFoundError(f"Ground truth mask for {image_stem} not found in {gt_dir}")
    return matches[0]


def create_mask(image_path: Path, defect_type: str, gt_root: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        width, height = img.size
    mask_array = np.zeros((height, width), dtype=np.uint8)

    if defect_type == "good":
        return mask_array

    class_id = DEFECT_CLASS_MAP[defect_type]
    gt_mask_path = find_ground_truth_mask(gt_root, defect_type, image_path.stem)
    gt_mask = Image.open(gt_mask_path).convert("L")
    if gt_mask.size != (width, height):
        gt_mask = gt_mask.resize((width, height), resample=Image.NEAREST)
    gt_array = np.array(gt_mask, dtype=np.uint8)
    mask_array[gt_array > 0] = class_id
    return mask_array


def copy_and_save(
    image_path: Path,
    mask_array: np.ndarray,
    images_dir: Path,
    masks_dir: Path,
    new_name: str | None = None,
) -> None:
    filename = new_name or image_path.name
    target_image = images_dir / filename
    target_mask = masks_dir / filename
    if target_image.exists():
        os.chmod(target_image, stat.S_IWRITE)
    if target_mask.exists():
        os.chmod(target_mask, stat.S_IWRITE)
    shutil.copy2(image_path, target_image)
    Image.fromarray(mask_array, mode="L").save(target_mask)


def split_test_images(
    test_root: Path, train_ratio: float, val_ratio: float, rng: random.Random
) -> Dict[str, List[Tuple[Path, str]]]:
    splits: Dict[str, List[Tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    allowed_types = {"good", *DEFECT_CLASS_MAP.keys()}
    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        if defect_type not in allowed_types:
            continue
        image_paths = sorted([p for p in defect_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        if not image_paths:
            continue
        rng.shuffle(image_paths)
        n = len(image_paths)
        train_count = int(n * train_ratio)
        val_count = int(n * val_ratio)
        splits["train"].extend([(p, defect_type) for p in image_paths[:train_count]])
        splits["val"].extend([(p, defect_type) for p in image_paths[train_count : train_count + val_count]])
        splits["test"].extend([(p, defect_type) for p in image_paths[train_count + val_count :]])
    return splits


def process_dataset(
    raw_dir: Path, processed_dir: Path, train_ratio: float, val_ratio: float, seed: int
) -> dict:
    rng = random.Random(seed)
    train_good_dir = raw_dir / "train" / "good"
    test_root = raw_dir / "test"
    gt_root = raw_dir / "ground_truth"

    if not train_good_dir.exists():
        raise FileNotFoundError(f"Missing train/good directory at {train_good_dir}")
    if not test_root.exists():
        raise FileNotFoundError(f"Missing test directory at {test_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"Missing ground_truth directory at {gt_root}")

    split_dirs = ensure_dirs(processed_dir)
    stats = defaultdict(int)

    # Train split: only good samples from the raw train set, masks are zeros.
    for img_path in sorted(train_good_dir.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        mask_array = create_mask(img_path, "good", gt_root)
        # Use a distinct prefix so these do not collide with `test/good` files
        # when those are also assigned to the processed train split.
        new_name = f"good_train_{img_path.name}"
        copy_and_save(
            img_path,
            mask_array,
            split_dirs["train"]["images"],
            split_dirs["train"]["masks"],
            new_name=new_name,
        )
        stats["train"] += 1

    # Validation/Test splits from the test folder.
    split_map = split_test_images(test_root, train_ratio, val_ratio, rng)
    for split_name, items in split_map.items():
        for img_path, defect_type in items:
            mask_array = create_mask(img_path, defect_type, gt_root)
            # `test/good` file names often overlap with `train/good`; keep them distinct.
            prefix = "good_test" if defect_type == "good" else defect_type
            new_name = f"{prefix}_{img_path.name}"
            copy_and_save(
                img_path,
                mask_array,
                split_dirs[split_name]["images"],
                split_dirs[split_name]["masks"],
                new_name=new_name,
            )
            stats[split_name] += 1

    return stats


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0 (remaining goes to test).")

    def _on_rm_error(func, path, exc_info):
        # Clear read-only and retry removal on Windows.
        import stat
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            raise

    if args.processed_dir.exists() and args.overwrite:
        shutil.rmtree(args.processed_dir, onerror=_on_rm_error)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    stats = process_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("Preprocessing complete.")
    for split in ("train", "val", "test"):
        print(f"{split}: {stats.get(split, 0)} images")


if __name__ == "__main__":
    main()
