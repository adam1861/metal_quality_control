from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Allow running as `python training/train_unet_metalnut.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.data_utils import MetalNutSegmentationDataset
from training.model import UNet

NUM_CLASSES = 3
CLASS_IDS = {"background": 0, "color": 1, "scratch": 2}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, masks, _ in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()

    avg_loss = running_loss / len(dataloader.dataset)
    pixel_acc = correct / total if total > 0 else 0.0
    return avg_loss, pixel_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    iou_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    union_sums = np.zeros(NUM_CLASSES, dtype=np.float64)

    for images, masks, _ in tqdm(dataloader, desc="Eval", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()

        preds_cpu = preds.cpu()
        masks_cpu = masks.cpu()
        for cls in range(NUM_CLASSES):
            pred_c = preds_cpu == cls
            target_c = masks_cpu == cls
            iou_sums[cls] += (pred_c & target_c).sum().item()
            union_sums[cls] += (pred_c | target_c).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    pixel_acc = correct / total if total > 0 else 0.0
    per_class_iou = [
        (iou_sums[cls] / union_sums[cls]) if union_sums[cls] > 0 else 0.0 for cls in range(NUM_CLASSES)
    ]
    mean_iou = float(np.mean(per_class_iou))
    return avg_loss, pixel_acc, per_class_iou, mean_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on metal_nut segmentation (color/scratch).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/metal_nut"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g., cpu or cuda.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", help="Enable simple geometric augmentations on train split.")
    parser.add_argument(
        "--balance-sampler",
        action="store_true",
        help="Use a weighted sampler to balance image types (good vs defect classes).",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "sqrt_bg1"),
        default="sqrt_bg1",
        help="Class weighting strategy for CrossEntropyLoss.",
    )
    parser.add_argument(
        "--init-weights-path",
        type=Path,
        default=None,
        help="Optional checkpoint to initialize the model (fine-tuning). The output layer will be re-initialized.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("models/best_unet_metalnut_colorscratch.pth"),
        help="Where to save the best model weights.",
    )
    return parser.parse_args()


def _compute_train_class_weights(
    masks_dir: Path, image_size: int, *, mode: str
) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    target_size = (image_size, image_size)

    for mask_path in sorted(masks_dir.glob("*.png")):
        mask_img = Image.open(mask_path).convert("L").resize(target_size, resample=Image.NEAREST)
        arr = np.array(mask_img, dtype=np.int64)
        counts += np.bincount(arr.reshape(-1), minlength=NUM_CLASSES)

    # sqrt-balanced weights (normalized so background weight becomes 1).
    total = float(counts.sum())
    denom = np.maximum(counts.astype(np.float64), 1.0)
    raw = np.sqrt(total / (NUM_CLASSES * denom))
    raw = raw / raw[0] if raw[0] > 0 else raw

    return torch.tensor(raw.astype(np.float32), dtype=torch.float32)


def _build_balanced_sampler(
    image_paths: list[Path],
) -> WeightedRandomSampler:
    prefixes = []
    for p in image_paths:
        stem = p.stem
        prefix = stem.split("_", 1)[0] if "_" in stem else "unknown"
        prefixes.append(prefix)

    counts = Counter(prefixes)
    weights = []
    for prefix in prefixes:
        w = 1.0 / float(counts.get(prefix, 1))
        weights.append(w)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    *,
    augment: bool,
    balance_sampler: bool,
    seed: int,
):
    train_ds = MetalNutSegmentationDataset(
        images_dir=data_dir / "images/train",
        masks_dir=data_dir / "masks/train",
        image_size=(image_size, image_size),
        augment=augment,
        seed=seed,
    )
    val_ds = MetalNutSegmentationDataset(
        images_dir=data_dir / "images/val",
        masks_dir=data_dir / "masks/val",
        image_size=(image_size, image_size),
        augment=False,
    )
    test_ds = MetalNutSegmentationDataset(
        images_dir=data_dir / "images/test",
        masks_dir=data_dir / "masks/test",
        image_size=(image_size, image_size),
        augment=False,
    )

    if balance_sampler:
        sampler = _build_balanced_sampler(train_ds.image_paths)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    expected = [
        args.data_dir / "images/train",
        args.data_dir / "images/val",
        args.data_dir / "images/test",
        args.data_dir / "masks/train",
        args.data_dir / "masks/val",
        args.data_dir / "masks/test",
    ]
    for path in expected:
        if not path.exists():
            raise FileNotFoundError(f"Expected directory not found: {path}. Run preprocessing first.")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        balance_sampler=args.balance_sampler,
        seed=args.seed,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=32).to(device)
    if args.init_weights_path and args.init_weights_path.exists():
        try:
            init_state = torch.load(args.init_weights_path, map_location=device, weights_only=True)
        except TypeError:  # torch<2.4 compatibility
            init_state = torch.load(args.init_weights_path, map_location=device)
        # Allow initializing from older checkpoints with a different output head.
        model_state = model.state_dict()
        filtered = {k: v for k, v in init_state.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Initialized weights from {args.init_weights_path} (loaded {len(filtered)}/{len(init_state)} tensors)")

    class_weights = _compute_train_class_weights(
        masks_dir=args.data_dir / "masks/train",
        image_size=args.image_size,
        mode=args.class_weighting,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print("Using class weights:", [float(x) for x in class_weights.detach().cpu().numpy().tolist()])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    args.weights_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_iou, val_miou = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} mIoU: {val_miou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.weights_path)
            print(f"  Saved new best model to {args.weights_path}")

    # Final evaluation on the test split.
    if args.weights_path.exists():
        try:
            best_state = torch.load(args.weights_path, map_location=device, weights_only=True)
        except TypeError:  # torch<2.4 compatibility
            best_state = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(best_state)
    test_loss, test_acc, test_iou, test_miou = evaluate(model, test_loader, criterion, device)
    print("Test Results")
    print(f"  Loss: {test_loss:.4f} | Pixel Acc: {test_acc:.4f} | mIoU: {test_miou:.4f}")
    for cls_id, iou in enumerate(test_iou):
        print(f"    Class {cls_id} IoU: {iou:.4f}")


if __name__ == "__main__":
    main()
