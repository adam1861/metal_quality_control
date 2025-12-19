from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as `python training/evaluate_unet_metalnut.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.data_utils import MetalNutSegmentationDataset
from training.inference import CLASS_ID_TO_NAME, segment_nut_mask
from training.model import UNet

NUM_CLASSES = max(CLASS_ID_TO_NAME.keys()) + 1


def _pil_resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    # Pillow changed resampling constants; keep backward compatible.
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return img.resize(size, resample=resample)


def _confusion_matrix_from_flat(
    targets: torch.Tensor, preds: torch.Tensor, *, num_classes: int
) -> torch.Tensor:
    """Build confusion matrix for integer tensors of same shape (flattened already)."""
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    targets = targets.to(torch.int64)
    preds = preds.to(torch.int64)
    k = num_classes * targets + preds
    cm = torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.to(torch.int64)


@dataclass(frozen=True)
class PerClassMetrics:
    class_id: int
    class_name: str
    support: int
    precision: float
    recall: float
    f1: float
    iou: float
    dice: float
    specificity: float


@dataclass(frozen=True)
class SummaryMetrics:
    pixel_accuracy: float
    mean_iou_all: float
    mean_iou_no_background: float
    macro_f1_no_background: float
    macro_precision_no_background: float
    macro_recall_no_background: float
    defect_binary_precision: float
    defect_binary_recall: float
    defect_binary_f1: float
    defect_binary_iou: float
    defect_binary_dice: float


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    num_images: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def _safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    return np.divide(numer, denom, out=np.zeros_like(numer, dtype=np.float64), where=denom != 0)


def metrics_from_confusion_matrix(cm: np.ndarray) -> tuple[list[PerClassMetrics], SummaryMetrics, dict[str, Any]]:
    cm = cm.astype(np.int64, copy=False)
    total = int(cm.sum())
    if total == 0:
        empty_summary = SummaryMetrics(
            pixel_accuracy=0.0,
            mean_iou_all=0.0,
            mean_iou_no_background=0.0,
            macro_f1_no_background=0.0,
            macro_precision_no_background=0.0,
            macro_recall_no_background=0.0,
            defect_binary_precision=0.0,
            defect_binary_recall=0.0,
            defect_binary_f1=0.0,
            defect_binary_iou=0.0,
            defect_binary_dice=0.0,
        )
        return [], empty_summary, {"confusion_matrix": cm.tolist()}

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    tn = float(total) - tp - fp - fn
    support = cm.sum(axis=1).astype(np.int64)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    specificity = _safe_div(tn, tn + fp)

    per_class: list[PerClassMetrics] = []
    for cls_id in range(cm.shape[0]):
        per_class.append(
            PerClassMetrics(
                class_id=int(cls_id),
                class_name=CLASS_ID_TO_NAME.get(int(cls_id), str(cls_id)),
                support=int(support[cls_id]),
                precision=float(precision[cls_id]),
                recall=float(recall[cls_id]),
                f1=float(f1[cls_id]),
                iou=float(iou[cls_id]),
                dice=float(dice[cls_id]),
                specificity=float(specificity[cls_id]),
            )
        )

    pixel_accuracy = float(tp.sum() / float(total))

    mean_iou_all = float(iou.mean())
    defect_ids = np.arange(1, cm.shape[0], dtype=np.int64)
    mean_iou_no_bg = float(iou[defect_ids].mean()) if len(defect_ids) else 0.0

    macro_precision_no_bg = float(precision[defect_ids].mean()) if len(defect_ids) else 0.0
    macro_recall_no_bg = float(recall[defect_ids].mean()) if len(defect_ids) else 0.0
    macro_f1_no_bg = float(f1[defect_ids].mean()) if len(defect_ids) else 0.0

    # Binary "any defect" metrics (defect classes vs background).
    # TP: true defect predicted as defect (any class 1-4).
    tp_def = float(cm[1:, 1:].sum())
    fp_def = float(cm[0, 1:].sum())
    fn_def = float(cm[1:, 0].sum())
    tn_def = float(cm[0, 0])
    defect_precision = float(tp_def / (tp_def + fp_def)) if (tp_def + fp_def) > 0 else 0.0
    defect_recall = float(tp_def / (tp_def + fn_def)) if (tp_def + fn_def) > 0 else 0.0
    defect_f1 = (
        float(2 * defect_precision * defect_recall / (defect_precision + defect_recall))
        if (defect_precision + defect_recall) > 0
        else 0.0
    )
    defect_iou = float(tp_def / (tp_def + fp_def + fn_def)) if (tp_def + fp_def + fn_def) > 0 else 0.0
    defect_dice = float((2 * tp_def) / (2 * tp_def + fp_def + fn_def)) if (2 * tp_def + fp_def + fn_def) > 0 else 0.0

    summary = SummaryMetrics(
        pixel_accuracy=pixel_accuracy,
        mean_iou_all=mean_iou_all,
        mean_iou_no_background=mean_iou_no_bg,
        macro_f1_no_background=macro_f1_no_bg,
        macro_precision_no_background=macro_precision_no_bg,
        macro_recall_no_background=macro_recall_no_bg,
        defect_binary_precision=defect_precision,
        defect_binary_recall=defect_recall,
        defect_binary_f1=defect_f1,
        defect_binary_iou=defect_iou,
        defect_binary_dice=defect_dice,
    )

    details: dict[str, Any] = {
        "confusion_matrix": cm.tolist(),  # rows=true, cols=pred
        "confusion_matrix_normalized_by_true": (
            _safe_div(cm.astype(np.float64), cm.sum(axis=1, keepdims=True).astype(np.float64)).tolist()
        ),
        "total_pixels": total,
    }
    return per_class, summary, details


def binary_metrics_from_counts(*, tp: int, fp: int, fn: int, tn: int) -> BinaryClassificationMetrics:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return BinaryClassificationMetrics(
        num_images=total,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def dominant_class_from_flat(mask_flat: torch.Tensor, *, num_classes: int) -> int:
    """Return the dominant non-background class id for an image.

    - Returns 0 if no defect pixels are present.
    - If multiple defect classes are present, returns the class with the most pixels.
    """
    if mask_flat.numel() == 0:
        return 0
    counts = torch.bincount(mask_flat.to(torch.int64), minlength=num_classes)
    if int(counts[1:].sum().item()) == 0:
        return 0
    return int(torch.argmax(counts[1:]).item() + 1)


def _print_image_cm(cm: np.ndarray) -> None:
    labels = [f"{i}:{CLASS_ID_TO_NAME.get(i, str(i))}" for i in range(cm.shape[0])]
    col_widths = [max(len(l), 7) for l in labels]
    header = "true\\pred".ljust(10) + "  " + "  ".join(labels[i].ljust(col_widths[i]) for i in range(len(labels)))
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        line = labels[i].ljust(10) + "  " + "  ".join(str(int(row[j])).ljust(col_widths[j]) for j in range(len(labels)))
        print(line)


def _image_precision_recall_from_cm(cm: np.ndarray, cls_id: int) -> tuple[float, float, int, int, int]:
    """Return (precision, recall, correct, predicted, support) for a class id."""
    correct = int(cm[cls_id, cls_id])
    predicted = int(cm[:, cls_id].sum())
    support = int(cm[cls_id, :].sum())
    precision = (correct / predicted) if predicted else 0.0
    recall = (correct / support) if support else 0.0
    return float(precision), float(recall), correct, predicted, support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained U-Net metal_nut segmentation model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/metal_nut"))
    parser.add_argument("--weights-path", type=Path, default=Path("models/best_unet_metalnut_colorscratch.pth"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g., cpu or cuda.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write metrics JSON.")
    parser.add_argument(
        "--image-class-precision-only",
        action="store_true",
        help="Print only image-level class precision for color/scratch (dominant class), instead of full pixel metrics.",
    )
    parser.add_argument(
        "--no-nut-metrics",
        action="store_true",
        help="Skip metrics restricted to nut pixels (computed via Otsu-based nut segmentation).",
    )
    return parser.parse_args()


def _print_table(per_class: list[PerClassMetrics]) -> None:
    headers = ["class", "support", "precision", "recall", "f1", "iou", "dice"]
    rows = []
    for m in per_class:
        rows.append(
            [
                f"{m.class_id}:{m.class_name}",
                str(m.support),
                f"{m.precision:.4f}",
                f"{m.recall:.4f}",
                f"{m.f1:.4f}",
                f"{m.iou:.4f}",
                f"{m.dice:.4f}",
            ]
        )

    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for r in rows:
        print("  ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))))


def main() -> None:
    args = parse_args()

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        images_dir = args.data_dir / f"images/{split}"
        masks_dir = args.data_dir / f"masks/{split}"
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f"Missing expected split directories: {images_dir} and {masks_dir}")
    if not args.weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights_path}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=32).to(device)
    try:
        state = torch.load(args.weights_path, map_location=device, weights_only=True)
    except TypeError:  # torch<2.4 compatibility
        state = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cm_all = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    cm_on_nut = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    img_tp = img_fp = img_fn = img_tn = 0
    img_tp_nut = img_fp_nut = img_fn_nut = img_tn_nut = 0
    img_cm_all = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    img_cm_nut = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    for split in splits:
        images_dir = args.data_dir / f"images/{split}"
        masks_dir = args.data_dir / f"masks/{split}"
        ds = MetalNutSegmentationDataset(
            images_dir=images_dir, masks_dir=masks_dir, image_size=(args.image_size, args.image_size)
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for images, masks, names in tqdm(dl, desc=f"Eval ({split})"):
            images = images.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                logits = model(images)
                preds = torch.argmax(logits, dim=1)

            preds_cpu = preds.detach().to("cpu")
            masks_cpu = masks.detach().to("cpu")

            cm_all += _confusion_matrix_from_flat(masks_cpu.flatten(), preds_cpu.flatten(), num_classes=NUM_CLASSES)

            # Image-level detection: "any defect present?"
            for i in range(len(names)):
                gt_def = bool((masks_cpu[i] != 0).any().item())
                pred_def = bool((preds_cpu[i] != 0).any().item())
                if gt_def and pred_def:
                    img_tp += 1
                elif (not gt_def) and pred_def:
                    img_fp += 1
                elif gt_def and (not pred_def):
                    img_fn += 1
                else:
                    img_tn += 1
                gt_label = dominant_class_from_flat(masks_cpu[i].flatten(), num_classes=NUM_CLASSES)
                pred_label = dominant_class_from_flat(preds_cpu[i].flatten(), num_classes=NUM_CLASSES)
                img_cm_all[gt_label, pred_label] += 1

            if not args.no_nut_metrics:
                # Restrict metrics to nut pixels only.
                for i, name in enumerate(names):
                    img_path = images_dir / name
                    if not img_path.exists():
                        continue
                    pil_img = Image.open(img_path).convert("RGB")
                    pil_img = _pil_resize(pil_img, (args.image_size, args.image_size))
                    nut = segment_nut_mask(pil_img)
                    nut_flat = torch.from_numpy(nut.astype(np.bool_)).flatten()

                    t = masks_cpu[i].flatten()
                    p = preds_cpu[i].flatten()
                    t_sel = t[nut_flat]
                    p_sel = p[nut_flat]
                    cm_on_nut += _confusion_matrix_from_flat(t_sel, p_sel, num_classes=NUM_CLASSES)

                    gt_def_nut = bool((t_sel != 0).any().item()) if t_sel.numel() else bool((t != 0).any().item())
                    pred_def_nut = bool((p_sel != 0).any().item()) if p_sel.numel() else bool((p != 0).any().item())
                    if gt_def_nut and pred_def_nut:
                        img_tp_nut += 1
                    elif (not gt_def_nut) and pred_def_nut:
                        img_fp_nut += 1
                    elif gt_def_nut and (not pred_def_nut):
                        img_fn_nut += 1
                    else:
                        img_tn_nut += 1

                    gt_label_nut = (
                        dominant_class_from_flat(t_sel, num_classes=NUM_CLASSES)
                        if t_sel.numel()
                        else dominant_class_from_flat(t, num_classes=NUM_CLASSES)
                    )
                    pred_label_nut = (
                        dominant_class_from_flat(p_sel, num_classes=NUM_CLASSES)
                        if p_sel.numel()
                        else dominant_class_from_flat(p, num_classes=NUM_CLASSES)
                    )
                    img_cm_nut[gt_label_nut, pred_label_nut] += 1

    per_class_all, summary_all, details_all = metrics_from_confusion_matrix(cm_all.numpy())

    img_cm_np = img_cm_all.numpy()
    img_acc = float(np.trace(img_cm_np) / float(img_cm_np.sum())) if int(img_cm_np.sum()) else 0.0
    color_prec, color_rec, color_correct, color_pred, color_sup = _image_precision_recall_from_cm(img_cm_np, 1)
    scratch_prec, scratch_rec, scratch_correct, scratch_pred, scratch_sup = _image_precision_recall_from_cm(img_cm_np, 2)

    if args.image_class_precision_only and args.no_nut_metrics:
        print(f"Image-level precision (dominant class) — color: {color_prec:.4f}  scratch: {scratch_prec:.4f}")
        print(f"  color: {color_correct}/{color_pred} (predicted color)")
        print(f"  scratch: {scratch_correct}/{scratch_pred} (predicted scratch)")

    if not args.image_class_precision_only:
        print("\n=== Metrics (all pixels) ===")
        print(f"Pixel accuracy: {summary_all.pixel_accuracy:.4f}")
        print(f"Mean IoU (all classes): {summary_all.mean_iou_all:.4f}")
        print(f"Mean IoU (defects only, no background): {summary_all.mean_iou_no_background:.4f}")
        print(
            "Macro (defects only) — "
            f"Precision: {summary_all.macro_precision_no_background:.4f}  "
            f"Recall: {summary_all.macro_recall_no_background:.4f}  "
            f"F1: {summary_all.macro_f1_no_background:.4f}"
        )
        print(
            "Binary (any defect vs background) — "
            f"Precision: {summary_all.defect_binary_precision:.4f}  "
            f"Recall: {summary_all.defect_binary_recall:.4f}  "
            f"F1: {summary_all.defect_binary_f1:.4f}  "
            f"IoU: {summary_all.defect_binary_iou:.4f}  "
            f"Dice: {summary_all.defect_binary_dice:.4f}"
        )
        img_metrics_all = binary_metrics_from_counts(tp=img_tp, fp=img_fp, fn=img_fn, tn=img_tn)
        print(
            "Image-level (any defect present?) — "
            f"Acc: {img_metrics_all.accuracy:.4f}  "
            f"Precision: {img_metrics_all.precision:.4f}  "
            f"Recall: {img_metrics_all.recall:.4f}  "
            f"F1: {img_metrics_all.f1:.4f}  "
            f"(TP/FP/FN/TN: {img_metrics_all.tp}/{img_metrics_all.fp}/{img_metrics_all.fn}/{img_metrics_all.tn})"
        )
        print(f"Image-level (dominant class: good/color/scratch) — Acc: {img_acc:.4f}")
        _print_image_cm(img_cm_np)
        print(
            f"  color — precision: {color_prec:.4f} ({color_correct}/{color_pred})  "
            f"recall: {color_rec:.4f} ({color_correct}/{color_sup})"
        )
        print(
            f"  scratch — precision: {scratch_prec:.4f} ({scratch_correct}/{scratch_pred})  "
            f"recall: {scratch_rec:.4f} ({scratch_correct}/{scratch_sup})"
        )
        print("\nPer-class (all pixels):")
        _print_table(per_class_all)

    results: dict[str, Any] = {
        "split": args.split,
        "weights_path": str(args.weights_path),
        "data_dir": str(args.data_dir),
        "image_size": int(args.image_size),
        "device": str(device),
        "all_pixels": {
            "summary": asdict(summary_all),
            "per_class": [asdict(m) for m in per_class_all],
            "image_level_any_defect": asdict(binary_metrics_from_counts(tp=img_tp, fp=img_fp, fn=img_fn, tn=img_tn)),
            "image_level_dominant_class": {
                "accuracy": img_acc,
                "confusion_matrix": img_cm_np.tolist(),  # rows=true, cols=pred
                "color": {
                    "precision": color_prec,
                    "recall": color_rec,
                    "correct": color_correct,
                    "predicted": color_pred,
                    "support": color_sup,
                },
                "scratch": {
                    "precision": scratch_prec,
                    "recall": scratch_rec,
                    "correct": scratch_correct,
                    "predicted": scratch_pred,
                    "support": scratch_sup,
                },
            },
            **details_all,
        },
    }

    if not args.no_nut_metrics:
        per_class_nut, summary_nut, details_nut = metrics_from_confusion_matrix(cm_on_nut.numpy())
        img_cm_nut_np = img_cm_nut.numpy()
        img_acc_nut = float(np.trace(img_cm_nut_np) / float(img_cm_nut_np.sum())) if int(img_cm_nut_np.sum()) else 0.0
        color_prec_n, color_rec_n, color_correct_n, color_pred_n, color_sup_n = _image_precision_recall_from_cm(
            img_cm_nut_np, 1
        )
        scratch_prec_n, scratch_rec_n, scratch_correct_n, scratch_pred_n, scratch_sup_n = _image_precision_recall_from_cm(
            img_cm_nut_np, 2
        )

        if args.image_class_precision_only:
            print(f"Image-level precision (dominant class, on nut) — color: {color_prec_n:.4f}  scratch: {scratch_prec_n:.4f}")
            print(f"  color: {color_correct_n}/{color_pred_n} (predicted color)")
            print(f"  scratch: {scratch_correct_n}/{scratch_pred_n} (predicted scratch)")
        else:
            print("\n=== Metrics (nut pixels only) ===")
            print(f"Pixel accuracy (on nut): {summary_nut.pixel_accuracy:.4f}")
            print(f"Mean IoU (all classes, on nut): {summary_nut.mean_iou_all:.4f}")
            print(f"Mean IoU (defects only, on nut): {summary_nut.mean_iou_no_background:.4f}")
            print(
                "Macro (defects only, on nut) — "
                f"Precision: {summary_nut.macro_precision_no_background:.4f}  "
                f"Recall: {summary_nut.macro_recall_no_background:.4f}  "
                f"F1: {summary_nut.macro_f1_no_background:.4f}"
            )
            print(
                "Binary (any defect vs background, on nut) — "
                f"Precision: {summary_nut.defect_binary_precision:.4f}  "
                f"Recall: {summary_nut.defect_binary_recall:.4f}  "
                f"F1: {summary_nut.defect_binary_f1:.4f}  "
                f"IoU: {summary_nut.defect_binary_iou:.4f}  "
                f"Dice: {summary_nut.defect_binary_dice:.4f}"
            )
            img_metrics_nut = binary_metrics_from_counts(tp=img_tp_nut, fp=img_fp_nut, fn=img_fn_nut, tn=img_tn_nut)
            print(
                "Image-level (any defect present?, on nut) — "
                f"Acc: {img_metrics_nut.accuracy:.4f}  "
                f"Precision: {img_metrics_nut.precision:.4f}  "
                f"Recall: {img_metrics_nut.recall:.4f}  "
                f"F1: {img_metrics_nut.f1:.4f}  "
                f"(TP/FP/FN/TN: {img_metrics_nut.tp}/{img_metrics_nut.fp}/{img_metrics_nut.fn}/{img_metrics_nut.tn})"
            )
            print(f"Image-level (dominant class, on nut) — Acc: {img_acc_nut:.4f}")
            _print_image_cm(img_cm_nut_np)
            print(
                f"  color — precision: {color_prec_n:.4f} ({color_correct_n}/{color_pred_n})  "
                f"recall: {color_rec_n:.4f} ({color_correct_n}/{color_sup_n})"
            )
            print(
                f"  scratch — precision: {scratch_prec_n:.4f} ({scratch_correct_n}/{scratch_pred_n})  "
                f"recall: {scratch_rec_n:.4f} ({scratch_correct_n}/{scratch_sup_n})"
            )
            print("\nPer-class (nut pixels only):")
            _print_table(per_class_nut)

        results["nut_pixels_only"] = {
            "summary": asdict(summary_nut),
            "per_class": [asdict(m) for m in per_class_nut],
            "image_level_any_defect": asdict(
                binary_metrics_from_counts(tp=img_tp_nut, fp=img_fp_nut, fn=img_fn_nut, tn=img_tn_nut)
            ),
            "image_level_dominant_class": {
                "accuracy": img_acc_nut,
                "confusion_matrix": img_cm_nut_np.tolist(),  # rows=true, cols=pred
                "color": {
                    "precision": color_prec_n,
                    "recall": color_rec_n,
                    "correct": color_correct_n,
                    "predicted": color_pred_n,
                    "support": color_sup_n,
                },
                "scratch": {
                    "precision": scratch_prec_n,
                    "recall": scratch_rec_n,
                    "correct": scratch_correct_n,
                    "predicted": scratch_pred_n,
                    "support": scratch_sup_n,
                },
            },
            **details_nut,
        }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
