
"""Evaluate HipMRI segmentation models using Dice and IoU."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import HipMRIDataset
from predict import load_model


NUM_CLASSES = 6


def evaluate(model, loader, device):
    """Compute dataset-level per-class Dice and IoU."""

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int64
    )

    total_inference_time = 0.0
    total_images = 0

    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            logits = model(images)
            predictions = logits.argmax(dim=1)

            if device.type == "cuda":
                torch.cuda.synchronize()

            total_inference_time += time.perf_counter() - start
            total_images += images.shape[0]

            predictions = predictions.cpu().long()
            targets = targets.cpu().long()

            flat = (
                targets.reshape(-1) * NUM_CLASSES
                + predictions.reshape(-1)
            )

            counts = torch.bincount(
                flat, minlength=NUM_CLASSES * NUM_CLASSES
            )

            confusion += counts.reshape(
                NUM_CLASSES, NUM_CLASSES
            )

    results = []

    for label in range(NUM_CLASSES):
        tp = confusion[label, label].item()
        gt_pixels = confusion[label, :].sum().item()
        pred_pixels = confusion[:, label].sum().item()

        dice_denom = gt_pixels + pred_pixels
        iou_denom = gt_pixels + pred_pixels - tp

        dice = (
            2 * tp / dice_denom
            if dice_denom > 0 else None
        )

        iou = (
            tp / iou_denom
            if iou_denom > 0 else None
        )

        results.append({
            "label": label,
            "ground_truth_pixels": gt_pixels,
            "predicted_pixels": pred_pixels,
            "dice": dice,
            "iou": iou,
        })

    foreground_dice = [
        row["dice"]
        for row in results[1:]
        if row["dice"] is not None
    ]

    mean_dice = (
        sum(foreground_dice) / len(foreground_dice)
        if foreground_dice else None
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    return {
        "num_images": total_images,
        "mean_foreground_dice": mean_dice,
        "seconds_per_image": (
            total_inference_time / total_images
            if total_images else None
        ),
        "peak_gpu_allocated_mb": peak_memory_mb,
        "per_class": results,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["baseline", "improved"],
        default="baseline"
    )

    parser.add_argument(
        "--split",
        choices=["validate", "test"],
        default="validate"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    checkpoint = (
        "checkpoints/baseline_unet.pt"
        if args.model == "baseline"
        else "checkpoints/improved_unet.pt"
    )

    model = load_model(args.model, checkpoint)
    model = model.to(device)

    dataset = HipMRIDataset(split=args.split)
    full_dataset_size = len(dataset)

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")

        dataset = Subset(
            dataset,
            range(min(args.max_samples, len(dataset)))
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Model:", args.model)
    print("Split:", args.split)
    print("Samples:", len(dataset))
    print("Device:", device)

    result = evaluate(model, loader, device)

    result["model"] = args.model
    result["split"] = args.split
    result["device"] = str(device)
    result["full_dataset_size"] = full_dataset_size
    result["is_partial"] = (
        len(dataset) != full_dataset_size
    )

    result["parameters"] = sum(
        p.numel() for p in model.parameters()
    )

    print("\nEvaluation results:")

    for row in result["per_class"]:
        dice = row["dice"]
        iou = row["iou"]

        dice_text = f"{dice:.4f}" if dice is not None else "N/A"
        iou_text = f"{iou:.4f}" if iou is not None else "N/A"

        print(
            f"Label {row['label']}: "
            f"Dice={dice_text}, IoU={iou_text}"
        )

    print(
        "Mean foreground Dice:",
        result["mean_foreground_dice"]
    )

    print(
        "Inference seconds/image:",
        result["seconds_per_image"]
    )

    print(
        "Peak allocated GPU memory (MB):",
        result["peak_gpu_allocated_mb"]
    )

    if result["is_partial"]:
        print("PARTIAL CHECK ONLY: not a final evaluation.")

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"{args.model}_{args.split}_{args.device}"
        + ("_partial" if result["is_partial"] else "")
    )

    json_path = output_dir / f"{suffix}.json"
    csv_path = output_dir / f"{suffix}.csv"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "ground_truth_pixels",
                "predicted_pixels",
                "dice",
                "iou"
            ]
        )

        writer.writeheader()
        writer.writerows(result["per_class"])

    print("Saved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
