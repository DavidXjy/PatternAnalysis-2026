
"""Find and visualise segmentation failure cases on HipMRI."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from predict import load_model


def calculate_dice(gt, pred, label=5):
    """Dice for one raw class in one image."""
    gt_binary = gt == label
    pred_binary = pred == label

    intersection = np.logical_and(
        gt_binary, pred_binary
    ).sum()

    denominator = gt_binary.sum() + pred_binary.sum()

    if denominator == 0:
        return None

    return float(2 * intersection / denominator)


def plot_overlay(ax, image, gt, pred, title):
    """Show ground-truth and prediction boundaries."""

    ax.imshow(image, cmap="gray")

    if np.any(gt):
        ax.contour(
            gt.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=1.5
        )

    if np.any(pred):
        ax.contour(
            pred.astype(float),
            levels=[0.5],
            colors="lime",
            linewidths=1.5
        )

    ax.set_title(title)
    ax.axis("off")


def plot_error(ax, image, error_mask, title, color):
    """Highlight false-positive or false-negative pixels."""

    ax.imshow(image, cmap="gray")

    overlay = np.ma.masked_where(
        ~error_mask,
        error_mask.astype(float)
    )

    ax.imshow(
        overlay,
        cmap=color,
        vmin=0,
        vmax=1,
        alpha=0.75
    )

    ax.set_title(title)
    ax.axis("off")


def create_figure(
    image, gt_mask, baseline_mask, improved_mask,
    label, baseline_dice, improved_dice, path
):
    gt = gt_mask == label
    baseline = baseline_mask == label
    improved = improved_mask == label

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Ground Truth: Label {label}")
    axes[0, 1].axis("off")

    plot_overlay(
        axes[0, 2], image, gt, baseline,
        f"Baseline Dice = {baseline_dice:.4f}"
    )

    plot_overlay(
        axes[0, 3], image, gt, improved,
        f"Improved Dice = {improved_dice:.4f}"
    )

    plot_error(
        axes[1, 0], image,
        baseline & ~gt,
        "Baseline False Positive",
        "Reds"
    )

    plot_error(
        axes[1, 1], image,
        gt & ~baseline,
        "Baseline False Negative",
        "Blues"
    )

    plot_error(
        axes[1, 2], image,
        improved & ~gt,
        "Improved False Positive",
        "Reds"
    )

    plot_error(
        axes[1, 3], image,
        gt & ~improved,
        "Improved False Negative",
        "Blues"
    )

    fig.suptitle(
        f"Raw Label {label}: Red GT / Green Prediction",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["validate", "test"],
        default="test"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    parser.add_argument("--label", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    if not 0 <= args.label < 6:
        raise ValueError("Label must be between 0 and 5.")

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Loading models...")

    baseline_model = load_model(
        "baseline", "checkpoints/baseline_unet.pt"
    ).to(device)

    improved_model = load_model(
        "improved", "checkpoints/improved_unet.pt"
    ).to(device)

    baseline_model.eval()
    improved_model.eval()

    dataset = HipMRIDataset(split=args.split)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    records = []
    offset = 0

    print("Evaluating slices...")

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)

            baseline_preds = baseline_model(
                images
            ).argmax(dim=1).cpu().numpy()

            improved_preds = improved_model(
                images
            ).argmax(dim=1).cpu().numpy()

            targets = targets.numpy()

            for j in range(len(targets)):
                gt = targets[j]

                # Only consider slices containing the target.
                if not np.any(gt == args.label):
                    continue

                baseline_dice = calculate_dice(
                    gt, baseline_preds[j], args.label
                )
                improved_dice = calculate_dice(
                    gt, improved_preds[j], args.label
                )

                records.append({
                    "index": offset + j,
                    "gt_pixels": int((gt == args.label).sum()),
                    "baseline_dice": baseline_dice,
                    "improved_dice": improved_dice,
                    "difference": improved_dice - baseline_dice
                })

            offset += len(targets)

    if not records:
        print("No ground-truth samples contain this label.")
        return

    # Select several distinct, informative failure patterns.
    candidates = [
        min(records, key=lambda x: x["improved_dice"]),
        min(records, key=lambda x: x["baseline_dice"]),
        max(records, key=lambda x: x["difference"]),
        min(records, key=lambda x: x["difference"])
    ]

    # Add other difficult cases when necessary.
    candidates += sorted(
        records,
        key=lambda x: (
            x["baseline_dice"] + x["improved_dice"]
        ) / 2
    )

    selected = []
    used = set()

    for record in candidates:
        if record["index"] not in used:
            selected.append(record)
            used.add(record["index"])

        if len(selected) >= args.top_k:
            break

    output_dir = Path("outputs/failure_cases")
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in selected:
        index = record["index"]

        image, gt = dataset[index]

        with torch.inference_mode():
            x = image.unsqueeze(0).to(device)

            baseline_pred = baseline_model(
                x
            ).argmax(dim=1).squeeze(0).cpu().numpy()

            improved_pred = improved_model(
                x
            ).argmax(dim=1).squeeze(0).cpu().numpy()

        filename = f"{args.split}_case_{index:04d}.png"

        create_figure(
            image.squeeze(0).numpy(),
            gt.numpy(),
            baseline_pred,
            improved_pred,
            args.label,
            record["baseline_dice"],
            record["improved_dice"],
            output_dir / filename
        )

        print(
            f"Case {index}: "
            f"Baseline={record['baseline_dice']:.4f}, "
            f"Improved={record['improved_dice']:.4f}"
        )

    csv_path = output_dir / f"{args.split}_case_metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(selected[0].keys())
        )
        writer.writeheader()
        writer.writerows(selected)

    print("Saved images and metrics to:", output_dir)
    print("Failure-case analysis complete.")


if __name__ == "__main__":
    main()
