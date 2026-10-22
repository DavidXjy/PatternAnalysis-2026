"""Visualise HipMRI segmentation predictions from a trained U-Net."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import HipMRIDataset
from modules import UNet2D
from improved_modules import ImprovedUNet2D


NUM_CLASSES = 6


def load_model(model_name, checkpoint_path):
    """Load a trained baseline or improved U-Net."""

    if model_name == "baseline":
        model = UNet2D(
            in_channels=1,
            out_channels=NUM_CLASSES,
            base_channels=16
        )
    else:
        model = ImprovedUNet2D(
            in_channels=1,
            out_channels=NUM_CLASSES,
            base_channels=16
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    # Support common checkpoint formats.
    state_dict = None

    for key in (
        "model_state_dict",
        "model_state",
        "state_dict",
        "model",
        "weights"
    ):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        if all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Checkpoint format not recognised. "
                f"Available keys: {list(checkpoint.keys())}"
            )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Model loaded:", model_name)

    if "epoch" in checkpoint:
        print("Checkpoint epoch:", checkpoint["epoch"])

    return model


def predict_sample(model, dataset, index):
    """Predict one MRI slice."""

    image, ground_truth = dataset[index]

    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        prediction = logits.argmax(dim=1).squeeze(0)

    return (
        image.squeeze(0).numpy(),
        ground_truth.numpy(),
        prediction.numpy()
    )


def save_visualisation(image, ground_truth, prediction, path):
    """Save MRI, ground truth and prediction panels."""

    fig, axes = plt.subplots(1, 4, figsize=(16, 7))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI")

    axes[1].imshow(
        ground_truth,
        cmap="tab10",
        vmin=0,
        vmax=9
    )
    axes[1].set_title("Ground Truth")

    axes[2].imshow(
        prediction,
        cmap="tab10",
        vmin=0,
        vmax=9
    )
    axes[2].set_title("Prediction")

    # Highlight raw Label 5 for anatomical inspection.
    axes[3].imshow(image, cmap="gray")

    gt5 = np.ma.masked_where(
        ground_truth != 5,
        ground_truth
    )

    pred5 = np.ma.masked_where(
        prediction != 5,
        prediction
    )

    axes[3].imshow(gt5, cmap="Reds", alpha=0.55)
    axes[3].contour(
        prediction == 5,
        levels=[0.5],
        colors="lime",
        linewidths=1
    ) if np.any(prediction == 5) else None

    axes[3].set_title("Label 5: GT red / Pred green")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    print("Saved visualisation:", path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["baseline", "improved"],
        default="baseline"
    )

    parser.add_argument(
        "--checkpoint",
        default=None
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        checkpoint_path = (
            "checkpoints/baseline_unet.pt"
            if args.model == "baseline"
            else "checkpoints/improved_unet.pt"
        )

    model = load_model(
        args.model,
        checkpoint_path
    )

    dataset = HipMRIDataset(split="test")

    if not 0 <= args.index < len(dataset):
        raise ValueError("Sample index outside test set.")

    image, ground_truth, prediction = predict_sample(
        model,
        dataset,
        args.index
    )

    print("MRI shape:", image.shape)
    print("GT labels:", np.unique(ground_truth).tolist())
    print("Predicted labels:", np.unique(prediction).tolist())

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{args.model}_test_{args.index}.png"
    )

    save_visualisation(
        image,
        ground_truth,
        prediction,
        output_path
    )

    print("Prediction complete.")


if __name__ == "__main__":
    main()