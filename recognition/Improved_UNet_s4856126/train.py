
"""Train/evaluate a standard 2D U-Net on the HipMRI 6-class slice masks.

IMPORTANT: The class index corresponding to the prostate is not yet verified.
Per-class metrics here are for raw label IDs, NOT confirmed prostate Dice.
Do not use the test split for tuning; --evaluate-test only after model selection.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, check_patient_splits
from modules import UNet2D


NUM_CLASSES = 6


# ==========================================
# 1. Command-line Arguments
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HipMRI 2D U-Net training"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="One tiny CPU optimization step with one real MRI slice"
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate a saved checkpoint on the held-out test set"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/baseline_unet.pt"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs"
    )

    return parser.parse_args()


# ==========================================
# 2. Train One Epoch
# ==========================================

def train_one_epoch(
    model, loader, optimizer, criterion, device
):
    model.train()

    total_loss = 0.0
    total_images = 0

    for images, masks in loader:

        # Move tensors to CPU or GPU
        images = images.to(device)
        masks = masks.to(device)

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward propagation
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, masks)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / total_images


# ==========================================
# 3. Validation and Dice Score
# ==========================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_images = 0

    intersections = torch.zeros(
        NUM_CLASSES, dtype=torch.float64
    )

    denominators = torch.zeros(
        NUM_CLASSES, dtype=torch.float64
    )

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        logits = model(images)

        # Validation loss
        loss = criterion(logits, masks)

        # Convert logits to predicted class IDs
        predictions = logits.argmax(dim=1)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_images += batch_size

        # Calculate statistics for each class
        for label in range(NUM_CLASSES):

            pred_label = predictions == label
            true_label = masks == label

            intersections[label] += (
                pred_label & true_label
            ).sum().item()

            denominators[label] += (
                pred_label.sum() + true_label.sum()
            ).item()

    # Calculate Dice for every class
    dice = {}

    for label in range(NUM_CLASSES):

        if denominators[label] > 0:
            dice[label] = (
                2.0 * intersections[label].item()
                / denominators[label].item()
            )
        else:
            dice[label] = float("nan")

    # Mean Dice over non-background labels
    foreground_dice = [
        dice[k]
        for k in range(1, NUM_CLASSES)
        if np.isfinite(dice[k])
    ]

    macro_foreground_dice = (
        float(np.mean(foreground_dice))
        if foreground_dice
        else float("nan")
    )

    return (
        total_loss / total_images,
        dice,
        macro_foreground_dice
    )


# ==========================================
# 4. Smoke Test
# ==========================================

def smoke_test():
    """
    Test one small optimization step on CPU.

    This is NOT full training.
    It checks:
    - Dataset loading
    - Forward propagation
    - CrossEntropyLoss
    - Backpropagation
    - Adam optimizer
    """

    # Limit CPU usage on Rangpur login node
    torch.set_num_threads(1)

    # Load one real MRI slice
    dataset = HipMRIDataset(
        "train",
        binary_segmentation=False
    )

    image, mask = dataset[0]

    # Resize MRI to 64 x 64 for lightweight testing
    x = F.interpolate(
        image.unsqueeze(0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False
    )

    # Resize mask using nearest-neighbor interpolation
    y = F.interpolate(
        mask[None, None].float(),
        size=(64, 64),
        mode="nearest"
    )[:, 0].long()

    # Create a small U-Net
    model = UNet2D(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base_channels=8
    )

    model.train()

    # Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # Multi-class segmentation loss
    criterion = nn.CrossEntropyLoss()

    # Forward propagation
    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    # Calculate loss
    loss = criterion(logits, y)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print results
    print("MRI batch:", tuple(x.shape))
    print("Mask batch:", tuple(y.shape))
    print("Model output:", tuple(logits.shape))

    print(
        "CrossEntropyLoss:",
        round(loss.item(), 4)
    )

    print(
        "Forward + backward + optimizer: successful"
    )

    print(
        "This was a smoke test, "
        "NOT model training or a Dice result."
    )


# ==========================================
# 5. Save Training Results and Plots
# ==========================================

def save_history_and_plot(history, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = output_dir / "baseline_history.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_macro_dice"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:
        print(
            "matplotlib not installed: "
            "saved CSV but skipped plot"
        )
        return

    epochs = [
        item["epoch"]
        for item in history
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4)
    )

    # Training and validation loss
    axes[0].plot(
        epochs,
        [item["train_loss"] for item in history],
        label="Train"
    )

    axes[0].plot(
        epochs,
        [item["val_loss"] for item in history],
        label="Validation"
    )

    axes[0].set(
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
        title="Loss"
    )

    axes[0].legend()

    # Validation Dice
    axes[1].plot(
        epochs,
        [item["val_macro_dice"] for item in history]
    )

    axes[1].set(
        xlabel="Epoch",
        ylabel="Dice",
        title="Validation mean Dice (labels 1-5)"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "baseline_training_curves.png",
        dpi=150
    )

    plt.close(fig)

    print(
        f"Saved training history and plots to {output_dir}"
    )


# ==========================================
# 6. Main Function
# ==========================================

def main():

    args = parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Run lightweight smoke test
    if args.smoke_test:
        smoke_test()
        return

    # Check CUDA availability for full training
    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA unavailable. "
            "Run full training in a Slurm GPU job."
        )

    device = torch.device(args.device)

    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(args.checkpoint)

    # ======================================
    # 7. Evaluate Test Set
    # ======================================

    if args.evaluate_test:

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        saved = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model = UNet2D(
            in_channels=1,
            out_channels=NUM_CLASSES,
            base_channels=saved["base_channels"]
        ).to(device)

        model.load_state_dict(
            saved["model_state_dict"]
        )

        test_loader = DataLoader(
            HipMRIDataset("test"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        loss, dice, macro_dice = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(
            f"TEST cross-entropy loss: {loss:.4f}"
        )

        print(
            "TEST Dice by RAW LABEL "
            "(not confirmed anatomy):",
            dice
        )

        print(
            f"TEST mean Dice for labels 1-5: "
            f"{macro_dice:.4f}"
        )

        return

    # ======================================
    # 8. Prepare Training Data
    # ======================================

    train_set = HipMRIDataset(
        "train",
        augment=True
    )

    val_set = HipMRIDataset("validate")

    test_set = HipMRIDataset("test")

    # Check patient-level separation
    case_counts = check_patient_splits(
        train_set,
        val_set,
        test_set
    )

    print(
        "Non-overlapping case counts "
        "(train/val/test):",
        case_counts
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    # ======================================
    # 9. Create U-Net and Optimizer
    # ======================================

    model = UNet2D(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base_channels=args.base_channels
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # ======================================
    # 10. Training Loop
    # ======================================

    history = []

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):

        # Training
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        # Validation
        val_loss, val_dice, val_macro_dice = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_macro_dice={val_macro_dice:.4f}"
        )

        print(
            "Validation Dice by RAW LABEL:",
            {
                k: round(v, 4)
                for k, v in val_dice.items()
            }
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_dice": val_macro_dice
        })

        # Save best-performing model
        if (
            np.isfinite(val_macro_dice)
            and val_macro_dice > best_dice
        ):

            best_dice = val_macro_dice

            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": args.base_channels,
                    "epoch": epoch,
                    "val_macro_dice": best_dice
                },
                checkpoint_path
            )

            print(
                f"Saved best checkpoint: "
                f"{checkpoint_path}"
            )

    # ======================================
    # 11. Save Training Curves
    # ======================================

    save_history_and_plot(
        history,
        Path(args.output_dir)
    )

    print(
        "Training complete. "
        "Prostate label mapping still requires verification."
    )


# ==========================================
# 12. Run Program
# ==========================================

if __name__ == "__main__":
    main()
