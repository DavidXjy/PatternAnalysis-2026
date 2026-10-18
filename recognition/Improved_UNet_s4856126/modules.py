
"""
COMP3710 Report
Standard 2D U-Net Baseline for HipMRI segmentation.

This module contains the neural network architecture.
It uses PyTorch only and does not load dataset files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Double Convolution Block
# ==========================================

class DoubleConv(nn.Module):
    """
    Two convolution layers with BatchNorm and ReLU.

    Input:  [B, in_channels, H, W]
    Output: [B, out_channels, H, W]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


# ==========================================
# 2. Downsampling Block (Encoder)
# ==========================================

class Down(nn.Module):
    """
    Reduce spatial resolution by a factor of 2.
    Then extract convolutional features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


# ==========================================
# 3. Upsampling Block (Decoder)
# ==========================================

class Up(nn.Module):
    """
    Upsample feature maps and concatenate
    corresponding encoder features.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        # Increase spatial resolution
        x = self.up(x)

        # Match the encoder's spatial dimensions
        # when the input has odd image dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Skip connection: concatenate along channels
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


# ==========================================
# 4. Standard 2D U-Net
# ==========================================

class UNet2D(nn.Module):
    """
    Standard U-Net with four downsampling
    and four upsampling stages.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, H, W]

    The output contains raw logits.
    No Sigmoid or Softmax is applied here.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=4,
        base_channels=16
    ):
        super().__init__()

        c = base_channels

        # ------------------------------
        # Encoder
        # ------------------------------

        self.inc = DoubleConv(
            in_channels, c
        )

        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        # ------------------------------
        # Decoder
        # ------------------------------

        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)

        # ------------------------------
        # Output Layer
        # ------------------------------

        # 1x1 convolution converts feature maps
        # into class prediction logits
        self.out_conv = nn.Conv2d(
            c,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Per-pixel prediction
        logits = self.out_conv(x)

        return logits


# ==========================================
# 5. Model Smoke Test
# ==========================================

if __name__ == "__main__":

    # Keep the small CPU test lightweight
    torch.set_num_threads(1)

    model = UNet2D(
        in_channels=1,
        out_channels=4,
        base_channels=16
    )

    model.eval()

    # Simulated MRI batch with HipMRI image dimensions
    test_input = torch.randn(
        1, 1, 256, 128
    )

    # Forward pass without gradient calculation
    with torch.no_grad():
        test_output = model(test_input)

    # Number of trainable parameters
    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("===== Standard 2D U-Net =====")
    print("Input shape:", tuple(test_input.shape))
    print("Output shape:", tuple(test_output.shape))
    print("Trainable parameters:", parameters)

    assert test_output.shape == (1, 4, 256, 128)

    print("U-Net forward pass successful!")
