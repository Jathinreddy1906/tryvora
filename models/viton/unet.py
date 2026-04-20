# models/viton/unet.py
# ─────────────────────────────────────────────────────────
# U-Net: Feature Extraction for VITON
#
# Simple explanation:
#   A U-Net is like a funnel:
#   → ENCODER: shrinks image, extracts features (patterns)
#   → BOTTLENECK: deepest features captured
#   → DECODER: grows back to full size using skip connections
#   Skip connections = "remember what the input looked like"
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two convolution layers with BatchNorm and ReLU.
    This is the basic building block of U-Net.

    Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder step: halve spatial size, double channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: double spatial size, halve channels.
    Uses bilinear upsampling (no checkerboard artifacts).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch between x and skip connection
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear',
                              align_corners=True)

        # Concatenate skip connection (this is the key U-Net feature!)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FeatureExtractorUNet(nn.Module):
    """
    Lightweight U-Net for feature extraction.

    Input:  image tensor [B, in_channels, H, W]
    Output: feature map  [B, out_channels, H, W]

    Used to extract features from:
      - Person image (after shirt removal)
      - Clothing image
    Then these features guide the warping network.
    """
    def __init__(self, in_channels=3, out_channels=256,
                 base_ch=32):
        super().__init__()

        # ── Encoder ──────────────────────────────────────
        self.inc   = DoubleConv(in_channels, base_ch)        # 32
        self.down1 = Down(base_ch,     base_ch * 2)          # 64
        self.down2 = Down(base_ch * 2, base_ch * 4)          # 128
        self.down3 = Down(base_ch * 4, base_ch * 8)          # 256

        # ── Bottleneck ────────────────────────────────────
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 8)

        # ── Decoder ───────────────────────────────────────
        self.up1 = Up(base_ch * 8 + base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4 + base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2 + base_ch * 2, base_ch)

        # ── Output head ───────────────────────────────────
        self.out_conv = nn.Conv2d(
            base_ch + base_ch, out_channels, 1
        )

    def forward(self, x):
        # Encoder path (save skip connections)
        s1 = self.inc(x)       # full resolution
        s2 = self.down1(s1)    # 1/2
        s3 = self.down2(s2)    # 1/4
        s4 = self.down3(s3)    # 1/8

        # Bottleneck
        b  = self.bottleneck(s4)

        # Decoder path (use skip connections)
        d1 = self.up1(b,  s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)

        # Final output
        out = self.out_conv(
            torch.cat([d3, s1], dim=1)
        )

        return out


class AppearanceEncoder(nn.Module):
    """
    Encodes the clothing appearance (texture, color, pattern).
    Smaller network focused on appearance features.
    """
    def __init__(self, in_channels=3, feat_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 4
            nn.Conv2d(128, feat_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, feat_dim, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded