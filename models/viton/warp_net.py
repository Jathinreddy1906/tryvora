# models/viton/warp_net.py
# ─────────────────────────────────────────────────────────
# Cloth Warping Network
#
# Simple explanation:
#   TPS (Thin Plate Spline) warping works like this:
#   Imagine the cloth image printed on a rubber sheet.
#   The network learns WHERE to push/pull each corner
#   of that rubber sheet to match the person's body.
#
#   Steps:
#   1. Extract features from person + clothing
#   2. Correlate features → find matching regions
#   3. Predict warp grid (where each pixel should move)
#   4. Apply grid to cloth image → warped cloth!
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────
# CORRELATION LAYER
# Compares person features vs clothing features
# Finds "what matches where" between the two images
# ─────────────────────────────────────────────────────────

class CorrelationLayer(nn.Module):
    """
    Computes feature correlation between person and clothing.

    Think of it as: for each location in the person image,
    find the most similar location in the clothing image.
    This tells us HOW to warp the clothing to match.
    """
    def __init__(self, max_displacement=4, stride=1):
        super().__init__()
        self.max_d  = max_displacement
        self.stride = stride

    def forward(self, feat_person, feat_cloth):
        b, c, h, w = feat_person.shape

        # Normalize features (makes comparison more stable)
        fp = F.normalize(feat_person, dim=1)
        fc = F.normalize(feat_cloth,  dim=1)

        # Correlation: dot product at each displacement
        correlations = []
        d = self.max_d

        for dy in range(-d, d+1, self.stride):
            for dx in range(-d, d+1, self.stride):
                # Shift clothing features
                fc_shifted = self._shift(fc, dy, dx)
                # Dot product correlation
                corr = (fp * fc_shifted).sum(dim=1, keepdim=True)
                correlations.append(corr)

        # Stack into correlation volume
        return torch.cat(correlations, dim=1)

    def _shift(self, x, dy, dx):
        """Shift feature map by (dy, dx) pixels."""
        b, c, h, w = x.shape
        shifted = torch.zeros_like(x)

        src_y_start = max(0, -dy)
        src_y_end   = min(h,  h - dy)
        src_x_start = max(0, -dx)
        src_x_end   = min(w,  w - dx)

        dst_y_start = max(0,  dy)
        dst_y_end   = min(h,  h + dy)
        dst_x_start = max(0,  dx)
        dst_x_end   = min(w,  w + dx)

        if (src_y_end > src_y_start and
                src_x_end > src_x_start and
                dst_y_end > dst_y_start and
                dst_x_end > dst_x_start):
            shifted[:, :,
                    dst_y_start:dst_y_end,
                    dst_x_start:dst_x_end] = \
                x[:, :,
                  src_y_start:src_y_end,
                  src_x_start:src_x_end]

        return shifted


# ─────────────────────────────────────────────────────────
# REGRESSION NETWORK
# Takes correlation map → predicts warp control points
# ─────────────────────────────────────────────────────────

class RegressionNet(nn.Module):
    """
    Predicts TPS (Thin Plate Spline) control point offsets.

    Input:  correlation volume [B, corr_channels, H, W]
    Output: control point offsets [B, num_points * 2]
            (x,y offsets for each control point)
    """
    def __init__(self, corr_channels, num_ctrl_pts=25):
        super().__init__()
        self.num_pts = num_ctrl_pts

        self.conv_layers = nn.Sequential(
            nn.Conv2d(corr_channels, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_ctrl_pts * 2),
        )

        # Initialize with small weights (near-identity warp)
        self._init_weights()

    def _init_weights(self):
        """Start with identity-like warping (no deformation)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.zeros_(m.bias)

    def forward(self, corr):
        feat = self.conv_layers(corr)
        feat = feat.view(feat.size(0), -1)  # flatten
        offsets = self.fc_layers(feat)
        return offsets


# ─────────────────────────────────────────────────────────
# TPS GRID GENERATOR
# Converts control points → full warp grid
# ─────────────────────────────────────────────────────────

class TPSGridGenerator(nn.Module):
    """
    Thin Plate Spline grid generator.

    Converts sparse control point movements into
    a dense warp field (one (x,y) offset per pixel).

    Think of it as: if I move 25 control points,
    what happens to ALL pixels in between?
    TPS gives a smooth, physically realistic answer.
    """
    def __init__(self, out_h=256, out_w=192, num_ctrl_pts=25,
                 device='cpu'):
        super().__init__()
        self.out_h     = out_h
        self.out_w     = out_w
        self.num_pts   = num_ctrl_pts
        self.device    = device

        # Create source control point grid
        self.register_buffer(
            'src_pts',
            self._make_grid(num_ctrl_pts)
        )
        self.register_buffer(
            'P_inv',
            self._compute_P_inv(self._make_grid(num_ctrl_pts))
        )

    def _make_grid(self, n):
        """Uniform grid of control points in [-1, 1]."""
        side = int(n ** 0.5)
        xs   = torch.linspace(-0.8, 0.8, side)
        ys   = torch.linspace(-0.8, 0.8, side)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        pts  = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return pts.float()  # [n, 2]

    def _compute_P_inv(self, src):
        """Precompute TPS kernel matrix inverse."""
        n   = src.shape[0]
        eps = 1e-6

        # Distance matrix
        dist  = torch.cdist(src, src)                   # [n, n]
        K     = dist ** 2 * torch.clamp(dist, min=eps).log()

        # Augmented system [K, P; P^T, 0]
        P     = torch.cat([
            torch.ones(n, 1),
            src
        ], dim=1)                                       # [n, 3]
        zeros = torch.zeros(3, 3)

        top   = torch.cat([K, P],    dim=1)             # [n, n+3]
        bot   = torch.cat([P.T, zeros], dim=1)          # [3, n+3]
        L     = torch.cat([top, bot], dim=0)            # [n+3, n+3]

        try:
            L_inv = torch.inverse(
                L + torch.eye(n+3) * eps
            )
        except RuntimeError:
            L_inv = torch.pinverse(L)

        return L_inv

    def forward(self, tgt_offsets):
        """
        tgt_offsets: [B, num_ctrl_pts * 2] predicted offsets
        Returns: sampling grid [B, H, W, 2]
        """
        b = tgt_offsets.shape[0]
        n = self.num_pts

        # Target control points = source + predicted offset
        offsets = tgt_offsets.view(b, n, 2)
        tgt_pts = self.src_pts.unsqueeze(0) + offsets * 0.1

        # Build target vector Y [B, n+3, 2]
        ones = torch.ones(b, n, 1, device=tgt_offsets.device)
        Y    = torch.cat([
            tgt_pts,
            torch.zeros(b, 3, 2, device=tgt_offsets.device)
        ], dim=1)

        # Solve for TPS weights
        # W = L_inv @ Y
        L_inv = self.P_inv.unsqueeze(0).expand(b, -1, -1)
        W     = torch.bmm(L_inv, Y)           # [B, n+3, 2]

        # Create output pixel grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.out_h,
                           device=tgt_offsets.device),
            torch.linspace(-1, 1, self.out_w,
                           device=tgt_offsets.device),
            indexing='ij'
        )
        # [H*W, 2]
        grid_pts = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=1
        )

        hw  = grid_pts.shape[0]
        src = self.src_pts.to(tgt_offsets.device)

        # Compute distances from grid to control points
        dist = torch.cdist(
            grid_pts.unsqueeze(0).expand(b, -1, -1),
            src.unsqueeze(0).expand(b, -1, -1)
        )  # [B, H*W, n]

        eps = 1e-6
        K   = dist**2 * torch.clamp(dist, min=eps).log()

        # Augmented design matrix
        ones_gp = torch.ones(b, hw, 1, device=tgt_offsets.device)
        P_grid  = torch.cat([ones_gp, grid_pts.unsqueeze(0)
                             .expand(b,-1,-1)], dim=2)
        design  = torch.cat([K, P_grid], dim=2)  # [B, H*W, n+3]

        # Compute warped positions
        warped  = torch.bmm(design, W)            # [B, H*W, 2]
        grid    = warped.view(b, self.out_h, self.out_w, 2)
        grid    = torch.clamp(grid, -1, 1)

        return grid


# ─────────────────────────────────────────────────────────
# FULL WARP NETWORK
# Combines all pieces into one model
# ─────────────────────────────────────────────────────────

class ClothWarpNet(nn.Module):
    """
    Complete cloth warping network.

    Pipeline:
      1. Extract features from person image (agnostic)
      2. Extract features from clothing image
      3. Compute feature correlation
      4. Predict TPS control point offsets
      5. Generate dense warp grid
      6. Apply grid to clothing → warped clothing!

    Input:
      person_agnostic : person image with shirt removed [B,3,H,W]
      cloth           : clothing image                  [B,3,H,W]

    Output:
      warped_cloth    : cloth deformed to fit person    [B,3,H,W]
      warp_grid       : the warp field used             [B,H,W,2]
    """
    def __init__(self, img_h=256, img_w=192,
                 num_ctrl_pts=25, device='cpu'):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w

        # Feature extractors
        from models.viton.unet import FeatureExtractorUNet
        self.person_feat = FeatureExtractorUNet(
            in_channels=3, out_channels=64, base_ch=32
        )
        self.cloth_feat  = FeatureExtractorUNet(
            in_channels=3, out_channels=64, base_ch=32
        )

        # Correlation
        max_d     = 4
        n_disps   = (2 * max_d + 1) ** 2
        self.corr = CorrelationLayer(max_displacement=max_d)

        # Regression (correlation → control points)
        self.regression = RegressionNet(
            corr_channels=n_disps,
            num_ctrl_pts=num_ctrl_pts
        )

        # TPS grid generator
        self.tps = TPSGridGenerator(
            out_h=img_h,
            out_w=img_w,
            num_ctrl_pts=num_ctrl_pts,
            device=device
        )

    def forward(self, person_agnostic, cloth):
        # 1. Extract features
        p_feat = self.person_feat(person_agnostic)
        c_feat = self.cloth_feat(cloth)

        # Downsample for correlation (faster)
        pf_small = F.avg_pool2d(p_feat, 4)
        cf_small = F.avg_pool2d(c_feat, 4)

        # 2. Correlate
        corr = self.corr(pf_small, cf_small)

        # 3. Predict control point offsets
        offsets = self.regression(corr)

        # 4. Generate warp grid
        grid = self.tps(offsets)

        # 5. Apply warp to clothing image
        warped = F.grid_sample(
            cloth, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped, grid


# ─────────────────────────────────────────────────────────
# TRY-ON GENERATOR
# Takes warped cloth + cleaned person → final output
# ─────────────────────────────────────────────────────────

class TryOnGenerator(nn.Module):
    """
    Blending network that combines:
      - Cleaned person (shirt removed)
      - Warped clothing
      - Segmentation mask
    → Produces final realistic try-on image

    This is a U-Net that learns to blend cloth onto person.
    """
    def __init__(self):
        super().__init__()

        # Input: person(3) + warped_cloth(3) + mask(1) = 7 channels
        self.enc1 = self._block(7,   64)
        self.enc2 = self._block(64,  128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.bottleneck = self._block(512, 512)

        self.dec4 = self._block(512+512, 256)
        self.dec3 = self._block(256+256, 128)
        self.dec2 = self._block(128+128, 64)
        self.dec1 = self._block(64+64,   32)

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )

        self.out  = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Tanh()
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, person, warped_cloth, mask):
        # Resize mask to match spatial dims
        mask_r = F.interpolate(
            mask, size=person.shape[2:],
            mode='bilinear', align_corners=True
        )

        # Concatenate inputs
        x = torch.cat([person, warped_cloth, mask_r], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([
            self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([
            self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([
            self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([
            self.up(d2), e1], dim=1))

        return self.out(d1)