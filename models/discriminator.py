"""
models/discriminator.py

Multi-Scale CNN Discriminator (MSD) -- built entirely from scratch.

The MSD consists of `num_scales` independent CNN discriminators, each
operating at a different resolution of the mel-spectrogram:
  - Scale 0: full resolution   (captures fine voice texture, timbre)
  - Scale 1: 2x downsampled    (captures mid-level patterns)
  - Scale 2: 4x downsampled    (captures coarse rhythm, prosody)

Each sub-discriminator outputs a scalar score per input.
The total discriminator loss is the sum over all scales.

This design is proven in MelGAN and HiFi-GAN literature to produce
much more natural-sounding voice than a single-scale discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────
# Single-Scale Sub-Discriminator
# ─────────────────────────────────────────────

class SingleScaleDiscriminator(nn.Module):
    """
    A single CNN discriminator that classifies mel-spectrograms as
    real or fake at one resolution.

    Architecture:
      Input (batch, 1, n_mels, frames)
        → Conv2d blocks with increasing channels
        → Flatten
        → Linear → scalar score

    We treat the mel-spectrogram as a single-channel 2D image,
    similar to how image GANs treat RGB images.
    """

    def __init__(
        self,
        base_channels: int = 32,
        max_channels: int = 512,
        kernel_size: int = 5,
        stride: int = 2,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # Layer 0: initial projection
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size=kernel_size,
                          stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # Strided downsampling blocks
        in_ch = base_channels
        for i in range(4):
            out_ch = min(in_ch * 2, max_channels)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2),
                    nn.InstanceNorm2d(out_ch, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_ch = out_ch

        # Final conv to scalar map
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, 1, n_mels, frames) -- mel-spectrogram

        Returns:
            score:          (batch, 1, h, w) -- discriminator output map
            feature_maps:   list of intermediate feature tensors (for feature matching loss)
        """
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        score = self.final_conv(x)
        return score, feature_maps


# ─────────────────────────────────────────────
# Multi-Scale Discriminator
# ─────────────────────────────────────────────

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD).

    Runs `num_scales` sub-discriminators on the mel-spectrogram at
    progressively downsampled resolutions. Each scale specializes in
    detecting different types of artifacts:
      - Fine scale: unnatural voice texture, wrong timbre
      - Mid scale:  inconsistent phoneme transitions
      - Coarse scale: wrong prosody, unnatural rhythm

    Used with LSGAN loss:
      L_D = sum_s [ (D_s(real) - 1)^2 + D_s(fake)^2 ]
      L_G = sum_s [ (D_s(fake) - 1)^2 ]
    """

    def __init__(self, config: dict):
        super().__init__()
        disc_cfg = config["discriminator"]

        self.num_scales = disc_cfg["num_scales"]
        self.downsample_factor = disc_cfg["downsample_factor"]

        # Build one sub-discriminator per scale
        self.discriminators = nn.ModuleList([
            SingleScaleDiscriminator(
                base_channels=disc_cfg["base_channels"],
                max_channels=disc_cfg["max_channels"],
                kernel_size=disc_cfg["kernel_size"],
                stride=disc_cfg["stride"],
            )
            for _ in range(self.num_scales)
        ])

        # Average pooling to downsample between scales
        self.downsamples = nn.ModuleList([
            nn.AvgPool2d(
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
                padding=0,
            )
            for _ in range(self.num_scales - 1)
        ])

    def forward(
        self, mel: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            mel: (batch, n_mels, frames) -- mel-spectrogram

        Returns:
            scores:       list of score tensors, one per scale
            feature_maps: list of feature map lists, one per scale
        """
        # Add channel dim: (batch, 1, n_mels, frames)
        x = mel.unsqueeze(1)

        all_scores = []
        all_feature_maps = []

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsamples[i - 1](x)
            score, fmaps = disc(x)
            all_scores.append(score)
            all_feature_maps.append(fmaps)

        return all_scores, all_feature_maps

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
