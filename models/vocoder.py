"""
models/vocoder.py

HiFi-GAN vocoder wrapper -- converts mel-spectrograms to audio waveforms.

HiFi-GAN is a pretrained neural vocoder that produces high-fidelity
audio at ~167x real-time speed with a 14MB model footprint.
It is used here as a frozen inference-only component.

Downloads the pretrained checkpoint automatically on first use.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union
import urllib.request


# ─────────────────────────────────────────────
# HiFi-GAN model URLs (NVIDIA pretrained)
# ─────────────────────────────────────────────
HIFIGAN_GENERATOR_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/dle/"
    "hifigan__pyt_dle/versions/21.08.0_amp/files/hifigan_gen_checkpoint_10000.pt"
)
HIFIGAN_CONFIG_URL = (
    "https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json"
)

# ─────────────────────────────────────────────
# Minimal HiFi-GAN Generator definition
# Reproduced from https://github.com/jik876/hifi-gan
# ─────────────────────────────────────────────

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=d,
                          padding=get_padding(kernel_size, d))
            )
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=get_padding(kernel_size, 1))
            )
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


import torch.nn.functional as F


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN V1 Generator.
    Input:  mel-spectrogram (batch, n_mels, frames)
    Output: waveform        (batch, 1, samples)
    """

    def __init__(self, h: dict):
        super().__init__()
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(h["num_mels"], h["upsample_initial_channel"], 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        in_ch = h["upsample_initial_channel"]
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            out_ch = in_ch // 2
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(in_ch, out_ch, k, u,
                                       padding=(k - u) // 2)
                )
            )
            in_ch = out_ch

        self.resblocks = nn.ModuleList()
        ch = h["upsample_initial_channel"]
        for i in range(len(self.ups)):
            ch //= 2
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"]):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(ch, 1, 7, 1, padding=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for up in self.ups:
            nn.utils.remove_weight_norm(up)
        for block in self.resblocks:
            for c in block.convs1:
                nn.utils.remove_weight_norm(c)
            for c in block.convs2:
                nn.utils.remove_weight_norm(c)
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)


# ─────────────────────────────────────────────
# LyreVoice Vocoder Wrapper
# ─────────────────────────────────────────────

# Default HiFi-GAN V1 config (matches LJSpeech pretrained weights)
DEFAULT_HIFIGAN_CONFIG = {
    "resblock": "1",
    "num_gpus": 0,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "sampling_rate": 22050,
    "fmin": 0,
    "fmax": 8000,
}


class Vocoder(nn.Module):
    """
    Frozen HiFi-GAN vocoder wrapper.

    Converts mel-spectrograms → audio waveforms.
    Always runs in eval mode with no gradient computation.
    """

    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device
        self.sample_rate = config["audio"]["sample_rate"]

        voc_cfg = config["vocoder"]
        ckpt_path = Path(voc_cfg["checkpoint"])
        config_path = Path(voc_cfg["config"])

        # Load or use default HiFi-GAN config
        if config_path.exists():
            with open(config_path, "r") as f:
                h = json.load(f)
        else:
            print("HiFi-GAN config not found, using default V1 config.")
            h = DEFAULT_HIFIGAN_CONFIG

        self.generator = HiFiGANGenerator(h)

        # Load pretrained weights
        if ckpt_path.exists():
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            # Handle checkpoints saved with 'generator' key
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            self.generator.load_state_dict(state_dict)
            print(f"HiFi-GAN loaded from {ckpt_path}")
        else:
            print(
                f"HiFi-GAN checkpoint not found at {ckpt_path}.\n"
                "Download it with: python scripts/download_pretrained.py"
            )

        self.generator.remove_weight_norm()
        self.generator.eval()

        # Freeze all parameters
        for param in self.generator.parameters():
            param.requires_grad = False

        self.generator.to(device)

    @torch.no_grad()
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convert a mel-spectrogram to a numpy audio waveform.

        Args:
            mel: (n_mels, frames) or (batch, n_mels, frames)

        Returns:
            audio: numpy array of shape (samples,) normalized to [-1, 1]
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel = mel.to(self.device)
        audio = self.generator(mel)  # (batch, 1, samples)
        audio = audio.squeeze().cpu().numpy()
        return audio.astype(np.float32)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning audio tensor.
        Args:
            mel: (batch, n_mels, frames)
        Returns:
            audio: (batch, 1, samples)
        """
        return self.generator(mel)
