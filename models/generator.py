"""
models/generator.py

LyreVoice Generator -- Tacotron2 fine-tuned for speaker-conditioned TTS.

The pretrained NVIDIA Tacotron2 model is loaded and extended with a
speaker conditioning layer that injects the Resemblyzer embedding into
the encoder output before decoding. This allows the model to generate
mel-spectrograms that sound like a specific target speaker.

Architecture:
  Text (characters) → [Tacotron2 Encoder] → encoder_outputs
  encoder_outputs + speaker_embedding → [Speaker Projection] → conditioned_outputs
  conditioned_outputs → [Tacotron2 Decoder] → mel-spectrogram + gate (stop token)

The speaker projection is a simple linear layer that projects
(encoder_dim + speaker_dim) → encoder_dim, keeping the decoder unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# Tacotron2 character vocabulary
# ─────────────────────────────────────────────

_SYMBOLS = (
    "_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'\",-.?:; "
)
SYMBOL_TO_ID = {s: i for i, s in enumerate(_SYMBOLS)}


def text_to_sequence(text: str) -> list:
    """Convert a text string to a list of symbol IDs."""
    return [SYMBOL_TO_ID[c] for c in text if c in SYMBOL_TO_ID]


# ─────────────────────────────────────────────
# Speaker Conditioning Layer
# ─────────────────────────────────────────────

class SpeakerConditioningLayer(nn.Module):
    """
    Projects (encoder_output + speaker_embedding) → encoder_dim.

    This layer is the only new trainable component added on top of Tacotron2.
    It learns to inject speaker identity into the encoder outputs before
    the attention-based decoder processes them.
    """

    def __init__(self, encoder_dim: int, speaker_dim: int):
        super().__init__()
        self.projection = nn.Linear(encoder_dim + speaker_dim, encoder_dim)
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_outputs:   (batch, time, encoder_dim)
            speaker_embedding: (batch, speaker_dim)

        Returns:
            conditioned: (batch, time, encoder_dim)
        """
        # Expand speaker embedding across the time dimension
        time_steps = encoder_outputs.size(1)
        spk_expanded = speaker_embedding.unsqueeze(1).expand(-1, time_steps, -1)

        # Concatenate and project back to encoder_dim
        combined = torch.cat([encoder_outputs, spk_expanded], dim=-1)
        conditioned = self.projection(combined)
        return self.norm(conditioned)


# ─────────────────────────────────────────────
# LyreVoice Generator
# ─────────────────────────────────────────────

class LyreVoiceGenerator(nn.Module):
    """
    Speaker-conditioned mel-spectrogram generator.

    Wraps NVIDIA's pretrained Tacotron2 and adds speaker conditioning.
    Only the SpeakerConditioningLayer and the Tacotron2 decoder are
    updated during GAN fine-tuning; the encoder is frozen to preserve
    the language representations learned during pretraining.
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
        self.config = config
        gen_cfg = config["generator"]

        encoder_dim = gen_cfg["encoder_embedding_dim"]
        speaker_dim = gen_cfg["speaker_embedding_dim"]

        # Load pretrained Tacotron2
        self.tacotron2 = self._load_tacotron2(gen_cfg["tacotron2_checkpoint"])

        # Speaker conditioning -- the only new trainable layer
        self.speaker_conditioning = SpeakerConditioningLayer(encoder_dim, speaker_dim)

        # Freeze the Tacotron2 encoder -- preserve pretrained language knowledge
        for param in self.tacotron2.encoder.parameters():
            param.requires_grad = False

        # Decoder remains trainable -- it learns to use speaker information
        for param in self.tacotron2.decoder.parameters():
            param.requires_grad = True

        self.to(device)

    def _load_tacotron2(self, checkpoint_path: str):
        """
        Load NVIDIA pretrained Tacotron2.

        Downloads from Torch Hub if checkpoint not found locally.
        """
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            # Load from local checkpoint
            tacotron2 = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub",
                "nvidia_tacotron2",
                model_math="fp32",
                pretrained=False,
            )
            state_dict = torch.load(str(ckpt), map_location="cpu")
            tacotron2.load_state_dict(state_dict)
            print(f"Tacotron2 loaded from local checkpoint: {ckpt}")
        else:
            # Download pretrained weights manually to avoid NVIDIA's
            # torch.load() call which lacks map_location and fails on non-CUDA
            print("Downloading pretrained Tacotron2 from NVIDIA Torch Hub...")
            CKPT_URL = (
                "https://api.ngc.nvidia.com/v2/models/nvidia/"
                "tacotron2_pyt_ckpt_fp32/versions/19.09.0/files/"
                "nvidia_tacotron2pyt_fp32_20190427"
            )
            # Load architecture only (no weights)
            tacotron2 = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub",
                "nvidia_tacotron2",
                model_math="fp32",
                pretrained=False,
            )
            # Download and load weights with map_location=cpu
            checkpoint = torch.hub.load_state_dict_from_url(
                CKPT_URL, map_location="cpu"
            )
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            # Strip "module." prefix from DataParallel-saved checkpoint
            checkpoint = {
                k.replace("module.", "", 1): v for k, v in checkpoint.items()
            }
            tacotron2.load_state_dict(checkpoint)
            # Save for future use
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tacotron2.state_dict(), str(ckpt))
            print(f"Tacotron2 checkpoint saved to {ckpt}")

        return tacotron2

    def forward(
        self,
        text_sequences: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_targets: Optional[torch.Tensor] = None,
        output_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the speaker-conditioned generator.

        Args:
            text_sequences:    (batch, max_text_len) -- tokenized text
            text_lengths:      (batch,) -- actual text lengths
            speaker_embeddings:(batch, 256) -- from Resemblyzer
            mel_targets:       (batch, n_mels, max_frames) -- for teacher forcing
            output_lengths:    (batch,) -- actual mel lengths

        Returns:
            mel_outputs:       (batch, n_mels, frames) -- post-net mel
            mel_outputs_pre:   (batch, n_mels, frames) -- pre post-net mel
            gate_outputs:      (batch, frames) -- stop token predictions
        """
        # Embed text (int token IDs → float embeddings) then encode
        embedded_inputs = self.tacotron2.embedding(text_sequences).transpose(1, 2)
        encoder_outputs = self.tacotron2.encoder(embedded_inputs, text_lengths)

        # Inject speaker identity into encoder outputs
        conditioned_outputs = self.speaker_conditioning(
            encoder_outputs, speaker_embeddings
        )

        # Decode conditioned outputs to mel-spectrogram
        if mel_targets is not None:
            # Teacher forcing during training
            mel_outputs_pre, gate_outputs, _ = self.tacotron2.decoder(
                conditioned_outputs, mel_targets, memory_lengths=text_lengths
            )
        else:
            # Autoregressive inference
            mel_outputs_pre, gate_outputs, _ = self.tacotron2.decoder.infer(
                conditioned_outputs, memory_lengths=text_lengths
            )

        # Post-net refinement
        mel_outputs = self.tacotron2.postnet(mel_outputs_pre) + mel_outputs_pre

        return mel_outputs, mel_outputs_pre, gate_outputs

    @torch.no_grad()
    def infer(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a mel-spectrogram from text and a speaker embedding.

        Used during app inference -- no teacher forcing, pure autoregressive.

        Args:
            text:              Input text string
            speaker_embedding: (256,) tensor from SpeakerEncoder

        Returns:
            mel: (n_mels, frames) mel-spectrogram tensor
        """
        self.eval()

        sequence = text_to_sequence(text)
        sequence_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([len(sequence)]).to(self.device)
        spk_emb = speaker_embedding.unsqueeze(0).to(self.device)

        embedded_inputs = self.tacotron2.embedding(sequence_tensor).transpose(1, 2)
        encoder_outputs = self.tacotron2.encoder(embedded_inputs, lengths)
        conditioned_outputs = self.speaker_conditioning(encoder_outputs, spk_emb)

        mel_outputs_pre, _, _ = self.tacotron2.decoder.infer(
            conditioned_outputs, memory_lengths=lengths
        )
        mel_outputs = self.tacotron2.postnet(mel_outputs_pre) + mel_outputs_pre

        return mel_outputs.squeeze(0)
