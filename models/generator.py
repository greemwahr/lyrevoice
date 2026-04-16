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

        # Monkey-patch decoder to replace .view() with .reshape() --
        # NVIDIA's decoder uses .transpose().view() which fails on MPS
        # during backward because gradient tensors are non-contiguous.
        self._patch_decoder_view(self.tacotron2.decoder)

        self.to(device)

    @staticmethod
    def _patch_decoder_view(decoder):
        """Replace .view() with .reshape() in Tacotron2 decoder methods.

        NVIDIA's parse_decoder_inputs and parse_decoder_outputs use
        .transpose().view() which requires contiguous memory. On MPS,
        gradient tensors in the backward pass can be non-contiguous,
        causing RuntimeError. .reshape() handles both cases.
        """
        def patched_parse_decoder_inputs(self, decoder_inputs):
            # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
            decoder_inputs = decoder_inputs.transpose(1, 2)
            decoder_inputs = decoder_inputs.reshape(
                decoder_inputs.size(0),
                int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
            # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
            decoder_inputs = decoder_inputs.transpose(0, 1)
            return decoder_inputs

        def patched_parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
            # (T_out, B) -> (B, T_out)
            alignments = alignments.transpose(0, 1).contiguous()
            # (T_out, B) -> (B, T_out)
            gate_outputs = gate_outputs.transpose(0, 1).contiguous()
            # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
            mel_outputs = mel_outputs.transpose(0, 1).contiguous()
            # decouple frames per step
            shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
            mel_outputs = mel_outputs.reshape(*shape)
            # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
            mel_outputs = mel_outputs.transpose(1, 2)
            return mel_outputs, gate_outputs, alignments

        import types
        decoder.parse_decoder_inputs = types.MethodType(patched_parse_decoder_inputs, decoder)
        decoder.parse_decoder_outputs = types.MethodType(patched_parse_decoder_outputs, decoder)

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
        mel_targets: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass -- mirrors Tacotron2.forward() with speaker
        conditioning injected between encoder and decoder.

        Args:
            text_sequences:    (batch, max_text_len) -- tokenized text (sorted by descending length)
            text_lengths:      (batch,) -- actual text lengths (descending, for pack_padded_sequence)
            speaker_embeddings:(batch, 256) -- from Resemblyzer
            mel_targets:       (batch, n_mels, max_frames) -- for teacher forcing
            output_lengths:    (batch,) -- actual mel lengths

        Returns:
            mel_outputs_postnet: (batch, n_mels, frames) -- post-net mel
            mel_outputs:         (batch, n_mels, frames) -- raw decoder mel
            gate_outputs:        (batch, frames) -- stop token predictions
        """
        # ── Encode (same as Tacotron2.forward lines 663-665) ──────────────
        embedded_inputs = self.tacotron2.embedding(text_sequences).transpose(1, 2)
        encoder_outputs = self.tacotron2.encoder(embedded_inputs, text_lengths)

        # ── Inject speaker identity ───────────────────────────────────────
        conditioned_outputs = self.speaker_conditioning(
            encoder_outputs, speaker_embeddings
        )

        # ── Decode with teacher forcing (Decoder.forward returns 3 values) ─
        # .contiguous() required: batch sorting in trainer makes mel_targets
        # non-contiguous, but Decoder.parse_decoder_inputs does .transpose().view()
        mel_outputs, gate_outputs, _ = self.tacotron2.decoder(
            conditioned_outputs, mel_targets.contiguous(), memory_lengths=text_lengths
        )

        # ── Post-net refinement (residual, same as Tacotron2.forward l670-671)
        mel_outputs_postnet = self.tacotron2.postnet(mel_outputs) + mel_outputs

        # ── Mask padded frames (same as Tacotron2.parse_output) ───────────
        outputs = self.tacotron2.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs],
            output_lengths,
        )

        return outputs[1], outputs[0], outputs[2]

    @torch.no_grad()
    def infer(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a mel-spectrogram from text and a speaker embedding.

        Used during app inference -- no teacher forcing, pure autoregressive.
        Mirrors Tacotron2.infer() with speaker conditioning injected.

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

        # encoder.infer() — no pack_padded_sequence, no sort requirement
        embedded_inputs = self.tacotron2.embedding(sequence_tensor).transpose(1, 2)
        encoder_outputs = self.tacotron2.encoder.infer(embedded_inputs, lengths)
        conditioned_outputs = self.speaker_conditioning(encoder_outputs, spk_emb)

        # decoder.infer() returns 4 values: mel_outputs, gate_outputs, alignments, mel_lengths
        mel_outputs, _, _, _ = self.tacotron2.decoder.infer(
            conditioned_outputs, memory_lengths=lengths
        )
        mel_outputs_postnet = self.tacotron2.postnet(mel_outputs) + mel_outputs

        return mel_outputs_postnet.squeeze(0)
