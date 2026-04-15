"""
models/speaker_encoder.py

Wrapper around Resemblyzer for extracting speaker embeddings.

Resemblyzer encodes variable-length audio into a fixed 256-dim
embedding vector that captures the unique identity of a voice.
It is used here in two places:
  1. During training  -- encode reference wavs per batch item
  2. During inference -- encode the user's 3 sample sentences
                         to produce their voice fingerprint

The encoder is always frozen (no gradient updates).
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from pathlib import Path
from typing import List, Union

from resemblyzer import VoiceEncoder, preprocess_wav


class SpeakerEncoder(nn.Module):
    """
    Frozen Resemblyzer-based speaker encoder.

    Produces a 256-dimensional L2-normalized embedding for any
    speaker given one or more reference audio clips.
    """

    EMBEDDING_DIM = 256

    def __init__(self, device: torch.device = None):
        super().__init__()

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device

        # Resemblyzer runs on CPU internally; we move the embedding to target device
        self.encoder = VoiceEncoder(device="cpu")

        # Freeze all parameters -- encoder is never trained
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_wav_paths(self, wav_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of wav file paths into a single speaker embedding.

        Multiple files are averaged into one embedding, which is more
        robust than using a single reference clip.

        Args:
            wav_paths: List of paths to reference audio files (.wav or .flac)

        Returns:
            Tensor of shape (256,) -- L2-normalized speaker embedding
        """
        embeddings = []
        for path in wav_paths:
            wav = preprocess_wav(Path(path))
            emb = self.encoder.embed_utterance(wav)
            embeddings.append(emb)

        # Average embeddings and re-normalize
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        return torch.FloatTensor(mean_emb).to(self.device)

    @torch.no_grad()
    def encode_audio_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Encode a raw numpy audio array into a speaker embedding.

        Used during app inference when the user records audio directly
        (no file path available).

        Args:
            audio:       Raw waveform as float32 numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Tensor of shape (256,) -- L2-normalized speaker embedding
        """
        wav = preprocess_wav(audio, source_sr=sample_rate)
        emb = self.encoder.embed_utterance(wav)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return torch.FloatTensor(emb).to(self.device)

    @torch.no_grad()
    def encode_batch_wav_paths(
        self,
        batch_wav_paths: List[List[str]],
    ) -> torch.Tensor:
        """
        Encode a batch of speakers, each with multiple reference wavs.

        Args:
            batch_wav_paths: List of lists -- outer dim is batch,
                             inner dim is reference wav paths per speaker

        Returns:
            Tensor of shape (batch_size, 256)
        """
        embeddings = []
        for wav_paths in batch_wav_paths:
            emb = self.encode_wav_paths(wav_paths)
            embeddings.append(emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, wav_paths: List[List[str]]) -> torch.Tensor:
        """
        Forward pass for training loop compatibility.

        Args:
            wav_paths: Batch of reference wav path lists

        Returns:
            Tensor of shape (batch_size, 256)
        """
        return self.encode_batch_wav_paths(wav_paths)
