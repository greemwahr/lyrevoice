"""
training/losses.py

All loss functions for LyreVoice GAN training.

  LSGANDiscriminatorLoss  -- LSGAN loss for the Discriminator
  LSGANGeneratorLoss      -- LSGAN adversarial loss for the Generator
  ReconstructionLoss      -- L1 loss on mel-spectrogram (Generator)
  SpeakerConsistencyLoss  -- Cosine similarity on speaker embeddings (Generator)
  GeneratorTotalLoss      -- Combines all three Generator losses with lambda schedule

Why LSGAN over standard GAN (BCE):
  Standard GAN saturates when the Discriminator becomes too confident,
  causing vanishing gradients and Generator stagnation.
  LSGAN uses squared error instead, keeping gradients alive even when
  the Discriminator is winning -- critical for audio GAN stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────
# Discriminator Loss (LSGAN)
# ─────────────────────────────────────────────

class LSGANDiscriminatorLoss(nn.Module):
    """
    LSGAN Discriminator loss.

    L_D = mean[(D(real) - 1)^2] + mean[D(fake)^2]

    Real samples should score 1, fake samples should score 0.
    Summed over all scales of the Multi-Scale Discriminator.
    """

    def forward(
        self,
        real_scores: List[torch.Tensor],
        fake_scores: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            real_scores: list of score tensors for real mel-spectrograms (one per scale)
            fake_scores: list of score tensors for generated mel-spectrograms (one per scale)

        Returns:
            loss:  scalar discriminator loss
            stats: dict of per-scale losses for logging
        """
        total_loss = torch.tensor(0.0, device=real_scores[0].device)
        stats = {}

        for i, (real, fake) in enumerate(zip(real_scores, fake_scores)):
            real_loss = torch.mean((real - 1.0) ** 2)
            fake_loss = torch.mean(fake ** 2)
            scale_loss = real_loss + fake_loss
            total_loss = total_loss + scale_loss
            stats[f"d_loss_scale_{i}"] = scale_loss.item()
            stats[f"d_real_score_scale_{i}"] = real.mean().item()
            stats[f"d_fake_score_scale_{i}"] = fake.mean().item()

        stats["d_loss_total"] = total_loss.item()
        return total_loss, stats


# ─────────────────────────────────────────────
# Generator Losses
# ─────────────────────────────────────────────

class LSGANGeneratorLoss(nn.Module):
    """
    LSGAN adversarial loss for the Generator.

    L_adv = mean[(D(fake) - 1)^2]

    The Generator wants fake scores to be close to 1 (fool Discriminator).
    Summed over all scales.
    """

    def forward(
        self,
        fake_scores: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            fake_scores: list of score tensors for generated mels (one per scale)

        Returns:
            loss:  scalar adversarial loss
            stats: dict of per-scale losses for logging
        """
        total_loss = torch.tensor(0.0, device=fake_scores[0].device)
        stats = {}

        for i, fake in enumerate(fake_scores):
            scale_loss = torch.mean((fake - 1.0) ** 2)
            total_loss = total_loss + scale_loss
            stats[f"g_adv_loss_scale_{i}"] = scale_loss.item()

        stats["g_adv_loss_total"] = total_loss.item()
        return total_loss, stats


class ReconstructionLoss(nn.Module):
    """
    L1 reconstruction loss on mel-spectrograms.

    L_rec = mean(|mel_generated - mel_real|)

    Keeps the Generator grounded to real speech patterns,
    preventing it from drifting to produce bizarre but
    "discriminator-fooling" spectrograms.

    Applied with mask to ignore padding frames.
    """

    def forward(
        self,
        mel_generated: torch.Tensor,
        mel_real: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            mel_generated: (batch, n_mels, frames)
            mel_real:      (batch, n_mels, frames)
            mel_lengths:   (batch,) -- actual frame lengths (for masking)

        Returns:
            loss:  scalar reconstruction loss
            stats: dict for logging
        """
        batch_size, n_mels, max_frames = mel_generated.shape

        # Build mask to ignore padded frames
        mask = torch.zeros(batch_size, max_frames, device=mel_generated.device)
        for i, length in enumerate(mel_lengths):
            mask[i, :length] = 1.0
        mask = mask.unsqueeze(1).expand_as(mel_generated)

        loss = F.l1_loss(mel_generated * mask, mel_real * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-8)

        return loss, {"g_rec_loss": loss.item()}


class SpeakerConsistencyLoss(nn.Module):
    """
    Speaker consistency loss via Resemblyzer cosine similarity.

    L_spk = 1 - CosineSimilarity(embed_generated, embed_target)

    Minimizing this loss maximizes the similarity between the speaker
    embedding of the generated speech and the target speaker embedding,
    ensuring the generated voice sounds like the intended person.

    Note: Computing speaker embeddings during training requires converting
    generated mels back to audio (via vocoder). To avoid this overhead,
    we compare mel-space embeddings using a lightweight proxy: the mean
    of the mel-spectrogram across time, which correlates with voice quality.
    For full speaker consistency evaluation, the app uses Resemblyzer directly.
    """

    def forward(
        self,
        mel_generated: torch.Tensor,
        speaker_embedding_target: torch.Tensor,
        speaker_embedding_generated: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            mel_generated:               (batch, n_mels, frames) -- not used directly here
            speaker_embedding_target:    (batch, 256) -- Resemblyzer embedding of real voice
            speaker_embedding_generated: (batch, 256) -- Resemblyzer embedding of generated voice

        Returns:
            loss:  scalar speaker consistency loss
            stats: dict for logging
        """
        # Cosine similarity: 1 = identical, -1 = opposite
        cos_sim = F.cosine_similarity(
            speaker_embedding_generated,
            speaker_embedding_target,
            dim=-1,
        )
        # Loss = 1 - similarity (minimize to maximize similarity)
        loss = torch.mean(1.0 - cos_sim)

        return loss, {
            "g_spk_loss": loss.item(),
            "speaker_cosine_sim": cos_sim.mean().item(),
        }


# ─────────────────────────────────────────────
# Combined Generator Loss with Lambda Schedule
# ─────────────────────────────────────────────

class GeneratorTotalLoss(nn.Module):
    """
    Combined Generator loss with 3-phase lambda schedule.

    L_G = L_adv + λ1 * L_rec + λ2 * L_spk

    Lambda schedule (from config):
      Phase 1 (epochs 1-30):  λ1=10,  λ2=0.1
      Phase 2 (epochs 31-60): λ1=7,   λ2=0.5
      Phase 3 (epochs 61+):   λ1=5,   λ2=1.0
    """

    def __init__(self, config: dict):
        super().__init__()
        self.schedule = config["training"]["lambda_schedule"]
        self.adv_loss = LSGANGeneratorLoss()
        self.rec_loss = ReconstructionLoss()
        self.spk_loss = SpeakerConsistencyLoss()

    def get_lambdas(self, epoch: int) -> Tuple[float, float]:
        """Return (lambda_reconstruction, lambda_speaker) for the current epoch."""
        lambda_rec = self.schedule[0]["lambda_reconstruction"]
        lambda_spk = self.schedule[0]["lambda_speaker"]

        for phase in self.schedule:
            if epoch >= phase["epoch"]:
                lambda_rec = phase["lambda_reconstruction"]
                lambda_spk = phase["lambda_speaker"]
            else:
                break

        return lambda_rec, lambda_spk

    def forward(
        self,
        fake_scores: List[torch.Tensor],
        mel_generated: torch.Tensor,
        mel_real: torch.Tensor,
        mel_lengths: torch.Tensor,
        speaker_embedding_target: torch.Tensor,
        speaker_embedding_generated: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            fake_scores:                 list of Discriminator scores on generated mels
            mel_generated:               (batch, n_mels, frames)
            mel_real:                    (batch, n_mels, frames)
            mel_lengths:                 (batch,)
            speaker_embedding_target:    (batch, 256) -- real speaker embedding
            speaker_embedding_generated: (batch, 256) -- embedding of generated speech
            epoch:                       current training epoch (for lambda schedule)

        Returns:
            total_loss: scalar
            stats:      dict of all component losses for WandB logging
        """
        lambda_rec, lambda_spk = self.get_lambdas(epoch)

        adv_loss, adv_stats = self.adv_loss(fake_scores)
        rec_loss, rec_stats = self.rec_loss(mel_generated, mel_real, mel_lengths)
        spk_loss, spk_stats = self.spk_loss(
            mel_generated, speaker_embedding_target, speaker_embedding_generated
        )

        total_loss = adv_loss + lambda_rec * rec_loss + lambda_spk * spk_loss

        stats = {
            **adv_stats,
            **rec_stats,
            **spk_stats,
            "lambda_reconstruction": lambda_rec,
            "lambda_speaker": lambda_spk,
            "g_loss_total": total_loss.item(),
        }

        return total_loss, stats
