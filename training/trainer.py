"""
training/trainer.py

GAN training loop for LyreVoice.

Orchestrates the adversarial training between Generator and Discriminator:
  1. Generator produces a fake mel-spectrogram from text + speaker embedding
  2. Discriminator scores real and fake mels
  3. Discriminator is updated to better distinguish real from fake
  4. Generator is updated to better fool the Discriminator
     while staying close to real speech (reconstruction + speaker loss)

The lambda schedule is applied automatically based on current epoch.
All metrics are logged to WandB.
"""

import json
import os
import yaml
import torch
import torch.nn as nn

# Workaround: wandb server returns flags=None in the viewer query,
# causing json.loads(None) to crash. Patch before importing wandb.
_orig_json_loads = json.loads
def _safe_json_loads(s, *args, **kwargs):
    if s is None:
        return {}
    return _orig_json_loads(s, *args, **kwargs)
json.loads = _safe_json_loads

import wandb
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from models.generator import LyreVoiceGenerator, text_to_sequence
from models.discriminator import MultiScaleDiscriminator
from models.speaker_encoder import SpeakerEncoder
from models.vocoder import Vocoder
from training.losses import LSGANDiscriminatorLoss, GeneratorTotalLoss
from data.dataset import get_dataloader


class Trainer:
    """
    LyreVoice GAN Trainer.

    Manages the full training loop including:
      - Optimizer setup (separate for G and D)
      - Lambda-scheduled loss computation
      - WandB logging (losses, audio samples, spectrograms)
      - Checkpoint saving and resuming
    """

    def __init__(self, config: dict, resume_checkpoint: Optional[str] = None):
        self.config = config
        self.train_cfg = config["training"]

        # Device selection -- MPS for Apple M2, CUDA for GPU, else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Training on device: {self.device}")

        # Models
        self.speaker_encoder = SpeakerEncoder(device=self.device)
        self.generator = LyreVoiceGenerator(config, device=self.device)
        self.discriminator = MultiScaleDiscriminator(config).to(self.device)
        self.vocoder = Vocoder(config, device=self.device)

        # NOTE: torch.compile() disabled — NVIDIA Tacotron2 uses .item() calls
        # that cause excessive graph breaks, making compilation slower than eager mode.

        # Losses
        self.d_loss_fn = LSGANDiscriminatorLoss()
        self.g_loss_fn = GeneratorTotalLoss(config)

        # Optimizers -- separate learning rates for G and D
        self.optimizer_g = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=self.train_cfg["learning_rate_generator"],
            betas=(self.train_cfg["adam_beta1"], self.train_cfg["adam_beta2"]),
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.train_cfg["learning_rate_discriminator"],
            betas=(self.train_cfg["adam_beta1"], self.train_cfg["adam_beta2"]),
        )

        # AMP -- automatic mixed precision for CUDA Tensor Cores
        self.use_amp = (self.device.type == "cuda")
        self.scaler_g = torch.amp.GradScaler(enabled=self.use_amp)
        self.scaler_d = torch.amp.GradScaler(enabled=self.use_amp)

        # Data
        self.train_loader = get_dataloader(config, dataset_name="vctk", split="train")
        self.val_loader = get_dataloader(config, dataset_name="vctk", split="val")

        # State
        self.start_epoch = 1
        self.global_step = 0

        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

        # WandB
        self._init_wandb()

    def _init_wandb(self) -> None:
        wb_cfg = self.config["wandb"]
        try:
            wandb.init(
                project=wb_cfg["project"],
                entity=wb_cfg.get("entity"),
                config=self.config,
            )
            wandb.watch(self.generator, log="gradients", log_freq=200)
            wandb.watch(self.discriminator, log="gradients", log_freq=200)
            self.use_wandb = True
        except Exception as e:
            print(f"WARNING: WandB init failed ({e}). Training will continue without WandB.")
            self.use_wandb = False

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt_dir = Path(self.config["paths"]["checkpoints"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"lyrevoice_epoch_{epoch:04d}.pt"

        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "scaler_g_state_dict": self.scaler_g.state_dict() if self.use_amp else None,
            "scaler_d_state_dict": self.scaler_d.state_dict() if self.use_amp else None,
        }, str(ckpt_path))
        print(f"Checkpoint saved: {ckpt_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator_state_dict"])
        self.discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
        if self.use_amp and ckpt.get("scaler_g_state_dict"):
            self.scaler_g.load_state_dict(ckpt["scaler_g_state_dict"])
            self.scaler_d.load_state_dict(ckpt["scaler_d_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        print(f"Resumed from checkpoint: {checkpoint_path} (epoch {ckpt['epoch']})")

    def _text_batch_to_tensors(self, texts, device):
        """Convert a batch of text strings to padded tensors."""
        sequences = [text_to_sequence(t) for t in texts]
        lengths = torch.LongTensor([len(s) for s in sequences])
        max_len = lengths.max().item()
        padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.LongTensor(seq)
        return padded.to(device), lengths.to(device)

    def _train_step(self, batch: dict, epoch: int) -> dict:
        """Single training step -- one batch."""
        mel_real = batch["mel"].to(self.device)          # (B, n_mels, frames)
        mel_lengths = batch["mel_len"].to(self.device)   # (B,)
        texts = batch["text"]
        ref_wav_paths = batch["ref_wav_paths"]

        # ── Speaker embeddings (frozen Resemblyzer) ──────────────────────
        # Use pre-computed embeddings if available, otherwise fall back to live encoder
        if "speaker_embedding" in batch:
            speaker_embeddings = batch["speaker_embedding"].to(self.device)
        else:
            speaker_embeddings = self.speaker_encoder(ref_wav_paths)  # (B, 256)

        # ── Text to tensor ────────────────────────────────────────────────
        text_seqs, text_lengths = self._text_batch_to_tensors(texts, self.device)

        # Sort batch by text length (descending) -- required by Tacotron2's
        # pack_padded_sequence in the encoder.
        # .contiguous() after indexing -- fancy indexing can produce
        # non-contiguous tensors that cause .view() failures in PyTorch's
        # C++ autograd backward kernels on MPS.
        sorted_idx = torch.argsort(text_lengths, descending=True)
        text_seqs = text_seqs[sorted_idx].contiguous()
        text_lengths = text_lengths[sorted_idx].contiguous()
        speaker_embeddings = speaker_embeddings[sorted_idx].contiguous()
        mel_real = mel_real[sorted_idx].contiguous()
        mel_lengths = mel_lengths[sorted_idx].contiguous()

        # ── Generator forward pass ────────────────────────────────────────
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            mel_generated, mel_pre, gate_outputs = self.generator(
                text_sequences=text_seqs,
                text_lengths=text_lengths,
                speaker_embeddings=speaker_embeddings,
                mel_targets=mel_real,
                output_lengths=mel_lengths,
            )

        # ── Discriminator step ────────────────────────────────────────────
        self.optimizer_d.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            real_scores, _ = self.discriminator(mel_real.detach())
            fake_scores, _ = self.discriminator(mel_generated.detach())
            d_loss, d_stats = self.d_loss_fn(real_scores, fake_scores)

        self.scaler_d.scale(d_loss).backward()

        nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.train_cfg["grad_clip_threshold"],
        )
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        # ── Generator step ────────────────────────────────────────────────
        self.optimizer_g.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            fake_scores_for_g, _ = self.discriminator(mel_generated)

            spk_emb_generated = speaker_embeddings

            g_loss, g_stats = self.g_loss_fn(
                fake_scores=fake_scores_for_g,
                mel_generated=mel_generated,
                mel_real=mel_real,
                mel_lengths=mel_lengths,
                speaker_embedding_target=speaker_embeddings,
                speaker_embedding_generated=spk_emb_generated,
                epoch=epoch,
            )

        self.scaler_g.scale(g_loss).backward()

        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            self.train_cfg["grad_clip_threshold"],
        )
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        return {**d_stats, **g_stats}

    @torch.no_grad()
    def _log_audio_sample(self, batch: dict, epoch: int) -> None:
        """Generate and log a voice sample to WandB."""
        self.generator.eval()

        sample_text = batch["text"][0]
        ref_wavs = batch["ref_wav_paths"][0]
        speaker_id = batch["speaker_id"][0]

        spk_emb = self.speaker_encoder.encode_wav_paths(ref_wavs)
        mel = self.generator.infer(sample_text, spk_emb)
        audio = self.vocoder.mel_to_audio(mel)

        if self.use_wandb:
            wandb.log({
                "generated_audio": wandb.Audio(
                    audio,
                    sample_rate=self.config["audio"]["sample_rate"],
                    caption=f"[{speaker_id}] {sample_text[:60]}",
                ),
                "epoch": epoch,
            })

        self.generator.train()

    def train(self) -> None:
        """Full GAN training loop."""
        num_epochs = self.train_cfg["num_epochs"]
        log_interval = self.train_cfg["log_interval"]
        sample_interval = self.train_cfg["sample_interval"]
        checkpoint_interval = self.train_cfg["checkpoint_interval"]

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(self.start_epoch, num_epochs + 1):
            self.generator.train()
            self.discriminator.train()

            epoch_stats = {}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for batch in pbar:
                step_stats = self._train_step(batch, epoch)
                self.global_step += 1

                # Accumulate for epoch summary
                for k, v in step_stats.items():
                    epoch_stats[k] = epoch_stats.get(k, 0) + v

                pbar.set_postfix({
                    "G": f"{step_stats.get('g_loss_total', 0):.3f}",
                    "D": f"{step_stats.get('d_loss_total', 0):.3f}",
                    "λ1": f"{step_stats.get('lambda_reconstruction', 0):.1f}",
                    "λ2": f"{step_stats.get('lambda_speaker', 0):.2f}",
                })

                if self.global_step % log_interval == 0 and self.use_wandb:
                    wandb.log({**step_stats, "step": self.global_step, "epoch": epoch})

                if self.global_step % sample_interval == 0:
                    self._log_audio_sample(batch, epoch)

            # Epoch-level logging
            n_steps = len(self.train_loader)
            avg_g_loss = epoch_stats.get("g_loss_total", 0) / n_steps
            if self.use_wandb:
                wandb.log({
                    f"epoch/{k}": v / n_steps
                    for k, v in epoch_stats.items()
                } | {"epoch": epoch})

            if epoch % checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        print("Training complete.")
        if self.use_wandb:
            wandb.finish()
