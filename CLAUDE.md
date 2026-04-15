# LyreVoice - Claude Code Context

## Project Overview

LyreVoice is a voice cloning GAN for LIAI 1009 (Deep Learning) at Georgian College.
Users record 3 sentences → system clones their voice → speaks any text in that voice.

Full architecture decisions and rationale are in `docs/SPECIFICATION.md`.
Step-by-step setup and training instructions are in `docs/RUNBOOK.md`.

---

## Package Manager & Tooling

**Always use `uv` -- never `pip`.**

```bash
uv sync --group dev          # install dependencies
uv run python <script>       # run any script
uv run ruff check .          # lint
uv run ruff check --fix .    # lint + autofix
uv run ruff format .         # format
uv add <package>             # add a new dependency
```

Dependencies and ruff config live in `pyproject.toml`.
UTMOS (git-only package) is in `[tool.uv.sources]` as a dev dependency.

---

## Architecture at a Glance

```
User records 3 sentences
        ↓
[Resemblyzer] → 256-dim speaker embedding  (frozen, models/speaker_encoder.py)
        ↓
User types text
        ↓
[Tacotron2 Generator] + speaker embedding → mel-spectrogram  (models/generator.py)
        ↓
[HiFi-GAN Vocoder] → audio waveform  (frozen, models/vocoder.py)
```

GAN components:
- **Generator**: Pretrained Tacotron2 fine-tuned with a `SpeakerConditioningLayer` (only new trainable part of generator)
- **Discriminator**: Multi-Scale CNN built from scratch -- 3 scales, operates at full/2x/4x downsampled mel resolution (`models/discriminator.py`)
- **Loss**: LSGAN (not BCE) + L1 reconstruction + speaker cosine similarity

---

## Key Decisions (do not change without good reason)

| Decision | Choice | Why |
|----------|--------|-----|
| Audio representation | Mel-spectrogram | Industry standard for TTS/voice cloning |
| Speaker encoder | Resemblyzer (frozen) | Production-grade, focus budget on GAN |
| Generator backbone | Pretrained Tacotron2 | Already knows English speech |
| Discriminator | Multi-Scale CNN (3 scales) | Captures fine texture + coarse rhythm |
| Vocoder | HiFi-GAN (frozen) | 14MB, 167x real-time, MOS 4.57 |
| Dataset | VCTK (train) + LJSpeech (vocoder ref) | VCTK has 110 speakers; LJSpeech fits disk |
| Adversarial loss | LSGAN | Stable gradients when D dominates |
| Lambda schedule | 3-phase (epochs 1-30, 31-60, 61+) | Curriculum: speech first, voice identity second |
| Monitoring | WandB | Audio playback in dashboard |
| App | Gradio | Built-in mic recording, self-hostable |

---

## Project Structure

```
lyrevoice/
├── app/app.py                   # Gradio UI -- voice enrollment + TTS
├── checkpoints/
│   ├── pretrained/              # tacotron2_statedict.pt, hifigan_generator.pt
│   └── training/                # lyrevoice_epoch_XXXX.pt (saved every 10 epochs)
├── configs/config.yaml          # ALL hyperparameters live here -- edit this, not code
├── data/
│   ├── datasets/                # Raw VCTK-Corpus/ and LJSpeech-1.1/ (not committed)
│   ├── preprocessed/            # Mel .npy files + metadata .txt (not committed)
│   ├── dataset.py               # VCTKDataset, LJSpeechDataset, collate_fn
│   └── preprocess.py            # Run once: audio → mel-spectrograms on disk
├── docs/
│   ├── RUNBOOK.md               # Setup and training guide
│   └── SPECIFICATION.md         # Architecture decisions with rationale
├── models/
│   ├── discriminator.py         # MultiScaleDiscriminator + SingleScaleDiscriminator
│   ├── generator.py             # LyreVoiceGenerator, SpeakerConditioningLayer, text_to_sequence
│   ├── speaker_encoder.py       # SpeakerEncoder (Resemblyzer wrapper)
│   └── vocoder.py               # Vocoder (HiFi-GAN), HiFiGANGenerator, ResBlock
├── scripts/
│   ├── evaluate.py              # UTMOS + FAD + speaker cosine similarity
│   └── train.py                 # Entry point: uv run python scripts/train.py
├── training/
│   ├── losses.py                # LSGANDiscriminatorLoss, GeneratorTotalLoss, etc.
│   └── trainer.py               # Trainer class -- full GAN loop + WandB logging
├── CLAUDE.md                    # This file
└── pyproject.toml               # Dependencies + ruff config
```

---

## Common Commands

```bash
# Preprocessing (run once before first training)
uv run python data/preprocess.py --dataset all

# Training from scratch
uv run python scripts/train.py

# Resume from checkpoint
uv run python scripts/train.py --resume checkpoints/training/lyrevoice_epoch_0050.pt

# Evaluate a checkpoint
uv run python scripts/evaluate.py --checkpoint checkpoints/training/lyrevoice_epoch_0100.pt

# Launch the app (local)
uv run python app/app.py

# Launch the app (server, accessible externally)
uv run python app/app.py --server_name 0.0.0.0 --server_port 7860 \
  --checkpoint checkpoints/training/lyrevoice_epoch_0100.pt
```

---

## Config -- Most Frequently Edited Settings

All in `configs/config.yaml`:

```yaml
training:
  batch_size: 16           # Reduce to 8 if running out of memory
  num_epochs: 100
  learning_rate_generator: 0.0001
  learning_rate_discriminator: 0.0004
  checkpoint_interval: 10  # Save every N epochs

wandb:
  entity: your_wandb_username   # Must be set before training

data:
  vctk_path: data/datasets/VCTK-Corpus
  ljspeech_path: data/datasets/LJSpeech-1.1
```

---

## Device Handling

Device selection is automatic in all model files:
1. Apple MPS (M1/M2/M3) -- detected via `torch.backends.mps.is_available()`
2. NVIDIA CUDA -- detected via `torch.cuda.is_available()`
3. CPU -- fallback

No manual device configuration needed.

---

## Loss Function Summary

```
L_D = (D(real) - 1)² + D(fake)²                          # LSGAN Discriminator

L_G = (D(fake) - 1)²                                      # LSGAN adversarial
    + λ1 * L1(mel_generated, mel_real)                    # Reconstruction
    + λ2 * CosineSimilarity(spk_emb_gen, spk_emb_target)  # Speaker consistency

Lambda schedule:
  Epochs 1-30:  λ1=10,  λ2=0.1
  Epochs 31-60: λ1=7,   λ2=0.5
  Epochs 61+:   λ1=5,   λ2=1.0
```

---

## Evaluation Metrics

| Metric | Tool | Scale | Goal |
|--------|------|-------|------|
| Audio naturalness | UTMOS | 1-5 | Higher |
| GAN output quality | FAD (Fréchet Audio Distance) | 0-∞ | Lower |
| Voice cloning accuracy | Speaker Cosine Similarity (Resemblyzer) | 0-1 | Higher |

---

## Git Rules

- **Always commit changes** after completing any task
- **Never push** -- the user handles all pushes manually
- Active development branch: `claude/count-repo-files-Pa8A5`
