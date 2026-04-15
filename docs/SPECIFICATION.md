# LyreVoice - Architecture Specification Document

**Project:** LyreVoice - Voice Cloning GAN  
**Course:** LIAI 1009 - Deep Learning  
**Assignment:** Development and Application of a Generative Adversarial Network  
**Weight:** 40% of grade  

---

## 1. Project Overview

LyreVoice is a voice cloning system that allows a user to speak 3 sample sentences,
capture the unique characteristics of their voice, and then synthesize new speech in
that person's voice from any input text. The system is deployed as a web application
where users can record their voice and receive cloned audio output in real time.

### Application Flow

```
User records 3 sample sentences
        ↓
[Resemblyzer] → speaker embedding (voice fingerprint)
        ↓
User types text
        ↓
[Pretrained Tacotron2 Generator] + speaker embedding → mel-spectrogram
        ↓
[HiFi-GAN Vocoder] → audio waveform
        ↓
User hears their cloned voice
```

---

## 2. Architecture Decisions

### 2.1 Audio Representation
- **Decision:** Mel-Spectrogram
- **Rationale:** Best balance of quality, trainability, and compatibility with modern
  TTS systems. Captures voice tone, pitch, and texture effectively. Well-supported
  by vocoders for conversion back to audio.

### 2.2 Speaker Encoder
- **Decision:** Resemblyzer (pretrained)
- **Rationale:** Production-grade voice embeddings without requiring additional training
  data or model development. Allows focus on the GAN component which is the core of
  the assignment. Resemblyzer is based on Google's GE2E (Generalized End-to-End) paper.

### 2.3 Generator
- **Decision:** Pretrained Tacotron2 (fine-tuned)
- **Rationale:** Tacotron2 is the industry standard sequence-to-sequence TTS synthesizer
  (text → mel-spectrogram). Using a pretrained model means it already knows how to
  produce intelligible speech -- fine-tuning with GAN loss teaches it to produce speech
  in specific target voices. NVIDIA's official PyTorch implementation is used.

### 2.4 Discriminator
- **Decision:** Multi-Scale CNN Discriminator (MSD)
- **Rationale:** Voice identity requires capturing both coarse patterns (speech rhythm,
  cadence, prosody) and fine details (voice texture, breathiness, timbre). A single-scale
  discriminator misses fine-grained voice characteristics. Multiple CNN discriminators
  operating at different mel-spectrogram resolutions address both levels.

### 2.5 Vocoder
- **Decision:** HiFi-GAN (pretrained)
- **Rationale:** Converts mel-spectrograms to high-fidelity audio waveforms. HiFi-GAN
  supersedes WaveGlow on every metric -- smaller (14MB vs 268MB), faster inference
  (167x real-time), and equal or better MOS score (4.57 vs 4.56). Critical for app
  quality since users hear their own voice.

### 2.6 Training Dataset
- **Decision:** VCTK + LJSpeech
- **Rationale:** VCTK provides 110 speakers (~44 hours) for speaker diversity required
  by voice cloning. LJSpeech provides clean single-speaker data for vocoder fine-tuning.
  LibriTTS (~60GB) was ruled out due to storage constraints.

---

## 3. GAN Training Design

### 3.1 Loss Functions

**Discriminator Loss (LSGAN):**
```
L_D = (D(real) - 1)² + D(fake)²
```

**Generator Loss (combined):**
```
L_G = (D(fake) - 1)²                                         # LSGAN adversarial
    + λ1 * L1(mel_generated, mel_real)                        # Reconstruction
    + λ2 * CosineSimilarity(embed_generated, embed_target)    # Speaker consistency
```

**Why LSGAN over standard BCE:**
- Prevents vanishing gradients when Discriminator outperforms Generator
- More stable training -- critical for audio GANs
- Penalizes samples based on distance from decision boundary rather than binary classification

### 3.2 Lambda Schedule

A 3-phase curriculum schedule is used so the Generator first learns to produce
intelligible speech, then refines voice identity:

| Phase | Epochs | λ1 (Reconstruction) | λ2 (Speaker Consistency) |
|-------|--------|---------------------|--------------------------|
| 1     | 1-30   | 10                  | 0.1                      |
| 2     | 31-60  | 7                   | 0.5                      |
| 3     | 61+    | 5                   | 1.0                      |

All schedule values are configurable via `configs/config.yaml`.

---

## 4. Tech Stack

| Purpose             | Library/Tool              | Notes                                      |
|---------------------|---------------------------|--------------------------------------------|
| Framework           | PyTorch                   | MPS backend for Apple M2                   |
| Audio processing    | Librosa + Torchaudio      | Librosa for mel analysis, torchaudio for   |
|                     |                           | data loading pipeline                      |
| Speaker encoder     | Resemblyzer               | Pretrained, frozen during training         |
| Generator backbone  | Tacotron2 (NVIDIA)        | Pretrained, fine-tuned with GAN loss       |
| Vocoder             | HiFi-GAN                  | Pretrained, frozen during training         |
| Training monitoring | Weights & Biases (WandB)  | Audio playback, loss curves, experiment    |
|                     |                           | comparison                                 |
| Config management   | PyYAML                    | All hyperparameters in config.yaml         |
| App UI              | Gradio                    | Self-hostable, open source (Apache 2.0)    |
| Audio recording     | Gradio built-in           | gr.Audio(source="microphone")              |
| Package manager     | uv (Astral)               | Replaces pip; pyproject.toml as single     |
|                     |                           | source of truth for dependencies           |
| Linter/Formatter    | Ruff (Astral)             | Replaces flake8, black, isort; configured  |
|                     |                           | in pyproject.toml                          |

**Development Environment:** Apple M2 MacBook Air 24GB

### Package Management Commands

```bash
# Install all dependencies
uv sync

# Install including dev tools (ruff, utmos)
uv sync --group dev

# Run a script
uv run python scripts/train.py
uv run python app/app.py

# Lint and format
uv run ruff check .
uv run ruff format .
```

---

## 5. Evaluation Metrics

| Dimension        | Metric                          | Description                                      |
|------------------|---------------------------------|--------------------------------------------------|
| Speaker similarity | Speaker Embedding Cosine      | Resemblyzer embeddings of generated vs real      |
|                  | Similarity                      | voice compared via cosine similarity (0-1)       |
| GAN quality      | FAD (Fréchet Audio Distance)    | Audio equivalent of FID, measures statistical    |
|                  |                                 | similarity between real and generated audio      |

> **Note:** UTMOS22 (MOS predictor) was removed -- it pins `torch==1.11.0` which is
> incompatible with Python 3.11+. FAD and Speaker Cosine Similarity together
> provide sufficient quantitative evaluation for this project.

---

## 6. Project Folder Structure

```
lyrevoice/
├── models/
│   ├── speaker_encoder.py       # Resemblyzer wrapper
│   ├── generator.py             # Tacotron2 fine-tuning logic
│   ├── discriminator.py         # Multi-Scale CNN Discriminator (from scratch)
│   └── vocoder.py               # HiFi-GAN wrapper
├── data/
│   ├── dataset.py               # VCTK + LJSpeech dataset classes
│   └── preprocess.py            # Mel-spectrogram extraction, audio preprocessing
├── training/
│   ├── trainer.py               # GAN training loop
│   └── losses.py                # LSGAN + reconstruction + speaker consistency losses
├── app/
│   └── app.py                   # Gradio web application
├── configs/
│   └── config.yaml              # All hyperparameters and lambda schedule
├── scripts/
│   ├── train.py                 # Training entry point
│   └── evaluate.py              # Evaluation entry point (UTMOS, FAD, cosine sim)
├── checkpoints/                 # Saved model weights
├── docs/
│   └── SPECIFICATION.md         # This document
└── README.md
```

---

## 7. Deployment

- **Development:** Local Gradio server on M2 MacBook (`python app/app.py`)
- **Production/Demo:** Self-hosted on personal server behind Nginx reverse proxy -- shared URL used for both production and class presentation

---

## 8. Assignment Deliverables Mapping

| Assignment Requirement       | LyreVoice Implementation                          |
|------------------------------|---------------------------------------------------|
| Choice of application        | Voice cloning / Text-to-Speech                    |
| GAN model development        | Multi-Scale Discriminator (from scratch) +        |
|                              | fine-tuned Tacotron2 Generator                    |
| Training the GAN             | LSGAN + reconstruction + speaker consistency loss |
|                              | with 3-phase lambda schedule                      |
| Quality assessment           | UTMOS + FAD + Speaker Cosine Similarity           |
| Creative application         | Deployable voice cloning web app (Gradio)         |
| Ethical considerations       | To be addressed in report                         |
| Report and presentation      | To be compiled during/after implementation        |
