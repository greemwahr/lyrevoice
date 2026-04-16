# GPU-Optimized Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Maximize A100/H100 GPU utilization during training by eliminating the speaker encoder CPU bottleneck, enabling mixed-precision (FP16) training, adding multi-worker data prefetching, and using `torch.compile()` for kernel fusion — targeting ~1-2s/step down from ~28s/step on A100.

**Architecture:** Four optimizations layered on top of the existing training loop, each independent: (1) pre-compute speaker embeddings to disk, eliminating per-step CPU work; (2) AMP (Automatic Mixed Precision) with GradScaler for FP16 forward/backward passes on Tensor Cores; (3) increase DataLoader workers and enable persistent workers + prefetch; (4) `torch.compile()` on generator and discriminator for CUDA kernel fusion. All changes are gated behind config flags or hardware detection so local M2 training still works.

**Tech Stack:** PyTorch AMP, torch.compile, Resemblyzer, CUDA

---

## Current Bottleneck Analysis

| Bottleneck | Where | Impact |
|-----------|-------|--------|
| Speaker encoder on CPU | `trainer.py:163` — 384 wav loads + Resemblyzer per step | **~25s/step** (dominant) |
| FP32 everywhere | All forward/backward passes | A100/H100 Tensor Cores idle (designed for FP16/BF16) |
| 2 DataLoader workers | `config.yaml:78` | CPU underutilized, GPU starved between steps |
| No kernel fusion | Eager-mode PyTorch | Redundant memory reads/writes per op |

---

### Task 1: Pre-compute speaker embeddings in `data/preprocess.py`

**Files:**
- Modify: `data/preprocess.py`

**Step 1: Add `import torch` at the top**

Add after line 6 (`from tqdm import tqdm`):

```python
import torch
```

**Step 2: Add `deep_merge` and `--config-override` support**

Add before the `if __name__ == "__main__":` block:

```python
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base
```

Update the argparse block to add `--config-override` and `speaker_embeddings` choice:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio datasets for LyreVoice")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help="Path to override config YAML (merged on top of base config)",
    )
    parser.add_argument("--dataset", choices=["vctk", "ljspeech", "speaker_embeddings", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.config_override:
        override = load_config(args.config_override)
        cfg = deep_merge(cfg, override)

    if args.dataset in ("vctk", "all"):
        preprocess_vctk(cfg)
    if args.dataset in ("ljspeech", "all"):
        preprocess_ljspeech(cfg)
    if args.dataset in ("speaker_embeddings", "all"):
        preprocess_speaker_embeddings(cfg)
```

**Step 3: Add the pre-compute function**

Add after `preprocess_ljspeech()`:

```python
def preprocess_speaker_embeddings(config: dict) -> None:
    """
    Pre-compute speaker embeddings for all VCTK speakers.

    For each speaker, loads all their wav files, runs Resemblyzer,
    averages the per-utterance embeddings, and saves the result.
    Output: {speaker_id: tensor(256,)} saved as a .pt file.
    """
    from models.speaker_encoder import SpeakerEncoder
    from resemblyzer import preprocess_wav
    import random

    data_cfg = config["data"]
    vctk_path = Path(data_cfg["vctk_path"])
    wav_dir = vctk_path / "wav48_silence_trimmed"
    output_path = Path(data_cfg["preprocessed_path"])

    if not wav_dir.exists():
        raise FileNotFoundError(f"VCTK wav directory not found: {wav_dir}")

    encoder = SpeakerEncoder(device=torch.device("cpu"))
    speakers = sorted([d.name for d in wav_dir.iterdir() if d.is_dir()])

    # Limit speakers if configured (use same seed as dataset.py for consistency)
    max_speakers = data_cfg.get("max_speakers")
    if max_speakers and len(speakers) > max_speakers:
        random.seed(42)
        speakers = sorted(random.sample(sorted(speakers), max_speakers))

    print(f"Pre-computing speaker embeddings for {len(speakers)} speakers...")

    embeddings = {}
    for speaker in tqdm(speakers, desc="Speaker embeddings"):
        speaker_wav_dir = wav_dir / speaker
        wav_files = sorted(speaker_wav_dir.glob("*.flac"))

        if not wav_files:
            print(f"  Skipping {speaker}: no wav files found")
            continue

        spk_embs = []
        for wav_file in wav_files:
            try:
                wav = preprocess_wav(Path(wav_file))
                emb = encoder.encoder.embed_utterance(wav)
                spk_embs.append(emb)
            except Exception:
                continue

        if not spk_embs:
            print(f"  Skipping {speaker}: no valid embeddings")
            continue

        mean_emb = np.mean(spk_embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        embeddings[speaker] = torch.FloatTensor(mean_emb)

    out_file = output_path / "speaker_embeddings.pt"
    torch.save(embeddings, str(out_file))
    print(f"Saved {len(embeddings)} speaker embeddings to {out_file}")
```

**Step 4: Commit**

```bash
git add data/preprocess.py
git commit -m "feat: add speaker embedding pre-computation and --config-override to preprocess.py"
```

---

### Task 2: Update `data/dataset.py` to load and return pre-computed embeddings

**Files:**
- Modify: `data/dataset.py:63-228`

**Step 1: Load pre-computed embeddings in `VCTKDataset.__init__`**

After `self.speaker_to_idx` is built (line 130), add:

```python
# Load pre-computed speaker embeddings if available
emb_path = Path(self.data_cfg["preprocessed_path"]) / "speaker_embeddings.pt"
if emb_path.exists():
    self.speaker_embeddings = torch.load(str(emb_path), weights_only=True)
    print(f"  Loaded pre-computed speaker embeddings from {emb_path}")
else:
    self.speaker_embeddings = None
    print(f"  WARNING: No pre-computed embeddings at {emb_path}, using live encoder")
```

**Step 2: Return embedding in `__getitem__`**

Replace the return dict in `__getitem__` (line 151-158) with:

```python
result = {
    "mel": torch.FloatTensor(mel),
    "mel_len": torch.LongTensor([mel_len]),
    "text": entry["text"],
    "speaker_id": spk,
    "speaker_idx": torch.LongTensor([self.speaker_to_idx[spk]]),
    "ref_wav_paths": ref_wavs,
}
if self.speaker_embeddings and spk in self.speaker_embeddings:
    result["speaker_embedding"] = self.speaker_embeddings[spk]
return result
```

**Step 3: Update `collate_fn` to stack embeddings**

In `collate_fn()` (line 212-228), add after the `ref_wav_paths` line:

```python
if "speaker_embedding" in batch[0]:
    result["speaker_embedding"] = torch.stack([item["speaker_embedding"] for item in batch])
```

**Step 4: Commit**

```bash
git add data/dataset.py
git commit -m "feat: load and return pre-computed speaker embeddings from dataset"
```

---

### Task 3: Update `training/trainer.py` — pre-computed embeddings + AMP + torch.compile

This is the main task. Three optimizations are applied to the trainer.

**Files:**
- Modify: `training/trainer.py`

**Step 1: Use pre-computed embeddings when available**

In `_train_step()`, replace line 163:

```python
speaker_embeddings = self.speaker_encoder(ref_wav_paths)  # (B, 256)
```

With:

```python
# Use pre-computed embeddings if available, otherwise fall back to live encoder
if "speaker_embedding" in batch:
    speaker_embeddings = batch["speaker_embedding"].to(self.device)
else:
    speaker_embeddings = self.speaker_encoder(ref_wav_paths)  # (B, 256)
```

**Step 2: Add AMP (Automatic Mixed Precision)**

AMP runs forward passes and loss computation in FP16 on Tensor Cores, then scales gradients back to FP32 for the optimizer. This roughly doubles throughput on A100/H100 with no code-level FP16 management needed.

In `__init__()`, after the optimizer setup (after line 89), add:

```python
# AMP -- automatic mixed precision for CUDA Tensor Cores
self.use_amp = (self.device.type == "cuda")
self.scaler_g = torch.amp.GradScaler(enabled=self.use_amp)
self.scaler_d = torch.amp.GradScaler(enabled=self.use_amp)
```

In `_train_step()`, wrap the forward/backward passes with `torch.amp.autocast` and use the scalers. Replace the entire discriminator step block (lines 190-203):

```python
# ── Discriminator step ────────────────────────────────────────
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
```

Replace the generator step block (lines 205-231):

```python
# ── Generator step ────────────────────────────────────────────
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
```

Also wrap the generator forward pass (lines 180-187):

```python
# ── Generator forward pass ────────────────────────────────────
with torch.amp.autocast("cuda", enabled=self.use_amp):
    mel_generated, mel_pre, gate_outputs = self.generator(
        text_sequences=text_seqs,
        text_lengths=text_lengths,
        speaker_embeddings=speaker_embeddings,
        mel_targets=mel_real,
        output_lengths=mel_lengths,
    )
```

**Step 3: Add `torch.compile()` for kernel fusion**

`torch.compile()` fuses GPU operations to reduce memory reads/writes. On A100/H100 with PyTorch ≥2.0 this gives 10-30% speedup with zero code changes. We only apply it on CUDA — MPS doesn't support it.

In `__init__()`, after the model setup (after line 73), add:

```python
# torch.compile -- fuses CUDA kernels for ~10-30% speedup on A100/H100
if self.device.type == "cuda":
    try:
        self.generator = torch.compile(self.generator)
        self.discriminator = torch.compile(self.discriminator)
        print("Models compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile() not available: {e}")
```

**Step 4: Add AMP state to checkpoint save/load**

In `_save_checkpoint()` (line 120-133), add the scaler states:

```python
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
```

In `_load_checkpoint()` (line 135-143), restore them:

```python
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
```

**Step 5: Commit**

```bash
git add training/trainer.py
git commit -m "feat: add AMP, torch.compile, and pre-computed embedding support to trainer"
```

---

### Task 4: Update `configs/config_colab.yaml` with GPU-optimized settings

**Files:**
- Modify: `configs/config_colab.yaml`

**Step 1: Add DataLoader and training overrides**

```yaml
# Colab path overrides — merged on top of configs/config.yaml
# Data is extracted from Drive archives to /content/lyrevoice_data/ for fast local I/O.
# Checkpoints are saved to Drive so they survive Colab disconnects.

data:
  vctk_path: /content/lyrevoice_data/datasets/VCTK-Corpus
  ljspeech_path: /content/lyrevoice_data/datasets/LJSpeech-1.1
  preprocessed_path: /content/lyrevoice_data/preprocessed

generator:
  tacotron2_checkpoint: /content/lyrevoice_data/pretrained/tacotron2_statedict.pt

vocoder:
  checkpoint: /content/lyrevoice_data/pretrained/hifigan_generator.pt
  config: /content/lyrevoice_data/pretrained/hifigan_config.json

training:
  batch_size: 128
  num_workers: 4            # A100/H100 host CPUs can handle more workers

paths:
  checkpoints: /content/drive/MyDrive/lyrevoice/checkpoints
  logs: /content/lyrevoice/logs
```

**Step 2: Commit**

```bash
git add configs/config_colab.yaml
git commit -m "feat: add GPU-optimized training settings to Colab config"
```

---

### Task 5: Update Colab notebook with pre-compute cell and branch clone

**Files:**
- Modify: `notebooks/train_colab.ipynb`

**Step 1: Update the clone cell to use the feature branch**

Cell 1:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repo (feature branch)
!git clone -b speaker_encoder_optimiser https://github.com/greemwahr/lyrevoice.git /content/lyrevoice
%cd /content/lyrevoice
```

**Step 2: Add a pre-compute cell after the extraction cell (cell-4) and before WandB login (cell-5)**

New cell (insert between cell-4 and cell-5):

```python
# Pre-compute speaker embeddings (run once — takes ~5-10 min)
# Eliminates the Resemblyzer CPU bottleneck during training.
import os
emb_path = '/content/lyrevoice_data/preprocessed/speaker_embeddings.pt'
if os.path.exists(emb_path):
    print(f'Speaker embeddings already exist at {emb_path}, skipping.')
else:
    !python data/preprocess.py \
        --dataset speaker_embeddings \
        --config configs/config.yaml \
        --config-override configs/config_colab.yaml
    print('Done! Speaker embeddings pre-computed.')
```

**Step 3: Commit**

```bash
git add notebooks/train_colab.ipynb
git commit -m "feat: add speaker embedding pre-compute cell and feature branch clone to notebook"
```

---

### Task 6: Push branch and test on Colab

**Step 1: Push to GitHub**

```bash
git push -u origin speaker_encoder_optimiser
```

**Step 2: Test on Colab**

1. Open Colab, set runtime to A100 (or H100 if available)
2. Open notebook from GitHub: `greemwahr/lyrevoice` branch `speaker_encoder_optimiser`
3. Run all cells in order:
   - Mount Drive + clone
   - pip install
   - Verify GPU (expect A100 80GB or H100)
   - Extract archives
   - Pre-compute speaker embeddings (expect ~5-10 min, one-time)
   - WandB login
   - Train
4. **Verify these in the training output:**
   - `Models compiled with torch.compile()` — kernel fusion active
   - `Loaded pre-computed speaker embeddings` — no CPU bottleneck
   - Step time should be **~1-2s/step** (down from ~28s)
5. **Verify GPU utilization:** Colab sidebar should show GPU RAM usage significantly higher than 5.7 GB
6. Watch WandB for loss curves — verify they look similar to previous runs

**Expected performance with all optimizations on A100:**

| Metric | Before | After |
|--------|--------|-------|
| Step time | ~28s | ~1-2s |
| Steps/epoch (batch 128) | 174 | 174 |
| Time/epoch | ~83 min | ~3-6 min |
| 100 epochs | ~5.7 days | ~5-10 hours |
| GPU RAM usage | ~6 GB / 80 GB | ~20-40 GB / 80 GB |

---

## Summary of Changes

| File | Change |
|------|--------|
| `data/preprocess.py` | Add `preprocess_speaker_embeddings()`, `--config-override`, `deep_merge()` |
| `data/dataset.py` | Load `speaker_embeddings.pt` in init, return in `__getitem__`, stack in `collate_fn` |
| `training/trainer.py` | Pre-computed embeddings, AMP with GradScaler, `torch.compile()`, scaler checkpoint save/load |
| `configs/config_colab.yaml` | batch_size 128, num_workers 4 |
| `notebooks/train_colab.ipynb` | Feature branch clone, pre-compute cell |

**No changes to:** `models/speaker_encoder.py` (still used for inference/Gradio app), `models/generator.py`, `models/discriminator.py`, `training/losses.py`, `configs/config.yaml`
