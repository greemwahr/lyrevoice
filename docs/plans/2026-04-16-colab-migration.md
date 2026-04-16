# Colab Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Colab notebook that trains LyreVoice on a T4 GPU using data from Google Drive, with zero changes to core model code.

**Architecture:** A single `train_colab.ipynb` notebook that: mounts Google Drive, installs pip dependencies, creates a Colab-specific config overlay pointing at Drive paths, and runs the existing `Trainer` class. All model/training code stays untouched — only paths and package manager differ.

**Tech Stack:** Google Colab, Google Drive, pip, PyTorch CUDA, WandB

---

## Prerequisites (user does manually)

Before running the notebook, the user must upload these to Google Drive:

```
My Drive/
└── lyrevoice/
    ├── datasets/
    │   └── VCTK-Corpus/          # 11 GB — raw wavs needed by speaker encoder
    │       └── wav48_silence_trimmed/
    ├── preprocessed/
    │   ├── vctk/                  # 7.4 GB — mel .npy files
    │   ├── vctk_metadata.txt
    │   ├── ljspeech/
    │   └── ljspeech_metadata.txt
    └── pretrained/
        ├── tacotron2_statedict.pt  # 113 MB
        ├── hifigan_generator.pt    # 56 MB
        └── hifigan_config.json
```

Total upload: ~18.5 GB to Google Drive.

---

### Task 1: Create the Colab notebook — setup cells

**Files:**
- Create: `notebooks/train_colab.ipynb`

**Step 1: Create notebook with setup cells**

Cell 1 — Mount Drive and clone repo:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repo
!git clone https://github.com/greemwahr/lyrevoice.git /content/lyrevoice
%cd /content/lyrevoice
```

Cell 2 — Install dependencies:
```python
# Install dependencies (pip, not uv — Colab doesn't have uv)
!pip install torch torchaudio --quiet
!pip install resemblyzer librosa soundfile pyyaml wandb gradio \
    frechet-audio-distance numpy scipy tqdm matplotlib setuptools --quiet
```

Cell 3 — Verify GPU:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Step 2: Commit**

```bash
git add notebooks/train_colab.ipynb
git commit -m "feat: add Colab training notebook — setup cells"
```

---

### Task 2: Create Colab config overlay

**Files:**
- Create: `configs/config_colab.yaml`

**Step 1: Create Colab-specific config**

This overrides only the paths that differ on Colab. The notebook will load the base config and merge this on top.

```yaml
# Colab path overrides — merged on top of configs/config.yaml
# All other settings (model, training, wandb) inherited from base config.

data:
  vctk_path: /content/drive/MyDrive/lyrevoice/datasets/VCTK-Corpus
  ljspeech_path: /content/drive/MyDrive/lyrevoice/datasets/LJSpeech-1.1
  preprocessed_path: /content/drive/MyDrive/lyrevoice/preprocessed

generator:
  tacotron2_checkpoint: /content/drive/MyDrive/lyrevoice/pretrained/tacotron2_statedict.pt

vocoder:
  checkpoint: /content/drive/MyDrive/lyrevoice/pretrained/hifigan_generator.pt
  config: /content/drive/MyDrive/lyrevoice/pretrained/hifigan_config.json

paths:
  checkpoints: /content/drive/MyDrive/lyrevoice/checkpoints
  logs: /content/lyrevoice/logs
```

**Step 2: Commit**

```bash
git add configs/config_colab.yaml
git commit -m "feat: add Colab config overlay with Drive paths"
```

---

### Task 3: Add config merge logic to train.py

**Files:**
- Modify: `scripts/train.py`

**Step 1: Add --config-override flag and deep merge**

Add a `--config-override` argument that loads a second YAML and merges it on top of the base config. This keeps the base config untouched.

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

Add argument:
```python
parser.add_argument(
    "--config-override",
    type=str,
    default=None,
    help="Path to override config YAML (merged on top of base config)",
)
```

After loading base config:
```python
if args.config_override:
    override = load_config(args.config_override)
    config = deep_merge(config, override)
```

**Step 2: Commit**

```bash
git add scripts/train.py
git commit -m "feat: add --config-override flag for Colab path overrides"
```

---

### Task 4: Fix hardcoded wav_path in metadata

**Files:**
- Modify: `data/dataset.py`

**Step 1: Make wav_path resolution use config paths**

The metadata file stores `wav_path` as `data/datasets/VCTK-Corpus/wav48_silence_trimmed/...` which is a local path. On Colab, the VCTK corpus is at a different location. The dataset should resolve wav paths relative to the configured `vctk_path`.

In `VCTKDataset.__getitem__()` where `ref_wavs` are built from `self.speaker_wavs`, the wav paths come from metadata. Update `__init__` to rewrite wav paths using the configured `vctk_path`:

```python
# In VCTKDataset.__init__, after loading metadata:
# Rewrite wav_path to use configured vctk_path
vctk_root = Path(self.data_cfg["vctk_path"])
for entry in all_entries:
    raw = entry["wav_path"]
    # Strip the local prefix up to and including "VCTK-Corpus/"
    relative = raw.split("VCTK-Corpus/", 1)[-1]
    entry["wav_path"] = str(vctk_root / relative)
```

**Step 2: Commit**

```bash
git add data/dataset.py
git commit -m "fix: resolve wav_path relative to configured vctk_path"
```

---

### Task 5: Add training cell to Colab notebook

**Files:**
- Modify: `notebooks/train_colab.ipynb`

**Step 1: Add WandB login cell**

```python
import wandb
wandb.login()
```

**Step 2: Add training cell**

```python
# Train with Colab config overlay
!python scripts/train.py \
    --config configs/config.yaml \
    --config-override configs/config_colab.yaml
```

**Step 3: Add resume cell (for reconnecting after disconnect)**

```python
# Resume training after Colab disconnect
# Find the latest checkpoint:
!ls -la /content/drive/MyDrive/lyrevoice/checkpoints/

# Then resume:
# !python scripts/train.py \
#     --config configs/config.yaml \
#     --config-override configs/config_colab.yaml \
#     --resume /content/drive/MyDrive/lyrevoice/checkpoints/lyrevoice_epoch_XXXX.pt
```

**Step 4: Commit**

```bash
git add notebooks/train_colab.ipynb
git commit -m "feat: complete Colab training notebook with resume support"
```

---

### Task 6: Push and test on Colab

**Step 1: Push all commits to GitHub**

```bash
git push origin main
```

**Step 2: Open Colab and test**

1. Go to colab.research.google.com
2. Open notebook from GitHub: `greemwahr/lyrevoice/notebooks/train_colab.ipynb`
3. Set runtime to T4 GPU
4. Run all setup cells
5. Verify GPU is detected as T4
6. Run training cell
7. Confirm WandB is logging and training speed is ~1-2s/step

---

## Data Upload Checklist

Before running the notebook, user must upload to Google Drive (`My Drive/lyrevoice/`):

- [ ] `datasets/VCTK-Corpus/wav48_silence_trimmed/` — raw wavs for speaker encoder (11 GB)
- [ ] `preprocessed/vctk/` — mel spectrograms (needs `vctk_metadata.txt` too)
- [ ] `preprocessed/ljspeech/` — mel spectrograms (needs `ljspeech_metadata.txt` too)
- [ ] `pretrained/tacotron2_statedict.pt`
- [ ] `pretrained/hifigan_generator.pt`
- [ ] `pretrained/hifigan_config.json`

Total: ~18.5 GB. Upload time depends on internet speed (1-3 hours typical).
