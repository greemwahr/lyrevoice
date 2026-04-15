# LyreVoice Training Runbook

Step-by-step guide to go from a fresh clone to a trained, running voice cloning model.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | macOS 13+, Ubuntu 20.04+, Windows 11 (WSL2) | macOS 14+ / Ubuntu 22.04 |
| RAM | 16 GB | 24 GB |
| Disk space | 25 GB free | 50 GB free |
| Python | 3.10 | 3.10 (pinned via `.python-version`) |
| GPU | None (CPU works, very slow) | Apple M2/M3, NVIDIA GPU (CUDA 11.8+) |

> **Apple M2 users:** Training uses MPS (Metal Performance Shaders) automatically.
> No additional GPU setup is needed.

> **NVIDIA GPU users:** Ensure CUDA drivers are installed before running `uv sync`.

---

## Step 1 -- Install uv

uv is the package manager used for this project. Install it once on your machine.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

---

## Step 2 -- Clone the Repository

```bash
git clone https://github.com/greemwahr/lyrevoice.git
cd lyrevoice
```

---

## Step 3 -- Install Dependencies

```bash
uv sync --group dev
```

This installs all core dependencies plus ruff.
uv creates a virtual environment automatically at `.venv/`.

Verify the environment is working:
```bash
uv run python -c "import torch; print(torch.__version__)"
```

---

## Step 4 -- Download Datasets

You need both VCTK (multi-speaker, for GAN training) and LJSpeech (single-speaker, for vocoder reference).

### 4a. VCTK Dataset (~11 GB)

**Kaggle (recommended -- fastest download):**

The dataset at https://www.kaggle.com/datasets/asadsama/vctk-corpus contains the correct
`wav48_silence_trimmed/` structure with FLAC files. Requires a free Kaggle account and
API token configured at `~/.kaggle/kaggle.json`.

```bash
mkdir -p data/datasets
uvx kaggle datasets download -d asadsama/vctk-corpus -p data/datasets/
```

After extraction, rename the folder if needed so it matches the expected path:
```bash
# Check what was extracted
ls data/datasets/

# Rename if the folder name differs from VCTK-Corpus
mv data/datasets/<extracted-folder-name> data/datasets/VCTK-Corpus
```

**Edinburgh DataShare (fallback):**

The Edinburgh DataShare server throttles downloads heavily after the first few minutes
(drops from MB/s to KB/s). Use wget with `-c` to resume if interrupted.

```bash
mkdir -p data/datasets
wget -c -P data/datasets \
  https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip

unzip data/datasets/VCTK-Corpus-0.92.zip -d data/datasets/
```

Expected structure (both sources):
```
data/datasets/VCTK-Corpus/
├── wav48_silence_trimmed/
│   ├── p225/
│   │   ├── p225_001_mic1.flac
│   │   └── ...
│   └── ...
└── txt/
    ├── p225/
    │   ├── p225_001.txt
    │   └── ...
    └── ...
```

### 4b. LJSpeech Dataset (~2.6 GB)

```bash
cd data/datasets
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
cd ../..
# Result: data/datasets/LJSpeech-1.1/
```

Expected structure:
```
data/datasets/LJSpeech-1.1/
├── wavs/
│   ├── LJ001-0001.wav
│   └── ...
└── metadata.csv
```

---

## Step 5 -- Download Pretrained Model Weights

### 5a. Tacotron2 (auto-downloaded)

Tacotron2 weights are downloaded automatically from NVIDIA's PyTorch Hub on first run.
No manual action needed -- just ensure you have an internet connection for the first training run.

The checkpoint will be saved to `checkpoints/pretrained/tacotron2_statedict.pt`.

### 5b. HiFi-GAN Vocoder (~14 MB)

**Download both files from the official Google Drive folder:**

1. Open the official Google Drive folder:
   https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing
2. Open the **`LJ_FT_T2_V1`** folder -- V1 fine-tuned with Tacotron2, matches our generator
3. Download both `generator` and `config.json` from inside that folder
4. Move them to the correct locations:
```bash
mkdir -p checkpoints/pretrained
mv ~/Downloads/generator   checkpoints/pretrained/hifigan_generator.pt
mv ~/Downloads/config.json checkpoints/pretrained/hifigan_config.json
```

> Use the `config.json` from the Drive folder, not the one from GitHub --
> it is guaranteed to match the downloaded weights exactly.

---

## Step 6 -- Configure WandB (Training Monitor)

WandB tracks your training losses and lets you listen to generated voice samples in real time.

1. Create a free account at https://wandb.ai
2. Get your API key from https://wandb.ai/settings → API keys
3. Add it to your shell profile:

```bash
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.zshrc
source ~/.zshrc
```

> No login command needed -- wandb picks up `WANDB_API_KEY` automatically when training starts.

4. Set your WandB username in `configs/config.yaml`:
```yaml
wandb:
  entity: your_wandb_username   # ← update this line
```

---

## Step 7 -- Update Data Paths in Config

Open `configs/config.yaml` and verify these paths match your setup:

```yaml
data:
  vctk_path: data/datasets/VCTK-Corpus          # ← check this
  ljspeech_path: data/datasets/LJSpeech-1.1     # ← check this
  preprocessed_path: data/preprocessed
```

If you extracted the datasets to a different location, update the paths accordingly.

---

## Step 8 -- Preprocess the Datasets

This step converts raw audio files into mel-spectrograms and saves them to disk.
**Run once before training.** Takes approximately 30--60 minutes depending on hardware.

```bash
# Preprocess both datasets
uv run python data/preprocess.py --dataset all

# Or preprocess individually
uv run python data/preprocess.py --dataset vctk
uv run python data/preprocess.py --dataset ljspeech
```

Expected output:
```
Found 110 speakers in VCTK
Processing VCTK speakers: 100%|████████| 110/110
VCTK preprocessing complete. 44085 utterances saved.
Metadata written to data/preprocessed/vctk_metadata.txt

Processing LJSpeech: 100%|████████| 13100/13100
LJSpeech preprocessing complete. 13099 utterances saved.
Metadata written to data/preprocessed/ljspeech_metadata.txt
```

---

## Step 9 -- Start Training

```bash
uv run python scripts/train.py
```

Training runs for 100 epochs by default (configurable in `configs/config.yaml`).

### What you will see

In the terminal:
```
Training on device: mps
VCTKDataset [train]: 41880 utterances, 110 speakers
Epoch 1/100: 100%|████| 2617/2617 [G: 4.231, D: 0.892, λ1: 10.0, λ2: 0.10]
Checkpoint saved: checkpoints/training/lyrevoice_epoch_0010.pt
```

In WandB (https://wandb.ai):
- Live loss curves for Generator and Discriminator
- Audio samples generated every 500 steps -- listen to voice quality improve
- Lambda schedule tracked automatically

### Adjusting training for your hardware

Edit `configs/config.yaml`:
```yaml
training:
  batch_size: 8       # Reduce if you run out of memory (default: 16)
  num_workers: 2      # Reduce on machines with fewer CPU cores (default: 4)
```

---

## Step 10 -- Resume Interrupted Training

If training is interrupted, resume from the last checkpoint:

```bash
uv run python scripts/train.py \
  --resume checkpoints/training/lyrevoice_epoch_0050.pt
```

Checkpoints are saved every 10 epochs to `checkpoints/training/`.

---

## Step 11 -- Evaluate the Model

Run evaluation after training completes (or at any checkpoint):

```bash
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/training/lyrevoice_epoch_0100.pt
```

This computes:
- **UTMOS** -- audio naturalness score (1-5, higher is better)
- **FAD** -- Fréchet Audio Distance (lower is better)
- **Speaker Cosine Similarity** -- voice cloning accuracy (0-1, higher is better)

Results are saved to `eval/outputs/results.json`.

---

## Step 12 -- Launch the App

```bash
# Local development (accessible only on your machine)
uv run python app/app.py

# Server deployment (accessible on your network)
uv run python app/app.py --server_name 0.0.0.0 --server_port 7860

# With a specific trained checkpoint
uv run python app/app.py \
  --checkpoint checkpoints/training/lyrevoice_epoch_0100.pt \
  --server_name 0.0.0.0 \
  --server_port 7860
```

Open your browser at `http://localhost:7860` (local) or `http://your-server-ip:7860` (server).

---

## Linting and Formatting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

---

## Project Structure Reference

```
lyrevoice/
├── app/app.py                   # Gradio web app (Step 12)
├── checkpoints/
│   ├── pretrained/              # Tacotron2 + HiFi-GAN weights (Step 5)
│   └── training/                # Saved checkpoints during training
├── configs/config.yaml          # All hyperparameters (edit before training)
├── data/
│   ├── datasets/                # Raw VCTK + LJSpeech audio (Step 4)
│   ├── preprocessed/            # Mel-spectrograms generated by preprocess.py
│   ├── dataset.py               # PyTorch Dataset classes
│   └── preprocess.py            # Preprocessing script (Step 8)
├── docs/
│   ├── RUNBOOK.md               # This file
│   └── SPECIFICATION.md         # Architecture decisions
├── eval/outputs/                # Evaluation results (Step 11)
├── models/
│   ├── discriminator.py         # Multi-Scale CNN Discriminator
│   ├── generator.py             # Speaker-conditioned Tacotron2
│   ├── speaker_encoder.py       # Resemblyzer wrapper
│   └── vocoder.py               # HiFi-GAN wrapper
├── scripts/
│   ├── evaluate.py              # Evaluation entry point (Step 11)
│   └── train.py                 # Training entry point (Step 9)
├── training/
│   ├── losses.py                # LSGAN + reconstruction + speaker losses
│   └── trainer.py               # GAN training loop
└── pyproject.toml               # Dependencies + ruff config
```

---

## Troubleshooting

**`ModuleNotFoundError` on import:**
```bash
uv sync --group dev   # Re-run to ensure all packages installed
```

**MPS out of memory (Apple Silicon):**
```yaml
# In configs/config.yaml, reduce:
training:
  batch_size: 8
```

**CUDA out of memory (NVIDIA GPU):**
```yaml
training:
  batch_size: 8
  num_workers: 2
```

**Tacotron2 download fails (network issue):**
- Download manually from NVIDIA NGC and place at `checkpoints/pretrained/tacotron2_statedict.pt`

**WandB connection error:**
- Training still works offline; WandB logs are queued and synced when connection resumes
- To disable WandB entirely: `uv run wandb disabled`

**Training loss not decreasing after 20+ epochs:**
- Check WandB audio samples -- if they sound like noise, lower the learning rate in `configs/config.yaml`
- Try reducing `learning_rate_generator` from `0.0001` to `0.00005`
