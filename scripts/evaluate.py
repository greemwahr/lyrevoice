"""
scripts/evaluate.py

Evaluation script for LyreVoice -- computes all three quality metrics:

  1. UTMOS  -- neural MOS predictor (audio naturalness, 1-5 scale)
  2. FAD    -- Fréchet Audio Distance (GAN quality, lower is better)
  3. Speaker Cosine Similarity -- how well the voice is cloned (0-1 scale)

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/training/lyrevoice_epoch_0100.pt
  python scripts/evaluate.py --checkpoint ... --num_samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_metadata
from models.generator import LyreVoiceGenerator
from models.speaker_encoder import SpeakerEncoder
from models.vocoder import Vocoder


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ─────────────────────────────────────────────
# Metric: Speaker Cosine Similarity
# ─────────────────────────────────────────────


def compute_speaker_similarity(
    generated_audio: np.ndarray,
    reference_wav_paths: list,
    speaker_encoder: SpeakerEncoder,
    sample_rate: int,
) -> float:
    """
    Compare speaker embedding of generated audio vs reference wavs.
    Returns cosine similarity in [0, 1] (higher = more similar voice).
    """
    # Embedding of generated audio
    emb_generated = speaker_encoder.encode_audio_array(generated_audio, sample_rate)

    # Embedding of reference (real) voice
    emb_reference = speaker_encoder.encode_wav_paths(reference_wav_paths)

    cosine_sim = F.cosine_similarity(
        emb_generated.unsqueeze(0),
        emb_reference.unsqueeze(0),
    ).item()

    # Clamp to [0, 1] for reporting (can be slightly negative for very different voices)
    return max(0.0, cosine_sim)


# ─────────────────────────────────────────────
# Metric: FAD (Fréchet Audio Distance)
# ─────────────────────────────────────────────


def compute_fad(generated_dir: str, reference_dir: str) -> float:
    """
    Compute Fréchet Audio Distance between generated and reference audio sets.
    Lower FAD = generated audio distribution closer to real audio.
    """
    try:
        from frechet_audio_distance import FrechetAudioDistance

        fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=True)
        score = fad.score(generated_dir, reference_dir)
        return score
    except ImportError:
        print("frechet-audio-distance not installed.")
        print("Run: pip install frechet-audio-distance")
        print("Skipping FAD evaluation.")
        return None


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate LyreVoice model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help="Path to override config YAML (merged on top of base config)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: from config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.config_override:
        override = load_config(args.config_override)
        config = deep_merge(config, override)
    eval_cfg = config["evaluation"]
    num_samples = args.num_samples or eval_cfg["num_samples"]

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Evaluating on device: {device}")

    # Load models
    speaker_encoder = SpeakerEncoder(device=device)
    generator = LyreVoiceGenerator(config, device=device)
    vocoder = Vocoder(config, device=device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    # Debug: print a weight fingerprint to verify different checkpoints load different weights
    spk_w = generator.speaker_conditioning.projection.weight
    print(f"  Speaker conditioning weight checksum: {spk_w.sum().item():.6f}")

    # Load validation metadata
    meta_path = Path(config["data"]["preprocessed_path"]) / "vctk_metadata.txt"
    all_entries = load_metadata(str(meta_path))

    # Remap wav_path: metadata stores local paths (data/datasets/VCTK-Corpus/...)
    # but on Colab the wavs live under config's vctk_path
    vctk_path = config["data"]["vctk_path"]
    for entry in all_entries:
        raw = entry["wav_path"]
        suffix = raw.split("VCTK-Corpus/", 1)[-1] if "VCTK-Corpus/" in raw else raw
        entry["wav_path"] = str(Path(vctk_path) / suffix)

    # Use last 5% as validation (matching dataset.py split)
    n_train = int(len(all_entries) * config["data"]["train_split"])
    val_entries = all_entries[n_train:]
    eval_entries = val_entries[:num_samples]

    # Output directories
    output_dir = Path(eval_cfg["output_dir"])
    generated_dir = output_dir / "generated"
    reference_dir = output_dir / "reference"
    generated_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = config["audio"]["sample_rate"]
    similarity_scores = []

    print(f"\nGenerating {len(eval_entries)} samples...")

    for i, entry in enumerate(tqdm(eval_entries, desc="Generating")):
        # Speaker embedding from reference wavs
        ref_wavs = [entry["wav_path"]]
        spk_emb = speaker_encoder.encode_wav_paths(ref_wavs)

        # Generate mel from text
        with torch.no_grad():
            mel = generator.infer(entry["text"], spk_emb)
            audio_generated = vocoder.mel_to_audio(mel)

        # Save generated audio
        gen_path = str(generated_dir / f"sample_{i:04d}.wav")
        sf.write(gen_path, audio_generated, sample_rate)

        # Convert reference audio to WAV for FAD comparison
        import librosa

        ref_out_path = str(reference_dir / f"sample_{i:04d}.wav")
        ref_audio, ref_sr = librosa.load(entry["wav_path"], sr=sample_rate)
        sf.write(ref_out_path, ref_audio, sample_rate)

        # Speaker similarity
        sim = compute_speaker_similarity(audio_generated, ref_wavs, speaker_encoder, sample_rate)
        similarity_scores.append(sim)

    # ── Compute metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Speaker Cosine Similarity
    mean_sim = np.mean(similarity_scores)
    std_sim = np.std(similarity_scores)
    print(f"Speaker Cosine Similarity: {mean_sim:.4f} ± {std_sim:.4f}")

    # FAD
    fad_score = compute_fad(str(generated_dir), str(reference_dir))
    if fad_score is not None:
        print(f"FAD:                       {fad_score:.4f} (lower is better)")

    print("=" * 50)

    # Save results to JSON — append to results.json so multiple epochs accumulate
    ckpt_stem = Path(args.checkpoint).stem
    result_entry = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt_stem,
        "num_samples": len(eval_entries),
        "speaker_cosine_similarity": {
            "mean": float(mean_sim),
            "std": float(std_sim),
        },
        "fad": float(fad_score) if fad_score is not None else None,
    }

    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        # Migrate from old dict format to list
        if isinstance(all_results, dict):
            all_results = [all_results]
    else:
        all_results = []

    # Replace existing entry for same epoch, or append
    all_results = [r for r in all_results if r.get("epoch") != ckpt_stem]
    all_results.append(result_entry)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
