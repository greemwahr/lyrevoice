"""
data/preprocess.py

Handles all audio preprocessing for LyreVoice:
  - Loading and resampling audio files
  - Extracting mel-spectrograms
  - Normalizing and saving preprocessed data

Run this script once before training to convert raw VCTK/LJSpeech
audio into mel-spectrograms cached on disk.
"""

import os
import yaml
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_audio(file_path: str, sample_rate: int) -> np.ndarray:
    """Load an audio file and resample to target sample rate."""
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Convert a raw audio waveform to a mel-spectrogram.

    Returns a numpy array of shape (n_mels, time_frames) in log scale.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )
    # Convert to log scale (dB), clamp to avoid log(0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [-1, 1]
    mel_norm = mel_db / 80.0
    mel_norm = np.clip(mel_norm, -1.0, 1.0)
    return mel_norm.astype(np.float32)


def trim_silence(audio: np.ndarray, sample_rate: int, top_db: int = 25) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio_trimmed


def preprocess_vctk(config: dict) -> None:
    """
    Preprocess the VCTK dataset.

    VCTK structure:
      VCTK-Corpus/
        wav48_silence_trimmed/
          p225/
            p225_001_mic1.flac
            ...
        txt/
          p225/
            p225_001.txt
            ...
    """
    audio_cfg = config["audio"]
    data_cfg = config["data"]

    vctk_path = Path(data_cfg["vctk_path"])
    output_path = Path(data_cfg["preprocessed_path"]) / "vctk"
    output_path.mkdir(parents=True, exist_ok=True)

    wav_dir = vctk_path / "wav48_silence_trimmed"
    txt_dir = vctk_path / "txt"

    if not wav_dir.exists():
        raise FileNotFoundError(f"VCTK wav directory not found: {wav_dir}")

    speakers = sorted([d.name for d in wav_dir.iterdir() if d.is_dir()])
    print(f"Found {len(speakers)} speakers in VCTK")

    metadata = []

    for speaker in tqdm(speakers, desc="Processing VCTK speakers"):
        speaker_wav_dir = wav_dir / speaker
        speaker_out_dir = output_path / speaker
        speaker_out_dir.mkdir(exist_ok=True)

        for wav_file in sorted(speaker_wav_dir.glob("*.flac")):
            utterance_id = wav_file.stem.replace("_mic1", "").replace("_mic2", "")

            # Load transcript
            txt_file = txt_dir / speaker / f"{utterance_id}.txt"
            if not txt_file.exists():
                continue
            text = txt_file.read_text().strip()

            # Load and preprocess audio
            audio = load_audio(str(wav_file), audio_cfg["sample_rate"])
            audio = trim_silence(audio, audio_cfg["sample_rate"])

            # Skip clips that are too long
            max_samples = data_cfg["max_wav_length"] * audio_cfg["sample_rate"]
            if len(audio) > max_samples:
                continue

            # Extract mel-spectrogram
            mel = extract_mel_spectrogram(
                audio,
                sample_rate=audio_cfg["sample_rate"],
                n_mels=audio_cfg["n_mels"],
                n_fft=audio_cfg["n_fft"],
                hop_length=audio_cfg["hop_length"],
                win_length=audio_cfg["win_length"],
                fmin=audio_cfg["fmin"],
                fmax=audio_cfg["fmax"],
            )

            # Save mel as numpy array
            mel_path = speaker_out_dir / f"{wav_file.stem}.npy"
            np.save(mel_path, mel)

            metadata.append({
                "speaker": speaker,
                "utterance_id": utterance_id,
                "text": text,
                "mel_path": str(mel_path.relative_to(output_path.parent)),
                "wav_path": str(wav_file),
                "num_frames": mel.shape[1],
            })

    # Save metadata as a simple text file (pipe-separated)
    meta_path = Path(data_cfg["preprocessed_path"]) / "vctk_metadata.txt"
    with open(meta_path, "w") as f:
        f.write("speaker|utterance_id|text|mel_path|wav_path|num_frames\n")
        for entry in metadata:
            f.write(
                f"{entry['speaker']}|{entry['utterance_id']}|{entry['text']}|"
                f"{entry['mel_path']}|{entry['wav_path']}|{entry['num_frames']}\n"
            )

    print(f"VCTK preprocessing complete. {len(metadata)} utterances saved.")
    print(f"Metadata written to {meta_path}")


def preprocess_ljspeech(config: dict) -> None:
    """
    Preprocess the LJSpeech dataset.

    LJSpeech structure:
      LJSpeech-1.1/
        wavs/
          LJ001-0001.wav
          ...
        metadata.csv  (id|transcription|normalized_transcription)
    """
    audio_cfg = config["audio"]
    data_cfg = config["data"]

    lj_path = Path(data_cfg["ljspeech_path"])
    output_path = Path(data_cfg["preprocessed_path"]) / "ljspeech"
    output_path.mkdir(parents=True, exist_ok=True)

    wav_dir = lj_path / "wavs"
    meta_csv = lj_path / "metadata.csv"

    if not wav_dir.exists():
        raise FileNotFoundError(f"LJSpeech wav directory not found: {wav_dir}")

    metadata_out = []

    with open(meta_csv, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing LJSpeech"):
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        utterance_id, _, text = parts[0], parts[1], parts[2]

        wav_file = wav_dir / f"{utterance_id}.wav"
        if not wav_file.exists():
            continue

        audio = load_audio(str(wav_file), audio_cfg["sample_rate"])
        audio = trim_silence(audio, audio_cfg["sample_rate"])

        max_samples = data_cfg["max_wav_length"] * audio_cfg["sample_rate"]
        if len(audio) > max_samples:
            continue

        mel = extract_mel_spectrogram(
            audio,
            sample_rate=audio_cfg["sample_rate"],
            n_mels=audio_cfg["n_mels"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
            fmin=audio_cfg["fmin"],
            fmax=audio_cfg["fmax"],
        )

        mel_path = output_path / f"{utterance_id}.npy"
        np.save(mel_path, mel)

        metadata_out.append({
            "utterance_id": utterance_id,
            "text": text,
            "mel_path": str(mel_path.relative_to(Path(data_cfg["preprocessed_path"]).parent)),
            "wav_path": str(wav_file),
            "num_frames": mel.shape[1],
        })

    meta_path = Path(data_cfg["preprocessed_path"]) / "ljspeech_metadata.txt"
    with open(meta_path, "w") as f:
        f.write("utterance_id|text|mel_path|wav_path|num_frames\n")
        for entry in metadata_out:
            f.write(
                f"{entry['utterance_id']}|{entry['text']}|"
                f"{entry['mel_path']}|{entry['wav_path']}|{entry['num_frames']}\n"
            )

    print(f"LJSpeech preprocessing complete. {len(metadata_out)} utterances saved.")
    print(f"Metadata written to {meta_path}")


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

    # Always compute embeddings for ALL speakers — the dataset handles
    # speaker limiting itself, and the selected speakers depend on the
    # metadata file which may differ from the wav directory listing.

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


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


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
