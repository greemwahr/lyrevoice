"""
data/dataset.py

PyTorch Dataset classes for LyreVoice training.

VCTKDataset  -- multi-speaker dataset for GAN training
LJSpeechDataset -- single-speaker dataset for vocoder fine-tuning
LyreVoiceDataset -- combined dataset wrapping both
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_metadata(meta_path: str) -> List[Dict]:
    """Load pipe-separated metadata file into list of dicts."""
    entries = []
    with open(meta_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return entries

    headers = lines[0].strip().split("|")
    for line in lines[1:]:
        parts = line.strip().split("|")
        if len(parts) != len(headers):
            continue
        entries.append(dict(zip(headers, parts)))
    return entries


def pad_or_trim_mel(mel: np.ndarray, max_frames: int) -> Tuple[np.ndarray, int]:
    """
    Pad or trim a mel-spectrogram to a fixed number of frames.
    Returns the processed mel and the original (unpadded) length.
    """
    n_mels, num_frames = mel.shape
    original_len = num_frames

    if num_frames > max_frames:
        mel = mel[:, :max_frames]
        original_len = max_frames
    elif num_frames < max_frames:
        pad_width = max_frames - num_frames
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=-1.0)

    return mel, original_len


class VCTKDataset(Dataset):
    """
    Multi-speaker VCTK dataset.

    Each item returns:
      - mel: mel-spectrogram of the utterance (n_mels, max_frames)
      - mel_len: original length before padding
      - text: raw transcript string
      - speaker_id: speaker label string (e.g. "p225")
      - speaker_wavs: list of wav paths for this speaker (used by speaker encoder)
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        max_frames: int = 600,
    ):
        self.config = config
        self.split = split
        self.max_frames = max_frames
        self.audio_cfg = config["audio"]
        self.data_cfg = config["data"]

        meta_path = Path(self.data_cfg["preprocessed_path"]) / "vctk_metadata.txt"
        all_entries = load_metadata(str(meta_path))

        # Group entries by speaker
        speaker_to_entries: Dict[str, List[Dict]] = {}
        for entry in all_entries:
            spk = entry["speaker"]
            speaker_to_entries.setdefault(spk, []).append(entry)

        # Limit number of speakers if configured
        max_speakers = self.data_cfg.get("max_speakers")
        if max_speakers and len(speaker_to_entries) > max_speakers:
            all_spks = sorted(speaker_to_entries.keys())
            random.seed(42)
            selected = random.sample(all_spks, max_speakers)
            speaker_to_entries = {s: speaker_to_entries[s] for s in selected}

        # Build speaker wav lookup for speaker encoder enrollment
        self.speaker_wavs: Dict[str, List[str]] = {
            spk: [e["wav_path"] for e in entries]
            for spk, entries in speaker_to_entries.items()
        }

        # Train/val split per speaker
        train_ratio = self.data_cfg["train_split"]
        self.entries = []
        for spk, entries in speaker_to_entries.items():
            random.seed(42)
            random.shuffle(entries)
            n_train = max(1, int(len(entries) * train_ratio))
            if split == "train":
                self.entries.extend(entries[:n_train])
            else:
                self.entries.extend(entries[n_train:])

        self.speakers = sorted(speaker_to_entries.keys())
        self.speaker_to_idx = {spk: i for i, spk in enumerate(self.speakers)}

        print(f"VCTKDataset [{split}]: {len(self.entries)} utterances, "
              f"{len(self.speakers)} speakers")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]

        # Load preprocessed mel-spectrogram
        mel_path = Path(self.data_cfg["preprocessed_path"]) / entry["mel_path"]
        mel = np.load(mel_path)
        mel, mel_len = pad_or_trim_mel(mel, self.max_frames)

        # Sample 3 reference wav paths for speaker encoder
        spk = entry["speaker"]
        all_wavs = self.speaker_wavs[spk]
        ref_wavs = random.sample(all_wavs, min(3, len(all_wavs)))

        return {
            "mel": torch.FloatTensor(mel),
            "mel_len": torch.LongTensor([mel_len]),
            "text": entry["text"],
            "speaker_id": spk,
            "speaker_idx": torch.LongTensor([self.speaker_to_idx[spk]]),
            "ref_wav_paths": ref_wavs,
        }


class LJSpeechDataset(Dataset):
    """
    Single-speaker LJSpeech dataset.
    Used for vocoder fine-tuning and baseline quality checks.

    Each item returns:
      - mel: mel-spectrogram (n_mels, max_frames)
      - mel_len: original length before padding
      - text: transcript string
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        max_frames: int = 600,
    ):
        self.config = config
        self.split = split
        self.max_frames = max_frames
        self.data_cfg = config["data"]

        meta_path = Path(self.data_cfg["preprocessed_path"]) / "ljspeech_metadata.txt"
        all_entries = load_metadata(str(meta_path))

        train_ratio = self.data_cfg["train_split"]
        n_train = max(1, int(len(all_entries) * train_ratio))

        if split == "train":
            self.entries = all_entries[:n_train]
        else:
            self.entries = all_entries[n_train:]

        print(f"LJSpeechDataset [{split}]: {len(self.entries)} utterances")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        mel_path = Path(self.data_cfg["preprocessed_path"]) / entry["mel_path"]
        mel = np.load(mel_path)
        mel, mel_len = pad_or_trim_mel(mel, self.max_frames)

        return {
            "mel": torch.FloatTensor(mel),
            "mel_len": torch.LongTensor([mel_len]),
            "text": entry["text"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate for DataLoader.
    Stacks tensors and keeps text/path lists as-is.
    """
    mels = torch.stack([item["mel"] for item in batch])
    mel_lens = torch.cat([item["mel_len"] for item in batch])
    texts = [item["text"] for item in batch]

    result = {"mel": mels, "mel_len": mel_lens, "text": texts}

    if "speaker_id" in batch[0]:
        result["speaker_id"] = [item["speaker_id"] for item in batch]
        result["speaker_idx"] = torch.cat([item["speaker_idx"] for item in batch])
        result["ref_wav_paths"] = [item["ref_wav_paths"] for item in batch]

    return result


def get_dataloader(
    config: dict,
    dataset_name: str = "vctk",
    split: str = "train",
    max_frames: int = 600,
) -> DataLoader:
    """Convenience function to build a DataLoader from config."""
    train_cfg = config["training"]

    if dataset_name == "vctk":
        dataset = VCTKDataset(config, split=split, max_frames=max_frames)
    elif dataset_name == "ljspeech":
        dataset = LJSpeechDataset(config, split=split, max_frames=max_frames)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(split == "train"),
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
