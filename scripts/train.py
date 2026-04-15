"""
scripts/train.py

Entry point for LyreVoice GAN training.

Usage:
  # Full training from scratch
  python scripts/train.py

  # Resume from a checkpoint
  python scripts/train.py --resume checkpoints/training/lyrevoice_epoch_0030.pt

  # Use a custom config
  python scripts/train.py --config configs/config.yaml
"""

import argparse
import yaml
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train LyreVoice GAN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("LyreVoice GAN Training")
    print("=" * 60)
    print(f"Config:    {args.config}")
    print(f"Resume:    {args.resume or 'No (training from scratch)'}")
    print(f"Epochs:    {config['training']['num_epochs']}")
    print(f"Batch:     {config['training']['batch_size']}")
    print(f"Dataset:   VCTK ({config['data']['vctk_path']})")
    print("=" * 60)

    trainer = Trainer(config, resume_checkpoint=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
