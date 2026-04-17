# Continued Training (Epochs 101-150) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Continue training from epoch 100 checkpoint with LR decay and smaller batch size to fine-tune the generator and improve GAN equilibrium.

**Architecture:** Resume from the epoch 100 checkpoint with halved learning rates (LR decay) and reduced batch size (64) to give the generator noisier gradients and finer optimization steps. No model or code changes — config-only adjustments.

**Tech Stack:** PyTorch, Google Colab (A100/H100)

---

## Context

After 100 epochs of training:
- G loss converged from 5.5 → ~3.2
- D loss remained very low (~0.003-0.02) — discriminator dominant
- All 3 lambda phases completed successfully
- Checkpoints saved at epochs 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

The goal of continued training is to fine-tune with smaller steps (LR decay) and noisier gradients (smaller batch) to push G loss further down and potentially improve GAN equilibrium.

---

### Task 1: Update Colab config for continued training

**Files:**
- Modify: `configs/config_colab.yaml`

**Step 1: Update training overrides**

Change the training section to:

```yaml
training:
  batch_size: 64
  num_epochs: 150
  learning_rate_generator: 0.00005
  learning_rate_discriminator: 0.00005
  num_workers: 4
```

| Setting | Old | New | Reason |
|---------|-----|-----|--------|
| `batch_size` | 128 | 64 | Noisier gradients to handicap D |
| `num_epochs` | 100 | 150 | 50 more epochs |
| `learning_rate_generator` | 0.0001 | 0.00005 | LR decay — finer optimization |
| `learning_rate_discriminator` | 0.0001 | 0.00005 | Keep G/D balanced |
| `num_workers` | 4 | 4 | No change — keeps data pipeline smooth |

**Step 2: Commit**

```bash
git add configs/config_colab.yaml
git commit -m "feat: update Colab config for continued training (epochs 101-150)"
```

---

### Task 2: Pull changes on Colab and resume training

**Step 1: Pull latest code on Colab**

```python
!cd /content/lyrevoice && git pull
```

**Step 2: Resume from epoch 100 checkpoint**

```python
!python scripts/train.py \
    --config configs/config.yaml \
    --config-override configs/config_colab.yaml \
    --resume /content/drive/MyDrive/lyrevoice/checkpoints/lyrevoice_epoch_0100.pt
```

**Step 3: Verify training output**

Expected output should show:
- `Resumed from checkpoint: ... (epoch 100)`
- `Batch: 64`
- `Starting training for 150 epochs...`
- Training continues from epoch 101
- `λ1=5.0, λ2=1.00` (stays in phase 3)
- Step time ~0.6-1.0s (smaller batch)
- ~335 steps/epoch

**Step 4: Monitor on WandB**

Watch for:
- G loss continuing to decrease below 3.2
- D loss potentially rising (smaller batch = noisier signal for D)
- Stable oscillation without sudden spikes

---

### Task 3: Push checkpoint after training completes

**Step 1: Verify final checkpoint saved**

```python
!ls -lh /content/drive/MyDrive/lyrevoice/checkpoints/
```

Expected: `lyrevoice_epoch_0150.pt` plus all previous checkpoints.

---

## Expected Performance

- ~335 steps/epoch at ~0.6-1.0s/step = ~3-5 min/epoch
- 50 epochs = ~2.5-4 hours
- Compute units: ~19 units/hour × 4 hours = ~76 units

## Rollback

If continued training degrades quality (G loss increases significantly), the epoch 100 checkpoint is still available on Drive. Simply use that for evaluation/inference instead.
