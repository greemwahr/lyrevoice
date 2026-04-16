# LyreVoice — TODO / Known Warnings

## Warnings to Address

- [ ] `webrtcvad` imports deprecated `pkg_resources` (slated for removal after 2025-11-30) — appears on every import including dataloader workers
- [ ] `resemblyzer` uses `torch.load` with `weights_only=False` — future PyTorch will default to `True`; may break
- [ ] `generator.py:146` — same `torch.load` `weights_only=False` warning when loading Tacotron2 checkpoint
- [ ] `vocoder.py:210` — same `torch.load` `weights_only=False` warning when loading HiFi-GAN checkpoint
- [ ] NVIDIA hub ConvNets code warns `pytorch_quantization` module not found — harmless, we don't use quantization
- [ ] `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm` — comes from HiFi-GAN ResBlock
