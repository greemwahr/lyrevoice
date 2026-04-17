# AMP & Mixed Precision Learning Roadmap

## Level 1: Understand the basics
- [PyTorch Blog: What Every User Should Know About Mixed Precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) — best starting point, explains why and how
- [PyTorch AMP Recipe Tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — hands-on code walkthrough with `autocast` and `GradScaler`

## Level 2: See it in action
- [PyTorch AMP Examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) — official examples for different training patterns (single GPU, GAN, gradient accumulation)
- [DigitalOcean AMP Tutorial](https://www.digitalocean.com/community/tutorials/automatic-mixed-precision-using-pytorch) — step-by-step practical guide

## Level 3: Understand the hardware
- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — the definitive guide from NVIDIA on how Tensor Cores work with FP16
- [NVIDIA Video Series: Mixed Precision with Tensor Cores](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/) — video tutorials explaining Tensor Core math

## Level 4: Go deeper
- [NVIDIA A100 Architecture Whitepaper (PDF)](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) — how A100/H100 Tensor Cores are built
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html) — optimizing batch sizes, memory, and compute for Tensor Cores

## Key takeaway

FP32 uses 32 bits per number. FP16 uses 16. Tensor Cores on A100/H100 are physically designed to multiply FP16 matrices faster. AMP automatically picks which ops can safely use FP16 and which need FP32 — you get the speed without losing accuracy.
