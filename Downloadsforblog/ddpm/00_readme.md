# DDPM (for blog)

## Files

- `unet.py`: diffusion UNet that predicts noise `eps`.
- `schedules.py`: beta schedule helpers (cosine by default).
- `diffusion.py`: forward noising `q(x_t|x_0)`, training loss, DDPM sampling, DDIM sampling.

## Core ideas (very short)

- Forward process: add Gaussian noise step-by-step.
- Model: learn to predict the noise that was added.
- Sampling: start from pure noise and iteratively denoise.

Run training/sampling with:
- `scripts/train_ddpm_cifar10.py`
- `scripts/sample_ddpm.py`
