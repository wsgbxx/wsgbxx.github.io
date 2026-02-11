# UNet (for blog)

This folder contains a small, reusable UNet for supervised image-to-image tasks.

## Files

- `blocks.py`: `ResBlock`, `Downsample`, `Upsample`
- `unet_sr.py`: `SimpleUNetSR` (a small SR-style UNet)

## Why separate UNet from DDPM UNet?

- The **supervised UNet** is used for tasks like denoising / SR where you have paired targets.
- The **diffusion UNet** (in `ddpm/unet.py`) additionally takes a timestep embedding and is trained to predict noise.
