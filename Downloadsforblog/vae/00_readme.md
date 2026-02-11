# VAE (for blog)

## Files

- `model.py`: `ConvVAE` (encode/reparameterize/decode)
- `losses.py`: `vae_loss` (reconstruction + KL)

## What this VAE does

- Input: image tensor `x` in `[0, 1]` (when using `out_activation="sigmoid"`).
- Encoder: strided convs to a feature map, then MLP to `mu` and `logvar`.
- Latent sampling: `z = mu + eps * exp(0.5*logvar)`.
- Decoder: MLP + transposed convs to reconstruct the image.

## Typical training recipe

- Reconstruction loss: MSE (simple and stable).
- KL loss: standard Gaussian prior.
- Total: `loss = recon + beta * kl`.

Use `scripts/train_vae_cifar10.py` for a minimal runnable example.
