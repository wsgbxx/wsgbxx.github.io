from __future__ import annotations

import argparse
import os

import torch

from ddpm import DiffusionConfig, DiffusionUNet, GaussianDiffusion, UNetConfig
from shared.utils import pick_amp_dtype, set_seed


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="./out_ddpm", help="输出目录。")
    p.add_argument("--ckpt", type=str, default=None, help="可选：checkpoint 路径（默认用 out-dir/last.pt）。")

    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=50, help="DDIM 采样步数（越少越快，越多越精细）。")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta（0=确定性；>0 会引入随机性）。")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    unet = DiffusionUNet(UNetConfig(image_size=args.image_size)).to(device)
    diffusion = GaussianDiffusion(DiffusionConfig())

    ckpt_path = args.ckpt or os.path.join(args.out_dir, "last.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        unet.load_state_dict(state, strict=True)
    else:
        # 注意：不强制要求 ckpt 存在，方便你先验证采样流程是否能跑通。
        print(f"warn: checkpoint not found, sampling from random init: {ckpt_path}")

    unet.eval()

    use_amp = torch.cuda.is_available()
    amp_dtype = pick_amp_dtype()

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        x = diffusion.ddim_sample_loop(
            unet,
            (args.batch, 3, args.image_size, args.image_size),
            device=device,
            steps=args.steps,
            eta=args.eta,
        )

    # Convert [-1,1] -> [0,1] for saving.
    x = (x.clamp(-1, 1) + 1) / 2

    try:
        from torchvision.utils import save_image

        save_image(x, os.path.join(args.out_dir, "samples.png"), nrow=int(args.batch**0.5))
        print("saved:", os.path.join(args.out_dir, "samples.png"))
    except Exception:
        # torchvision might not be installed; in that case just save a tensor.
        out_path = os.path.join(args.out_dir, "samples.pt")
        torch.save(x.cpu(), out_path)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
