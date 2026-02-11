from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from ddpm import DiffusionConfig, DiffusionUNet, GaussianDiffusion, UNetConfig
from shared.utils import get_data_root, pick_amp_dtype, set_seed


def build_dataloader(*, data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    ds = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tfm)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None, help="CIFAR-10 根目录（可选；否则使用环境变量 DATA_ROOT）。")
    p.add_argument("--out-dir", type=str, default="./out_ddpm", help="输出目录。")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # 注意：默认读 DATA_ROOT 是为了博客演示命令更短。
    data_root = get_data_root(args.data_root, env_name="DATA_ROOT")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    unet = DiffusionUNet(UNetConfig(image_size=32)).to(device)
    diffusion = GaussianDiffusion(DiffusionConfig(timesteps=args.timesteps))

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = torch.cuda.is_available()
    amp_dtype = pick_amp_dtype()

    dl = build_dataloader(data_root=data_root, batch_size=args.batch_size, num_workers=args.num_workers)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        unet.train()
        running = 0.0

        for x, _y in dl:
            x = x.to(device, non_blocking=True)
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                losses = diffusion.training_losses(unet, x, t)

            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            opt.step()

            running += float(losses["loss"].detach().cpu())
            global_step += 1

        avg = running / max(1, len(dl))
        print(f"epoch={epoch} loss={avg:.4f}")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": unet.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))


if __name__ == "__main__":
    main()
