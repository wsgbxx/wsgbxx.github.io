from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from shared.utils import get_data_root, pick_amp_dtype, set_seed
from vae.losses import vae_loss
from vae.model import ConvVAE


@dataclass(frozen=True)
class TrainCfg:
    image_size: int = 64
    batch_size: int = 256
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    beta: float = 1.0


def build_dataloader(*, data_root: str, batch_size: int, image_size: int, num_workers: int) -> DataLoader:
    """CIFAR-10 loader.

    注意事项：
    - 这里不 download（服务器通常无网），请你提前把 CIFAR-10 放到 data_root。
    - data_root 的传入优先级：--data-root > 环境变量 DATA_ROOT。

    目录结构需兼容 torchvision.datasets.CIFAR10。
    """

    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
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
    p.add_argument("--out-dir", type=str, default="./out_vae", help="checkpoint 输出目录。")

    p.add_argument("--image-size", type=int, default=TrainCfg.image_size)
    p.add_argument("--batch-size", type=int, default=TrainCfg.batch_size)
    p.add_argument("--epochs", type=int, default=TrainCfg.epochs)
    p.add_argument("--lr", type=float, default=TrainCfg.lr)
    p.add_argument("--weight-decay", type=float, default=TrainCfg.weight_decay)
    p.add_argument("--num-workers", type=int, default=TrainCfg.num_workers)
    p.add_argument("--seed", type=int, default=TrainCfg.seed)
    p.add_argument("--beta", type=float, default=TrainCfg.beta)

    args = p.parse_args()

    data_root = get_data_root(args.data_root, env_name="DATA_ROOT")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model = ConvVAE(image_size=args.image_size, out_activation="sigmoid").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = torch.cuda.is_available()
    amp_dtype = pick_amp_dtype()

    dl = build_dataloader(
        data_root=data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, _y in dl:
            x = x.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                losses = vae_loss(
                    recon=out.recon,
                    x=x,
                    mu=out.mu,
                    logvar=out.logvar,
                    beta=args.beta,
                    recon_type="mse",
                    reduction="mean",
                )

            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running += float(losses["loss"].detach().cpu())
            global_step += 1

        avg = running / max(1, len(dl))
        print(f"epoch={epoch} loss={avg:.4f}")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))


if __name__ == "__main__":
    main()
