"""Shared minimal utilities for teaching code.

安全 / 隐私注意：
- 不要在代码里硬编码任何本地数据集路径（博客/分享时很容易泄露隐私目录）。
- 所有脚本都应通过 CLI 参数或环境变量传入数据集根目录。

这个文件只放“跨脚本复用、且不引入额外依赖”的小工具。
"""

from __future__ import annotations

import os
import random
from typing import Optional


def get_data_root(cli_value: Optional[str], *, env_name: str = "DATA_ROOT") -> str:
    """解析数据集根目录。

    优先级：
    1) CLI 显式传入
    2) 环境变量（默认 DATA_ROOT）

    注意：只做字符串校验，不在这里做 I/O（比如检查目录是否存在），
    以免在教学脚本里引入不必要的副作用。
    """

    if cli_value is not None and cli_value.strip():
        return cli_value

    v = os.environ.get(env_name)
    if v is not None and v.strip():
        return v

    raise RuntimeError(f"Please provide --data-root or set {env_name}.")


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """尽力保证可复现。

    deterministic=True 会关闭 cuDNN benchmark 并开启确定性算子，通常会更慢。
    教学/博客 Demo 默认 deterministic=False，让代码更快更贴近日常训练。
    """

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def pick_amp_dtype() -> "torch.dtype":
    """选择 AMP dtype。

    注意事项：
    - 这里“偏向 bf16”，因为在 Hopper/Ampere+ 上通常更稳。
    - 但并不保证所有算子都对 bf16 友好；如果你遇到 NaN/inf，先试试禁用 AMP。
    """

    import torch

    if not torch.cuda.is_available():
        return torch.float32

    return torch.bfloat16
