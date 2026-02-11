from __future__ import annotations

import sys


def main() -> None:
    import os

    # 轻量 runner：不强依赖 pytest。
    # 注意：这里做 sys.path 注入是为了“仓库拷贝到任意机器上都能直接跑”。
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from test_shapes import (
        test_ddpm_forward_and_sample_shapes,
        test_unet_sr_shapes,
        test_vae_forward_shapes,
    )

    test_vae_forward_shapes()
    test_unet_sr_shapes()
    test_ddpm_forward_and_sample_shapes()

    print("smoke: ok")


if __name__ == "__main__":
    if "./" not in sys.path:
        sys.path.insert(0, "./")
    if "./tests" not in sys.path:
        sys.path.insert(0, "./tests")
    main()
