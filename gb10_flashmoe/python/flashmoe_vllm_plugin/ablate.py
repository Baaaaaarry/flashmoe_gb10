from __future__ import annotations

import argparse
from pathlib import Path

from .config import FlashMoEConfig
from .runtime import FlashMoERuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Print FlashMoE ablation matrix and vLLM args.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = FlashMoEConfig.from_file(args.config)
    runtime = FlashMoERuntime(config)
    print(runtime.describe())
    print("\nvLLM serve args:")
    print(" ".join(runtime.to_vllm_serve_args()))


if __name__ == "__main__":
    main()
