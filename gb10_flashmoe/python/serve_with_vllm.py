from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from flashmoe_vllm_plugin.config import FlashMoEConfig
from flashmoe_vllm_plugin.runtime import FlashMoERuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch vLLM with FlashMoE plugin config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()

    config = FlashMoEConfig.from_file(args.config)
    runtime = FlashMoERuntime(config)

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *runtime.to_vllm_serve_args(), *passthrough]
    print("Command:")
    print(" ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
