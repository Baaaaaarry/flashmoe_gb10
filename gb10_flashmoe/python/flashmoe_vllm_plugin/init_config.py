from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

try:
    from .config import FlashMoEConfig
except ImportError:
    import sys
    from pathlib import Path as _Path

    current_dir = _Path(__file__).resolve().parent
    package_root = current_dir.parent
    sys.path.append(str(package_root))
    sys.path.append(str(current_dir))
    try:
        from flashmoe_vllm_plugin.config import FlashMoEConfig
    except ImportError:
        from config import FlashMoEConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local FlashMoE config JSON from paths.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--expert-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expert-cache-gb", type=float, default=96.0)
    parser.add_argument("--pread-workers", type=int, default=8)
    parser.add_argument("--pread-chunks", type=int, default=4)
    parser.add_argument("--prefetch-window", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--expert-format", default="dense", choices=["dense", "q3like"])
    args = parser.parse_args()

    expert_path = args.expert_path.resolve()
    manifest_path = expert_path / "expert_manifest.json"

    raw_kwargs = {
        "model_path": str(args.model_path.resolve()),
        "expert_path": str(expert_path),
        "expert_manifest_path": str(manifest_path),
        "expert_format": args.expert_format,
        "expert_cache_gb": args.expert_cache_gb,
        "pread_workers": args.pread_workers,
        "pread_chunks": args.pread_chunks,
        "prefetch_window": args.prefetch_window,
        "max_model_len": args.max_model_len,
        "top_k": args.top_k,
        "dense_path": str(args.model_path.resolve()),
        "predictor_path": str((expert_path / "predictor.json").resolve()),
        "cache_policy_path": str((expert_path / "cache_policy.json").resolve()),
        "routing_trace_path": str((expert_path / "routing_trace.txt").resolve()),
        "base_model_class": "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM",
        "vllm_quantization": "model",
    }
    supported = {item.name for item in fields(FlashMoEConfig)}
    cfg = FlashMoEConfig(**{key: value for key, value in raw_kwargs.items() if key in supported})
    cfg.to_file(args.output)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
