from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_FLASHMOE_EXT = None


def _require_torch():
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for FlashMoE CUDA ops") from exc
    return torch, load


def load_flashmoe_cuda_extension(verbose: bool = False):
    global _FLASHMOE_EXT
    if _FLASHMOE_EXT is not None:
        return _FLASHMOE_EXT

    torch, load = _require_torch()
    root = Path(__file__).resolve().parents[1]
    sources = [
        str(root / "csrc" / "flashmoe_ops.cpp"),
        str(root / "csrc" / "flashmoe_ops_cuda.cu"),
    ]
    build_directory = os.environ.get(
        "FLASHMOE_EXT_BUILD_DIR",
        str(root / "build_ext"),
    )
    Path(build_directory).mkdir(parents=True, exist_ok=True)
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]
    _FLASHMOE_EXT = load(
        name="flashmoe_cuda_ops",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-O3"],
        build_directory=build_directory,
        verbose=verbose,
    )
    return _FLASHMOE_EXT


def flashmoe_moe_forward(
    hidden_states,
    gate_proj_weights,
    up_proj_weights,
    down_proj_weights,
    route_ids,
    route_weights,
):
    ext = load_flashmoe_cuda_extension()
    return ext.moe_forward(
        hidden_states,
        gate_proj_weights,
        up_proj_weights,
        down_proj_weights,
        route_ids,
        route_weights,
    )
