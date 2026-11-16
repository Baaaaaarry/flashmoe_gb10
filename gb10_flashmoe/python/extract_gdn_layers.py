#!/usr/bin/env python3
"""Extract Qwen3.5 GDN layer metadata without loading tensor payloads."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


GDN_REQUIRED = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
    "out_proj.weight",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def load_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("missing 8-byte safetensors header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            header = handle.read(header_length)
            if len(header) != header_length:
                raise ValueError("truncated safetensors header")
        parsed = json.loads(header)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read safetensors header {path}: {exc}") from exc
    return parsed


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def parse_layer_tensor(name: str) -> tuple[int, str] | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.linear_attn\.(.+)$", name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = load_json(index_path)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise SystemExit(f"weight_map is invalid in {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}

    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"neither model.safetensors.index.json nor model.safetensors exists under {model_dir}")
    header = load_safetensors_header(single)
    return {key: single.name for key in header if key != "__metadata__"}


def tensor_metadata(model_dir: Path, weight_map: dict[str, str], names: list[str]) -> dict[str, dict[str, Any]]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_shard[weight_map[name]].append(name)

    output: dict[str, dict[str, Any]] = {}
    for shard, shard_names in sorted(by_shard.items()):
        header = load_safetensors_header(model_dir / shard)
        for name in shard_names:
            item = header.get(name)
            if not isinstance(item, dict):
                output[name] = {"dtype": "missing", "shape": [], "shard": shard}
                continue
            output[name] = {
                "dtype": str(item.get("dtype", "unknown")),
                "shape": item.get("shape", []),
                "shard": shard,
                "data_offsets": item.get("data_offsets", []),
            }
    return output


def derive_dimensions(config: dict[str, Any]) -> dict[str, Any]:
    key_heads = int(config.get("linear_num_key_heads", 0))
    value_heads = int(config.get("linear_num_value_heads", 0))
    key_head_dim = int(config.get("linear_key_head_dim", 0))
    value_head_dim = int(config.get("linear_value_head_dim", 0))
    hidden_size = int(config.get("hidden_size", 0))
    conv_kernel = int(config.get("linear_conv_kernel_dim", 0))
    key_dim = key_heads * key_head_dim
    value_dim = value_heads * value_head_dim
    qkv_dim = key_dim * 2 + value_dim
    return {
        "hidden_size": hidden_size,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "key_head_dim": key_head_dim,
        "value_head_dim": value_head_dim,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "qkv_dim": qkv_dim,
        "conv_kernel": conv_kernel,
        "qkv_split": {
            "q": [0, key_dim],
            "k": [key_dim, key_dim * 2],
            "v": [key_dim * 2, qkv_dim],
        },
        "conv_state_shape": ["batch", qkv_dim, max(0, conv_kernel - 1)],
        "recurrent_state_shape": ["batch", value_heads, key_head_dim, value_head_dim],
    }


def operator_decomposition() -> dict[str, list[str]]:
    common = [
        "RMSNorm(hidden)",
        "Linear/GEMM: in_proj_qkv(hidden)",
        "Linear/GEMM: in_proj_z(hidden)",
        "Linear/GEMM: in_proj_a(hidden)",
        "Linear/GEMM: in_proj_b(hidden)",
        "Depthwise causal Conv1D(QKV)",
        "SiLU(QKV)",
        "Split Q/K/V using configured dimensions",
        "beta = sigmoid(B)",
        "g = -exp(A_log) * softplus(A + dt_bias)",
        "L2Norm(Q), L2Norm(K), repeat K-heads to V-heads",
    ]
    tail = [
        "Gated RMSNorm(core_output, Z)",
        "Linear/GEMM: out_proj",
        "Residual add",
    ]
    return {
        "prefill": common + ["Chunk Gated Delta Rule"] + tail,
        "decode": common + ["Recurrent Gated Delta Rule; update recurrent state"] + tail,
    }


def build_report(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    config = load_json(config_path)
    text_cfg = text_config(config)
    layer_types = text_cfg.get("layer_types", [])
    if not isinstance(layer_types, list) or not layer_types:
        raise SystemExit(f"layer_types is missing from {config_path}")

    weight_map = load_weight_map(model_dir)
    layer_names: dict[int, list[str]] = defaultdict(list)
    suffix_by_name: dict[str, str] = {}
    for name in weight_map:
        parsed = parse_layer_tensor(name)
        if parsed is None:
            continue
        layer, suffix = parsed
        layer_names[layer].append(name)
        suffix_by_name[name] = suffix

    gdn_layers = [idx for idx, layer_type in enumerate(layer_types) if layer_type == "linear_attention"]
    all_gdn_names = sorted(name for layer in gdn_layers for name in layer_names.get(layer, []))
    metadata = tensor_metadata(model_dir, weight_map, all_gdn_names)
    dimensions = derive_dimensions(text_cfg)

    layers: list[dict[str, Any]] = []
    for layer in gdn_layers:
        tensors = []
        suffixes: set[str] = set()
        for name in sorted(layer_names.get(layer, []), key=lambda item: suffix_by_name[item]):
            suffix = suffix_by_name[name]
            suffixes.add(suffix)
            tensors.append({"name": name, "component": suffix, **metadata[name]})
        missing = [required for required in GDN_REQUIRED if required not in suffixes]
        layers.append(
            {
                "layer": layer,
                "layer_type": layer_types[layer],
                "complete": not missing,
                "missing_required_tensors": missing,
                "tensors": tensors,
            }
        )

    return {
        "model_dir": str(model_dir),
        "model_type": text_cfg.get("model_type", config.get("model_type", "unknown")),
        "model_dtype": text_cfg.get("dtype", config.get("torch_dtype", "unknown")),
        "num_layers": len(layer_types),
        "gdn_layer_count": len(gdn_layers),
        "full_attention_layer_count": sum(item == "full_attention" for item in layer_types),
        "gdn_layer_indices": gdn_layers,
        "dimensions": dimensions,
        "operators": operator_decomposition(),
        "layers": layers,
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("layer", "complete", "component", "dtype", "shape", "shard", "tensor_name"),
        )
        writer.writeheader()
        for layer in report["layers"]:
            for tensor in layer["tensors"]:
                writer.writerow(
                    {
                        "layer": layer["layer"],
                        "complete": layer["complete"],
                        "component": tensor["component"],
                        "dtype": tensor["dtype"],
                        "shape": "x".join(str(dim) for dim in tensor["shape"]),
                        "shard": tensor["shard"],
                        "tensor_name": tensor["name"],
                    }
                )


def print_summary(report: dict[str, Any], show_tensors: bool) -> None:
    dims = report["dimensions"]
    print(f"model_type: {report['model_type']}")
    print(f"model_dtype: {report['model_dtype']}")
    print(
        f"layers: total={report['num_layers']} gdn={report['gdn_layer_count']} "
        f"full_attention={report['full_attention_layer_count']}"
    )
    print(f"gdn_layer_indices: {report['gdn_layer_indices']}")
    print(
        f"qkv: Q={dims['key_dim']} K={dims['key_dim']} V={dims['value_dim']} "
        f"total={dims['qkv_dim']} split={dims['qkv_split']}"
    )
    print(f"conv_state_shape: {dims['conv_state_shape']}")
    print(f"recurrent_state_shape: {dims['recurrent_state_shape']} dtype=FP32")

    print("\nlayer completeness:")
    for layer in report["layers"]:
        state = "complete" if layer["complete"] else "missing=" + ",".join(layer["missing_required_tensors"])
        print(f"  layer {layer['layer']:>2}: {state}; tensors={len(layer['tensors'])}")
        if show_tensors:
            for tensor in layer["tensors"]:
                shape = "x".join(str(dim) for dim in tensor["shape"])
                print(f"    {tensor['component']:<36} {tensor['dtype']:<9} {shape:<20} {tensor['shard']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract per-layer GDN metadata from a Qwen3.5 checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Model directory containing config.json and safetensors.")
    parser.add_argument("--json-out", type=Path, help="Output complete JSON report.")
    parser.add_argument("--csv-out", type=Path, help="Output one tensor per CSV row.")
    parser.add_argument("--show-tensors", action="store_true", help="Print every GDN tensor in the terminal.")
    args = parser.parse_args()

    report = build_report(args.model.resolve())
    print_summary(report, args.show_tensors)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    if args.csv_out:
        write_csv(report, args.csv_out)
        print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
