#!/usr/bin/env python3
"""Compare the operators and dtypes used by official_hf and streamed paths.

This intentionally does not load model weights. It combines the checkpoint
configuration with the exact Python/C++ sources available in the runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Evidence:
    file: str
    line: int
    text: str


@dataclass
class Finding:
    value: str
    status: str
    evidence: list[Evidence]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def find_evidence(path: Path, patterns: list[str], *, regex: bool = False) -> list[Evidence]:
    text = read_text(path)
    if not text:
        return []
    found: list[Evidence] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        matched = any(re.search(pattern, line) if regex else pattern in line for pattern in patterns)
        if matched:
            found.append(Evidence(str(path), line_no, line.strip()))
    return found


def first_evidence(path: Path, patterns: list[str], *, regex: bool = False) -> list[Evidence]:
    found = find_evidence(path, patterns, regex=regex)
    return found[:1]


def locate_transformers_source() -> Path | None:
    try:
        spec = importlib.util.find_spec("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    except Exception:
        # Importing a package parent may trigger an unavailable optional backend.
        return None
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).resolve()
    return path if path.is_file() else None


def model_config_path(model: Path | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    if model is None:
        return None
    candidate = model.resolve() / "config.json"
    return candidate if candidate.is_file() else None


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read model config {path}: {exc}") from exc


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def config_evidence(path: Path | None, keys: list[str]) -> list[Evidence]:
    if path is None:
        return []
    return find_evidence(path, [f'"{key}"' for key in keys])


def finding(value: str, status: str, *groups: list[Evidence]) -> Finding:
    evidence: list[Evidence] = []
    for group in groups:
        evidence.extend(group)
    return Finding(value=value, status=status, evidence=evidence)


def inspect_official(config: dict[str, Any], config_path: Path | None, hf_source: Path | None) -> dict[str, Finding]:
    text_cfg = text_config(config)
    layer_types = text_cfg.get("layer_types", [])
    gdn_layers = sum(1 for item in layer_types if item == "linear_attention")
    full_layers = sum(1 for item in layer_types if item == "full_attention")
    dtype = str(text_cfg.get("dtype", config.get("torch_dtype", "unknown")))
    quant = config.get("quantization_config", {})
    if not isinstance(quant, dict):
        quant = {}
    quant_method = str(quant.get("quant_method", "none"))
    activation_scheme = str(quant.get("activation_scheme", "none"))
    block = quant.get("weight_block_size", [])
    excluded = quant.get("modules_to_not_convert", [])
    if not isinstance(excluded, list):
        excluded = []
    spec_fields = {
        "num_layers": text_cfg.get("num_hidden_layers", "unknown"),
        "hidden_size": text_cfg.get("hidden_size", "unknown"),
        "experts": text_cfg.get("num_experts", "unknown"),
        "top_k": text_cfg.get("num_experts_per_tok", "unknown"),
    }
    spec_value = ", ".join(f"{key}={value}" for key, value in spec_fields.items())

    source_available = hf_source is not None
    source = hf_source or Path("<transformers-source-unavailable>")
    class_ev = first_evidence(source, ["class Qwen3_5MoeGatedDeltaNet"])
    qkv_ev = first_evidence(source, ["mixed_qkv = self.in_proj_qkv(hidden_states)"])
    conv_prefill_ev = first_evidence(source, ["mixed_qkv = causal_conv1d_fn("])
    conv_decode_ev = first_evidence(source, ["mixed_qkv = causal_conv1d_update("])
    chunk_ev = first_evidence(source, ["torch_chunk_gated_delta_rule("])
    recurrent_ev = first_evidence(source, ["torch_recurrent_gated_delta_rule("])
    fp32_ev = first_evidence(source, ["contiguous().to(torch.float32)"])
    norm_ev = first_evidence(source, ["core_attn_out = self.norm(core_attn_out, z)"])
    output_ev = first_evidence(source, ["output = self.out_proj(core_attn_out)"])

    gdn_value = f"yes ({gdn_layers} GDN/linear-attention, {full_layers} full-attention)"
    if not layer_types:
        gdn_value = "unknown (layer_types missing from config)"

    return {
        "backend": finding("Transformers AutoModelForCausalLM", "confirmed"),
        "model_spec": finding(
            spec_value,
            "confirmed" if text_cfg else "unknown",
            config_evidence(config_path, ["num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok"]),
        ),
        "gdn": finding(
            gdn_value,
            "confirmed" if layer_types and class_ev else "partial",
            config_evidence(config_path, ["layer_types", "full_attention_interval"]),
            class_ev,
        ),
        "qkv_projection": finding("nn.Linear -> CUDA GEMM/quantized GEMM", "confirmed" if qkv_ev else "unknown", qkv_ev),
        "conv1d": finding(
            "causal Conv1D: prefill fn + decode state update",
            "confirmed" if conv_prefill_ev and conv_decode_ev else "unknown",
            conv_prefill_ev,
            conv_decode_ev,
        ),
        "silu": finding(
            "configured SiLU; passed into causal Conv1D",
            "confirmed" if text_cfg.get("hidden_act") == "silu" and conv_prefill_ev else "partial",
            config_evidence(config_path, ["hidden_act"]),
            conv_prefill_ev,
        ),
        "prefill_core": finding("chunk_gated_delta_rule", "confirmed" if chunk_ev else "unknown", chunk_ev),
        "decode_core": finding("recurrent_gated_delta_rule", "confirmed" if recurrent_ev else "unknown", recurrent_ev),
        "gated_rmsnorm": finding("Gated RMSNorm followed by out_proj", "confirmed" if norm_ev else "unknown", norm_ev, output_ev),
        "external_dtype": finding(dtype, "confirmed" if dtype != "unknown" else "unknown", config_evidence(config_path, ["dtype"])),
        "core_dtype": finding(
            "FP32 GDN fallback/core state; output cast back to input dtype",
            "confirmed" if fp32_ev else "partial",
            config_evidence(config_path, ["mamba_ssm_dtype"]),
            fp32_ev,
        ),
        "weight_quantization": finding(
            f"{quant_method}, block={block}; exclusions={len(excluded)}",
            "confirmed" if quant_method != "none" else "unknown",
            config_evidence(config_path, ["quant_method", "weight_block_size", "modules_to_not_convert"]),
        ),
        "activation_quantization": finding(
            f"{activation_scheme} for converted FP8 linear ops; model boundary remains {dtype}",
            "confirmed" if activation_scheme != "none" else "unknown",
            config_evidence(config_path, ["activation_scheme"]),
        ),
        "source": finding(str(source), "confirmed" if source_available else "missing"),
    }


def parse_streamed_spec(model_spec: Path) -> dict[str, str]:
    text = read_text(model_spec)
    match = re.search(r"constexpr ModelSpec kQwen35Spec\s*\{(.*?)\n\};", text, re.DOTALL)
    if not match:
        return {}
    result: dict[str, str] = {}
    for key in ("short_name", "num_layers", "routed_experts_per_layer", "top_k", "hidden_size", "expert_intermediate_size"):
        field = re.search(rf"\.{key}\s*=\s*([^,]+)", match.group(1))
        if field:
            result[key] = field.group(1).strip().strip('"')
    return result


def inspect_streamed(repo: Path) -> dict[str, Finding]:
    src = repo / "src"
    engine = src / "streamed_engine.cpp"
    dense = src / "dense_operator_chain.cpp"
    artifact = src / "dense_artifact.cpp"
    kv = src / "kv_cache.cpp"
    expert = src / "expert_cuda_backend.cu"
    model_spec = src / "model_spec.cpp"

    gdn_markers = [
        "Qwen3_5MoeGatedDeltaNet",
        "chunk_gated_delta_rule",
        "recurrent_gated_delta_rule",
        "causal_conv1d",
    ]
    all_runtime_text = "\n".join(read_text(path) for path in (engine, dense, artifact, expert))
    missing_gdn = not any(marker in all_runtime_text for marker in gdn_markers)
    spec = parse_streamed_spec(model_spec)
    spec_value = ", ".join(f"{key}={value}" for key, value in spec.items()) or "unparsed"

    return {
        "backend": finding("custom C++ StreamedEngine", "confirmed", first_evidence(engine, ["StreamedEngine::load"])),
        "model_spec": finding(spec_value, "confirmed" if spec else "unknown", first_evidence(model_spec, ["constexpr ModelSpec kQwen35Spec"])),
        "gdn": finding(
            "no: GDN class/core calls absent; replaced by custom attention",
            "confirmed" if missing_gdn else "review",
            first_evidence(dense, ["run_layer_attention_direct"]),
        ),
        "qkv_projection": finding(
            "BF16-cached weight matvec with FP32 accumulation; in_proj_qkv split into thirds",
            "confirmed",
            first_evidence(artifact, ["const auto chunk = std::max<std::size_t>(1, rows / 3)"]),
            first_evidence(artifact, ["acc += bf16_to_float(qkv_bf16"]),
        ),
        "conv1d": finding("not executed in attention path", "confirmed" if "conv1d" not in read_text(dense) else "review"),
        "silu": finding(
            "not in attention; executed only in Expert SwiGLU",
            "confirmed",
            first_evidence(expert, ["silu_mul_kernel<<<"]),
        ),
        "prefill_core": finding(
            "token-serial loop; 32-token sliding-window softmax attention",
            "confirmed",
            first_evidence(engine, ["for (std::size_t token_idx = 0; token_idx < token_ids.size();"]),
            first_evidence(dense, ["const std::size_t window = std::min<std::size_t>(position + 1U, 32U)"]),
        ),
        "decode_core": finding(
            "same custom attention path per token; no recurrent GDN state",
            "confirmed",
            first_evidence(engine, ["for (std::size_t token_idx = 0; token_idx < max_tokens;"]),
            first_evidence(engine, ["run_layer_attention_direct("]),
        ),
        "gated_rmsnorm": finding(
            "not implemented; ordinary RMSNorm before attention/MoE",
            "confirmed",
            first_evidence(dense, ["apply_rms_norm(normed, input_norm)"]),
        ),
        "external_dtype": finding(
            "FP32 hidden/activation vectors",
            "confirmed",
            first_evidence(dense, ["state.hidden.assign(spec.hidden_size, 0.0f)"]),
        ),
        "core_dtype": finding(
            "FP32 accumulation; FP32 K/V cache",
            "confirmed",
            first_evidence(artifact, ["float acc = 0.0f"]),
            first_evidence(kv, ["2.0 * sizeof(float)"]),
        ),
        "weight_quantization": finding(
            "source FP8/F16/BF16 decoded, then persisted as BF16 cache; Expert q3like decoded to FP32",
            "confirmed",
            first_evidence(artifact, ["std::vector<std::uint16_t> tensor_bf16"]),
            first_evidence(artifact, ["fp8_e4m3_to_float"]),
        ),
        "activation_quantization": finding(
            "none in C++ streamed kernels",
            "confirmed",
            first_evidence(expert, ["__global__ void matvec_rows_kernel(const float* weights"]),
        ),
    }


DISPLAY_ROWS = [
    ("Backend", "backend"),
    ("Model spec", "model_spec"),
    ("GDN executed", "gdn"),
    ("QKV", "qkv_projection"),
    ("Conv1D", "conv1d"),
    ("SiLU", "silu"),
    ("Prefill core", "prefill_core"),
    ("Decode core", "decode_core"),
    ("Gated RMSNorm", "gated_rmsnorm"),
    ("Block I/O dtype", "external_dtype"),
    ("Core/state dtype", "core_dtype"),
    ("Weight quantization", "weight_quantization"),
    ("Activation quantization", "activation_quantization"),
]


def clipped(value: str, width: int) -> str:
    return value if len(value) <= width else value[: width - 3] + "..."


def print_report(official: dict[str, Finding], streamed: dict[str, Finding], show_evidence: bool) -> None:
    label_width = 23
    value_width = 67
    print(f"{'Question':<{label_width}} {'official_hf':<{value_width}} streamed")
    print("-" * (label_width + value_width * 2 + 2))
    for label, key in DISPLAY_ROWS:
        off = official.get(key, Finding("n/a", "unknown", [])).value
        stream = streamed.get(key, Finding("n/a", "unknown", [])).value
        print(f"{label:<{label_width}} {clipped(off, value_width):<{value_width}} {stream}")

    if not show_evidence:
        return
    print("\nEvidence")
    for mode, findings in (("official_hf", official), ("streamed", streamed)):
        print(f"\n[{mode}]")
        for label, key in DISPLAY_ROWS:
            item = findings.get(key)
            if item is None or not item.evidence:
                continue
            print(f"  {label}: {item.status}")
            for evidence in item.evidence:
                print(f"    {evidence.file}:{evidence.line}: {evidence.text}")


def main() -> int:
    script = Path(__file__).resolve()
    default_repo = script.parent.parent
    parser = argparse.ArgumentParser(
        description="Inspect the real operator and dtype paths used by official_hf and flashmoe_streamed_cli."
    )
    parser.add_argument("--model", type=Path, help="Local Qwen model directory containing config.json.")
    parser.add_argument("--config", type=Path, help="Explicit config.json path; overrides --model/config.json.")
    parser.add_argument("--repo", type=Path, default=default_repo, help="gb10_flashmoe repository root.")
    parser.add_argument("--transformers-source", type=Path, help="Explicit modeling_qwen3_5_moe.py path.")
    parser.add_argument("--json-out", type=Path, help="Write the complete machine-readable report.")
    parser.add_argument("--no-evidence", action="store_true", help="Suppress source evidence lines in terminal output.")
    args = parser.parse_args()

    config_path = model_config_path(args.model, args.config)
    config = load_config(config_path)
    hf_source = args.transformers_source.resolve() if args.transformers_source else locate_transformers_source()
    official = inspect_official(config, config_path, hf_source)
    streamed = inspect_streamed(args.repo.resolve())

    payload = {
        "model_config": str(config_path) if config_path else None,
        "transformers_source": str(hf_source) if hf_source else None,
        "official_hf": {key: asdict(value) for key, value in official.items()},
        "streamed": {key: asdict(value) for key, value in streamed.items()},
    }
    print_report(official, streamed, not args.no_evidence)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
