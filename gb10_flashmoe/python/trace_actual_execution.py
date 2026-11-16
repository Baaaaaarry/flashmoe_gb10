#!/usr/bin/env python3
"""Trace modules, weights, tensor dtypes, and operators used by a real HF forward."""

from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace actual official-HF prefill and one-token decode execution."
    )
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory.")
    parser.add_argument("--prompt", default="Explain unified memory briefly.")
    parser.add_argument("--prompt-file", help="Read the prompt from this UTF-8 file.")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument(
        "--module-regex",
        default=r"(^|\.)(linear_attn|self_attn)(\.|$)",
        help="Only record modules whose qualified name matches this expression.",
    )
    parser.add_argument(
        "--operator-regex",
        default=r"(linear|gemm|mm|conv|silu|delta|rms|norm|mul|add)",
        help="Only print profiler operators matching this expression; traces retain all events.",
    )
    parser.add_argument("--output-dir", default="execution-trace")
    parser.add_argument("--no-profiler", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def resolve_dtype(torch: Any, name: str) -> Any:
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def tensor_info(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [tensor_info(torch, item) for item in value[:8]]
    if isinstance(value, dict):
        return {str(key): tensor_info(torch, item) for key, item in list(value.items())[:16]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def direct_tensors(module: Any) -> list[dict[str, Any]]:
    tensors: list[dict[str, Any]] = []
    for kind, values in (("parameter", module.named_parameters(recurse=False)), ("buffer", module.named_buffers(recurse=False))):
        for name, value in values:
            tensors.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "device": str(value.device),
                    "requires_grad": bool(getattr(value, "requires_grad", False)),
                }
            )
    return tensors


def first_tensor_signature(value: Any) -> str:
    if isinstance(value, dict) and {"shape", "dtype", "device"}.issubset(value):
        shape = "x".join(map(str, value["shape"])) or "scalar"
        return f"{value['dtype']}:{shape}@{value['device']}"
    if isinstance(value, dict):
        for item in value.values():
            signature = first_tensor_signature(item)
            if signature != "none":
                return signature
    if isinstance(value, list):
        for item in value:
            signature = first_tensor_signature(item)
            if signature != "none":
                return signature
    return "none"


def make_profiler(torch: Any, enabled: bool):
    if not enabled:
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def profiler_rows(profile: Any, operator_pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    if profile is None:
        return []
    rows: list[dict[str, Any]] = []
    for event in profile.key_averages(group_by_input_shape=True):
        if not operator_pattern.search(event.key):
            continue
        rows.append(
            {
                "operator": event.key,
                "calls": event.count,
                "cpu_total_us": event.cpu_time_total,
                "cuda_total_us": getattr(
                    event, "device_time_total", getattr(event, "cuda_time_total", 0.0)
                ),
                "input_shapes": event.input_shapes,
            }
        )
    rows.sort(key=lambda row: row["cuda_total_us"] or row["cpu_total_us"], reverse=True)
    return rows


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt

    device_map: Any = args.device_map
    if args.device_map in {"cuda", "cuda:0", "gpu", "gpu:0"}:
        device_map = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        torch_dtype=resolve_dtype(torch, args.torch_dtype),
    )
    model.eval()

    module_pattern = re.compile(args.module_regex, re.IGNORECASE)
    operator_pattern = re.compile(args.operator_regex, re.IGNORECASE)
    phase = {"name": "load"}
    calls: dict[tuple[str, str], dict[str, Any]] = {}
    handles = []

    def hook_for(name: str, module: Any):
        def hook(_module: Any, inputs: Any, output: Any):
            key = (phase["name"], name)
            entry = calls.setdefault(
                key,
                {
                    "phase": phase["name"],
                    "module": name,
                    "class": f"{module.__class__.__module__}.{module.__class__.__name__}",
                    "calls": 0,
                    "weights": direct_tensors(module),
                    "input": tensor_info(torch, inputs),
                    "output": tensor_info(torch, output),
                },
            )
            entry["calls"] += 1
            entry["output"] = tensor_info(torch, output)

        return hook

    for name, module in model.named_modules():
        if name and module_pattern.search(name):
            handles.append(module.register_forward_hook(hook_for(name, module)))

    encoded = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in encoded.items()}
    profiles: dict[str, Any] = {}

    with torch.inference_mode():
        phase["name"] = "prefill"
        with make_profiler(torch, not args.no_profiler) as profile:
            prefill = model(**inputs, use_cache=True, return_dict=True)
        profiles["prefill"] = profile

        next_token = prefill.logits[:, -1:].argmax(dim=-1)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)], dim=-1
            )
        decode_inputs = model.prepare_inputs_for_generation(
            next_token,
            past_key_values=prefill.past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        phase["name"] = "decode"
        with make_profiler(torch, not args.no_profiler) as profile:
            model(**decode_inputs, return_dict=True)
        profiles["decode"] = profile

    for handle in handles:
        handle.remove()

    operators: dict[str, list[dict[str, Any]]] = {}
    for phase_name, profile in profiles.items():
        operators[phase_name] = profiler_rows(profile, operator_pattern)
        if profile is not None:
            profile.export_chrome_trace(str(output_dir / f"{phase_name}-chrome-trace.json"))

    report = {
        "model": str(Path(args.model).resolve()),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "module_regex": args.module_regex,
        "modules": sorted(calls.values(), key=lambda item: (item["phase"], item["module"])),
        "operators": operators,
    }
    report_path = output_dir / "execution-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"model_class: {report['model_class']}")
    print(f"prompt_tokens: {report['prompt_tokens']}")
    for phase_name in ("prefill", "decode"):
        print(f"\n[{phase_name}: executed modules]")
        for entry in report["modules"]:
            if entry["phase"] != phase_name:
                continue
            weights = ", ".join(
                f"{item['name']}:{item['dtype']}:{'x'.join(map(str, item['shape']))}"
                for item in entry["weights"]
            ) or "no-direct-weights"
            print(
                f"  {entry['module']} [{entry['class'].rsplit('.', 1)[-1]}] calls={entry['calls']} "
                f"input={first_tensor_signature(entry['input'])} "
                f"output={first_tensor_signature(entry['output'])} weights={weights}"
            )
        print(f"\n[{phase_name}: matching profiler operators]")
        for row in operators[phase_name][:80]:
            print(
                f"  {row['operator']} calls={row['calls']} cpu_us={row['cpu_total_us']:.1f} "
                f"cuda_us={row['cuda_total_us']:.1f} shapes={row['input_shapes']}"
            )
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
