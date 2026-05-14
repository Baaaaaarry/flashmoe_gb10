from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import re

try:
    from .q3like import pack_q3_tensor
except ImportError:
    import sys
    from pathlib import Path as _Path

    current_dir = _Path(__file__).resolve().parent
    package_root = current_dir.parent
    sys.path.append(str(package_root))
    sys.path.append(str(current_dir))
    try:
        from flashmoe_vllm_plugin.q3like import pack_q3_tensor
    except ImportError:
        from q3like import pack_q3_tensor


EXPERT_KEY_PATTERN = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
EXPERT_KEY_PATTERN_ALT = re.compile(
    r"^(?:model\.)?language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
FUSED_GATE_UP_PATTERN = re.compile(
    r"^(?:model\.)?language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.gate_up_proj$"
)
FUSED_DOWN_PATTERN = re.compile(
    r"^(?:model\.)?language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.down_proj$"
)


def require_deps():
    try:
        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for expert export. Install: pip install huggingface_hub safetensors"
        ) from exc
    return torch, snapshot_download, LocalEntryNotFoundError, HfHubHTTPError, safe_open, save_file


def _looks_like_downloaded_model(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").exists()


def resolve_model_dir(model: str, cache_dir: str | None, snapshot_download, LocalEntryNotFoundError, HfHubHTTPError) -> Path:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return candidate.resolve()

    download_root = Path(cache_dir).expanduser().resolve() if cache_dir else (Path.home() / ".cache" / "flashmoe_hf")
    download_root.mkdir(parents=True, exist_ok=True)
    local_dir = download_root / model.replace("/", "__")
    if local_dir.exists() and _looks_like_downloaded_model(local_dir):
        return local_dir

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        snapshot_download(
            repo_id=model,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
    except LocalEntryNotFoundError as exc:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        raise SystemExit(
            "Unable to download model from Hugging Face because the host is unreachable.\n"
            f"repo_id: {model}\n"
            f"endpoint: {endpoint}\n\n"
            "Use one of these paths:\n"
            "1. Pass a local model directory instead of a repo id.\n"
            "2. Download the model on another machine and copy it to the GB10 host.\n"
            "3. Configure network access or a mirror, for example HF_ENDPOINT / proxy / VPN.\n"
            "4. If the model is gated, make sure HF_TOKEN is set.\n\n"
            f"Original error: {exc}"
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(
            "Hugging Face request failed.\n"
            f"repo_id: {model}\n"
            "Check repository name, token permissions, or network access.\n"
            f"Original error: {exc}"
        ) from exc
    return local_dir


def normalize_dtype(dtype_name: str, torch):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise SystemExit(f"Unsupported export dtype: {dtype_name}") from exc


def load_index(model_dir: Path) -> dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        return {}
    raw = json.loads(index_file.read_text())
    return {key: str(model_dir / value) for key, value in raw.get("weight_map", {}).items()}


def list_weight_files(model_dir: Path, weight_map: dict[str, str]) -> list[Path]:
    if weight_map:
        return sorted({Path(path) for path in weight_map.values()})
    files = sorted(model_dir.glob("*.safetensors"))
    if files:
        return files
    raise SystemExit(f"No safetensors weights found under {model_dir}")


def merge_manifest_entries(manifest_path: Path, new_entries: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[int, int], dict[str, object]] = {}
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text())
        for item in raw.get("entries", []):
            merged[(int(item["layer_id"]), int(item["expert_id"]))] = item
    for item in new_entries:
        merged[(int(item["layer_id"]), int(item["expert_id"]))] = item
    return [merged[key] for key in sorted(merged)]


def save_expert_bundle(
    gate_tensor,
    up_tensor,
    down_tensor,
    target: Path,
    save_file,
    export_format: str,
    layer_id: int,
    expert_id: int,
) -> dict[str, object]:
    if export_format == "dense":
        payload = {
            "gate_proj.weight": gate_tensor,
            "up_proj.weight": up_tensor,
            "down_proj.weight": down_tensor,
        }
        metadata = {"layer_id": str(layer_id), "expert_id": str(expert_id), "format": "dense"}
    elif export_format == "q3like":
        gate_q = pack_q3_tensor(gate_tensor)
        up_q = pack_q3_tensor(up_tensor)
        down_q = pack_q3_tensor(down_tensor)
        payload = {
            "gate_proj.qweight": gate_q["qweight"],
            "gate_proj.scale": gate_q["scale"],
            "gate_proj.shape": gate_q["shape"],
            "up_proj.qweight": up_q["qweight"],
            "up_proj.scale": up_q["scale"],
            "up_proj.shape": up_q["shape"],
            "down_proj.qweight": down_q["qweight"],
            "down_proj.scale": down_q["scale"],
            "down_proj.shape": down_q["shape"],
        }
        metadata = {"layer_id": str(layer_id), "expert_id": str(expert_id), "format": "q3like"}
    else:
        raise SystemExit(f"Unsupported export format: {export_format}")

    save_file(payload, str(target), metadata=metadata)
    return {
        "layer_id": layer_id,
        "expert_id": expert_id,
        "path": str(target.resolve()),
        "offset": 0,
        "size_bytes": target.stat().st_size,
        "format": export_format,
    }


def export_fused_qwen_streaming(
    model_dir: Path,
    output_dir: Path,
    export_dtype_name: str,
    export_format: str,
    manifest_path: Path,
    layer_start: int | None,
    layer_end: int | None,
) -> int:
    torch, _, _, _, safe_open, save_file = require_deps()
    export_dtype = normalize_dtype(export_dtype_name, torch)
    weight_map = load_index(model_dir)
    entries: list[dict[str, object]] = []

    fused_by_layer: dict[int, dict[str, str]] = defaultdict(dict)
    for key, filename in weight_map.items():
        gate_match = FUSED_GATE_UP_PATTERN.match(key)
        if gate_match:
            fused_by_layer[int(gate_match.group("layer"))]["gate_up_proj"] = filename
            continue
        down_match = FUSED_DOWN_PATTERN.match(key)
        if down_match:
            fused_by_layer[int(down_match.group("layer"))]["down_proj"] = filename

    if not fused_by_layer:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_id in sorted(fused_by_layer):
        if layer_start is not None and layer_id < layer_start:
            continue
        if layer_end is not None and layer_id > layer_end:
            continue
        files = fused_by_layer[layer_id]
        if "gate_up_proj" not in files or "down_proj" not in files:
            raise SystemExit(f"Layer {layer_id} fused expert tensors are incomplete: {files}")

        gate_path = files["gate_up_proj"]
        down_path = files["down_proj"]

        with safe_open(gate_path, framework="pt", device="cpu") as gate_handle:
            gate_key = f"model.language_model.layers.{layer_id}.mlp.experts.gate_up_proj"
            if gate_key not in gate_handle.keys():
                gate_key = f"language_model.layers.{layer_id}.mlp.experts.gate_up_proj"
            gate_slice = gate_handle.get_slice(gate_key)
            gate_shape = gate_slice.get_shape()
            if len(gate_shape) != 3 or gate_shape[1] % 2 != 0:
                raise SystemExit(f"Unexpected gate_up shape for layer {layer_id}: {gate_shape}")
            num_experts = gate_shape[0]
            split = gate_shape[1] // 2

            with safe_open(down_path, framework="pt", device="cpu") as down_handle:
                down_key = f"model.language_model.layers.{layer_id}.mlp.experts.down_proj"
                if down_key not in down_handle.keys():
                    down_key = f"language_model.layers.{layer_id}.mlp.experts.down_proj"
                down_slice = down_handle.get_slice(down_key)
                down_shape = down_slice.get_shape()
                if len(down_shape) != 3 or down_shape[0] != num_experts:
                    raise SystemExit(f"Unexpected down_proj shape for layer {layer_id}: {down_shape}")

                for expert_id in range(num_experts):
                    gate_up_tensor = gate_slice[expert_id : expert_id + 1, :, :]
                    if gate_up_tensor.dim() == 3:
                        gate_up_tensor = gate_up_tensor.squeeze(0)
                    gate_up_tensor = gate_up_tensor.to(dtype=export_dtype).contiguous()
                    gate_tensor = gate_up_tensor[:split, :].contiguous()
                    up_tensor = gate_up_tensor[split:, :].contiguous()

                    down_tensor = down_slice[expert_id : expert_id + 1, :, :]
                    if down_tensor.dim() == 3:
                        down_tensor = down_tensor.squeeze(0)
                    down_tensor = down_tensor.to(dtype=export_dtype).contiguous()

                    target = output_dir / f"layer_{layer_id:03d}_expert_{expert_id:05d}.safetensors"
                    entries.append(
                        save_expert_bundle(
                            gate_tensor=gate_tensor,
                            up_tensor=up_tensor,
                            down_tensor=down_tensor,
                            target=target,
                            save_file=save_file,
                            export_format=export_format,
                            layer_id=layer_id,
                            expert_id=expert_id,
                        )
                    )
                    del gate_up_tensor, gate_tensor, up_tensor, down_tensor
                gc.collect()

    merged_entries = merge_manifest_entries(manifest_path, entries)
    manifest_path.write_text(json.dumps({"entries": merged_entries}, indent=2, sort_keys=True))
    return len(entries)


def export_nonfused_reference(
    weight_files: list[Path],
    output_dir: Path,
    export_dtype_name: str,
    export_format: str,
    manifest_path: Path,
    layer_start: int | None,
    layer_end: int | None,
) -> int:
    torch, _, _, _, safe_open, save_file = require_deps()
    export_dtype = normalize_dtype(export_dtype_name, torch)
    grouped: dict[tuple[int, int], dict[str, object]] = defaultdict(dict)

    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                match = EXPERT_KEY_PATTERN.match(key) or EXPERT_KEY_PATTERN_ALT.match(key)
                if not match:
                    continue
                layer_id = int(match.group("layer"))
                if layer_start is not None and layer_id < layer_start:
                    continue
                if layer_end is not None and layer_id > layer_end:
                    continue
                expert_id = int(match.group("expert"))
                proj_name = match.group("proj")
                grouped[(layer_id, expert_id)][proj_name] = handle.get_tensor(key).to(dtype=export_dtype).contiguous()

    if not grouped:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for (layer_id, expert_id), tensors in sorted(grouped.items()):
        missing = {"gate_proj", "up_proj", "down_proj"} - set(tensors)
        if missing:
            raise SystemExit(
                f"Expert export incomplete for layer={layer_id} expert={expert_id}: missing {sorted(missing)}"
            )
        target = output_dir / f"layer_{layer_id:03d}_expert_{expert_id:05d}.safetensors"
        entries.append(
            save_expert_bundle(
                gate_tensor=tensors["gate_proj"],
                up_tensor=tensors["up_proj"],
                down_tensor=tensors["down_proj"],
                target=target,
                save_file=save_file,
                export_format=export_format,
                layer_id=layer_id,
                expert_id=expert_id,
            )
        )
    merged_entries = merge_manifest_entries(manifest_path, entries)
    manifest_path.write_text(json.dumps({"entries": merged_entries}, indent=2, sort_keys=True))
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen routed experts from a local or Hugging Face model directory."
    )
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face repo id.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", default="", help="Optional local cache dir for Hugging Face download.")
    parser.add_argument("--export-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--export-format", default="dense", choices=["dense", "q3like"])
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional explicit output path for expert manifest JSON. Defaults to <output-dir>/expert_manifest.json",
    )
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    args = parser.parse_args()

    _, snapshot_download, LocalEntryNotFoundError, HfHubHTTPError, _, _ = require_deps()
    model_dir = resolve_model_dir(
        args.model,
        args.cache_dir or None,
        snapshot_download,
        LocalEntryNotFoundError,
        HfHubHTTPError,
    )
    weight_map = load_index(model_dir)
    weight_files = list_weight_files(model_dir, weight_map)
    manifest_path = args.manifest_output or (args.output_dir / "expert_manifest.json")

    exported = export_fused_qwen_streaming(
        model_dir=model_dir,
        output_dir=args.output_dir,
        export_dtype_name=args.export_dtype,
        export_format=args.export_format,
        manifest_path=manifest_path,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )
    if exported == 0:
        exported = export_nonfused_reference(
            weight_files=weight_files,
            output_dir=args.output_dir,
            export_dtype_name=args.export_dtype,
            export_format=args.export_format,
            manifest_path=manifest_path,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
        )
    if exported == 0:
        raise SystemExit(
            "No routed expert tensors were detected in the checkpoint.\n"
            "This usually means the exporter key patterns do not match the model format."
        )

    summary_path = args.output_dir / "export_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_model": str(model_dir),
                "export_dtype": args.export_dtype,
                "export_format": args.export_format,
                "num_experts": exported,
                "manifest_path": str(manifest_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(f"model_dir={model_dir}")
    print(f"weight_files={len(weight_files)}")
    print(f"exported_experts={exported}")
    print(f"output_dir={args.output_dir.resolve()}")
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
