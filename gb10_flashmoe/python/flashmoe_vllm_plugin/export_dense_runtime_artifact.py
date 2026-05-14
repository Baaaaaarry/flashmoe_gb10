from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .export_qwen_experts import (
        require_deps,
        resolve_model_dir,
        load_index,
    )
except ImportError:
    import sys
    from pathlib import Path as _Path

    current_dir = _Path(__file__).resolve().parent
    package_root = current_dir.parent
    sys.path.append(str(package_root))
    sys.path.append(str(current_dir))
    try:
        from flashmoe_vllm_plugin.export_qwen_experts import (
            require_deps,
            resolve_model_dir,
            load_index,
        )
    except ImportError:
        from export_qwen_experts import (
            require_deps,
            resolve_model_dir,
            load_index,
        )


def _find_first_key(keys: set[str], candidates: list[str]) -> str:
    for key in candidates:
        if key in keys:
            return key
    raise SystemExit(f"None of the expected keys were found: {candidates}")


def _maybe_find_first_key(keys: set[str], candidates: list[str]) -> str:
    for key in candidates:
        if key in keys:
            return key
    return ""


def _find_router_key(keys: set[str], layer: int) -> str:
    return _find_first_key(
        keys,
        [
            f"model.language_model.layers.{layer}.mlp.gate.weight",
            f"language_model.layers.{layer}.mlp.gate.weight",
            f"model.layers.{layer}.mlp.gate.weight",
            f"layers.{layer}.mlp.gate.weight",
        ],
    )


def _find_attention_key(keys: set[str], layer: int, proj: str) -> str:
    aliases = [proj]
    if proj == "o_proj":
        aliases.append("out_proj")
    return _find_first_key(
        keys,
        [
            candidate
            for alias in aliases
            for candidate in (
                f"model.language_model.layers.{layer}.self_attn.{alias}.weight",
                f"language_model.layers.{layer}.self_attn.{alias}.weight",
                f"model.layers.{layer}.self_attn.{alias}.weight",
                f"layers.{layer}.self_attn.{alias}.weight",
            )
        ],
    )


def _find_attention_key_optional(keys: set[str], layer: int, proj: str) -> str:
    aliases = [proj]
    if proj == "o_proj":
        aliases.append("out_proj")
    return _maybe_find_first_key(
        keys,
        [
            candidate
            for alias in aliases
            for candidate in (
                f"model.language_model.layers.{layer}.self_attn.{alias}.weight",
                f"language_model.layers.{layer}.self_attn.{alias}.weight",
                f"model.layers.{layer}.self_attn.{alias}.weight",
                f"layers.{layer}.self_attn.{alias}.weight",
            )
        ],
    )


def _find_linear_attn_key_optional(keys: set[str], layer: int, name: str) -> str:
    return _maybe_find_first_key(
        keys,
        [
            f"model.language_model.layers.{layer}.linear_attn.{name}.weight",
            f"language_model.layers.{layer}.linear_attn.{name}.weight",
            f"model.layers.{layer}.linear_attn.{name}.weight",
            f"layers.{layer}.linear_attn.{name}.weight",
        ],
    )


def _find_norm_key(keys: set[str], layer: int, name: str) -> str:
    return _find_first_key(
        keys,
        [
            f"model.language_model.layers.{layer}.{name}.weight",
            f"language_model.layers.{layer}.{name}.weight",
            f"model.layers.{layer}.{name}.weight",
            f"layers.{layer}.{name}.weight",
        ],
    )


def _find_norm_key_optional(keys: set[str], layer: int, name: str) -> str:
    return _maybe_find_first_key(
        keys,
        [
            f"model.language_model.layers.{layer}.{name}.weight",
            f"language_model.layers.{layer}.{name}.weight",
            f"model.layers.{layer}.{name}.weight",
            f"layers.{layer}.{name}.weight",
        ],
    )


def _sampled_indices(hidden_size: int, sample_stride: int, max_dims: int) -> list[int]:
    indices = list(range(0, hidden_size, sample_stride))
    if len(indices) > max_dims:
        indices = indices[:max_dims]
    if not indices:
        indices = [0]
    return indices


def _sample_square_weight(tensor, sampled_tensor):
    if tensor.dim() != 2:
        raise SystemExit(f"Expected 2D attention tensor, got shape {tuple(tensor.shape)}")
    return tensor.index_select(0, sampled_tensor).index_select(1, sampled_tensor).contiguous()


def _sample_norm_weight(tensor, sampled_tensor):
    if tensor.dim() != 1:
        raise SystemExit(f"Expected 1D norm tensor, got shape {tuple(tensor.shape)}")
    return tensor.index_select(0, sampled_tensor).contiguous()


def _sample_direct_projection(tensor, sampled_tensor, hidden_size: int):
    if tensor.dim() != 2:
        raise SystemExit(f"Expected 2D projection tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] == hidden_size:
        rows = tensor.shape[0]
        row_index = sampled_tensor.clamp_max(rows - 1)
        return tensor.index_select(0, row_index).index_select(1, sampled_tensor).contiguous()
    if tensor.shape[0] == hidden_size:
        cols = tensor.shape[1]
        col_index = sampled_tensor.clamp_max(cols - 1)
        return tensor.index_select(0, sampled_tensor).index_select(1, col_index).contiguous()
    raise SystemExit(f"Direct projection tensor does not consume hidden states: {tuple(tensor.shape)}")


def _sample_rectangular_projection(tensor, sampled_tensor):
    if tensor.dim() != 2:
        raise SystemExit(f"Expected 2D projection tensor, got shape {tuple(tensor.shape)}")
    rows = min(tensor.shape[0], sampled_tensor.numel())
    cols = min(tensor.shape[1], sampled_tensor.numel())
    return tensor[:rows, :].index_select(1, sampled_tensor[:cols]).contiguous()


def _materialize_low_rank_projection(left, right, sampled_tensor, hidden_size: int):
    if left.dim() != 2 or right.dim() != 2:
        raise SystemExit(f"Expected 2D low-rank projection tensors, got {tuple(left.shape)} and {tuple(right.shape)}")
    if left.shape[1] != hidden_size:
        raise SystemExit(f"Low-rank left tensor does not consume hidden states: {tuple(left.shape)}")
    if right.shape[1] != left.shape[0]:
        raise SystemExit(f"Low-rank projection dimensions do not align: {tuple(left.shape)} and {tuple(right.shape)}")
    if right.shape[0] != hidden_size:
        raise SystemExit(f"Low-rank right tensor does not project back to hidden size: {tuple(right.shape)}")
    sampled_left = left.index_select(1, sampled_tensor).contiguous()
    sampled_right = right.index_select(0, sampled_tensor).contiguous()
    return sampled_right @ sampled_left


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dense runtime artifact (embed/router/lm_head) for streamed runtime.")
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face repo id.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", default="", help="Optional HF cache dir.")
    parser.add_argument("--sample-stride", type=int, default=64)
    parser.add_argument("--max-sampled-dims", type=int, default=64)
    parser.add_argument("--runtime-vocab-size", type=int, default=32)
    args = parser.parse_args()

    torch, snapshot_download, LocalEntryNotFoundError, HfHubHTTPError, safe_open, _ = require_deps()
    model_dir = resolve_model_dir(
        args.model,
        args.cache_dir or None,
        snapshot_download,
        LocalEntryNotFoundError,
        HfHubHTTPError,
    )
    weight_map = load_index(model_dir)
    keys = set(weight_map)
    if not keys:
        raise SystemExit("Dense runtime artifact export requires model.safetensors.index.json")

    embed_key = _find_first_key(
        keys,
        [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        ],
    )
    lm_head_key = _find_first_key(
        keys,
        [
            "lm_head.weight",
            "model.lm_head.weight",
            "language_model.lm_head.weight",
        ],
    )

    with safe_open(weight_map[embed_key], framework="pt", device="cpu") as handle:
        embed = handle.get_tensor(embed_key).to(dtype=torch.float32)
    hidden_size = int(embed.shape[1])
    sampled = _sampled_indices(hidden_size, args.sample_stride, args.max_sampled_dims)
    sampled_tensor = torch.tensor(sampled, dtype=torch.long)

    runtime_vocab_size = min(args.runtime_vocab_size, int(embed.shape[0]))
    embed_sampled = embed.index_select(1, sampled_tensor)[:runtime_vocab_size].contiguous()

    with safe_open(weight_map[lm_head_key], framework="pt", device="cpu") as handle:
        lm_head = handle.get_tensor(lm_head_key).to(dtype=torch.float32)
    lm_head_sampled = lm_head.index_select(1, sampled_tensor)[:runtime_vocab_size].contiguous()

    router_rows = []
    q_rows = []
    k_rows = []
    v_rows = []
    o_rows = []
    input_norm_rows = []
    post_norm_rows = []
    layer = 0
    while True:
        try:
            router_key = _find_router_key(keys, layer)
        except SystemExit:
            break
        with safe_open(weight_map[router_key], framework="pt", device="cpu") as handle:
            router = handle.get_tensor(router_key).to(dtype=torch.float32)
        if router.dim() != 2:
            raise SystemExit(f"Unexpected router tensor shape at layer {layer}: {tuple(router.shape)}")
        if router.shape[1] == hidden_size:
            sampled_router = router.index_select(1, sampled_tensor).contiguous()
        elif router.shape[0] == hidden_size:
            sampled_router = router.index_select(0, sampled_tensor).transpose(0, 1).contiguous()
        else:
            raise SystemExit(f"Router tensor shape does not match hidden size at layer {layer}: {tuple(router.shape)}")
        router_rows.append(sampled_router)

        q_key = _find_attention_key_optional(keys, layer, "q_proj")
        k_key = _find_attention_key_optional(keys, layer, "k_proj")
        v_key = _find_attention_key_optional(keys, layer, "v_proj")
        o_key = _find_attention_key_optional(keys, layer, "o_proj")
        input_norm_key = _find_norm_key_optional(keys, layer, "input_layernorm")
        post_norm_key = _find_norm_key_optional(keys, layer, "post_attention_layernorm")

        if q_key and k_key and v_key and o_key:
            with safe_open(weight_map[q_key], framework="pt", device="cpu") as handle:
                q_rows.append(_sample_direct_projection(handle.get_tensor(q_key).to(dtype=torch.float32), sampled_tensor, hidden_size))
            with safe_open(weight_map[k_key], framework="pt", device="cpu") as handle:
                k_rows.append(_sample_direct_projection(handle.get_tensor(k_key).to(dtype=torch.float32), sampled_tensor, hidden_size))
            with safe_open(weight_map[v_key], framework="pt", device="cpu") as handle:
                v_rows.append(_sample_direct_projection(handle.get_tensor(v_key).to(dtype=torch.float32), sampled_tensor, hidden_size))
            with safe_open(weight_map[o_key], framework="pt", device="cpu") as handle:
                o_rows.append(_sample_direct_projection(handle.get_tensor(o_key).to(dtype=torch.float32), sampled_tensor, hidden_size))
        else:
            q_a_key = _find_attention_key_optional(keys, layer, "q_a_proj")
            q_b_key = _find_attention_key_optional(keys, layer, "q_b_proj")
            kv_a_key = _find_attention_key_optional(keys, layer, "kv_a_proj_with_mqa")
            kv_b_key = _find_attention_key_optional(keys, layer, "kv_b_proj")
            if q_a_key and q_b_key and kv_a_key and kv_b_key:
                o_key = _find_attention_key(keys, layer, "o_proj")
                with safe_open(weight_map[q_a_key], framework="pt", device="cpu") as handle_a, \
                     safe_open(weight_map[q_b_key], framework="pt", device="cpu") as handle_b:
                    q_rows.append(
                        _materialize_low_rank_projection(
                            handle_a.get_tensor(q_a_key).to(dtype=torch.float32),
                            handle_b.get_tensor(q_b_key).to(dtype=torch.float32),
                            sampled_tensor,
                            hidden_size,
                        )
                    )
                with safe_open(weight_map[kv_a_key], framework="pt", device="cpu") as handle_a, \
                     safe_open(weight_map[kv_b_key], framework="pt", device="cpu") as handle_b:
                    kv_effective = _materialize_low_rank_projection(
                        handle_a.get_tensor(kv_a_key).to(dtype=torch.float32),
                        handle_b.get_tensor(kv_b_key).to(dtype=torch.float32),
                        sampled_tensor,
                        hidden_size,
                    )
                    k_rows.append(kv_effective)
                    v_rows.append(kv_effective.clone())
                with safe_open(weight_map[o_key], framework="pt", device="cpu") as handle:
                    o_rows.append(_sample_direct_projection(handle.get_tensor(o_key).to(dtype=torch.float32), sampled_tensor, hidden_size))
            else:
                linear_qkv_key = _find_linear_attn_key_optional(keys, layer, "in_proj_qkv")
                linear_out_key = _find_linear_attn_key_optional(keys, layer, "out_proj")
                linear_a_key = _find_linear_attn_key_optional(keys, layer, "in_proj_a")
                linear_b_key = _find_linear_attn_key_optional(keys, layer, "in_proj_b")
                if linear_qkv_key and linear_out_key:
                    with safe_open(weight_map[linear_qkv_key], framework="pt", device="cpu") as handle:
                        qkv = handle.get_tensor(linear_qkv_key).to(dtype=torch.float32)
                    if qkv.dim() != 2:
                        raise SystemExit(f"Unexpected linear_attn in_proj_qkv shape at layer {layer}: {tuple(qkv.shape)}")
                    rows = qkv.shape[0]
                    chunk = max(1, rows // 3)
                    q_rows.append(_sample_rectangular_projection(qkv[:chunk, :], sampled_tensor))
                    k_rows.append(_sample_rectangular_projection(qkv[chunk : min(rows, 2 * chunk), :], sampled_tensor))
                    v_rows.append(_sample_rectangular_projection(qkv[min(rows, 2 * chunk) :, :], sampled_tensor))
                    with safe_open(weight_map[linear_out_key], framework="pt", device="cpu") as handle:
                        o_rows.append(_sample_rectangular_projection(handle.get_tensor(linear_out_key).to(dtype=torch.float32), sampled_tensor))
                elif linear_a_key and linear_b_key and linear_out_key:
                    with safe_open(weight_map[linear_a_key], framework="pt", device="cpu") as handle_a, \
                         safe_open(weight_map[linear_b_key], framework="pt", device="cpu") as handle_b:
                        effective = _materialize_low_rank_projection(
                            handle_a.get_tensor(linear_a_key).to(dtype=torch.float32),
                            handle_b.get_tensor(linear_b_key).to(dtype=torch.float32),
                            sampled_tensor,
                            hidden_size,
                        )
                    q_rows.append(effective)
                    k_rows.append(effective.clone())
                    v_rows.append(effective.clone())
                    with safe_open(weight_map[linear_out_key], framework="pt", device="cpu") as handle:
                        o_rows.append(_sample_rectangular_projection(handle.get_tensor(linear_out_key).to(dtype=torch.float32), sampled_tensor))
                else:
                    raise SystemExit(
                        f"Layer {layer} is missing supported self_attn and linear_attn projection weights"
                    )

        if not input_norm_key:
            input_norm_key = _find_norm_key_optional(keys, layer, "self_attn.q_a_layernorm")
        if not post_norm_key:
            post_norm_key = _find_norm_key_optional(keys, layer, "self_attn.kv_a_layernorm")
        if not input_norm_key:
            input_norm_key = _find_norm_key_optional(keys, layer, "linear_attn.in_norm")
        if not post_norm_key:
            post_norm_key = _find_norm_key_optional(keys, layer, "linear_attn.out_norm")
        if not input_norm_key or not post_norm_key:
            raise SystemExit(f"Layer {layer} is missing supported norm weights for dense runtime artifact export")
        with safe_open(weight_map[input_norm_key], framework="pt", device="cpu") as handle:
            input_norm_rows.append(_sample_norm_weight(handle.get_tensor(input_norm_key).to(dtype=torch.float32), sampled_tensor))
        with safe_open(weight_map[post_norm_key], framework="pt", device="cpu") as handle:
            post_norm_rows.append(_sample_norm_weight(handle.get_tensor(post_norm_key).to(dtype=torch.float32), sampled_tensor))
        layer += 1

    if not router_rows:
        raise SystemExit("No router gate weights detected for dense runtime artifact export")

    router_stacked = torch.stack(router_rows, dim=0).contiguous()
    q_stacked = torch.stack(q_rows, dim=0).contiguous()
    k_stacked = torch.stack(k_rows, dim=0).contiguous()
    v_stacked = torch.stack(v_rows, dim=0).contiguous()
    o_stacked = torch.stack(o_rows, dim=0).contiguous()
    input_norm_stacked = torch.stack(input_norm_rows, dim=0).contiguous()
    post_norm_stacked = torch.stack(post_norm_rows, dim=0).contiguous()
    routed_experts_per_layer = int(router_stacked.shape[1])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embed_sampled.bin"
    q_proj_path = output_dir / "q_proj_sampled.bin"
    k_proj_path = output_dir / "k_proj_sampled.bin"
    v_proj_path = output_dir / "v_proj_sampled.bin"
    o_proj_path = output_dir / "o_proj_sampled.bin"
    input_norm_path = output_dir / "input_norm_sampled.bin"
    post_norm_path = output_dir / "post_norm_sampled.bin"
    router_path = output_dir / "router_sampled.bin"
    lm_head_path = output_dir / "lm_head_sampled.bin"
    embedding_path.write_bytes(embed_sampled.numpy().tobytes())
    q_proj_path.write_bytes(q_stacked.numpy().tobytes())
    k_proj_path.write_bytes(k_stacked.numpy().tobytes())
    v_proj_path.write_bytes(v_stacked.numpy().tobytes())
    o_proj_path.write_bytes(o_stacked.numpy().tobytes())
    input_norm_path.write_bytes(input_norm_stacked.numpy().tobytes())
    post_norm_path.write_bytes(post_norm_stacked.numpy().tobytes())
    router_path.write_bytes(router_stacked.numpy().tobytes())
    lm_head_path.write_bytes(lm_head_sampled.numpy().tobytes())

    manifest = {
        "family": "qwen3.5-397b-a17b",
        "sampled_indices": sampled,
        "runtime_vocab_size": runtime_vocab_size,
        "num_layers": len(router_rows),
        "routed_experts_per_layer": routed_experts_per_layer,
        "embedding_path": str(embedding_path),
        "q_proj_path": str(q_proj_path),
        "k_proj_path": str(k_proj_path),
        "v_proj_path": str(v_proj_path),
        "o_proj_path": str(o_proj_path),
        "input_norm_path": str(input_norm_path),
        "post_norm_path": str(post_norm_path),
        "router_path": str(router_path),
        "lm_head_path": str(lm_head_path),
    }
    manifest_path = output_dir / "dense_runtime_artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"model_dir={model_dir}")
    print(f"output_dir={output_dir}")
    print(f"dense_artifact={manifest_path}")
    print(f"sampled_dims={len(sampled)}")
    print(f"num_layers={len(router_rows)}")
    print(f"runtime_vocab_size={runtime_vocab_size}")


if __name__ == "__main__":
    main()
