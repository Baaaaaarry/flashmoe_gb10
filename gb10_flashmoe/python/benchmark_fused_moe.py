from __future__ import annotations

import argparse
import inspect
import json
import time
from typing import NamedTuple

from flashmoe_vllm_plugin.custom_op import FlashMoEFusedExpertsOp


class BenchmarkStats(NamedTuple):
    avg_ms: float
    est_payload_gbps: float
    active_experts: int
    routes_per_active_expert: float
    cuda_peak_alloc_gb: float
    cuda_peak_reserved_gb: float
    process_rss_gb: float
    system_used_gb: float
    system_total_gb: float


def _bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024 ** 3)


def _read_process_rss_bytes() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _read_system_memory_bytes() -> tuple[int, int]:
    total_kb = 0
    available_kb = 0
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
    total_bytes = total_kb * 1024
    used_bytes = (total_kb - available_kb) * 1024
    return used_bytes, total_bytes


def _estimate_payload_bytes(x, gate_w, up_w, down_w, route_ids, route_w) -> tuple[int, int]:
    import torch

    dtype_bytes = x.element_size()
    route_dtype_bytes = route_ids.element_size()
    route_weight_bytes = route_w.element_size()

    num_tokens = x.shape[0]
    hidden_dim = x.shape[1]
    top_k = route_ids.shape[1]
    inter_dim = gate_w.shape[1]

    active_experts = int(torch.unique(route_ids).numel())
    total_routes = num_tokens * top_k

    input_output_bytes = (2 * num_tokens * hidden_dim * dtype_bytes)
    routing_bytes = total_routes * (route_dtype_bytes + route_weight_bytes)
    packed_hidden_bytes = total_routes * hidden_dim * dtype_bytes
    mlp_activation_bytes = total_routes * ((3 * inter_dim) + hidden_dim) * dtype_bytes
    expert_weight_bytes = active_experts * (
        gate_w[0].numel() + up_w[0].numel() + down_w[0].numel()
    ) * dtype_bytes

    total_bytes = (
        input_output_bytes
        + routing_bytes
        + packed_hidden_bytes
        + mlp_activation_bytes
        + expert_weight_bytes
    )
    return total_bytes, active_experts


def torch_eager_impl(x, gate_w, up_w, down_w, route_ids, route_w):
    import torch

    num_tokens, hidden_dim = x.shape
    top_k = route_ids.shape[1]
    output = x.new_zeros((num_tokens, hidden_dim))
    for token_idx in range(num_tokens):
        token = x[token_idx:token_idx + 1]
        for k in range(top_k):
            expert_idx = int(route_ids[token_idx, k].item())
            prob = route_w[token_idx, k]
            gate = token @ gate_w[expert_idx].transpose(0, 1)
            up = token @ up_w[expert_idx].transpose(0, 1)
            act = torch.nn.functional.silu(gate) * up
            down = act @ down_w[expert_idx].transpose(0, 1)
            output[token_idx:token_idx + 1] += prob * down
    return output


def try_build_vllm_runner():
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    except Exception:
        return None

    signature = inspect.signature(fused_experts)
    required = {"hidden_states", "w1", "w2", "topk_weights", "topk_ids"}
    if not required.issubset(signature.parameters):
        return None

    activation = getattr(MoEActivation, "SILU", "silu")

    def runner(x, gate_w, up_w, down_w, route_ids, route_w):
        import torch

        w1 = torch.cat([gate_w, up_w], dim=1).contiguous()
        return fused_experts(
            hidden_states=x,
            w1=w1,
            w2=down_w,
            topk_weights=route_w,
            topk_ids=route_ids.to(torch.int32),
            inplace=False,
            activation=activation,
            apply_router_weight_on_input=False,
        )

    return runner


def build_fixed_active_routes(num_tokens: int,
                              top_k: int,
                              active_experts: int,
                              total_experts: int,
                              device,
                              dtype):
    import torch

    if active_experts < 1:
        raise ValueError("active_experts must be >= 1")
    if active_experts > total_experts:
        raise ValueError("active_experts cannot exceed total_experts")

    route_ids = torch.empty((num_tokens, top_k), device=device, dtype=torch.int32)
    for token_idx in range(num_tokens):
        for k in range(top_k):
            route_ids[token_idx, k] = (token_idx * top_k + k) % active_experts

    route_w = torch.full(
        (num_tokens, top_k),
        1.0 / max(top_k, 1),
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    return route_ids, route_w


def benchmark(label, fn, x, gate_w, up_w, down_w, route_ids, route_w, iters):
    import torch

    estimated_payload_bytes, active_experts = _estimate_payload_bytes(
        x, gate_w, up_w, down_w, route_ids, route_w)
    routes_per_active_expert = route_ids.numel() / max(active_experts, 1)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(10):
            y = fn(x, gate_w, up_w, down_w, route_ids, route_w)
    torch.cuda.synchronize()

    rss_before = _read_process_rss_bytes()
    system_used_before, system_total = _read_system_memory_bytes()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            y = fn(x, gate_w, up_w, down_w, route_ids, route_w)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rss_after = _read_process_rss_bytes()
    system_used_after, _ = _read_system_memory_bytes()

    avg_ms = (elapsed / iters) * 1000
    est_payload_gbps = estimated_payload_bytes / max(elapsed / iters, 1e-9) / 1e9
    stats = BenchmarkStats(
        avg_ms=avg_ms,
        est_payload_gbps=est_payload_gbps,
        active_experts=active_experts,
        routes_per_active_expert=routes_per_active_expert,
        cuda_peak_alloc_gb=_bytes_to_gb(torch.cuda.max_memory_allocated()),
        cuda_peak_reserved_gb=_bytes_to_gb(torch.cuda.max_memory_reserved()),
        process_rss_gb=_bytes_to_gb(max(rss_before, rss_after)),
        system_used_gb=_bytes_to_gb(max(system_used_before, system_used_after)),
        system_total_gb=_bytes_to_gb(system_total),
    )
    print(
        f"{label}: avg_ms={stats.avg_ms:.3f} "
        f"est_payload_gbps={stats.est_payload_gbps:.2f} "
        f"active_experts={stats.active_experts} "
        f"routes_per_active_expert={stats.routes_per_active_expert:.2f} "
        f"cuda_peak_alloc_gb={stats.cuda_peak_alloc_gb:.3f} "
        f"cuda_peak_reserved_gb={stats.cuda_peak_reserved_gb:.3f} "
        f"process_rss_gb={stats.process_rss_gb:.3f} "
        f"system_used_gb={stats.system_used_gb:.3f}/{stats.system_total_gb:.3f} "
        f"shape={tuple(y.shape)} dtype={y.dtype}"
    )
    return stats


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Benchmark FlashMoE fused CUDA op.")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--backend", default="all", choices=["all", "torch", "vllm", "flashmoe"])
    parser.add_argument("--emit-compute-profile", default="")
    parser.add_argument("--profile-active-experts", default="")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    device = torch.device("cuda")
    x = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
    gate_w = torch.randn(args.experts, args.intermediate, args.hidden, device=device, dtype=dtype)
    up_w = torch.randn(args.experts, args.intermediate, args.hidden, device=device, dtype=dtype)
    down_w = torch.randn(args.experts, args.hidden, args.intermediate, device=device, dtype=dtype)
    route_ids = torch.randint(0, args.experts, (args.tokens, args.topk), device=device, dtype=torch.int32)
    route_w = torch.softmax(torch.randn(args.tokens, args.topk, device=device, dtype=torch.float32), dim=-1).to(dtype)

    results = {}

    if args.backend in ("all", "torch"):
        results["torch_eager"] = benchmark(
            "torch_eager", torch_eager_impl, x, gate_w, up_w, down_w, route_ids, route_w, args.iters)

    if args.backend in ("all", "vllm"):
        vllm_runner = try_build_vllm_runner()
        if vllm_runner is None:
            print("vllm_fused_experts: unavailable in current vLLM install")
        else:
            results["vllm_fused_experts"] = benchmark(
                "vllm_fused_experts", vllm_runner, x, gate_w, up_w, down_w, route_ids, route_w, args.iters)

    if args.backend in ("all", "flashmoe"):
        op = FlashMoEFusedExpertsOp()
        results["flashmoe_custom"] = benchmark(
            "flashmoe_custom", op, x, gate_w, up_w, down_w, route_ids, route_w, args.iters)

    if args.emit_compute_profile:
        active_values = []
        if args.profile_active_experts:
            active_values = [int(part) for part in args.profile_active_experts.split(",") if part.strip()]
        if not active_values:
            active_values = list(range(1, args.topk + 1))

        op = FlashMoEFusedExpertsOp()
        points = []
        for active_experts in active_values:
            fixed_route_ids, fixed_route_w = build_fixed_active_routes(
                args.tokens,
                args.topk,
                active_experts,
                args.experts,
                device,
                dtype,
            )
            stats = benchmark(
                f"flashmoe_profile_active_{active_experts}",
                op,
                x,
                gate_w,
                up_w,
                down_w,
                fixed_route_ids,
                fixed_route_w,
                args.iters,
            )
            points.append({
                "active_experts": active_experts,
                "avg_ms": stats.avg_ms,
                "payload_gbps": stats.est_payload_gbps,
            })

        payload = {
            "model_hint": {
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "topk": args.topk,
                "dtype": args.dtype,
                "tokens": args.tokens,
            },
            "backend": "flashmoe_custom",
            "points": points,
        }
        with open(args.emit_compute_profile, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"wrote_compute_profile={args.emit_compute_profile}")


if __name__ == "__main__":
    main()
