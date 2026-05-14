from __future__ import annotations

import importlib
from typing import Any

from .custom_op import FlashMoEFusedExpertsOp


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for FlashMoE layer replacement") from exc
    return torch, nn, F


class FlashMoERoutedMoELayer:
    def __new__(cls, original_layer, layer_idx: int, runtime):
        torch, nn, F = _require_torch()

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = layer_idx
                self.runtime = runtime
                self.fused_op = FlashMoEFusedExpertsOp()
                self._resident_gate = None
                self._resident_up = None
                self._resident_down = None
                self.shared_expert = getattr(original_layer, "shared_expert", None)
                self.gate = getattr(original_layer, "gate", None)
                if runtime.has_streaming_backend():
                    self.experts = nn.ModuleList()
                else:
                    self.experts = getattr(original_layer, "experts", None)
                self.top_k = runtime.config.top_k

            def _materialize_resident_weights(self, device, dtype):
                if self._resident_gate is not None:
                    return
                gate_ws = []
                up_ws = []
                down_ws = []
                for expert in self.experts:
                    gate_ws.append(expert.gate_proj.weight.detach().to(device=device, dtype=dtype).contiguous())
                    up_ws.append(expert.up_proj.weight.detach().to(device=device, dtype=dtype).contiguous())
                    down_ws.append(expert.down_proj.weight.detach().to(device=device, dtype=dtype).contiguous())
                self._resident_gate = torch.stack(gate_ws, dim=0)
                self._resident_up = torch.stack(up_ws, dim=0)
                self._resident_down = torch.stack(down_ws, dim=0)

            def _streamed_weights(self, hidden_states, route_ids):
                active_ids = route_ids.detach().to("cpu").unique(sorted=True).tolist()
                expert_ids, gate_w, up_w, down_w = self.runtime.load_layer_experts(
                    self.layer_idx,
                    active_ids,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                local_route_ids = route_ids.detach().to("cpu").clone()
                for local_idx, expert_id in enumerate(expert_ids):
                    local_route_ids[local_route_ids == expert_id] = local_idx
                return gate_w, up_w, down_w, local_route_ids.to(device=hidden_states.device, dtype=torch.int32)

            def forward(self, hidden_states, *args, **kwargs):
                logits = self.gate(hidden_states)
                probs = torch.softmax(logits, dim=-1)
                route_weights, route_ids = torch.topk(probs, k=self.top_k, dim=-1)
                if self.runtime.trace_writer is not None:
                    self.runtime.trace_writer.append_routes(self.layer_idx, route_ids)

                if self.runtime.has_streaming_backend():
                    gate_w, up_w, down_w, local_route_ids = self._streamed_weights(hidden_states, route_ids)
                else:
                    self._materialize_resident_weights(hidden_states.device, hidden_states.dtype)
                    gate_w = self._resident_gate
                    up_w = self._resident_up
                    down_w = self._resident_down
                    local_route_ids = route_ids.to(dtype=torch.int32)

                routed = self.fused_op(
                    hidden_states,
                    gate_w,
                    up_w,
                    down_w,
                    local_route_ids,
                    route_weights,
                )

                if self.shared_expert is not None:
                    routed = routed + self.shared_expert(hidden_states)
                return routed

        return _Impl()


def import_base_model_class(import_path: str):
    module_name, _, class_name = import_path.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"Invalid base model class path: {import_path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def replace_routed_moe_layers(module, runtime, prefix: str = "") -> int:
    torch, nn, _ = _require_torch()
    replaced = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if hasattr(child, "experts") and hasattr(child, "gate"):
            layer_idx = replaced
            setattr(module, name, FlashMoERoutedMoELayer(child, layer_idx=layer_idx, runtime=runtime))
            replaced += 1
            continue
        replaced += replace_routed_moe_layers(child, runtime, child_prefix)
    return replaced
