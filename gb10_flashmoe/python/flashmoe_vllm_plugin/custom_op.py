from __future__ import annotations

from typing import Any

from .ops import flashmoe_moe_forward


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for FlashMoE custom ops") from exc
    return torch, nn


def _get_custom_op_base():
    try:
        from vllm.model_executor.custom_op import CustomOp
        return CustomOp
    except Exception:
        _, nn = _require_torch()
        return nn.Module


class FlashMoEFusedExpertsOp(_get_custom_op_base()):
    def __init__(self):
        _, nn = _require_torch()
        self._standalone_mode = False
        try:
            super().__init__()
        except AssertionError as exc:
            if "Current vLLM config is not set" not in str(exc):
                raise
            nn.Module.__init__(self)
            self._standalone_mode = True
        self._compiled_native = None

    def forward_cuda(self, hidden_states, gate_proj_weights, up_proj_weights,
                     down_proj_weights, route_ids, route_weights):
        return flashmoe_moe_forward(
            hidden_states,
            gate_proj_weights,
            up_proj_weights,
            down_proj_weights,
            route_ids,
            route_weights,
        )

    def forward_native(self, hidden_states, gate_proj_weights, up_proj_weights,
                       down_proj_weights, route_ids, route_weights):
        torch, _ = _require_torch()
        num_tokens, hidden_dim = hidden_states.shape
        top_k = route_ids.shape[1]
        output = hidden_states.new_zeros((num_tokens, hidden_dim))
        for token_idx in range(num_tokens):
            x = hidden_states[token_idx:token_idx + 1]
            for k in range(top_k):
                expert_idx = int(route_ids[token_idx, k].item())
                prob = route_weights[token_idx, k]
                gate = torch.matmul(x, gate_proj_weights[expert_idx].transpose(0, 1))
                up = torch.matmul(x, up_proj_weights[expert_idx].transpose(0, 1))
                act = torch.nn.functional.silu(gate) * up
                down = torch.matmul(act, down_proj_weights[expert_idx].transpose(0, 1))
                output[token_idx:token_idx + 1] += prob * down
        return output

    def _run_impl(self, hidden_states, gate_proj_weights, up_proj_weights,
                  down_proj_weights, route_ids, route_weights):
        if hidden_states.is_cuda:
            return self.forward_cuda(
                hidden_states,
                gate_proj_weights,
                up_proj_weights,
                down_proj_weights,
                route_ids,
                route_weights,
            )
        return self.forward_native(
            hidden_states,
            gate_proj_weights,
            up_proj_weights,
            down_proj_weights,
            route_ids,
            route_weights,
        )

    def forward(self, hidden_states, gate_proj_weights, up_proj_weights,
                down_proj_weights, route_ids, route_weights):
        if not self._standalone_mode and hasattr(self, "_forward_method"):
            return self._forward_method(
                hidden_states,
                gate_proj_weights,
                up_proj_weights,
                down_proj_weights,
                route_ids,
                route_weights,
            )
        return self._run_impl(
            hidden_states,
            gate_proj_weights,
            up_proj_weights,
            down_proj_weights,
            route_ids,
            route_weights,
        )
