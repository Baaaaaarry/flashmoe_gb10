from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .config import FlashMoEConfig
from .layer_replacement import import_base_model_class, replace_routed_moe_layers
from .runtime import FlashMoERuntime


try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - import-time fallback for non-runtime environments
    torch = None

    class nn:  # type: ignore[no-redef]
        class Module:  # noqa: D401 - tiny fallback shim
            """Fallback base when torch is unavailable at import time."""

            def __init__(self, *args, **kwargs):
                raise RuntimeError("torch is required to instantiate FlashMoEQwenForCausalLM")


class FlashMoEQwenForCausalLM(nn.Module):
    """vLLM-visible wrapper that preserves generation interfaces.

    vLLM inspects the registered model class itself to determine whether the
    architecture supports `runner=generate`. The previous `__new__`-based lazy
    wrapper hid the real generation methods from class inspection, so vLLM
    rejected the model before startup. This wrapper is a normal `nn.Module`
    and forwards the generation methods to the underlying base model.
    """

    def __init__(self, vllm_config=None, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        config_path = ""
        if vllm_config is not None:
            config_path = getattr(vllm_config.model_config.hf_config, "flashmoe_config", "")
        self.flashmoe = FlashMoERuntime(
            FlashMoEConfig.from_file(config_path) if config_path else FlashMoEConfig()
        )
        base_model_class = self.flashmoe.config.base_model_class
        hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
        if (
            hf_config is not None
            and hf_config.__class__.__name__.startswith("Qwen3_5")
            and base_model_class == "vllm.model_executor.models.qwen3_moe:Qwen3MoeForCausalLM"
        ):
            base_model_class = "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM"
            self.flashmoe.config.base_model_class = base_model_class

        if not base_model_class:
            raise RuntimeError(
                "flashmoe base_model_class must point to the original vLLM Qwen MoE model class"
            )
        base_cls = import_base_model_class(base_model_class)
        self.base_model = base_cls(vllm_config=vllm_config, prefix=prefix)
        self.replaced_layers = replace_routed_moe_layers(self.base_model, self.flashmoe)

    def embed_input_ids(self, input_ids):
        return self.base_model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, **kwargs):
        return self.base_model(input_ids=input_ids, positions=positions, **kwargs)

    def load_weights(self, weights):
        if hasattr(self.base_model, "load_weights"):
            if self.flashmoe.has_streaming_backend():
                return self.base_model.load_weights(self._filtered_streaming_weights(weights))
            return self.base_model.load_weights(weights)
        return None

    def _filtered_streaming_weights(self, weights: Iterable):
        for item in weights:
            if not item:
                continue
            name = item[0]
            if isinstance(name, str) and ".mlp.experts." in name:
                continue
            yield item

    def compute_logits(self, hidden_states, *args, **kwargs):
        return self.base_model.compute_logits(hidden_states=hidden_states, *args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.base_model.sample(*args, **kwargs)

    def make_empty_intermediate_tensors(self, *args, **kwargs):
        if hasattr(self.base_model, "make_empty_intermediate_tensors"):
            return self.base_model.make_empty_intermediate_tensors(*args, **kwargs)
        raise AttributeError("base model does not implement make_empty_intermediate_tensors")

    def __getattr__(self, name: str) -> Any:
        if name in {"base_model", "flashmoe", "vllm_config", "prefix", "replaced_layers"}:
            return super().__getattribute__(name)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            base_model = super().__getattribute__("base_model")
            return getattr(base_model, name)
