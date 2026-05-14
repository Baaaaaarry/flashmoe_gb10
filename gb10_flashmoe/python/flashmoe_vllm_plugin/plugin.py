from __future__ import annotations


def register() -> None:
    try:
        from vllm import ModelRegistry
    except ModuleNotFoundError:
        return

    ModelRegistry.register_model(
        "FlashMoEQwenForCausalLM",
        "flashmoe_vllm_plugin.vllm_adapter:FlashMoEQwenForCausalLM",
    )
