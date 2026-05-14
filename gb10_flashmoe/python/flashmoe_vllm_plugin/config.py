from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TechniqueFlags:
    enable_predictor: bool = False
    enable_cache_policy: bool = True
    enable_sliding_window: bool = False
    enable_row_column_bundling: bool = False
    enable_dram_sparse_cache: bool = True
    enable_expert_streaming: bool = True
    enable_layer_major_scheduling: bool = True
    enable_fused_moe_kernel: bool = True
    enable_parallel_pread: bool = True
    enable_expert_prefetch: bool = True
    enable_chunked_pread: bool = True
    enable_shared_expert_overlap: bool = True


@dataclass(slots=True)
class FlashMoEConfig:
    model_name: str = "Qwen3.5-397B-A17B"
    model_path: str = ""
    base_model_class: str = ""
    expert_path: str = ""
    expert_manifest_path: str = ""
    expert_format: str = "dense"
    dense_path: str = ""
    predictor_path: str = ""
    cache_policy_path: str = ""
    routing_trace_path: str = ""
    expert_cache_gb: float = 96.0
    expert_page_size_kb: int = 16
    pread_workers: int = 8
    pread_chunks: int = 4
    prefetch_window: int = 2
    max_hot_experts: int = 16384
    top_k: int = 4
    shared_experts: int = 1
    max_model_len: int = 32768
    gpu_streams: int = 3
    cuda_graphs: bool = True
    use_vllm_paged_attention: bool = True
    vllm_disable_sliding_window: bool = False
    vllm_quantization: str = "model"
    attention_backend: str = "flash-attn"
    flags: TechniqueFlags = field(default_factory=TechniqueFlags)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> "FlashMoEConfig":
        raw = json.loads(Path(path).read_text())
        flags = TechniqueFlags(**raw.pop("flags", {}))
        return cls(flags=flags, **raw)

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
