from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import FlashMoEConfig
from .policy import CachePolicyModel
from .streaming import ExpertManifest, StreamingExpertCache
from .trace_utils import RoutingTraceWriter


@dataclass(slots=True)
class FeatureStatus:
    name: str
    enabled: bool
    note: str


class FlashMoERuntime:
    def __init__(self, config: FlashMoEConfig):
        self.config = config
        self.cache_policy = CachePolicyModel.maybe_load(config.cache_policy_path)
        self.trace_writer = RoutingTraceWriter(config.routing_trace_path) if config.routing_trace_path else None
        self.manifest = None
        self.streaming_cache = None
        self._step = 0
        manifest_path = config.expert_manifest_path
        if config.flags.enable_expert_streaming and manifest_path and Path(manifest_path).exists():
            self.manifest = ExpertManifest.from_file(manifest_path)
            self.streaming_cache = StreamingExpertCache(config, self.manifest, self.cache_policy)

    def feature_matrix(self) -> list[FeatureStatus]:
        flags = self.config.flags
        return [
            FeatureStatus("predictor", flags.enable_predictor,
                          "Routing predictor for speculative expert prefetch."),
            FeatureStatus("cache_policy", flags.enable_cache_policy,
                          "FFN or heuristic eviction score for DRAM expert cache."),
            FeatureStatus("sliding_window", flags.enable_sliding_window,
                          "vLLM attention-side switch, not used by the Apple repos."),
            FeatureStatus("row_column_bundling", flags.enable_row_column_bundling,
                          "Pack expert rows/columns into larger sequential bundles."),
            FeatureStatus("dram_sparse_cache", flags.enable_dram_sparse_cache,
                          "Explicit hot expert cache in unified DRAM."),
            FeatureStatus("expert_level_stream", flags.enable_expert_streaming,
                          "SSD to RAM on-demand expert streaming."),
            FeatureStatus("layer_major_scheduling", flags.enable_layer_major_scheduling,
                          "Layer-major packed expert files and decode traversal."),
            FeatureStatus("fused_moe_kernel", flags.enable_fused_moe_kernel,
                          "Shared+routed expert fused kernel path."),
            FeatureStatus("parallel_pread", flags.enable_parallel_pread,
                          "Persistent worker pool and parallel pread fanout."),
            FeatureStatus("chunked_pread", flags.enable_chunked_pread,
                          "Page-aligned split per expert read."),
            FeatureStatus("shared_expert_overlap", flags.enable_shared_expert_overlap,
                          "Overlap shared expert work with routed-expert I/O."),
        ]

    def to_vllm_serve_args(self) -> list[str]:
        args = [
            "--model", self.config.model_path,
            "--max-model-len", str(self.config.max_model_len),
            "--hf-overrides",
            (
                "{"
                "\"architectures\":[\"FlashMoEQwenForCausalLM\"],"
                f"\"flashmoe_config\":\"{self.config_path_hint()}\","
                f"\"flashmoe_top_k\":{self.config.top_k}"
                "}"
            ),
        ]
        quantization = (self.config.vllm_quantization or "").strip().lower()
        if quantization and quantization not in {"model", "auto", "inherit"}:
            args[2:2] = ["--quantization", self.config.vllm_quantization]
        if self.config.vllm_disable_sliding_window:
            args.append("--disable-sliding-window")
        return args

    def config_path_hint(self) -> str:
        return str(Path(self.config.model_path).parent / "flashmoe_config.json")

    def describe(self) -> str:
        lines = [
            f"model={self.config.model_name}",
            f"expert_cache_gb={self.config.expert_cache_gb}",
            f"expert_format={self.config.expert_format}",
            f"expert_manifest_path={self.config.expert_manifest_path or '<unset>'}",
            f"pread_workers={self.config.pread_workers}",
            f"pread_chunks={self.config.pread_chunks}",
            f"prefetch_window={self.config.prefetch_window}",
        ]
        for feature in self.feature_matrix():
            lines.append(f"{feature.name}={'on' if feature.enabled else 'off'} # {feature.note}")
        return "\n".join(lines)

    def has_streaming_backend(self) -> bool:
        return self.streaming_cache is not None

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def load_layer_experts(self, layer_id: int, expert_ids: Iterable[int], *, device, dtype):
        if self.streaming_cache is None:
            raise RuntimeError("streaming backend is not initialized")

        torch = __import__("torch")

        active_ids = sorted({int(expert_id) for expert_id in expert_ids})
        step = self.next_step()
        gate_ws = []
        up_ws = []
        down_ws = []
        for expert_id in active_ids:
            resident = self.streaming_cache.ensure(layer_id, expert_id, step=step, prefetched=False)
            tensors = self.streaming_cache.materialize_tensors(resident)
            gate_ws.append(tensors["gate_proj.weight"].to(device=device, dtype=dtype).contiguous())
            up_ws.append(tensors["up_proj.weight"].to(device=device, dtype=dtype).contiguous())
            down_ws.append(tensors["down_proj.weight"].to(device=device, dtype=dtype).contiguous())

        return (
            active_ids,
            torch.stack(gate_ws, dim=0),
            torch.stack(up_ws, dim=0),
            torch.stack(down_ws, dim=0),
        )
