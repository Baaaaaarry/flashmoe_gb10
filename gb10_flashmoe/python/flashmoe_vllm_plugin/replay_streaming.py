from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

try:
    from .config import FlashMoEConfig
    from .policy import CachePolicyModel
    from .streaming import ExpertManifest, StreamingExpertCache, format_stats
except ImportError:
    import sys
    from pathlib import Path as _Path

    current_dir = _Path(__file__).resolve().parent
    package_root = current_dir.parent
    sys.path.append(str(package_root))
    sys.path.append(str(current_dir))
    try:
        from flashmoe_vllm_plugin.config import FlashMoEConfig
        from flashmoe_vllm_plugin.policy import CachePolicyModel
        from flashmoe_vllm_plugin.streaming import ExpertManifest, StreamingExpertCache, format_stats
    except ImportError:
        from config import FlashMoEConfig
        from policy import CachePolicyModel
        from streaming import ExpertManifest, StreamingExpertCache, format_stats


def load_trace(path: Path) -> list[tuple[int, list[int]]]:
    requests: list[tuple[int, list[int]]] = []
    for raw in path.read_text().splitlines():
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        left, right = row.split(":")
        layer_id = int(left)
        experts = [int(token) for token in right.split(",") if token.strip()]
        requests.append((layer_id, experts))
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay routing trace against FlashMoE SSD/cache control plane.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    args = parser.parse_args()

    config = FlashMoEConfig.from_file(args.config)
    if not config.expert_manifest_path:
        raise SystemExit("config.expert_manifest_path is required for streaming replay")

    manifest = ExpertManifest.from_file(config.expert_manifest_path)
    policy = CachePolicyModel.maybe_load(config.cache_policy_path)
    cache = StreamingExpertCache(config, manifest, policy)
    requests = load_trace(args.trace)

    window: deque[list[tuple[int, int]]] = deque(maxlen=max(1, config.prefetch_window))
    start = time.perf_counter()
    for step, (layer_id, experts) in enumerate(requests):
        if config.flags.enable_predictor and config.flags.enable_expert_prefetch:
            for predicted in window:
                cache.prefetch(predicted, step)

        current = []
        for expert_id in experts:
            current.append((layer_id, expert_id))
            if (layer_id, expert_id) in manifest:
                cache.ensure(layer_id, expert_id, step=step, prefetched=False)
        window.append(current)
    elapsed = time.perf_counter() - start

    print(f"manifest_entries={len(manifest)}")
    print(f"trace_steps={len(requests)} elapsed_s={elapsed:.3f}")
    print(format_stats(cache.stats, elapsed))


if __name__ == "__main__":
    main()
