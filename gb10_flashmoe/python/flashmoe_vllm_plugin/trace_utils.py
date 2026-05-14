from __future__ import annotations

import json
import random
from pathlib import Path


class RoutingTraceWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_routes(self, layer_id: int, route_ids) -> None:
        rows = route_ids.detach().to("cpu").tolist()
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                experts = ",".join(str(int(expert_id)) for expert_id in row)
                handle.write(f"{layer_id}:{experts}\n")


def load_manifest_layer_sizes(manifest_path: Path) -> dict[int, int]:
    raw = json.loads(manifest_path.read_text())
    counts: dict[int, int] = {}
    for item in raw.get("entries", []):
        layer_id = int(item["layer_id"])
        expert_id = int(item["expert_id"])
        counts[layer_id] = max(counts.get(layer_id, -1), expert_id)
    return {layer_id: max_expert + 1 for layer_id, max_expert in counts.items()}


def generate_synthetic_trace(
    manifest_path: Path,
    output_path: Path,
    steps: int,
    top_k: int,
    locality: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    layer_sizes = load_manifest_layer_sizes(manifest_path)
    if not layer_sizes:
        raise SystemExit(f"No layer/expert entries found in manifest: {manifest_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_choice: dict[int, list[int]] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for _ in range(steps):
            for layer_id in sorted(layer_sizes):
                num_experts = layer_sizes[layer_id]
                if layer_id in last_choice and rng.random() < locality:
                    chosen = list(last_choice[layer_id])
                else:
                    chosen = rng.sample(range(num_experts), k=min(top_k, num_experts))
                    while len(chosen) < top_k:
                        chosen.append(chosen[-1])
                last_choice[layer_id] = chosen
                handle.write(f"{layer_id}:{','.join(str(v) for v in chosen)}\n")
