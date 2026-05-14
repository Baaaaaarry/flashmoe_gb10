from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class Access:
    step: int
    layer: int
    expert: int


@dataclass(slots=True)
class Meta:
    last_touch: int
    access_count: int
    layer_pressure: int


def load_trace(path: Path) -> list[Access]:
    accesses: list[Access] = []
    step = 0
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = [int(piece) for piece in raw.split()]
        _, layer, *experts = parts
        for expert in experts:
            accesses.append(Access(step=step, layer=layer, expert=expert))
            step += 1
    return accesses


def build_dataset(trace_path: Path, output_csv: Path, cache_entries: int) -> None:
    accesses = load_trace(trace_path)
    future_positions: dict[tuple[int, int], list[int]] = {}
    for access in accesses:
        future_positions.setdefault((access.layer, access.expert), []).append(access.step)

    resident: dict[tuple[int, int], Meta] = {}
    layer_counts: dict[int, int] = {}

    fieldnames = [
        "recency",
        "frequency",
        "reuse_distance",
        "size_ratio",
        "layer_pressure",
        "is_prefetched",
        "label",
    ]
    rows: list[dict[str, float]] = []

    for access in accesses:
        key = (access.layer, access.expert)
        positions = future_positions[key]
        while positions and positions[0] <= access.step:
            positions.pop(0)

        if key in resident:
            meta = resident[key]
            meta.last_touch = access.step
            meta.access_count += 1
            continue

        if len(resident) >= cache_entries:
            future_distance: dict[tuple[int, int], float] = {}
            for cand_key, meta in resident.items():
                cand_positions = future_positions.get(cand_key, [])
                next_use = cand_positions[0] if cand_positions else float("inf")
                future_distance[cand_key] = next_use - access.step if next_use != float("inf") else 1e9

            victim = max(future_distance, key=future_distance.get)
            max_access = max(item.access_count for item in resident.values())
            for cand_key, meta in resident.items():
                recency = float(access.step - meta.last_touch)
                frequency = float(meta.access_count) / float(max_access or 1)
                reuse_distance = float(future_distance[cand_key])
                size_ratio = 1.0
                layer_pressure = float(layer_counts.get(cand_key[0], 0)) / float(max(len(resident), 1))
                rows.append({
                    "recency": recency,
                    "frequency": frequency,
                    "reuse_distance": reuse_distance,
                    "size_ratio": size_ratio,
                    "layer_pressure": layer_pressure,
                    "is_prefetched": 0.0,
                    "label": 1.0 if cand_key == victim else 0.0,
                })

            layer_counts[victim[0]] -= 1
            del resident[victim]

        resident[key] = Meta(last_touch=access.step, access_count=1, layer_pressure=layer_counts.get(access.layer, 0) + 1)
        layer_counts[access.layer] = layer_counts.get(access.layer, 0) + 1

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Belady-style cache-policy dataset from routing trace.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-entries", type=int, default=16384)
    args = parser.parse_args()
    build_dataset(args.trace, args.output, args.cache_entries)


if __name__ == "__main__":
    main()
