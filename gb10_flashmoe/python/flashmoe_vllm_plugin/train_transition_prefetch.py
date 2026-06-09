from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def canonicalize(experts: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(int(v) for v in experts)))


def load_trace(path: Path) -> dict[int, list[tuple[int, ...]]]:
    per_layer: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for raw in path.read_text().splitlines():
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        toks = row.split()
        if len(toks) < 3:
            continue
        layer = int(toks[1])
        experts = [int(tok) for tok in toks[2:]]
        key = canonicalize(experts)
        if key:
            per_layer[layer].append(key)
    return per_layer


def train_transition_rules(
    per_layer: dict[int, list[tuple[int, ...]]],
    top_m: int,
    min_support: int,
) -> list[str]:
    lines: list[str] = [
        "# layer current_count current_experts... predicted_count predicted_experts..."
    ]
    for layer in sorted(per_layer):
        seq = per_layer[layer]
        next_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for i in range(len(seq) - 1):
            cur = seq[i]
            nxt = seq[i + 1]
            if cur == nxt:
                # still count; temporal stability is useful
                pass
            for expert in nxt:
                next_counts[cur][expert] += 1
        for cur in sorted(next_counts):
            counts = next_counts[cur]
            support = sum(counts.values())
            if support < min_support:
                continue
            ranked = [expert for expert, _ in counts.most_common(top_m)]
            toks = [str(layer), str(len(cur)), *(str(v) for v in cur), str(len(ranked)), *(str(v) for v in ranked)]
            lines.append(" ".join(toks))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a transition-based FlashMoE prefetch predictor from cache trace.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-m", type=int, default=8, help="Max predicted experts stored per current set.")
    parser.add_argument("--min-support", type=int, default=2, help="Minimum transition support to keep a rule.")
    args = parser.parse_args()

    per_layer = load_trace(args.trace)
    lines = train_transition_rules(per_layer, top_m=args.top_m, min_support=args.min_support)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output} rules={max(0, len(lines)-1)} layers={len(per_layer)}")


if __name__ == "__main__":
    main()
