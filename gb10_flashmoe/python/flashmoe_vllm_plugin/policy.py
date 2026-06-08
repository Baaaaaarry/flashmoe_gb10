from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class CachePolicyFeatures:
    recency: float
    frequency: float
    reuse_distance: float
    size_ratio: float
    layer_pressure: float
    is_prefetched: float

    def as_list(self) -> list[float]:
        return [
            self.recency,
            self.frequency,
            self.reuse_distance,
            self.size_ratio,
            self.layer_pressure,
            self.is_prefetched,
        ]


class CachePolicyModel:
    def __init__(self, layers: list[dict[str, list[list[float]] | list[float]]], feature_names: list[str] | None = None):
        self.layers = layers
        self.feature_names = feature_names or [
            "recency",
            "frequency",
            "size_ratio",
            "layer_pressure",
            "is_prefetched",
        ]

    @classmethod
    def maybe_load(cls, path: str) -> "CachePolicyModel | None":
        if not path:
            return None
        target = Path(path)
        if not target.exists():
            return None
        raw = json.loads(target.read_text())
        return cls(raw["layers"], raw.get("feature_names"))

    def score(self, features: CachePolicyFeatures) -> float:
        values = {
            "recency": features.recency,
            "frequency": features.frequency,
            "reuse_distance": features.reuse_distance,
            "size_ratio": features.size_ratio,
            "layer_pressure": features.layer_pressure,
            "is_prefetched": features.is_prefetched,
        }
        x = [values.get(name, 0.0) for name in self.feature_names]
        for layer in self.layers:
            x = _linear(layer["weight"], x, layer["bias"])
            if layer.get("activation", "relu") == "relu":
                x = [max(0.0, value) for value in x]
            elif layer["activation"] == "tanh":
                x = [math.tanh(value) for value in x]
        return float(x[0])


def _linear(weight: list[list[float]], x: list[float], bias: list[float]) -> list[float]:
    out: list[float] = []
    for row, row_bias in zip(weight, bias, strict=True):
        acc = row_bias
        for w, value in zip(row, x, strict=True):
            acc += w * value
        out.append(acc)
    return out
