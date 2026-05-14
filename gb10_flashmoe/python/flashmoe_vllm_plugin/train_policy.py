from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Iterable


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required for policy training") from exc
    return torch, nn


def _load_rows(path: Path) -> tuple[list[list[float]], list[float]]:
    feats: list[list[float]] = []
    labels: list[float] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feats.append([
                float(row["recency"]),
                float(row["frequency"]),
                float(row["reuse_distance"]),
                float(row["size_ratio"]),
                float(row["layer_pressure"]),
                float(row.get("is_prefetched", 0.0)),
            ])
            labels.append(float(row["label"]))
    return feats, labels


def train(input_csv: Path, output_json: Path, epochs: int, hidden_dim: int, lr: float) -> None:
    torch, nn = _require_torch()
    features, labels = _load_rows(input_csv)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    layers = []
    for module in model:
        if isinstance(module, nn.Linear):
            layers.append({
                "weight": module.weight.detach().cpu().tolist(),
                "bias": module.bias.detach().cpu().tolist(),
                "activation": "relu",
            })
    if layers:
        layers[-1]["activation"] = "identity"
    output_json.write_text(json.dumps({"layers": layers}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlashMoE cache-policy FFN.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.input, args.output, args.epochs, args.hidden_dim, args.lr)


if __name__ == "__main__":
    main()
