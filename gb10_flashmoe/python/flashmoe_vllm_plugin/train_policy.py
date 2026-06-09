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


def _load_rows(path: Path, feature_names: list[str]) -> tuple[list[list[float]], list[float]]:
    feats: list[list[float]] = []
    labels: list[float] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feats.append([float(row.get(name, 0.0)) for name in feature_names])
            labels.append(float(row["label"]))
    return feats, labels


def train(input_csv: Path, output_json: Path, epochs: int, hidden_dim: int, lr: float, feature_names: list[str]) -> None:
    torch, nn = _require_torch()
    features, labels = _load_rows(input_csv, feature_names)
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
    output_json.write_text(json.dumps({"feature_names": feature_names, "layers": layers}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlashMoE cache-policy FFN.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--features", default="recency,frequency,layer_pressure,slot_age,recency_ratio,reuse_density,log_recency,log_frequency,is_prefetched")
    args = parser.parse_args()
    feature_names = [piece.strip() for piece in args.features.split(",") if piece.strip()]
    train(args.input, args.output, args.epochs, args.hidden_dim, args.lr, feature_names)


if __name__ == "__main__":
    main()
