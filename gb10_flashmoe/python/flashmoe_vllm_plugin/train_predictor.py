from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required for predictor training") from exc
    return torch, nn


def load_routing_bin(path: Path, hidden_dim: int = 4096) -> tuple[list[int], list[list[float]], list[list[int]]]:
    blob = path.read_bytes()
    layers: list[int] = []
    states: list[list[float]] = []
    experts: list[list[int]] = []
    cursor = 0
    while cursor + 8 <= len(blob):
        layer = struct.unpack_from("<i", blob, cursor)[0]
        cursor += 4
        k = struct.unpack_from("<i", blob, cursor)[0]
        cursor += 4
        fmt = f"<{hidden_dim}f"
        state = list(struct.unpack_from(fmt, blob, cursor))
        cursor += 4 * hidden_dim
        exp_fmt = f"<{k}i"
        routed = list(struct.unpack_from(exp_fmt, blob, cursor))
        cursor += 4 * k
        layers.append(layer)
        states.append(state)
        experts.append(routed)
    return layers, states, experts


def train(input_bin: Path, output_json: Path, hidden_dim: int, num_experts: int, epochs: int, lr: float) -> None:
    torch, nn = _require_torch()
    layers, states, experts = load_routing_bin(input_bin, hidden_dim=hidden_dim)
    x = torch.tensor(states, dtype=torch.float32)
    l = torch.tensor(layers, dtype=torch.long)
    y = torch.zeros((len(experts), num_experts), dtype=torch.float32)
    for row_idx, routed in enumerate(experts):
        for expert_idx in routed:
            y[row_idx, expert_idx] = 1.0

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_emb = nn.Embedding(128, 32)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + 32, 512),
                nn.ReLU(),
                nn.Linear(512, num_experts),
            )

        def forward(self, states, layers):
            return self.net(torch.cat([states, self.layer_emb(layers)], dim=-1))

    model = Predictor()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        pred = model(x, l)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    exported = {
        "layer_emb.weight": model.layer_emb.weight.detach().cpu().tolist(),
        "net.0.weight": model.net[0].weight.detach().cpu().tolist(),
        "net.0.bias": model.net[0].bias.detach().cpu().tolist(),
        "net.2.weight": model.net[2].weight.detach().cpu().tolist(),
        "net.2.bias": model.net[2].bias.detach().cpu().tolist(),
    }
    output_json.write_text(json.dumps(exported, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlashMoE routing predictor.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.input, args.output, args.hidden_dim, args.num_experts, args.epochs, args.lr)


if __name__ == "__main__":
    main()
