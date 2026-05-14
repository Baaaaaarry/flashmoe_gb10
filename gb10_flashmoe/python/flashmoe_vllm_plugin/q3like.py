from __future__ import annotations

import torch


def pack_q3_tensor(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().view(tensor.shape[0], -1)
    scales = flat.abs().amax(dim=1).clamp_min(1e-8) / 3.0
    quant = torch.round(flat / scales.unsqueeze(1)).clamp(-4, 3).to(torch.int16) + 4
    packed = _pack_uint3(quant.to(torch.uint8))
    return {
        "qweight": packed,
        "scale": scales.to(torch.float16).contiguous(),
        "shape": torch.tensor(list(tensor.shape), dtype=torch.int32),
    }


def unpack_q3_tensor(bundle: dict[str, torch.Tensor]) -> torch.Tensor:
    shape = [int(v) for v in bundle["shape"].tolist()]
    rows = shape[0]
    cols = 1
    for dim in shape[1:]:
        cols *= dim
    quant = _unpack_uint3(bundle["qweight"], rows * cols).to(torch.int16) - 4
    quant = quant.view(rows, cols).to(torch.float32)
    scale = bundle["scale"].to(torch.float32).unsqueeze(1)
    out = quant * scale
    return out.view(*shape)


def _pack_uint3(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.int32)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int32)], dim=0)
    groups = flat.view(-1, 8)
    shifts = torch.tensor([0, 3, 6, 9, 12, 15, 18, 21], dtype=torch.int32)
    packed24 = torch.sum(groups << shifts, dim=1, dtype=torch.int32)
    out = torch.empty((packed24.numel(), 3), dtype=torch.uint8)
    out[:, 0] = (packed24 & 0xFF).to(torch.uint8)
    out[:, 1] = ((packed24 >> 8) & 0xFF).to(torch.uint8)
    out[:, 2] = ((packed24 >> 16) & 0xFF).to(torch.uint8)
    return out.reshape(-1).contiguous()


def _unpack_uint3(packed: torch.Tensor, count: int) -> torch.Tensor:
    bytes_view = packed.reshape(-1, 3).to(torch.int32)
    packed24 = bytes_view[:, 0] | (bytes_view[:, 1] << 8) | (bytes_view[:, 2] << 16)
    shifts = torch.tensor([0, 3, 6, 9, 12, 15, 18, 21], dtype=torch.int32)
    values = ((packed24.unsqueeze(1) >> shifts) & 0x7).reshape(-1)
    return values[:count].to(torch.uint8).contiguous()
