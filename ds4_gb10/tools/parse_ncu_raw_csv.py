#!/usr/bin/env python3
import csv
import json
import math
import re
import sys
from pathlib import Path


METRIC_ALIASES = {
    "dram_util_pct": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "tensor_util_pct": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    ],
    "sm_util_pct": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "dram_read_bytes": ["dram__bytes_read.sum"],
    "dram_write_bytes": ["dram__bytes_write.sum"],
}


def parse_float(text: str):
    try:
        return float(text.replace(",", "").strip())
    except Exception:
        return None


def weighted_avg(vals):
    total_w = 0.0
    total = 0.0
    for value, weight in vals:
        if value is None:
            continue
        w = weight if weight and weight > 0.0 else 1.0
        total += value * w
        total_w += w
    if total_w <= 0.0:
        return None
    return total / total_w


def parse_structured(rows):
    header = rows[0]
    name_idx = None
    value_idx = None
    time_idx = None
    for i, col in enumerate(header):
        c = col.strip().lower()
        if c == "metric name":
            name_idx = i
        elif c == "metric value":
            value_idx = i
        elif c in ("kernel time", "gpu time", "gpu__time_duration.sum"):
            time_idx = i
    if name_idx is None or value_idx is None:
        return None

    out = {
        "dram_util_pct": [],
        "tensor_util_pct": [],
        "sm_util_pct": [],
        "dram_read_bytes": 0.0,
        "dram_write_bytes": 0.0,
        "mac_metric": "",
    }
    for row in rows[1:]:
        if len(row) <= max(name_idx, value_idx):
            continue
        metric = row[name_idx].strip()
        value = parse_float(row[value_idx])
        weight = None
        if time_idx is not None and len(row) > time_idx:
            weight = parse_float(row[time_idx])
        for key, aliases in METRIC_ALIASES.items():
            if metric not in aliases:
                continue
            if key.endswith("_bytes"):
                if value is not None:
                    out[key] += value
            else:
                out[key].append((value, weight))

    dram = weighted_avg(out["dram_util_pct"])
    tensor = weighted_avg(out["tensor_util_pct"])
    sm = weighted_avg(out["sm_util_pct"])
    mac = tensor if tensor is not None else sm
    mac_metric = "tensor_util_pct" if tensor is not None else ("sm_util_pct" if sm is not None else "")
    return {
        "eta_mem_pct": dram,
        "eta_mac_pct": mac,
        "mac_metric": mac_metric,
        "dram_read_bytes": out["dram_read_bytes"],
        "dram_write_bytes": out["dram_write_bytes"],
        "sm_util_pct": sm,
        "tensor_util_pct": tensor,
        "dram_util_pct": dram,
    }


def parse_fallback(text: str):
    result = {
        "eta_mem_pct": None,
        "eta_mac_pct": None,
        "mac_metric": "",
        "dram_read_bytes": 0.0,
        "dram_write_bytes": 0.0,
        "sm_util_pct": None,
        "tensor_util_pct": None,
        "dram_util_pct": None,
    }
    for key, aliases in METRIC_ALIASES.items():
        for metric in aliases:
            m = re.search(rf"{re.escape(metric)}.*?([-+]?[0-9][0-9,]*\.?[0-9]*)", text)
            if not m:
                continue
            value = parse_float(m.group(1))
            if key.endswith("_bytes"):
                result[key] = value or 0.0
            else:
                result[key] = value
            break
    if result["tensor_util_pct"] is not None:
        result["eta_mac_pct"] = result["tensor_util_pct"]
        result["mac_metric"] = "tensor_util_pct"
    elif result["sm_util_pct"] is not None:
        result["eta_mac_pct"] = result["sm_util_pct"]
        result["mac_metric"] = "sm_util_pct"
    result["eta_mem_pct"] = result["dram_util_pct"]
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} NCU_RAW.csv")
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = list(csv.reader(text.splitlines()))
    parsed = parse_structured(rows) if rows else None
    if parsed is None:
        parsed = parse_fallback(text)
    print(json.dumps(parsed, sort_keys=True))


if __name__ == "__main__":
    main()
