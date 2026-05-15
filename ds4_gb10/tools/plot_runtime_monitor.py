#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path


WIDTH = 1600
HEIGHT = 1200
MARGIN_L = 70
MARGIN_R = 20
MARGIN_T = 40
MARGIN_B = 40
PANEL_GAP_X = 28
PANEL_GAP_Y = 40
PANEL_COLS = 2
PANEL_ROWS = 2


def load_rows(path: Path):
    rows = []
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "phase":
                    parsed[key] = value or "idle"
                elif key in ("current", "total", "gpu_util_pct", "gpu_mem_util_pct"):
                    try:
                        parsed[key] = int(value)
                    except Exception:
                        parsed[key] = -1
                else:
                    try:
                        parsed[key] = float(value)
                    except Exception:
                        parsed[key] = float("nan")
            rows.append(parsed)
    if not rows:
        raise SystemExit(f"no samples found in {path}")
    return rows


def finite(v):
    return v is not None and not math.isnan(v) and not math.isinf(v) and v >= 0.0


def panel_box(index):
    panel_w = (WIDTH - MARGIN_L - MARGIN_R - PANEL_GAP_X) / PANEL_COLS
    panel_h = (HEIGHT - MARGIN_T - MARGIN_B - PANEL_GAP_Y) / PANEL_ROWS
    col = index % PANEL_COLS
    row = index // PANEL_COLS
    x = MARGIN_L + col * (panel_w + PANEL_GAP_X)
    y = MARGIN_T + row * (panel_h + PANEL_GAP_Y)
    return x, y, panel_w, panel_h


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def scale_points(xs, ys, x0, y0, w, h, ymin=None, ymax=None):
    good = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if not good:
        return ""
    x_min = min(x for x, _ in good)
    x_max = max(x for x, _ in good)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if ymin is None:
        ymin = min(y for _, y in good)
    if ymax is None:
        ymax = max(y for _, y in good)
    if ymax <= ymin:
        ymax = ymin + 1.0
    pts = []
    for x, y in good:
        sx = x0 + (x - x_min) / (x_max - x_min) * w
        sy = y0 + h - (y - ymin) / (ymax - ymin) * h
        pts.append(f"{sx:.2f},{sy:.2f}")
    return " ".join(pts), x_min, x_max, ymin, ymax


def draw_axes(parts, x0, y0, w, h, title, ymin, ymax, x_max):
    parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="white" stroke="#d0d7de"/>')
    parts.append(f'<text x="{x0:.1f}" y="{y0 - 12:.1f}" font-size="18" font-family="sans-serif" fill="#111">{svg_escape(title)}</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 + h - frac * h
        val = ymin + frac * (ymax - ymin)
        parts.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + w:.1f}" y2="{yy:.1f}" stroke="#eef2f6"/>')
        parts.append(f'<text x="{x0 - 8:.1f}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#555">{val:.1f}</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx = x0 + frac * w
        val = frac * x_max
        parts.append(f'<line x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" y2="{y0 + h:.1f}" stroke="#f3f5f7"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + h + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">{val:.0f}s</text>')


def draw_series(parts, xs, rows, specs, panel_idx, title):
    x0, y0, w, h = panel_box(panel_idx)
    all_vals = []
    for key, _, _ in specs:
        for row in rows:
            v = row.get(key, float("nan"))
            if finite(v):
                all_vals.append(v)
    if not all_vals:
        ymin, ymax = 0.0, 1.0
    else:
        ymin, ymax = min(all_vals), max(all_vals)
        if ymax <= ymin:
            ymax = ymin + 1.0
    x_max = max(xs) if xs else 1.0
    draw_axes(parts, x0, y0, w, h, title, ymin, ymax, x_max)
    legend_y = y0 - 28
    legend_x = x0 + 180
    for idx, (key, label, color) in enumerate(specs):
        ys = [row.get(key, float("nan")) for row in rows]
        scaled = scale_points(xs, ys, x0, y0, w, h, ymin, ymax)
        if not scaled:
            continue
        pts, _, _, _, _ = scaled
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        lx = legend_x + idx * 150
        parts.append(f'<line x1="{lx:.1f}" y1="{legend_y:.1f}" x2="{lx + 18:.1f}" y2="{legend_y:.1f}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 24:.1f}" y="{legend_y + 4:.1f}" font-size="12" font-family="sans-serif" fill="#333">{svg_escape(label)}</text>')


def draw_phase_marks(parts, rows, xs):
    x0, y0, w, h = panel_box(0)
    last_phase = None
    x_max = max(xs) if xs else 1.0
    if x_max <= 0.0:
        x_max = 1.0
    for row, x in zip(rows, xs):
        phase = row.get("phase", "idle")
        if phase == last_phase:
            continue
        last_phase = phase
        xx = x0 + (x / x_max) * w
        parts.append(f'<line x1="{xx:.1f}" y1="{MARGIN_T - 4:.1f}" x2="{xx:.1f}" y2="{HEIGHT - MARGIN_B + 4:.1f}" stroke="#f0c674" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{xx + 4:.1f}" y="{MARGIN_T - 10:.1f}" font-size="11" font-family="sans-serif" fill="#8a6d3b">{svg_escape(phase)}</text>')


def render(rows, out_path: Path):
    xs = [row["time_s"] for row in rows]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fafbfc"/>',
        '<text x="24" y="28" font-size="24" font-family="sans-serif" fill="#111">ds4 runtime monitor</text>',
    ]
    draw_phase_marks(parts, rows, xs)
    draw_series(parts, xs, rows, [
        ("rss_gib", "RSS", "#1f77b4"),
        ("hwm_gib", "HWM", "#ff7f0e"),
        ("cuda_model_gib", "CUDA model cache", "#2ca02c"),
        ("gpu_mem_used_gib", "GPU mem used", "#d62728"),
    ], 0, "Memory (GiB)")
    draw_series(parts, xs, rows, [
        ("io_read_mibs", "Read MiB/s", "#9467bd"),
        ("io_write_mibs", "Write MiB/s", "#8c564b"),
    ], 1, "Process I/O bandwidth")
    draw_series(parts, xs, rows, [
        ("gpu_util_pct", "GPU util %", "#e377c2"),
        ("gpu_mem_util_pct", "GPU mem util %", "#7f7f7f"),
        ("power_w", "Power W", "#bcbd22"),
    ], 2, "GPU utilization / power")

    progress_rows = []
    for row in rows:
        cur = row.get("current", -1)
        total = row.get("total", -1)
        pct = float("nan")
        if total and total > 0 and cur >= 0:
            pct = 100.0 * float(cur) / float(total)
        copied = dict(row)
        copied["progress_pct"] = pct
        progress_rows.append(copied)
    draw_series(parts, xs, progress_rows, [
        ("progress_pct", "Progress %", "#17becf"),
        ("q8f16_gib", "Q8->F16 cache", "#ff9896"),
        ("cuda_free_gib", "CUDA free", "#98df8a"),
    ], 3, "Progress / cache / free memory")
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        raise SystemExit(f"usage: {argv[0]} INPUT.csv [OUTPUT.svg]")
    in_path = Path(argv[1])
    out_path = Path(argv[2]) if len(argv) == 3 else in_path.with_suffix(".svg")
    rows = load_rows(in_path)
    render(rows, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv)
