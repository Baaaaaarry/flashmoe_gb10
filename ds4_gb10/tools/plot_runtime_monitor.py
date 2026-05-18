#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path


WIDTH = 1800
HEIGHT = 1400
MARGIN_L = 70
MARGIN_R = 20
MARGIN_T = 40
MARGIN_B = 40
PANEL_GAP_X = 28
PANEL_GAP_Y = 50
PANEL_COLS = 1
PANEL_ROWS = 3


def load_rows(path: Path):
    rows = []
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "phase":
                    parsed[key] = value or "idle"
                elif key.endswith("_name"):
                    parsed[key] = value or ""
                elif key.endswith("_tid"):
                    try:
                        parsed[key] = int(float(value))
                    except Exception:
                        parsed[key] = 0
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


def collect_top_threads(rows, top_n=5):
    peaks = {}
    names = {}
    for row in rows:
        for i in range(1, 6):
            tid = row.get(f"t{i}_tid", 0)
            name = row.get(f"t{i}_name", "") or f"tid-{tid}"
            cpu = row.get(f"t{i}_cpu_pct", float("nan"))
            if not tid or not finite(cpu):
                continue
            peaks[tid] = max(peaks.get(tid, 0.0), cpu)
            names[tid] = name
    ordered = sorted(peaks.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(tid, names.get(tid, f"tid-{tid}")) for tid, _ in ordered]


def collect_top_processes(rows, top_n=5):
    peaks = {}
    names = {}
    for row in rows:
        for i in range(1, 6):
            pid = row.get(f"p{i}_pid", 0)
            name = row.get(f"p{i}_name", "") or f"pid-{pid}"
            mem = row.get(f"p{i}_mem_gib", float("nan"))
            if not pid or not finite(mem):
                continue
            peaks[pid] = max(peaks.get(pid, 0.0), mem)
            names[pid] = name
    ordered = sorted(peaks.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(pid, names.get(pid, f"pid-{pid}")) for pid, _ in ordered]


def build_thread_series(rows, tids, suffix):
    out = {tid: [] for tid, _ in tids}
    for row in rows:
        rowmap = {}
        for i in range(1, 6):
            tid = row.get(f"t{i}_tid", 0)
            if tid:
                rowmap[tid] = row.get(f"t{i}_{suffix}", float("nan"))
        for tid, _ in tids:
            out[tid].append(rowmap.get(tid, 0.0 if suffix == "cpu_pct" else float("nan")))
    return out


def build_process_series(rows, pids):
    out = {pid: [] for pid, _ in pids}
    for row in rows:
        rowmap = {}
        for i in range(1, 6):
            pid = row.get(f"p{i}_pid", 0)
            if pid:
                rowmap[pid] = row.get(f"p{i}_mem_gib", float("nan"))
        for pid, _ in pids:
            out[pid].append(rowmap.get(pid, float("nan")))
    return out


def render(rows, out_path: Path):
    xs = [row["time_s"] for row in rows]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fafbfc"/>',
        '<text x="24" y="28" font-size="24" font-family="sans-serif" fill="#111">ds4 runtime monitor</text>',
    ]
    draw_phase_marks(parts, rows, xs)
    top_threads = [(tid, name) for tid, name in collect_top_threads(rows, top_n=5)]
    top_procs = [(pid, name) for pid, name in collect_top_processes(rows, top_n=5)]
    thread_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    thread_cpu = build_thread_series(rows, top_threads, "cpu_pct")
    proc_mem = build_process_series(rows, top_procs)

    # Panel 0: top thread CPU 0..100%
    x0, y0, w, h = panel_box(0)
    draw_axes(parts, x0, y0, w, h, "Top-5 CPU threads (% of one core)", 0.0, 100.0, max(xs) if xs else 1.0)
    legend_x = x0 + w - 280
    legend_y = y0 + 18
    if top_threads:
        parts.append(f'<rect x="{legend_x - 10:.1f}" y="{y0 + 4:.1f}" width="300" height="{24 + max(0, len(top_threads)-1) * 18:.1f}" fill="rgba(255,255,255,0.85)" stroke="#e5e7eb"/>')
    for idx, (tid, name) in enumerate(top_threads):
        color = thread_colors[idx % len(thread_colors)]
        ys = thread_cpu[tid]
        scaled = scale_points(xs, ys, x0, y0, w, h, 0.0, 100.0)
        if not scaled:
            continue
        pts, _, _, _, _ = scaled
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        lx = legend_x
        ly = legend_y + idx * 18
        parts.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx + 18:.1f}" y2="{ly:.1f}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 24:.1f}" y="{ly + 4:.1f}" font-size="12" font-family="sans-serif" fill="#333">{svg_escape(str(name))} ({tid})</text>')

    # Panel 1: system + top process memory
    x0, y0, w, h = panel_box(1)
    mem_vals = []
    mem_vals.extend([r.get("mem_total_used_gib", float("nan")) for r in rows])
    for pid, _ in top_procs:
        mem_vals.extend([v for v in proc_mem[pid] if finite(v)])
    mem_ymax = max(mem_vals) if mem_vals else 1.0
    if mem_ymax <= 0.0:
        mem_ymax = 1.0
    draw_axes(parts, x0, y0, w, h, "System memory and Top-5 process RSS (GiB)", 0.0, mem_ymax, max(xs) if xs else 1.0)
    legend_x = x0 + w - 280
    legend_y = y0 + 18
    legend_rows = 1 + len(top_procs)
    parts.append(f'<rect x="{legend_x - 10:.1f}" y="{y0 + 4:.1f}" width="300" height="{24 + max(0, legend_rows-1) * 18:.1f}" fill="rgba(255,255,255,0.85)" stroke="#e5e7eb"/>')
    sys_mem = [r.get("mem_total_used_gib", float("nan")) for r in rows]
    scaled = scale_points(xs, sys_mem, x0, y0, w, h, 0.0, mem_ymax)
    if scaled:
        pts, _, _, _, _ = scaled
        parts.append(f'<polyline fill="none" stroke="#111827" stroke-width="2.5" points="{pts}"/>')
    parts.append(f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 18:.1f}" y2="{legend_y:.1f}" stroke="#111827" stroke-width="3"/>')
    parts.append(f'<text x="{legend_x + 24:.1f}" y="{legend_y + 4:.1f}" font-size="12" font-family="sans-serif" fill="#333">System memory used</text>')
    for idx, (pid, name) in enumerate(top_procs):
        color = thread_colors[idx % len(thread_colors)]
        ys = proc_mem[pid]
        scaled = scale_points(xs, ys, x0, y0, w, h, 0.0, mem_ymax)
        if not scaled:
            continue
        pts, _, _, _, _ = scaled
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        lx = legend_x
        ly = legend_y + (idx + 1) * 18
        parts.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx + 18:.1f}" y2="{ly:.1f}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 24:.1f}" y="{ly + 4:.1f}" font-size="12" font-family="sans-serif" fill="#333">{svg_escape(str(name))} ({pid})</text>')

    # Panel 2: GPU util + CUDA model cache (dual axis)
    x0, y0, w, h = panel_box(2)
    gpu_vals = [r.get("gpu_util_pct", float("nan")) for r in rows]
    cache_vals = [r.get("cuda_model_gib", float("nan")) for r in rows]
    draw_axes(parts, x0, y0, w, h, "GPU util % and CUDA model cache GiB", 0.0, 100.0, max(xs) if xs else 1.0)
    good_cache = [v for v in cache_vals if finite(v)]
    cache_max = max(good_cache) if good_cache else 1.0
    if cache_max <= 0.0:
        cache_max = 1.0
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 + h - frac * h
        val = frac * cache_max
        parts.append(f'<text x="{x0 + w + 8:.1f}" y="{yy + 4:.1f}" font-size="11" font-family="sans-serif" fill="#555">{val:.1f}</text>')
    parts.append(f'<text x="{x0 + w + 8:.1f}" y="{y0 - 12:.1f}" font-size="14" font-family="sans-serif" fill="#555">GiB</text>')
    scaled = scale_points(xs, gpu_vals, x0, y0, w, h, 0.0, 100.0)
    if scaled:
        pts, _, _, _, _ = scaled
        parts.append(f'<polyline fill="none" stroke="#e377c2" stroke-width="2.5" points="{pts}"/>')
    scaled = scale_points(xs, cache_vals, x0, y0, w, h, 0.0, cache_max)
    if scaled:
        pts, _, _, _, _ = scaled
        parts.append(f'<polyline fill="none" stroke="#2ca02c" stroke-width="2.5" points="{pts}"/>')
    legend_x = x0 + w - 280
    legend_y = y0 + 18
    parts.append(f'<rect x="{legend_x - 10:.1f}" y="{y0 + 4:.1f}" width="300" height="42" fill="rgba(255,255,255,0.85)" stroke="#e5e7eb"/>')
    parts.append(f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 18:.1f}" y2="{legend_y:.1f}" stroke="#e377c2" stroke-width="3"/>')
    parts.append(f'<text x="{legend_x + 24:.1f}" y="{legend_y + 4:.1f}" font-size="12" font-family="sans-serif" fill="#333">GPU util %</text>')
    parts.append(f'<line x1="{legend_x:.1f}" y1="{legend_y + 18:.1f}" x2="{legend_x + 18:.1f}" y2="{legend_y + 18:.1f}" stroke="#2ca02c" stroke-width="3"/>')
    parts.append(f'<text x="{legend_x + 24:.1f}" y="{legend_y + 22:.1f}" font-size="12" font-family="sans-serif" fill="#333">CUDA model cache GiB</text>')
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
