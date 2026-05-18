#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            out = {}
            for k, v in row.items():
                if k == "phase":
                    out[k] = v or "idle"
                elif k.endswith("_name"):
                    out[k] = v or ""
                elif k.endswith("_tid"):
                    try:
                        out[k] = int(float(v))
                    except Exception:
                        out[k] = 0
                elif k in ("current", "total", "gpu_util_pct", "gpu_mem_util_pct"):
                    try:
                        out[k] = int(v)
                    except Exception:
                        out[k] = -1
                else:
                    try:
                        out[k] = float(v)
                    except Exception:
                        out[k] = float("nan")
            rows.append(out)
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def finite(v):
    return isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v) and v >= 0


def avg(vals):
    vals = [v for v in vals if finite(v)]
    return statistics.fmean(vals) if vals else float("nan")


def peak(vals):
    vals = [v for v in vals if finite(v)]
    return max(vals) if vals else float("nan")


def summarize_phase(rows, phase):
    sub = [r for r in rows if r.get("phase") == phase]
    if not sub:
        return None
    return {
        "samples": len(sub),
        "duration_s": (sub[-1]["time_s"] - sub[0]["time_s"]) if len(sub) > 1 else 0.0,
        "avg_proc_cpu_pct": avg([r.get("proc_cpu_pct") for r in sub]),
        "avg_sys_cpu_pct": avg([r.get("sys_cpu_pct") for r in sub]),
        "avg_cpu_freq_mhz": avg([r.get("cpu_freq_mhz") for r in sub]),
        "avg_gpu_util_pct": avg([r.get("gpu_util_pct") for r in sub]),
        "peak_gpu_util_pct": peak([r.get("gpu_util_pct") for r in sub]),
        "avg_gpu_mem_util_pct": avg([r.get("gpu_mem_util_pct") for r in sub]),
        "avg_sm_clock_mhz": avg([r.get("gpu_sm_clock_mhz") for r in sub]),
        "avg_mem_clock_mhz": avg([r.get("gpu_mem_clock_mhz") for r in sub]),
        "avg_power_w": avg([r.get("power_w") for r in sub]),
        "peak_power_w": peak([r.get("power_w") for r in sub]),
        "avg_rss_gib": avg([r.get("rss_gib") for r in sub]),
        "peak_rss_gib": peak([r.get("rss_gib") for r in sub]),
        "avg_gpu_mem_used_gib": avg([r.get("gpu_mem_used_gib") for r in sub]),
        "peak_gpu_mem_used_gib": peak([r.get("gpu_mem_used_gib") for r in sub]),
        "avg_cuda_model_gib": avg([r.get("cuda_model_gib") for r in sub]),
        "peak_cuda_model_gib": peak([r.get("cuda_model_gib") for r in sub]),
        "avg_read_mibs": avg([r.get("io_read_mibs") for r in sub]),
        "peak_read_mibs": peak([r.get("io_read_mibs") for r in sub]),
        "avg_write_mibs": avg([r.get("io_write_mibs") for r in sub]),
        "peak_write_mibs": peak([r.get("io_write_mibs") for r in sub]),
    }


def fmt(v, unit=""):
    if not finite(v):
        return "n/a"
    return f"{v:.2f}{unit}"


def main(argv):
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} MONITOR.csv")
    path = Path(argv[1])
    rows = load_rows(path)
    phase_order = ["startup", "inspect", "prefill", "decode", "generation", "flashmoe-export", "kv-cache-report", "imatrix", "repl"]
    phases = []
    for row in rows:
        p = row.get("phase", "idle")
        if p not in phases:
            phases.append(p)
    phases.sort(key=lambda p: phase_order.index(p) if p in phase_order else len(phase_order))
    print(f"runtime monitor summary: {path}")
    print(f"total samples: {len(rows)}")
    print("")
    # Top CPU threads by peak slot sample.
    thread_peaks = {}
    thread_names = {}
    thread_mem_peaks = {}
    for row in rows:
        for i in range(1, 6):
            tid = row.get(f"t{i}_tid", 0)
            if not tid:
                continue
            cpu = row.get(f"t{i}_cpu_pct", float("nan"))
            mem = row.get(f"t{i}_mem_gib", float("nan"))
            thread_names[tid] = row.get(f"t{i}_name", "") or f"tid-{tid}"
            if finite(cpu):
                thread_peaks[tid] = max(thread_peaks.get(tid, 0.0), cpu)
            if finite(mem):
                thread_mem_peaks[tid] = max(thread_mem_peaks.get(tid, 0.0), mem)
    if thread_peaks:
        print("[top CPU threads]")
        for tid, cpu in sorted(thread_peaks.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            print(f"  {thread_names.get(tid, f'tid-{tid}')} ({tid}): cpu_peak={fmt(cpu, '%')} mem_peak={fmt(thread_mem_peaks.get(tid, float('nan')), 'GiB')}")
        print("")
    for phase in phases:
        summary = summarize_phase(rows, phase)
        if not summary:
            continue
        print(f"[{phase}]")
        print(f"  duration_s: {fmt(summary['duration_s'])}")
        print(f"  samples: {summary['samples']}")
        print(f"  cpu: proc_avg={fmt(summary['avg_proc_cpu_pct'], '%')} sys_avg={fmt(summary['avg_sys_cpu_pct'], '%')} freq_avg={fmt(summary['avg_cpu_freq_mhz'], 'MHz')}")
        print(f"  gpu: util_avg={fmt(summary['avg_gpu_util_pct'], '%')} util_peak={fmt(summary['peak_gpu_util_pct'], '%')} mem_util_avg={fmt(summary['avg_gpu_mem_util_pct'], '%')}")
        print(f"  gpu_clocks: sm_avg={fmt(summary['avg_sm_clock_mhz'], 'MHz')} mem_avg={fmt(summary['avg_mem_clock_mhz'], 'MHz')} power_avg={fmt(summary['avg_power_w'], 'W')} power_peak={fmt(summary['peak_power_w'], 'W')}")
        print(f"  memory: rss_avg={fmt(summary['avg_rss_gib'], 'GiB')} rss_peak={fmt(summary['peak_rss_gib'], 'GiB')} gpu_mem_avg={fmt(summary['avg_gpu_mem_used_gib'], 'GiB')} gpu_mem_peak={fmt(summary['peak_gpu_mem_used_gib'], 'GiB')} cuda_model_peak={fmt(summary['peak_cuda_model_gib'], 'GiB')}")
        print(f"  bandwidth: read_avg={fmt(summary['avg_read_mibs'], 'MiB/s')} read_peak={fmt(summary['peak_read_mibs'], 'MiB/s')} write_avg={fmt(summary['avg_write_mibs'], 'MiB/s')} write_peak={fmt(summary['peak_write_mibs'], 'MiB/s')}")
        print(f"  cache+gpu: cuda_model_peak={fmt(summary['peak_cuda_model_gib'], 'GiB')} sm_clock_avg={fmt(summary['avg_sm_clock_mhz'], 'MHz')}")
        print("")


if __name__ == "__main__":
    main(sys.argv)
