#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
usage: run_measured_util_bench.sh MODEL.gguf OUT.csv [prompt_token_count ...]

Runs ds4 full-generation sweeps and records per-phase metrics:
  - real prefill throughput from the prompt processing phase
  - real generation throughput from the decode phase
  - measured DRAM/GPU utilization proxies split into prefill and decode windows

Requirements:
  - ncu available in PATH

Environment:
  DS4_UTIL_PROMPT_FILE     Prompt corpus file. Default: tests/long_context_story_prompt.txt
  DS4_UTIL_BACKEND         cuda|metal|cpu. Default: cuda
  DS4_UTIL_CTX_MARGIN      Extra ctx slots beyond prompt+1. Default: 16
  DS4_UTIL_EXTRA_ARGS      Extra ds4 CLI args appended to every run
  DS4_UTIL_GEN_TOKENS      Number of generated tokens. Default: 128
  DS4_UTIL_MEM_BW_GIBS     Peak UMA bandwidth for reference. Default: 273
  DS4_UTIL_COMPUTE_PEAK_TF Peak compute throughput for reference. Default: 123
  DS4_UTIL_PREFILL_MODEL_GIB
                           Prefill resident-weight model per 2048-token chunk.
                           Default: 80.76
  DS4_UTIL_DECODE_MODEL_GIB
                           Decode active-weight model per token. Default: 10.97
  DS4_UTIL_NCU_BIN         Nsight Compute binary. Default: /usr/local/cuda/bin/ncu
  DS4_UTIL_USE_NCU         Try Nsight Compute collection. Default: 0
  DS4_UTIL_NCU_USE_SUDO    Use sudo -E for ncu. Default: 0
  DS4_UTIL_NCU_NVTX_INCLUDE
                           Optional NVTX include filter, passed with --nvtx.
  DS4_UTIL_NCU_KERNEL      Optional kernel filter, passed with -k.
  DS4_UTIL_NCU_COUNT       ncu -c count. Default: 10
  DS4_UTIL_NCU_SKIP        ncu -s skip. Default: 8
EOF
  exit 2
fi

MODEL=$1
OUT_CSV=$2
shift 2

PROMPT_FILE=${DS4_UTIL_PROMPT_FILE:-tests/long_context_story_prompt.txt}
BACKEND=${DS4_UTIL_BACKEND:-cuda}
GEN_TOKENS=${DS4_UTIL_GEN_TOKENS:-128}
CTX_MARGIN=${DS4_UTIL_CTX_MARGIN:-16}
EXTRA_ARGS_STR=${DS4_UTIL_EXTRA_ARGS:-}
MEM_BW_GIBS=${DS4_UTIL_MEM_BW_GIBS:-273}
COMPUTE_PEAK_TF=${DS4_UTIL_COMPUTE_PEAK_TF:-123}
PREFILL_MODEL_GIB=${DS4_UTIL_PREFILL_MODEL_GIB:-80.76}
DECODE_MODEL_GIB=${DS4_UTIL_DECODE_MODEL_GIB:-10.97}
NCU_BIN=${DS4_UTIL_NCU_BIN:-/usr/local/cuda/bin/ncu}
USE_NCU=${DS4_UTIL_USE_NCU:-0}
NCU_USE_SUDO=${DS4_UTIL_NCU_USE_SUDO:-0}
NCU_NVTX_INCLUDE=${DS4_UTIL_NCU_NVTX_INCLUDE:-}
NCU_KERNEL=${DS4_UTIL_NCU_KERNEL:-}
NCU_COUNT=${DS4_UTIL_NCU_COUNT:-10}
NCU_SKIP=${DS4_UTIL_NCU_SKIP:-8}

if [[ $# -eq 0 ]]; then
  set -- 128 256 512 1024 2048 4096 8192 65536 131072
fi

if [[ "$USE_NCU" == "1" ]] && [[ ! -x "$NCU_BIN" ]] && ! command -v "$NCU_BIN" >/dev/null 2>&1; then
  echo "missing ncu binary: $NCU_BIN" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  if [[ -f tests/generate_long_context_story_prompt.py ]]; then
    max_req=0
    for v in "$@"; do
      if (( v > max_req )); then max_req=$v; fi
    done
    python3 tests/generate_long_context_story_prompt.py --min-tokens "$max_req"
  fi
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "missing prompt file: $PROMPT_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"
ART_DIR="${OUT_CSV%.csv}.artifacts"
mkdir -p "$ART_DIR"
LOCK_DIR="$(dirname "$OUT_CSV")/locks"
mkdir -p "$LOCK_DIR"

if [[ -z "${DS4_LOCK_FILE:-}" ]]; then
  export DS4_LOCK_FILE="$LOCK_DIR/ds4_${USER}_$$.lock"
fi
rm -f "$DS4_LOCK_FILE"

count_prompt_tokens() {
  local dump_file=$1
  ./ds4 -m "$MODEL" --dump-tokens --prompt-file "$PROMPT_FILE" >"$dump_file" 2>/dev/null
  python3 - "$dump_file" <<'PY'
import ast, sys
line = open(sys.argv[1], 'r', encoding='utf-8').readline().strip()
tokens = ast.literal_eval(line)
print(len(tokens))
PY
}

run_ncu_profile() {
  local rep_base=$1
  local stdout_log=$2
  local stderr_log=$3
  shift 3
  local -a target_cmd=( "$@" )
  local -a ncu_cmd=()
  if [[ "$NCU_USE_SUDO" == "1" ]]; then
    ncu_cmd+=( sudo -E )
  fi
  ncu_cmd+=(
    "$NCU_BIN"
    --target-processes all
    --force-overwrite
  )
  if [[ -n "$NCU_NVTX_INCLUDE" ]]; then
    ncu_cmd+=( --nvtx --nvtx-include "$NCU_NVTX_INCLUDE" )
  fi
  ncu_cmd+=(
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --section MemoryWorkloadAnalysis_Chart
    --section MemoryWorkloadAnalysis_Tables
    --section ComputeWorkloadAnalysis
    --section InstructionStats
    --section SpeedOfLight_HierarchicalHalfRooflineChart
    --section SpeedOfLight_HierarchicalTensorRooflineChart
    --section SpeedOfLight_RooflineChart
  )
  if [[ -n "$NCU_KERNEL" ]]; then
    ncu_cmd+=( -k "$NCU_KERNEL" )
  fi
  ncu_cmd+=( -c "$NCU_COUNT" -s "$NCU_SKIP" -o "$rep_base" )
  "${ncu_cmd[@]}" "${target_cmd[@]}" >"$stdout_log" 2>"$stderr_log"
}

TOTAL_TOKENS=$(count_prompt_tokens "$ART_DIR/prompt_tokens.txt")
MAX_REQ=0
for v in "$@"; do
  if (( v > MAX_REQ )); then MAX_REQ=$v; fi
done
if (( TOTAL_TOKENS < MAX_REQ )); then
  if [[ -f tests/generate_long_context_story_prompt.py ]]; then
    python3 tests/generate_long_context_story_prompt.py --min-tokens "$MAX_REQ" --output "$PROMPT_FILE"
    TOTAL_TOKENS=$(count_prompt_tokens "$ART_DIR/prompt_tokens.txt")
  fi
fi
if (( TOTAL_TOKENS < MAX_REQ )); then
  echo "prompt corpus only has ${TOTAL_TOKENS} tokens, need at least ${MAX_REQ}" >&2
  exit 1
fi

printf "prompt_tokens,gen_tokens,ctx_alloc,prefill_tps,generation_tps,prefill_s,decode_s,prefill_eta_mem_pct,prefill_eta_mac_pct,decode_eta_mem_pct,decode_eta_mac_pct,util_source,prefill_mac_metric,decode_mac_metric,dram_read_gib,dram_write_gib,prefill_gpu_util_proxy_pct,prefill_gpu_mem_util_proxy_pct,decode_gpu_util_proxy_pct,decode_gpu_mem_util_proxy_pct,mem_bw_gibs_ref,compute_peak_tf_ref,ncu_status,parse_status,stderr_log,ncu_stderr_log,nvidia_smi_log,ncu_raw_csv,ncu_rep\n" >"$OUT_CSV"

for prompt_tokens in "$@"; do
  ctx_alloc=$((prompt_tokens + 1 + CTX_MARGIN))
  run_name="p${prompt_tokens}"
  stderr_log="$ART_DIR/${run_name}.stderr.log"
  stdout_log="$ART_DIR/${run_name}.stdout.log"
  ncu_stdout="$ART_DIR/${run_name}.ncu.stdout.log"
  ncu_stderr="$ART_DIR/${run_name}.ncu.stderr.log"
  smi_log="$ART_DIR/${run_name}.nvidia-smi.log"
  dmon_log="$ART_DIR/${run_name}.nvidia-smi.dmon.log"
  rep_base="$ART_DIR/${run_name}"
  rep_file="${rep_base}.ncu-rep"
  raw_csv="$ART_DIR/${run_name}.ncu.raw.csv"

  cmd=(
    ./ds4
    -m "$MODEL"
    "--${BACKEND}"
    --ctx "$ctx_alloc"
    --nothink
    --temp 0
    -n "$GEN_TOKENS"
    --prompt-file "$PROMPT_FILE"
    --prompt-token-limit "$prompt_tokens"
  )

  if [[ -n "$EXTRA_ARGS_STR" ]]; then
    # shellcheck disable=SC2206
    extra=( $EXTRA_ARGS_STR )
    cmd+=("${extra[@]}")
  fi

  smi_pid=""
  dmon_pid=""
  : >"$smi_log"
  : >"$dmon_log"
  if [[ "$BACKEND" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory,power.draw \
      --format=csv,noheader,nounits \
      -lms 100 >"$smi_log" 2>/dev/null &
    smi_pid=$!
    nvidia-smi dmon -s u -d 1 >"$dmon_log" 2>/dev/null &
    dmon_pid=$!
  fi

  run_t0=$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)

  "${cmd[@]}" >"$stdout_log" 2>"$stderr_log"

  run_t1=$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)

  if [[ -n "$smi_pid" ]]; then
    kill "$smi_pid" >/dev/null 2>&1 || true
    wait "$smi_pid" 2>/dev/null || true
  fi
  if [[ -n "$dmon_pid" ]]; then
    kill "$dmon_pid" >/dev/null 2>&1 || true
    wait "$dmon_pid" 2>/dev/null || true
  fi

  ncu_status="disabled"
  parse_status="ok"
  : >"$raw_csv"
  full_metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum"
  fallback_metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum"
  if [[ "$USE_NCU" == "1" ]]; then
    ncu_status="ok"
    if ! run_ncu_profile "$rep_base" "$ncu_stdout" "$ncu_stderr" "${cmd[@]}"; then
      if [[ "$NCU_USE_SUDO" == "1" ]]; then
        ncu_status="profile_failed"
      else
        if ! "$NCU_BIN" \
          --target-processes all \
          --force-overwrite \
          --page raw \
          --csv \
          --metrics \
          "$fallback_metrics" \
          -o "$rep_base" \
          "${cmd[@]}" >>"$ncu_stdout" 2>>"$ncu_stderr"; then
          ncu_status="profile_failed"
        else
          ncu_status="fallback_metrics"
        fi
      fi
    elif ! "$NCU_BIN" --import "$rep_file" --csv --page raw >"$raw_csv" 2>>"$ncu_stderr"; then
      ncu_status="import_failed"
    else
      ncu_status="sections"
    fi

    if [[ "$ncu_status" == "fallback_metrics" ]]; then
      if ! "$NCU_BIN" --import "$rep_file" --csv --page raw >"$raw_csv" 2>>"$ncu_stderr"; then
        ncu_status="import_failed"
      fi
    fi
  fi

  python3 - "$stderr_log" "$raw_csv" "$OUT_CSV" "$prompt_tokens" "$GEN_TOKENS" "$ctx_alloc" "$MEM_BW_GIBS" "$COMPUTE_PEAK_TF" "$PREFILL_MODEL_GIB" "$DECODE_MODEL_GIB" "$stderr_log" "$ncu_stderr" "$smi_log" "$dmon_log" "$raw_csv" "$rep_file" "$ncu_status" "$parse_status" "$run_t0" "$run_t1" <<'PY'
import csv, json, pathlib, re, subprocess, sys
from datetime import datetime
stderr_path, raw_csv, out_csv, prompt_tokens, gen_tokens, ctx_alloc, mem_bw_gibs, compute_peak_tf, prefill_model_gib, decode_model_gib, stderr_log, ncu_stderr_log, smi_log, dmon_log, raw_csv_log, rep_file, ncu_status, parse_status, run_t0, run_t1 = sys.argv[1:]
text = pathlib.Path(stderr_path).read_text(encoding='utf-8', errors='replace')
prefill_m = re.search(r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s", text)
if not prefill_m:
    prefill_m = re.search(r"ds4: prefill-only: ([0-9.]+) t/s", text)
if not prefill_m:
    print(f"failed to parse throughput from {stderr_path}", file=sys.stderr)
    sys.exit(1)
if len(prefill_m.groups()) == 2:
    prefill_tps = float(prefill_m.group(1))
    generation_tps = float(prefill_m.group(2))
else:
    prefill_tps = float(prefill_m.group(1))
    generation_tps = 0.0
prompt_tokens_i = int(prompt_tokens)
gen_tokens_i = int(gen_tokens)
prefill_s = prompt_tokens_i / prefill_tps if prefill_tps > 0 else 0.0
decode_s = gen_tokens_i / generation_tps if generation_tps > 0 else 0.0
mem_bw_gibs_f = float(mem_bw_gibs)
prefill_model_gib_f = float(prefill_model_gib)
decode_model_gib_f = float(decode_model_gib)
parsed = {
    "eta_mem_pct": "",
    "eta_mac_pct": "",
    "util_source": "",
    "mac_metric": "",
    "dram_read_bytes": 0.0,
    "dram_write_bytes": 0.0,
    "gpu_util_proxy_pct": "",
    "gpu_mem_util_proxy_pct": "",
}
if pathlib.Path(raw_csv).exists() and pathlib.Path(raw_csv).stat().st_size > 0:
    try:
        parsed = json.loads(subprocess.check_output(
            [sys.executable, "tools/parse_ncu_raw_csv.py", raw_csv],
            text=True,
        ))
    except Exception:
        parse_status = "parse_failed"
else:
    if ncu_status == "ok":
        parse_status = "empty_raw_csv"
    else:
        parse_status = "skipped"

def parse_smi_time(s):
    s = s.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except Exception:
            pass
    return None

def parse_smi_proxy(path, prefill_t0, prefill_t1, decode_t1):
    pre_gpu = []
    pre_mem = []
    dec_gpu = []
    dec_mem = []
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            ts = parse_smi_time(parts[0])
            gpu = float(parts[1])
            mem = float(parts[2])
        except Exception:
            continue
        if ts is None:
            continue
        if prefill_t0 <= ts <= prefill_t1:
            pre_gpu.append(gpu)
            pre_mem.append(mem)
        elif prefill_t1 < ts <= decode_t1:
            dec_gpu.append(gpu)
            dec_mem.append(mem)
    def avg(xs):
        return sum(xs) / len(xs) if xs else None
    return {
        "prefill_gpu": avg(pre_gpu),
        "prefill_mem": avg(pre_mem),
        "decode_gpu": avg(dec_gpu),
        "decode_mem": avg(dec_mem),
    }

def parse_dmon_proxy(path, prefill_s, decode_s):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            gpu = float(parts[1])
            mem = float(parts[2])
        except Exception:
            continue
        rows.append((gpu, mem))
    if not rows:
        return None
    pre_n = max(1, round(prefill_s)) if prefill_s > 0 else 0
    dec_n = max(1, round(decode_s)) if decode_s > 0 else 0
    pre = rows[:pre_n] if pre_n else []
    dec = rows[pre_n:pre_n + dec_n] if dec_n else []
    def avg_pairs(xs, idx):
        return sum(v[idx] for v in xs) / len(xs) if xs else None
    return {
        "prefill_gpu": avg_pairs(pre, 0),
        "prefill_mem": avg_pairs(pre, 1),
        "decode_gpu": avg_pairs(dec, 0),
        "decode_mem": avg_pairs(dec, 1),
    }

def estimate_prefill_mem_util(prompt_tokens, prefill_s, mem_bw_gibs, per_chunk_gib):
    if prefill_s <= 0 or mem_bw_gibs <= 0:
        return ""
    chunks = (prompt_tokens + 2048 - 1) // 2048
    bw = (chunks * per_chunk_gib) / prefill_s
    return 100.0 * bw / mem_bw_gibs

def estimate_decode_mem_util(generation_tps, mem_bw_gibs, per_token_gib):
    if generation_tps <= 0 or mem_bw_gibs <= 0:
        return ""
    bw = generation_tps * per_token_gib
    return 100.0 * bw / mem_bw_gibs

run_t0_f = float(run_t0)
run_t1_f = float(run_t1)
prefill_t0 = run_t0_f
prefill_t1 = min(run_t0_f + prefill_s, run_t1_f)
decode_t1 = min(prefill_t1 + decode_s, run_t1_f)
smi = parse_smi_proxy(smi_log, prefill_t0, prefill_t1, decode_t1)
dmon = parse_dmon_proxy(dmon_log, prefill_s, decode_s)

prefill_gpu_util_proxy = ""
prefill_gpu_mem_util_proxy = ""
decode_gpu_util_proxy = ""
decode_gpu_mem_util_proxy = ""
prefill_eta_mem = parsed.get("eta_mem_pct", "")
prefill_eta_mac = parsed.get("eta_mac_pct", "")
decode_eta_mem = ""
decode_eta_mac = ""
prefill_mac_metric = parsed.get("mac_metric", "")
decode_mac_metric = ""

if smi:
    if smi["prefill_gpu"] is not None:
        prefill_gpu_util_proxy = smi["prefill_gpu"]
    if smi["prefill_mem"] is not None:
        prefill_gpu_mem_util_proxy = smi["prefill_mem"]
    if smi["decode_gpu"] is not None:
        decode_gpu_util_proxy = smi["decode_gpu"]
    if smi["decode_mem"] is not None:
        decode_gpu_mem_util_proxy = smi["decode_mem"]

if dmon:
    if prefill_gpu_util_proxy == "" and dmon["prefill_gpu"] is not None:
        prefill_gpu_util_proxy = dmon["prefill_gpu"]
    if (prefill_gpu_mem_util_proxy == "" or float(prefill_gpu_mem_util_proxy) == 0.0) and dmon["prefill_mem"] is not None:
        prefill_gpu_mem_util_proxy = dmon["prefill_mem"]
    if decode_gpu_util_proxy == "" and dmon["decode_gpu"] is not None:
        decode_gpu_util_proxy = dmon["decode_gpu"]
    if (decode_gpu_mem_util_proxy == "" or float(decode_gpu_mem_util_proxy) == 0.0) and dmon["decode_mem"] is not None:
        decode_gpu_mem_util_proxy = dmon["decode_mem"]

prefill_mem_missing = (prefill_gpu_mem_util_proxy == "" or float(prefill_gpu_mem_util_proxy) == 0.0)
decode_mem_missing = (decode_gpu_mem_util_proxy == "" or float(decode_gpu_mem_util_proxy) == 0.0)

if not prefill_eta_mem and ncu_status == "profile_failed":
    ncu_err = pathlib.Path(ncu_stderr_log).read_text(encoding="utf-8", errors="replace")
    if "ERR_NVGPUCTRPERM" in ncu_err and smi:
        prefill_eta_mac = prefill_gpu_util_proxy if prefill_gpu_util_proxy != "" else ""
        prefill_eta_mem = prefill_gpu_mem_util_proxy if prefill_gpu_mem_util_proxy != "" else ""
        decode_eta_mac = decode_gpu_util_proxy if decode_gpu_util_proxy != "" else ""
        decode_eta_mem = decode_gpu_mem_util_proxy if decode_gpu_mem_util_proxy != "" else ""
        parsed["util_source"] = "nvidia-smi-proxy"
        prefill_mac_metric = "gpu_util_pct_proxy"
        decode_mac_metric = "gpu_util_pct_proxy"
        parse_status = "proxy_fallback"
elif not prefill_eta_mem and ncu_status == "disabled" and smi:
    prefill_eta_mac = prefill_gpu_util_proxy if prefill_gpu_util_proxy != "" else ""
    prefill_eta_mem = prefill_gpu_mem_util_proxy if prefill_gpu_mem_util_proxy != "" else ""
    decode_eta_mac = decode_gpu_util_proxy if decode_gpu_util_proxy != "" else ""
    decode_eta_mem = decode_gpu_mem_util_proxy if decode_gpu_mem_util_proxy != "" else ""
    parsed["util_source"] = "nvidia-smi-proxy"
    prefill_mac_metric = "gpu_util_pct_proxy"
    decode_mac_metric = "gpu_util_pct_proxy"
    parse_status = "proxy_only"
elif parsed.get("eta_mem_pct") or parsed.get("eta_mac_pct"):
    parsed["util_source"] = "ncu"
    prefill_eta_mem = parsed.get("eta_mem_pct", "")
    prefill_eta_mac = parsed.get("eta_mac_pct", "")
    prefill_mac_metric = parsed.get("mac_metric", "")

if prefill_eta_mem in ("", 0, 0.0, "0", "0.0"):
    est = estimate_prefill_mem_util(prompt_tokens_i, prefill_s, mem_bw_gibs_f, prefill_model_gib_f)
    if est != "":
        prefill_eta_mem = est
        if parsed.get("util_source"):
            parsed["util_source"] += "+prefill-mem-model"
        else:
            parsed["util_source"] = "prefill-mem-model"

if decode_eta_mem in ("", 0, 0.0, "0", "0.0"):
    est = estimate_decode_mem_util(generation_tps, mem_bw_gibs_f, decode_model_gib_f)
    if est != "":
        decode_eta_mem = est
        if parsed.get("util_source"):
            parsed["util_source"] += "+decode-mem-model"
        else:
            parsed["util_source"] = "decode-mem-model"

with open(out_csv, "a", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow([
        prompt_tokens_i,
        gen_tokens_i,
        int(ctx_alloc),
        prefill_tps,
        generation_tps,
        prefill_s,
        decode_s,
        prefill_eta_mem,
        prefill_eta_mac,
        decode_eta_mem,
        decode_eta_mac,
        parsed.get("util_source", ""),
        prefill_mac_metric,
        decode_mac_metric,
        (parsed.get("dram_read_bytes", 0.0) or 0.0) / (1024 ** 3),
        (parsed.get("dram_write_bytes", 0.0) or 0.0) / (1024 ** 3),
        prefill_gpu_util_proxy,
        prefill_gpu_mem_util_proxy,
        decode_gpu_util_proxy,
        decode_gpu_mem_util_proxy,
        float(mem_bw_gibs),
        float(compute_peak_tf),
        ncu_status,
        parse_status,
        stderr_log,
        ncu_stderr_log,
        smi_log,
        raw_csv_log,
        rep_file,
    ])
PY

  printf "done prompt_tokens=%s ctx_alloc=%s ncu_status=%s\n" "$prompt_tokens" "$ctx_alloc" "$ncu_status"
done

echo "wrote $OUT_CSV"
echo "artifacts dir: $ART_DIR"
