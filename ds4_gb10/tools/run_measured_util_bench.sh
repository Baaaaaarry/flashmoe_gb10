#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
usage: run_measured_util_bench.sh MODEL.gguf OUT.csv [prompt_token_count ...]

Runs ds4 prefill-only sweeps and records:
  - real prefill throughput from a normal run
  - measured DRAM utilization proxy from Nsight Compute
  - measured MAC utilization proxy from Nsight Compute

Requirements:
  - ncu available in PATH

Environment:
  DS4_UTIL_PROMPT_FILE     Prompt corpus file. Default: tests/long_context_story_prompt.txt
  DS4_UTIL_BACKEND         cuda|metal|cpu. Default: cuda
  DS4_UTIL_CTX_MARGIN      Extra ctx slots beyond prompt+1. Default: 16
  DS4_UTIL_EXTRA_ARGS      Extra ds4 CLI args appended to every run
  DS4_UTIL_MEM_BW_GIBS     Peak UMA bandwidth for reference. Default: 273
  DS4_UTIL_COMPUTE_PEAK_TF Peak compute throughput for reference. Default: 123
EOF
  exit 2
fi

MODEL=$1
OUT_CSV=$2
shift 2

PROMPT_FILE=${DS4_UTIL_PROMPT_FILE:-tests/long_context_story_prompt.txt}
BACKEND=${DS4_UTIL_BACKEND:-cuda}
CTX_MARGIN=${DS4_UTIL_CTX_MARGIN:-16}
EXTRA_ARGS_STR=${DS4_UTIL_EXTRA_ARGS:-}
MEM_BW_GIBS=${DS4_UTIL_MEM_BW_GIBS:-273}
COMPUTE_PEAK_TF=${DS4_UTIL_COMPUTE_PEAK_TF:-123}

if [[ $# -eq 0 ]]; then
  set -- 128 256 512 1024 2048 4096 8192 65536 131072
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "missing ncu in PATH" >&2
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

TOTAL_TOKENS=$(count_prompt_tokens "$ART_DIR/prompt_tokens.txt")
MAX_REQ=0
for v in "$@"; do
  if (( v > MAX_REQ )); then MAX_REQ=$v; fi
done
if (( TOTAL_TOKENS < MAX_REQ )); then
  echo "prompt corpus only has ${TOTAL_TOKENS} tokens, need at least ${MAX_REQ}" >&2
  exit 1
fi

printf "prompt_tokens,ctx_alloc,prefill_tps,prefill_s,eta_mem_pct,eta_mac_pct,mac_metric,dram_read_gib,dram_write_gib,mem_bw_gibs_ref,compute_peak_tf_ref,ncu_status,parse_status,stderr_log,ncu_stderr_log,ncu_raw_csv,ncu_rep\n" >"$OUT_CSV"

for prompt_tokens in "$@"; do
  ctx_alloc=$((prompt_tokens + 1 + CTX_MARGIN))
  run_name="p${prompt_tokens}"
  stderr_log="$ART_DIR/${run_name}.stderr.log"
  stdout_log="$ART_DIR/${run_name}.stdout.log"
  ncu_stdout="$ART_DIR/${run_name}.ncu.stdout.log"
  ncu_stderr="$ART_DIR/${run_name}.ncu.stderr.log"
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
    --prefill-only
    --prompt-file "$PROMPT_FILE"
    --prompt-token-limit "$prompt_tokens"
  )

  if [[ -n "$EXTRA_ARGS_STR" ]]; then
    # shellcheck disable=SC2206
    extra=( $EXTRA_ARGS_STR )
    cmd+=("${extra[@]}")
  fi

  "${cmd[@]}" >"$stdout_log" 2>"$stderr_log"

  ncu_status="ok"
  parse_status="ok"
  : >"$raw_csv"
  full_metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum"
  fallback_metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum"
  if ! ncu \
    --target-processes all \
    --force-overwrite \
    --page raw \
    --csv \
    --metrics \
    "$full_metrics" \
    -o "$rep_base" \
    "${cmd[@]}" >"$ncu_stdout" 2>"$ncu_stderr"; then
    if ! ncu \
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
  elif ! ncu --import "$rep_file" --csv --page raw >"$raw_csv" 2>>"$ncu_stderr"; then
    ncu_status="import_failed"
  else
    ncu_status="full_metrics"
  fi

  if [[ "$ncu_status" == "fallback_metrics" ]]; then
    if ! ncu --import "$rep_file" --csv --page raw >"$raw_csv" 2>>"$ncu_stderr"; then
      ncu_status="import_failed"
    fi
  fi

  python3 - "$stderr_log" "$raw_csv" "$OUT_CSV" "$prompt_tokens" "$ctx_alloc" "$MEM_BW_GIBS" "$COMPUTE_PEAK_TF" "$stderr_log" "$ncu_stderr" "$raw_csv" "$rep_file" "$ncu_status" "$parse_status" <<'PY'
import csv, json, pathlib, re, subprocess, sys
stderr_path, raw_csv, out_csv, prompt_tokens, ctx_alloc, mem_bw_gibs, compute_peak_tf, stderr_log, ncu_stderr_log, raw_csv_log, rep_file, ncu_status, parse_status = sys.argv[1:]
text = pathlib.Path(stderr_path).read_text(encoding='utf-8', errors='replace')
m = re.search(r"ds4: prefill-only: ([0-9.]+) t/s", text)
if not m:
    m = re.search(r"ds4: prefill: ([0-9.]+) t/s", text)
if not m:
    print(f"failed to parse prefill throughput from {stderr_path}", file=sys.stderr)
    sys.exit(1)
prefill_tps = float(m.group(1))
prompt_tokens_i = int(prompt_tokens)
prefill_s = prompt_tokens_i / prefill_tps if prefill_tps > 0 else 0.0
parsed = {
    "eta_mem_pct": "",
    "eta_mac_pct": "",
    "mac_metric": "",
    "dram_read_bytes": 0.0,
    "dram_write_bytes": 0.0,
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
with open(out_csv, "a", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow([
        prompt_tokens_i,
        int(ctx_alloc),
        prefill_tps,
        prefill_s,
        parsed.get("eta_mem_pct", ""),
        parsed.get("eta_mac_pct", ""),
        parsed.get("mac_metric", ""),
        (parsed.get("dram_read_bytes", 0.0) or 0.0) / (1024 ** 3),
        (parsed.get("dram_write_bytes", 0.0) or 0.0) / (1024 ** 3),
        float(mem_bw_gibs),
        float(compute_peak_tf),
        ncu_status,
        parse_status,
        stderr_log,
        ncu_stderr_log,
        raw_csv_log,
        rep_file,
    ])
PY

  printf "done prompt_tokens=%s ctx_alloc=%s ncu_status=%s\n" "$prompt_tokens" "$ctx_alloc" "$ncu_status"
done

echo "wrote $OUT_CSV"
echo "artifacts dir: $ART_DIR"
