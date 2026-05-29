#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
usage: run_e2e_bench.sh MODEL.gguf OUT.csv [prompt_token_count ...]

Runs end-to-end ds4 CLI generation across fixed prompt token frontiers using
one shared long prompt file plus --prompt-token-limit. Outputs CSV with
prefill/decode throughput.

Environment:
  DS4_E2E_PROMPT_FILE   Prompt corpus file. Default: tests/long_context_story_prompt.txt
  DS4_E2E_GEN_TOKENS    Decode tokens per run. Default: 128
  DS4_E2E_BACKEND       cuda|metal|cpu. Default: cuda
  DS4_E2E_CTX_MARGIN    Extra ctx slots beyond prompt+gen+1. Default: 16
  DS4_E2E_EXTRA_ARGS    Extra ds4 CLI args appended to every run
  DS4_E2E_MEM_BW_GIBS   Peak memory bandwidth used for utilization. Default: 273
  DS4_E2E_COMPUTE_PEAK_TF
                        Peak effective compute used for utilization. Default: 250
  DS4_E2E_SSD_BW_GIBS   Optional SSD cold-read bandwidth for utilization. Default: 0 (disabled)
EOF
  exit 2
fi

MODEL=$1
OUT_CSV=$2
shift 2

PROMPT_FILE=${DS4_E2E_PROMPT_FILE:-tests/long_context_story_prompt.txt}
GEN_TOKENS=${DS4_E2E_GEN_TOKENS:-128}
BACKEND=${DS4_E2E_BACKEND:-cuda}
CTX_MARGIN=${DS4_E2E_CTX_MARGIN:-16}
EXTRA_ARGS_STR=${DS4_E2E_EXTRA_ARGS:-}
MEM_BW_GIBS=${DS4_E2E_MEM_BW_GIBS:-273}
COMPUTE_PEAK_TF=${DS4_E2E_COMPUTE_PEAK_TF:-250}
SSD_BW_GIBS=${DS4_E2E_SSD_BW_GIBS:-0}

if [[ $# -eq 0 ]]; then
  set -- 128 1024 8192 65536 131072
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  if [[ -f tests/generate_long_context_story_prompt.py ]]; then
    python3 tests/generate_long_context_story_prompt.py
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

printf "prompt_tokens,ctx_alloc,gen_tokens,prefill_tps,generation_tps,prefill_flops_gf_per_token,prefill_weight_gib_per_token,prefill_act_rw_gib_per_token,prefill_kv_state_gib_per_token,prefill_total_bw_gib_per_token,prefill_ssd_cold_gib_per_token,prefill_bw_gibs,prefill_bw_util_pct,prefill_compute_tflops,prefill_compute_util_pct,prefill_ssd_gibs,prefill_ssd_util_pct,stderr_log,stdout_log\n" >"$OUT_CSV"

for prompt_tokens in "$@"; do
  ctx_alloc=$((prompt_tokens + GEN_TOKENS + 1 + CTX_MARGIN))
  run_name="p${prompt_tokens}"
  stderr_log="$ART_DIR/${run_name}.stderr.log"
  stdout_log="$ART_DIR/${run_name}.stdout.log"
  graph_log="$ART_DIR/${run_name}.graph.log"

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

  ./ds4 -m "$MODEL" "--${BACKEND}" --ctx "$ctx_alloc" --graph-profile --graph-profile-prompt-len "$prompt_tokens" >"$graph_log" 2>/dev/null
  "${cmd[@]}" >"$stdout_log" 2>"$stderr_log"

  python3 - "$stderr_log" "$graph_log" "$OUT_CSV" "$prompt_tokens" "$ctx_alloc" "$GEN_TOKENS" "$MEM_BW_GIBS" "$COMPUTE_PEAK_TF" "$SSD_BW_GIBS" "$stderr_log" "$stdout_log" <<'PY'
import csv, pathlib, re, sys
stderr_path, graph_path, out_csv, prompt_tokens, ctx_alloc, gen_tokens, mem_bw_gibs, compute_peak_tf, ssd_bw_gibs, stderr_log, stdout_log = sys.argv[1:]
text = pathlib.Path(stderr_path).read_text(encoding='utf-8', errors='replace')
m = re.search(r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s", text)
if not m:
    print(f"failed to parse throughput from {stderr_path}", file=sys.stderr)
    sys.exit(1)
graph_text = pathlib.Path(graph_path).read_text(encoding='utf-8', errors='replace')
g = re.search(
    r"total_ssd_cold=([0-9.]+) GiB\s+per-token-average: flops=([0-9.]+) GF weight=([0-9.]+) GiB act_rw=([0-9.]+) GiB kv_state=([0-9.]+) GiB",
    graph_text,
    re.MULTILINE,
)
if not g:
    print(f"failed to parse graph profile from {graph_path}", file=sys.stderr)
    sys.exit(1)
prefill_tps = float(m.group(1))
generation_tps = float(m.group(2))
prompt_tokens_i = int(prompt_tokens)
mem_bw = float(mem_bw_gibs)
compute_peak = float(compute_peak_tf)
ssd_bw = float(ssd_bw_gibs)
ssd_total = float(g.group(1))
flops_gf = float(g.group(2))
weight_gib = float(g.group(3))
act_rw_gib = float(g.group(4))
kv_state_gib = float(g.group(5))
total_bw_gib = weight_gib + act_rw_gib + kv_state_gib
ssd_per_token_gib = ssd_total / prompt_tokens_i if prompt_tokens_i > 0 else 0.0
prefill_bw_gibs = prefill_tps * total_bw_gib
prefill_bw_util_pct = (prefill_bw_gibs / mem_bw * 100.0) if mem_bw > 0 else 0.0
prefill_compute_tflops = prefill_tps * flops_gf / 1000.0
prefill_compute_util_pct = (prefill_compute_tflops / compute_peak * 100.0) if compute_peak > 0 else 0.0
prefill_ssd_gibs = prefill_tps * ssd_per_token_gib
prefill_ssd_util_pct = (prefill_ssd_gibs / ssd_bw * 100.0) if ssd_bw > 0 else 0.0
with open(out_csv, "a", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow([
        prompt_tokens_i,
        int(ctx_alloc),
        int(gen_tokens),
        prefill_tps,
        generation_tps,
        flops_gf,
        weight_gib,
        act_rw_gib,
        kv_state_gib,
        total_bw_gib,
        ssd_per_token_gib,
        prefill_bw_gibs,
        prefill_bw_util_pct,
        prefill_compute_tflops,
        prefill_compute_util_pct,
        prefill_ssd_gibs,
        prefill_ssd_util_pct,
        stderr_log,
        stdout_log,
    ])
PY

  printf "done prompt_tokens=%s ctx_alloc=%s\n" "$prompt_tokens" "$ctx_alloc"
done

echo "wrote $OUT_CSV"
echo "artifacts dir: $ART_DIR"
