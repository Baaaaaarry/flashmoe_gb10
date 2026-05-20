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

printf "prompt_tokens,ctx_alloc,gen_tokens,prefill_tps,generation_tps,stderr_log,stdout_log\n" >"$OUT_CSV"

for prompt_tokens in "$@"; do
  ctx_alloc=$((prompt_tokens + GEN_TOKENS + 1 + CTX_MARGIN))
  run_name="p${prompt_tokens}"
  stderr_log="$ART_DIR/${run_name}.stderr.log"
  stdout_log="$ART_DIR/${run_name}.stdout.log"

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

  "${cmd[@]}" >"$stdout_log" 2>"$stderr_log"

  python3 - "$stderr_log" "$OUT_CSV" "$prompt_tokens" "$ctx_alloc" "$GEN_TOKENS" "$stderr_log" "$stdout_log" <<'PY'
import csv, pathlib, re, sys
stderr_path, out_csv, prompt_tokens, ctx_alloc, gen_tokens, stderr_log, stdout_log = sys.argv[1:]
text = pathlib.Path(stderr_path).read_text(encoding='utf-8', errors='replace')
m = re.search(r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s", text)
if not m:
    print(f"failed to parse throughput from {stderr_path}", file=sys.stderr)
    sys.exit(1)
with open(out_csv, "a", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow([
        int(prompt_tokens),
        int(ctx_alloc),
        int(gen_tokens),
        float(m.group(1)),
        float(m.group(2)),
        stderr_log,
        stdout_log,
    ])
PY

  printf "done prompt_tokens=%s ctx_alloc=%s\n" "$prompt_tokens" "$ctx_alloc"
done

echo "wrote $OUT_CSV"
echo "artifacts dir: $ART_DIR"
