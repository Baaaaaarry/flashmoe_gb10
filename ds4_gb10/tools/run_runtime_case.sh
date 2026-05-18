#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODEL.gguf OUT_PREFIX [extra ds4 args...]" >&2
  exit 2
fi

MODEL=$1
OUT_PREFIX=$2
shift 2

PROMPT_FILE=${DS4_RUNTIME_CASE_PROMPT:-tests/runtime_case_qa.txt}
TOKENS=${DS4_RUNTIME_CASE_TOKENS:-256}
CTX=${DS4_RUNTIME_CASE_CTX:-8192}
INTERVAL=${DS4_RUNTIME_MONITOR_INTERVAL:-1}

CSV="${OUT_PREFIX}.csv"
SVG="${OUT_PREFIX}.svg"

./ds4 \
  -m "$MODEL" \
  --cuda \
  --runtime-monitor \
  --runtime-monitor-interval "$INTERVAL" \
  --runtime-monitor-log "$CSV" \
  --ctx "$CTX" \
  --nothink \
  --temp 0 \
  -n "$TOKENS" \
  --prompt-file "$PROMPT_FILE" \
  "$@"

./tools/analyze_runtime_monitor.py "$CSV"
./tools/plot_runtime_monitor.sh "$CSV"

if [[ -f "${CSV%.csv}.svg" ]]; then
  mv "${CSV%.csv}.svg" "$SVG"
fi

echo "artifacts:"
echo "  csv: $CSV"
echo "  svg: $SVG"
