#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB10_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"

MANIFEST_PATH="${1:?usage: gb10_generate_synthetic_trace.sh expert_manifest.json output_trace.txt}"
OUTPUT_TRACE="${2:?usage: gb10_generate_synthetic_trace.sh expert_manifest.json output_trace.txt}"

source "${VENV_DIR}/bin/activate"

python "${GB10_ROOT}/python/flashmoe_vllm_plugin/generate_synthetic_trace.py" \
  --manifest "${MANIFEST_PATH}" \
  --output "${OUTPUT_TRACE}" \
  --steps "${TRACE_STEPS:-256}" \
  --top-k "${TRACE_TOPK:-4}" \
  --locality "${TRACE_LOCALITY:-0.85}" \
  --seed "${TRACE_SEED:-7}"
