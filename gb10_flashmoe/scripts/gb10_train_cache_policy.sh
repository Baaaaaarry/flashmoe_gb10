#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PKG_DIR="${PYTHON_PKG_DIR:-${REPO_ROOT}/python}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"
TRACE_TXT="${1:?usage: gb10_train_cache_policy.sh routing_trace.txt cache_policy.json}"
OUTPUT_JSON="${2:?usage: gb10_train_cache_policy.sh routing_trace.txt cache_policy.json}"
TMP_CSV="${TMP_CSV:-/tmp/flashmoe_cache_policy_train.csv}"

source "$VENV_DIR/bin/activate"
PYTHONPATH="$PYTHON_PKG_DIR" \
  flashmoe-build-cache-dataset \
  --trace "$TRACE_TXT" \
  --output "$TMP_CSV" \
  --cache-entries "${CACHE_ENTRIES:-16384}"

PYTHONPATH="$PYTHON_PKG_DIR" \
  flashmoe-train-policy \
  --input "$TMP_CSV" \
  --output "$OUTPUT_JSON" \
  --epochs "${EPOCHS:-20}" \
  --hidden-dim "${HIDDEN_DIM:-32}" \
  --lr "${LR:-1e-3}"
