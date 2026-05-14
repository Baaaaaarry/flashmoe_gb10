#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PKG_DIR="${PYTHON_PKG_DIR:-${REPO_ROOT}/python}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"
INPUT_BIN="${1:?usage: gb10_train_predictor.sh routing_data.bin output.json}"
OUTPUT_JSON="${2:?usage: gb10_train_predictor.sh routing_data.bin output.json}"

source "$VENV_DIR/bin/activate"
PYTHONPATH="$PYTHON_PKG_DIR" \
  flashmoe-train-predictor \
  --input "$INPUT_BIN" \
  --output "$OUTPUT_JSON" \
  --epochs "${EPOCHS:-10}" \
  --lr "${LR:-1e-3}"
