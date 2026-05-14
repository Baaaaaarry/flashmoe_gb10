#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PKG_DIR="${PYTHON_PKG_DIR:-${REPO_ROOT}/python}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"
CONFIG_JSON="${1:?usage: gb10_serve_vllm.sh flashmoe_config.json [extra vllm args...]}"
shift || true

source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PYTHON_PKG_DIR}:${PYTHONPATH:-}"

python "$PYTHON_PKG_DIR/serve_with_vllm.py" \
  --config "$CONFIG_JSON" \
  "$@"
