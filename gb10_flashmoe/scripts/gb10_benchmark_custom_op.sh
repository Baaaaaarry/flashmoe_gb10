#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PKG_DIR="${PYTHON_PKG_DIR:-${REPO_ROOT}/python}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"
source "$VENV_DIR/bin/activate"

PYTHONPATH="$PYTHON_PKG_DIR" \
  python "$PYTHON_PKG_DIR/benchmark_fused_moe.py" "$@"
