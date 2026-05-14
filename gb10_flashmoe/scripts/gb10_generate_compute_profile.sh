#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PKG_DIR="${PYTHON_PKG_DIR:-${REPO_ROOT}/python}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"

OUT_PATH="${1:?usage: gb10_generate_compute_profile.sh <output.json> [tokens] [hidden] [intermediate] [experts] [topk] [dtype] [iters] [active_experts_csv]}"
TOKENS="${2:-1}"
HIDDEN="${3:-4096}"
INTERMEDIATE="${4:-1536}"
EXPERTS="${5:-64}"
TOPK="${6:-4}"
DTYPE="${7:-bfloat16}"
ITERS="${8:-200}"
ACTIVE_EXPERTS_CSV="${9:-}"

source "$VENV_DIR/bin/activate"

ARGS=(
  "--backend" "flashmoe"
  "--tokens" "${TOKENS}"
  "--hidden" "${HIDDEN}"
  "--intermediate" "${INTERMEDIATE}"
  "--experts" "${EXPERTS}"
  "--topk" "${TOPK}"
  "--dtype" "${DTYPE}"
  "--iters" "${ITERS}"
  "--emit-compute-profile" "${OUT_PATH}"
)

if [[ -n "${ACTIVE_EXPERTS_CSV}" ]]; then
  ARGS+=("--profile-active-experts" "${ACTIVE_EXPERTS_CSV}")
fi

PYTHONPATH="$PYTHON_PKG_DIR" \
  python "$PYTHON_PKG_DIR/benchmark_fused_moe.py" "${ARGS[@]}"
