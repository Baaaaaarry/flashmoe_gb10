#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PKG_DIR="${PYTHON_PKG_DIR:-${REPO_ROOT}/python}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
export MAX_JOBS="${MAX_JOBS:-8}"

require_cmd() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: missing required command: $cmd"
    echo "Install hint: $hint"
    exit 1
  fi
}

check_python_headers() {
  python - <<'PY'
import pathlib
import sys
import sysconfig

include_dir = pathlib.Path(sysconfig.get_paths()["include"])
python_h = include_dir / "Python.h"
if python_h.exists():
    print(f"Found Python headers: {python_h}")
    raise SystemExit(0)

ver = f"{sys.version_info.major}.{sys.version_info.minor}"
print("ERROR: Python development headers not found.")
print(f"Expected header: {python_h}")
print("Install one of:")
print(f"  sudo apt-get install -y python{ver}-dev")
print("or")
print("  sudo apt-get install -y python3-dev")
raise SystemExit(1)
PY
}

source "$VENV_DIR/bin/activate"
require_cmd g++ "sudo apt-get install -y build-essential"
require_cmd nvcc "Install NVIDIA CUDA toolkit / use the DGX Spark CUDA image"
require_cmd cmake "sudo apt-get install -y cmake"
require_cmd ninja "sudo apt-get install -y ninja-build"
check_python_headers
PYTHONPATH="$PYTHON_PKG_DIR" \
  python "$PYTHON_PKG_DIR/build_cuda_extension.py"
