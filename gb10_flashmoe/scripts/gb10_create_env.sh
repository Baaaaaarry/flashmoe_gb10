#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PKG_DIR="${PYTHON_PKG_DIR:-${REPO_ROOT}/python}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"
VLLM_REF="${VLLM_REF:-main}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_RETRIES="${PIP_RETRIES:-8}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
VLLM_INSTALL_MODE="${VLLM_INSTALL_MODE:-wheel}"
VLLM_SOURCE="${VLLM_SOURCE:-pypi}"
FLASHMOE_INSTALL_MODE="${FLASHMOE_INSTALL_MODE:-wheel}"

pip_install() {
  python -m pip --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" install "$@"
}

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
print("On Ubuntu/Debian install one of:")
print(f"  sudo apt-get update && sudo apt-get install -y python{ver}-dev")
print("or")
print("  sudo apt-get update && sudo apt-get install -y python3-dev")
raise SystemExit(1)
PY
}

preflight_system() {
  require_cmd git "sudo apt-get install -y git"
  require_cmd g++ "sudo apt-get install -y build-essential"
  require_cmd cmake "sudo apt-get install -y cmake"
  require_cmd ninja "sudo apt-get install -y ninja-build"
  check_python_headers
}

patch_vllm_pyproject_license() {
  local pyproject="$HOME/src/vllm/pyproject.toml"
  if [[ ! -f "$pyproject" ]]; then
    return 0
  fi
  python - "$pyproject" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
orig = text
text = text.replace('license = "Apache-2.0"', 'license = { text = "Apache-2.0" }')
text = text.replace('license-files = ["LICENSE"]\n', '')
if text != orig:
    path.write_text(text)
    print(f"Patched {path} project metadata for setuptools compatibility")
else:
    print(f"No vLLM pyproject patch needed for {path}")
PY
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
preflight_system
pip_install --upgrade pip wheel "setuptools==77.0.3"
pip_install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
pip_install ninja packaging cmake setuptools-scm jinja2 pybind11 huggingface_hub safetensors

if [[ "$VLLM_SOURCE" == "source" ]]; then
  if [[ ! -d "$HOME/src/vllm" ]]; then
    git clone https://github.com/vllm-project/vllm.git "$HOME/src/vllm"
  fi
  git -C "$HOME/src/vllm" fetch origin
  git -C "$HOME/src/vllm" checkout "$VLLM_REF"
  patch_vllm_pyproject_license
  if [[ "$VLLM_INSTALL_MODE" == "editable" ]]; then
    pip_install --no-build-isolation -e "$HOME/src/vllm"
  else
    pip_install --no-build-isolation "$HOME/src/vllm"
  fi
else
  pip_install vllm
fi

if [[ "$FLASHMOE_INSTALL_MODE" == "editable" ]]; then
  pip_install --no-build-isolation -e "$PYTHON_PKG_DIR"
else
  pip_install --no-build-isolation "$PYTHON_PKG_DIR"
fi

echo "Environment ready at $VENV_DIR"
