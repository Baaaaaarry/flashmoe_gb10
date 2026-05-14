#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB10_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"

MODEL_PATH="${1:?usage: gb10_init_config.sh /path/to/model /path/to/expert_dir /path/to/flashmoe_config.json}"
EXPERT_PATH="${2:?usage: gb10_init_config.sh /path/to/model /path/to/expert_dir /path/to/flashmoe_config.json}"
OUTPUT_PATH="${3:?usage: gb10_init_config.sh /path/to/model /path/to/expert_dir /path/to/flashmoe_config.json}"

source "${VENV_DIR}/bin/activate"

python "${GB10_ROOT}/python/flashmoe_vllm_plugin/init_config.py" \
  --model-path "${MODEL_PATH}" \
  --expert-path "${EXPERT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --expert-format "${EXPERT_FORMAT:-dense}"
