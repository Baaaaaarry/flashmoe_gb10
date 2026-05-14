#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB10_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"

MODEL_SPEC="${1:?usage: gb10_export_qwen_experts_chunked.sh /local/model/or/hf_repo_id /output/expert_dir [cache_dir]}"
OUTPUT_DIR="${2:?usage: gb10_export_qwen_experts_chunked.sh /local/model/or/hf_repo_id /output/expert_dir [cache_dir]}"
CACHE_DIR="${3:-}"
LAYER_START_DEFAULT="${LAYER_START_DEFAULT:-0}"
LAYER_END_DEFAULT="${LAYER_END_DEFAULT:-61}"

source "${VENV_DIR}/bin/activate"

for (( layer=LAYER_START_DEFAULT; layer<=LAYER_END_DEFAULT; layer++ )); do
  if compgen -G "${OUTPUT_DIR}/layer_$(printf "%03d" "${layer}")_expert_*.safetensors" > /dev/null; then
    echo "skip layer ${layer}: already exported"
    continue
  fi

  echo "export layer ${layer}"
  LAYER_START="${layer}" LAYER_END="${layer}" \
    "${GB10_ROOT}/scripts/gb10_export_qwen_experts.sh" \
    "${MODEL_SPEC}" \
    "${OUTPUT_DIR}" \
    "${CACHE_DIR}"
done
