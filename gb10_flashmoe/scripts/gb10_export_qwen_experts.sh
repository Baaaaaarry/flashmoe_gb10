#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB10_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"

MODEL_SPEC="${1:?usage: gb10_export_qwen_experts.sh /local/model/or/hf_repo_id /output/expert_dir [cache_dir]}"
OUTPUT_DIR="${2:?usage: gb10_export_qwen_experts.sh /local/model/or/hf_repo_id /output/expert_dir [cache_dir]}"
CACHE_DIR="${3:-}"

source "${VENV_DIR}/bin/activate"

ARGS=(
  --model "${MODEL_SPEC}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  ARGS+=(--cache-dir "${CACHE_DIR}")
fi
ARGS+=(--export-format "${EXPORT_FORMAT:-dense}")
ARGS+=(--export-dtype "${EXPORT_DTYPE:-bf16}")

if [[ -n "${LAYER_START:-}" ]]; then
  ARGS+=(--layer-start "${LAYER_START}")
fi
if [[ -n "${LAYER_END:-}" ]]; then
  ARGS+=(--layer-end "${LAYER_END}")
fi

python "${GB10_ROOT}/python/flashmoe_vllm_plugin/export_qwen_experts.py" "${ARGS[@]}"
