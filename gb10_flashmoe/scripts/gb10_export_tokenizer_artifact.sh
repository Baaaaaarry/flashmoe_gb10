#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_DIR="${REPO_ROOT}/python"

MODEL="${1:?usage: gb10_export_tokenizer_artifact.sh <model_path_or_repo_id> <output_json> [cache_dir]}"
OUTPUT="${2:?usage: gb10_export_tokenizer_artifact.sh <model_path_or_repo_id> <output_json> [cache_dir]}"
CACHE_DIR="${3:-}"

ARGS=(
  --model "${MODEL}"
  --output "${OUTPUT}"
)
if [[ -n "${CACHE_DIR}" ]]; then
  ARGS+=(--cache-dir "${CACHE_DIR}")
fi
if [[ -n "${TOKENIZER_RUNTIME_VOCAB_SIZE:-}" ]]; then
  ARGS+=(--runtime-vocab-size "${TOKENIZER_RUNTIME_VOCAB_SIZE}")
fi

python3 "${PYTHON_DIR}/flashmoe_vllm_plugin/export_tokenizer_artifact.py" "${ARGS[@]}"
