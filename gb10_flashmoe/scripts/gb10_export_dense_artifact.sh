#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_DIR="${REPO_ROOT}/python"

MODEL="${1:?usage: gb10_export_dense_artifact.sh <model_path_or_repo_id> <output_dir> [cache_dir]}"
OUTPUT_DIR="${2:?usage: gb10_export_dense_artifact.sh <model_path_or_repo_id> <output_dir> [cache_dir]}"
CACHE_DIR="${3:-}"

ARGS=(
  --model "${MODEL}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${CACHE_DIR}" ]]; then
  ARGS+=(--cache-dir "${CACHE_DIR}")
fi
if [[ -n "${DENSE_SAMPLE_STRIDE:-}" ]]; then
  ARGS+=(--sample-stride "${DENSE_SAMPLE_STRIDE}")
fi
if [[ -n "${DENSE_MAX_SAMPLED_DIMS:-}" ]]; then
  ARGS+=(--max-sampled-dims "${DENSE_MAX_SAMPLED_DIMS}")
fi
if [[ -n "${DENSE_RUNTIME_VOCAB_SIZE:-}" ]]; then
  ARGS+=(--runtime-vocab-size "${DENSE_RUNTIME_VOCAB_SIZE}")
fi

python3 "${PYTHON_DIR}/flashmoe_vllm_plugin/export_dense_runtime_artifact.py" "${ARGS[@]}"
