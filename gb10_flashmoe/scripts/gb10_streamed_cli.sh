#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-streamed-service}"

MODEL_FAMILY="${1:?usage: gb10_streamed_cli.sh <model_family> <expert_manifest.json> <routing_trace.txt|''> [compute_profile.json] [cache_budget_gb] [extra cli args...]}"
MANIFEST_PATH="${2:?usage: gb10_streamed_cli.sh <model_family> <expert_manifest.json> <routing_trace.txt|''> [compute_profile.json] [cache_budget_gb] [extra cli args...]}"
TRACE_PATH="${3:-}"
COMPUTE_PROFILE="${4:-}"
CACHE_BUDGET_GB="${5:-}"
shift $(( $# >= 5 ? 5 : $# ))
EXTRA_ARGS=("$@")

USE_RUNTIME_ROUTER="${USE_RUNTIME_ROUTER:-0}"
USE_CPU_EXPERT_BACKEND="${USE_CPU_EXPERT_BACKEND:-0}"
MODEL_PATH="${MODEL_PATH:-}"
DENSE_ARTIFACT_PATH="${DENSE_ARTIFACT_PATH:-}"
TOKENIZER_ARTIFACT_PATH="${TOKENIZER_ARTIFACT_PATH:-}"
FLASHMOE_PROGRESS="${FLASHMOE_PROGRESS:-1}"
FLASHMOE_PROGRESS_EVERY="${FLASHMOE_PROGRESS_EVERY:-1}"
EXECUTION_BACKEND="${EXECUTION_BACKEND:-streamed}"

if [[ "${EXECUTION_BACKEND}" == "official_hf" ]]; then
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required when EXECUTION_BACKEND=official_hf" >&2
    exit 1
  fi
  PYTHON_DIR="${REPO_ROOT}/python"
  python3 "${PYTHON_DIR}/flashmoe_vllm_plugin/reference_cli.py" \
    --model "${MODEL_PATH}" \
    "${EXTRA_ARGS[@]}"
  exit 0
fi

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

ARGS=("${MODEL_FAMILY}" "${MANIFEST_PATH}")
if [[ "${USE_RUNTIME_ROUTER}" == "1" ]]; then
  ARGS+=("")
else
  ARGS+=("${TRACE_PATH}")
fi
if [[ -n "${COMPUTE_PROFILE}" ]]; then
  ARGS+=("${COMPUTE_PROFILE}")
fi
if [[ -n "${CACHE_BUDGET_GB}" ]]; then
  if [[ -z "${COMPUTE_PROFILE}" ]]; then
    ARGS+=("")
  fi
  ARGS+=("${CACHE_BUDGET_GB}")
fi
if [[ "${USE_RUNTIME_ROUTER}" == "1" ]]; then
  ARGS+=("--runtime-router")
fi
if [[ "${USE_CPU_EXPERT_BACKEND}" == "1" ]]; then
  ARGS+=("--cpu-expert-backend")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=("--model-path" "${MODEL_PATH}")
fi
if [[ -n "${DENSE_ARTIFACT_PATH}" ]]; then
  ARGS+=("--dense-artifact" "${DENSE_ARTIFACT_PATH}")
fi
if [[ -n "${TOKENIZER_ARTIFACT_PATH}" ]]; then
  ARGS+=("--tokenizer-artifact" "${TOKENIZER_ARTIFACT_PATH}")
fi
ARGS+=("${EXTRA_ARGS[@]}")

FLASHMOE_PROGRESS="${FLASHMOE_PROGRESS}" \
FLASHMOE_PROGRESS_EVERY="${FLASHMOE_PROGRESS_EVERY}" \
"${BUILD_DIR}/flashmoe_streamed_cli" "${ARGS[@]}"
