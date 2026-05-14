#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-streamed-runtime}"

MODEL_FAMILY="${1:?usage: gb10_streamed_runtime.sh <model_family> <expert_manifest.json> <routing_trace.txt> [max_steps] [compute_profile.json] [cache_budget_gb]}"
MANIFEST_PATH="${2:?usage: gb10_streamed_runtime.sh <model_family> <expert_manifest.json> <routing_trace.txt> [max_steps] [compute_profile.json] [cache_budget_gb]}"
TRACE_PATH="${3:?usage: gb10_streamed_runtime.sh <model_family> <expert_manifest.json> <routing_trace.txt> [max_steps] [compute_profile.json] [cache_budget_gb]}"
MAX_STEPS="${4:-}"
COMPUTE_PROFILE="${5:-}"
CACHE_BUDGET_GB="${6:-}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

ARGS=("${MODEL_FAMILY}" "${MANIFEST_PATH}" "${TRACE_PATH}")
if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=("${MAX_STEPS}")
fi
if [[ -n "${COMPUTE_PROFILE}" ]]; then
  if [[ -z "${MAX_STEPS}" ]]; then
    ARGS+=("0")
  fi
  ARGS+=("${COMPUTE_PROFILE}")
fi
if [[ -n "${CACHE_BUDGET_GB}" ]]; then
  if [[ -z "${MAX_STEPS}" ]]; then
    ARGS+=("0")
  fi
  if [[ -z "${COMPUTE_PROFILE}" ]]; then
    echo "cache_budget_gb requires compute_profile.json position; pass a real compute profile path before cache budget" >&2
    exit 1
  fi
  ARGS+=("${CACHE_BUDGET_GB}")
fi

"${BUILD_DIR}/flashmoe_streamed_runtime" "${ARGS[@]}"
