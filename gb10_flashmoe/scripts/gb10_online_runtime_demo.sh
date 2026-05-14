#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-online-runtime}"

MODEL_FAMILY="${1:?usage: gb10_online_runtime_demo.sh <model_family> <expert_manifest.json> <routing_trace.txt> [cache_gb] [max_steps]}"
MANIFEST_PATH="${2:?usage: gb10_online_runtime_demo.sh <model_family> <expert_manifest.json> <routing_trace.txt> [cache_gb] [max_steps]}"
TRACE_PATH="${3:?usage: gb10_online_runtime_demo.sh <model_family> <expert_manifest.json> <routing_trace.txt> [cache_gb] [max_steps]}"
CACHE_GB="${4:-}"
MAX_STEPS="${5:-}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

ARGS=("${MODEL_FAMILY}" "${MANIFEST_PATH}" "${TRACE_PATH}")
if [[ -n "${CACHE_GB}" ]]; then
  ARGS+=("${CACHE_GB}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  if [[ -z "${CACHE_GB}" ]]; then
    ARGS+=("-1")
  fi
  ARGS+=("${MAX_STEPS}")
fi

"${BUILD_DIR}/flashmoe_online_runtime_demo" "${ARGS[@]}"
