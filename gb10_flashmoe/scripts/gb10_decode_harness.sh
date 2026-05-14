#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-decode-harness}"

MODEL_FAMILY="${1:?usage: gb10_decode_harness.sh <model_family> <expert_manifest.json> <routing_trace.txt> [max_steps] [compute_profile.json]}"
MANIFEST_PATH="${2:?usage: gb10_decode_harness.sh <model_family> <expert_manifest.json> <routing_trace.txt> [max_steps] [compute_profile.json]}"
TRACE_PATH="${3:?usage: gb10_decode_harness.sh <model_family> <expert_manifest.json> <routing_trace.txt> [max_steps] [compute_profile.json]}"
MAX_STEPS="${4:-}"
COMPUTE_PROFILE="${5:-}"

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

"${BUILD_DIR}/flashmoe_decode_harness" "${ARGS[@]}"
