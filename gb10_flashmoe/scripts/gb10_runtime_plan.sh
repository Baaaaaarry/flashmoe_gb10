#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-runtime-plan}"
MODEL_FAMILY="${1:?usage: gb10_runtime_plan.sh <qwen3.5-397b-a17b|deepseek-v4-flash>}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

"${BUILD_DIR}/flashmoe_runtime_plan" "${MODEL_FAMILY}"
