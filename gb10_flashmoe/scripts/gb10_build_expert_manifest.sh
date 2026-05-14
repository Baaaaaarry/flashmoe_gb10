#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"

EXPERT_ROOT="${1:?usage: gb10_build_expert_manifest.sh /path/to/expert/files output_manifest.json}"
OUTPUT_JSON="${2:?usage: gb10_build_expert_manifest.sh /path/to/expert/files output_manifest.json}"

source "${VENV_DIR}/bin/activate"

flashmoe-build-expert-manifest \
  --root "${EXPERT_ROOT}" \
  --output "${OUTPUT_JSON}"
