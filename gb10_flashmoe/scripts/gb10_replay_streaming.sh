#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/flashmoe-gb10}"

CONFIG_PATH="${1:?usage: gb10_replay_streaming.sh /path/to/flashmoe_config.json routing_trace.txt}"
TRACE_PATH="${2:?usage: gb10_replay_streaming.sh /path/to/flashmoe_config.json routing_trace.txt}"

source "${VENV_DIR}/bin/activate"

flashmoe-replay-streaming \
  --config "${CONFIG_PATH}" \
  --trace "${TRACE_PATH}"
