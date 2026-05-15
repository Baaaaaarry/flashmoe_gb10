#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 INPUT.csv [OUTPUT.svg]" >&2
  exit 2
fi

python3 "${SCRIPT_DIR}/plot_runtime_monitor.py" "$@"
