#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Deprecated two-stage launcher: running joint Phase1 SRT instead." >&2
exec bash "${SCRIPT_DIR}/run_phase1_srt_4gpu.sh" "$@"
