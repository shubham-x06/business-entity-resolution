#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/code/business_entity_resolution/src:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/scripts/run_train.py" --stage all "$@"
