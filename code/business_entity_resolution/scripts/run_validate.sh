#!/usr/bin/env bash
# run_validate.sh  –  Convenience wrapper around the submission validator.
#
# Usage:
#   bash scripts/run_validate.sh [ROOT]
#
# ROOT defaults to the student_resource/ directory (three levels up).

set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

python3 "${ROOT}/utils/validate_submission.py" \
    --matching "${ROOT}/output/matching_results.tsv" \
    --candidate "${ROOT}/output/candidate_pairs.tsv" \
    --test-dir  "${ROOT}/dataset/test"
