#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# build_submission.sh
# Packages submission bundle according to Amazon ML Challenge 2026 specs.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TEAM_NAME="${TEAM_NAME:-team}"
ZIP_NAME="${TEAM_NAME}_submission.zip"
ZIP_PATH="${REPO_ROOT}/${ZIP_NAME}"

MATCHING_FILE="output/matching_results.tsv"
CANDIDATE_FILE="output/candidate_pairs.tsv"
DOC_FILE="Documentation_template.md"
CODE_DIR="code/business_entity_resolution"

echo "=== Building submission package for team: ${TEAM_NAME} ==="

# 1. Validate required files exist
if [[ ! -f "${MATCHING_FILE}" ]]; then
    echo "ERROR: Missing required file '${MATCHING_FILE}'!" >&2
    echo "Please run the matching pipeline to produce '${MATCHING_FILE}' before packaging." >&2
    exit 1
fi

if [[ ! -f "${CANDIDATE_FILE}" ]]; then
    echo "ERROR: Missing required file '${CANDIDATE_FILE}'!" >&2
    echo "Please run candidate generation to produce '${CANDIDATE_FILE}' before packaging." >&2
    exit 1
fi

if [[ ! -f "${DOC_FILE}" ]]; then
    echo "ERROR: Missing documentation template '${DOC_FILE}'!" >&2
    exit 1
fi

if [[ ! -d "${CODE_DIR}" ]]; then
    echo "ERROR: Missing code directory '${CODE_DIR}'!" >&2
    exit 1
fi

TEST_DIR="${TEST_DIR:-dataset/test}"

# 2. Run validator
echo "Validating submission files with utils/validate_submission.py..."
PYTHON_BIN="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_BIN="python"
fi

VALIDATOR_OUTPUT=$("${PYTHON_BIN}" utils/validate_submission.py \
    --matching "${MATCHING_FILE}" \
    --candidate "${CANDIDATE_FILE}" \
    --test-dir "${TEST_DIR}" 2>&1 || true)

echo "${VALIDATOR_OUTPUT}"

if ! echo "${VALIDATOR_OUTPUT}" | grep -q "PASS"; then
    echo "ERROR: Submission validation did not pass! Aborting." >&2
    exit 1
fi

echo "Submission files validated successfully."

# 3. Create zip archive
echo "Packaging submission archive: ${ZIP_NAME}..."
rm -f "${ZIP_PATH}"

"${PYTHON_BIN}" -c "
import os
import sys
import zipfile
from pathlib import Path

repo_root = Path('.').resolve()
zip_name = '${ZIP_NAME}'
zip_path = repo_root / zip_name

files_to_pack = []

# output/
output_dir = repo_root / 'output'
if output_dir.is_dir():
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.tsv'):
                p = Path(root) / f
                files_to_pack.append((p, p.relative_to(repo_root)))

# Documentation_template.md
doc = repo_root / 'Documentation_template.md'
if doc.is_file():
    files_to_pack.append((doc, Path('Documentation_template.md')))

# code/business_entity_resolution/ (excluding __pycache__ and *.pyc/*.pyo)
code_dir = repo_root / 'code' / 'business_entity_resolution'
if code_dir.is_dir():
    for root, dirs, files in os.walk(code_dir):
        if '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.pyc') or f.endswith('.pyo'):
                continue
            p = Path(root) / f
            files_to_pack.append((p, p.relative_to(repo_root)))

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for src_path, arc_path in files_to_pack:
        zf.write(src_path, str(arc_path).replace('\\\\', '/'))

print('Packaging complete.')
"

# 4. Verify zip archive was created and report size
if [[ -f "${ZIP_PATH}" ]]; then
    ZIP_SIZE=$(wc -c < "${ZIP_PATH}" 2>/dev/null || stat -c%s "${ZIP_PATH}" 2>/dev/null || stat -f%z "${ZIP_PATH}" 2>/dev/null)
    echo "=== Submission package created successfully ==="
    echo "File: ${ZIP_PATH}"
    echo "Size: ${ZIP_SIZE} bytes"
else
    echo "ERROR: Failed to create ${ZIP_PATH}!" >&2
    exit 1
fi
