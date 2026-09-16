#!/usr/bin/env bash
#
# check-image-sizes.sh
#
# Fails if any newly added or modified image file in the current diff
# exceeds the configured size threshold.
#
# Usage:
#   scripts/check-image-sizes.sh [BASE_REF]
#
# Environment variables:
#   MAX_IMAGE_SIZE_KB   Maximum allowed size in kilobytes (default: 500)
#   BASE_REF            Base git ref to diff against (default: origin/main)
#
# Exit codes:
#   0  All added/modified images are within the size limit (or none found).
#   1  One or more added/modified images exceed the size limit.
#   2  Script usage/environment error.

set -euo pipefail

MAX_IMAGE_SIZE_KB="${MAX_IMAGE_SIZE_KB:-500}"
BASE_REF="${1:-${BASE_REF:-origin/main}}"

# Convert KB threshold to bytes.
MAX_IMAGE_SIZE_BYTES=$((MAX_IMAGE_SIZE_KB * 1024))

# Image extensions to check (case-insensitive).
IMAGE_EXTENSIONS_REGEX='\.(png|jpg|jpeg|gif|webp|svg)$'

echo "Checking added/modified image files against ${MAX_IMAGE_SIZE_KB} KB limit..."
echo "Base ref: ${BASE_REF}"

# Verify the base ref exists so we fail loudly rather than silently passing.
if ! git rev-parse --verify --quiet "${BASE_REF}" >/dev/null; then
  echo "ERROR: Base ref '${BASE_REF}' not found. Ensure the repository is fetched with sufficient history." >&2
  exit 2
fi

# Collect added/modified files (A=added, M=modified) between BASE_REF and HEAD.
# Use -z to safely handle filenames with spaces/newlines.
mapfile -d '' -t CHANGED_FILES < <(
  git diff --name-only --diff-filter=AM -z "${BASE_REF}...HEAD" || true
)

if [ "${#CHANGED_FILES[@]}" -eq 0 ]; then
  echo "No added or modified files in diff. Nothing to check."
  exit 0
fi

FAILED=0
CHECKED=0

for file in "${CHANGED_FILES[@]}"; do
  # Skip empty entries (defensive).
  [ -z "${file}" ] && continue

  # Filter for image extensions (case-insensitive).
  if ! printf '%s' "${file}" | grep -Eiq "${IMAGE_EXTENSIONS_REGEX}"; then
    continue
  fi

  # Skip files that no longer exist in the working tree (e.g., deleted after rename).
  if [ ! -f "${file}" ]; then
    continue
  fi

  # Prefer git's blob size for the file at HEAD; fall back to stat.
  size_bytes="$(git cat-file -s "HEAD:${file}" 2>/dev/null || true)"
  if [ -z "${size_bytes}" ]; then
    size_bytes="$(stat -c '%s' "${file}" 2>/dev/null || stat -f '%z' "${file}" 2>/dev/null || echo 0)"
  fi

  CHECKED=$((CHECKED + 1))

  if [ "${size_bytes}" -gt "${MAX_IMAGE_SIZE_BYTES}" ]; then
    size_kb=$(( (size_bytes + 1023) / 1024 ))
    printf 'FAIL: %s is %s KB (limit: %s KB)\n' "${file}" "${size_kb}" "${MAX_IMAGE_SIZE_KB}" >&2
    FAILED=1
  else
    size_kb=$(( (size_bytes + 1023) / 1024 ))
    printf 'OK:   %s is %s KB\n' "${file}" "${size_kb}"
  fi
done

if [ "${CHECKED}" -eq 0 ]; then
  echo "No added or modified image files detected. Nothing to check."
  exit 0
fi

if [ "${FAILED}" -ne 0 ]; then
  echo "" >&2
  echo "One or more image files exceed the ${MAX_IMAGE_SIZE_KB} KB limit." >&2
  echo "Please optimize or compress the offending image(s) before merging." >&2
  exit 1
fi

echo ""
echo "All ${CHECKED} added/modified image file(s) are within the ${MAX_IMAGE_SIZE_KB} KB limit."
exit 0
