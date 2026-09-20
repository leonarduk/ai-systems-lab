#!/usr/bin/env bash
#
# Verify that every MCP server under
# projects/01-mcp-server-suite/servers/*/requirements.txt pins mcp<2.0.0.
#
# Usage:
#   check_mcp_pin.sh [ROOT_DIR]
#
# ROOT_DIR defaults to the current working directory. The script looks for
# server requirements files under
#   "$ROOT_DIR/projects/01-mcp-server-suite/servers/*/requirements.txt"
#
# Exit codes:
#   0  every discovered requirements.txt contains the pin
#   1  at least one requirements.txt is missing the pin, OR no requirements
#      files were discovered (a vacuous pass is treated as a failure so the
#      guard cannot silently degrade into a no-op)
#
# This script is invoked both by the PR-triggered CI step and by the
# workflow_dispatch self-test, so the self-test exercises the real code path
# rather than a copy.

set -euo pipefail

ROOT_DIR="${1:-.}"
SERVERS_GLOB="$ROOT_DIR/projects/01-mcp-server-suite/servers"

shopt -s nullglob
requirements_files=("$SERVERS_GLOB"/*/requirements.txt)
shopt -u nullglob

if [ "${#requirements_files[@]}" -eq 0 ]; then
  echo "::error::FAIL: no requirements.txt files found under $SERVERS_GLOB/*/requirements.txt"
  exit 1
fi

status=0
for req in "${requirements_files[@]}"; do
  # Anchored to the start of the line so a comment merely mentioning
  # "mcp<2.0.0" (e.g. explaining why the line below needs it) can't
  # satisfy the check — only an actual requirement line can.
  if grep -qE '^mcp.*<2\.0\.0' "$req"; then
    echo "OK: $req pins mcp<2.0.0"
  else
    echo "::error::FAIL: $req is missing the mcp<2.0.0 pin"
    status=1
  fi
done

exit "$status"
