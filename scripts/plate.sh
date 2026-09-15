#!/usr/bin/env bash
# Neighbor plate (Linux). One argument is the model; the rest go to
# python -m gpu.cli.plate. Creates a folder of metrics to send back.
#
#   bash scripts/plate.sh 3b
#   bash scripts/plate.sh Qwen/Qwen2.5-3B-Instruct
#   bash scripts/plate.sh 32b --skip-verify
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

exec "$PY" -m gpu.cli.plate "$@"
