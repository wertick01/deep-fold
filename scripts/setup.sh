#!/usr/bin/env bash
# Neighbor install (Linux). Creates repo-local .venv with CUDA torch.
# Does not install the NVIDIA driver. No published Linux tok/s.
# If chr is missing, Python fetches portable Go 1.22 from go.dev.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
VENV="$REPO/.venv"
VENV_PY="$VENV/bin/python"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Creating venv at $VENV"
  PYTHON="${PYTHON:-python3}"
  "$PYTHON" -m venv "$VENV"
fi

echo "Using $VENV_PY"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install torch --index-url "$TORCH_INDEX"
"$VENV_PY" -m pip install -e ".[hub,chat]"

"$VENV_PY" -m gpu.cli setup --chr-only

"$VENV_PY" -m gpu.cli doctor
echo
echo "Next:"
echo "  source .venv/bin/activate"
echo "  deepfold pull Qwen/Qwen2.5-3B-Instruct --yes"
echo "  deepfold chat --model <that directory>"
if [[ -x "$VENV/bin/deepfold" ]]; then
  echo "Without activate: $VENV/bin/deepfold doctor"
else
  echo "WARN: .venv/bin/deepfold missing. Use: $VENV_PY -m gpu.cli doctor"
fi
echo "Do not expect the 3080 tok/s plate on another card."
