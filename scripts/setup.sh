#!/usr/bin/env bash
# Neighbor install (Linux). Creates repo-local .venv with CUDA torch.
# Does not install the NVIDIA driver. No published Linux tok/s.
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

if command -v go >/dev/null 2>&1; then
  go build -o chr ./cmd/chr
else
  echo "Go not on PATH: skip chr build. Install Go 1.22+ and re-run, or set DEEPFOLD_CHR_BIN."
fi

"$VENV_PY" -m gpu.cli doctor
echo
echo "Next:"
echo "  source .venv/bin/activate"
echo "  deepfold pull Qwen/Qwen2.5-3B-Instruct --yes"
echo "  deepfold chat --model <that directory>"
echo "Do not expect the 3080 tok/s plate on another card."
