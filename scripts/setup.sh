#!/usr/bin/env bash
# Neighbor install (Linux). Creates repo-local .venv with CUDA torch.
# Does not install the NVIDIA driver. No published Linux tok/s.
# If chr is missing, Python fetches portable Go 1.22 from go.dev.
# Compiles gpu/nf4 when g++ and nvcc are already on PATH (no sudo apt).
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "macOS: compress may work; generate needs NVIDIA CUDA. Use Linux or Windows." >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
VENV="$REPO/.venv"
VENV_PY="$VENV/bin/python"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

py_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1
}

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if py_ok "$PYTHON"; then
      printf '%s\n' "$PYTHON"
      return 0
    fi
    echo "PYTHON=$PYTHON is not Python 3.11+." >&2
    exit 1
  fi
  local c
  for c in python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && py_ok "$c"; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  echo "Need Python 3.11 or 3.12 (Ubuntu 22.04: python3 is 3.10; install python3.11 and python3.11-venv)." >&2
  exit 1
}

if [[ ! -x "$VENV_PY" ]]; then
  PY="$(pick_python)"
  echo "Creating venv at $VENV with $PY"
  "$PY" -m venv "$VENV"
fi

if ! py_ok "$VENV_PY"; then
  echo "$VENV_PY is not Python 3.11+. Delete .venv and re-run." >&2
  exit 1
fi

echo "Using $VENV_PY"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install torch --index-url "$TORCH_INDEX"
"$VENV_PY" -m pip install -e ".[hub,chat]"
"$VENV_PY" -m pip install ninja

"$VENV_PY" -m gpu.cli setup --chr-only
"$VENV_PY" -m gpu.cli setup --kernel-only

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
