"""``deepfold`` command line: doctor, compress, run.

Without installing: ``python -m gpu.cli doctor``,
``python -m gpu.cli run --model DIR``, and ``python -m gpu.cli from-ollama``.
After ``pip install -e .`` the same entry point is ``deepfold`` on PATH.

Generate is Ampere ``sm_86`` only; everything else refuses with a reason.
GGUF is refused (the path is never opened). A sibling ``.chr`` is used only
when the CHR0 header matches this model's ``config.json``.

Two things are always needed at generate time: a HuggingFace **directory**
for ``config.json``, the tokenizer and (InternLM2) its remote-code Python,
and one **`.chr``** of packed NF4 weights. The safetensor shards are opened
once, by ``chr compress``, and never again. Wrapped around the Go ``chr``
compressor, ``gpu.host.load_model`` and ``gpu.loop.TokenLoop``.
"""

from __future__ import annotations

from .arch import Gate, gate
from .doctor import Machine, Verdict, checks, exit_code, probe, verdict
from .main import build_parser, main

__all__ = [
    "main",
    "build_parser",
    "Machine",
    "Verdict",
    "probe",
    "verdict",
    "checks",
    "exit_code",
    "Gate",
    "gate",
]
