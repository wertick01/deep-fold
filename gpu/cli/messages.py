"""Refuse / diagnostic copy for the CLI. One module so tests assert wording.

GGUF, unknown architecture, `.chr` without a sidecar, no compiler, the 3B
speed line, CPU torch, wrong capability, macOS, missing ``chr``, CUDA OOM,
InternLM extras. Generate verdicts stay ASCII so a Windows console on a
non-UTF-8 code page prints them.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# wave7-ux.md §8
# --------------------------------------------------------------------------- #

GGUF = """\
Deepfold cannot load GGUF (including Ollama blobs under ~/.ollama).
chr compress reads HuggingFace BF16/FP16 safetensors only.
If this tag came from the Ollama library, retry:

  deepfold from-ollama <tag>

That command downloads the same HuggingFace id; it does not convert the GGUF."""


def unknown_ollama_tag(tag: str) -> str:
    """WAVE 10 P2 §3.4. Unknown library name, including ``llama3.1:8b``."""
    return (
        f"from-ollama: unknown tag {tag!r}.\n"
        "Deepfold maps allowlisted Ollama library names to HuggingFace ids and\n"
        "downloads BF16 safetensors. It never reads ~/.ollama and never loads GGUF.\n"
        "Allowlisted tags: qwen2.5:3b, qwen2.5:3b-instruct, qwen2.5:14b,\n"
        "qwen2.5:14b-instruct.\n"
        "Pass --hf <id> only for a row in that table\n"
        "(e.g. internlm/internlm2_5-20b-chat).\n"
        "Download the HuggingFace repo yourself, then:\n"
        "\n"
        "  deepfold run --model <that directory>"
    )


NEED_HUB = """\
deepfold pull / from-ollama needs huggingface_hub to download BF16 safetensors.
Install the extra:

  pip install "deepfold[hub]"

Or download the HuggingFace repo yourself, then:

  deepfold run --model <that directory>"""


def unknown_hf_id(hf_id: str, table: tuple[str, ...]) -> str:
    """Closed Hub lookup: not an allowlisted BF16 tree."""
    listed = ", ".join(table)
    return (
        f"pull: unknown HuggingFace id {hf_id!r}.\n"
        "Deepfold downloads an allowlisted BF16 tree, not arbitrary Hub repos.\n"
        f"Allowlisted ids: {listed}.\n"
        "This is not a general HuggingFace runtime."
    )


CHAT_NEED_TTY = (
    "deepfold chat needs a TTY. For scripts and pipes: deepfold run --prompt ..."
)

CHAT_NEED_TOOLKIT = """\
deepfold chat needs prompt_toolkit. Install the extra:

  pip install "deepfold[chat]"

Or use deepfold run --prompt."""

SETUP_REFUSE_TORCH_GPU = """\
deepfold setup refuses to pip-install into conda env torch-gpu
(that interpreter is the author's lab; a broken wheel there takes the kernel).
Create a neighbor venv instead:

  powershell -File scripts/setup.ps1
  bash scripts/setup.sh"""

CHAT_HELP = (
    "Enter sends. Ctrl+J newline. Ctrl+C stops a reply; "
    "at an empty prompt, twice to quit. "
    "Slash: /help /quit /exit /clear /stats /new /chats /copy /save /agent. "
    "History is JSON on disk; each turn prefills it (KV is not reused). "
    "Ctrl+C does not copy; use /copy or the terminal's copy (Ctrl+Shift+C). "
    "Agent mode (--agent or /agent on) can list/read/write the workspace and "
    "run pytest; writes and tests ask first. There is no general shell."
)

#: Kept for the doctor / catalog copy. It is *not* the generate authority any
#: more: after wave10 P1 that is `gpu.host.attach` (the walker) plus the
#: config-only pre-refuse in `gpu.graphs`.
SUPPORTED_MODEL_TYPES = ("qwen2", "internlm2", "llama", "mistral (no SWA)")


def unknown_arch(model_type: str | None, refusal: str = "") -> str:
    """The walker's own refusal, plus which glue families exist.

    Points at the missing **family id**, not at ``model_type``: after the walker
    ships, "supports qwen2 and internlm2 layouts only" is the wrong sentence --
    a Llama attaches and a Qwen2 with a vision tower does not
    (wave8-arch §2.3, wave10-product §2.3).
    """
    found = model_type if model_type else "<missing>"
    head = refusal.rstrip() + "\n\n" if refusal else ""
    return (
        f"{head}"
        f"config.json model_type={found}.\n"
        "TokenLoop implements two glue families: llama_swiglu (split q/k/v/o + "
        "SwiGLU,\n"
        "measured on Qwen2.5-3B/14B; also Llama 3.x without qk-norm and Mistral "
        "when its\n"
        "sliding window is off) and internlm_gqa (fused wqkv, measured on "
        "internlm2.5-20B).\n"
        "This is not a general GGUF/HuggingFace runtime."
    )


def deferred_arch(model_type: str | None) -> str:
    """Not a refusal: the config is off the measured list, so the walker decides.

    Printed to stderr by ``deepfold run`` before it compresses. The gate cannot
    say yes from ``config.json`` alone here, but ``model_type`` is also not
    evidence of *no*: the graph is what counts.
    """
    found = model_type if model_type else "<missing>"
    return (
        f"[warn] model_type={found} is not one of the measured layouts "
        "(qwen2, internlm2, llama, mistral).\n"
        "        Nothing in config.json rules it out, so attach() will walk the "
        "module tree\n"
        "        at load time and refuse if it is not llama_swiglu or "
        "internlm_gqa."
    )


CHR_WITHOUT_SIDECAR = """\
Need a HuggingFace directory (config.json + tokenizer) plus a .chr.
The .chr is weights only; it is not a full model file."""

NO_COMPILER = """\
NF4 kernel is not built. Install Visual Studio Build Tools (C++), or
build gpu/nf4 with vcvars64.bat. Jupyter is not required."""

NO_COMPILER_POSIX = """\
NF4 kernel is not built. Install the CUDA toolkit (nvcc) and a C++ compiler
(g++), or build it in place:

  python gpu/nf4/setup.py build_ext --inplace"""

THREE_B_SPEED = """\
On this 12 GB card both copies of 3B fit. Packed NF4 decode is now ahead of
the committed BF16 row; time to first token is still slower.
NF4 pays off when the 16-bit model does not fit (14B, 20B)."""


# --------------------------------------------------------------------------- #
# wave8-install.md §8
# --------------------------------------------------------------------------- #

CPU_TORCH = """\
PyTorch has no CUDA. Deepfold's kernel is Ampere-family CUDA (sm_80/86/89), not CPU.
Default "pip install torch" is often the CPU wheel.
Install a CUDA 12.4 wheel, then re-run doctor:

  pip install torch --index-url https://download.pytorch.org/whl/cu124

Deepfold does not install the NVIDIA driver."""

NO_TORCH = """\
PyTorch is not installed in this interpreter. Deepfold's kernel is Ampere
CUDA (sm_86 ship; sm_80/sm_89 experimental); the CUDA wheel is not on the default PyPI index:

  pip install torch --index-url https://download.pytorch.org/whl/cu124

Deepfold does not install the NVIDIA driver."""


def wrong_capability(capability: tuple[int, int] | None) -> str:
    """wave8-install.md §8, with the capability doctor actually saw."""
    sm = f"sm_{capability[0]}{capability[1]}" if capability else "unknown"
    return (
        f"This GPU is {sm}. The NF4 kernel ships as an Ampere-family fatbinary\n"
        f"(sm_80 / sm_86 / sm_89 + PTX compute_80).\n"
        "Measured machine: RTX 3080 (sm_86). Turing / Hopper / Blackwell are "
        "refused, not a silent fallback."
    )


MACOS_RUN = """\
deepfold run needs the Ampere CUDA kernel. There is no CUDA kernel on macOS.
chr compress on this Mac is supported; copy the .chr to a CUDA Ampere/Ada machine."""

MISSING_CHR = """\
chr (Go compressor) was not found. Deepfold does not pack weights in Python.
Run deepfold setup (fetches portable Go 1.22 from go.dev if needed),
put chr.exe on PATH, set DEEPFOLD_CHR_BIN, or from a checkout:

  go build -o chr.exe ./cmd/chr"""

MISSING_CHR_POSIX = """\
chr (Go compressor) was not found. Deepfold does not pack weights in Python.
Run deepfold setup (fetches portable Go 1.22 from go.dev if needed),
put chr on PATH, set DEEPFOLD_CHR_BIN, or from a checkout:

  go build -o chr ./cmd/chr"""


def missing_chr() -> str:
    import os

    return MISSING_CHR if os.name == "nt" else MISSING_CHR_POSIX


def cuda_oom(
    used_mib: int | None, total_mib: int | None, weight_mib: float | None
) -> str:
    """wave8-install.md §8. OOM is a recorded miss (wave8-runtime D9), not a crash."""
    used = str(used_mib) if used_mib is not None else "?"
    total = str(total_mib) if total_mib is not None else "12288"
    weights = f"{weight_mib:.0f}" if weight_mib is not None else "?"
    return (
        "CUDA OOM. Recorded miss, not a crash.\n"
        f"nvidia-smi: {used} / {total} MiB (this card is 12 GB). "
        f"Packed weights: {weights} MiB.\n"
        "Close other GPU apps or use a model whose NF4 working set fits "
        "(14B/20B are\n"
        "the reason to use NF4; 3B also fits BF16 on this card)."
    )


INTERNLM_EXTRAS = """\
internlm2 needs the InternLM extra: einops, and sentencepiece==0.1.99
(0.2.2 rejects InternLM's <0x00> pieces). Install it, then re-run:

  pip install "deepfold[internlm]"
  pip install einops "sentencepiece==0.1.99"

This is a package gate, not a notebook."""

COMPRESS_FAILED = "Deepfold did not load anything."


# --------------------------------------------------------------------------- #
# wave8-runtime.md §2.1 -- the one-line generate verdict
# --------------------------------------------------------------------------- #

GENERATE_SHIP = "generate: yes (ship, sm_86)"

GENERATE_EXPERIMENTAL = (
    "generate: experimental (Ampere-family, unmeasured; plate is RTX 3080 sm_86)"
)


def generate_unmeasured(capability: tuple[int, int]) -> str:
    sm = f"sm_{capability[0]}{capability[1]}"
    return (
        f"generate: experimental -- this GPU is {sm}; the plate is sm_86. "
        "Ampere-family fatbinary (sm_80/86/89 + PTX)."
    )


GENERATE_TURING = (
    "generate: no -- Turing sm_75 has no BF16 tensor cores and no cp.async. "
    "This GEMM is Ampere fused reconstruct+HMMA."
)

GENERATE_CPU = (
    "generate: no -- CPU only. chr compress / chr verify work. "
    "There is no CPU fused GEMM."
)

GENERATE_APPLE = (
    "generate: no -- Apple GPU is not CUDA. chr compress / verify on CPU. "
    "This runtime does not run on M2."
)

GENERATE_ROCM = (
    "generate: no -- ROCm is not implemented. NVIDIA CUDA Ampere-family only."
)


def generate_unsupported(capability: tuple[int, int]) -> str:
    sm = f"sm_{capability[0]}{capability[1]}"
    return (
        f"generate: no -- this GPU is {sm}; Ampere-family is sm_80/86/89. "
        "Hopper / Blackwell are refused (D1)."
    )


GENERATE_NO_TORCH = "generate: no -- torch is not importable; cannot probe the device."

DOCTOR_OK = "doctor: run is possible"
DOCTOR_COMPRESS_ONLY = "doctor: compress is possible, generate is not"
