"""Pick NF4 vs VQ from the model's config and the card's VRAM.

No torch. The Go compressor still packs; this module only answers *which*
``--codec`` to pass it. Default is ``auto``: NF4 when the packed file plus a
fixed runtime overhead still sits on the card; otherwise NF4 with
``overflow=True`` (H2 ring) when 2-bit VQ would have fit. It never picks VQ:
the 3B canary (``qwen25-3b.vq2.chr``) failed greedy Paris/Berlin/323 because
``gate_proj`` alone at 2×8 residual k-means collapses SwiGLU (rel_mse ~0.12
vs NF4 ~0.009). Force ``--codec vq`` only for the kernel oracle. 70B-class
(even VQ misses the card) still raises :class:`CodecFitError`.

The overhead (CUDA context, KV, activations, Windows desktop) is the 20B
measured peak minus packed weights, rounded up: 11 828 − 10 062 → 1 800 MiB.
It is not a kernel benchmark.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MIB = 1024 * 1024
NF4_BITS = 4.25
VQ_BITS = 2.0
CODEBOOK_BYTES = 8192  # 2 × 256 × 8 × FP16, one book per VQ matrix
# Slightly above the internlm2.5-20B measured (1 766 MiB). Auto must not
# promise a fit that the 20B peak already spent.
RUNTIME_OVERHEAD_MIB = 1800
DEFAULT_VRAM_MIB = 12288  # RTX 3080 12 GB, the machine of record

CODECS = ("auto", "nf4", "vq")


class CodecFitError(ValueError):
    """Neither requested nor auto codec fits the card. Message is user-facing."""


@dataclass(frozen=True)
class Budget:
    params: int
    n_matrices: int
    nf4_mib: float
    vq_mib: float
    vram_mib: int
    leftover_nf4: float
    leftover_vq: float


@dataclass(frozen=True)
class Decision:
    codec: str  # "nf4" or "vq", never "auto"
    requested: str
    reason: str
    budget: Budget
    overflow: bool = False


def load_config(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def params_from_config(cfg: Mapping[str, Any]) -> int:
    """Linear + embed (+ untied lm_head) weight count from HuggingFace shapes.

    Matches the Qwen2.5-14B / 32B arithmetic in docs/vram-3080.md: GQA q/k/v/o
    plus three MLP matrices per layer. InternLM2's fused ``wqkv`` is the same
    number of weights as split q/k/v.
    """
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    layers = int(cfg["num_hidden_layers"])
    vocab = int(cfg["vocab_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads") or n_q)
    head_dim = int(cfg["head_dim"]) if cfg.get("head_dim") else hidden // n_q
    qkv = (n_q + 2 * n_kv) * head_dim * hidden
    o = hidden * n_q * head_dim
    mlp = 3 * intermediate * hidden
    embed = vocab * hidden
    tied = bool(cfg.get("tie_word_embeddings", False))
    lm_head = 0 if tied else vocab * hidden
    return layers * (qkv + o + mlp) + embed + lm_head


def n_vq_matrices(cfg: Mapping[str, Any]) -> int:
    """Codebooks on disk: one 8 KiB book per quantized matrix.

    Split attention is seven GEMMs per layer; InternLM2 fuses qkv (five).
    Embed is always one; lm_head is a second when not tied.
    """
    layers = int(cfg["num_hidden_layers"])
    per_layer = 5 if str(cfg.get("model_type") or "") == "internlm2" else 7
    extra = 1 if bool(cfg.get("tie_word_embeddings", False)) else 2
    return layers * per_layer + extra


def packed_mib(params: int, bits: float, *, extra_bytes: int = 0) -> float:
    return (params * bits / 8 + extra_bytes) / MIB


def budget_for(cfg: Mapping[str, Any], vram_mib: int) -> Budget:
    params = params_from_config(cfg)
    n_mat = n_vq_matrices(cfg)
    nf4 = packed_mib(params, NF4_BITS)
    vq = packed_mib(params, VQ_BITS, extra_bytes=n_mat * CODEBOOK_BYTES)
    card = int(vram_mib)
    return Budget(
        params=params,
        n_matrices=n_mat,
        nf4_mib=nf4,
        vq_mib=vq,
        vram_mib=card,
        leftover_nf4=card - nf4 - RUNTIME_OVERHEAD_MIB,
        leftover_vq=card - vq - RUNTIME_OVERHEAD_MIB,
    )


def decide(
    cfg: Mapping[str, Any],
    vram_mib: int,
    *,
    requested: str = "auto",
) -> Decision:
    """Return the codec to pack / load, or raise :class:`CodecFitError`."""
    req = (requested or "auto").lower()
    if req not in CODECS:
        raise ValueError(f"codec={requested!r}; expected {CODECS}")
    b = budget_for(cfg, vram_mib)

    def nf4_ok() -> bool:
        return b.leftover_nf4 >= 0

    def vq_ok() -> bool:
        return b.leftover_vq >= 0

    too_big = (
        f"even VQ 2-bit ({b.vq_mib:.0f} MiB weights + {RUNTIME_OVERHEAD_MIB} MiB "
        f"runtime) misses {b.vram_mib} MiB VRAM by {abs(b.leftover_vq):.0f} MiB. "
        "70B-class models are out of scope on this card."
    )
    overflow_reason = (
        f"NF4 packed weights {b.nf4_mib:.0f} MiB plus "
        f"{RUNTIME_OVERHEAD_MIB} MiB runtime do not fit "
        f"{b.vram_mib} MiB VRAM (short {abs(b.leftover_nf4):.0f} MiB); "
        "NF4 overflow (H2), not VQ."
    )

    if req == "nf4":
        if nf4_ok():
            return Decision("nf4", req, "requested NF4; it fits this card", b)
        if vq_ok():
            return Decision("nf4", req, overflow_reason, b, overflow=True)
        raise CodecFitError(too_big)
    if req == "vq":
        if not vq_ok():
            raise CodecFitError(
                f"VQ packed weights {b.vq_mib:.0f} MiB plus "
                f"{RUNTIME_OVERHEAD_MIB} MiB runtime do not fit "
                f"{b.vram_mib} MiB VRAM (short {abs(b.leftover_vq):.0f} MiB). "
                "A model this large is not a 12 GB fit even at 2 bits."
            )
        return Decision("vq", req, "requested VQ 2-bit; it fits this card", b)

    if nf4_ok():
        return Decision(
            "nf4",
            req,
            "NF4 fits this card; keeping 4.25-bit (better quality, measured path)",
            b,
        )
    if vq_ok():
        return Decision("nf4", req, overflow_reason, b, overflow=True)
    raise CodecFitError(too_big)


def detect_vram_mib() -> int:
    """Dedicated VRAM of GPU 0 from nvidia-smi, else the 3080 default.

    ``--codec auto`` follows the card in front of the user, not a hardcoded
    12 GB. No torch: ``deepfold compress`` is a CPU path.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.total",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return DEFAULT_VRAM_MIB
    if out.returncode != 0:
        return DEFAULT_VRAM_MIB
    try:
        value = int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return DEFAULT_VRAM_MIB
    return value if value > 0 else DEFAULT_VRAM_MIB


def suffix(codec: str) -> str:
    if codec == "vq":
        return "vq2.chr"
    if codec == "nf4":
        return "nf4.chr"
    raise ValueError(f"suffix: {codec!r} is not a packed codec")
