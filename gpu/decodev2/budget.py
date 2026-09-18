"""Decode V2 device-byte estimate. No torch. Used before ``--executor auto``.

``gpu.cli.codec.RUNTIME_OVERHEAD_MIB`` is TokenLoop / 20B codec fit (packed
weights plus a lump). This module counts V2 pieces: packed tensors already
on the card, KV at ``max_seq``, the decode arena, RoPE tables, a CUDA-graph
allowance, and a WDDM reserve. Packed embed is inside ``packed_mib``; a dense
vocab table is extra and is the 20B ~1 GiB hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "GRAPH_MIB",
    "MIB",
    "WDDM_RESERVE_MIB",
    "V2Estimate",
    "estimate_v2_from_config",
    "estimate_v2_mib",
    "shapes_from_config",
]

MIB = 1024 * 1024
# Capture workspace + replay buffers. Not a measured Nsight number.
GRAPH_MIB = 256.0
# Desktop driver / 3080 12 GB headroom. Live 20B dense peak sat at 12153 / 12256.
WDDM_RESERVE_MIB = 512.0
_ATTN_SPLIT = 32  # must match gpu.decodev2.arena.Arena


@dataclass(frozen=True)
class V2Estimate:
    packed_mib: float
    kv_mib: float
    arena_mib: float
    rope_mib: float
    dense_embed_extra_mib: float
    graph_mib: float
    wddm_reserve_mib: float
    total_mib: float


def _get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def shapes_from_config(cfg: Any, *, max_seq: int) -> dict[str, int]:
    """Hidden / GQA / vocab from a HuggingFace config or a dict."""
    hidden = int(_get(cfg, "hidden_size"))
    n_q = int(_get(cfg, "num_attention_heads"))
    n_kv = int(_get(cfg, "num_key_value_heads") or n_q)
    head_dim = _get(cfg, "head_dim")
    head_dim = int(head_dim) if head_dim else hidden // n_q
    return {
        "n_layers": int(_get(cfg, "num_hidden_layers")),
        "n_q": n_q,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "hidden": hidden,
        "intermediate": int(_get(cfg, "intermediate_size")),
        "vocab": int(_get(cfg, "vocab_size")),
        "max_seq": int(max_seq),
    }


def estimate_v2_mib(
    *,
    packed_mib: float,
    n_layers: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
    hidden: int,
    intermediate: int,
    vocab: int,
    max_seq: int,
    packed_embed: bool = True,
    elem_bytes: int = 2,
) -> V2Estimate:
    """Peak Decode V2 bytes after ``load_model``, before the first request.

    ``packed_mib`` is ``LoadReport.device_mib`` (linears + norms + packed embed).
    Dense embed, when requested, is an extra full ``[vocab, hidden]`` table on
    top of that packed copy (the old ``dequant_table`` path).
    """
    kv = 2 * int(n_layers) * int(max_seq) * int(n_kv) * int(head_dim) * int(elem_bytes)
    act = (
        hidden
        + hidden
        + n_q * head_dim
        + n_kv * head_dim
        + n_kv * head_dim
        + n_q * head_dim
        + 3 * intermediate
        + vocab
    ) * int(elem_bytes)
    act += int(max_seq) * 4  # attn_mask float32
    act += int(n_q) * _ATTN_SPLIT * (int(head_dim) + 2) * 4  # attn_ws
    rope = 2 * int(max_seq) * int(head_dim) * int(elem_bytes)
    dense = int(vocab) * int(hidden) * int(elem_bytes)
    extra = 0.0 if packed_embed else dense / MIB
    kv_mib = kv / MIB
    arena_mib = act / MIB
    rope_mib = rope / MIB
    total = (
        float(packed_mib)
        + kv_mib
        + arena_mib
        + rope_mib
        + extra
        + GRAPH_MIB
        + WDDM_RESERVE_MIB
    )
    return V2Estimate(
        packed_mib=float(packed_mib),
        kv_mib=kv_mib,
        arena_mib=arena_mib,
        rope_mib=rope_mib,
        dense_embed_extra_mib=extra,
        graph_mib=GRAPH_MIB,
        wddm_reserve_mib=WDDM_RESERVE_MIB,
        total_mib=total,
    )


def estimate_v2_from_config(
    packed_mib: float,
    config: Any,
    *,
    max_seq: int,
    packed_embed: bool = True,
) -> V2Estimate:
    shapes = shapes_from_config(config, max_seq=max_seq)
    return estimate_v2_mib(packed_mib=float(packed_mib), packed_embed=packed_embed, **shapes)
