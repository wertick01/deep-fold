"""Resident Decode V2. Synthetic stack; production TokenLoop is unchanged."""

from __future__ import annotations

from .arena import Arena
from .budget import V2Estimate, estimate_v2_from_config, estimate_v2_mib
from .embed import PackedEmbed, bind_embed, gather_embed
from .graph import GreedyGraph, capture_greedy
from .kv import GraphSafeKV
from .linear import DeviceWeights, nf4_linear
from .load import LoadedDecode, load_chr
from .oracle import greedy_ids, teacher_logits
from .plan import TINY_INTERNLM, TINY_LLAMA, MEDIUM_LLAMA, ArchSpec, spec_from_loaded
from .prefill import forward_prefill, prefill_chunk_width
from .rope import apply_rope_numpy, apply_rope_torch, rope_tables_numpy, rope_tables_torch
from .runner import consume_prompt, generate
from .session import BoundV2, DecodeV2Loop, decodev2_refused, pick_executor
from .state import DecodeState
from .step import forward_decode, greedy_decode, teacher_force_token
from .synth import SynthModel, build

__all__ = [
    "ArchSpec",
    "Arena",
    "DecodeState",
    "DeviceWeights",
    "GraphSafeKV",
    "GreedyGraph",
    "PackedEmbed",
    "V2Estimate",
    "bind_embed",
    "estimate_v2_from_config",
    "estimate_v2_mib",
    "gather_embed",
    "load_chr",
    "LoadedDecode",
    "MEDIUM_LLAMA",
    "TINY_INTERNLM",
    "TINY_LLAMA",
    "SynthModel",
    "apply_rope_numpy",
    "apply_rope_torch",
    "build",
    "capture_greedy",
    "consume_prompt",
    "decodev2_refused",
    "BoundV2",
    "DecodeV2Loop",
    "forward_decode",
    "forward_prefill",
    "generate",
    "greedy_decode",
    "greedy_ids",
    "nf4_linear",
    "pick_executor",
    "prefill_chunk_width",
    "rope_tables_numpy",
    "rope_tables_torch",
    "spec_from_loaded",
    "teacher_force_token",
    "teacher_logits",
]
