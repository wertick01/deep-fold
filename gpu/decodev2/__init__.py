"""Resident Decode V2. Synthetic stack; production TokenLoop is unchanged."""

from __future__ import annotations

from .arena import Arena
from .graph import GreedyGraph, capture_greedy
from .kv import GraphSafeKV
from .linear import DeviceWeights, nf4_linear
from .load import LoadedDecode, load_chr
from .oracle import greedy_ids, teacher_logits
from .plan import TINY_INTERNLM, TINY_LLAMA, MEDIUM_LLAMA, ArchSpec, spec_from_loaded
from .rope import apply_rope_numpy, apply_rope_torch, rope_tables_numpy, rope_tables_torch
from .runner import generate
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
    "forward_decode",
    "generate",
    "greedy_decode",
    "greedy_ids",
    "nf4_linear",
    "rope_tables_numpy",
    "rope_tables_torch",
    "spec_from_loaded",
    "teacher_force_token",
    "teacher_logits",
]
