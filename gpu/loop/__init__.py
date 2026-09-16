"""Our token loop: preallocated KV, one forward per step, optional CUDA graph A.

Wave 3, agent 7. Owns ``gpu/loop/`` and nothing else: the file format is
``gpu/chr0``, the GEMM is ``gpu/nf4``, the module tree and the NF4 ``nn.Linear``
seat are ``gpu/host``. This package is the driver on top of them --
``transformers.generate`` is never called.

    import sys; sys.path.insert(0, r"C:\\dev\\deep-fold")
    from gpu.host import load_model
    from gpu.loop import TokenLoop

    model, report = load_model(r"C:\\dev\\models\\Qwen2.5-3B-Instruct",
                               r"C:\\dev\\models\\qwen25-3b.nf4.chr")
    loop = TokenLoop(model, max_seq=512)   # binds from load_model's DriverPlan
    loop.warmup()            # eager, before any capture
    loop.capture_graphs()    # plan A: NF4 GEMMs only; falls back to eager
    out = loop.generate(prompt_ids, 64, stop=stop_token_ids(tok))
    print(out.prefill_ms, out.decode_tok_s, out.graph)

``stop`` is never a literal: :func:`gpu.loop.stop.stop_token_ids` asks the
tokenizer, because Qwen closes a turn on 151645 and InternLM2 on 92542.

Contracts: ``docs/token-loop.md`` §1-5.
Acceptance: ``python gpu/loop/smoke.py``.
"""

from __future__ import annotations

from .generate import (
    PACKERS,
    Generation,
    TokenLoop,
    repeat_kv,
    rms_norm_exact,
    split_concat_qkv,
    split_internlm_wqkv,
    split_neox_qkv,
)
from .graph import Gemm, GemmGroup, GraphedGemmGroup, capture, group_is_resident, linear_max_n, nf4_max_n
from .kv_cache import KVCache
from .ring import CopyRing
from .stop import stop_token_ids

__all__ = [
    "TokenLoop",
    "Generation",
    "KVCache",
    "Gemm",
    "GemmGroup",
    "GraphedGemmGroup",
    "PACKERS",
    "capture",
    "group_is_resident",
    "nf4_max_n",
    "linear_max_n",
    "repeat_kv",
    "rms_norm_exact",
    "split_concat_qkv",
    "split_internlm_wqkv",
    "split_neox_qkv",
    "stop_token_ids",
    "CopyRing",
]
