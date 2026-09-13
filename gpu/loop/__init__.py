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
    loop = TokenLoop(model, max_seq=512)
    loop.warmup()            # eager, before any capture
    loop.capture_graphs()    # plan A: NF4 GEMMs only; falls back to eager
    out = loop.generate(prompt_ids, 64, stop=(151645,))
    print(out.prefill_ms, out.decode_tok_s, out.graph)

Contracts: ``docs/token-loop.md`` §1-5.
Acceptance: ``python gpu/loop/smoke.py``.
"""

from __future__ import annotations

from .generate import Generation, TokenLoop, rms_norm_exact, split_internlm_wqkv
from .graph import Gemm, GemmGroup, GraphedGemmGroup, capture, nf4_max_n
from .kv_cache import KVCache

__all__ = [
    "TokenLoop",
    "Generation",
    "KVCache",
    "Gemm",
    "GemmGroup",
    "GraphedGemmGroup",
    "capture",
    "nf4_max_n",
    "rms_norm_exact",
    "split_internlm_wqkv",
]
