"""InternLM2 fused-wqkv packing, no GPU and no 20B weights."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.loop import split_internlm_wqkv  # noqa: E402
from gpu.tests.skips import Skip, requires_module  # noqa: E402


def test_split_matches_internlm_rearrange() -> None:
    """Same slice as modeling_internlm2.py: gs = 2 + n_q/n_kv, K then V at the end."""
    n, n_q, n_kv, hd = 3, 48, 8, 128
    n_rep = n_q // n_kv
    y = torch.arange(n * (n_q + 2 * n_kv) * hd, dtype=torch.float32).view(n, -1)
    q, k, v = split_internlm_wqkv(y, n_q, n_kv, hd)

    # einops ships in the optional `internlm` extra, so a box without it must
    # skip the reference comparison rather than report it as passed (D12).
    requires_module("einops", extra="internlm")
    from einops import rearrange

    packed = rearrange(
        y.view(n, 1, -1),
        "b q (h gs d) -> b q h gs d",
        gs=2 + n_rep,
        d=hd,
    )
    q_ref = rearrange(packed[..., :n_rep, :], "b q h gs d -> b q (h gs) d").squeeze(1)
    k_ref = packed[..., -2, :].squeeze(1)
    v_ref = packed[..., -1, :].squeeze(1)
    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)
    assert torch.equal(v, v_ref)
    assert q.shape == (n, n_q, hd)
    assert k.shape == (n, n_kv, hd)
    assert v.shape == (n, n_kv, hd)


if __name__ == "__main__":
    try:
        test_split_matches_internlm_rearrange()
    except Skip as exc:
        print(f"SKIP split_internlm_wqkv: {exc}")
    else:
        print("PASS split_internlm_wqkv")
