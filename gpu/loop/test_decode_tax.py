"""Decode-path glue: no extra copies, prefetch-before-head, fused RMS on N==1.

    python gpu/loop/test_decode_tax.py

CPU dummy TokenLoop (no .chr, no 3B). CUDA RoPE graph is skipped without a device.
"""

from __future__ import annotations

import inspect
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.loop import generate as generate_mod  # noqa: E402
from gpu.loop.generate import (  # noqa: E402
    TokenLoop,
    _rope,
    _RopePairGraph,
    rms_norm_exact,
    split_concat_qkv,
    split_internlm_wqkv,
    split_neox_qkv,
)
from gpu.loop.kv_cache import KVCache  # noqa: E402
from gpu.loop.test_attach import _bound, qwen2_model  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _loop(model, **over) -> TokenLoop:
    original = generate_mod.linear_max_n
    generate_mod.linear_max_n = lambda codec="nf4", probe=16: 1
    try:
        return TokenLoop(model, max_seq=8, overlap=False, **over)
    finally:
        generate_mod.linear_max_n = original


def _bf16_norms(loop: TokenLoop) -> None:
    loop.embed.weight.data = loop.embed.weight.data.to(torch.bfloat16)
    for w in (*loop._norm1, *loop._norm2, loop.final_norm):
        w.data = w.data.to(torch.bfloat16)


class _FakeGroup:
    def __init__(self, ms: tuple[int, ...]) -> None:
        self.ms = ms

    def run(self, x: torch.Tensor, ring=None) -> tuple[torch.Tensor, ...]:
        n = int(x.shape[0])
        return tuple(x.new_zeros(n, m) for m in self.ms)


def _install_fake_gemms(loop: TokenLoop) -> None:
    """Replace GEMM runners with zeros so glue can run without a kernel."""

    def ms(runner) -> tuple[int, ...]:
        g = getattr(runner, "eager", runner)
        return tuple(int(m.M) for m in g.gemms)

    loop._layer = [
        tuple(_FakeGroup(ms(g)) for g in row)  # type: ignore[misc]
        for row in loop._layer
    ]
    loop._head = _FakeGroup(ms(loop._head))  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# contracts that must not move
# --------------------------------------------------------------------------- #


def test_defaults_exact_norm_and_quiet_ring() -> None:
    sig = inspect.signature(TokenLoop.__init__)
    assert sig.parameters["norm"].default == "exact"
    assert sig.parameters["ring_timing"].default is False
    model, _ = _bound(qwen2_model())
    loop = _loop(model)
    assert loop.norm_mode == "exact"
    assert loop.rms is rms_norm_exact
    assert loop._ring is None
    assert loop._rope_pair is None
    assert tuple(loop.cos.shape) == (loop.max_seq, 1, loop.head_dim)
    assert tuple(loop.sin.shape) == (loop.max_seq, 1, loop.head_dim)


def test_kv_decode_write_is_slot_not_cat() -> None:
    kv = KVCache(2, 8, 2, 4, device="cpu", dtype=torch.bfloat16)
    k_ptr, v_ptr = kv.k.data_ptr(), kv.v.data_ptr()
    nbytes = kv.k.untyped_storage().nbytes()
    knew = torch.arange(8, dtype=torch.bfloat16).view(1, 2, 4)
    vnew = knew + 10
    kv.write(1, 3, knew, vnew)
    assert kv.k.data_ptr() == k_ptr and kv.v.data_ptr() == v_ptr
    assert kv.k.untyped_storage().nbytes() == nbytes
    assert torch.equal(kv.k[1, 3], knew[0])
    assert torch.equal(kv.v[1, 3], vnew[0])
    k_all, v_all = kv.view(1, 4)
    assert tuple(k_all.shape) == (1, 2, 4, 4)
    assert k_all.untyped_storage().data_ptr() == kv.k.untyped_storage().data_ptr()
    assert v_all.untyped_storage().data_ptr() == kv.v.untyped_storage().data_ptr()
    assert torch.equal(k_all[0, :, 3, :], knew[0])


def test_kv_prefill_write_still_assigns_a_slice() -> None:
    kv = KVCache(1, 8, 2, 4, device="cpu", dtype=torch.float32)
    k = torch.randn(3, 2, 4)
    v = torch.randn(3, 2, 4)
    kv.write(0, 1, k, v)
    assert torch.equal(kv.k[0, 1:4], k)
    got, _ = kv.view(0, 4)
    assert tuple(got.shape) == (1, 2, 4, 4)
    assert torch.equal(got[0, :, 1:4, :], k.permute(1, 0, 2))


def test_packers_return_views_not_contiguous_copies() -> None:
    n_q, n_kv, hd, n = 8, 2, 4, 1
    y = torch.arange((n_q + 2 * n_kv) * hd, dtype=torch.float32).view(n, -1)
    _, k_i, v_i = split_internlm_wqkv(y, n_q, n_kv, hd)
    _, k_c, v_c = split_concat_qkv(y, n_q, n_kv, hd)
    assert k_i.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    assert k_c.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    assert v_i.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    assert v_c.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    y_neox = torch.arange(n_q * 3 * hd, dtype=torch.float32).view(n, -1)
    q_n, k_n, v_n = split_neox_qkv(y_neox, n_q, n_q, hd)
    assert q_n.untyped_storage().data_ptr() == y_neox.untyped_storage().data_ptr()
    assert k_n.untyped_storage().data_ptr() == y_neox.untyped_storage().data_ptr()
    assert v_n.untyped_storage().data_ptr() == y_neox.untyped_storage().data_ptr()


# --------------------------------------------------------------------------- #
# dummy forward: prefetch order, fused RMS on N==1, exact on N!=1
# --------------------------------------------------------------------------- #


def test_prefetch_next_is_before_lm_head_on_n1() -> None:
    model, _ = _bound(qwen2_model())
    loop = _loop(model)
    _bf16_norms(loop)
    _install_fake_gemms(loop)
    order: list[str] = []
    ring = SimpleNamespace(
        arm=lambda tape: order.append("arm"),
        prefetch=lambda: order.append("prefetch"),
        prefetch_next=lambda: order.append("prefetch_next"),
    )
    loop._ring = ring  # type: ignore[assignment]
    loop._host_tape = ()
    orig_head = loop._head

    class _Head:
        def run(self, hidden, ring=None):
            order.append("head")
            return orig_head.run(hidden, ring=ring)

    loop._head = _Head()  # type: ignore[assignment]
    ids = torch.tensor([1], dtype=torch.long)
    loop.forward(ids, 0)
    assert order[:2] == ["arm", "prefetch"], order
    assert "prefetch_next" in order and "head" in order, order
    assert order.index("prefetch_next") < order.index("head"), order


def test_logits_false_and_n_gt_1_skip_lm_head_prefetch() -> None:
    model, _ = _bound(qwen2_model())
    loop = _loop(model)
    _bf16_norms(loop)
    _install_fake_gemms(loop)
    hits = {"n": []}
    orig = loop._prefetch_next_token

    def wrapped(n: int) -> None:
        hits["n"].append(n)
        orig(n)

    loop._prefetch_next_token = wrapped  # type: ignore[method-assign]
    loop.forward(torch.tensor([1], dtype=torch.long), 0, logits=False)
    assert hits["n"] == [], f"logits=False must not prefetch, got {hits['n']}"

    ring = SimpleNamespace(prefetch_next=lambda: hits.setdefault("next", []).append(1))
    loop._ring = ring  # type: ignore[assignment]
    loop._prefetch_next_token(2)
    assert "next" not in hits

    loop.reset()
    hits["n"].clear()
    loop._ring = None
    # n>1 uses exact RMS / prefill body; prefetch_next_token still runs at the
    # head but is a no-op without a ring. Probe that the hook saw n==2.
    out = loop.forward(torch.tensor([1, 2], dtype=torch.long), 0)
    assert out is not None and out.shape[-1] == loop.vocab
    assert hits["n"] == [2]


def test_decode_n1_uses_fused_rms_prefill_keeps_exact() -> None:
    model, _ = _bound(qwen2_model())
    loop = _loop(model)
    _bf16_norms(loop)
    _install_fake_gemms(loop)
    assert loop.norm_mode == "exact"
    counts = {"fast": 0, "exact": 0}
    orig_fast = F.rms_norm
    orig_exact = generate_mod.rms_norm_exact

    def fast(*args, **kwargs):
        counts["fast"] += 1
        return orig_fast(*args, **kwargs)

    def exact(*args, **kwargs):
        counts["exact"] += 1
        return orig_exact(*args, **kwargs)

    F.rms_norm = fast  # type: ignore[assignment]
    generate_mod.rms_norm_exact = exact
    loop.rms = exact
    try:
        loop.reset()
        loop.forward(torch.tensor([1], dtype=torch.long), 0)
        decode_fast, decode_exact = counts["fast"], counts["exact"]
        # 2 layers * 2 norms + final = 5 fused; exact unused on N==1.
        assert decode_fast == 2 * loop.n_layers + 1, decode_fast
        assert decode_exact == 0, decode_exact
        counts["fast"] = counts["exact"] = 0
        loop.reset()
        loop.forward(torch.tensor([1, 2], dtype=torch.long), 0)
        assert counts["exact"] == 2 * loop.n_layers + 1, counts
        assert counts["fast"] == 0, counts
    finally:
        F.rms_norm = orig_fast
        generate_mod.rms_norm_exact = orig_exact


def test_decode_gqa_view_not_repeat_kv() -> None:
    """N==1 attention keeps the n_kv * n_rep view; output is [1, q_dim]."""
    q = torch.randn(1, 8, 4)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    n_kv, n_rep, hd, q_dim = 2, 4, 4, 32
    a = F.scaled_dot_product_attention(
        q.view(1, n_kv, n_rep, hd), k, v, scale=hd**-0.5
    )
    assert tuple(a.shape) == (1, n_kv, n_rep, hd)
    flat = a.reshape(1, q_dim)
    assert tuple(flat.shape) == (1, q_dim)
    assert q.view(1, n_kv, n_rep, hd).untyped_storage().data_ptr() == q.untyped_storage().data_ptr()


def test_rope_pair_graph_matches_eager() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    n_q, n_kv, hd = 8, 2, 4
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pair = _RopePairGraph(n_q, n_kv, hd, device, dtype)
    q = torch.randn(1, n_q, hd, device=device, dtype=dtype)
    k = torch.randn(1, n_kv, hd, device=device, dtype=dtype)
    cos = torch.randn(1, 1, hd, device=device, dtype=dtype)
    sin = torch.randn(1, 1, hd, device=device, dtype=dtype)
    pair.load_pos(cos, sin)
    qg, kg = pair.apply(q, k)
    qe, ke = _rope(q, cos, sin), _rope(k, cos, sin)
    assert torch.equal(qg, qe), "rope graph Q must match eager"
    assert torch.equal(kg, ke), "rope graph K must match eager"
    assert qg.data_ptr() != pair.q_out.data_ptr(), "token path must not alias graph-pool Q"
    assert kg.data_ptr() != pair.k_out.data_ptr(), "token path must not alias graph-pool K"
    # Replay must not allocate a new cache object; outputs are static live buffers.
    qg2, kg2 = pair.apply(q, k)
    assert qg2.data_ptr() == qg.data_ptr()
    assert kg2.data_ptr() == kg.data_ptr()


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/loop decode-tax, {len(TESTS)} tests\n")
    for fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
        except Skip as exc:
            SKIPPED.append((fn.__name__, str(exc)))
            print(f"  SKIP  {fn.__name__}  -- {exc}")
        except AssertionError as exc:
            check(fn.__name__, False, str(exc).splitlines()[0] if str(exc) else "assert")
        except Exception as exc:  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
        else:
            check(fn.__name__, True)

    failed = [name for name, ok, _ in CHECKS if not ok]
    tail = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed{tail}")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())
