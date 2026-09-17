"""CUDA-core NF4 GEMV vs CPU oracle. No checkpoints.

    python gpu/nf4/test_gemv.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.numerics import KERNEL_ABS_LIMIT, QWEN25_3B_LINEARS, error_metrics  # noqa: E402
from gpu.tests.nf4_oracle import decode_nf4, encode_nf4, k_pad, matmul_f32  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _run_pair(m: int, k: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.05)
    x = rng.standard_normal((k, 1), dtype=np.float32)
    packed, scale = encode_nf4(w)
    hat = decode_nf4(packed, scale, m, k)
    y_cpu = matmul_f32(hat, x)
    from gpu.nf4 import nf4_gemv

    pk = torch.from_numpy(packed).cuda()
    sc = torch.from_numpy(scale).cuda()
    xt = torch.from_numpy(x).to(device="cuda", dtype=torch.bfloat16)
    y = nf4_gemv(pk, sc, xt, m, k, k_pad(k)).float().cpu().numpy()
    met = error_metrics(y, y_cpu)
    return float(met.max_abs), float(met.cosine)


def test_tiny_shapes() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    for m, k, seed in ((64, 64, 0), (32, 64, 1), (128, 128, 2), (65, 65, 3)):
        max_abs, cosine = _run_pair(m, k, seed)
        assert max_abs <= max(KERNEL_ABS_LIMIT, 0.2), (m, k, max_abs)
        assert cosine > 0.99, (m, k, cosine)


def test_catalog_q_and_kv() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    kinds = {n: (m, k) for n, m, k in QWEN25_3B_LINEARS}
    for name in ("q_proj", "k_proj", "down_proj"):
        m, k = kinds[name]
        max_abs, cosine = _run_pair(m, k, seed=11)
        assert max_abs <= 0.5, (name, max_abs)
        assert cosine > 0.99, (name, cosine)


def test_gemv_multirow() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.nf4 import nf4_gemv, nf4_swiglu

    for m, k, seed in ((1025, 64, 20), (4101, 64, 21), (8193, 64, 22)):
        max_abs, cosine = _run_pair(m, k, seed)
        assert max_abs <= max(KERNEL_ABS_LIMIT, 0.25), (m, k, max_abs)
        assert cosine > 0.99, (m, k, cosine)
    import os

    os.environ["CHR_NF4_GEMV_NR"] = "8"
    try:
        max_abs, cosine = _run_pair(8193, 64, 24)
        assert max_abs <= max(KERNEL_ABS_LIMIT, 0.25), (8193, 8, max_abs)
        assert cosine > 0.99, (8193, 8, cosine)
    finally:
        os.environ.pop("CHR_NF4_GEMV_NR", None)
    rng = np.random.default_rng(23)
    m, k = 4101, 64
    xt = torch.from_numpy(rng.standard_normal((k, 1), dtype=np.float32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    packed = []
    for _ in range(2):
        w = rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.05)
        pk, sc = encode_nf4(w)
        packed.append((torch.from_numpy(pk).cuda(), torch.from_numpy(sc).cuda()))
    g = nf4_gemv(packed[0][0], packed[0][1], xt, m, k, k_pad(k)).view(-1).float()
    u = nf4_gemv(packed[1][0], packed[1][1], xt, m, k, k_pad(k)).view(-1).float()
    want = (torch.nn.functional.silu(g) * u).cpu().numpy()
    out = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    nf4_swiglu(
        packed[0][0],
        packed[0][1],
        packed[1][0],
        packed[1][1],
        xt,
        m,
        k,
        k_pad(k),
        out,
    )
    met = error_metrics(out.float().cpu().numpy(), want)
    assert float(met.max_abs) <= 0.08, met.max_abs
    assert float(met.cosine) > 0.99, met.cosine


def test_qkv_matches_three_gemv() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.nf4 import nf4_gemv, nf4_qkv

    rng = np.random.default_rng(4)
    k = 64
    shapes = ((64, k), (32, k), (32, k))
    xt = torch.from_numpy(rng.standard_normal((k, 1), dtype=np.float32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    mats = []
    outs = []
    for m, kk in shapes:
        w = rng.standard_normal((m, kk), dtype=np.float32) * np.float32(0.05)
        packed, scale = encode_nf4(w)
        pk = torch.from_numpy(packed).cuda()
        sc = torch.from_numpy(scale).cuda()
        mats.append((pk, sc, m))
        outs.append(nf4_gemv(pk, sc, xt, m, kk, k_pad(kk)).view(-1))
    yq = torch.empty(64, device="cuda", dtype=torch.bfloat16)
    yk = torch.empty(32, device="cuda", dtype=torch.bfloat16)
    yv = torch.empty(32, device="cuda", dtype=torch.bfloat16)
    nf4_qkv(
        mats[0][0],
        mats[0][1],
        mats[1][0],
        mats[1][1],
        mats[2][0],
        mats[2][1],
        xt,
        64,
        32,
        32,
        k,
        k_pad(k),
        yq,
        yk,
        yv,
    )
    for got, want in ((yq, outs[0]), (yk, outs[1]), (yv, outs[2])):
        met = error_metrics(got.float().cpu().numpy(), want.float().cpu().numpy())
        assert float(met.max_abs) <= 1e-3, met.max_abs
        assert float(met.cosine) > 0.999, met.cosine

    bq = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    bk = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    bv = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    nf4_qkv(
        mats[0][0],
        mats[0][1],
        mats[1][0],
        mats[1][1],
        mats[2][0],
        mats[2][1],
        xt,
        64,
        32,
        32,
        k,
        k_pad(k),
        yq,
        yk,
        yv,
        bq,
        bk,
        bv,
    )
    for got, gemv, bias in (
        (yq, outs[0], bq),
        (yk, outs[1], bk),
        (yv, outs[2], bv),
    ):
        want = gemv.float() + bias.float()
        met = error_metrics(got.float().cpu().numpy(), want.cpu().numpy())
        assert float(met.max_abs) <= 0.02, met.max_abs
        assert float(met.cosine) > 0.999, met.cosine


def test_swiglu_matches_pair() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.nf4 import nf4_gemv, nf4_swiglu

    rng = np.random.default_rng(5)
    m, k = 128, 64
    xt = torch.from_numpy(rng.standard_normal((k, 1), dtype=np.float32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    packed = []
    for seed in (0, 1):
        w = rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.05)
        pk, sc = encode_nf4(w)
        packed.append((torch.from_numpy(pk).cuda(), torch.from_numpy(sc).cuda()))
    g = nf4_gemv(packed[0][0], packed[0][1], xt, m, k, k_pad(k)).view(-1).float()
    u = nf4_gemv(packed[1][0], packed[1][1], xt, m, k, k_pad(k)).view(-1).float()
    want = (torch.nn.functional.silu(g) * u).cpu().numpy()
    out = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    nf4_swiglu(
        packed[0][0],
        packed[0][1],
        packed[1][0],
        packed[1][1],
        xt,
        m,
        k,
        k_pad(k),
        out,
    )
    met = error_metrics(out.float().cpu().numpy(), want)
    assert float(met.max_abs) <= 0.05, met.max_abs
    assert float(met.cosine) > 0.99, met.cosine


def test_gemv_residual_add() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.nf4 import nf4_gemv

    rng = np.random.default_rng(6)
    m, k = 64, 64
    w = rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.05)
    packed, scale = encode_nf4(w)
    pk = torch.from_numpy(packed).cuda()
    sc = torch.from_numpy(scale).cuda()
    xt = torch.from_numpy(rng.standard_normal((k, 1), dtype=np.float32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    dest = torch.randn(m, device="cuda", dtype=torch.bfloat16)
    base = dest.clone()
    y = nf4_gemv(pk, sc, xt, m, k, k_pad(k))
    want = (base.float() + y.view(-1).float()).cpu().numpy()
    nf4_gemv(pk, sc, xt, m, k, k_pad(k), dest, dest)
    met = error_metrics(dest.float().cpu().numpy(), want)
    assert float(met.max_abs) <= 0.02, met.max_abs
    assert float(met.cosine) > 0.999, met.cosine


def test_attn_matches_sdpa() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.nf4 import nf4_attn

    n_q, n_kv, hd, max_seq, vl = 4, 2, 16, 16, 7
    torch.manual_seed(7)
    q = torch.randn(1, n_q, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(max_seq, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(max_seq, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, n_q * hd, device="cuda", dtype=torch.bfloat16)
    valid = torch.tensor(vl, device="cuda", dtype=torch.int32)
    scale = hd**-0.5
    nf4_attn(q, k, v, out, valid, scale)
    q4 = q.reshape(1, n_kv, n_q // n_kv, hd)
    k4 = k.permute(1, 0, 2).unsqueeze(0)
    v4 = v.permute(1, 0, 2).unsqueeze(0)
    mask = torch.zeros(1, 1, 1, max_seq, device="cuda", dtype=q.dtype)
    mask[..., vl:] = -1.0e9
    want = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, attn_mask=mask, scale=scale
    ).reshape(1, n_q * hd)
    met = error_metrics(out.float().cpu().numpy(), want.float().cpu().numpy())
    assert float(met.max_abs) <= 0.05, met.max_abs
    assert float(met.cosine) > 0.99, met.cosine

    n_q, n_kv, hd, max_seq, vl = 16, 2, 128, 512, 48
    q = torch.randn(1, n_q, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(max_seq, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(max_seq, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, n_q * hd, device="cuda", dtype=torch.bfloat16)
    valid = torch.tensor(vl, device="cuda", dtype=torch.int32)
    nf4_attn(q, k, v, out, valid, hd**-0.5)
    q4 = q.reshape(1, n_kv, n_q // n_kv, hd)
    k4 = k.permute(1, 0, 2).unsqueeze(0)
    v4 = v.permute(1, 0, 2).unsqueeze(0)
    mask = torch.zeros(1, 1, 1, max_seq, device="cuda", dtype=q.dtype)
    mask[..., vl:] = -1.0e9
    want = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, attn_mask=mask, scale=hd**-0.5
    ).reshape(1, n_q * hd)
    met = error_metrics(out.float().cpu().numpy(), want.float().cpu().numpy())
    assert float(met.max_abs) <= 0.08, met.max_abs
    assert float(met.cosine) > 0.99, met.cosine


def test_attn_fused_rope() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.decodev2.rope import apply_rope_torch, rope_tables_torch
    from gpu.nf4 import nf4_attn

    n_q, n_kv, hd, max_seq, vl, pos = 8, 2, 64, 32, 5, 4
    torch.manual_seed(9)
    q = torch.randn(1, n_q, hd, device="cuda", dtype=torch.bfloat16)
    k_act = torch.randn(1, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    v_act = torch.randn(1, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(max_seq, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(max_seq, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    cos, sin = rope_tables_torch(max_seq, hd, 10000.0, device="cuda", dtype=torch.bfloat16)
    want_k = k.clone()
    want_v = v.clone()
    q_rot = apply_rope_torch(q, cos[pos : pos + 1], sin[pos : pos + 1]).contiguous()
    k_rot = apply_rope_torch(k_act, cos[pos : pos + 1], sin[pos : pos + 1])
    want_k[pos] = k_rot.reshape(n_kv, hd)
    want_v[pos] = v_act.reshape(n_kv, hd)
    valid = torch.tensor(vl, device="cuda", dtype=torch.int32)
    scale = hd**-0.5
    ref = torch.empty(1, n_q * hd, device="cuda", dtype=torch.bfloat16)
    nf4_attn(q_rot, want_k, want_v, ref, valid, scale)
    got = torch.empty(1, n_q * hd, device="cuda", dtype=torch.bfloat16)
    k_in = k.clone()
    v_in = v.clone()
    nf4_attn(
        q,
        k_in,
        v_in,
        got,
        valid,
        scale,
        k_act=k_act,
        v_act=v_act,
        cos=cos,
        sin=sin,
        position=torch.tensor(pos, device="cuda", dtype=torch.int64),
    )
    met = error_metrics(got.float().cpu().numpy(), ref.float().cpu().numpy())
    assert float(met.max_abs) <= 0.08, met.max_abs
    assert float(met.cosine) > 0.99, met.cosine
    kmet = error_metrics(k_in[pos].float().cpu().numpy(), want_k[pos].float().cpu().numpy())
    vmet = error_metrics(v_in[pos].float().cpu().numpy(), want_v[pos].float().cpu().numpy())
    assert float(kmet.max_abs) <= 0.02, kmet.max_abs
    assert float(vmet.max_abs) <= 0.02, vmet.max_abs


def test_rms_matches_torch() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.nf4 import nf4_rms

    torch.manual_seed(8)
    n = 2048
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    nf4_rms(x, w, y, 1e-6)
    want = torch.nn.functional.rms_norm(x, (n,), w, 1e-6)
    met = error_metrics(y.float().cpu().numpy(), want.float().cpu().numpy())
    assert float(met.max_abs) <= 0.1, met.max_abs
    assert float(met.cosine) > 0.999, met.cosine


def test_gemv_fused_rms() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    from gpu.nf4 import nf4_gemv, nf4_qkv, nf4_rms, nf4_swiglu

    torch.manual_seed(10)
    m, k = 64, 64
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    h = torch.empty(k, device="cuda", dtype=torch.bfloat16)
    nf4_rms(x, w, h, 1e-6)
    rng = np.random.default_rng(10)
    mat = rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.05)
    packed, scale = encode_nf4(mat)
    pk = torch.from_numpy(packed).cuda()
    sc = torch.from_numpy(scale).cuda()
    want = nf4_gemv(pk, sc, h, m, k, k_pad(k)).view(-1)
    got = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    nf4_gemv(pk, sc, x, m, k, k_pad(k), got, rms_w=w, rms_eps=1e-6)
    met = error_metrics(got.float().cpu().numpy(), want.float().cpu().numpy())
    assert float(met.max_abs) <= 0.05, met.max_abs
    assert float(met.cosine) > 0.99, met.cosine

    yq = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    yk = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    yv = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    nf4_qkv(pk, sc, pk, sc, pk, sc, x, m, m, m, k, k_pad(k), yq, yk, yv, rms_w=w)
    ref_q = nf4_gemv(pk, sc, h, m, k, k_pad(k)).view(-1)
    qmet = error_metrics(yq.float().cpu().numpy(), ref_q.float().cpu().numpy())
    assert float(qmet.max_abs) <= 0.05, qmet.max_abs

    gout = torch.empty(m, device="cuda", dtype=torch.bfloat16)
    nf4_swiglu(pk, sc, pk, sc, x, m, k, k_pad(k), gout, rms_w=w)
    g = nf4_gemv(pk, sc, h, m, k, k_pad(k)).view(-1).float()
    u = nf4_gemv(pk, sc, h, m, k, k_pad(k)).view(-1).float()
    want_s = (torch.nn.functional.silu(g) * u).cpu().numpy()
    smet = error_metrics(gout.float().cpu().numpy(), want_s)
    assert float(smet.max_abs) <= 0.08, smet.max_abs
    assert float(smet.cosine) > 0.99, smet.cosine


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/nf4 gemv, {len(TESTS)} tests\n")
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
