"""CPU: nf4_gemm_cpu vs oracle, home=cpu, no pin. No GPU, no CUDA extension.

    python gpu/host/test_cpu_linear.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.cpu_linear import LIVE_MAX_N, nf4_gemm_cpu, nf4_linear_cpu  # noqa: E402
from gpu.host.host_image import cpu_is_pinned  # noqa: E402
from gpu.host.linear import CompressedLinear  # noqa: E402
from gpu.loop.generate import repeat_kv  # noqa: E402
from gpu.loop.graph import Gemm  # noqa: E402
from gpu.tests.nf4_oracle import decode_nf4, k_pad, matmul_f32, toy_nf4  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _maxabs(a, b) -> float:
    return float((a - b).abs().max())


def test_oracle_match_n1_and_n32() -> None:
    for n, m, k, seed in ((1, 32, 64, 1), (32, 48, 96, 2), (1, 17, 65, 3)):
        packed_np, scale_np = toy_nf4(m, k, seed=seed)
        x_np = (
            torch.randn(k, n, generator=torch.Generator().manual_seed(seed))
            .float()
            .numpy()
        )
        w = decode_nf4(packed_np, scale_np, m, k)
        y_ref = matmul_f32(w, x_np.astype("float32"))
        packed = torch.from_numpy(packed_np)
        scale = torch.from_numpy(scale_np)
        x = torch.from_numpy(x_np.astype("float32"))
        y32 = nf4_gemm_cpu(
            packed, scale, x, m, k, k_pad(k), row_chunk=8, dtype=torch.float32
        )
        err = _maxabs(y32, torch.from_numpy(y_ref))
        check(
            f"oracle N={n} M={m} K={k}",
            err < 1e-4,
            f"maxabs={err:.2e}",
        )
        y_bf = nf4_gemm_cpu(packed, scale, x, m, k, k_pad(k), row_chunk=8)
        err_bf = _maxabs(y_bf.float(), torch.from_numpy(y_ref).to(torch.bfloat16).float())
        check(
            f"bf16 roundtrip N={n} M={m} K={k}",
            err_bf < 1e-5,
            f"maxabs={err_bf:.2e}",
        )


def test_row_chunk_never_full_w() -> None:
    m, k, n = 1000, 64, 4
    packed_np, scale_np = toy_nf4(m, k, seed=4)
    packed = torch.from_numpy(packed_np)
    scale = torch.from_numpy(scale_np)
    x = torch.randn(k, n)
    import gpu.host.cpu_linear as cpu_mod

    orig = cpu_mod.dequant_nf4_rows
    seen: list[int] = []

    def spy(packed, scale, ids, K, **kw):  # noqa: N803
        seen.append(int(ids.numel()))
        return orig(packed, scale, ids, K, **kw)

    cpu_mod.dequant_nf4_rows = spy
    try:
        y = cpu_mod.nf4_gemm_cpu(packed, scale, x, m, k, k_pad(k), row_chunk=256)
    finally:
        cpu_mod.dequant_nf4_rows = orig
    check("chunk sizes <= 256", bool(seen) and all(s <= 256 for s in seen), str(seen[:8]))
    check("several chunks", len(seen) >= 4, f"n_chunks={len(seen)}")
    check("out shape", tuple(y.shape) == (m, n), str(tuple(y.shape)))


def test_refuse_cuda_x() -> None:
    if not torch.cuda.is_available():
        packed_np, scale_np = toy_nf4(8, 64, seed=5)
        packed = torch.from_numpy(packed_np)
        scale = torch.from_numpy(scale_np)
        x = torch.randn(64, 1)
        raised = False
        try:
            # CPU path still refuses if we lie with a CUDA-tagged... skip
            nf4_gemm_cpu(packed, scale, x, 8, 64, 64)
        except RuntimeError:
            raised = True
        check("cpu gemm on cpu x works", not raised, "")
        return
    packed_np, scale_np = toy_nf4(8, 64, seed=5)
    packed = torch.from_numpy(packed_np)
    scale = torch.from_numpy(scale_np)
    x = torch.randn(64, 1, device="cuda")
    raised = False
    msg = ""
    try:
        nf4_gemm_cpu(packed, scale, x, 8, 64, 64)
    except RuntimeError as exc:
        raised = True
        msg = str(exc)
    check("refuse CUDA x", raised and "H2D" in msg, msg[:120])


def test_attach_cpu_home_not_pinned() -> None:
    m, k = 16, 64
    lin = CompressedLinear(k, m)
    packed_np, scale_np = toy_nf4(m, k, seed=6)
    packed = torch.from_numpy(packed_np)
    scale = torch.from_numpy(scale_np)
    mat = SimpleNamespace(
        name="toy.down",
        M=m,
        K=k,
        K_pad=k_pad(k),
        packed=packed,
        scale=scale,
    )
    lin.attach_cpu(mat)
    check("home cpu", lin.home == "cpu", lin.home)
    check("is_loaded", lin.is_loaded, "")
    check("packed live", int(lin.packed.numel()) > 0, "")
    check("host_image None", lin.host_image is None, "")
    check("packed not pinned", not cpu_is_pinned(lin.packed), "")
    check("scale not pinned", not cpu_is_pinned(lin.scale), "")
    g = Gemm.of(lin, "L0.down")
    check("Gemm.of home=cpu", g.home == "cpu", g.home)
    check("Gemm.of host_image None", g.host_image is None, "")
    x = torch.randn(1, k, dtype=torch.bfloat16)
    y = lin(x)
    check("forward shape", tuple(y.shape) == (1, m), str(tuple(y.shape)))
    if torch.cuda.is_available():
        raised = False
        try:
            lin(x.to("cuda"))
        except RuntimeError as exc:
            raised = "H2D" in str(exc) or "CPU-resident" in str(exc)
        check("forward refuses CUDA x", raised, "")


def test_nf4_linear_cpu_n32() -> None:
    m, k, n = 24, 64, LIVE_MAX_N
    packed_np, scale_np = toy_nf4(m, k, seed=7)
    packed = torch.from_numpy(packed_np)
    scale = torch.from_numpy(scale_np)
    x = torch.randn(n, k, dtype=torch.bfloat16)
    y = nf4_linear_cpu(packed=packed, scale=scale, x=x, M=m, K=k, K_pad=k_pad(k))
    check("linear N=32 out", tuple(y.shape) == (n, m), str(tuple(y.shape)))
    w = decode_nf4(packed_np, scale_np, m, k)
    y_ref = matmul_f32(w, x.float().transpose(0, 1).contiguous().numpy())
    err = _maxabs(y.float(), torch.from_numpy(y_ref.T.copy()))
    # bf16 activations vs fp32 oracle
    check("linear vs oracle bf16 x", err < 0.05, f"maxabs={err:.3e}")


def test_repeat_kv_gqa_n_rep_5() -> None:
    """32B is 40/8, n_rep=5. Expand path for CPU SDPA without enable_gqa."""
    k = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
    out = repeat_kv(k, 5)
    check("repeat heads 2*5=10", out.shape == (1, 10, 2, 4), str(tuple(out.shape)))
    check("head0 copies", torch.equal(out[:, 0], k[:, 0]) and torch.equal(out[:, 4], k[:, 0]), "")
    check("n_rep 1 identity", repeat_kv(k, 1) is k or torch.equal(repeat_kv(k, 1), k), "")


TESTS = [
    test_oracle_match_n1_and_n32,
    test_row_chunk_never_full_w,
    test_refuse_cuda_x,
    test_attach_cpu_home_not_pinned,
    test_nf4_linear_cpu_n32,
    test_repeat_kv_gqa_n_rep_5,
]


def main() -> int:
    print("gpu/host nf4_gemm_cpu vs oracle, CPU only\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
