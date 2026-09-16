"""i4c pack + AVX2 GEMV vs decode oracle. No GPU.

    python gpu/host/test_i4c.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.i4c import i4c_gemm_cpu, pack_i4c_from_nf4, pack_i4c_torch  # noqa: E402
from gpu.tests.i4c_oracle import GROUP, decode_i4c, k_pad_i4c, pack_i4c  # noqa: E402
from gpu.tests.nf4_oracle import decode_nf4, toy_nf4  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_pack_roundtrip() -> None:
    torch.manual_seed(0)
    w = torch.randn(24, 65)
    packed, scale, kp = pack_i4c_torch(w)
    hat = torch.from_numpy(decode_i4c(packed.numpy(), scale.numpy(), 24, 65))
    err = float((hat - w).abs().max())
    # INT4 absmax/8: worst step is 0.5*scale; scale ~ max/8 so step ~ max/16.
    check("pack shape", packed.shape == (24, kp // 2) and scale.shape[1] == kp // GROUP, "")
    check("roundtrip finite", torch.isfinite(hat).all().item(), "")
    check("roundtrip maxabs < 1.0", err < 1.0, f"maxabs={err:.4f}")


def test_kernel_matches_decode() -> None:
    for m, k, seed in ((32, 64, 1), (17, 65, 2), (8, 32, 3)):
        torch.manual_seed(seed)
        w = torch.randn(m, k)
        packed, scale, kp = pack_i4c_torch(w)
        x = torch.randn(k)
        w_hat = torch.from_numpy(decode_i4c(packed.numpy(), scale.numpy(), m, k))
        y_ref = w_hat @ x.float()
        y_py = i4c_gemm_cpu(packed, scale, x, m, k, kp, dtype=torch.float32, impl="python")
        err_py = float((y_py.view(-1) - y_ref).abs().max())
        check(f"python vs decode M={m} K={k}", err_py < 1e-3, f"maxabs={err_py:.2e}")
        y_av = i4c_gemm_cpu(packed, scale, x, m, k, kp, dtype=torch.float32, impl="avx2")
        err_av = float((y_av.view(-1) - y_ref).abs().max())
        check(f"avx2 vs decode M={m} K={k}", err_av < 1e-3, f"maxabs={err_av:.2e}")


def test_pack_from_nf4() -> None:
    m, k = 16, 65
    packed_np, scale_np = toy_nf4(m, k, seed=9)
    w = torch.from_numpy(decode_nf4(packed_np, scale_np, m, k).astype("float32"))
    p_ref, s_ref, kp_ref = pack_i4c_torch(w)
    p, s, kp = pack_i4c_from_nf4(
        torch.from_numpy(packed_np),
        torch.from_numpy(scale_np),
        m,
        k,
        row_chunk=8,
    )
    check("from_nf4 K_pad", kp == kp_ref, f"{kp} vs {kp_ref}")
    check("from_nf4 packed", torch.equal(p, p_ref), "")
    check("from_nf4 scale", torch.equal(s, s_ref), "")


def test_n_gt1_python() -> None:
    torch.manual_seed(4)
    m, k, n = 12, 64, 4
    w = torch.randn(m, k)
    packed, scale, kp = pack_i4c_torch(w)
    x = torch.randn(k, n)
    y = i4c_gemm_cpu(packed, scale, x, m, k, kp, dtype=torch.float32, impl="python")
    hat = torch.from_numpy(decode_i4c(packed.numpy(), scale.numpy(), m, k))
    err = float((y - (hat @ x.float())).abs().max())
    check("N=4 python vs decode", err < 1e-3, f"maxabs={err:.2e}")


def main() -> int:
    print("i4c CPU")
    test_pack_roundtrip()
    test_kernel_matches_decode()
    test_pack_from_nf4()
    test_n_gt1_python()
    failed = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
