"""CPU: nf4_gemv_cpu vs dequant_nf4_rows @ x. No GPU.

    python gpu/lab/test_h2_gates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.embedding import GROUP_SIZE, dequant_nf4_rows
from gpu.host.host_image import k_pad
from gpu.lab.h2_gates import nf4_gemv_cpu, parse_k

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_parse_k() -> None:
    check("k list", parse_k("1,2,4,8") == [1, 2, 4, 8], "")
    try:
        parse_k("33")
    except Exception as exc:
        check("k=33 raises", "33" in str(exc), str(exc)[:80])
    else:
        check("k=33 raises", False, "no error")


def test_gemv_matches_dequant() -> None:
    torch.manual_seed(1)
    m, k = 32, 128
    pad = k_pad(k)
    packed = torch.randint(0, 256, (m, pad // 2), dtype=torch.uint8)
    n_groups = pad // GROUP_SIZE
    scale = torch.randn(m, n_groups, dtype=torch.float16) * 0.05
    x = torch.randn(k, dtype=torch.float32)
    y = nf4_gemv_cpu(packed, scale, x, k=k)
    w = dequant_nf4_rows(packed, scale, torch.arange(m), k, dtype=torch.float32)
    y_ref = w.matmul(x)
    err = float((y - y_ref).abs().max())
    check("shape [M]", int(y.numel()) == m, str(tuple(y.shape)))
    check("max abs vs dequant @ x < 2e-4", err < 2e-4, f"{err:.3e}")


TESTS = [test_parse_k, test_gemv_matches_dequant]


def main() -> int:
    print("gpu/lab h2_gates CPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
