"""GPU-free: ``nf4_linear`` does not pad tails; the kernel owns N=2..16.

    python -m gpu.host.test_linear
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host import linear as linear_mod  # noqa: E402

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        _FAILS.append(msg)


def main() -> int:
    calls: list[tuple[int, int]] = []

    def fake(packed, scale, x, M, K, K_pad):  # noqa: N803
        del packed, scale, K_pad
        calls.append((int(x.shape[0]), int(x.shape[1])))
        return x.new_zeros(M, x.shape[1])

    original = linear_mod.nf4_gemm
    linear_mod.nf4_gemm = fake
    try:
        m, k, k_pad = 8, 32, 64
        packed = torch.zeros(m, k_pad // 2, dtype=torch.uint8)
        scale = torch.zeros(m, k_pad // 64, dtype=torch.float16)

        y1 = linear_mod.nf4_linear(
            torch.zeros(1, k, dtype=torch.bfloat16), packed, scale, m, k, k_pad
        )
        check(calls[-1] == (k, 1), "N=1 stays decode width")
        check(tuple(y1.shape) == (1, m), f"N=1 out {tuple(y1.shape)}")

        y3 = linear_mod.nf4_linear(
            torch.zeros(1, 3, k, dtype=torch.bfloat16), packed, scale, m, k, k_pad
        )
        check(calls[-1] == (k, 3), "N=3 is not padded to 16 on the host")
        check(tuple(y3.shape) == (1, 3, m), f"N=3 out {tuple(y3.shape)}")

        linear_mod.nf4_linear(
            torch.zeros(1, 16, k, dtype=torch.bfloat16), packed, scale, m, k, k_pad
        )
        check(calls[-1] == (k, 16), "N=16 is one prefill launch")

        calls.clear()
        y19 = linear_mod.nf4_linear(
            torch.zeros(1, 19, k, dtype=torch.bfloat16), packed, scale, m, k, k_pad
        )
        check(calls == [(k, 16), (k, 3)], f"N=19 chunks 16+3 {calls}")
        check(tuple(y19.shape) == (1, 19, m), f"N=19 out {tuple(y19.shape)}")

        calls.clear()
        linear_mod.nf4_linear(
            torch.zeros(1, 17, k, dtype=torch.bfloat16), packed, scale, m, k, k_pad
        )
        check(calls == [(k, 16), (k, 1)], f"N=17 remainder stays decode {calls}")
    finally:
        linear_mod.nf4_gemm = original

    if _FAILS:
        print(f"\n{len(_FAILS)} FAIL")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
