"""CPU: VQ row reconstruct and the embedding seat. No GPU, no .chr.

    python gpu/host/test_vq_cpu.py
    python -m pytest gpu/host/test_vq_cpu.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.vq_blobs import (  # noqa: E402
    VqMatrix,
    dequant_vq_rows,
    k_pad_vq,
    reconstruct_vq,
)
from gpu.host.vq_linear import VqEmbedding  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _toy(m: int = 6, k: int = 10) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    torch.manual_seed(0)
    k_pad = k_pad_vq(k)
    g = k_pad // 8
    index = torch.randint(0, 256, (m, g, 2), dtype=torch.uint8)
    book = torch.randn(2, 256, 8, dtype=torch.float16)
    return index, book, m, k, k_pad


def test_dequant_rows_match_full_table_gather() -> None:
    index, book, m, k, k_pad = _toy()
    table = reconstruct_vq(index, book, m, k, k_pad)
    ids = torch.tensor([0, 2, 5, 2])
    rows = dequant_vq_rows(index, book, ids, k, dtype=torch.float32)
    want = table[ids]
    check(
        "row gather matches reconstruct",
        torch.allclose(rows, want, atol=0, rtol=0),
        f"max |d|={(rows - want).abs().max().item()}",
    )


def test_vq_embedding_forward_shape_and_pad() -> None:
    index, book, m, k, k_pad = _toy()
    emb = VqEmbedding(m, k, padding_idx=0, dtype=torch.float32)
    emb.attach(VqMatrix(name="embed", M=m, K=k, K_pad=k_pad, index=index, book=book))
    ids = torch.tensor([[0, 2], [5, 1]])
    out = emb(ids)
    check("embedding rank-3", tuple(out.shape) == (2, 2, k), str(tuple(out.shape)))
    table = reconstruct_vq(index, book, m, k, k_pad)
    check(
        "embedding matches reconstruct",
        torch.allclose(out, table[ids], atol=0, rtol=0),
        "",
    )


def test_encode_four_templates_near_exact() -> None:
    from gpu.host.vq_encode import encode_vq, reconstruct_encoded

    templates = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    weight = templates.repeat(4, 1)
    index, book = encode_vq(weight, iters=8, seed=0)
    hat = reconstruct_encoded(index, book, 8)
    err = (weight - hat).abs().max().item()
    check("encode 4 templates maxabs < 1e-3", err < 1e-3, f"maxabs={err:.2e}")


def test_vq_linear_refuses_prefill_width() -> None:
    from gpu.host.vq_linear import vq_linear

    index, book, m, k, k_pad = _toy(m=8, k=16)
    x = torch.zeros(3, k, dtype=torch.bfloat16)
    try:
        vq_linear(x, index, book, m, k, k_pad)
    except NotImplementedError as exc:
        check("N=3 is decode-only", "N=3" in str(exc) or "decode-only" in str(exc), str(exc)[:80])
        return
    check("N=3 is decode-only", False, "no error")


TESTS = [
    test_dequant_rows_match_full_table_gather,
    test_vq_embedding_forward_shape_and_pad,
    test_encode_four_templates_near_exact,
    test_vq_linear_refuses_prefill_width,
]


def main() -> int:
    print("gpu/host VQ CPU acceptance, no GPU\n")
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
