"""Packed NF4 embed gather. CPU only — no GPU plate.

    python gpu/decodev2/test_embed.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.embed import PackedEmbed, bind_embed, gather_embed  # noqa: E402
from gpu.decodev2.graph import capture_greedy  # noqa: E402
from gpu.decodev2.linear import DeviceWeights  # noqa: E402
from gpu.decodev2.plan import TINY_LLAMA  # noqa: E402
from gpu.decodev2.runner import generate  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.decodev2.step import teacher_force_token  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.host.embedding import NF4_LEVELS, dequant_nf4_rows, dequant_table  # noqa: E402
from gpu.tests.nf4_oracle import encode_nf4  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _packed_of(table: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    packed, scale = encode_nf4(np.ascontiguousarray(table, dtype=np.float32))
    return (
        torch.from_numpy(np.ascontiguousarray(packed)),
        torch.from_numpy(np.ascontiguousarray(scale)),
    )


def test_bind_embed_keeps_packed() -> None:
    spec = TINY_LLAMA
    rng = np.random.default_rng(0)
    table = rng.standard_normal((spec.vocab, spec.hidden), dtype=np.float32) * np.float32(0.05)
    packed, scale = _packed_of(table)
    mod = SimpleNamespace(
        packed=packed,
        scale=scale,
        num_embeddings=spec.vocab,
        embedding_dim=spec.hidden,
        weight=torch.empty(0),
    )
    bound = bind_embed(mod, spec, torch.float32)
    assert isinstance(bound, PackedEmbed)
    dense_bytes = spec.vocab * spec.hidden * 4
    assert bound.nbytes < dense_bytes // 2
    ids = torch.tensor([0, 3, 7], dtype=torch.long)
    got = bound.gather(ids, dtype=torch.float32)
    want = dequant_nf4_rows(packed, scale, ids, spec.hidden, lut=bound.lut, dtype=torch.float32)
    assert torch.equal(got, want)


def test_gather_matches_dequant_table() -> None:
    spec = TINY_LLAMA
    rng = np.random.default_rng(1)
    table = rng.standard_normal((spec.vocab, spec.hidden), dtype=np.float32) * np.float32(0.05)
    packed, scale = _packed_of(table)
    lut = torch.tensor(NF4_LEVELS, dtype=torch.float32)
    pe = PackedEmbed(
        packed=packed, scale=scale, vocab=spec.vocab, hidden=spec.hidden, lut=lut
    )
    ids = torch.tensor([1, 2, 5, 8], dtype=torch.long)
    got = gather_embed(pe, ids, dtype=torch.float32)
    dense = dequant_table(
        SimpleNamespace(packed=packed, scale=scale, M=spec.vocab, K=spec.hidden),
        dtype=torch.float32,
    )
    want = dense.index_select(0, ids)
    assert torch.equal(got, want)
    dense_got = gather_embed(dense, ids, dtype=torch.float32)
    assert torch.equal(dense_got, want)


def test_packed_embed_step_matches_densified_same_codes() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    packed, scale = _packed_of(model.embed)
    lut = torch.tensor(NF4_LEVELS, dtype=torch.float32)
    dense = dequant_table(
        SimpleNamespace(packed=packed, scale=scale, M=spec.vocab, K=spec.hidden),
        dtype=torch.float32,
    )
    w_dense = DeviceWeights.from_synth(model, "cpu", torch.float32)
    w_packed = DeviceWeights.from_synth(model, "cpu", torch.float32)
    w_dense.embed = dense
    w_packed.embed = PackedEmbed(
        packed=packed, scale=scale, vocab=spec.vocab, hidden=spec.hidden, lut=lut
    )
    s_dense = DecodeState.allocate(spec, w_dense.embed, device="cpu", dtype=torch.float32)
    s_packed = DecodeState.allocate(spec, w_packed.embed, device="cpu", dtype=torch.float32)
    prompt = [1, 3, 5, 7]
    for tok in prompt:
        teacher_force_token(s_dense, w_dense, tok)
        teacher_force_token(s_packed, w_packed, tok)
    assert int(s_dense.next_token.item()) == int(s_packed.next_token.item())
    assert torch.allclose(s_dense.arena.logits, s_packed.arena.logits, atol=1e-5, rtol=1e-5)


def test_cuda_packed_graph_matches_eager() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_LLAMA
    model = build(spec, 0)
    packed, scale = _packed_of(model.embed)
    w = DeviceWeights.from_synth(model, "cuda", torch.bfloat16)
    w.embed = PackedEmbed(
        packed=packed.to("cuda"),
        scale=scale.to("cuda"),
        vocab=spec.vocab,
        hidden=spec.hidden,
        lut=torch.tensor(NF4_LEVELS, dtype=torch.float32, device="cuda"),
    )
    state = DecodeState.allocate(spec, w.embed, dtype=torch.bfloat16)
    captured = capture_greedy(state, w, warmup=2)
    state.reset()
    eager = generate(state, w, [4, 5, 6], 8)
    state.reset()
    graphed = generate(state, w, [4, 5, 6], 8, step=captured)
    assert graphed == eager, f"packed graph {graphed} != eager {eager}"


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 embed, {len(TESTS)} tests\n")
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
        except Exception as exc:  # pragma: no cover
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
