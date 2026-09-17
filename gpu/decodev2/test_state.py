"""Decode V2 state / KV contract. No checkpoints.

    python gpu/decodev2/test_state.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.plan import MEDIUM_LLAMA, TINY_INTERNLM, TINY_LLAMA, ArchSpec  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_specs_validate() -> None:
    assert TINY_LLAMA.family == "llama_swiglu"
    assert TINY_INTERNLM.family == "internlm_gqa"
    assert TINY_LLAMA.hidden == 64 and TINY_LLAMA.n_rep == 2
    assert MEDIUM_LLAMA.hidden == 512 and MEDIUM_LLAMA.n_layers == 4
    assert TINY_INTERNLM.wqkv_out == (4 + 2 * 2) * 16
    try:
        ArchSpec(
            name="bad",
            family="llama_swiglu",
            n_layers=1,
            hidden=65,
            n_q=4,
            n_kv=2,
            head_dim=16,
            intermediate=128,
            vocab=8,
            max_seq=8,
        )
    except ValueError:
        return
    raise AssertionError("hidden not multiple of 64 must fail")


def _cpu_state() -> DecodeState:
    spec = TINY_LLAMA
    embed = torch.randn(spec.vocab, spec.hidden)
    return DecodeState.allocate(spec, embed, device="cpu", dtype=torch.float32)


def test_reset_keeps_dirty_kv() -> None:
    st = _cpu_state()
    st.kv.k[0, 3].fill_(7.0)
    st.valid_len.fill_(4)
    st.reset()
    assert int(st.position.item()) == 0
    assert int(st.valid_len.item()) == 0
    assert float(st.kv.k[0, 3, 0, 0].item()) == 7.0


def test_mask_hides_dirty_tail() -> None:
    st = _cpu_state()
    st.kv.k.fill_(0)
    st.kv.k[0, 7].fill_(99.0)
    st.valid_len.fill_(1)
    mask = st.kv.additive_mask(st.arena.attn_mask)
    assert float(mask[0, 0, 0, 0].item()) == 0.0
    assert float(mask[0, 0, 0, 7].item()) < -1.0e6


def test_write_at_gpu_position() -> None:
    st = _cpu_state()
    st.position.fill_(3)
    k = torch.ones(1, st.spec.n_kv, st.spec.head_dim)
    v = torch.full((1, st.spec.n_kv, st.spec.head_dim), 2.0)
    st.kv.write(0, st.position, k, v)
    st.kv.mark_written(st.position)
    assert int(st.valid_len.item()) == 4
    assert torch.equal(st.kv.k[0, 3], k[0])
    assert torch.equal(st.kv.v[0, 3], v[0])
    k_att, _ = st.kv.attn(0)
    assert k_att.shape == (1, st.spec.n_kv, st.spec.max_seq, st.spec.head_dim)


def test_write_range_and_view() -> None:
    st = _cpu_state()
    n = 3
    k = torch.arange(n * st.spec.n_kv * st.spec.head_dim, dtype=torch.float32).reshape(
        n, st.spec.n_kv, st.spec.head_dim
    )
    v = k + 1
    st.kv.write_range(0, 2, k, v)
    k_win, v_win = st.kv.view(0, 5)
    assert k_win.shape == (1, st.spec.n_kv, 5, st.spec.head_dim)
    assert torch.equal(st.kv.k[0, 2:5], k)
    assert torch.equal(st.kv.v[0, 2:5], v)
    assert torch.equal(k_win[0, :, 2:5, :], k.permute(1, 0, 2))


def test_commit_feeds_token_on_device() -> None:
    st = _cpu_state()
    st.next_token.fill_(st.spec.eos_id)
    st.commit_step()
    assert int(st.token.item()) == st.spec.eos_id
    assert int(st.position.item()) == 1
    assert int(st.finished.item()) == 1


def test_rope_index_uses_position_tensor() -> None:
    st = _cpu_state()
    st.position.fill_(5)
    cos, sin = st.rope_at_position()
    assert tuple(cos.shape) == (1, st.spec.head_dim)
    assert torch.equal(cos[0], st.cos[5])


def test_cuda_allocate() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_LLAMA
    embed = torch.randn(spec.vocab, spec.hidden, device="cuda")
    st = DecodeState.allocate(spec, embed, dtype=torch.bfloat16)
    assert st.token.is_cuda and st.kv.k.is_cuda
    ptr = st.arena.x.data_ptr()
    st.reset()
    st.load_token(4)
    assert st.arena.x.data_ptr() == ptr


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 state, {len(TESTS)} tests\n")
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
