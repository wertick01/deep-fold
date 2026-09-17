"""CPU oracle for Decode V2 synthetic models.

    python gpu/decodev2/test_oracle.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.oracle import OracleCache, greedy_ids, step_logits, teacher_logits  # noqa: E402
from gpu.decodev2.plan import TINY_INTERNLM, TINY_LLAMA  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.tests.nf4_oracle import selfcheck  # noqa: E402
from gpu.tests.skips import Skip  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []
PROMPT = [1, 3, 5, 7]


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_nf4_selfcheck() -> None:
    bad = [c.ident for c in selfcheck() if not c.ok]
    assert not bad, f"nf4_oracle.selfcheck failed {bad}"


def test_build_families() -> None:
    llama = build(TINY_LLAMA, 0)
    intern = build(TINY_INTERNLM, 1)
    assert llama.layers[0].wqkv is None and llama.layers[0].q is not None
    assert intern.layers[0].wqkv is not None and intern.layers[0].q is None
    assert intern.layers[0].wqkv.M == TINY_INTERNLM.wqkv_out
    assert np.isfinite(llama.embed).all()
    assert np.isfinite(intern.lm_head.packed).all()


def test_greedy_deterministic() -> None:
    m = build(TINY_LLAMA, 0)
    a = greedy_ids(m, PROMPT, 8)
    b = greedy_ids(m, PROMPT, 8)
    assert a == b and len(a) >= 1


def test_teacher_deterministic() -> None:
    m = build(TINY_INTERNLM, 1)
    a = teacher_logits(m, PROMPT).argmax(axis=-1).tolist()
    b = teacher_logits(m, PROMPT).argmax(axis=-1).tolist()
    assert a == b and len(a) == len(PROMPT)


def test_dirty_tail_masked() -> None:
    m = build(TINY_LLAMA, 0)
    clean = OracleCache(m.spec)
    dirty = OracleCache(m.spec)
    step_logits(m, clean, 1)
    step_logits(m, dirty, 1)
    dirty.k[:, -1] = np.float32(1.0e4)
    dirty.v[:, -1] = np.float32(1.0e4)
    y_c = step_logits(m, clean, 3)
    y_d = step_logits(m, dirty, 3)
    assert int(y_c.argmax()) == int(y_d.argmax()), "tail must be masked"
    assert np.allclose(y_c, y_d, atol=1e-5, rtol=1e-5)


def test_eos_stop() -> None:
    spec = TINY_LLAMA
    for seed in range(64):
        m = build(spec, seed)
        for tok in range(spec.vocab):
            logits = teacher_logits(m, [tok])
            if int(logits[0].argmax()) != spec.eos_id:
                continue
            ids = greedy_ids(m, [tok], 8)
            assert ids == [spec.eos_id], ids
            return
    raise AssertionError("no seed/token whose first generated id is eos")


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 oracle, {len(TESTS)} tests\n")
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
