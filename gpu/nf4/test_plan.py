"""Launch math for ``chr_nf4_gemm``, asserted without a GPU.

    python -m gpu.nf4.test_plan
    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe -m gpu.nf4.test_plan

``docs/tz/wave9-review.md`` §3 states the wave-2 occupancy bug as a grid size,
not as a profile: decode launched ``ceil(M / 128)`` CTAs, which is 16 for a
Qwen2.5-3B ``q_proj`` and **2** for the GQA ``k_proj`` on a 70-SM card. Grid
sizes are integers, so every claim on this page is checkable on any machine.
What this file does *not* check is speed -- that is ``gpu/nf4/bench.py`` and it
needs the card.

1. the wave-2 grid is reproduced exactly, so the "16 and 2 CTAs" number is
   this tree's and not a quoted review;
2. the current plan puts more than one wave of CTAs on every 3B decode shape;
3. the K splits tile ``n_ktiles`` exactly: no split without work, no K tile
   counted twice, which is what makes the FP32 reduce a sum and not a guess;
4. the split-K workspace stays a reduction buffer -- orders of magnitude under
   a dense ``[M, K]`` BF16 weight, because "no resident dense W" has to survive
   the occupancy fix;
5. ``N`` outside 1..64 and a wrong ``K_pad`` are refused; N=17..64 is the
   planned wide family (not the live TokenLoop path);
6. when ``chr_nf4_ext`` happens to be importable **and up to date**, the
   pure-Python planner and the one compiled into the ``.cu`` agree field by
   field on live N=1..16. The ``.cu`` is the truth; a mismatch is a bug in
   ``gpu/nf4/plan.py``. This check skips (does not JIT) if sources are newer
   than the extension -- the 3B lab owns the 3080.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.plan import (  # noqa: E402
    CLASSIC,
    LIVE_MAX_N,
    ONE_WAVE,
    PLAN_MAX_N,
    PREFILL,
    PREFILL_N32,
    PREFILL_N64,
    SMALL,
    Plan,
    k_pad,
    plan,
)
from gpu.tests.skips import Skip, skip  # noqa: E402

# (name, M, K) for Qwen2.5-3B-Instruct: hidden 2048, intermediate 11008,
# 16 heads / 2 KV heads x 128, vocab 151936.
SHAPES_3B = (
    ("q_proj", 2048, 2048),
    ("k_proj", 256, 2048),
    ("o_proj", 2048, 2048),
    ("gate_proj", 11008, 2048),
    ("down_proj", 2048, 11008),
    ("lm_head", 151936, 2048),
)

#: The launch the shipped wave-2 kernel made, from nf4_gemm.cu before this wave:
#: grid = (ceil(M / 128), 1, 1) for N == 1 and (ceil(M / 64), 1, 1) for prefill.
WAVE2_CTAS_N1 = {"q_proj": 16, "k_proj": 2, "o_proj": 16, "gate_proj": 86,
                 "down_proj": 16, "lm_head": 1187}

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> bool:
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        _FAILS.append(msg)
    return cond


def test_wave2_grid() -> None:
    """The bug, reproduced from this tree's own arithmetic."""
    ok = True
    for name, m, k in SHAPES_3B:
        p = plan(m, k, 1, have_ws=False, force_path=CLASSIC)
        ok &= p.ctas == WAVE2_CTAS_N1[name] and p.grid_y == 1 and p.bm == 128
    check(ok, f"wave-2 decode grid is ceil(M/128) x 1: {WAVE2_CTAS_N1}")
    starved = [n for n, c in WAVE2_CTAS_N1.items() if c < ONE_WAVE]
    check(
        starved == ["q_proj", "k_proj", "o_proj", "down_proj"],
        f"wave 2 left the card idle on {starved} ({ONE_WAVE} SMs)",
    )


def _filled(p: Plan) -> bool:
    """One CTA per SM, or every K tile already has its own CTA.

    ``k_proj`` (M=256, K=2048) is the honest exception: 4 row tiles x 16 K tiles
    is 64 CTAs and there is no 65th without splitting K below one scale group.
    64 of 70 SMs is the tile's limit, not an oversight, so the assertion is
    "the card is full *or* the K dimension is fully split".
    """
    return p.ctas >= ONE_WAVE or p.grid_y == p.n_ktiles


def test_new_grid_fills_card() -> None:
    """Every 3B shape either fills the card or runs out of K to split."""
    rows = []
    ok = True
    for name, m, k in SHAPES_3B:
        p = plan(m, k, 1)
        ok &= _filled(p) and p.ctas >= WAVE2_CTAS_N1[name]
        rows.append(f"{name} {WAVE2_CTAS_N1[name]}->{p.ctas}")
    check(ok, "decode N=1 fills the card or saturates split_k: " + ", ".join(rows))

    ok = True
    rows = []
    for name, m, k in SHAPES_3B:
        for n in (2, 3, 4, 8, 9, 16):
            p = plan(m, k, n)
            ok &= p.path == PREFILL and _filled(p)
        rows.append(f"{name} {plan(m, k, 16).ctas}")
    check(ok, "prefill N=2..16 likewise: " + ", ".join(rows))

    p8 = plan(2048, 2048, 8)
    p9 = plan(2048, 2048, 9)
    p16 = plan(2048, 2048, 16)
    check(
        p8.smem_bytes == 18432
        and p9.smem_bytes == 24576
        and p16.smem_bytes == 24576
        and p8.smem_bytes < 99 * 1024
        and p16.smem_bytes < 99 * 1024,
        f"prefill BN=8 smem {p8.smem_bytes}B, BN=16 {p16.smem_bytes}B "
        f"(N=9 {p9.smem_bytes}B), both under 99 KiB",
    )

    # No shape may come out of the planner with fewer CTAs than wave 2 had.
    worst = min(
        plan(m, k, n).ctas / plan(m, k, n, have_ws=False,
                                 force_path=CLASSIC if n == 1 else None).ctas
        for _, m, k in SHAPES_3B
        for n in (1, 2, 4, 8, 16)
    )
    check(worst >= 1.0, f"no shape/N loses CTAs vs wave 2 (worst ratio {worst:.2f}x)")


def test_split_tiles_exactly() -> None:
    """``split_k`` x ``tiles_per_split`` must cover the K tiles once each."""
    ok = True
    bad = []
    for _, m, k in SHAPES_3B:
        for n in (1, 2, 4, 8, 16, 32, 64):
            for one_wave in (1, 16, 70, 140):
                p = plan(m, k, n, one_wave=one_wave)
                covered = p.grid_y * p.tiles_per_split
                last = p.n_ktiles - (p.grid_y - 1) * p.tiles_per_split
                if not (covered >= p.n_ktiles
                        and covered - p.tiles_per_split < p.n_ktiles
                        and 1 <= last <= p.tiles_per_split
                        and p.grid_y <= p.n_ktiles):
                    ok = False
                    bad.append((m, k, n, one_wave, p.grid_y, p.tiles_per_split))
    check(ok, f"K splits tile n_ktiles exactly, no idle split (bad: {bad[:3]})")

    zero = all(plan(m, k, n, have_ws=False).ws_floats == 0
               for _, m, k in SHAPES_3B for n in (1, 2, 16))
    check(zero, "no workspace offered -> split_k = 1, i.e. the wave-2 contract")


def test_workspace_is_not_a_dense_weight() -> None:
    """The occupancy fix must not smuggle ``[M, K]`` BF16 into HBM.

    The structural claim is that the workspace is one FP32 partial per *output*
    element per split -- ``split_k * M * N`` -- so it is sized by the answer and
    by how many pieces the answer was computed in, never by ``K``. A dense
    ``[M, K]`` BF16 weight is the thing ``CompressedLinear`` refuses to own, and
    it is 256x larger than the worst 3B workspace below.
    """
    shape_ok = True
    small_ok = True
    worst_ratio = 0.0
    worst_bytes = 0
    rows = []
    for name, m, k in SHAPES_3B:
        dense = m * k_pad(k) * 2  # BF16 [M, K]: the weight we never materialise
        for n in (1, 2, 4, 8, 16):
            p = plan(m, k, n)
            n_eff = n if p.path not in (CLASSIC, SMALL) else 1
            shape_ok &= p.ws_floats == (p.grid_y * m * n_eff if p.grid_y > 1 else 0)
            ws = p.ws_floats * 4
            small_ok &= ws < dense
            if ws / dense > worst_ratio:
                worst_ratio = ws / dense
            worst_bytes = max(worst_bytes, ws)
            if n == 1:
                rows.append(f"{name} {ws / 1024:.0f}KiB")
    check(shape_ok, "workspace is exactly split_k x M x N FP32 partials "
                    "(sized by the output, independent of K)")
    check(
        small_ok and worst_bytes <= (1 << 20),
        f"workspace is always under that shape's dense W (largest share "
        f"{worst_ratio * 100:.1f}%, largest absolute {worst_bytes / 1024:.0f} KiB); "
        f"at N=1: " + ", ".join(rows),
    )


def test_rejects() -> None:
    for n in (0, -1, PLAN_MAX_N + 1, 128):
        try:
            plan(2048, 2048, n)
        except ValueError:
            continue
        check(False, f"N={n} should be refused")
        return
    check(True, f"N outside 1..{PLAN_MAX_N} refused")

    try:
        plan(2048, 100, 1, K_pad=100)
    except ValueError:
        check(True, "K_pad != 64*ceil(K/64) refused")
    else:
        check(False, "K_pad != 64*ceil(K/64) should be refused")

    try:
        plan(0, 2048, 1)
    except ValueError:
        check(True, "M < 1 refused")
    else:
        check(False, "M < 1 should be refused")


def test_decode_cta_freeze() -> None:
    """Do not change the split-K decode counts that produced 128/64."""
    q = plan(2048, 2048, 1)
    k = plan(256, 2048, 1)
    check(
        q.ctas == 128 and q.path == SMALL and q.bm == 64,
        f"q_proj decode stays 128 CTAs (got {q.ctas} path={q.path_name})",
    )
    check(
        k.ctas == 64 and k.path == SMALL and k.bm == 64,
        f"k_proj decode stays 64 CTAs (got {k.ctas} path={k.path_name})",
    )


def test_wide_family_is_plan_only() -> None:
    """N<=32 is live n8/n16/n32. N=33..64 is planned n64. Same BM/BK as n16."""
    check(LIVE_MAX_N == 32, "live N is 32")
    p16 = plan(2048, 2048, 16)
    p17 = plan(2048, 2048, 17)
    p32 = plan(2048, 2048, 32)
    p33 = plan(2048, 2048, 33)
    p64 = plan(2048, 2048, 64)
    check(
        p16.path == PREFILL and p16.live and p16.smem_bytes == 24576,
        f"N=16 stays live n16 (path={p16.path_name} smem={p16.smem_bytes})",
    )
    check(
        p17.path == PREFILL_N32 and p17.live and p17.smem_bytes == 36864,
        f"N=17 is live n32 (path={p17.path_name} smem={p17.smem_bytes})",
    )
    check(
        p32.path == PREFILL_N32 and p32.live and p32.smem_bytes == 36864
        and p32.bm == p16.bm and p32.bk == p16.bk
        and p32.grid_x == p16.grid_x and p32.grid_y == p16.grid_y
        and p32.ctas == p16.ctas,
        f"N=32 shares n16 BM/BK/split-K ({p32.grid_x},{p32.grid_y}) "
        f"ctas={p32.ctas}",
    )
    check(
        p33.path == PREFILL_N64 and not p33.live and p64.path == PREFILL_N64
        and p64.smem_bytes == 61440
        and p64.smem_bytes < 99 * 1024
        and p32.smem_bytes < 99 * 1024,
        f"N=33..64 is planned n64 smem={p64.smem_bytes}B",
    )
    padded = plan(2048, 2048, 16, force_path=PREFILL_N32)
    check(
        padded.path == PREFILL,
        "force_path cannot select n32 for live N<=16",
    )
    src = Path(__file__).resolve().parents[1] / "loop" / "generate.py"
    text = src.read_text(encoding="utf-8")
    check(
        "nf4_max_n(LIVE_MAX_N)" in text,
        "TokenLoop.prefill_chunk still probes LIVE_MAX_N (not 64)",
    )

    wide_ok = True
    worst_ws = 0
    for _, m, k in SHAPES_3B:
        dense = m * k_pad(k) * 2
        for n in (17, 32, 64):
            p = plan(m, k, n)
            wide_ok &= p.ws_floats == (p.grid_y * m * n if p.grid_y > 1 else 0)
            wide_ok &= p.ws_floats * 4 <= dense
            worst_ws = max(worst_ws, p.ws_floats * 4)
            wide_ok &= _filled(p)
    check(
        wide_ok,
        f"wide N workspace is still split_k x M x N and at most dense W "
        f"(largest {worst_ws / 1024:.0f} KiB; k_proj N=64 ties BF16 W bytes)",
    )


def test_matches_extension() -> None:
    """Cross-check against the planner compiled into nf4_gemm.cu, if present.

    Without the extension this check cannot be made, so it is a skip and not a
    pass: a green tick here would claim ``plan.py`` had been compared against
    the ``.cu`` on a box that never loaded it (wave8-runtime D12).
    Does not JIT and does not import ``gpu.nf4`` (torch) until the inplace
    binary is up to date -- the 3B lab owns the 3080.
    """
    nf4_dir = Path(__file__).resolve().parent
    sources = (nf4_dir / "bindings.cpp", nf4_dir / "nf4_gemm.cu")
    try:
        from gpu.ext_bin import find_ext
    except Exception as exc:  # noqa: BLE001
        skip(f"C planner cross-check: {type(exc).__name__}: {exc}")
    matches = find_ext(nf4_dir, "chr_nf4_ext")
    if not matches:
        skip("C planner cross-check: no chr_nf4_ext (not JIT-loading on the live 3B lab)")
    pyd = matches[0]
    src_newer = any(s.is_file() and s.stat().st_mtime > pyd.stat().st_mtime for s in sources)
    if src_newer:
        skip(
            "C planner cross-check: chr_nf4_ext older than gpu/nf4 sources "
            "(not JIT-loading on the live 3B lab)"
        )
    try:
        from gpu.nf4 import nf4_plan, nf4_set_tuning
    except Exception as exc:  # noqa: BLE001 - no torch / no card is not a failure
        skip(f"C planner cross-check: {type(exc).__name__}: {exc}")
    try:
        nf4_set_tuning(path=0, split_k=0, one_wave=0)
        probe = nf4_plan(2048, 2048, 1)
    except Exception as exc:  # noqa: BLE001
        skip(f"C planner cross-check: {type(exc).__name__}: {exc}")

    fields = [f for f in Plan.__dataclass_fields__ if f in probe]
    ok = True
    for _, m, k in SHAPES_3B:
        for n in (1, 2, 3, 4, 8, 9, 16, 17, 32, 64):
            for have_ws in (True, False):
                c = nf4_plan(m, k, n, have_ws=have_ws)
                p = plan(m, k, n, have_ws=have_ws)
                for f in fields:
                    if getattr(p, f) != c[f]:
                        ok = False
                        print(f"  MISMATCH M={m} K={k} N={n} ws={have_ws} "
                              f"{f}: py={getattr(p, f)} cu={c[f]}")
    check(ok, f"gpu/nf4/plan.py matches nf4_gemm.cu on {len(fields)} fields")

    # The forced paths the bench uses have to line up too, or its labels lie.
    try:
        for path, const in ((1, CLASSIC), (2, SMALL)):
            nf4_set_tuning(path=path, split_k=1)
            c = nf4_plan(256, 2048, 1)
            p = plan(256, 2048, 1, have_ws=False, force_path=const)
            ok &= c["path"] == p.path and c["ctas"] == p.ctas
        check(ok, "nf4_set_tuning(path=1|2) matches force_path in plan.py")
    finally:
        nf4_set_tuning(path=0, split_k=0, one_wave=0)


def main() -> int:
    print(f"launch math only, no GPU needed (one_wave = {ONE_WAVE} SMs)\n")
    skipped = 0
    for fn in (
        test_wave2_grid,
        test_new_grid_fills_card,
        test_split_tiles_exactly,
        test_workspace_is_not_a_dense_weight,
        test_rejects,
        test_decode_cta_freeze,
        test_wide_family_is_plan_only,
        test_matches_extension,
    ):
        try:
            fn()
        except Skip as exc:
            print(f"SKIP {exc}")
            skipped += 1
    print()
    if _FAILS:
        print(f"SOME FAIL ({len(_FAILS)})")
        return 2
    print("ALL PASS" + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
