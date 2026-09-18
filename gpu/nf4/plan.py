"""Host-side mirror of the ``chr_nf4_gemm`` launch planner.

Wave-2 decode launched ``grid = (ceil(M / 128), 1)``, ``split_k = 1``: 16
CTAs on a 3B ``q_proj`` and 2 on GQA ``k_proj`` against 70 SMs. Auto decode is
the occupancy fix: BM=64 plus split-K (128 / 64 CTAs on those shapes). This
file duplicates the ``.cu`` arithmetic so occupancy can be asserted without a
GPU; if they diverge, the kernel is the truth.

Kernel families (sequence ``N``; some notes call this M):

* ``N==1`` decode: ``SMALL`` BM=64 auto, ``CLASSIC`` BM=128 force only (wave-2).
* ``N=2..8`` / ``N=9..16``: BN=8 / BN=16. ``N=17..32`` BN=32 (live if
  ``LIVE_MAX_N>=32``). ``N=33..64`` planned BN=64.

``LIVE_MAX_N == 32`` is the TokenLoop / live ``chr_nf4_gemm`` ceiling. Tile
family is 8/16/32/64 by ``N``, not by this cap. n64 stays plan-only. Decode
N=1 is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CLASSIC",
    "SMALL",
    "PREFILL",
    "PREFILL_N32",
    "PREFILL_N64",
    "ONE_WAVE",
    "TARGET_CTAS",
    "GROUP_SIZE",
    "LIVE_MAX_N",
    "PLAN_MAX_N",
    "Plan",
    "k_pad",
    "plan",
]

CLASSIC = 0
SMALL = 1
PREFILL = 2
PREFILL_N32 = 3
PREFILL_N64 = 4

GROUP_SIZE = 64

#: SMs on GA102-200 (RTX 3080 12 GB) and on GB203-200 (RTX 5070 Ti).
#: A decode grid below this is the wave-2 bug, and the unit the split-K
#: target is expressed in when the caller does not pass ``one_wave``.
ONE_WAVE = 70
#: Two CTAs per SM on that 70-SM default. Live C planner uses
#: ``2 * cudaDeviceProp.multiProcessorCount``.
TARGET_CTAS = 140

#: Live ``chr_nf4_gemm`` / TokenLoop ceiling. Raising this is the unfreeze.
LIVE_MAX_N = 32
#: Planner can describe N=17..64 (BN=32 / BN=64). Launch still returns -2.
PLAN_MAX_N = 64

_TILES = {
    #        bm,  bk, block, smem
    CLASSIC: (128, 256, 256, 3 * 128 * 128 + 3 * 256 * 2),
    SMALL: (64, 128, 128, 3 * 64 * 64 + 3 * 128 * 2),
    PREFILL: (64, 128, 256, 3 * 64 * 64 + 3 * 128 * 16 * 2),
    PREFILL_N32: (64, 128, 256, 3 * 64 * 64 + 3 * 128 * 32 * 2),
    PREFILL_N64: (64, 128, 256, 3 * 64 * 64 + 3 * 128 * 64 * 2),
}

_PATH_NAME = {
    CLASSIC: "classic",
    SMALL: "small",
    PREFILL: "prefill",
    PREFILL_N32: "prefill_n32",
    PREFILL_N64: "prefill_n64",
}


def _prefill_smem(n: int) -> int:
    """BN for this N. Must match nf4_gemm.cu plan_impl.

    N=2..8 -> 8; 9..16 -> 16 (live); 17..32 -> 32; 33..64 -> 64 (planned).
    """
    n = int(n)
    if n <= 8:
        bn = 8
    elif n <= 16:
        bn = 16
    elif n <= 32:
        bn = 32
    else:
        bn = 64
    return 3 * 64 * 64 + 3 * 128 * bn * 2


def k_pad(k: int) -> int:
    """``64 * ceil(K / 64)`` -- the NF4 group size, docs/spec/nf4.md §1."""
    return GROUP_SIZE * ((int(k) + GROUP_SIZE - 1) // GROUP_SIZE)


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


@dataclass(frozen=True)
class Plan:
    """What ``chr_nf4_gemm_ws`` will launch. ``ctas`` is the occupancy number."""

    path: int
    grid_x: int
    grid_y: int
    block: int
    bm: int
    bk: int
    n_ktiles: int
    tiles_per_split: int
    ctas: int
    smem_bytes: int
    ws_floats: int

    @property
    def path_name(self) -> str:
        return _PATH_NAME[self.path]

    @property
    def waves(self) -> float:
        """CTAs per SM on this card. Below 1.0 some SM never gets work."""
        return self.ctas / ONE_WAVE

    @property
    def live(self) -> bool:
        """True for a tile TokenLoop may launch. n64 stays plan-only."""
        if self.path in (CLASSIC, SMALL, PREFILL):
            return True
        if LIVE_MAX_N >= 32 and self.path == PREFILL_N32:
            return True
        if LIVE_MAX_N >= 64 and self.path == PREFILL_N64:
            return True
        return False


def _pick_split(grid_x: int, n_ktiles: int, have_ws: bool, force_split: int,
                one_wave: int) -> tuple[int, int]:
    """``(split, tiles_per_split)``.

    Aim ``grid_x * split`` at ``2 * one_wave`` CTAs, then cap the split at
    ``n_ktiles``: a split with no K tiles to walk is a CTA that writes zeros.
    ``split`` is recomputed from ``tiles_per_split`` so the pair always agrees
    with what the kernel indexes. Default ``one_wave`` is 70 (RTX 3080 and
    RTX 5070 Ti).
    """
    want = 1
    if have_ws:
        target = max(1, int(one_wave) * 2)
        want = force_split if force_split > 0 else _ceil_div(target, grid_x)
    want = max(1, min(want, n_ktiles))
    tiles_per_split = _ceil_div(n_ktiles, want)
    return _ceil_div(n_ktiles, tiles_per_split), tiles_per_split


def plan(
    M: int,  # noqa: N803 - out_features, the kernel's name for it
    K: int,  # noqa: N803
    N: int = 1,  # noqa: N803 - sequence columns, 1..64 (1..16 live)
    *,
    K_pad: int | None = None,  # noqa: N803
    have_ws: bool = True,
    force_path: int | None = None,
    force_split: int = 0,
    one_wave: int = ONE_WAVE,
) -> Plan:
    """Launch plan for ``y[M,N] = dequant_nf4(W[M,K]) @ x[K,N]``.

    Auto decode (``N==1``) is BM=64 plus split-K (occupancy fix).
    ``have_ws=False`` pins ``split_k`` to 1 (plain ``chr_nf4_gemm``).

    ``N=2..16`` is n8/n16. ``N=17..32`` is n32 when ``LIVE_MAX_N>=32``.
    ``N=33..64`` is planned n64 (host still chunks above ``LIVE_MAX_N``).
    """
    M, K, N = int(M), int(K), int(N)
    if M < 1 or K < 1:
        raise ValueError(f"M and K must be >= 1, got M={M} K={K}")
    if not 1 <= N <= PLAN_MAX_N:
        raise ValueError(
            f"N={N} not in 1..{PLAN_MAX_N} (live launch is 1..{LIVE_MAX_N}; "
            f"the host chunks TokenLoop above {LIVE_MAX_N})"
        )
    kp = k_pad(K) if K_pad is None else int(K_pad)
    if kp != k_pad(K):
        raise ValueError(f"K_pad={kp} != 64*ceil(K/64) = {k_pad(K)}")

    if N == 1:
        # Auto is the small tile at every measured 3B shape, not only the ones
        # that starved the card -- see the table in nf4_gemm.cu's planner. The
        # 128-row tile stays reachable through force_path for the comparison.
        path = SMALL if force_path is None else (
            SMALL if force_path == SMALL else CLASSIC
        )
    elif N <= 16:
        # BN=8 / BN=16 tile width. LIVE_MAX_N is the TokenLoop launch cap, not
        # this cut: if it were LIVE_MAX_N, raising the cap to 32 would plan
        # N=32 as n16 (BN=16 smem, N=32) and overflow the x stage.
        path = PREFILL
    elif N <= 32:
        path = PREFILL_N32
    else:
        path = PREFILL_N64

    bm, bk, block, smem = _TILES[path]
    if path in (PREFILL, PREFILL_N32, PREFILL_N64):
        smem = _prefill_smem(N)
    grid_x = _ceil_div(M, bm)
    n_ktiles = _ceil_div(kp, bk)

    if path == CLASSIC:
        split, tiles_per_split = 1, n_ktiles
    else:
        split, tiles_per_split = _pick_split(
            grid_x, n_ktiles, have_ws, force_split, one_wave
        )

    uses_n = path not in (CLASSIC, SMALL)
    ws_floats = split * M * (N if uses_n else 1) if split > 1 else 0
    return Plan(
        path=path,
        grid_x=grid_x,
        grid_y=split,
        block=block,
        bm=bm,
        bk=bk,
        n_ktiles=n_ktiles,
        tiles_per_split=tiles_per_split,
        ctas=grid_x * split,
        smem_bytes=smem,
        ws_floats=ws_floats,
    )
