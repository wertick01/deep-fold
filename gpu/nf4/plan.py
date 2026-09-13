"""Host-side mirror of the ``chr_nf4_gemm`` launch planner.

Wave-2 decode launched ``grid = (ceil(M / 128), 1)``, ``split_k = 1``: 16
CTAs on a 3B ``q_proj`` and 2 on GQA ``k_proj`` against 70 SMs. Auto decode is
the occupancy fix: BM=64 plus split-K (128 / 64 CTAs on those shapes). This
file duplicates the ``.cu`` arithmetic so occupancy can be asserted without a
GPU; if they diverge, the kernel is the truth.

``CLASSIC`` BM=128 ``N==1``, force only (wave-2). ``SMALL`` BM=64 ``N==1`` auto.
``PREFILL`` BM=64, ``N`` in 2..16 (BN=8 smem when N<=8, BN=16 otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CLASSIC",
    "SMALL",
    "PREFILL",
    "ONE_WAVE",
    "TARGET_CTAS",
    "GROUP_SIZE",
    "Plan",
    "k_pad",
    "plan",
]

CLASSIC = 0
SMALL = 1
PREFILL = 2

GROUP_SIZE = 64

#: SMs on GA102-200 (RTX 3080 12 GB). A decode grid below this is the wave-2 bug,
#: and the unit the split-K target is expressed in.
ONE_WAVE = 70
#: Two CTAs per SM, which is what ``__launch_bounds__`` asks for.
TARGET_CTAS = 140

_TILES = {
    #        bm,  bk, block, smem
    CLASSIC: (128, 256, 256, 3 * 128 * 128 + 3 * 256 * 2),
    SMALL: (64, 128, 128, 3 * 64 * 64 + 3 * 128 * 2),
    PREFILL: (64, 128, 256, 3 * 64 * 64 + 3 * 128 * 16 * 2),
}


def _prefill_smem(n: int) -> int:
    """BN=8 for N<=8, BN=16 for N=9..16. Must match nf4_gemm.cu plan_impl."""
    bn = 8 if int(n) <= 8 else 16
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
        return {CLASSIC: "classic", SMALL: "small", PREFILL: "prefill"}[self.path]

    @property
    def waves(self) -> float:
        """CTAs per SM on this card. Below 1.0 some SM never gets work."""
        return self.ctas / ONE_WAVE


def _pick_split(grid_x: int, n_ktiles: int, have_ws: bool, force_split: int,
                one_wave: int) -> tuple[int, int]:
    """``(split, tiles_per_split)``.

    Aim ``grid_x * split`` at the CTA target, then cap the split at
    ``n_ktiles``: a split with no K tiles to walk is a CTA that writes zeros.
    ``split`` is recomputed from ``tiles_per_split`` so the pair always agrees
    with what the kernel indexes.
    """
    want = 1
    if have_ws:
        target = max(TARGET_CTAS, one_wave * 2)
        want = force_split if force_split > 0 else _ceil_div(target, grid_x)
    want = max(1, min(want, n_ktiles))
    tiles_per_split = _ceil_div(n_ktiles, want)
    return _ceil_div(n_ktiles, tiles_per_split), tiles_per_split


def plan(
    M: int,  # noqa: N803 - out_features, the kernel's name for it
    K: int,  # noqa: N803
    N: int = 1,  # noqa: N803 - sequence columns, 1..16
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
    """
    M, K, N = int(M), int(K), int(N)
    if M < 1 or K < 1:
        raise ValueError(f"M and K must be >= 1, got M={M} K={K}")
    if not 1 <= N <= 16:
        raise ValueError(f"N={N} not in 1..16 (the host chunks N>16)")
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
    else:
        path = PREFILL

    bm, bk, block, smem = _TILES[path]
    if path == PREFILL:
        smem = _prefill_smem(N)
    grid_x = _ceil_div(M, bm)
    n_ktiles = _ceil_div(kp, bk)

    if path == CLASSIC:
        split, tiles_per_split = 1, n_ktiles
    else:
        split, tiles_per_split = _pick_split(
            grid_x, n_ktiles, have_ws, force_split, one_wave
        )

    ws_floats = split * M * (N if path == PREFILL else 1) if split > 1 else 0
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
