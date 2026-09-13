"""How a run of NF4 linears is executed: eager, or captured as CUDA graph plan A.

``docs/token-loop.md`` §5 counts the launches on a decode token and finds that
on Ampere (5-10 us per launch) a 3B step spends milliseconds just *starting*
kernels. Plan A is the cheap half of the fix: **only the NF4 GEMMs go into
graphs**. Norms, RoPE, SDPA and the KV write stay eager, so the graph never sees
a shape that depends on ``seq_len`` and one capture is valid for the whole
session -- that is the entire point of plan A over plan B's length buckets.

The unit of capture is a :class:`GemmGroup`: a *contiguous run* of GEMMs that
share one input. On Qwen2.5-3B a decode step has four per layer

    (q, k, v) -> attention -> (o) -> (gate, up) -> (down)

plus ``lm_head``, i.e. 145 groups covering all 253 GEMM launches. Capturing a
single GEMM per graph would trade a launch for a replay and buy nothing; a group
of three trades three launches (plus its bias adds) for one replay.

:class:`GemmGroup` and :class:`GraphedGemmGroup` expose the same ``run(x)``, so
``generate.py`` has one code path and ``graph=off`` is not a second
implementation -- the graph is captured *from* the eager group. If capture fails
-- a driver that refuses ``cudaFuncSetAttribute`` mid-capture, say --
:func:`capture` returns the eager groups unchanged and the loop still runs (the
TZ requires eager to PASS on its own).

Against the wave-2 kernel, plan A did **not** move tok/s: a decode token spent
~58 ms inside the GEMMs and ~5 ms launching them, so the launch tax the graph
removes was already hidden behind the kernel, and the 145 static-input copies it
adds cost about as much (17.7 tok/s eager vs 16.8 captured). What moved tok/s was
the stream fork in :class:`GemmGroup`, which the graph then captures for free.

The prediction in that paragraph -- "plan A earns its keep on a faster kernel" --
is now measured. With the split-K decode tile the same 3B smoke run gives **23.3
tok/s eager vs 31.0 captured**: the GEMMs no longer hide the launch tax, so the
replay is worth more than the static copies it costs. 14B/32B should widen it
further, since the Python cost per layer is unchanged and there are more layers.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "Gemm",
    "GemmGroup",
    "GraphedGemmGroup",
    "capture",
    "nf4_gemm",
    "nf4_max_n",
]

_gemm = None


def nf4_gemm(packed, scale, x, M, K, K_pad):  # noqa: N803
    """Agent 2's kernel, resolved once. Importing it may trigger a JIT build."""
    global _gemm
    if _gemm is None:
        from gpu.nf4 import nf4_gemm as fn

        _gemm = fn
    return _gemm(packed, scale, x, M, K, K_pad)


def nf4_max_n(probe: int = 16) -> int:
    """Largest ``N`` (sequence columns) the installed kernel accepts.

    Wave 2 shipped a decode-only GEMM (``N != 1`` -> rc -2, docs/spec/stitch-gpu.md),
    which decides whether the prompt can be prefilled in one forward or has to be
    walked token by token. Asked, not assumed: agent 6 may have widened it.
    """
    if not torch.cuda.is_available():
        return 1
    m = k = 64
    k_pad = 64
    packed = torch.zeros((m, k_pad // 2), dtype=torch.uint8, device="cuda")
    scale = torch.ones((m, k_pad // 64), dtype=torch.float16, device="cuda")
    x = torch.zeros((k, 2), dtype=torch.bfloat16, device="cuda")
    try:
        nf4_gemm(packed, scale, x, m, k, k_pad)
    except Exception:
        return 1
    return int(probe)


@dataclass(frozen=True)
class Gemm:
    """One NF4 matrix, flattened out of its ``CompressedLinear`` seat.

    The token loop touches these fields 253 times per token; resolving
    ``module.packed`` through ``nn.Module.__getattr__`` that often is pure
    interpreter tax. Tensors are shared with the module, never copied.
    """

    name: str
    packed: torch.Tensor
    scale: torch.Tensor
    M: int  # noqa: N815
    K: int  # noqa: N815
    K_pad: int  # noqa: N815
    bias: torch.Tensor | None

    @classmethod
    def of(cls, linear, name: str) -> "Gemm":
        """Read a loaded ``gpu.host.CompressedLinear``."""
        if not linear.is_loaded:
            raise RuntimeError(f"{name}: CompressedLinear has no weights")
        return cls(
            name=name,
            packed=linear.packed,
            scale=linear.scale,
            M=int(linear.M),
            K=int(linear.K),
            K_pad=int(linear.K_pad),
            bias=linear.bias,
        )

    @property
    def nbytes(self) -> int:
        return self.packed.numel() + self.scale.numel() * 2


class GemmGroup:
    """Eager runner: GEMMs sharing one ``[N, K]`` input, optionally overlapped.

    ``x`` is ``[N, K]`` (HuggingFace's layout); the kernel wants ``[K, N]`` and
    answers ``[M, N]`` (gpu-abi.md §6), so both ends are a ``.t()``. For
    ``N == 1`` -- every decode step -- both transposes are contiguous views and
    cost nothing.

    **Why the streams.** The wave-2 GEMM tiled ``BM=128`` with ``split_k=1``, so
    a matrix got ``M/128`` blocks: 16 for ``q``/``o`` and **2** for ``k``/``v``
    on 3B. On 70 SMs that left the card mostly idle, and measured on a 3080 the
    three QKV GEMMs cost 445 us back to back but 162 us when each got its own
    stream -- they are independent (same input, disjoint outputs), so the only
    thing serializing them was the stream. The kernel has since grown a 64-row
    tile and ``grid.y = split_k`` (``gpu/nf4/plan.py``), which puts 64-160 CTAs
    inside *one* of these GEMMs, so the fork now overlaps work that already
    fills the card rather than hiding an empty one. It still helps, and it is
    still free under capture. Members ``1..n-1`` fork onto ``streams``,
    member ``0`` runs on the caller's stream, and everything joins before
    ``run`` returns, so callers still see a plain sequential result. Bit-exact
    against the serial order: same kernel, same inputs.
    """

    __slots__ = ("name", "gemms", "K", "streams")

    def __init__(self, name: str, gemms: Sequence[Gemm], streams: Sequence = ()) -> None:
        if not gemms:
            raise ValueError(f"{name}: empty group")
        ks = {g.K for g in gemms}
        if len(ks) != 1:
            raise ValueError(f"{name}: a group shares one input, got K={sorted(ks)}")
        self.name = name
        self.gemms = tuple(gemms)
        self.K = int(gemms[0].K)
        # One stream per member past the first; a single-member group stays serial.
        self.streams = tuple(streams)[: len(self.gemms) - 1]
        if self.streams and len(self.streams) != len(self.gemms) - 1:
            self.streams = ()  # not enough to cover the group: no partial forking

    @property
    def nbytes(self) -> int:
        return sum(g.nbytes for g in self.gemms)

    def _one(self, g: Gemm, xk: torch.Tensor) -> torch.Tensor:
        y = nf4_gemm(g.packed, g.scale, xk, g.M, g.K, g.K_pad).t()
        if g.bias is not None:
            y = y.add_(g.bias)  # y is this call's own buffer
        return y

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        xk = x.t()
        if not self.streams:
            return tuple(self._one(g, xk) for g in self.gemms)

        main = torch.cuda.current_stream()
        out: list[torch.Tensor] = [None] * len(self.gemms)  # type: ignore[list-item]
        for i, s in enumerate(self.streams, start=1):
            s.wait_stream(main)
            with torch.cuda.stream(s):
                out[i] = self._one(self.gemms[i], xk)
        out[0] = self._one(self.gemms[0], xk)
        for i, s in enumerate(self.streams, start=1):
            main.wait_stream(s)
            # Allocated on `s`, read on `main`: the allocator has to be told, or
            # it may hand the block to someone else while `main` is still reading.
            out[i].record_stream(main)
        return tuple(out)

    def __repr__(self) -> str:
        return (
            f"GemmGroup({self.name}, K={self.K}, M={[g.M for g in self.gemms]}, "
            f"streams={len(self.streams)})"
        )


class GraphedGemmGroup:
    """Plan A: the same group, replayed from a CUDA graph.

    Capture needs static addresses on both ends. The input is a preallocated
    ``[1, K]`` buffer that ``run`` copies into; the outputs are allocated *during*
    capture, so they come from the graph's private pool and keep their addresses
    across replays. Every captured output is held for the object's lifetime,
    which is also what makes a pool shared between groups safe -- the allocator
    cannot hand group *j* memory that group *i* is still holding.

    Consequence for the caller: the returned tensors are the same buffers every
    replay. The token loop consumes them (RoPE, KV write, residual add) before
    the next group replays, which is why they are never aliased across groups.
    """

    __slots__ = ("eager", "name", "K", "x", "graph", "outs")

    def __init__(self, eager: GemmGroup, x: torch.Tensor, pool=None) -> None:
        """``x`` is the caller's static ``[1, K]`` input, allocated *outside* capture.

        ``capture_begin``/``capture_end`` instead of the ``torch.cuda.graph``
        context manager on purpose: that manager runs
        ``synchronize + gc.collect + empty_cache`` on every ``__enter__``, which
        across 145 groups costs ~20 s and pumps ``empty_cache`` through a resident
        1.6 GiB model -- one of the named ways to fragment this card
        (token-loop.md §6.1). :func:`capture` does that housekeeping once instead.
        """
        self.eager = eager
        self.name = eager.name
        self.K = eager.K
        self.x = x
        self.graph = torch.cuda.CUDAGraph()
        pool_args = () if pool is None else (pool,)
        self.graph.capture_begin(*pool_args)
        try:
            # The eager group *is* the recipe, fork/join included: a graph
            # captured from several streams keeps their parallelism at replay.
            # Capturing it verbatim is what makes "same greedy token as eager" a
            # property rather than a coincidence.
            outs = eager.run(self.x)
        finally:
            # Leaving the stream in capture mode would poison every later launch.
            self.graph.capture_end()
        self.outs = outs

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.shape[0] != 1:
            # A prefill chunk. The graph is captured for N == 1 only (plan A does
            # not bucket shapes); prefill is eager by design (token-loop.md §5).
            return self.eager.run(x)
        self.x.copy_(x)
        self.graph.replay()
        return self.outs

    def __repr__(self) -> str:
        return f"GraphedGemmGroup({self.name}, K={self.K}, n={len(self.outs)})"


def capture(
    groups: Sequence[GemmGroup],
    *,
    warmup: int = 3,
) -> tuple[list[GemmGroup | GraphedGemmGroup], str, str | None]:
    """Try to capture every group. All or nothing.

    Returns ``(runners, mode, error)`` where ``mode`` is ``"linears"`` on success
    and ``"off"`` on failure -- in which case ``runners`` is the eager list, so
    the caller has nothing to undo. Call this *after* the eager warmup:
    capture must not be the first time a kernel runs, and it must not allocate
    outside the graph pool.
    """
    if not groups:
        return [], "off", "no groups"
    if not torch.cuda.is_available():
        return list(groups), "off", "no CUDA device"

    # Static inputs first, on the default stream: a buffer allocated inside the
    # capture stream would be that stream's, and `run` writes it from the default
    # one.
    dev = groups[0].gemms[0].packed.device
    statics = [torch.zeros((1, g.K), dtype=torch.bfloat16, device=dev) for g in groups]

    # Warm every group on the capture stream: the extension calls
    # cudaFuncSetAttribute on its first launch, and one-time work inside a
    # capture is how graphs come out subtly wrong (or get refused).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for g, x in zip(groups, statics):
            for _ in range(max(1, warmup)):
                g.run(x)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gc.collect()

    pool = torch.cuda.graph_pool_handle()
    captured: list[GemmGroup | GraphedGemmGroup] = []
    try:
        for g, x in zip(groups, statics):
            # Every stream idle before capture_begin: a group's fork streams still
            # holding uncaptured work make the join "a dependency on uncaptured
            # work in another stream" and the capture is refused.
            torch.cuda.synchronize()
            with torch.cuda.stream(side):
                captured.append(GraphedGemmGroup(g, x, pool))
    except Exception as exc:  # noqa: BLE001 - eager must still PASS
        captured.clear()
        torch.cuda.synchronize()
        return list(groups), "off", f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize()
    return captured, "linears", None
