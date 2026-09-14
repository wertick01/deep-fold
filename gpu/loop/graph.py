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
implementation -- the graph is captured *from* the eager group. H2 overflow
keeps HOST *groups* out of :func:`capture` (CopyRing H2D must not enter a
graph). Decode ``N == 1`` may still replay a per-slot ``nf4_gemm`` graph
inside :meth:`GemmGroup._one` after bind: the two SlotPair arenas never move,
so the kernel sees the same device pointers every token. Prefill ``N != 1``
stays eager. If a DEVICE capture fails -- a driver that refuses
``cudaFuncSetAttribute`` mid-capture, say -- :func:`capture` returns the eager
groups unchanged and the loop still runs (the TZ requires eager to PASS on
its own). A HOST-slot capture failure falls back to the same eager ``_one``.

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
    "HostSlotGemm",
    "capture",
    "group_is_resident",
    "host_slot_graphs",
    "linear_max_n",
    "nf4_gemm",
    "nf4_max_n",
    "reset_host_slot_graphs",
    "slot_gemm_key",
    "try_host_slot_gemm",
    "vq_gemm",
    "vq_max_n",
]

_gemm = None


def nf4_gemm(packed, scale, x, M, K, K_pad):  # noqa: N803
    """Agent 2's kernel, resolved once. Importing it may trigger a JIT build."""
    global _gemm
    if _gemm is None:
        from gpu.nf4 import nf4_gemm as fn

        _gemm = fn
    return _gemm(packed, scale, x, M, K, K_pad)


_vq = None


def vq_gemm(index, book, x, M, K, K_pad):  # noqa: N803
    """VQ kernel, resolved once. Importing it may trigger a JIT build."""
    global _vq
    if _vq is None:
        from gpu.vq import vq_gemm as fn

        _vq = fn
    return _vq(index, book, x, M, K, K_pad)


def nf4_max_n(probe: int = 16) -> int:
    """Largest ``N`` (sequence columns) the installed kernel accepts.

    Wave 2 shipped a decode-only GEMM (``N != 1`` -> rc -2, docs/spec/stitch-gpu.md),
    which decides whether the prompt can be prefilled in one forward or has to be
    walked token by token. Asked, not assumed: agent 6 may have widened it.
    ``probe`` is the TokenLoop chunk (``LIVE_MAX_N``). This function probes N=2,
    then returns ``probe`` -- it does not launch N=probe.
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


def vq_max_n(probe: int = 16) -> int:
    """Largest ``N`` the VQ kernel accepts. Wave-3 VQ is decode-only (N==1)."""
    if not torch.cuda.is_available():
        return 1
    m, k, k_pad = 64, 64, 64
    g = k_pad // 8
    index = torch.zeros((m, g, 2), dtype=torch.uint8, device="cuda")
    book = torch.zeros((2, 256, 8), dtype=torch.float16, device="cuda")
    x = torch.zeros((k, 2), dtype=torch.bfloat16, device="cuda")
    try:
        vq_gemm(index, book, x, m, k, k_pad)
    except Exception:
        return 1
    return int(probe)


def linear_max_n(codec: str, probe: int = 16) -> int:
    """Prefill width for the codec actually loaded. Does not import the other kernel."""
    if codec == "vq":
        return vq_max_n(probe)
    return nf4_max_n(probe)


@dataclass(frozen=True)
class Gemm:
    """One packed matrix, flattened out of its linear seat (NF4 or VQ).

    The token loop touches these fields hundreds of times per token; resolving
    ``module.packed`` through ``nn.Module.__getattr__`` that often is pure
    interpreter tax. Tensors are shared with the module, never copied.
    """

    name: str
    M: int  # noqa: N815
    K: int  # noqa: N815
    K_pad: int  # noqa: N815
    bias: torch.Tensor | None
    packed: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    index: torch.Tensor | None = None
    book: torch.Tensor | None = None
    codec: str = "nf4"
    host_image: object | None = None
    home: str = "device"

    @classmethod
    def of(cls, linear, name: str) -> "Gemm":
        """Read a loaded NF4 or VQ seat. HOST overflow keeps empty packed."""
        if not linear.is_loaded:
            raise RuntimeError(f"{name}: packed linear has no weights")
        index = getattr(linear, "index", None)
        book = getattr(linear, "book", None)
        packed = getattr(linear, "packed", None)
        if index is not None and book is not None and index.numel() > 0:
            return cls(
                name=name,
                M=int(linear.M),
                K=int(linear.K),
                K_pad=int(linear.K_pad),
                bias=linear.bias,
                index=index,
                book=book,
                codec="vq",
            )
        home = getattr(linear, "home", "device")
        if home == "host":
            img = getattr(linear, "host_image", None)
            if img is None:
                raise RuntimeError(f"{name}: home=host but host_image is None")
            return cls(
                name=name,
                M=int(linear.M),
                K=int(linear.K),
                K_pad=int(linear.K_pad),
                bias=linear.bias,
                packed=packed,
                scale=getattr(linear, "scale", None),
                codec="nf4",
                host_image=img,
                home="host",
            )
        if packed is None or packed.numel() == 0:
            raise RuntimeError(f"{name}: seat is loaded but has neither NF4 packed nor VQ index")
        return cls(
            name=name,
            M=int(linear.M),
            K=int(linear.K),
            K_pad=int(linear.K_pad),
            bias=linear.bias,
            packed=packed,
            scale=linear.scale,
            codec="nf4",
        )

    @property
    def nbytes(self) -> int:
        if self.home == "host":
            return int(self.host_image.nbytes)
        if self.codec == "vq":
            return int(self.index.numel()) + int(self.book.numel()) * 2
        return int(self.packed.numel()) + int(self.scale.numel()) * 2

    @property
    def device(self) -> torch.device:
        if self.home == "host":
            packed = self.packed
            if packed is not None and packed.device.type == "cuda":
                return packed.device
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        t = self.packed if self.packed is not None and self.packed.numel() else self.index
        return t.device


# Decode N=1 HOST compute: one CUDA graph per (M, K, slot data_ptr). Bind, H2D
# and prefetch stay in CopyRing; these recipes never record copy_ or host packed.
_MISSING = object()
_HOST_SLOT: dict[tuple[int, int, int], "HostSlotGemm | None"] = {}
_HOST_SLOT_POOL = None
_HOST_SLOT_CAPTURE_STREAM = None


def slot_gemm_key(packed: torch.Tensor, M: int, K: int) -> tuple[int, int, int]:  # noqa: N803
    """Cache key for a HOST-slot decode graph. Refuses host packed."""
    if packed.device.type != "cuda":
        raise RuntimeError(
            f"slot graph packed must be a device slot view, got {packed.device}; "
            "refuse host packed"
        )
    return (int(M), int(K), int(packed.data_ptr()))


def host_slot_graphs() -> dict[tuple[int, int, int], "HostSlotGemm | None"]:
    """Process-wide HOST-slot recipes. ``None`` values are failed captures."""
    return _HOST_SLOT


def reset_host_slot_graphs() -> None:
    """Drop captured HOST-slot graphs. Tests isolate canaries this way."""
    global _HOST_SLOT_POOL
    _HOST_SLOT.clear()
    _HOST_SLOT_POOL = None


def _host_slot_pool():
    global _HOST_SLOT_POOL
    if _HOST_SLOT_POOL is None:
        _HOST_SLOT_POOL = torch.cuda.graph_pool_handle()
    return _HOST_SLOT_POOL


def _host_slot_capture_stream():
    """Non-default stream: CUDA refuses ``capture_begin`` on the default stream.

    Replay still runs on the caller's stream (TokenLoop compute). This is not
    a GemmGroup member fork -- HOST ``streams=()`` stays.
    """
    global _HOST_SLOT_CAPTURE_STREAM
    if _HOST_SLOT_CAPTURE_STREAM is None:
        _HOST_SLOT_CAPTURE_STREAM = torch.cuda.Stream()
    return _HOST_SLOT_CAPTURE_STREAM


class HostSlotGemm:
    """``nf4_gemm`` captured against a static SlotPair view, decode ``N == 1``.

    Input is a preallocated ``[K, 1]`` buffer that :meth:`run` copies into.
    Packed/scale addresses are the slot arenas (stable for the process). Bias
    stays eager so layers that share ``(M, K, packed_ptr)`` share one recipe.
    """

    __slots__ = ("x", "graph", "y", "packed", "scale", "replays")

    def __init__(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        x: torch.Tensor,
        graph,
        y: torch.Tensor,
    ) -> None:
        self.packed = packed
        self.scale = scale
        self.x = x
        self.graph = graph
        self.y = y
        self.replays = 0

    def run(self, xk: torch.Tensor) -> torch.Tensor:
        self.x.copy_(xk.reshape(self.x.shape))
        self.graph.replay()
        self.replays += 1
        return self.y.t()


def _capture_host_slot(
    packed: torch.Tensor,
    scale: torch.Tensor,
    xk: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
) -> HostSlotGemm | None:
    """Warm then capture. No ``empty_cache``, no H2D, no host packed."""
    try:
        slot_gemm_key(packed, M, K)
    except RuntimeError:
        return None
    if scale.device.type != "cuda" or xk.device.type != "cuda":
        return None
    x_static = torch.empty((K, 1), dtype=torch.bfloat16, device=xk.device)
    x_static.copy_(xk.reshape(K, 1))
    main = torch.cuda.current_stream()
    side = _host_slot_capture_stream()
    side.wait_stream(main)
    try:
        with torch.cuda.stream(side):
            # First launch on this slot view sets cudaFuncSetAttribute; doing
            # that inside capture_begin is how graphs come out wrong or get
            # refused. side.synchronize, not torch.cuda.synchronize: the copy
            # engine may still be prefetching the other slot.
            nf4_gemm(packed, scale, x_static, M, K, K_pad)
            side.synchronize()
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(_host_slot_pool())
            try:
                y = nf4_gemm(packed, scale, x_static, M, K, K_pad)
            finally:
                graph.capture_end()
        main.wait_stream(side)
        y.record_stream(main)
    except Exception:  # noqa: BLE001 - eager _one must still PASS
        return None
    return HostSlotGemm(packed, scale, x_static, graph, y)


def try_host_slot_gemm(
    packed: torch.Tensor,
    scale: torch.Tensor,
    xk: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
) -> torch.Tensor | None:
    """Replay or capture N=1 nf4_gemm on a slot view. ``None`` means use eager."""
    n = 1 if xk.dim() == 1 else int(xk.shape[-1])
    if n != 1:
        return None
    if packed.device.type != "cuda" or xk.device.type != "cuda":
        return None
    key = (int(M), int(K), int(packed.data_ptr()))
    rec = _HOST_SLOT.get(key, _MISSING)
    if rec is None:
        return None
    if rec is _MISSING:
        rec = _capture_host_slot(packed, scale, xk, M, K, K_pad)
        _HOST_SLOT[key] = rec
        if rec is None:
            return None
        return rec.y.t()
    return rec.run(xk)


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

    __slots__ = ("name", "gemms", "K", "streams", "ring")

    def __init__(self, name: str, gemms: Sequence[Gemm], streams: Sequence = ()) -> None:
        if not gemms:
            raise ValueError(f"{name}: empty group")
        ks = {g.K for g in gemms}
        if len(ks) != 1:
            raise ValueError(f"{name}: a group shares one input, got K={sorted(ks)}")
        self.name = name
        self.gemms = tuple(gemms)
        self.K = int(gemms[0].K)
        self.ring = None
        # Fork only all-DEVICE. Any HOST member (including mixed) is serial so
        # CopyRing's per-slot events stay ordered; QKV overflow is copy/GEMM
        # per member, not a side-stream fork.
        if any(g.home == "host" for g in gemms):
            self.streams = ()
        else:
            # One stream per member past the first; a single-member group stays serial.
            self.streams = tuple(streams)[: len(self.gemms) - 1]
            if self.streams and len(self.streams) != len(self.gemms) - 1:
                self.streams = ()  # not enough to cover the group: no partial forking

    @property
    def nbytes(self) -> int:
        return sum(g.nbytes for g in self.gemms)

    def _one(self, g: Gemm, xk: torch.Tensor) -> torch.Tensor:
        if g.codec == "vq":
            y = vq_gemm(g.index, g.book, xk, g.M, g.K, g.K_pad).t()
        elif g.home == "host":
            ring = self.ring
            if ring is None:
                raise RuntimeError(
                    f"{g.name}: host-resident GEMM needs CopyRing; "
                    "refuse nf4_gemm on host packed"
                )
            packed, scale = ring.bind_for_gemm(g)
            if xk.device.type == "cuda" and packed.device.type != "cuda":
                raise RuntimeError(
                    f"{g.name}: nf4_gemm packed must be a device slot view, "
                    f"got {packed.device}"
                )
            y = try_host_slot_gemm(packed, scale, xk, g.M, g.K, g.K_pad)
            if y is None:
                y = nf4_gemm(packed, scale, xk, g.M, g.K, g.K_pad).t()
            ring.record_gemm(g)
        else:
            y = nf4_gemm(g.packed, g.scale, xk, g.M, g.K, g.K_pad).t()
        if g.bias is not None:
            y = y.add_(g.bias)  # y is this call's own buffer
        return y

    def run(self, x: torch.Tensor, ring=None) -> tuple[torch.Tensor, ...]:
        if ring is not None:
            self.ring = ring
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

    def run(self, x: torch.Tensor, ring=None) -> tuple[torch.Tensor, ...]:
        if x.shape[0] != 1:
            # A prefill chunk. The graph is captured for N == 1 only (plan A does
            # not bucket shapes); prefill is eager by design (token-loop.md §5).
            return self.eager.run(x, ring=ring)
        self.x.copy_(x)
        self.graph.replay()
        return self.outs

    def __repr__(self) -> str:
        return f"GraphedGemmGroup({self.name}, K={self.K}, n={len(self.outs)})"


def group_is_resident(group: GemmGroup) -> bool:
    """True when every GEMM is DEVICE: safe to CUDA-graph, no H2D in the recipe."""
    return all(g.home != "host" for g in group.gemms)


def capture(
    groups: Sequence[GemmGroup],
    *,
    warmup: int = 3,
) -> tuple[list[GemmGroup | GraphedGemmGroup], str, str | None]:
    """Capture all-DEVICE groups. HOST groups stay eager (H2 overflow).

    Fully resident: all-or-nothing as before -- one DEVICE failure returns the
    eager list, ``mode="off"``. Mixed: DEVICE groups are captured as a set
    (same all-or-nothing among them); HOST groups are never warmed here
    without CopyRing, so the kernel never sees host packed. Decode N=1 HOST
    compute is graphed later in :meth:`GemmGroup._one` on slot pointers, not
    by treating HOST groups as resident. ``mode`` is ``"linears"`` if at least
    one group is graphed, else ``"off"``. Failure is the exception text, not
    ``"overflow"``.

    Call this *after* the eager warmup. ``torch.cuda.synchronize`` is allowed
    on this boundary (copy engine idle) and not on the decode path.
    """
    if not groups:
        return [], "off", "no groups"

    resident = [(i, g) for i, g in enumerate(groups) if group_is_resident(g)]
    if not resident:
        return list(groups), "off", None
    if not torch.cuda.is_available():
        return list(groups), "off", "no CUDA device"

    # Static inputs first, on the default stream: a buffer allocated inside the
    # capture stream would be that stream's, and `run` writes it from the default
    # one. HOST groups are not captured and must not be warmed without CopyRing.
    dev = resident[0][1].gemms[0].device
    if dev.type != "cuda":
        return list(groups), "off", "weights not on CUDA"

    # Copy engine idle before capture (TokenLoop.warmup may have issued H2D).
    torch.cuda.synchronize()
    statics = {
        i: torch.zeros((1, g.K), dtype=torch.bfloat16, device=dev) for i, g in resident
    }

    # Warm DEVICE groups on the capture stream: the extension calls
    # cudaFuncSetAttribute on its first launch, and one-time work inside a
    # capture is how graphs come out subtly wrong (or get refused).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i, g in resident:
            for _ in range(max(1, warmup)):
                g.run(statics[i])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gc.collect()

    pool = torch.cuda.graph_pool_handle()
    runners: list[GemmGroup | GraphedGemmGroup] = list(groups)
    try:
        for i, g in resident:
            # Every stream idle before capture_begin: a group's fork streams still
            # holding uncaptured work make the join "a dependency on uncaptured
            # work in another stream" and the capture is refused.
            torch.cuda.synchronize()
            with torch.cuda.stream(side):
                runners[i] = GraphedGemmGroup(g, statics[i], pool)
    except Exception as exc:  # noqa: BLE001 - eager must still PASS
        torch.cuda.synchronize()
        return list(groups), "off", f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize()
    return runners, "linears", None
