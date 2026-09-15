"""H2D ring for HOST overflow GEMMs (H2-4).

Copy engines never the default stream. Compute stays on the caller's current
stream (TokenLoop's explicit compute stream). Events are **per slot**:

    copy.wait(e_gemm[s]) → one packed‖scale ``copy_`` → record e_copy[s]
    compute.wait(e_copy[s]) → nf4_gemm(slot views) → record e_gemm[s]

``max_ahead = n_slots - 1`` (2 slots → depth 1; 3 slots → depth 2). Filling
``ahead=2`` before the CPU join (prefetch at arm, two copies already queued)
is the same WDDM trap as joining after queueing copy_{i+1}: the join waits
out the extra in-flight H2D (32B: warmup ~462 s, 0.05 tok/s). Product stays
two slots / one copy stream. Two copy-stream handles still share one H2D
engine.

``e_copy`` is always a timing event (WDDM fast H2D). Bind CPU-joins the
current ``e_copy`` *before* prefetch (WDDM: joining after queueing copy_{i+1}
drained the copy stream and fell to 0.01 tok/s on 32B). ``elapsed_time``
stays behind ``timing=True``. No ``wait_stream(copy_stream)``.
``prefetch_next`` issues a prefix of the next decode tape during DEVICE
``lm_head``; the following ``arm()`` consumes those slots instead of a
second H2D.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Sequence

import torch

from gpu.host.slots import SlotPair

from .graph import Gemm

__all__ = ["CopyRing", "default_join_copy"]


def _same_gemm(a: Gemm, b: Gemm) -> bool:
    return a is b or a.name == b.name


def default_join_copy() -> bool:
    """CPU-join ``e_copy`` before prefetch.

    Windows (WDDM) needs the join; without it this 3080 fell to 0.01 tok/s.
    POSIX defaults off: compute still ``wait_event``s. Override with
    ``DEEPFOLD_COPY_JOIN=1`` or ``0``.
    """
    raw = os.environ.get("DEEPFOLD_COPY_JOIN", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return os.name == "nt"


class CopyRing:
    """H2D into :class:`~gpu.host.slots.SlotPair` arenas.

    ``ops`` / ``slot_order`` / ``issue_count`` / ``ahead`` are the CPU-visible
    protocol counter: tests assert wait/record order without launching a kernel.

    ``total_bytes`` / ``total_copies`` survive :meth:`arm` (one generate).
    ``forward_bytes`` resets each forward. ``e_copy`` is always
    ``enable_timing=True`` on CUDA (WDDM's slow path was the non-timing event).
    Product CPU-joins ``e_copy`` *before* prefetch. Joining after queueing the
    next copy made WDDM wait out the copy stream (32B: 0.01 tok/s).
    ``elapsed_time`` stays behind ``timing=True`` (H2 tracer only).
    """

    def __init__(
        self,
        slots: SlotPair,
        *,
        timing: bool = False,
        n_copy_streams: int | None = None,
    ) -> None:
        self.slots = slots
        self.timing = bool(timing)
        self.n_slots = int(slots.count)
        self.max_ahead = max(1, self.n_slots - 1)
        if n_copy_streams is None:
            n_copy_streams = 2 if self.n_slots >= 3 else 1
        n_cs = int(n_copy_streams)
        if n_cs < 1:
            raise ValueError(f"n_copy_streams must be >= 1, got {n_cs}")
        self.n_copy_streams = n_cs
        self.ops: list[tuple] = []
        self.slot_order: list[int] = []
        self.issue_count = 0
        self.forward_bytes = 0
        self.forward_copy_ms = 0.0
        self.total_copies = 0
        self.total_bytes = 0
        self.total_copy_ms = 0.0
        self.total_forwards = 0
        self._phase = 0
        self._ahead = 0
        self._queue: deque[tuple[Gemm, int]] = deque()
        self._tape: tuple[Gemm, ...] = ()
        self._tape_i = 0
        self._active_slot: int | None = None
        self._active_gemm: Gemm | None = None
        # Product: timing e_copy + CPU join. Tests may set False. elapsed_time
        # still needs a join and stays behind timing=True.
        self._join_copy = default_join_copy()
        self._cuda = (
            torch.cuda.is_available() and torch.device(slots.device).type == "cuda"
        )
        n = self.n_slots
        if self._cuda:
            n_cs = min(self.n_copy_streams, n)
            self._copy_streams = tuple(torch.cuda.Stream() for _ in range(n_cs))
            self.copy_stream = self._copy_streams[0]
            # Timing events even when we do not call elapsed_time: on this 3080
            # WDDM, a non-timing e_copy was the ~1.8 tok/s path vs ~2.3.
            self._e_copy = tuple(torch.cuda.Event(enable_timing=True) for _ in range(n))
            self._e_gemm = tuple(torch.cuda.Event(enable_timing=False) for _ in range(n))
            self._e_h2d_start = (
                tuple(torch.cuda.Event(enable_timing=True) for _ in range(n))
                if self.timing
                else (None,) * n
            )
            # Slots are free: do not wait on an event that was never recorded.
            for e in self._e_gemm:
                e.record()
        else:
            self._copy_streams = ()
            self.copy_stream = None
            self._e_copy = (None,) * n
            self._e_gemm = (None,) * n
            self._e_h2d_start = (None,) * n

    @property
    def ahead(self) -> int:
        """Issued copies not yet consumed by :meth:`bind_for_gemm` / :meth:`bind_hold`."""
        return self._ahead

    def _copy_stream_for(self, slot: int):
        if not self._copy_streams:
            return None
        return self._copy_streams[slot % len(self._copy_streams)]

    def arm(self, gemms: Sequence[Gemm]) -> None:
        """New forward: HOST gemms in consume order. Events from last step stay.

        If :meth:`prefetch_next` already issued a prefix of this tape, keep
        those slots and do not H2D them again. ``e_copy`` / ``e_gemm`` are
        not re-recorded.
        """
        tape = tuple(g for g in gemms if g.home == "host")
        n_carry = self._carry_prefix(tape)
        if n_carry is None:
            self._queue.clear()
            self._ahead = 0
            self._phase = 0
            self._tape_i = 0
        else:
            # Keep queue / ahead / phase: the next issue uses the next arena.
            self._tape_i = n_carry
        self._tape = tape
        self._active_slot = None
        self._active_gemm = None
        self.ops.clear()
        self.slot_order.clear()
        self.issue_count = 0
        self.forward_bytes = 0
        self.forward_copy_ms = 0.0
        self.total_forwards += 1

    def _carry_prefix(self, tape: tuple[Gemm, ...]) -> int | None:
        """In-flight prefix from :meth:`prefetch_next`, or None."""
        if not tape or not self._queue or self._active_slot is not None:
            return None
        if self._ahead != len(self._queue):
            return None
        for i, (g, _) in enumerate(self._queue):
            if i >= len(tape) or not _same_gemm(g, tape[i]):
                return None
        return len(self._queue)

    def prefetch(self) -> None:
        """Fill the ring up to ``max_ahead``. No-op when the tape is done."""
        while self._ahead < self.max_ahead and self._tape_i < len(self._tape):
            if self._active_slot is not None and self._phase == self._active_slot:
                break
            g = self._tape[self._tape_i]
            self._tape_i += 1
            self.issue(g)

    def prefetch_next(self) -> None:
        """Issue a prefix of this tape for the next forward (DEVICE ``lm_head``).

        Does not call :meth:`arm`; the next :meth:`arm` consumes the slots.
        No-op when the tape is empty, a GEMM is bound, or this forward still
        has unissued HOST GEMMs.
        """
        if not self._tape:
            return
        if self._tape_i < len(self._tape):
            return
        if self._active_slot is not None:
            return
        i = 0
        while self._ahead < self.max_ahead and i < len(self._tape):
            self.issue(self._tape[i])
            i += 1

    def issue(self, gemm: Gemm) -> int:
        """Start one H2D of ``host_image.arena`` into the next ring slot."""
        if self._ahead >= self.max_ahead:
            raise RuntimeError(
                f"CopyRing prefetch depth is {self.max_ahead}; issue({gemm.name}) "
                f"would make ahead={self._ahead + 1}"
            )
        if gemm.home != "host" or gemm.host_image is None:
            raise RuntimeError(f"{gemm.name}: CopyRing.issue expects a HOST gemm")
        s = self._phase
        if self._active_slot is not None and s == self._active_slot:
            raise RuntimeError(
                f"CopyRing.issue({gemm.name}) would overwrite in-GEMM slot {s}"
            )
        nbytes = int(gemm.host_image.nbytes)
        if nbytes > self.slots.nbytes:
            raise ValueError(
                f"{gemm.name}: host image {nbytes} bytes exceeds slot {self.slots.nbytes}"
            )
        src = gemm.host_image.arena
        if int(src.numel()) != nbytes:
            raise ValueError(
                f"{gemm.name}: arena {int(src.numel())} != nbytes {nbytes}"
            )
        dst = self.slots.arena[s][:nbytes]
        cs = self._copy_stream_for(s)
        self.ops.append(("wait_gemm", s))
        if self._cuda:
            cs.wait_event(self._e_gemm[s])
            with torch.cuda.stream(cs):
                if self.timing:
                    self._e_h2d_start[s].record(cs)
                dst.copy_(src, non_blocking=True)
            self._e_copy[s].record(cs)
        else:
            dst.copy_(src)
        self.ops.append(("h2d", s, nbytes))
        self.ops.append(("record_copy", s))
        self._queue.append((gemm, s))
        self._phase = (self._phase + 1) % self.n_slots
        self._ahead += 1
        self.issue_count += 1
        self.forward_bytes += nbytes
        self.total_copies += 1
        self.total_bytes += nbytes
        self.slot_order.append(s)
        return s

    def bind_for_gemm(self, gemm: Gemm) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for this matrix's copy; return packed/scale **slot** views.

        GPU-side ``wait_event(e_copy)``, CPU-join the current copy (WDDM), then
        prefetch further tape entries (up to ``max_ahead``). Packed is never
        the host tensor.

        Do not queue copy_{i+1} before the join: on this WDDM driver that
        waits out the whole copy stream, and copy_{i+1} is blocked on
        ``e_gemm`` of a slot whose GEMM has not launched yet (32B: 0.01 tok/s).
        ``elapsed_time`` runs only when ``timing`` is on.
        """
        if gemm.home != "host":
            return gemm.packed, gemm.scale
        if self._active_slot is not None:
            raise RuntimeError(
                f"CopyRing.bind_for_gemm({gemm.name}): previous GEMM not record_gemm'd"
            )
        if not self._queue:
            self._issue_missing(gemm)
        head, s = self._queue[0]
        if not _same_gemm(head, gemm):
            raise RuntimeError(
                f"CopyRing.bind_for_gemm({gemm.name}): queue head is {head.name}"
            )
        self._queue.popleft()
        self._ahead = max(0, self._ahead - 1)
        self.ops.append(("wait_copy", s))
        if self._cuda:
            torch.cuda.current_stream().wait_event(self._e_copy[s])
            # Join before prefetch: queuing copy_{i+1} first then synchronizing
            # e_copy[i] on this WDDM box waited out the whole copy_stream
            # (0.01 tok/s on 32B). elapsed_time also needs a completed end event.
            if self._join_copy or self.timing:
                self._e_copy[s].synchronize()
            if self.timing:
                dt = float(self._e_h2d_start[s].elapsed_time(self._e_copy[s]))
                self.forward_copy_ms += dt
                self.total_copy_ms += dt
        packed, scale = self.slots.view(s, gemm.M, gemm.K, gemm.K_pad)
        self._active_slot = s
        self._active_gemm = gemm
        self.prefetch()
        return packed, scale

    def bind_hold(self, gemm: Gemm) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait ``e_copy``, WDDM-join, return slot views. Do **not** prefetch.

        Prefill weight-stationary path (docs/plan-h2-accel.md §A): one H2D, then
        several ``N<=32`` GEMMs on the same slot. Decode stays on
        :meth:`bind_for_gemm` / :meth:`record_gemm`. Join happens before any
        hold GEMM and before the next prefetch (that is :meth:`release_hold`).
        """
        if gemm.home != "host":
            raise RuntimeError(f"CopyRing.bind_hold({gemm.name}): expected HOST gemm")
        if self._active_slot is not None:
            raise RuntimeError(
                f"CopyRing.bind_hold({gemm.name}): hold already active"
            )
        if not self._queue:
            self._issue_missing(gemm)
        head, s = self._queue[0]
        if not _same_gemm(head, gemm):
            raise RuntimeError(
                f"CopyRing.bind_hold({gemm.name}): queue head is {head.name}"
            )
        self._queue.popleft()
        self._ahead = max(0, self._ahead - 1)
        self.ops.append(("wait_copy", s))
        if self._cuda:
            torch.cuda.current_stream().wait_event(self._e_copy[s])
            if self._join_copy or self.timing:
                self._e_copy[s].synchronize()
            if self.timing:
                dt = float(self._e_h2d_start[s].elapsed_time(self._e_copy[s]))
                self.forward_copy_ms += dt
                self.total_copy_ms += dt
        packed, scale = self.slots.view(s, gemm.M, gemm.K, gemm.K_pad)
        self._active_slot = s
        self._active_gemm = gemm
        return packed, scale

    def gemm_hold(self) -> None:
        """No-op: slot stays pinned until :meth:`release_hold`. Asserts a hold."""
        if self._active_slot is None:
            raise RuntimeError("CopyRing.gemm_hold: no active hold")

    def release_hold(self, gemm: Gemm) -> None:
        """Record ``e_gemm``, clear active, prefetch next like after a bind/record."""
        self.record_gemm(gemm)
        self.prefetch()

    def record_gemm(self, gemm: Gemm | None = None) -> None:
        """Slot ``s`` may be overwritten after the compute stream reaches this event."""
        s = self._active_slot
        if s is None:
            raise RuntimeError("CopyRing.record_gemm: no active bind")
        if (
            gemm is not None
            and self._active_gemm is not None
            and not _same_gemm(gemm, self._active_gemm)
        ):
            raise RuntimeError(
                f"CopyRing.record_gemm({gemm.name}): active is {self._active_gemm.name}"
            )
        self.ops.append(("record_gemm", s))
        if self._cuda:
            self._e_gemm[s].record()
        self._active_slot = None
        self._active_gemm = None

    def _issue_missing(self, gemm: Gemm) -> None:
        """bind() without a prior prefetch: issue now, keep the tape index honest."""
        if self._tape_i < len(self._tape):
            nxt = self._tape[self._tape_i]
            if _same_gemm(nxt, gemm):
                self._tape_i += 1
        self.issue(gemm)

    def snapshot(self) -> dict:
        """Counters for one dump. ``issue_count`` is this forward only."""
        return {
            "slot_nbytes": int(self.slots.nbytes),
            "n_slots": int(self.n_slots),
            "max_ahead": int(self.max_ahead),
            "n_copy_streams": int(self.n_copy_streams),
            "ahead": int(self._ahead),
            "issue_count": int(self.issue_count),
            "forward_bytes": int(self.forward_bytes),
            "forward_copy_ms": float(self.forward_copy_ms) if self.timing else None,
            "total_copies": int(self.total_copies),
            "total_bytes": int(self.total_bytes),
            "total_copy_ms": float(self.total_copy_ms) if self.timing else None,
            "total_forwards": int(self.total_forwards),
            "timing": bool(self.timing),
            "cuda": bool(self._cuda),
        }

    def extra_repr(self) -> str:
        return (
            f"n_slots={self.n_slots}, max_ahead={self.max_ahead}, "
            f"n_copy_streams={self.n_copy_streams}, slot_nbytes={self.slots.nbytes}, "
            f"ahead={self._ahead}, issued={self.issue_count}, "
            f"total_bytes={self.total_bytes}, cuda={self._cuda}, timing={self.timing}"
        )

    def __repr__(self) -> str:
        return f"CopyRing({self.extra_repr()})"
