"""Two-slot H2D ring for HOST overflow GEMMs (H2-4).

One ``copy_stream`` (never the default). Compute stays on the caller's current
stream (TokenLoop's explicit compute stream). Events are **per slot**:

    copy.wait(e_gemm[s]) → one packed‖scale ``copy_`` → record e_copy[s]
    compute.wait(e_copy[s]) → nf4_gemm(slot views) → record e_gemm[s]

Prefetch depth is 1 (``ahead <= 1``). ``e_gemm`` starts already recorded so
both slots are free. ``e_copy`` is always a timing event (WDDM fast H2D).
Bind CPU-joins it before prefetch (32B: wait_event alone stayed ~1.8 tok/s).
``elapsed_time`` stays behind ``timing=True``. No ``wait_stream(copy_stream)``.
``prefetch_next`` issues tape[0] for the next decode step during DEVICE
``lm_head``; the following ``arm()`` consumes that slot instead of a second H2D.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import torch

from gpu.host.slots import SlotPair

from .graph import Gemm

__all__ = ["CopyRing"]


def _same_gemm(a: Gemm, b: Gemm) -> bool:
    return a is b or a.name == b.name


class CopyRing:
    """Ping-pong H2D into :class:`~gpu.host.slots.SlotPair`.

    ``ops`` / ``slot_order`` / ``issue_count`` / ``ahead`` are the CPU-visible
    protocol counter: tests assert wait/record order without launching a kernel.

    ``total_bytes`` / ``total_copies`` survive :meth:`arm` (one generate).
    ``forward_bytes`` resets each forward. ``e_copy`` is always
    ``enable_timing=True`` on CUDA (WDDM's slow path was the non-timing event).
    Product also CPU-joins ``e_copy`` before prefetch; ``elapsed_time`` stays
    behind ``timing=True`` (H2 tracer only).
    """

    def __init__(self, slots: SlotPair, *, timing: bool = False) -> None:
        self.slots = slots
        self.timing = bool(timing)
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
        self._join_copy = True
        self._cuda = (
            torch.cuda.is_available() and torch.device(slots.device).type == "cuda"
        )
        if self._cuda:
            self.copy_stream = torch.cuda.Stream()
            # Timing events even when we do not call elapsed_time: on this 3080
            # WDDM, a non-timing e_copy was the ~1.8 tok/s path vs ~2.3.
            self._e_copy = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self._e_gemm = (
                torch.cuda.Event(enable_timing=False),
                torch.cuda.Event(enable_timing=False),
            )
            self._e_h2d_start = (
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                if self.timing
                else (None, None)
            )
            # Slots are free: do not wait on an event that was never recorded.
            for e in self._e_gemm:
                e.record()
        else:
            self.copy_stream = None
            self._e_copy = (None, None)
            self._e_gemm = (None, None)
            self._e_h2d_start = (None, None)

    @property
    def ahead(self) -> int:
        """Issued copies not yet consumed by :meth:`bind_for_gemm`. Max 1."""
        return self._ahead

    def arm(self, gemms: Sequence[Gemm]) -> None:
        """New forward: HOST gemms in consume order. Events from last step stay.

        If :meth:`prefetch_next` already issued this tape's first GEMM, keep
        that slot (``ahead=1``) and do not H2D it again. ``e_copy`` / ``e_gemm``
        are not re-recorded.
        """
        tape = tuple(g for g in gemms if g.home == "host")
        carry = self._carry_first(tape)
        if carry is None:
            self._queue.clear()
            self._ahead = 0
            self._phase = 0
            self._tape_i = 0
        else:
            # Keep queue / ahead / phase: the next issue must use the other slot.
            self._tape_i = 1
        self._tape = tape
        self._active_slot = None
        self._active_gemm = None
        self.ops.clear()
        self.slot_order.clear()
        self.issue_count = 0
        self.forward_bytes = 0
        self.forward_copy_ms = 0.0
        self.total_forwards += 1

    def _carry_first(self, tape: tuple[Gemm, ...]) -> tuple[Gemm, int] | None:
        """In-flight first-of-tape copy from :meth:`prefetch_next`, or None."""
        if not tape or self._ahead != 1 or len(self._queue) != 1:
            return None
        if self._active_slot is not None:
            return None
        head, s = self._queue[0]
        if not _same_gemm(head, tape[0]):
            return None
        return head, s

    def prefetch(self) -> None:
        """Issue the next tape matrix if ``ahead == 0``. No-op when the tape is done."""
        if self._ahead >= 1:
            return
        if self._tape_i >= len(self._tape):
            return
        g = self._tape[self._tape_i]
        self._tape_i += 1
        self.issue(g)

    def prefetch_next(self) -> None:
        """Issue tape[0] for the next forward (overlaps DEVICE ``lm_head``).

        Depth stays 1. Does not call :meth:`arm`; the next :meth:`arm` consumes
        the slot. No-op when a copy is already ahead, the tape is empty, or
        this forward still has unissued HOST GEMMs.
        """
        if self._ahead >= 1:
            return
        if not self._tape:
            return
        if self._tape_i < len(self._tape):
            return
        if self._active_slot is not None:
            return
        self.issue(self._tape[0])

    def issue(self, gemm: Gemm) -> int:
        """Start one H2D of ``host_image.arena`` into the next ping-pong slot."""
        if self._ahead >= 1:
            raise RuntimeError(
                f"CopyRing prefetch depth is 1; issue({gemm.name}) would make ahead="
                f"{self._ahead + 1}"
            )
        if gemm.home != "host" or gemm.host_image is None:
            raise RuntimeError(f"{gemm.name}: CopyRing.issue expects a HOST gemm")
        s = self._phase
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
        self.ops.append(("wait_gemm", s))
        if self._cuda:
            self.copy_stream.wait_event(self._e_gemm[s])
            with torch.cuda.stream(self.copy_stream):
                if self.timing:
                    self._e_h2d_start[s].record(self.copy_stream)
                dst.copy_(src, non_blocking=True)
            self._e_copy[s].record(self.copy_stream)
        else:
            dst.copy_(src)
        self.ops.append(("h2d", s, nbytes))
        self.ops.append(("record_copy", s))
        self._queue.append((gemm, s))
        self._phase ^= 1
        self._ahead += 1
        self.issue_count += 1
        self.forward_bytes += nbytes
        self.total_copies += 1
        self.total_bytes += nbytes
        self.slot_order.append(s)
        return s

    def bind_for_gemm(self, gemm: Gemm) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for this matrix's copy; return packed/scale **slot** views.

        After the wait, prefetches the next tape entry (depth 1) so its H2D
        overlaps the caller's GEMM. Packed is never the host tensor.

        On CUDA the wait is ``current_stream().wait_event(e_copy)`` plus a CPU
        join (WDDM). ``elapsed_time`` runs only when ``timing`` is on.
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
            # Join before prefetch: on 32B, timing e_copy without this join
            # stayed at ~1.8 tok/s; timing+join is ~2.3. elapsed_time also
            # needs a completed end event.
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
            f"slot_nbytes={self.slots.nbytes}, ahead={self._ahead}, "
            f"issued={self.issue_count}, total_bytes={self.total_bytes}, "
            f"cuda={self._cuda}, timing={self.timing}"
        )

    def __repr__(self) -> str:
        return f"CopyRing({self.extra_repr()})"
