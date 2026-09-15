"""Device NF4 arenas for the overflow ring (H2-3). CopyRing is H2-4.

Allocate :class:`SlotPair` **before** scattering resident matrices so the
slot ``data_ptr``s stay stable. Each arena is ``slot_nbytes`` uint8; a view
is only the packed‖scale prefix for ``(M, K)``. The tail is unread junk —
the kernel touches ``M * K_pad / 2`` packed bytes.

Product overflow uses ``count=OVERFLOW_SLOT_COUNT`` (2). ``count=3`` /
``ahead=2`` is implemented and tested, but WDDM-dead on the 3080 (see
``CopyRing``).
"""

from __future__ import annotations

import torch

from .host_image import unpack_arena
from .residency import nf4_nbytes

__all__ = ["OVERFLOW_SLOT_COUNT", "SLOT_ALIGN", "SlotPair"]

SLOT_ALIGN = 16
# Product stays 2: ahead=2 (three arenas + two copy streams) on this WDDM
# box joined while a second H2D was already queued → warmup ~462 s, 0.05 tok/s.
OVERFLOW_SLOT_COUNT = 2


class SlotPair:
    """``count`` device arenas, each ``nbytes`` uint8 on ``device``.

    ``count=2`` is product ping-pong. ``count=3`` is implemented (ahead=2)
    but WDDM-dead on this 3080 if prefetch fills two copies before the join.
    ``device`` may be CPU in tests. Torch CUDA alloc is typically ≥256 aligned;
    we still refuse a ``data_ptr`` that is not 16-byte aligned.
    """

    def __init__(
        self, nbytes: int, device: str | torch.device, count: int = 2
    ) -> None:
        n = int(nbytes)
        if n < 0:
            raise ValueError(f"SlotPair nbytes must be >= 0, got {n}")
        k = int(count)
        if k < 2:
            raise ValueError(f"SlotPair count must be >= 2, got {k}")
        self.nbytes = n
        self.count = k
        self.device = torch.device(device)
        self.arena = tuple(
            torch.empty(n, dtype=torch.uint8, device=self.device) for _ in range(k)
        )
        for i, buf in enumerate(self.arena):
            ptr = int(buf.data_ptr())
            if n > 0 and ptr % SLOT_ALIGN != 0:
                raise RuntimeError(
                    f"SlotPair arena[{i}] data_ptr {ptr:#x} is not "
                    f"{SLOT_ALIGN}-byte aligned"
                )

    def view(self, i: int, M: int, K: int, K_pad: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: N803
        """packed / scale views into ``arena[i][:nf4_nbytes(M, K)]``."""
        if i < 0 or i >= self.count:
            raise IndexError(f"SlotPair.view index must be 0..{self.count - 1}, got {i}")
        n = nf4_nbytes(M, K)
        if n > self.nbytes:
            raise ValueError(
                f"view nf4_nbytes({M},{K})={n} exceeds slot {self.nbytes} bytes"
            )
        return unpack_arena(self.arena[i][:n], int(M), int(K), k_pad_=int(K_pad))

    def extra_repr(self) -> str:
        return f"nbytes={self.nbytes}, count={self.count}, device={self.device}"

    def __repr__(self) -> str:
        return f"SlotPair({self.extra_repr()})"
