"""Two device NF4 arenas for the overflow ring (H2-3). CopyRing is H2-4.

Allocate :class:`SlotPair` **before** scattering resident matrices so the
slot ``data_ptr``s stay stable. Each arena is ``slot_nbytes`` uint8; a view
is only the packed‖scale prefix for ``(M, K)``. The tail is unread junk —
the kernel touches ``M * K_pad / 2`` packed bytes.
"""

from __future__ import annotations

import torch

from .host_image import unpack_arena
from .residency import nf4_nbytes

__all__ = ["SLOT_ALIGN", "SlotPair"]

SLOT_ALIGN = 16


class SlotPair:
    """``arena[0]`` and ``arena[1]``, each ``nbytes`` uint8 on ``device``.

    ``device`` may be CPU in tests. Torch CUDA alloc is typically ≥256 aligned;
    we still refuse a ``data_ptr`` that is not 16-byte aligned.
    """

    def __init__(self, nbytes: int, device: str | torch.device) -> None:
        n = int(nbytes)
        if n < 0:
            raise ValueError(f"SlotPair nbytes must be >= 0, got {n}")
        self.nbytes = n
        self.device = torch.device(device)
        self.arena = (
            torch.empty(n, dtype=torch.uint8, device=self.device),
            torch.empty(n, dtype=torch.uint8, device=self.device),
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
        if i not in (0, 1):
            raise IndexError(f"SlotPair.view index must be 0 or 1, got {i}")
        n = nf4_nbytes(M, K)
        if n > self.nbytes:
            raise ValueError(
                f"view nf4_nbytes({M},{K})={n} exceeds slot {self.nbytes} bytes"
            )
        return unpack_arena(self.arena[i][:n], int(M), int(K), k_pad_=int(K_pad))

    def extra_repr(self) -> str:
        return f"nbytes={self.nbytes}, device={self.device}"

    def __repr__(self) -> str:
        return f"SlotPair({self.extra_repr()})"
