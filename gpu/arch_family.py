"""Which CUDA capabilities may generate, and which stay a named refuse.

Ampere-family cubins are the measured kernel. SM120 (GeForce RTX 50) is the
same ``mma.sync`` model and 99 KiB smem ceiling, so generate is allowed as
experimental. Hopper and datacenter Blackwell (SM100) are different ISAs.

The first remote SM120 SKU is RTX 5070 Ti: 70 SMs (same as the 3080 plate),
16 GB GDDR7, ~896 GB/s. That is public silicon, not a Deepfold plate.
"""

from __future__ import annotations

import re

__all__ = [
    "AMPERE_CAPABILITIES",
    "GENERATE_CAPABILITIES",
    "MEASURED_CAPABILITY",
    "REMOTE_SM120_SMS",
    "REMOTE_SM120_SKU",
    "REMOTE_SM120_VRAM_MIB",
    "SM100_CAPABILITIES",
    "SM120_CAPABILITIES",
    "TURING_CAPABILITY",
    "family_of",
    "generate_allowed",
    "looks_sm120",
]

MEASURED_CAPABILITY = (8, 6)
TURING_CAPABILITY = (7, 5)

AMPERE_CAPABILITIES = frozenset({(8, 0), (8, 6), (8, 7), (8, 9)})
SM120_CAPABILITIES = frozenset({(12, 0), (12, 1)})
SM100_CAPABILITIES = frozenset({(10, 0), (10, 1), (10, 3)})
HOPPER_CAPABILITIES = frozenset({(9, 0)})
GENERATE_CAPABILITIES = AMPERE_CAPABILITIES | SM120_CAPABILITIES

#: Friend's card. Occupancy default 70 SMs matches this SKU and the 3080.
REMOTE_SM120_SKU = "RTX 5070 Ti"
REMOTE_SM120_SMS = 70
REMOTE_SM120_VRAM_MIB = 16384

# RTX 3050 must not match. RTX 5070 / 5070 Ti / 5080 / 5090 must.
_RTX_50_NAME = re.compile(r"(?i)\brtx\s*50[0-9]{2}\b")


def family_of(capability: tuple[int, int] | None) -> str:
    """Short family name for doctor ``Verdict.arch`` / messages."""
    if capability is None:
        return "unknown"
    cap = tuple(capability)
    if cap == MEASURED_CAPABILITY:
        return "ship"
    if cap in SM120_CAPABILITIES:
        return "sm120"
    if cap in AMPERE_CAPABILITIES:
        return "experimental"
    if cap == TURING_CAPABILITY:
        return "turing"
    if cap in HOPPER_CAPABILITIES:
        return "hopper"
    if cap in SM100_CAPABILITIES:
        return "sm100"
    return "unsupported"


def generate_allowed(capability: tuple[int, int] | None) -> bool:
    if capability is None:
        return False
    return tuple(capability) in GENERATE_CAPABILITIES


def looks_sm120(
    *,
    capability: tuple[int, int] | None = None,
    device_name: str | None = None,
) -> bool:
    """True for GeForce RTX 50 / SM120, including the remote 5070 Ti."""
    if capability is not None and tuple(capability) in SM120_CAPABILITIES:
        return True
    if device_name and _RTX_50_NAME.search(device_name):
        return True
    return False
