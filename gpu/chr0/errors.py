"""Error taxonomy for the CHR0 GPU loader.

One base class so callers can do ``except Chr0Error``; one subclass per failure
mode listed in ``docs/spec/gpu-abi.md`` §5 so tests can be specific.
"""

from __future__ import annotations

__all__ = [
    "Chr0Error",
    "TruncatedError",
    "HeaderError",
    "AlignmentError",
    "SizeMismatchError",
    "OverlapError",
    "CodecError",
    "GroupSizeError",
    "TensorNotFoundError",
]


class Chr0Error(Exception):
    """Base class for every rejection of a ``.chr`` file."""


class TruncatedError(Chr0Error):
    """File is shorter than the header or a blob range claims."""


class HeaderError(Chr0Error):
    """``header_nbytes`` or the JSON header violates chr0.md §1.2-§2."""


class AlignmentError(Chr0Error):
    """A blob ``start`` is not a multiple of 64."""


class SizeMismatchError(Chr0Error):
    """``end - start`` disagrees with the size formula for the codec."""


class OverlapError(Chr0Error):
    """Two blob ranges intersect."""


class CodecError(Chr0Error):
    """Unknown codec, wrong key set for a codec, or a codec this wave refuses."""


class GroupSizeError(Chr0Error):
    """``group_size`` is not the canonical value for the codec."""


class TensorNotFoundError(Chr0Error):
    """No such CHR0 name in ``tensors``."""
