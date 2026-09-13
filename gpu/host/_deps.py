"""Import shim for the two sibling packages this host depends on.

``gpu/`` has no ``__init__.py`` on purpose (one agent per subdirectory), so this
package has to work both ways:

* repo root on ``sys.path``  -> ``import gpu.host``  (relative import works)
* ``gpu/`` on ``sys.path``   -> ``import host``      (relative import does not)

Nothing here reimplements the loader or the kernel; it only finds them.
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gpu.chr0 import (  # noqa: E402
    Chr0Error,
    Header,
    TensorInfo,
    TruncatedError,
    iter_linears,
    load_header,
    materialize_nf4,
)
from gpu.chr0.loader import ChrMatrix  # noqa: E402


_gemm = None


def nf4_gemm(packed, scale, x, M, K, K_pad):  # noqa: N803
    """Lazy handle on agent 2's extension: importing it may trigger a JIT build.

    Resolved once and cached -- this is on the token path, 253 calls per token on
    3B, and re-entering the import machinery there buys nothing.
    """
    global _gemm
    if _gemm is None:
        from gpu.nf4 import nf4_gemm as fn

        _gemm = fn
    return _gemm(packed, scale, x, M, K, K_pad)


__all__ = [
    "Chr0Error",
    "ChrMatrix",
    "Header",
    "TensorInfo",
    "TruncatedError",
    "iter_linears",
    "load_header",
    "materialize_nf4",
    "nf4_gemm",
]
