"""VRAM / allocation / file-immutability probes for the wave-2 safety gate.

Source of truth for VRAM is `nvidia-smi --query-gpu=memory.used`
(docs/spec/gpu-safety.md). The torch allocator counters are a second, finer
witness: they catch a dequantized W that is allocated and freed inside one call,
which smi would still show as grown *reserved*, but with less detail.

Coverage limits are stated in docs/spec/gpu-safety.md SS6 -- a raw cudaMalloc
inside a C++ extension is invisible to the torch-level probes and is caught by
smi only.
"""

from __future__ import annotations

import builtins
import os
import statistics
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

MIB = 1024 * 1024


def smi_used_mib(index: int = 0, samples: int = 3) -> int:
    """memory.used in MiB for one GPU, median of a few samples."""
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.used",
        "--format=csv,nounits,noheader",
    ]
    vals = []
    for _ in range(max(1, samples)):
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        vals.append(int(out.stdout.strip().splitlines()[0]))
    return int(statistics.median(vals))


def nf4_vram_budget(m: int, k: int, n: int) -> dict[str, float]:
    """Bytes/MiB that a *correct* NF4 linear may hold on device for one matrix."""
    k_pad = 64 * ((k + 63) // 64)
    packed = m * k_pad // 2
    scale = m * (k_pad // 64) * 2
    x = k * n * 2
    y = m * n * 2
    bf16_w = m * k * 2
    return {
        "packed_mib": packed / MIB,
        "scale_mib": scale / MIB,
        "x_mib": x / MIB,
        "y_mib": y / MIB,
        "legit_mib": (packed + scale + x + y) / MIB,
        "bf16_w_mib": bf16_w / MIB,
        "fp32_w_mib": (m * k * 4) / MIB,
    }


@dataclass
class Alloc:
    op: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


class DequantWatcher:
    """Records big float cuda tensors produced by any aten op inside the block.

    S2: a [M, K] fp16/bf16/fp32 tensor is a dequantized W in HBM -- forbidden by
    stitch-gpu.md ("draft W: none in HBM").
    """

    def __init__(self, min_numel: int, *, dtypes: Iterable[str] = ("torch.float16", "torch.bfloat16", "torch.float32")):
        self.min_numel = int(min_numel)
        self.dtypes = set(dtypes)
        self.hits: list[Alloc] = []
        self.error: str | None = None
        self._mode: Any = None

    def __enter__(self) -> "DequantWatcher":
        try:
            import torch
            from torch.utils._python_dispatch import TorchDispatchMode

            watcher = self

            class _Mode(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
                    out = func(*args, **(kwargs or {}))
                    try:
                        watcher._scan(str(func), out, torch)
                    except Exception:  # never let the probe break the gate
                        pass
                    return out

            self._mode = _Mode()
            self._mode.__enter__()
        except Exception as exc:  # torch too old / mode unavailable
            self.error = f"{type(exc).__name__}: {exc}"
            self._mode = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._mode is not None:
            try:
                self._mode.__exit__(*exc)
            except Exception:
                pass
            self._mode = None

    def _scan(self, op: str, out: Any, torch: Any) -> None:
        if isinstance(out, (tuple, list)):
            for o in out:
                self._scan(op, o, torch)
            return
        if not isinstance(out, torch.Tensor):
            return
        if not out.is_cuda or str(out.dtype) not in self.dtypes:
            return
        if out.numel() >= self.min_numel:
            self.hits.append(
                Alloc(
                    op=op,
                    shape=tuple(out.shape),
                    dtype=str(out.dtype),
                    nbytes=out.numel() * out.element_size(),
                )
            )


@dataclass
class LaunchCounter:
    """Counts gemm invocations that go through the test adapter (S4 wants 0)."""

    count: int = 0

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*a: Any, **kw: Any) -> Any:
            self.count += 1
            return fn(*a, **kw)

        return inner


@dataclass
class ReadOnlyFileGuard:
    """S3/S6: the .chr must be opened read-only and must not change on disk."""

    path: str
    modes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    _stat_before: tuple[int, int] = (0, 0)
    _open: Any = None
    _os_open: Any = None

    def __enter__(self) -> "ReadOnlyFileGuard":
        target = os.path.abspath(self.path)
        st = os.stat(target)
        self._stat_before = (st.st_mtime_ns, st.st_size)
        guard = self

        self._open = builtins.open
        real_open = self._open

        def patched_open(file: Any, mode: str = "r", *a: Any, **kw: Any) -> Any:
            try:
                same = os.path.abspath(str(file)) == target
            except Exception:
                same = False
            if same:
                guard.modes.append(mode)
                if any(ch in mode for ch in "wax+"):
                    guard.violations.append(f"builtins.open(mode={mode!r})")
            return real_open(file, mode, *a, **kw)

        builtins.open = patched_open  # type: ignore[assignment]

        self._os_open = os.open
        real_os_open = self._os_open
        write_flags = 0
        for name in ("O_WRONLY", "O_RDWR", "O_APPEND", "O_CREAT", "O_TRUNC"):
            write_flags |= getattr(os, name, 0)

        def patched_os_open(path: Any, flags: int, *a: Any, **kw: Any) -> Any:
            try:
                same = os.path.abspath(str(path)) == target
            except Exception:
                same = False
            if same:
                guard.modes.append(f"os.open(flags=0x{flags:x})")
                if flags & write_flags:
                    guard.violations.append(f"os.open(flags=0x{flags:x})")
            return real_os_open(path, flags, *a, **kw)

        os.open = patched_os_open  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._open is not None:
            builtins.open = self._open  # type: ignore[assignment]
            self._open = None
        if self._os_open is not None:
            os.open = self._os_open  # type: ignore[assignment]
            self._os_open = None
        st = os.stat(os.path.abspath(self.path))
        after = (st.st_mtime_ns, st.st_size)
        if after != self._stat_before:
            self.violations.append(f"stat changed: {self._stat_before} -> {after}")

    @property
    def ok(self) -> bool:
        return not self.violations
