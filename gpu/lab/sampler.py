"""Background recorder: one ``nvidia-smi`` shell-out plus torch counters, <= 0.15 s.

Same idea as ``VramTimeline`` in ``notebooks/build_02_nf4_gpu_driver.py``, minus
the SVG: this one only produces rows in the frozen ``timeline.csv`` schema and
events in ``events.csv``. Drawing is ``gpu.lab.plot``'s job.

Two rules worth stating:

* ``nvidia-smi memory.used`` is the product metric. It includes the CUDA context,
  the graph pool and the desktop compositor. We do not subtract any of that away.
* A field the driver reports as ``N/A`` becomes an empty CSV cell, never ``0``.
  Laptop-class cards do that for ``power.draw`` and nobody should read the gap as
  "zero watts".
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

from .bundle import CODECS

__all__ = ["SMI_FIELDS", "Sampler", "smi_query", "smi_used_mib"]

MIB = 1024 * 1024

# Order matters: it is the order of the timeline columns.
SMI_FIELDS = (
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)

_SMI_TO_COLUMN = {
    "memory.used": "used_mib",
    "memory.total": "total_mib",
    "utilization.gpu": "util_gpu",
    "utilization.memory": "util_mem",
    "power.draw": "power_w",
    "temperature.gpu": "temp_c",
    "clocks.sm": "clock_sm_mhz",
    "clocks.mem": "clock_mem_mhz",
}

_NOT_A_NUMBER = ("n/a", "[n/a]", "[not supported]", "[unknown error]", "")


def _to_float(text: str) -> float | None:
    cleaned = (text or "").strip()
    if cleaned.lower() in _NOT_A_NUMBER:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def smi_query(fields: tuple[str, ...] = SMI_FIELDS, *, index: int = 0) -> list[float | None]:
    """One ``nvidia-smi`` call. ``N/A`` comes back as ``None``."""
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,nounits,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    first_line = result.stdout.strip().splitlines()[0]
    return [_to_float(cell) for cell in first_line.split(",")]


def smi_used_mib(*, index: int = 0) -> float | None:
    """VRAM in use on the whole card, right now."""
    try:
        return smi_query(("memory.used",), index=index)[0]
    except Exception:  # noqa: BLE001 -- no driver, no numbers; the CSV cell stays empty
        return None


class Sampler:
    """Polls the card on a thread for the whole session, load to unload.

    One sampler per codec session. ``t_s`` is seconds since :meth:`start`, so the
    two sessions overlay on a single x axis without any post-hoc alignment.
    """

    def __init__(
        self,
        codec: str,
        *,
        interval_s: float = 0.12,
        gpu_index: int = 0,
        verbose: bool = False,
    ) -> None:
        if codec not in CODECS:
            raise ValueError(f"codec={codec!r}; expected one of {CODECS}")
        if interval_s < 0 or interval_s > 0.15:
            raise ValueError(
                f"interval_s={interval_s}; 0 disables the smi thread, else freeze says <= 0.15 s"
            )
        self.codec = codec
        self.interval_s = float(interval_s)
        self.gpu_index = int(gpu_index)
        self.verbose = bool(verbose)
        self.rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._t0: float | None = None
        self._message_id = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- clock ------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self._t0 is not None

    def now(self) -> float:
        """Session-local seconds. ``0.0`` before :meth:`start`."""
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    # --- one sample -------------------------------------------------------
    def _poll(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "t_s": self.now(),
            "codec": self.codec,
            "message_id": self._message_id,
        }
        try:
            values = smi_query(SMI_FIELDS, index=self.gpu_index)
        except Exception as error:  # noqa: BLE001 -- keep sampling, record the reason once
            message = f"{type(error).__name__}: {error}"
            if message not in self.errors:
                self.errors.append(message)
            values = [None] * len(SMI_FIELDS)
        for field, value in zip(SMI_FIELDS, values):
            row[_SMI_TO_COLUMN[field]] = value
        row.update(self._torch_counters())
        return row

    @staticmethod
    def _torch_counters() -> dict[str, float | None]:
        """Allocator view, next to the driver view. They disagree, and that gap
        (context + fragmentation) is exactly what the figure should show."""
        try:
            import torch
        except Exception:  # noqa: BLE001
            return {
                "torch_alloc_mib": None,
                "torch_reserved_mib": None,
                "torch_max_alloc_mib": None,
            }
        if not torch.cuda.is_available():
            return {
                "torch_alloc_mib": None,
                "torch_reserved_mib": None,
                "torch_max_alloc_mib": None,
            }
        return {
            "torch_alloc_mib": torch.cuda.memory_allocated() / MIB,
            "torch_reserved_mib": torch.cuda.memory_reserved() / MIB,
            "torch_max_alloc_mib": torch.cuda.max_memory_allocated() / MIB,
        }

    # --- marks ------------------------------------------------------------
    def mark(self, event: str, message_id: int | str | None = None, detail: str = "") -> float:
        """Record an event. ``msg_send`` / ``msg_done`` also open and close the
        turn, which is what tags the timeline rows in between."""
        identifier = "" if message_id in (None, "") else str(message_id)
        if event == "msg_send":
            self._message_id = identifier
        t_s = self.now()
        with self._lock:
            self.events.append(
                {
                    "t_s": t_s,
                    "codec": self.codec,
                    "event": event,
                    "message_id": identifier,
                    "detail": detail,
                }
            )
        if event == "msg_done":
            self._message_id = ""
        if self.verbose:
            tail = f"  {detail}" if detail else ""
            label = f"{event}" + (f"[{identifier}]" if identifier else "")
            print(f"[{self.codec}] {t_s:7.2f}s  {label}{tail}", flush=True)
        return t_s

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Begin sampling and emit ``start``. Call before the model loads."""
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001 -- no torch is fine, the columns stay empty
            pass
        self._t0 = time.perf_counter()
        with self._lock:
            self.rows.append(self._poll())
        self.mark("start", detail=f"codec={self.codec}, interval={self.interval_s:.2f}s")
        self._stop.clear()
        if self.interval_s <= 0:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"lab-sampler-{self.codec}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            row = self._poll()
            with self._lock:
                self.rows.append(row)
            self._stop.wait(self.interval_s)

    def stop(self, detail: str = "") -> None:
        """Stop sampling and emit ``stop``. Call after the model is unloaded."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self.rows.append(self._poll())
        peak = self.peak_used_mib
        peak_text = "unknown" if peak is None else f"{peak:.0f} MiB"
        self.mark("stop", detail=detail or f"peak_smi={peak_text}")

    # --- results ----------------------------------------------------------
    @property
    def peak_used_mib(self) -> float | None:
        values = [row.get("used_mib") for row in self.rows if row.get("used_mib") is not None]
        return max(values) if values else None  # type: ignore[type-var]

    def timeline_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.rows)

    def event_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events)

    def __repr__(self) -> str:
        return (
            f"Sampler(codec={self.codec!r}, interval_s={self.interval_s}, "
            f"rows={len(self.rows)}, events={len(self.events)}, "
            f"running={self._thread is not None})"
        )
