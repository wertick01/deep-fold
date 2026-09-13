"""The codec lab: BF16 vs our NF4 driver, same GPU, same chat, one figure.

Wave 4, agent 10. Owns ``gpu/lab/`` and nothing else. The notebook and the
README only import this package; the recorder, the two sessions, the CSV writers
and the Plotly figure all live here.

    import sys; sys.path.insert(0, r"C:\\dev\\deep-fold")
    from gpu.lab import run_both, comparison_figure, MESSAGES

    bundle = run_both(out_dir)              # BF16 worker, exit, NF4 worker
    fig = comparison_figure(bundle)         # TWO VRAM graphs, same Y scale
    fig.write_html(out_dir / "lab.html")

``comparison_figure`` also takes a finished run directory, so a reader with no
GPU can redraw the same figure from the CSVs::

    fig = comparison_figure(r"C:\\dev\\models\\runs\\lab-20260912-2300")

Artifacts, all with English headers and labels: ``timeline.csv``, ``events.csv``,
``messages.csv``, ``summary.csv``, ``lab.html`` and -- if kaleido is installed --
``lab.png``.

Contracts: ``docs/lab.md`` (CSV schema and figure). Acceptance:
``python -m gpu.lab.test_lab`` (fixture only, never loads the 3B).

``run_bf16`` / ``run_nf4`` / ``run_both`` are imported lazily, so pulling
``comparison_figure`` or ``MESSAGES`` into a notebook does not drag in torch.
"""

from __future__ import annotations

from typing import Any

from .bundle import (
    CODECS,
    EVENTS_COLUMNS,
    MESSAGES_COLUMNS,
    REQUIRED_EVENTS,
    SUMMARY_COLUMNS,
    TIMELINE_COLUMNS,
    LabBundle,
)
from .script import (
    CHR_PATH,
    MAX_NEW_TOKENS,
    MAX_SEQ,
    MESSAGES,
    MODEL_DIR,
    NEEDLES,
    POLL_INTERVAL_S,
    RUNS_DIR,
    quality_ok,
)

__all__ = [
    # the frozen public surface
    "MESSAGES",
    "comparison_figure",
    "run_both",
    "run_bf16",
    "run_nf4",
    "quality_ok",
    # schema and artifacts
    "LabBundle",
    "CODECS",
    "REQUIRED_EVENTS",
    "TIMELINE_COLUMNS",
    "EVENTS_COLUMNS",
    "MESSAGES_COLUMNS",
    "SUMMARY_COLUMNS",
    "fixture_bundle",
    "write_artifacts",
    "write_png",
    "CODEC_STYLE",
    "LabSession",
    "Sampler",
    "unload",
    # knobs
    "NEEDLES",
    "MODEL_DIR",
    "CHR_PATH",
    "RUNS_DIR",
    "MAX_NEW_TOKENS",
    "MAX_SEQ",
    "POLL_INTERVAL_S",
]

# Attribute -> module, resolved on first use (PEP 562). `plot` needs plotly and
# `sessions` needs torch + transformers; neither should be a cost of `import
# gpu.lab` for someone who only wants the schema or the chat script.
_LAZY = {
    "comparison_figure": ".plot",
    "write_artifacts": ".plot",
    "write_png": ".plot",
    "CODEC_STYLE": ".plot",
    "run_both": ".sessions",
    "run_bf16": ".sessions",
    "run_nf4": ".sessions",
    "unload": ".sessions",
    "LabSession": ".sessions",
    "Sampler": ".sampler",
    "fixture_bundle": ".fixture",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache, so the import cost is paid once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
