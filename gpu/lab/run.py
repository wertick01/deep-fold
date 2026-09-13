"""CLI for the codec lab.

    python -m gpu.lab.run --out C:\\dev\\models\\runs\\lab-test --dry-plot
    python -m gpu.lab.run --out C:\\dev\\models\\runs\\lab-%Y%m%d-%H%M%S

``--dry-plot`` builds the artifacts from the synthetic fixture: the four CSVs
plus ``lab.html``, no CUDA and no model required. Anything else is a live run
and loads the 3B, one codec at a time, on the paths from ``docs/lab.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.bundle import LabBundle  # noqa: E402
from gpu.lab.script import (  # noqa: E402
    CHR_PATH,
    MAX_NEW_TOKENS,
    MAX_SEQ,
    MODEL_DIR,
    POLL_INTERVAL_S,
    RUNS_DIR,
)

__all__ = ["main"]

DEFAULT_OUT = str(Path(RUNS_DIR) / "lab-%Y%m%d-%H%M%S")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.run",
        description="BF16 vs NF4 on one GPU: CSVs plus one interactive Plotly figure.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="output directory; strftime patterns are expanded (default: %(default)s)",
    )
    parser.add_argument(
        "--codec",
        choices=("both", "bf16", "nf4"),
        default="both",
        help="which session(s) to run (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-plot",
        action="store_true",
        help="build the artifacts from the synthetic fixture; no CUDA, no model",
    )
    parser.add_argument(
        "--from-csv",
        metavar="DIR",
        help="redraw the figure from a finished run's CSVs instead of measuring",
    )
    parser.add_argument("--model-dir", default=MODEL_DIR, help="HF directory, BF16 session only")
    parser.add_argument("--chr", dest="chr_path", default=CHR_PATH, help="NF4 .chr file")
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="greedy cap per message"
    )
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ, help="NF4 KV cache length")
    parser.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL_S,
        help="sampler period in seconds, must be <= 0.15 (default: %(default)s)",
    )
    parser.add_argument(
        "--no-graphs",
        dest="graphs",
        action="store_false",
        help="skip capture_graphs() on the NF4 session, stay eager",
    )
    parser.add_argument(
        "--no-png", dest="png", action="store_false", help="do not attempt the kaleido export"
    )
    parser.add_argument(
        "--no-telemetry",
        dest="telemetry",
        action="store_false",
        help="write only the plate, not the lab-telemetry.* instrument sheet",
    )
    parser.add_argument(
        "--embed-js",
        action="store_true",
        help="inline plotly.js into lab.html instead of linking the CDN",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print event marks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # so a Russian reply is readable
    except Exception:  # noqa: BLE001
        pass

    out_dir = Path(time.strftime(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        bundle = LabBundle.read(args.from_csv)
        print(f"read {bundle!r} from {args.from_csv}")
        bundle.write(out_dir)
    elif args.dry_plot:
        from gpu.lab.fixture import fixture_bundle

        codec = args.codec if args.codec != "both" else "both"
        bundle = fixture_bundle(interval_s=max(args.interval, 0.25), codec=codec)
        bundle.write(out_dir)
        print(f"dry-plot: fixture {bundle!r} (SYNTHETIC, not a measurement)")
    else:
        from gpu.lab.sessions import run_both

        bundle = run_both(
            out_dir,
            codec=args.codec,
            model_dir=args.model_dir,
            chr_path=args.chr_path,
            max_new_tokens=args.max_new_tokens,
            max_seq=args.max_seq,
            interval_s=args.interval,
            graphs=args.graphs,
            verbose=not args.quiet,
        )
        print(f"live run: {bundle!r}")

    from gpu.lab.plot import comparison_figure, telemetry_figure, write_artifacts

    plotly_js = True if args.embed_js else "cdn"
    figure = comparison_figure(bundle)
    artifacts = write_artifacts(figure, out_dir, png=args.png, include_plotlyjs=plotly_js)
    extra = (
        write_artifacts(
            telemetry_figure(bundle),
            out_dir,
            png=args.png,
            include_plotlyjs=plotly_js,
            name="lab-telemetry",
        )
        if args.telemetry
        else {}
    )

    print(f"\nout: {out_dir}")
    for name in ("timeline", "events", "messages", "summary"):
        path = out_dir / f"{name}.csv"
        rows = max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)
        print(f"  {path.name:<14} {rows} row(s)")
    for label, written in (("lab", artifacts), ("lab-telemetry", extra)):
        if not written:
            continue
        print(f"  {label + '.html':<19} {Path(written['html']).stat().st_size // 1024} KiB")
        if written.get("png"):
            print(f"  {label + '.png':<19} {Path(written['png']).stat().st_size // 1024} KiB")
        else:
            print(
                f"  {label + '.png':<19} SKIPPED "
                f"({written.get('png_skip_reason') or 'unknown'})"
            )

    _report(bundle)
    return 0


def _report(bundle: LabBundle) -> None:
    """The comparison table, in the same numbers the CSVs carry."""
    if not bundle.summary:
        print("\nno session recorded")
        return
    print("\ncodec  peak_smi_mib  after_load_smi  weight_mib  mean_ttft_ms  mean_tok_s  quality")
    for row in bundle.summary:

        def cell(key: str, width: int, decimals: int = 0) -> str:
            value = row.get(key)
            text = "n/a" if value is None else f"{float(value):.{decimals}f}"
            return text.rjust(width)

        quality = row.get("quality_all_ok")
        print(
            f"{str(row['codec']):<6} {cell('vram_peak_smi_mib', 12)} "
            f"{cell('vram_after_load_smi_mib', 15)} {cell('weight_mib', 11)} "
            f"{cell('mean_ttft_ms', 13)} {cell('mean_decode_tok_s', 11, 1)} "
            f"  {'ok' if quality else 'NOT ok'}"
        )
    for row in bundle.summary:
        print(f"\n{row['codec']} notes: {row.get('notes', '')}")
    if bundle.synthetic:
        print("\nSYNTHETIC FIXTURE DATA -- do not quote these numbers anywhere.")


if __name__ == "__main__":
    raise SystemExit(main())
