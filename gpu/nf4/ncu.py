"""Scripted Nsight Compute for ``chr_nf4_gemm`` at N=1 and N=16.

    python -m gpu.nf4.ncu --dry-run
    python -m gpu.nf4.ncu --profile
    python -m gpu.nf4.ncu --export

``gpu.nf4.bench`` CUDA-event microseconds are not DRAM / tensor-pipe /
occupancy. This file prints (and optionally runs) one isolated ``ncu``
invocation per (shape, N). Weight buffers rotate the same way as the bench
so a 2 MiB ``q_proj`` is not timed L2-resident.

If ``ncu`` is missing, the driver session refuses profiling, or a metric name
vanished: print ``SKIP: <reason>``. Do not invent percentages. Do not write
empty ``docs/runs/ncu/`` artifacts from a skip.

``--log-file`` is ncu's own log, not the metric table. After a successful
``.ncu-rep``, ``--export`` (and ``--profile``) run ``ncu --import --page details``
into ``*.metrics.csv`` and a median ``summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.bench import CORE, QWEN25_3B, Shape  # noqa: E402
from gpu.nf4.plan import plan  # noqa: E402

__all__ = [
    "CASES",
    "METRICS",
    "NCU_CANDIDATES",
    "export_report",
    "find_ncu",
    "ncu_command",
    "parse_details_csv",
    "summarize_launches",
    "write_summary",
    "main",
]

_DEFAULT_OUT = _REPO / "docs" / "runs" / "ncu"

#: Nsight Compute 2024.2 on this Windows box; also whatever is on PATH.
NCU_CANDIDATES = (
    Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.bat"),
    Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.bat"),
    Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\ncu.bat"),
)

#: Ampere names from docs/tz/wave10-perf.md §5. Unknown names are skipped by
#: ncu itself; we never fill a percentage from a blog.
METRICS = (
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__shared_mem_per_block_dynamic",
    "launch__registers_per_thread",
    "launch__block_size",
    "launch__grid_size",
)

#: Pinned cases: 3B q_proj and k_proj, decode N=1 and prefill N=16.
CASES: tuple[tuple[str, int], ...] = (
    ("q_proj", 1),
    ("q_proj", 16),
    ("k_proj", 1),
    ("k_proj", 16),
)

SUMMARY_COLUMNS = (
    "case",
    "kernel",
    "n_launches",
    "dram_pct",
    "tensor_pct",
    "occupancy_pct",
    "dram_read",
    "dram_read_unit",
    "dram_write",
    "dram_write_unit",
    "registers",
    "block",
    "grid",
    "smem",
    "smem_unit",
)


def find_ncu() -> Path | None:
    """Installed ncu, or None. A miss is a skip, not a fake profile."""
    which = shutil.which("ncu")
    if which:
        return Path(which)
    for candidate in NCU_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _shape(name: str) -> Shape:
    for shape in QWEN25_3B:
        if shape.name == name:
            return shape
    raise KeyError(name)


def case_stem(name: str, n: int) -> str:
    return f"qwen25-3b-{name}-n{n}"


def ncu_command(
    ncu: Path,
    *,
    name: str,
    n: int,
    python_exe: str,
    out_dir: Path,
    l2_bytes: int,
    iters: int,
) -> list[str]:
    """One ncu process: one matrix × one N. Kernel regex matches both tiles."""
    stem = case_stem(name, n)
    report = out_dir / f"{stem}.ncu-rep"
    log_out = out_dir / f"{stem}.log"
    return [
        str(ncu),
        "--target-processes",
        "all",
        "--kernel-name",
        "regex:chr_nf4_gemm",
        "--metrics",
        ",".join(METRICS),
        "--csv",
        "--log-file",
        str(log_out),
        "-o",
        str(report),
        "--force-overwrite",
        str(python_exe),
        "-m",
        "gpu.nf4.bench",
        "--n",
        str(n),
        "--only",
        name,
        "--modes",
        "wave9",
        "--iters",
        str(iters),
        "--reps",
        "1",
        "--l2-bytes",
        str(l2_bytes),
    ]


def _plan_note(name: str, n: int) -> str:
    shape = _shape(name)
    row = plan(shape.M, shape.K, n, have_ws=True)
    return (
        f"{name} M={shape.M} K={shape.K} N={n} "
        f"path={row.path} grid=({row.grid_x},{row.grid_y}) ctas={row.ctas} "
        f"smem={row.smem_bytes} ws_floats={row.ws_floats}"
    )


def _parse_number(text: str) -> float | None:
    """ncu --csv on a Russian Windows locale writes 5,46 not 5.46."""
    raw = (text or "").strip().replace(" ", "").replace("\u00a0", "")
    if not raw:
        return None
    if "," in raw and "." in raw:
        raw = raw.replace(",", "")
    elif "," in raw:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_details_csv(text: str) -> list[dict[str, str]]:
    """Rows from ``ncu --import --csv --page details``."""
    handle = io.StringIO(text)
    # Skip ncu chatter before the header.
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            start = i
            break
    handle = io.StringIO("\n".join(lines[start:]))
    return list(csv.DictReader(handle))


def summarize_launches(rows: list[dict[str, str]]) -> dict[str, object]:
    """Median over profiled launches. Drop ID 0 (first pass is the warmup)."""
    by_id: dict[str, dict[str, dict[str, str]]] = {}
    kernel = ""
    for row in rows:
        launch_id = row.get("ID") or row.get("Id") or ""
        metric = row.get("Metric Name") or ""
        if not metric:
            continue
        kernel = row.get("Kernel Name") or kernel
        by_id.setdefault(launch_id, {})[metric] = row

    ids = sorted(by_id, key=lambda x: int(x) if str(x).isdigit() else 0)
    if len(ids) > 1:
        ids = ids[1:]
    launches = [by_id[i] for i in ids]

    def values(metric: str) -> list[float]:
        out: list[float] = []
        for launch in launches:
            row = launch.get(metric)
            if not row:
                continue
            number = _parse_number(row.get("Metric Value") or "")
            if number is not None:
                out.append(number)
        return out

    def median(metric: str) -> float | None:
        got = values(metric)
        return statistics.median(got) if got else None

    sample = launches[0] if launches else {}

    def unit(metric: str) -> str:
        row = sample.get(metric) or {}
        return (row.get("Metric Unit") or "").strip()

    kernel_short = kernel
    if "chr_nf4_gemm" in kernel:
        kernel_short = kernel[kernel.find("chr_nf4_gemm"):].split("(")[0]

    return {
        "kernel": kernel_short,
        "n_launches": len(launches),
        "dram_pct": median("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "tensor_pct": median(
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
        ),
        "occupancy_pct": median("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "dram_read": median("dram__bytes_read.sum"),
        "dram_read_unit": unit("dram__bytes_read.sum"),
        "dram_write": median("dram__bytes_write.sum"),
        "dram_write_unit": unit("dram__bytes_write.sum"),
        "registers": median("launch__registers_per_thread"),
        "block": median("launch__block_size"),
        "grid": median("launch__grid_size"),
        "smem": median("launch__shared_mem_per_block_dynamic"),
        "smem_unit": unit("launch__shared_mem_per_block_dynamic"),
    }


def export_report(ncu: Path, report: Path, dest: Path) -> str:
    """``ncu --import --csv --page details`` into ``dest``. Returns stdout."""
    env = os.environ.copy()
    env["PATH"] = str(ncu.parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [str(ncu), "--import", str(report), "--csv", "--page", "details"],
        cwd=_REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ncu --import {report.name} exit {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '')[-500:]}"
        )
    dest.write_text(proc.stdout, encoding="utf-8")
    return proc.stdout


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})
    return path


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ncu_env(ncu: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(ncu.parent) + os.pathsep + env.get("PATH", "")
    return env


def _export_all(ncu: Path, out_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, n in CASES:
        stem = case_stem(name, n)
        report = out_dir / f"{stem}.ncu-rep"
        if not report.is_file():
            print(f"SKIP: {stem}: no {report.name}")
            continue
        metrics_csv = out_dir / f"{stem}.metrics.csv"
        print(f"export {report.name} -> {metrics_csv.name}", flush=True)
        text = export_report(ncu, report, metrics_csv)
        summary = summarize_launches(parse_details_csv(text))
        summary["case"] = stem
        rows.append(summary)
        print(
            f"  {stem} kernel={summary['kernel']} "
            f"dram={_fmt(summary['dram_pct'])}% "
            f"tensor={_fmt(summary['tensor_pct'])}% "
            f"occupancy={_fmt(summary['occupancy_pct'])}% "
            f"launches={summary['n_launches']}"
        )
    if rows:
        path = write_summary(out_dir, rows)
        print(f"wrote {path}")
    return rows


def _readme(ncu: Path, python_exe: str, l2_bytes: int, iters: int, notes: list[str], rows: list[dict[str, object]]) -> str:
    lines = [
        "WAVE 10 K2 ncu",
        f"ncu={ncu}",
        f"python={python_exe}",
        f"l2_bytes={l2_bytes} iters={iters}",
        "Weight buffers L2-rotated (--l2-bytes). Isolated one shape x N per ncu process.",
        "Not TokenLoop. Not bench.py microseconds.",
        "metrics:",
        *("  " + m for m in METRICS),
        "",
        *notes,
        "",
    ]
    if rows:
        lines.append(
            "Median over profiled launches after dropping ID 0 (warmup). "
            "Locale on this box uses a decimal comma in ncu CSV; summary.csv uses dots."
        )
        lines.append(
            "case                         kernel                         DRAM%  tensor%  occ%  regs  block  grid"
        )
        for row in rows:
            lines.append(
                f"{row['case']:<28} {str(row['kernel']):<28} "
                f"{_fmt(row['dram_pct']):>6}  {_fmt(row['tensor_pct']):>7}  "
                f"{_fmt(row['occupancy_pct']):>4}  {_fmt(row['registers'], 0):>4}  "
                f"{_fmt(row['block'], 0):>5}  {_fmt(row['grid'], 0):>4}"
            )
        lines.append("")
        lines.append(
            "These percentages are ncu counters on the GEMM, not live tok/s. "
            "Do not paste them into the README speed line."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print commands, no ncu")
    ap.add_argument("--profile", action="store_true", help="run ncu (needs GPU performance counters)")
    ap.add_argument(
        "--export",
        action="store_true",
        help="import existing .ncu-rep files into metrics.csv + summary.csv",
    )
    ap.add_argument("--out", default=str(_DEFAULT_OUT), help="docs/runs/ncu")
    ap.add_argument("--python", dest="python_exe", default=sys.executable)
    ap.add_argument("--l2-bytes", type=int, default=24 << 20)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args(argv)

    ncu = find_ncu()
    out_dir = Path(args.out)
    print("Nsight driver for chr_nf4_gemm (WAVE 10 K2). bench.py is not this.")
    print("Pinned: q_proj / k_proj at N=1 and N=16. No TokenLoop (SDPA would drown it).")
    print()
    for name, n in CASES:
        if name not in CORE:
            print(f"SKIP: {name} is not in bench --shapes core")
            continue
        print(_plan_note(name, n))

    if ncu is None:
        print("\nSKIP: ncu not on PATH and not under Nsight Compute 2024.2/2024.3/2025.1")
        print("Install Nsight Compute or add ncu.bat to PATH. No empty ncu files written.")
        return 0

    print(f"\nncu: {ncu}")

    if args.export and not args.profile:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _export_all(ncu, out_dir)
        readme = out_dir / "README.txt"
        readme.write_text(
            _readme(ncu, args.python_exe, args.l2_bytes, args.iters, ["export of existing .ncu-rep"], rows),
            encoding="utf-8",
        )
        print(f"wrote {readme}")
        return 0 if rows else 1

    commands = [
        ncu_command(
            ncu,
            name=name,
            n=n,
            python_exe=args.python_exe,
            out_dir=out_dir,
            l2_bytes=args.l2_bytes,
            iters=args.iters,
        )
        for name, n in CASES
    ]
    for cmd in commands:
        print("\n ", " ".join(cmd))

    if args.dry_run or not args.profile:
        if not args.profile:
            print("\nNot profiling (pass --profile). Nothing written under docs/runs/ncu.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.txt"
    env_ncu = _ncu_env(ncu)

    notes: list[str] = []
    rc = 0
    for cmd, (name, n) in zip(commands, CASES):
        print(f"\n--- ncu {case_stem(name, n)} ---", flush=True)
        try:
            proc = subprocess.run(
                cmd,
                cwd=_REPO,
                env=env_ncu,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            line = f"SKIP: {case_stem(name, n)}: could not launch ncu: {exc}"
            print(line)
            notes.append(line)
            rc = 1
            continue
        text = (proc.stdout or "") + (proc.stderr or "")
        if text.strip():
            print(text[-4000:], end="" if text.endswith("\n") else "\n")
        report = out_dir / f"{case_stem(name, n)}.ncu-rep"
        log_out = out_dir / f"{case_stem(name, n)}.log"
        if proc.returncode != 0:
            if "ERR_NVGPUCTRPERM" in text:
                line = (
                    f"SKIP: {case_stem(name, n)}: ERR_NVGPUCTRPERM "
                    "(NVIDIA GPU Performance Counters not enabled for this user). "
                    "No DRAM/tensor percentages invented."
                )
            else:
                line = (
                    f"SKIP: {case_stem(name, n)}: ncu exit {proc.returncode} "
                    "(profiling session denied, driver, or metric name)"
                )
            print(line)
            notes.append(line)
            for leftover in (report, log_out):
                leftover.unlink(missing_ok=True)
            rc = 1
            continue
        notes.append(f"OK {case_stem(name, n)}")

    rows: list[dict[str, object]] = []
    if rc == 0:
        try:
            rows = _export_all(ncu, out_dir)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"SKIP: export failed: {type(exc).__name__}: {exc}")
            rc = 1

    readme.write_text(
        _readme(ncu, args.python_exe, args.l2_bytes, args.iters, notes, rows),
        encoding="utf-8",
    )
    print(f"\nwrote {readme}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
