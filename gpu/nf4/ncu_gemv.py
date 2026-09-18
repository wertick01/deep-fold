"""Scripted Nsight Compute for Decode V2 CUDA-core GEMV / SwiGLU.

    python -m gpu.nf4.ncu_gemv --dry-run
    python -m gpu.nf4.ncu_gemv --profile
    python -m gpu.nf4.ncu_gemv --export

Does not touch ``docs/runs/ncu/`` (that plate is ``chr_nf4_gemm``). Does not
edit ``nf4_gemm.cu``. Isolated shapes, L2-rotated, not a live 3B graph.
If ncu or counters refuse: ``SKIP``, no empty artifacts, no invented %.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.bench import QWEN25_3B, Shape  # noqa: E402
from gpu.nf4.ncu import (  # noqa: E402
    METRICS,
    SUMMARY_COLUMNS,
    find_ncu,
    parse_details_csv,
    summarize_launches,
    write_summary,
)
from gpu.nf4.ncu import _ncu_env, _parse_number  # noqa: E402

__all__ = ["CASES", "export_report_gemv", "ncu_command", "main"]

_DEFAULT_OUT = _REPO / "docs" / "runs" / "ncu-gemv"
_RENAMES = _REPO / "gpu" / "nf4" / "ncu_gemv_renames.yaml"

#: ``launch__grid_size`` on GEMV is ``(2048, 1, 1)``; ncu --import --csv dies
#: with ``bad conversion`` on this Russian locale. Grid comes from the CSV
#: ``Grid Size`` column instead.
_IMPORT_METRICS = tuple(m for m in METRICS if m != "launch__grid_size")

#: 3B decode shapes that dominate the greedy-step mix (lab log §4).
CASES: tuple[tuple[str, str], ...] = (
    ("o_proj", "regex:gemv_splitk"),
    ("down_proj", "regex:gemv_splitk"),
    ("swiglu", "regex:gemv_swiglu"),
)


def _shape_for(kind: str) -> Shape:
    name = "gate_proj" if kind == "swiglu" else kind
    for shape in QWEN25_3B:
        if shape.name == name:
            return shape
    raise KeyError(kind)


def case_stem(kind: str) -> str:
    return f"qwen25-3b-{kind}-gemv"


def ncu_command(
    ncu: Path,
    *,
    kind: str,
    kernel_regex: str,
    python_exe: str,
    out_dir: Path,
    l2_bytes: int,
    iters: int,
) -> list[str]:
    stem = case_stem(kind)
    report = out_dir / f"{stem}.ncu-rep"
    log_out = out_dir / f"{stem}.log"
    return [
        str(ncu),
        "--target-processes",
        "all",
        # demangled so regex:gemv_splitk hits anonymous-namespace kernels.
        # function-base matched 0 kernels on this cubin. Import of
        # launch__grid_size dies ("bad conversion" on (2048,1,1) + RU locale);
        # _export_all pulls every other metric one name at a time.
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        kernel_regex,
        "--metrics",
        ",".join(_IMPORT_METRICS),
        "--log-file",
        str(log_out),
        "-o",
        str(report),
        "--force-overwrite",
        str(python_exe),
        "-m",
        "gpu.nf4.bench_gemv_ncu",
        "--kind",
        kind,
        "--iters",
        str(iters),
        "--reps",
        "1",
        "--l2-bytes",
        str(l2_bytes),
    ]


def _run_ncu(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """``cmd /c call "ncu.BAT" args`` — CreateProcess on the .BAT drops flags."""
    del env
    if os.name == "nt":
        line = "call " + subprocess.list2cmdline(cmd)
        return subprocess.run(line, shell=True, cwd=_REPO, check=False)
    return subprocess.run(cmd, cwd=_REPO, check=False)


def _short_kernel(name: str) -> str:
    for token in ("gemv_swiglu", "gemv_splitk", "chr_nf4_gemv"):
        i = name.find(token)
        if i < 0:
            continue
        rest = name[i:]
        gt = rest.rfind(">")
        if gt >= 0:
            return rest[: gt + 1]
        return rest.split("(")[0]
    return name


def _grid_from_size_column(text: str) -> float | None:
    """``(2048, 1, 1)`` → 2048. Missing/empty → None."""
    inner = (text or "").strip().strip("()")
    if not inner:
        return None
    nums = [_parse_number(part) for part in inner.split(",")]
    if not nums or any(n is None for n in nums):
        return None
    prod = 1.0
    for number in nums:
        prod *= number
    return prod


def export_report_gemv(ncu: Path, report: Path, dest: Path, rename_yaml: Path) -> str:
    """One ``--metrics`` name per ncu --import. Full-table CSV hits grid conversion."""
    ncu_bin = ncu
    if ncu.suffix.lower() == ".bat":
        exe = ncu.parent / "target" / "windows-desktop-win7-x64" / "ncu.exe"
        if exe.is_file():
            ncu_bin = exe
    env = _ncu_env(ncu_bin)
    header = ""
    body: list[str] = []
    skipped: list[str] = []
    for metric in _IMPORT_METRICS:
        proc = subprocess.run(
            [
                str(ncu_bin),
                "--import",
                str(report),
                "--rename-kernels-path",
                str(rename_yaml),
                "--kernel-name-base",
                "demangled",
                "--metrics",
                metric,
                "--csv",
                "--page",
                "details",
            ],
            cwd=_REPO,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            skipped.append(metric)
            continue
        lines = proc.stdout.splitlines()
        start = 0
        for i, line in enumerate(lines):
            if line.startswith('"ID"') or line.startswith("ID,"):
                start = i
                break
        else:
            skipped.append(metric)
            continue
        if not header:
            header = lines[start]
        body.extend(lines[start + 1 :])
    if not header or not body:
        raise RuntimeError(
            f"ncu --import {report.name}: no metric rows "
            f"(skipped {skipped or 'none'})"
        )
    text = "\n".join([header, *body]) + "\n"
    dest.write_text(text, encoding="utf-8")
    if skipped:
        print(f"  skipped metrics: {', '.join(skipped)}")
    return text


def _export_all(ncu: Path, out_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not _RENAMES.is_file():
        print(f"SKIP: export needs {_RENAMES}")
        return rows
    for kind, _regex in CASES:
        stem = case_stem(kind)
        report = out_dir / f"{stem}.ncu-rep"
        if not report.is_file():
            print(f"SKIP: {stem}: no {report.name}")
            continue
        metrics_csv = out_dir / f"{stem}.metrics.csv"
        print(f"export {report.name} -> {metrics_csv.name}", flush=True)
        text = export_report_gemv(ncu, report, metrics_csv, _RENAMES)
        parsed = parse_details_csv(text)
        summary = summarize_launches(parsed)
        summary["case"] = stem
        summary["kernel"] = _short_kernel(str(summary.get("kernel") or ""))
        if parsed:
            grid = _grid_from_size_column(parsed[0].get("Grid Size") or "")
            if grid is not None:
                summary["grid"] = grid
        rows.append(summary)
        print(
            f"  {stem} kernel={summary['kernel']} "
            f"dram={summary['dram_pct']} occ={summary['occupancy_pct']} "
            f"grid={summary['grid']} launches={summary['n_launches']}"
        )
    if rows:
        write_summary(out_dir, rows)
        print(f"wrote {out_dir / 'summary.csv'}")
    return rows


def _readme(
    ncu: Path,
    python_exe: str,
    l2_bytes: int,
    iters: int,
    notes: list[str],
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "Decode V2 CUDA-core GEMV / SwiGLU ncu",
        f"ncu={ncu}",
        f"python={python_exe}",
        f"l2_bytes={l2_bytes} iters={iters}",
        "L2-rotated isolated shapes. Not TokenLoop. Not chr_nf4_gemm.",
        "launch__grid_size omitted: ncu --import --csv dies (bad conversion) on (2048,1,1).",
        "Do not paste these percentages into the README tok/s line.",
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
                f"{_fmt(row.get('dram_pct')):>6}  {_fmt(row.get('tensor_pct')):>7}  "
                f"{_fmt(row.get('occupancy_pct')):>4}  {_fmt(row.get('registers'), 0):>4}  "
                f"{_fmt(row.get('block'), 0):>5}  {_fmt(row.get('grid'), 0):>4}"
            )
        lines.append("")
        lines.append("These are ncu counters on isolated GEMV, not live tok/s.")
    return "\n".join(lines) + "\n"


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    ap.add_argument("--python", dest="python_exe", default=sys.executable)
    ap.add_argument("--l2-bytes", type=int, default=24 << 20)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args(argv)

    ncu = find_ncu()
    out_dir = Path(args.out)
    print("Nsight driver for chr_nf4_gemv / gemv_swiglu. Isolated 3B shapes.")
    print("Pinned: o_proj, down_proj, swiglu. Not lm_head (152k CTAs).")
    print()
    for kind, _regex in CASES:
        shape = _shape_for(kind)
        nr = 4 if shape.M >= 4096 else 1
        grid = (shape.M + nr - 1) // nr
        print(
            f"{kind} M={shape.M} K={shape.K} nr={nr} grid={grid} "
            f"block=128 kernel={_regex}"
        )

    if ncu is None:
        print("\nSKIP: ncu not installed. No empty ncu-gemv files written.")
        return 0
    print(f"\nncu: {ncu}")

    if args.export and not args.profile:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _export_all(ncu, out_dir)
        (out_dir / "README.txt").write_text(
            _readme(ncu, args.python_exe, args.l2_bytes, args.iters, ["export of existing .ncu-rep"], rows),
            encoding="utf-8",
        )
        return 0 if rows else 1

    commands = [
        ncu_command(
            ncu,
            kind=kind,
            kernel_regex=regex,
            python_exe=args.python_exe,
            out_dir=out_dir,
            l2_bytes=args.l2_bytes,
            iters=args.iters,
        )
        for kind, regex in CASES
    ]
    for cmd in commands:
        print("\n ", " ".join(cmd))

    if args.dry_run or not args.profile:
        if not args.profile:
            print("\nNot profiling (pass --profile). Nothing written under docs/runs/ncu-gemv.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    env_ncu = _ncu_env(ncu)
    notes: list[str] = []
    rc = 0
    for cmd, (kind, _regex) in zip(commands, CASES):
        stem = case_stem(kind)
        print(f"\n--- ncu {stem} ---", flush=True)
        try:
            proc = _run_ncu(cmd, env_ncu)
        except OSError as exc:
            line = f"SKIP: {stem}: could not launch ncu: {exc}"
            print(line)
            notes.append(line)
            rc = 1
            continue
        report = out_dir / f"{stem}.ncu-rep"
        log_out = out_dir / f"{stem}.log"
        text = ""
        if log_out.is_file():
            text = log_out.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            if "ERR_NVGPUCTRPERM" in text:
                line = (
                    f"SKIP: {stem}: ERR_NVGPUCTRPERM "
                    "(NVIDIA GPU Performance Counters not enabled for this user). "
                    "No DRAM/occupancy invented."
                )
            else:
                line = (
                    f"SKIP: {stem}: ncu exit {proc.returncode} "
                    f"or no kernel profiles in log"
                )
            print(line)
            notes.append(line)
            for leftover in (report, log_out):
                leftover.unlink(missing_ok=True)
            rc = 1
            continue
        notes.append(f"OK {stem}")

    rows: list[dict[str, object]] = []
    if rc == 0:
        try:
            rows = _export_all(ncu, out_dir)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"SKIP: export failed: {type(exc).__name__}: {exc}")
            rc = 1

    (out_dir / "README.txt").write_text(
        _readme(ncu, args.python_exe, args.l2_bytes, args.iters, notes, rows),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir / 'README.txt'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
