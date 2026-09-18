"""Neighbor protocol for GeForce RTX 50 / SM120. First SKU: RTX 5070 Ti.

    python -m gpu.lab.sm120_remote --out docs/runs/sm120-5070ti

Does not download a model. Without a generate-capable GPU this is a skip,
not a pass. 3080 tok/s are not copied onto this plate.

5070 Ti: 70 SMs (same occupancy freeze as the 3080), 16 GB — Qwen2.5-3B/14B
NF4 fit; 32B NF4 still overflow (H2). Native sm_120 cubin needs CUDA 12.8+
and a cu128 torch wheel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from gpu.ampere_gencode import kernel_gencode
from gpu.arch_family import (
    REMOTE_SM120_SMS,
    REMOTE_SM120_SKU,
    REMOTE_SM120_VRAM_MIB,
    SM120_CAPABILITIES,
    family_of,
)
from gpu.cli.doctor import Machine, Verdict, checks, exit_code, probe, render, verdict
from gpu.cli.paths import REPO
from gpu.cuda_env import nvcc_supports_sm120, nvcc_version


def _write(out: Path, name: str, text: str) -> None:
    dest = out / name
    dest.write_text(text, encoding="utf-8")
    print(dest)


def _nvidia_smi_dump() -> str:
    from gpu.cli.smi import executable

    exe = executable()
    if exe is None:
        return "nvidia-smi not found\n"
    chunks: list[str] = []
    for args in (
        [exe, "-L"],
        [
            exe,
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv",
        ],
    ):
        try:
            run = subprocess.run(
                args, capture_output=True, text=True, timeout=20, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            chunks.append(f"{' '.join(args)}: {exc}\n")
            continue
        chunks.append(run.stdout or run.stderr or f"exit {run.returncode}\n")
    return "".join(chunks) if chunks else "nvidia-smi produced no output\n"


def checklist_text(m: Machine, v: Verdict, *, code: int) -> str:
    cubin = "yes" if nvcc_supports_sm120() else "no (PTX compute_80 JIT)"
    ver = nvcc_version()
    nvcc = f"{ver[0]}.{ver[1]}" if ver else "missing"
    return "\n".join(
        [
            f"SM120 remote checklist  SKU hint: {REMOTE_SM120_SKU}",
            f"This GPU: {m.device_name or '?'}  sm={m.sm}  "
            f"SMs={m.sm_count}  VRAM={m.vram_total_mib} MiB",
            f"doctor: {v.line}  (exit {code})",
            f"nvcc {nvcc}  native sm_120 cubin: {cubin}",
            f"kernel gencode: {kernel_gencode()}",
            "",
            "Occupancy: 5070 Ti is 70 SMs — same CTA freeze as the 3080 plate.",
            f"VRAM: {REMOTE_SM120_VRAM_MIB} MiB public. 3B/14B NF4 resident; "
            "32B NF4 is still H2 overflow (packed ~16.6 GB).",
            "Do not quote RTX 3080 tok/s on this plate.",
            "",
            "Next (this box, after doctor is experimental / exit 0):",
            "  1. python -m gpu.nf4.verify",
            "  2. deepfold run Qwen2.5-3B-Instruct --max-new-tokens 32",
            "  3. ncu DRAM % / SM busy if ncu exists",
            "  4. one generate with CUDA_FORCE_PTX_JIT=1 vs native cubin",
            "",
        ]
    )


def write_bundle(
    out: Path,
    m: Machine,
    v: Verdict,
    report: list,
    code: int,
    *,
    smi_text: str | None = None,
) -> None:
    """CPU-testable artifact writer. ``probe`` stays in :func:`main`."""
    out.mkdir(parents=True, exist_ok=True)
    caps = {
        "sku_hint": REMOTE_SM120_SKU,
        "sku_sms": REMOTE_SM120_SMS,
        "sku_vram_mib": REMOTE_SM120_VRAM_MIB,
        "device_name": m.device_name,
        "capability": list(m.capability) if m.capability else None,
        "family": family_of(m.capability),
        "sm_count": m.sm_count,
        "vram_total_mib": m.vram_total_mib,
        "torch": m.torch,
        "torch_cuda": m.torch_cuda,
        "generate": v.generate,
        "arch": v.arch,
        "line": v.line,
        "doctor_exit": code,
        "kernel_gencode": kernel_gencode(),
        "nvcc_supports_sm120": nvcc_supports_sm120(),
        "ncu": shutil.which("ncu"),
    }
    text = render(m, v, report)
    _write(out, "doctor.txt", text + "\n")
    _write(out, "caps.json", json.dumps(caps, indent=2) + "\n")
    _write(out, "nvidia-smi.txt", smi_text if smi_text is not None else _nvidia_smi_dump())
    _write(out, "CHECKLIST.txt", checklist_text(m, v, code=code))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    source = [
        f"SM120 remote lab  {stamp}",
        f"Expected first SKU: {REMOTE_SM120_SKU} "
        f"({REMOTE_SM120_SMS} SMs, {REMOTE_SM120_VRAM_MIB // 1024} GB, sm_120).",
        "Do not quote RTX 3080 tok/s. Native sm_120 cubin needs CUDA 12.8+;",
        "otherwise PTX compute_80 JIT (first generate may take ~1 min).",
        "Torch wheel: cu128. Occupancy matches 3080 (70 SM). 32B still overflows.",
        "",
        v.line,
        f"doctor exit {code}",
        "",
    ]
    if m.capability and tuple(m.capability) not in SM120_CAPABILITIES:
        source.append(
            f"NOTE: this GPU is {m.sm} ({m.device_name}), not SM120. "
            "Artifacts still written; generate allowance follows doctor."
        )
        source.append("")
    _write(out, "SOURCE.txt", "\n".join(source))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for doctor dump + SOURCE.txt (default: timestamp under docs/runs)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="also run python -m gpu.nf4.verify (needs a generate-capable GPU)",
    )
    args = p.parse_args(argv)

    m = probe()
    v = verdict(m)
    report = checks(m, v)
    text = render(m, v, report)
    code = exit_code(m, v, report)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or (REPO / "docs" / "runs" / f"sm120-{stamp}")
    write_bundle(out, m, v, report, code)
    print(text)

    if args.verify:
        if not v.allowed:
            print("SKIP: --verify needs generate-capable GPU", file=sys.stderr)
            return 0
        verify = subprocess.call(
            [sys.executable, "-m", "gpu.nf4.verify"], cwd=str(REPO)
        )
        _write(out, "verify.exit", f"{verify}\n")
        if verify != 0:
            print(f"SKIP: gpu.nf4.verify exit {verify}", file=sys.stderr)
            return 0

    if not v.allowed:
        print("SKIP: generate not allowed on this machine", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
