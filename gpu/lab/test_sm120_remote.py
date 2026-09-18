"""CPU tests for the SM120 neighbor bundle. No generate, no Hub.

    python -m gpu.lab.test_sm120_remote
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def test_bundle_names_5070_ti() -> None:
    from gpu.cli.doctor import Machine, checks, exit_code, verdict
    from gpu.lab.sm120_remote import write_bundle

    m = Machine(
        system="Windows",
        python=(3, 12, 0),
        torch="2.7.0+cu128",
        torch_cuda="12.8",
        cuda_available=True,
        capability=(12, 0),
        device_name="NVIDIA GeForce RTX 5070 Ti",
        vram_total_mib=16384,
        sm_count=70,
        chr_bin=Path("chr.exe"),
        chr_runs=True,
        transformers=True,
        safetensors=True,
        nf4_ext=Path("gpu/nf4/chr_nf4_ext.pyd"),
        nf4_ext_abi_ok=True,
    )
    v = verdict(m)
    report = checks(m, v)
    code = exit_code(m, v, report)
    with tempfile.TemporaryDirectory() as raw:
        out = Path(raw)
        write_bundle(out, m, v, report, code, smi_text="NVIDIA GeForce RTX 5070 Ti\n")
        source = (out / "SOURCE.txt").read_text(encoding="utf-8")
        checklist = (out / "CHECKLIST.txt").read_text(encoding="utf-8")
        caps = json.loads((out / "caps.json").read_text(encoding="utf-8"))
        smi = (out / "nvidia-smi.txt").read_text(encoding="utf-8")
        check("doctor.txt exists", (out / "doctor.txt").is_file())
        check("SOURCE names 5070 Ti", "5070 Ti" in source, source.splitlines()[1])
        check("SOURCE says 70 SMs", "70 SMs" in source)
        check("SOURCE says 16 GB", "16 GB" in source)
        check("SOURCE does not quote 3080 tok/s", "Do not quote RTX 3080 tok/s" in source)
        check("checklist 32B overflow", "32B" in checklist and "overflow" in checklist)
        check("checklist 3B run", "Qwen2.5-3B-Instruct" in checklist)
        check("caps sku", caps["sku_hint"] == "RTX 5070 Ti")
        check("caps 70 SM", caps["sm_count"] == 70)
        check("caps 16 GB", caps["vram_total_mib"] == 16384)
        check("smi dump used", "5070 Ti" in smi)
        check("experimental generate", v.generate == "experimental" and v.allowed)
        check("doctor exit 0 on healthy fake", code == 0, str(code))


def test_bundle_on_3080_is_not_a_green_sm120_plate() -> None:
    from gpu.cli.doctor import Machine, checks, exit_code, verdict
    from gpu.lab.sm120_remote import write_bundle

    m = Machine(
        system="Windows",
        python=(3, 11, 0),
        torch="2.5.1+cu124",
        torch_cuda="12.4",
        cuda_available=True,
        capability=(8, 6),
        device_name="NVIDIA GeForce RTX 3080",
        vram_total_mib=12288,
        sm_count=70,
        chr_bin=Path("chr.exe"),
        chr_runs=True,
        transformers=True,
        safetensors=True,
        nf4_ext=Path("gpu/nf4/chr_nf4_ext.pyd"),
        nf4_ext_abi_ok=True,
    )
    v = verdict(m)
    report = checks(m, v)
    code = exit_code(m, v, report)
    with tempfile.TemporaryDirectory() as raw:
        out = Path(raw)
        write_bundle(out, m, v, report, code, smi_text="RTX 3080\n")
        source = (out / "SOURCE.txt").read_text(encoding="utf-8")
        check("3080 note is not SM120", "not SM120" in source)
        check("3080 still ship generate", v.arch == "ship" and code == 0, v.arch)


def main() -> int:
    print("gpu.lab.sm120_remote, no GPU\n")
    test_bundle_names_5070_ti()
    test_bundle_on_3080_is_not_a_green_sm120_plate()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
