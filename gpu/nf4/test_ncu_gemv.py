"""GPU-free tests for the GEMV Nsight driver. Does not run ncu.

    python -m gpu.nf4.test_ncu_gemv
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4 import ncu_gemv as ncu_mod  # noqa: E402

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        _FAILS.append(msg)


def main() -> int:
    kinds = [k for k, _ in ncu_mod.CASES]
    check(kinds == ["o_proj", "down_proj", "swiglu"], f"kinds {kinds}")
    ncu = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.bat")
    cmd = ncu_mod.ncu_command(
        ncu,
        kind="down_proj",
        kernel_regex="regex:gemv_splitk",
        python_exe=r"C:\python.exe",
        out_dir=_REPO / "docs" / "runs" / "ncu-gemv",
        l2_bytes=24 << 20,
        iters=8,
    )
    joined = " ".join(cmd)
    check("--kernel-name-base demangled" in joined, "match demangled gemv_splitk")
    check("launch__grid_size" not in joined, "grid size skipped (ncu bad conversion)")
    check("regex:gemv_splitk" in joined, "down kernel regex is gemv_splitk")
    check("-m gpu.nf4.bench_gemv_ncu" in joined, "workload is bench_gemv_ncu")
    check("--kind down_proj" in joined, "one shape per ncu process")
    check("ncu-gemv" in joined, "reports stay out of docs/runs/ncu GEMM plate")
    check("chr_nf4_gemm" not in joined, "does not profile GEMM")
    sw = ncu_mod.ncu_command(
        ncu,
        kind="swiglu",
        kernel_regex="regex:gemv_swiglu",
        python_exe=r"C:\python.exe",
        out_dir=_REPO / "docs" / "runs" / "ncu-gemv",
        l2_bytes=24 << 20,
        iters=8,
    )
    check("regex:gemv_swiglu" in " ".join(sw), "swiglu kernel regex")
    check(ncu_mod.case_stem("down_proj") == "qwen25-3b-down_proj-gemv", "stem")
    check(ncu_mod._grid_from_size_column("(2048, 1, 1)") == 2048.0, "grid column product")
    check(ncu_mod._grid_from_size_column("") is None, "empty grid")
    check("launch__grid_size" not in ncu_mod._IMPORT_METRICS, "import metrics omit grid")
    check("sm__warps_active.avg.pct_of_peak_sustained_active" in ncu_mod._IMPORT_METRICS, "import occupancy")
    check(ncu_mod._RENAMES.is_file(), "kernel rename yaml")
    if _FAILS:
        print(f"\n{len(_FAILS)} FAIL")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
