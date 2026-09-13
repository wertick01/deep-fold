"""GPU-free tests for the Nsight driver. Does not run ncu.

    python -m gpu.nf4.test_ncu
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4 import ncu as ncu_mod  # noqa: E402

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        _FAILS.append(msg)


def main() -> int:
    check(len(ncu_mod.CASES) == 4, "four pinned cases")
    check(("q_proj", 1) in ncu_mod.CASES and ("q_proj", 16) in ncu_mod.CASES, "q_proj N=1 and N=16")
    check(("k_proj", 1) in ncu_mod.CASES and ("k_proj", 16) in ncu_mod.CASES, "k_proj N=1 and N=16")
    check(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed" in ncu_mod.METRICS,
        "DRAM throughput metric listed",
    )
    check(
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed" in ncu_mod.METRICS,
        "tensor-pipe metric listed",
    )
    ncu = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.bat")
    cmd = ncu_mod.ncu_command(
        ncu,
        name="q_proj",
        n=1,
        python_exe=r"C:\python.exe",
        out_dir=_REPO / "docs" / "runs" / "ncu",
        l2_bytes=24 << 20,
        iters=8,
    )
    joined = " ".join(cmd)
    check("regex:chr_nf4_gemm" in joined, "kernel regex is chr_nf4_gemm")
    check("-m gpu.nf4.bench" in joined, "workload is bench, not TokenLoop")
    check(".log" in joined and "--log-file" in joined, "ncu log is .log, not the metric table")
    check("--n 1" in joined, "N=1 is one isolated invocation")
    check("--only q_proj" in joined, "one shape per ncu process")
    check("qwen25-3b-q_proj-n1" in joined, "report stem matches TZ names")
    found = ncu_mod.find_ncu()
    if found is None:
        print("SKIP find_ncu: ncu not installed on this box")
    else:
        check(found.is_file(), f"find_ncu -> {found}")

    snippet = '''"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","chr_nf4_gemm_decode_small()","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","9,00"
"0","chr_nf4_gemm_decode_small()","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed","%","1,00"
"0","chr_nf4_gemm_decode_small()","sm__warps_active.avg.pct_of_peak_sustained_active","%","10,00"
"1","chr_nf4_gemm_decode_small()","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","5,40"
"1","chr_nf4_gemm_decode_small()","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed","%","1,38"
"1","chr_nf4_gemm_decode_small()","sm__warps_active.avg.pct_of_peak_sustained_active","%","15,72"
"2","chr_nf4_gemm_decode_small()","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","5,60"
"2","chr_nf4_gemm_decode_small()","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed","%","1,38"
"2","chr_nf4_gemm_decode_small()","sm__warps_active.avg.pct_of_peak_sustained_active","%","15,74"
'''
    parsed = ncu_mod.parse_details_csv(snippet)
    summary = ncu_mod.summarize_launches(parsed)
    check(len(parsed) == 9, f"parsed {len(parsed)} metric rows")
    check(summary["n_launches"] == 2, "warmup ID 0 is dropped")
    check(summary["dram_pct"] == 5.5, f"median DRAM {summary['dram_pct']}")
    check(summary["kernel"] == "chr_nf4_gemm_decode_small", f"kernel {summary['kernel']}")

    if _FAILS:
        print(f"\n{len(_FAILS)} FAIL")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
