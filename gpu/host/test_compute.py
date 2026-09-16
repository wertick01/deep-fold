"""CPU: ComputePlan defaults, 32B sizes, CLI-equivalent flag math. No GPU, no .chr.

    python gpu/host/test_compute.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.compute import (  # noqa: E402
    DEFAULT_CPU_SUFFIX_GPU_LAYERS,
    DEFAULT_HYBRID_GPU_LAYERS,
    SLACK_MIB,
    VRAM_MIB_3080,
    ComputePlanError,
    format_compute_stderr,
    kv_mib_per_layer,
    net_fits_resident,
    plan_compute,
    prefix_budget_mib,
)
from gpu.host.residency import MIB, descs_from_qwen  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


QWEN_32B = dict(
    hidden_size=5120,
    intermediate_size=27648,
    num_hidden_layers=64,
    num_attention_heads=40,
    num_key_value_heads=8,
    vocab_size=152064,
    tie_word_embeddings=False,
    model_type="qwen2",
)
QWEN_3B = dict(
    hidden_size=2048,
    intermediate_size=11008,
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=2,
    vocab_size=151936,
    tie_word_embeddings=True,
    model_type="qwen2",
)


def test_32b_layer_and_table() -> None:
    descs = descs_from_qwen(QWEN_32B)
    layer = sum(d.nbytes for d in descs if d.layer == 0) / MIB
    embed = next(d.nbytes for d in descs if d.kind == "embed") / MIB
    head = next(d.nbytes for d in descs if d.kind == "lm_head") / MIB
    check("32B layer 247.03125 MiB", layer == 247.03125, f"{layer}")
    check("embed 394.453125", embed == 394.453125, f"{embed}")
    check("lm_head 394.453125", head == 394.453125, f"{head}")
    kv = kv_mib_per_layer(QWEN_32B, 2048)
    check("KV 8 MiB / layer @2048", kv == 8.0, f"{kv}")
    b32 = prefix_budget_mib(descs, QWEN_32B, 32, max_seq=2048, kv="split")
    check("N=32 split used ~8949.9", abs(b32["used_mib"] - 8949.9) < 0.1, f"{b32['used_mib']}")
    b36 = prefix_budget_mib(descs, QWEN_32B, 36, max_seq=2048, kv="split")
    check("N=36 split used ~9970.0", abs(b36["used_mib"] - 9970.0) < 0.1, f"{b36['used_mib']}")
    check(
        "N=36 used+1800 ~11770",
        abs(b36["used_plus_overhead_mib"] - 11770) < 0.1,
        f"{b36['used_plus_overhead_mib']}",
    )
    b38 = prefix_budget_mib(descs, QWEN_32B, 38, max_seq=2048, kv="split")
    check(
        "N=38 split used+1800 ~12280",
        abs(b38["used_plus_overhead_mib"] - 12280) < 0.1,
        f"{b38['used_plus_overhead_mib']}",
    )
    check(
        "N=38 + slack over 12 GB",
        b38["used_plus_overhead_mib"] + SLACK_MIB > VRAM_MIB_3080,
        f"{b38['used_plus_overhead_mib'] + SLACK_MIB}",
    )


def test_cpu_suffix_default_32() -> None:
    descs = descs_from_qwen(QWEN_32B)
    plan = plan_compute(descs, QWEN_32B, compute="cpu-suffix", max_seq=2048)
    check("cpu-suffix n_gpu 32", plan.n_gpu == DEFAULT_CPU_SUFFIX_GPU_LAYERS, f"{plan.n_gpu}")
    check("cpu-suffix n_cpu 32", plan.n_cpu == 32, f"{plan.n_cpu}")
    check("cpu-suffix kv split", plan.kv == "split", plan.kv)
    check("cpu-suffix ring none", plan.ring == "none", plan.ring)
    check("lm_head device", plan.lm_head == "device", plan.lm_head)
    check("embed device", plan.embed == "device", plan.embed)
    check("cpu ids start at 32", plan.cpu_layer_ids[0] == 32, str(plan.cpu_layer_ids[:3]))


def test_hybrid_default_36() -> None:
    descs = descs_from_qwen(QWEN_32B)
    plan = plan_compute(descs, QWEN_32B, compute="hybrid", max_seq=2048)
    check("hybrid n_gpu 36", plan.n_gpu == DEFAULT_HYBRID_GPU_LAYERS, f"{plan.n_gpu}")
    check("hybrid n_cpu 28", plan.n_cpu == 28, f"{plan.n_cpu}")
    check("hybrid ring none", plan.ring == "none", plan.ring)
    check("hybrid kv split", plan.kv == "split", plan.kv)
    check("kv gpu 288", abs(plan.kv_gpu_mib - 288) < 0.1, f"{plan.kv_gpu_mib}")
    check("kv cpu 224", abs(plan.kv_cpu_mib - 224) < 0.1, f"{plan.kv_cpu_mib}")
    dump = format_compute_stderr(plan)
    check("stderr compute=hybrid", "compute=hybrid gpu_layers=36/64 cpu_layers=28" in dump, dump)
    check("stderr ring=off", "ring=off" in dump, dump)
    check("stderr pageable", "pageable" in dump, dump)


def test_gpu_32b_policy_d_tape() -> None:
    descs = descs_from_qwen(QWEN_32B)
    plan = plan_compute(descs, QWEN_32B, compute="gpu", max_seq=2048)
    check("gpu n_cpu 0", plan.n_cpu == 0, f"{plan.n_cpu}")
    check("gpu n_gpu 64", plan.n_gpu == 64, f"{plan.n_gpu}")
    check("gpu kv gpu-all", plan.kv == "gpu-all", plan.kv)
    check("gpu ring D", plan.ring == "D", plan.ring)
    check("gpu tape 96", plan.ring_matrices == 96, f"{plan.ring_matrices}")
    check(
        "gpu streamed ~6885",
        abs(plan.ring_streamed_mib - 6885.0) < 0.1,
        f"{plan.ring_streamed_mib}",
    )


def test_gpu_layers_64_with_compute_gpu_is_noop() -> None:
    descs = descs_from_qwen(QWEN_32B)
    plan = plan_compute(descs, QWEN_32B, compute="gpu", gpu_layers=64)
    check("noop n_cpu 0", plan.n_cpu == 0 and plan.compute == "gpu", str(plan.compute))


def test_n38_split_slack_refuses() -> None:
    descs = descs_from_qwen(QWEN_32B)
    try:
        plan_compute(descs, QWEN_32B, compute="hybrid", gpu_layers=38)
    except ComputePlanError as exc:
        check("N=38 raises", "38" in str(exc) or "fit" in str(exc).lower(), str(exc)[:160])
        return
    check("N=38 raises", False, "no error")


def test_gpu_plus_cpu_layers_sum() -> None:
    descs = descs_from_qwen(QWEN_32B)
    plan = plan_compute(
        descs, QWEN_32B, compute="hybrid", gpu_layers=36, cpu_layers=28
    )
    check("sum 36+28", plan.n_gpu == 36 and plan.n_cpu == 28, f"{plan.n_gpu}+{plan.n_cpu}")
    try:
        plan_compute(descs, QWEN_32B, compute="hybrid", gpu_layers=32, cpu_layers=28)
    except ComputePlanError as exc:
        check("bad sum raises", "sum" in str(exc), str(exc)[:160])
    else:
        check("bad sum raises", False, "no error")


def test_gpu_frac_floor() -> None:
    descs = descs_from_qwen(QWEN_32B)
    plan = plan_compute(descs, QWEN_32B, compute="hybrid", gpu_frac=36 / 64)
    check("frac 36/64 -> 36", plan.n_gpu == 36, f"{plan.n_gpu}")
    plan32 = plan_compute(descs, QWEN_32B, compute="cpu-suffix", gpu_frac=0.5)
    check("frac 0.5 -> 32", plan32.n_gpu == 32, f"{plan32.n_gpu}")
    check("floor 0.51*64", math.floor(0.51 * 64) == 32, "")


def test_3b_refuse_without_explicit() -> None:
    descs = descs_from_qwen(QWEN_3B)
    check("3B fits resident", net_fits_resident(descs, QWEN_3B), "")
    try:
        plan_compute(descs, QWEN_3B, compute="hybrid")
    except ComputePlanError as exc:
        check("3B hybrid refuse", "explicit" in str(exc) or "3B" in str(exc), str(exc)[:160])
    else:
        check("3B hybrid refuse", False, "no error")
    try:
        plan_compute(descs, QWEN_3B, compute="cpu-suffix")
    except ComputePlanError as exc:
        check("3B cpu-suffix refuse", "explicit" in str(exc), str(exc)[:160])
    else:
        check("3B cpu-suffix refuse", False, "no error")
    canary = plan_compute(descs, QWEN_3B, compute="hybrid", gpu_layers=8)
    check("3B canary n_gpu 8", canary.n_gpu == 8 and canary.n_cpu == 28, f"{canary.n_gpu}/{canary.n_cpu}")
    check("3B canary no ring", canary.ring == "none", canary.ring)


def test_flags_require_split_mode() -> None:
    descs = descs_from_qwen(QWEN_32B)
    try:
        plan_compute(descs, QWEN_32B, compute="gpu", gpu_layers=36)
    except ComputePlanError as exc:
        check("gpu + gpu-layers 36 raises", "cpu-suffix" in str(exc), str(exc)[:160])
    else:
        check("gpu + gpu-layers 36 raises", False, "no error")


def test_suffix_contiguous_tail() -> None:
    descs = descs_from_qwen(QWEN_32B)
    plan = plan_compute(descs, QWEN_32B, compute="hybrid")
    check("gpu ids 0..35", plan.gpu_layer_ids == tuple(range(36)), str(plan.gpu_layer_ids[-1:]))
    check(
        "cpu ids 36..63",
        plan.cpu_layer_ids == tuple(range(36, 64)),
        str(plan.cpu_layer_ids[:2]),
    )


TESTS = [
    test_32b_layer_and_table,
    test_cpu_suffix_default_32,
    test_hybrid_default_36,
    test_gpu_32b_policy_d_tape,
    test_gpu_layers_64_with_compute_gpu_is_noop,
    test_n38_split_slack_refuses,
    test_gpu_plus_cpu_layers_sum,
    test_gpu_frac_floor,
    test_3b_refuse_without_explicit,
    test_flags_require_split_mode,
    test_suffix_contiguous_tail,
]


def main() -> int:
    print("gpu/host ComputePlan, CPU only\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
