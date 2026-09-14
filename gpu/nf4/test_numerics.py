"""CPU-only NF4 numerics: quantization vs kernel, no 3B, no JIT.

    python -m gpu.nf4.test_numerics
    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe -m gpu.nf4.test_numerics
"""

from __future__ import annotations

import ast
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4 import numerics as num  # noqa: E402
from gpu.tests.nf4_oracle import (  # noqa: E402
    GOLDEN2X64_METRICS,
    GOLDEN2X64_ROW1_HEX,
    GOLDEN64_HEX,
    decode_nf4,
    encode_nf4,
    matmul_f32,
)

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> bool:
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        _FAILS.append(msg)
    return cond


def test_catalog_shapes() -> None:
    names = [n for n, _, _ in num.QWEN25_3B_LINEARS]
    check(
        names == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        f"seven Linear kinds in catalog order: {names}",
    )
    want = {
        "q_proj": (2048, 2048),
        "k_proj": (256, 2048),
        "v_proj": (256, 2048),
        "o_proj": (2048, 2048),
        "gate_proj": (11008, 2048),
        "up_proj": (11008, 2048),
        "down_proj": (2048, 11008),
    }
    got = {n: (m, k) for n, m, k in num.QWEN25_3B_LINEARS}
    check(got == want, "Qwen2.5-3B GQA / SwiGLU shapes match gpu.nf4.bench")
    check("lm_head" not in names, "lm_head is not a per-layer Linear kind here")


def test_golden_ss94_metrics() -> None:
    """docs/spec/nf4.md §9.4: our five-metric helper agrees on the L2 trio."""
    i = np.arange(64, dtype=np.float32)
    w = np.stack(
        [i / np.float32(63.0), (np.float32(2.0) * i - np.float32(63.0)) / np.float32(63.0)]
    )
    packed, scale = encode_nf4(w)
    hat = decode_nf4(packed, scale, 2, 64)
    m = num.error_metrics(hat, w)
    check(
        m.max_abs <= GOLDEN2X64_METRICS["maxabs"][1]
        and m.rmse <= GOLDEN2X64_METRICS["rmse"][1]
        and m.mean_abs <= GOLDEN2X64_METRICS["mae"][1],
        f"SS9.4 quant/W max_abs={m.max_abs:.10f} rmse={m.rmse:.10f} "
        f"mean_abs={m.mean_abs:.10f} cosine={m.cosine:.6f} rel={m.rel:.6f}",
    )
    check(m.cosine > 0.99, f"SS9.4 cosine {m.cosine:.6f} is a reconstruction, not a kernel")
    packed_hex = packed.tobytes().hex().upper()
    check(
        packed_hex[:64] == GOLDEN64_HEX and packed[1].tobytes().hex().upper() == GOLDEN2X64_ROW1_HEX,
        "SS9.4 packed hex still matches the oracle (encoder not the kernel)",
    )


def test_zero_is_exact() -> None:
    w = np.zeros((4, 64), dtype=np.float32)
    packed, scale = encode_nf4(w)
    hat = decode_nf4(packed, scale, 4, 64)
    m = num.error_metrics(hat, w)
    check(
        m.max_abs == 0.0 and m.rmse == 0.0 and m.cosine == 1.0 and m.rel == 0.0,
        f"zero W reconstructs exactly: {m}",
    )


def test_chunked_encode_matches() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((70, 96), dtype=np.float32).astype(np.float32) * np.float32(0.02)
    a, sa = encode_nf4(w)
    b, sb = num.encode_nf4_chunked(w, row_chunk=17)
    check(np.array_equal(a, b) and np.array_equal(sa, sb), "chunked encode == encode_nf4")


def test_legs_are_not_conflated() -> None:
    """A fake kernel that equals the CPU GEMM must not inherit quant error."""
    w = num.random_bf16_w(32, 64, seed=7)
    packed, scale = encode_nf4(w)
    x = num.rms_norm_x(64, 1, seed=11)
    w_hat = decode_nf4(packed, scale, 32, 64)
    y_cpu = matmul_f32(w_hat, x)

    def exact_kernel(packed_t, scale_t, x_t, m, k, kp):  # noqa: ANN001, ARG001
        packed_np = packed_t.detach().cpu().numpy()
        scale_np = np.asarray(scale_t.detach().cpu().numpy(), dtype=np.float16)
        x_np = x_t.float().detach().cpu().numpy().astype(np.float32)
        return matmul_f32(decode_nf4(packed_np, scale_np, m, k), x_np)

    rows = num.run_kind(
        "q_proj",
        w,
        packed,
        scale,
        x,
        source="random",
        gemm=exact_kernel,
        kernel_reason=None,
    )
    by = {(r.leg, r.surface): r for r in rows}
    q_w = by[(num.LEG_QUANT, "W")].metrics
    q_y = by[(num.LEG_QUANT, "y")].metrics
    k_y = by[(num.LEG_KERNEL, "y")].metrics
    check(q_w is not None and q_w.max_abs > 0.0, "quant/W is the reconstruction error")
    check(q_y is not None and q_y.max_abs > 0.0, "quant/y is BF16 GEMM vs CPU NF4 GEMM")
    check(
        k_y is not None and k_y.max_abs == 0.0 and k_y.rel == 0.0 and k_y.cosine == 1.0,
        f"kernel/y vs exact oracle is 0 (got {k_y}), not the quantization number",
    )
    check(
        k_y is not None and q_y is not None and k_y.max_abs < q_y.max_abs,
        "kernel max_abs is not the mixed CUDA-vs-BF16 figure",
    )
    noisy = y_cpu + np.float32(0.01)

    def offset_kernel(*_a, **_k):  # noqa: ANN001
        return noisy

    rows2 = num.run_kind(
        "q_proj", w, packed, scale, x, source="random", gemm=offset_kernel
    )
    k2 = next(r for r in rows2 if r.leg == num.LEG_KERNEL).metrics
    q2 = next(r for r in rows2 if r.leg == num.LEG_QUANT and r.surface == "y").metrics
    check(k2 is not None and abs(k2.max_abs - 0.01) < 1e-5, f"injected kernel error is 0.01, got {k2}")
    check(q2 is not None and abs(q2.max_abs - q_y.max_abs) < 1e-7, "quant/y unchanged by a bad kernel")


def test_random_all_kinds_cpu() -> None:
    """Realistic K, capped M: no GPU, no 3B file."""
    rows = num.run_random(
        num.QWEN25_3B_LINEARS,
        ns=(1,),
        seed=3,
        max_rows=32,
        row_chunk=16,
        kernel_reason="cpu test: kernel not launched",
    )
    kinds = {r.kind for r in rows if r.leg == num.LEG_QUANT and r.status == "ok"}
    check(kinds == {n for n, _, _ in num.QWEN25_3B_LINEARS}, f"quant rows for every kind: {sorted(kinds)}")
    kernel_skips = [r for r in rows if r.leg == num.LEG_KERNEL]
    check(
        kernel_skips and all(r.status == "skip" for r in kernel_skips),
        f"{len(kernel_skips)} kernel rows skipped on the CPU path",
    )
    for r in rows:
        if r.leg == num.LEG_QUANT and r.status == "ok":
            m = r.metrics
            check(
                m is not None and np.isfinite([m.max_abs, m.mean_abs, m.rmse, m.cosine, m.rel]).all(),
                f"{r.kind} {r.surface} metrics finite",
            )
            if r.surface == "W":
                check(m.cosine > 0.9, f"{r.kind} W cosine {m.cosine:.4f}")


def test_n16_cpu_gemm() -> None:
    rows = num.run_random(
        (("k_proj", 256, 2048),),
        ns=(16,),
        seed=5,
        max_rows=32,
        kernel_reason="cpu test",
    )
    y = next(r for r in rows if r.leg == num.LEG_QUANT and r.surface == "y")
    check(y.N == 16 and y.status == "ok", f"N=16 CPU GEMM y row {y}")
    check(y.metrics is not None and y.metrics.cosine > 0.9, f"N=16 quant/y cosine {y.metrics}")


def test_kernel_floor_bf16_ulp() -> None:
    """The 2026-09-14 q_proj spike is BF16 store rounding, not a kernel miss."""
    got = np.array([[-17.5]], dtype=np.float32)
    ref = np.array([[-17.441530227661133]], dtype=np.float32)
    check(
        num.kernel_floor_ok(got, ref),
        "0.058 at |y|~17 is half BF16 ULP (allow 0.0625)",
    )
    check(
        not num.kernel_floor_ok(
            np.array([[0.06]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
        ),
        "0.06 vs 0 still fails the O(1) 0.05 floor",
    )
    check(
        num.kernel_floor_ok(
            np.array([[0.04]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
        ),
        "0.04 vs 0 passes 0.05",
    )


def test_n_refused() -> None:
    try:
        num.rms_norm_x(64, 33, seed=1)
    except ValueError:
        check(True, "N=33 refused (TokenLoop ceiling stays 32)")
    else:
        check(False, "N=33 should be refused")
    try:
        num._parse_ns("33")
    except ValueError:
        check(True, "CLI N=33 refused without --plan-n")
    else:
        check(False, "CLI N=33 should be refused")
    got = num._parse_ns("17,32,64", max_n=num.PLAN_MAX_N)
    check(got == [17, 32, 64], f"--plan-n parses 17,32,64: {got}")


def test_no_eager_kernel_import() -> None:
    check("chr_nf4_ext" not in sys.modules, "chr_nf4_ext was not loaded by importing numerics")
    src = Path(__file__).resolve().parent / "numerics.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    bad = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            if "nf4_gemm" in names or node.module in {"gpu.nf4", "nf4"}:
                bad.append(f"{node.module}:{names}")
    check(not bad, f"no module-level gpu.nf4 / nf4_gemm import (lazy kernel only); {bad}")


def test_cuda_default_is_skip() -> None:
    reason = num.cuda_launch_reason(want_cuda=False, force=False)
    check(reason is not None and "--cuda" in reason, f"default refuses CUDA: {reason}")
    busy = num.gpu_busy_reason()
    print(f"  nvidia-smi: {busy or 'looks free (still not launching)'}")
    src = (Path(__file__).resolve().parent / "numerics.py").read_text(encoding="utf-8")
    check("torch.cuda.is_available" not in src, "busy probe does not call torch.cuda.is_available")
    check("python -m gpu.lab.run" not in src, "numerics does not start the 3B lab")


def test_cli_tiny() -> None:
    code = num.main(["--tiny", "--kinds", "k_proj", "--n", "1"])
    check(code == 0, f"python -m gpu.nf4.numerics --tiny --kinds k_proj exited {code}")


def test_safetensors_cpu_roundtrip() -> None:
    from gpu.chr0._fixtures import write_safetensors

    w = np.arange(32, dtype=np.float32).reshape(4, 8) / np.float32(31.0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_safetensors(root / "model.safetensors", {"toy.weight": w})
        got = num.load_hf_weight_f32(root, "toy.weight")
    # F32 file is rounded to BF16 on load (that is the BF16-W contract).
    want = num.to_bf16_f32(w)
    check(got.shape == (4, 8), f"loaded shape {got.shape}")
    check(np.allclose(got, want, atol=0, rtol=0), "F32 shard rounded to BF16 matches to_bf16_f32")


def test_bf16_safetensors_bits() -> None:
    """Widening BF16 is a left shift, not a second round-trip through torch."""
    import torch

    src = torch.tensor([[0.5, -1.0], [0.0, 1.5]], dtype=torch.bfloat16)
    payload = src.view(torch.int16).cpu().numpy().tobytes()
    header = {
        "layer.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(payload)]}
    }
    js = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.safetensors"
        with path.open("wb") as f:
            f.write(struct.pack("<Q", len(js)))
            f.write(js)
            f.write(payload)
        got = num.load_hf_weight_f32(Path(tmp), "layer.weight")
    want = src.float().numpy()
    check(np.allclose(got, want, atol=0, rtol=0), "BF16 shard widens exactly")


def main() -> int:
    print("NF4 numerics: CPU only, no 3B, no chr_nf4_gemm\n")
    for fn in (
        test_catalog_shapes,
        test_golden_ss94_metrics,
        test_zero_is_exact,
        test_chunked_encode_matches,
        test_legs_are_not_conflated,
        test_random_all_kinds_cpu,
        test_n16_cpu_gemm,
        test_kernel_floor_bf16_ulp,
        test_n_refused,
        test_no_eager_kernel_import,
        test_cuda_default_is_skip,
        test_safetensors_cpu_roundtrip,
        test_bf16_safetensors_bits,
        test_cli_tiny,
    ):
        fn()
    print()
    if _FAILS:
        print(f"SOME FAIL ({len(_FAILS)})")
        return 2
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
