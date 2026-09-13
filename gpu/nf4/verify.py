"""Oracle checks for chr_nf4_gemm. Run with torch-gpu python from repo root:

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe gpu/nf4/verify.py
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

GPU_DIR = Path(__file__).resolve().parent.parent
REPO = GPU_DIR.parent
sys.path.insert(0, str(GPU_DIR))
sys.path.insert(0, str(GPU_DIR / "tests"))

from nf4 import nf4_gemm  # noqa: E402
from nf4_oracle import decode_nf4, matmul_f32  # noqa: E402

# docs/spec/nf4.md §1 literals (same decimal strings as the Go codec).
NF4_LITERALS = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
NF4_BITS = [
    0xBF800000,
    0xBF3239B1,
    0xBF066B30,
    0xBECA32A0,
    0xBE91A24D,
    0xBE3D353F,
    0xBDBA7871,
    0x00000000,
    0x3DA2FAFF,
    0x3E24CAE3,
    0x3E7C04DD,
    0x3EAD033A,
    0x3EE1A4B8,
    0x3F1007AB,
    0x3F3913B3,
    0x3F800000,
]
NF4 = np.array(NF4_LITERALS, dtype=np.float32)

CHR_PATH = Path(r"C:\dev\models\qwen25-3b.nf4.chr")
GATE_NAME = "model.layers.0.mlp.gate_proj"
MAXABS_LIMIT = 0.05


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def dequant_nf4(packed: np.ndarray, scale: np.ndarray, M: int, K: int, K_pad: int) -> np.ndarray:
    packed = np.ascontiguousarray(packed)
    lo = packed & 0x0F
    hi = packed >> 4
    idx = np.empty((M, K_pad), dtype=np.uint8)
    idx[:, 0::2] = lo
    idx[:, 1::2] = hi
    idx = idx[:, :K]
    scale_f32 = np.asarray(scale, dtype=np.float16).astype(np.float32)
    g = np.arange(K, dtype=np.int32) // 64
    return NF4[idx] * scale_f32[:, g]


def rms_norm_x(K: int, seed: int, N: int = 1) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(K, N, dtype=torch.bfloat16, device="cuda", generator=g)
    rms = x.float().pow(2).mean().sqrt().clamp_min(1e-6)
    return (x.float() / rms).to(torch.bfloat16)


def stats(y_gpu: torch.Tensor, y_cpu: np.ndarray) -> tuple[float, float]:
    yg = y_gpu.float().detach().cpu().numpy().reshape(-1)
    yc = np.asarray(y_cpu, dtype=np.float32).reshape(-1)
    err = yg - yc
    maxabs = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))
    return maxabs, rmse


def run_case(name: str, packed: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
             M: int, K: int, K_pad: int) -> tuple[bool, float, float]:
    y = nf4_gemm(packed, scale, x, M, K, K_pad)
    torch.cuda.synchronize()
    packed_np = packed.detach().cpu().numpy().reshape(M, K_pad // 2)
    scale_np = np.asarray(scale.detach().cpu().numpy().reshape(M, K_pad // 64), dtype=np.float16)
    W = decode_nf4(packed_np, scale_np, M, K)
    x_f = x.float().detach().cpu().numpy().reshape(K, -1).astype(np.float32)
    y_cpu = matmul_f32(W, x_f)
    maxabs, rmse = stats(y, y_cpu)
    W_bf = torch.from_numpy(np.ascontiguousarray(W)).to(torch.bfloat16).float().numpy()
    maxabs_bf, rmse_bf = stats(y, W_bf @ x_f)
    ok = maxabs <= MAXABS_LIMIT
    status = "PASS" if ok else "FAIL"
    print(
        f"{status} {name}: maxabs={maxabs:.6g} rmse={rmse:.6g} "
        f"(limit {MAXABS_LIMIT}; vs bf16(W) maxabs={maxabs_bf:.6g} rmse={rmse_bf:.6g})"
    )
    return ok, maxabs, rmse


def test_lut_bits() -> bool:
    ok = True
    for i, (lit, bits) in enumerate(zip(NF4_LITERALS, NF4_BITS)):
        got = f32_bits(lit)
        if got != bits:
            print(f"FAIL LUT[{i}] bits {got:#010x} want {bits:#010x}")
            ok = False
    if ok:
        print("PASS LUT literals match nf4.md section 1 bits")
    return ok


def test_toy() -> bool:
    M, K, K_pad = 128, 256, 256
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    packed = torch.randint(0, 256, (M, K_pad // 2), dtype=torch.int32, device="cuda", generator=g)
    packed = packed.to(torch.uint8)
    sg = torch.Generator(device="cuda")
    sg.manual_seed(11)
    scale = (0.04 + 0.20 * torch.rand(M, K_pad // 64, device="cuda", generator=sg)).to(
        torch.float16
    )
    x = rms_norm_x(K, seed=2)
    ok, _, _ = run_case("toy 128x256 N=1", packed, scale, x, M, K, K_pad)
    return ok


def test_tails() -> bool:
    M, K, K_pad = 130, 65, 128
    # Pad columns are nibble 0xF so a K-overread would blow up vs the CPU oracle.
    packed = torch.full((M, K_pad // 2), 0xFF, dtype=torch.uint8, device="cuda")
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    live = torch.randint(0, 256, (M, (K + 1) // 2), dtype=torch.int32, device="cuda", generator=g)
    packed[:, : (K + 1) // 2] = live.to(torch.uint8)
    sg = torch.Generator(device="cuda")
    sg.manual_seed(13)
    scale = (0.04 + 0.20 * torch.rand(M, K_pad // 64, device="cuda", generator=sg)).to(
        torch.float16
    )
    x = rms_norm_x(K, seed=4)
    ok, maxabs, rmse = run_case("tails 130x65 K_pad=128", packed, scale, x, M, K, K_pad)
    y = nf4_gemm(packed, scale, x, M, K, K_pad)
    torch.cuda.synchronize()
    if y.shape != (M, 1):
        print(f"FAIL tails shape {tuple(y.shape)} want {(M, 1)}")
        return False
    print(f"  y shape {tuple(y.shape)} (pad must not extend y); maxabs={maxabs:.6g} rmse={rmse:.6g}")
    return ok


def test_zero() -> bool:
    M, K, K_pad = 128, 256, 256
    packed = torch.full((M, K_pad // 2), 0x77, dtype=torch.uint8, device="cuda")
    scale = torch.ones(M, K_pad // 64, dtype=torch.float16, device="cuda")
    x = rms_norm_x(K, seed=5)
    y = nf4_gemm(packed, scale, x, M, K, K_pad)
    torch.cuda.synchronize()
    maxabs = float(y.float().abs().max().item())
    ok = maxabs <= 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"{status} zero nibble=7 scale=1: max|y|={maxabs:.6g} (want ~0)")
    return ok


def test_n_gt_16() -> bool:
    M, K, K_pad = 32, 64, 64
    packed = torch.full((M, K_pad // 2), 0x77, dtype=torch.uint8, device="cuda")
    scale = torch.ones(M, 1, dtype=torch.float16, device="cuda")
    x = torch.randn(K, 17, dtype=torch.bfloat16, device="cuda")
    try:
        nf4_gemm(packed, scale, x, M, K, K_pad)
        torch.cuda.synchronize()
        print("FAIL N=17: expected error")
        return False
    except RuntimeError as e:
        print(f"PASS N>16 rejected: {e}")
        return True


def test_toy_n16() -> bool:
    M, K, K_pad = 128, 256, 256
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    packed = torch.randint(0, 256, (M, K_pad // 2), dtype=torch.int32, device="cuda", generator=g)
    packed = packed.to(torch.uint8)
    sg = torch.Generator(device="cuda")
    sg.manual_seed(11)
    scale = (0.04 + 0.20 * torch.rand(M, K_pad // 64, device="cuda", generator=sg)).to(
        torch.float16
    )
    x = rms_norm_x(K, seed=21, N=16)
    ok, _, _ = run_case("toy 128x256 N=16", packed, scale, x, M, K, K_pad)
    return ok


def test_n3_tails() -> bool:
    M, K, K_pad, N = 130, 65, 128, 3
    packed = torch.full((M, K_pad // 2), 0xFF, dtype=torch.uint8, device="cuda")
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    live = torch.randint(0, 256, (M, (K + 1) // 2), dtype=torch.int32, device="cuda", generator=g)
    packed[:, : (K + 1) // 2] = live.to(torch.uint8)
    sg = torch.Generator(device="cuda")
    sg.manual_seed(13)
    scale = (0.04 + 0.20 * torch.rand(M, K_pad // 64, device="cuda", generator=sg)).to(
        torch.float16
    )
    x3 = rms_norm_x(K, seed=4, N=N)
    ok, maxabs, rmse = run_case("tails 130x65 N=3", packed, scale, x3, M, K, K_pad)
    y3 = nf4_gemm(packed, scale, x3, M, K, K_pad)
    torch.cuda.synchronize()
    if tuple(y3.shape) != (M, N):
        print(f"FAIL N=3 shape {tuple(y3.shape)} want {(M, N)} (pad must not extend y)")
        return False

    # Same kernel, explicit zero pad to 16: live columns must match; pad columns ~0.
    x16 = torch.zeros(K, 16, dtype=torch.bfloat16, device="cuda")
    x16[:, :N] = x3
    y16 = nf4_gemm(packed, scale, x16, M, K, K_pad)
    torch.cuda.synchronize()
    live_err = (y3.float() - y16[:, :N].float()).abs().max().item()
    pad_max = float(y16[:, N:].float().abs().max().item())
    leak_ok = live_err <= 1e-3 and pad_max <= 1e-3
    status = "PASS" if leak_ok else "FAIL"
    print(
        f"{status} N=3 vs N=16 zero-pad: live maxabs={live_err:.6g} "
        f"pad-cols max|y|={pad_max:.6g} (pad must not leak); "
        f"y shape {tuple(y3.shape)}; maxabs={maxabs:.6g} rmse={rmse:.6g}"
    )
    return ok and leak_ok


def test_gate() -> bool:
    if not CHR_PATH.is_file():
        print(f"SKIP gate_proj: {CHR_PATH} missing")
        return True
    try:
        import chr0
    except ImportError:
        print("SKIP gate_proj: gpu/chr0 loader not importable")
        return True
    w = chr0.materialize_nf4(str(CHR_PATH), GATE_NAME)
    print(f"  loaded {w.name} M={w.M} K={w.K} K_pad={w.K_pad} packed={tuple(w.packed.shape)}")
    if w.M != 11008 or w.K != 2048:
        print(f"FAIL unexpected gate_proj shape M={w.M} K={w.K}")
        return False
    x = rms_norm_x(w.K, seed=7)
    ok, maxabs, rmse = run_case(
        f"gate_proj {w.M}x{w.K} N=1", w.packed, w.scale, x, w.M, w.K, w.K_pad
    )
    grid = (w.M + 127) // 128
    print(f"  grid ceil(M/128)={grid}")
    return ok


def test_gate_n16() -> bool:
    if not CHR_PATH.is_file():
        print(f"SKIP gate_proj N=16: {CHR_PATH} missing")
        return True
    try:
        import chr0
    except ImportError:
        print("SKIP gate_proj N=16: gpu/chr0 loader not importable")
        return True
    w = chr0.materialize_nf4(str(CHR_PATH), GATE_NAME)
    x = rms_norm_x(w.K, seed=17, N=16)
    ok, maxabs, rmse = run_case(
        f"gate_proj {w.M}x{w.K} N=16", w.packed, w.scale, x, w.M, w.K, w.K_pad
    )
    grid = (w.M + 63) // 64
    print(f"  grid ceil(M/64)={grid} maxabs={maxabs:.6g} rmse={rmse:.6g}")
    return ok


def test_linear() -> bool:
    """nf4_linear permute [...,K] -> [K,N] for N=1,3,16."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from gpu.host.linear import nf4_linear

    M, K, K_pad = 128, 256, 256
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    packed = torch.randint(0, 256, (M, K_pad // 2), dtype=torch.int32, device="cuda", generator=g)
    packed = packed.to(torch.uint8)
    sg = torch.Generator(device="cuda")
    sg.manual_seed(11)
    scale = (0.04 + 0.20 * torch.rand(M, K_pad // 64, device="cuda", generator=sg)).to(
        torch.float16
    )
    ok = True
    for n, seed in ((1, 2), (3, 4), (16, 21)):
        xk = rms_norm_x(K, seed=seed, N=n)
        yk = nf4_gemm(packed, scale, xk, M, K, K_pad)
        x_hf = xk.T.contiguous().view(1, n, K)
        y = nf4_linear(x_hf, packed, scale, M, K, K_pad)
        if tuple(y.shape) != (1, n, M):
            print(f"FAIL nf4_linear N={n} shape {tuple(y.shape)} want {(1, n, M)}")
            ok = False
            continue
        err = (y.float().reshape(n, M).T - yk.float()).abs().max().item()
        n_ok = err <= 1e-5
        status = "PASS" if n_ok else "FAIL"
        print(f"{status} nf4_linear N={n} vs nf4_gemm: maxabs={err:.6g} y{tuple(y.shape)}")
        ok = ok and n_ok
    return ok


def dump_sass() -> None:
    cuobjdump = os.environ.get(
        "CUOBJDUMP",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\cuobjdump.exe",
    )
    if not Path(cuobjdump).is_file():
        print("SASS: cuobjdump not found (not a blocker)")
        return
    import glob

    try:
        from torch.utils.cpp_extension import _get_build_directory

        jit_pyd = str(Path(_get_build_directory("chr_nf4_ext", False)) / "*.pyd")
    except Exception:
        jit_pyd = str(_DIR_EXT_SEARCH())
    patterns = [jit_pyd, str(_DIR_EXT_SEARCH())]
    files = []
    files.extend(glob.glob(str(GPU_DIR / "nf4" / "chr_nf4_ext*.pyd")))
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        print("SASS: extension binary not found (not a blocker)")
        return
    target = files[0]
    print(f"SASS: cuobjdump {target}")
    import subprocess

    out = subprocess.run(
        [cuobjdump, "-sass", target],
        capture_output=True,
        text=True,
        errors="replace",
    )
    text = out.stdout + out.stderr
    hmma = text.count("HMMA")
    ffma = text.count("FFMA")
    hmma16816 = "HMMA.16816" in text or "HMMA.16816.F32.BF16" in text
    print(f"  HMMA count={hmma} FFMA count={ffma} has_HMMA.16816={hmma16816}")
    if hmma16816:
        print("PASS SASS contains HMMA.16816")
    elif hmma:
        print("NOTE SASS has HMMA but not the 16816 tag string")
    else:
        print("NOTE no HMMA in cuobjdump (check arch / binary)")


def _DIR_EXT_SEARCH() -> str:
    try:
        from torch.utils.cpp_extension import _get_build_directory

        return str(Path(_get_build_directory("chr_nf4_ext", False)) / "*")
    except Exception:
        return str(Path.home() / "AppData" / "Local" / "torch_extensions" / "*" / "chr_nf4_ext*")


def grep_kernel_allocs() -> bool:
    src = (GPU_DIR / "nf4" / "nf4_gemm.cu").read_text(encoding="utf-8")
    bad = []
    for needle in ("cudaMalloc", "cudaMallocAsync", "new ", "malloc("):
        if needle in src:
            bad.append(needle)
    if bad:
        print(f"FAIL kernel source contains {bad}")
        return False
    print("PASS no cudaMalloc/new/malloc in nf4_gemm.cu")
    return True


def main() -> int:
    print("device", torch.cuda.get_device_name(0))
    results = [
        test_lut_bits(),
        grep_kernel_allocs(),
        test_toy(),
        test_tails(),
        test_zero(),
        test_toy_n16(),
        test_n3_tails(),
        test_n_gt_16(),
        test_gate(),
        test_gate_n16(),
        test_linear(),
    ]
    dump_sass()
    if all(results):
        print("ALL PASS")
        return 0
    print("SOME FAIL")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
