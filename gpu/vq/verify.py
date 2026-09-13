"""Oracle checks for chr_vq_gemm. Run with torch-gpu python from repo root:

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe gpu/vq/verify.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

GPU_DIR = Path(__file__).resolve().parent.parent
REPO = GPU_DIR.parent
sys.path.insert(0, str(GPU_DIR))

from vq import vq_gemm  # noqa: E402

MAXABS_LIMIT = 0.05
MODELS_DIR = Path(r"C:\dev\models")


def k_pad_vq(K: int) -> int:
    return 8 * ((K + 7) // 8)


def dequant_vq(index: np.ndarray, book: np.ndarray, M: int, K: int, K_pad: int) -> np.ndarray:
    G = K_pad // 8
    idx = np.ascontiguousarray(index).reshape(M, G, 2)
    book_f = np.asarray(book, dtype=np.float16).astype(np.float32).reshape(2, 256, 8)
    i1 = idx[:, :, 0].astype(np.intp)
    i2 = idx[:, :, 1].astype(np.intp)
    g = book_f[0, i1] + book_f[1, i2]
    return g.reshape(M, K_pad)[:, :K]


def four_vec_book_index(M: int, K: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Exact (fp16) reconstruct of vq.md §9.1 templates via additive 2x8."""
    K_pad = k_pad_vq(K)
    G = K_pad // 8
    book = np.zeros((2, 256, 8), dtype=np.float16)
    book[0, 1] = np.float16([1, 0, 0, 0, 0, 0, 0, 0])
    book[0, 2] = np.float16([0, 0.5, 0, 0, 0, 0, 0, 0])
    book[1, 2] = np.float16([0, 0.5, 0, 0, 0, 0, 0, 0])
    book[0, 3] = np.float16([0, 0, 0.25, 0, 0, 0, 0, 0])
    book[1, 7] = np.float16([0, 0, 0.75, 0, 0, 0, 0, 0])
    pairs = ((0, 0), (1, 0), (2, 2), (3, 7))
    index = np.zeros((M, G, 2), dtype=np.uint8)
    for r in range(M):
        i1, i2 = pairs[r % 4]
        for j in range(G):
            if j * 8 < K:
                index[r, j, 0] = i1
                index[r, j, 1] = i2
    return index, book, K_pad


def layout_1x16() -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """vq.md §6.2 / §7.3 golden layout (no k-means)."""
    M, K = 1, 16
    K_pad = 16
    book = np.zeros((2, 256, 8), dtype=np.float16)
    book[0, 0, 0] = np.float16(1.0)
    book[0, 1, 1] = np.float16(1.0)
    book[1, 7, 2] = np.float16(0.5)
    index = np.zeros((M, K_pad // 8, 2), dtype=np.uint8)
    index[0, 0, 0] = 0
    index[0, 0, 1] = 7
    index[0, 1, 0] = 1
    index[0, 1, 1] = 0
    return index, book, M, K, K_pad


def rms_norm_x(K: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(K, 1, dtype=torch.bfloat16, device="cuda", generator=g)
    rms = x.float().pow(2).mean().sqrt().clamp_min(1e-6)
    return (x.float() / rms).to(torch.bfloat16)


def stats(y_gpu: torch.Tensor, y_cpu: np.ndarray) -> tuple[float, float]:
    yg = y_gpu.float().detach().cpu().numpy().reshape(-1)
    yc = np.asarray(y_cpu, dtype=np.float32).reshape(-1)
    err = yg - yc
    maxabs = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))
    return maxabs, rmse


def run_case(name: str, index: torch.Tensor, book: torch.Tensor, x: torch.Tensor,
             M: int, K: int, K_pad: int) -> tuple[bool, float, float]:
    y = vq_gemm(index, book, x, M, K, K_pad)
    torch.cuda.synchronize()
    W = dequant_vq(
        index.detach().cpu().numpy(),
        book.detach().cpu().numpy(),
        M,
        K,
        K_pad,
    )
    x_f = x.float().detach().cpu().numpy().reshape(K, -1)
    y_cpu = W @ x_f
    maxabs, rmse = stats(y, y_cpu)
    W_bf = torch.from_numpy(np.ascontiguousarray(W)).to(torch.bfloat16).float().numpy()
    maxabs_bf, rmse_bf = stats(y, W_bf @ x_f)
    rms_x = float(np.sqrt(np.mean(x_f.astype(np.float64) ** 2)))
    ok = maxabs <= MAXABS_LIMIT
    status = "PASS" if ok else "FAIL"
    print(
        f"{status} {name}: maxabs={maxabs:.6g} rmse={rmse:.6g} rms(x)={rms_x:.6g} "
        f"(limit {MAXABS_LIMIT}; vs bf16(W) maxabs={maxabs_bf:.6g} rmse={rmse_bf:.6g})"
    )
    return ok, maxabs, rmse


def to_cuda(index: np.ndarray, book: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(index)).to(device="cuda", dtype=torch.uint8),
        torch.from_numpy(np.ascontiguousarray(book)).to(device="cuda", dtype=torch.float16),
    )


def test_toy() -> bool:
    M, K = 128, 256
    index_np, book, K_pad = four_vec_book_index(M, K)
    index, book_t = to_cuda(index_np, book)
    x = rms_norm_x(K, seed=2)
    ok, _, _ = run_case("toy 128x256 four-vec", index, book_t, x, M, K, K_pad)
    return ok


def test_tails() -> bool:
    M, K = 130, 65
    index_np, book, K_pad = four_vec_book_index(M, K)
    assert K_pad == 72, K_pad
    # Junk in the pad columns of the last mixed group must not change y:
    # those k>=K are masked in the kernel. Live k=64 still uses this group.
    index, book_t = to_cuda(index_np, book)
    x = rms_norm_x(K, seed=4)
    ok, maxabs, rmse = run_case("tails 130x65 K_pad=72", index, book_t, x, M, K, K_pad)
    y = vq_gemm(index, book_t, x, M, K, K_pad)
    torch.cuda.synchronize()
    if y.shape != (M, 1):
        print(f"FAIL tails shape {tuple(y.shape)} want {(M, 1)}")
        return False
    print(f"  y shape {tuple(y.shape)} (pad must not extend y); maxabs={maxabs:.6g} rmse={rmse:.6g}")
    return ok


def test_layout_1x16() -> bool:
    index_np, book, M, K, K_pad = layout_1x16()
    W = dequant_vq(index_np, book, M, K, K_pad)
    want = np.array(
        [[1, 0, 0.5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    if not np.array_equal(W, want):
        print(f"FAIL layout reconstruct {W} want {want}")
        return False
    index, book_t = to_cuda(index_np, book)
    x = rms_norm_x(K, seed=9)
    ok, _, _ = run_case("layout 1x16 golden", index, book_t, x, M, K, K_pad)
    return ok


def test_n_not_one() -> bool:
    M, K = 32, 64
    K_pad = k_pad_vq(K)
    index_np, book, _ = four_vec_book_index(M, K)
    index, book_t = to_cuda(index_np, book)
    x = torch.randn(K, 2, dtype=torch.bfloat16, device="cuda")
    try:
        vq_gemm(index, book_t, x, M, K, K_pad)
        torch.cuda.synchronize()
        print("FAIL N=2: expected error")
        return False
    except RuntimeError as e:
        print(f"PASS N!=1 rejected: {e}")
        return True


def grep_kernel_allocs() -> bool:
    src = (GPU_DIR / "vq" / "vq_gemm.cu").read_text(encoding="utf-8")
    bad = []
    for needle in ("cudaMalloc", "cudaMallocAsync", "new ", "malloc("):
        if needle in src:
            bad.append(needle)
    if bad:
        print(f"FAIL kernel source contains {bad}")
        return False
    if "65536" in src or "1 << 16" in src or "1<<16" in src:
        print("FAIL kernel source looks like a 2^16 book gather")
        return False
    print("PASS no cudaMalloc/new/malloc and no 2^16 book in vq_gemm.cu")
    return True


def find_vq_chr() -> list[Path]:
    if not MODELS_DIR.is_dir():
        return []
    out: list[Path] = []
    for pat in ("*.vq2.chr", "*.vq.chr", "*vq*.chr"):
        out.extend(sorted(MODELS_DIR.glob(pat)))
    # de-dup while keeping order
    seen = set()
    uniq = []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def test_real_chr() -> bool:
    files = find_vq_chr()
    if not files:
        print(f"SKIP real .vq2.chr: none found under {MODELS_DIR}")
        return True
    path = files[0]
    print(f"  found {path}")
    try:
        import chr0
    except ImportError:
        print("SKIP real vq chr: gpu/chr0 loader not importable")
        return True
    hdr = chr0.load_header(str(path))
    names = [n for n, t in hdr.tensors.items() if t.codec == "vq"]
    if not names:
        print(f"SKIP {path.name}: header has no codec=vq tensors")
        return True
    name = names[0]
    info = hdr.tensor(name)
    M, K, K_pad = info.M, info.K, info.K_pad
    idx_blob = info.blobs["index"]
    book_blob = info.blobs["codebook"]
    with open(path, "rb") as f:
        f.seek(idx_blob.start)
        idx_bytes = f.read(idx_blob.nbytes)
        f.seek(book_blob.start)
        book_bytes = f.read(book_blob.nbytes)
    G = K_pad // 8
    index_np = np.frombuffer(idx_bytes, dtype=np.uint8)[: M * G * 2].reshape(M, G, 2).copy()
    book = np.frombuffer(book_bytes, dtype=np.float16).reshape(2, 256, 8).copy()
    index, book_t = to_cuda(index_np, book)
    x = rms_norm_x(K, seed=7)
    ok, _, _ = run_case(f"real {path.name} {name} {M}x{K}", index, book_t, x, M, K, K_pad)
    return ok


def main() -> int:
    print("device", torch.cuda.get_device_name(0))
    results = [
        grep_kernel_allocs(),
        test_toy(),
        test_tails(),
        test_layout_1x16(),
        test_n_not_one(),
        test_real_chr(),
    ]
    if all(results):
        print("ALL PASS")
        return 0
    print("SOME FAIL")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
