"""Split NF4 error into quantization vs kernel. Default path is CPU-only.

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe -m gpu.nf4.numerics
    python -m gpu.nf4.numerics --tiny
    python -m gpu.nf4.numerics --cuda --n 1,16          # after the 3B lab
    python -m gpu.nf4.numerics --chr PATH --model-dir DIR --cuda

Two error legs, never added into one number:

  quant   BF16 ``W`` vs CPU NF4 reconstruction (``docs/spec/nf4.md`` encode+decode).
          The GEMM of each weight @ the same ``x`` is the same leg on surface ``y``.
  kernel  CPU NF4 GEMM (``nf4_oracle``) vs ``chr_nf4_gemm``. CUDA only.

``gpu/nf4/verify.py`` prints an extra ``vs bf16(W)`` figure that mixes both legs;
this table does not. It never loads the 3B, never JIT-compiles the extension
unless ``--cuda`` is set *and* nvidia-smi says the card is free (or
``--cuda-force`` after the lab). Live ``N`` stays in 1..``LIVE_MAX_N``.
``--plan-n`` allows up to 64 for the kernel oracle only; TokenLoop chunks
above ``LIVE_MAX_N``. Kernel rows ``fail`` when any |Y_gpu−Y_cpu| exceeds
max(0.05, half BF16 ULP).
Kernel rows ``fail`` when any |Y_gpu−Y_cpu| exceeds max(0.05, half BF16 ULP).

Table / CSV columns: ``leg, surface, kind, source, M, K, N, status,
max_abs, mean_abs, rmse, cosine, rel, skip_reason``. ``leg=quant`` is BF16 vs
CPU NF4; ``leg=kernel`` is CPU NF4 GEMM vs CUDA. ``surface=W`` is the weight
reconstruct; ``surface=y`` is the GEMM. Do not compare CUDA ``y`` to BF16 ``y``
as a single mixed error.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.plan import LIVE_MAX_N, PLAN_MAX_N  # noqa: E402
from gpu.tests.nf4_oracle import decode_nf4, encode_nf4, k_pad, matmul_f32  # noqa: E402

__all__ = [
    "BUSY_PROC_MIB",
    "BUSY_USED_MIB",
    "CHR_SLOT",
    "HF_WEIGHT",
    "KERNEL_ABS_LIMIT",
    "LEG_KERNEL",
    "LEG_QUANT",
    "Metrics",
    "QWEN25_3B_LINEARS",
    "Row",
    "cuda_launch_reason",
    "cuda_nf4_gemm",
    "encode_nf4_chunked",
    "error_metrics",
    "kernel_floor_ok",
    "format_table",
    "gpu_busy_reason",
    "load_chr_nf4_cpu",
    "load_hf_weight_f32",
    "random_bf16_w",
    "rms_norm_x",
    "run_kind",
    "main",
]

LEG_QUANT = "quant"
LEG_KERNEL = "kernel"

#: Stitch floor 1 for |Y| ~ O(1). When |Y_cpu| sits in a wide BF16 binade the
#: store is allowed a half-ULP of that value (0.0625 on [16, 32)). A flat
#: 0.05 otherwise flags rounding of y≈17 as a kernel bug; n16 on the same x
#: hits the same element.
KERNEL_ABS_LIMIT = 0.05

#: nvidia-smi used-MiB. Display is inside that meter (~2 GiB on this WDDM
#: 3080) and is not a lab. A live 3B NF4 session sits near 3900 MiB after
#: load; BF16 around 8200. Desktop idle is ~2100, so 3072 is above the
#: desktop and below a resident 3B. Do not treat 1536 as busy: that would
#: skip ``--cuda`` on every idle boot and force ``--cuda-force``.
BUSY_USED_MIB = 3072
BUSY_PROC_MIB = 256

#: Qwen2.5-3B-Instruct Linear slots: hidden 2048, intermediate 11008, GQA
#: 16/2 heads x 128. Same numbers as ``gpu.nf4.bench.QWEN25_3B`` minus ``lm_head``.
QWEN25_3B_LINEARS: tuple[tuple[str, int, int], ...] = (
    ("q_proj", 2048, 2048),
    ("k_proj", 256, 2048),
    ("v_proj", 256, 2048),
    ("o_proj", 2048, 2048),
    ("gate_proj", 11008, 2048),
    ("up_proj", 11008, 2048),
    ("down_proj", 2048, 11008),
)

CHR_SLOT = {
    "q_proj": "model.layers.{layer}.self_attn.q_proj",
    "k_proj": "model.layers.{layer}.self_attn.k_proj",
    "v_proj": "model.layers.{layer}.self_attn.v_proj",
    "o_proj": "model.layers.{layer}.self_attn.o_proj",
    "gate_proj": "model.layers.{layer}.mlp.gate_proj",
    "up_proj": "model.layers.{layer}.mlp.up_proj",
    "down_proj": "model.layers.{layer}.mlp.down_proj",
}

HF_WEIGHT = {kind: name + ".weight" for kind, name in CHR_SLOT.items()}

LEGEND = (
    "Two error legs (do not add them, do not compare CUDA to BF16 as one number):\n"
    "  quant   BF16 W vs CPU NF4 reconstruction; surface y is those weights @ the same x\n"
    "  kernel  CPU NF4 GEMM vs CUDA chr_nf4_gemm (needs --cuda and a free GPU)\n"
)


@dataclass(frozen=True)
class Metrics:
    """Elementwise error of ``got`` vs ``ref``. ``rel`` is ||err||_2 / ||ref||_2."""

    max_abs: float
    mean_abs: float
    rmse: float
    cosine: float
    rel: float


@dataclass(frozen=True)
class Row:
    """One printed line. ``metrics`` is None when ``status`` is skip."""

    leg: str
    surface: str
    kind: str
    source: str
    M: int  # noqa: N815
    K: int  # noqa: N815
    N: int  # noqa: N815
    status: str
    metrics: Metrics | None = None
    skip_reason: str = ""


def error_metrics(got: np.ndarray, ref: np.ndarray) -> Metrics:
    """max abs, mean abs, RMSE, cosine, relative L2. Matches nf4.md §9 on the L2 trio.

    ``e_i`` is the float32 difference; RMSE / max / MAE accumulate that error in
    float64. Cosine and ``rel`` are over the flattened float64 values.
    """
    if got.shape != ref.shape:
        raise ValueError(f"shape {got.shape} != {ref.shape}")
    err32 = np.asarray(got, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
    e64 = err32.astype(np.float64).reshape(-1)
    if e64.size == 0:
        return Metrics(0.0, 0.0, 0.0, 1.0, 0.0)
    max_abs = float(np.max(np.abs(e64)))
    mean_abs = float(np.mean(np.abs(e64)))
    rmse = float(np.sqrt(np.mean(e64 * e64)))
    g = np.asarray(got, dtype=np.float64).reshape(-1)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    ng = float(np.linalg.norm(g))
    nr = float(np.linalg.norm(r))
    if ng == 0.0 and nr == 0.0:
        cosine = 1.0
    elif ng == 0.0 or nr == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.clip(np.dot(g, r) / (ng * nr), -1.0, 1.0))
    if nr == 0.0:
        rel = 0.0 if max_abs == 0.0 else float("inf")
    else:
        rel = float(np.linalg.norm(e64) / nr)
    return Metrics(max_abs=max_abs, mean_abs=mean_abs, rmse=rmse, cosine=cosine, rel=rel)


def bf16_half_ulp(ref: np.ndarray) -> np.ndarray:
    """Half ULP of BF16 at each |ref| (7 mantissa bits, same exponent as FP32)."""
    ax = np.maximum(np.abs(np.asarray(ref, dtype=np.float32)), np.float32(2.0**-126))
    exp = np.floor(np.log2(ax.astype(np.float64)))
    return 0.5 * np.power(2.0, exp - 7.0)


def kernel_floor_ok(got: np.ndarray, ref: np.ndarray) -> bool:
    """Kernel vs CPU NF4 GEMM. Per element: |err| ≤ max(0.05, half BF16 ULP).

    ``docs/spec/stitch-gpu.md``: 0.05 is the O(1) floor. BF16 ``y`` store of a
    large logit is allowed its own half-ULP. 2026-09-14 live 3B ``q_proj`` N=32
    was 0.05847 at y≈−17.44 (one element, bit-identical to 2×n16).
    """
    if got.shape != ref.shape:
        raise ValueError(f"shape {got.shape} != {ref.shape}")
    err = np.abs(np.asarray(got, dtype=np.float32) - np.asarray(ref, dtype=np.float32))
    allow = np.maximum(KERNEL_ABS_LIMIT, bf16_half_ulp(ref))
    return bool(np.all(err <= allow))


def encode_nf4_chunked(
    w: np.ndarray, row_chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """``encode_nf4`` in row batches so a 3B ``gate_proj`` does not allocate 1.4 GiB."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise TypeError(f"encode wants [M, K], got {w.shape}")
    m = w.shape[0]
    chunk = max(1, int(row_chunk))
    packed_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for lo in range(0, m, chunk):
        p, s = encode_nf4(w[lo : min(lo + chunk, m)])
        packed_parts.append(p)
        scale_parts.append(s)
    return np.concatenate(packed_parts, axis=0), np.concatenate(scale_parts, axis=0)


def to_bf16_f32(w: np.ndarray) -> np.ndarray:
    """Round to BF16 on CPU, return the values as float32 (nf4.md §2.1)."""
    import torch

    arr = np.array(w, dtype=np.float32, copy=True, order="C")
    return torch.from_numpy(arr).to(torch.bfloat16).float().numpy()


def random_bf16_w(m: int, k: int, seed: int, *, scale: float = 0.02) -> np.ndarray:
    """LLM-like BF16 weights: N(0, ``scale``), rounded the way a checkpoint is."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = (float(scale) * torch.randn(int(m), int(k), dtype=torch.float32, generator=g))
    return w.to(torch.bfloat16).float().numpy()


def rms_norm_x(k: int, n: int, seed: int, *, max_n: int = LIVE_MAX_N) -> np.ndarray:
    """BF16 activations, RMS-normalised, returned as float32 [K, N] (the kernel's x)."""
    import torch

    if not 1 <= int(n) <= int(max_n):
        raise ValueError(f"N={n} not in 1..{max_n}")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(int(k), int(n), dtype=torch.bfloat16, generator=g)
    rms = x.float().pow(2).mean().sqrt().clamp_min(1e-6)
    return (x.float() / rms).to(torch.bfloat16).float().numpy()


def gpu_busy_reason(*, index: int = 0) -> str | None:
    """Why the kernel must not launch, from nvidia-smi only (no CUDA context).

    ``None`` means the card looks idle enough for a one-matrix GEMM. A live
    ``qwen25-3b-paired-*`` lab is busy.
    """
    try:
        used = subprocess.run(
            [
                "nvidia-smi",
                f"--id={int(index)}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi unusable ({type(exc).__name__}: {exc})"
    if used.returncode != 0:
        err = (used.stderr or used.stdout or "").strip() or f"exit {used.returncode}"
        return f"nvidia-smi memory.used failed: {err[:200]}"
    line = used.stdout.strip().splitlines()[0] if used.stdout.strip() else ""
    try:
        used_mib = int(float(line.split()[0]))
    except (TypeError, ValueError, IndexError):
        return f"nvidia-smi memory.used unreadable: {line!r}"

    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    heavy: list[str] = []
    if apps.returncode == 0:
        body = (apps.stdout or "").strip()
        if body and "no running processes" not in body.lower():
            for raw in body.splitlines():
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    proc_mib = int(float(parts[2].split()[0]))
                except (TypeError, ValueError):
                    proc_mib = 0
                if proc_mib >= BUSY_PROC_MIB:
                    heavy.append(f"{parts[0]} {Path(parts[1]).name} {proc_mib}MiB")
    if heavy:
        return (
            f"GPU busy: compute apps {'; '.join(heavy[:4])} "
            f"(smi used={used_mib} MiB). Wait for the 3B lab, then --cuda"
        )
    if used_mib >= BUSY_USED_MIB:
        return (
            f"GPU busy: nvidia-smi memory.used={used_mib} MiB "
            f"(threshold {BUSY_USED_MIB}). Wait for the 3B lab, then --cuda"
        )
    return None


def cuda_launch_reason(*, want_cuda: bool, force: bool = False, index: int = 0) -> str | None:
    """``None`` when the CUDA kernel may run. Anything else is a skip reason."""
    if not want_cuda:
        return "pass --cuda to compare the kernel (GPU must be free)"
    if force:
        return None
    return gpu_busy_reason(index=index)


def cuda_nf4_gemm(
    packed: np.ndarray,
    scale: np.ndarray,
    x: np.ndarray,
    m: int,
    k: int,
    kp: int,
    *,
    gemm: Callable[..., object] | None = None,
    max_n: int = LIVE_MAX_N,
) -> np.ndarray:
    """``y[M, N]`` float32 from the CUDA kernel (or an injected host ``gemm``).

    The ``gpu.nf4.nf4_gemm`` import is inside this function on purpose: it is
    what loads ``chr_nf4_ext`` and may JIT. An injected ``gemm`` stays on CPU
    so tests can split the two legs without touching the 3080. The real
    kernel is used only when ``gemm is None`` *and* ``--cuda`` cleared
    :func:`cuda_launch_reason`.
    """
    import torch

    packed_t = torch.from_numpy(np.array(packed, copy=True, order="C"))
    scale_t = torch.from_numpy(np.array(scale, copy=True, order="C"))
    x_host = torch.from_numpy(np.array(x, dtype=np.float32, copy=True, order="C")).to(
        torch.bfloat16
    )
    if gemm is not None:
        y = gemm(packed_t, scale_t, x_host, int(m), int(k), int(kp))
        if isinstance(y, np.ndarray):
            return np.ascontiguousarray(y, dtype=np.float32)
        return np.ascontiguousarray(y.float().detach().cpu().numpy(), dtype=np.float32)

    from gpu.nf4 import nf4_gemm  # noqa: PLC0415

    packed_t = packed_t.to(device="cuda", dtype=torch.uint8)
    scale_t = scale_t.to(device="cuda", dtype=torch.float16)
    x_t = x_host.to(device="cuda")
    y = nf4_gemm(packed_t, scale_t, x_t, int(m), int(k), int(kp), int(max_n))
    torch.cuda.synchronize()
    out = y.float().detach().cpu().numpy()
    del y, packed_t, scale_t, x_t, x_host
    return np.ascontiguousarray(out, dtype=np.float32)


def _row(
    leg: str,
    surface: str,
    kind: str,
    source: str,
    m: int,
    k: int,
    n: int,
    metrics: Metrics | None,
    *,
    status: str = "ok",
    skip_reason: str = "",
) -> Row:
    return Row(
        leg=leg,
        surface=surface,
        kind=kind,
        source=source,
        M=int(m),
        K=int(k),
        N=int(n),
        status=status,
        metrics=metrics,
        skip_reason=skip_reason,
    )


def _skip_kernel(kind: str, source: str, m: int, k: int, n: int, reason: str) -> Row:
    return _row(
        LEG_KERNEL, "y", kind, source, m, k, n, None, status="skip", skip_reason=reason
    )


def run_kind(
    kind: str,
    w_bf16: np.ndarray | None,
    packed: np.ndarray,
    scale: np.ndarray,
    x: np.ndarray,
    *,
    source: str,
    gemm: Callable[..., object] | None = None,
    kernel_reason: str | None = None,
    max_n: int = LIVE_MAX_N,
) -> list[Row]:
    """One matrix: quant (if ``w_bf16`` given) then kernel (or a skip row).

    ``w_bf16`` is the original / synthetic BF16 matrix as float32. Packed
    bytes may come from :func:`encode_nf4_chunked` or from a ``.chr``.
    """
    m, k = packed.shape[0], x.shape[0]
    n = int(x.shape[1])
    if not 1 <= n <= int(max_n):
        raise ValueError(f"N={n} not in 1..{max_n}")
    kp = k_pad(k)
    w_hat = decode_nf4(packed, scale, m, k)
    y_cpu = matmul_f32(w_hat, np.ascontiguousarray(x, dtype=np.float32))
    rows: list[Row] = []
    if w_bf16 is None:
        rows.append(
            _row(
                LEG_QUANT,
                "W",
                kind,
                source,
                m,
                k,
                n,
                None,
                status="skip",
                skip_reason="no BF16 W (--model-dir, or random encode)",
            )
        )
        rows.append(
            _row(
                LEG_QUANT,
                "y",
                kind,
                source,
                m,
                k,
                n,
                None,
                status="skip",
                skip_reason="no BF16 W; cannot form y_bf16",
            )
        )
    else:
        if w_bf16.shape != (m, k):
            raise ValueError(f"W {w_bf16.shape} != packed logical ({m}, {k})")
        w_ref = np.ascontiguousarray(w_bf16, dtype=np.float32)
        rows.append(_row(LEG_QUANT, "W", kind, source, m, k, n, error_metrics(w_hat, w_ref)))
        y_bf16 = matmul_f32(w_ref, np.ascontiguousarray(x, dtype=np.float32))
        rows.append(_row(LEG_QUANT, "y", kind, source, m, k, n, error_metrics(y_cpu, y_bf16)))

    if kernel_reason is not None:
        rows.append(_skip_kernel(kind, source, m, k, n, kernel_reason))
        return rows
    try:
        y_gpu = cuda_nf4_gemm(
            packed, scale, x, m, k, kp, gemm=gemm, max_n=max_n
        )
    except Exception as exc:  # noqa: BLE001 - reporter, not a kernel
        rows.append(
            _skip_kernel(kind, source, m, k, n, f"{type(exc).__name__}: {exc}")
        )
        return rows
    if y_gpu.shape != y_cpu.shape:
        rows.append(
            _skip_kernel(
                kind,
                source,
                m,
                k,
                n,
                f"CUDA y{tuple(y_gpu.shape)} != CPU y{tuple(y_cpu.shape)}",
            )
        )
        return rows
    metrics = error_metrics(y_gpu, y_cpu)
    if kernel_floor_ok(y_gpu, y_cpu):
        rows.append(_row(LEG_KERNEL, "y", kind, source, m, k, n, metrics))
        return rows
    rows.append(
        _row(
            LEG_KERNEL,
            "y",
            kind,
            source,
            m,
            k,
            n,
            metrics,
            status="fail",
            skip_reason=(
                f"kernel floor: maxabs={metrics.max_abs:.5g} "
                f"(0.05 or half BF16 ULP of |Y_cpu|)"
            ),
        )
    )
    return rows


def _cap_rows(
    packed: np.ndarray,
    scale: np.ndarray,
    w: np.ndarray | None,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_rows <= 0 or packed.shape[0] <= max_rows:
        return packed, scale, w
    packed = packed[:max_rows]
    scale = scale[:max_rows]
    if w is not None:
        w = w[:max_rows]
    return packed, scale, w


def run_random(
    kinds: Sequence[tuple[str, int, int]],
    *,
    ns: Sequence[int],
    seed: int,
    max_rows: int = 0,
    row_chunk: int = 256,
    kernel_reason: str | None = None,
    gemm: Callable[..., object] | None = None,
    max_n: int = LIVE_MAX_N,
) -> list[Row]:
    """Synthetic BF16 W, Python NF4 encode, CPU GEMM; kernel only if allowed."""
    rows: list[Row] = []
    for i, (kind, m, k) in enumerate(kinds):
        m_use = min(int(m), max_rows) if max_rows > 0 else int(m)
        w = random_bf16_w(m_use, int(k), seed + i)
        packed, scale = encode_nf4_chunked(w, row_chunk=row_chunk)
        for n in ns:
            x = rms_norm_x(int(k), int(n), seed + 1000 + i + n, max_n=max_n)
            rows.extend(
                run_kind(
                    kind,
                    w,
                    packed,
                    scale,
                    x,
                    source="random",
                    gemm=gemm,
                    kernel_reason=kernel_reason,
                    max_n=max_n,
                )
            )
    return rows


def load_chr_nf4_cpu(path: str, name: str) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Packed/scale of one CHR0 tensor on the host. No torch, no CUDA."""
    from gpu.chr0 import load_header

    hdr = load_header(str(path))
    info = hdr.tensor(name)
    data = info.blobs["data"]
    scale_blob = info.blobs["scale"]
    with open(path, "rb") as f:
        f.seek(data.start)
        packed = np.frombuffer(f.read(data.nbytes), dtype=np.uint8).copy()
        f.seek(scale_blob.start)
        scale = np.frombuffer(f.read(scale_blob.nbytes), dtype=np.float16).copy()
    packed = packed.reshape(info.M, info.K_pad // 2)
    scale = scale.reshape(info.M, info.n_groups)
    return packed, scale, info.M, info.K, info.K_pad


def _safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        (n,) = struct.unpack("<Q", raw_len)
        raw = f.read(n)
    if len(raw) != n:
        raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(raw.decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + n


def _decode_safetensors_bytes(raw: bytes, dtype: str, shape: Sequence[int]) -> np.ndarray:
    shape = tuple(int(a) for a in shape)
    if dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        bits = u16.astype(np.uint32) << 16
        arr = bits.view(np.float32)
    else:
        raise TypeError(f"unsupported safetensors dtype {dtype!r}")
    return np.ascontiguousarray(arr.reshape(shape), dtype=np.float32)


def load_hf_weight_f32(model_dir: str | Path, tensor_name: str) -> np.ndarray:
    """One shard tensor as float32. CPU only; does not mmap the whole model."""
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"model dir missing: {root}")
    index = root / "model.safetensors.index.json"
    shard: Path | None = None
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        rel = weight_map.get(tensor_name)
        if rel:
            shard = root / rel
    if shard is None:
        for name in ("model.safetensors", "pytorch_model.bin"):
            candidate = root / name
            if candidate.is_file() and candidate.suffix == ".safetensors":
                shard = candidate
                break
        if shard is None:
            shards = sorted(root.glob("model-*.safetensors"))
            for candidate in shards:
                header, _ = _safetensors_header(candidate)
                if tensor_name in header:
                    shard = candidate
                    break
    if shard is None or not shard.is_file():
        raise FileNotFoundError(f"{tensor_name} not found under {root}")
    header, data_start = _safetensors_header(shard)
    info = header.get(tensor_name)
    if not isinstance(info, dict):
        raise KeyError(f"{shard.name} has no {tensor_name}")
    start, end = info["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(data_start + int(start))
        raw = f.read(int(end) - int(start))
    w = _decode_safetensors_bytes(raw, str(info["dtype"]), info["shape"])
    return to_bf16_f32(w) if str(info["dtype"]) != "BF16" else w


def run_real(
    kinds: Sequence[tuple[str, int, int]],
    *,
    ns: Sequence[int],
    seed: int,
    layer: int,
    chr_path: str | None,
    model_dir: str | None,
    max_rows: int = 0,
    row_chunk: int = 256,
    kernel_reason: str | None = None,
    gemm: Callable[..., object] | None = None,
    max_n: int = LIVE_MAX_N,
) -> list[Row]:
    """Optional ``.chr`` / HuggingFace dir. CPU reads only; kernel still gated."""
    rows: list[Row] = []
    for n in ns:
        if not 1 <= int(n) <= int(max_n):
            raise ValueError(f"N={n} not in 1..{max_n}")
    for kind, m_expect, k_expect in kinds:
        chr_name = CHR_SLOT[kind].format(layer=int(layer))
        hf_name = HF_WEIGHT[kind].format(layer=int(layer))
        packed = scale = w_bf16 = None
        source = "real"
        try:
            if chr_path:
                packed, scale, m, k, _kp = load_chr_nf4_cpu(chr_path, chr_name)
                if (m, k) != (m_expect, k_expect) and max_rows <= 0:
                    rows.append(
                        _row(
                            LEG_QUANT,
                            "W",
                            kind,
                            "chr",
                            m,
                            k,
                            ns[0],
                            None,
                            status="skip",
                            skip_reason=f"chr shape [{m},{k}] != catalog [{m_expect},{k_expect}]",
                        )
                    )
                    continue
            if model_dir:
                w_bf16 = load_hf_weight_f32(model_dir, hf_name)
            if packed is None:
                if w_bf16 is None:
                    continue
                if w_bf16.shape != (m_expect, k_expect):
                    rows.append(
                        _row(
                            LEG_QUANT,
                            "W",
                            kind,
                            "model",
                            w_bf16.shape[0],
                            w_bf16.shape[1],
                            ns[0],
                            None,
                            status="skip",
                            skip_reason=f"HF shape {tuple(w_bf16.shape)} != catalog",
                        )
                    )
                    continue
                packed, scale = encode_nf4_chunked(w_bf16, row_chunk=row_chunk)
                source = "model"
            elif w_bf16 is not None:
                source = "chr+model"
                if w_bf16.shape[0] != packed.shape[0] or w_bf16.shape[1] != k_expect:
                    rows.append(
                        _row(
                            LEG_QUANT,
                            "W",
                            kind,
                            source,
                            packed.shape[0],
                            k_expect,
                            ns[0],
                            None,
                            status="skip",
                            skip_reason=f"HF {tuple(w_bf16.shape)} vs chr packed M={packed.shape[0]}",
                        )
                    )
                    continue
            else:
                source = "chr"
            packed, scale, w_bf16 = _cap_rows(packed, scale, w_bf16, max_rows)
            k = int(k_expect if w_bf16 is None else w_bf16.shape[1])
            for n in ns:
                x = rms_norm_x(k, int(n), seed + 3000 + n, max_n=max_n)
                rows.extend(
                    run_kind(
                        kind,
                        w_bf16,
                        packed,
                        scale,
                        x,
                        source=source,
                        gemm=gemm,
                        kernel_reason=kernel_reason,
                        max_n=max_n,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                _row(
                    LEG_QUANT,
                    "W",
                    kind,
                    source,
                    m_expect,
                    k_expect,
                    ns[0],
                    None,
                    status="skip",
                    skip_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def _fmt(x: float | None) -> str:
    if x is None:
        return "-"
    if x == float("inf"):
        return "inf"
    return f"{x:.6g}"


def format_table(rows: Sequence[Row]) -> str:
    """Fixed-width table with the five metrics plus skip reasons."""
    head = (
        f"{'leg':<7} {'surf':<7} {'kind':<10} {'source':<10} "
        f"{'M':>6} {'K':>6} {'N':>3} {'status':<6} "
        f"{'max_abs':>10} {'mean_abs':>10} {'rmse':>10} {'cosine':>9} {'rel':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        m = r.metrics
        lines.append(
            f"{r.leg:<7} {r.surface:<7} {r.kind:<10} {r.source:<10} "
            f"{r.M:6d} {r.K:6d} {r.N:3d} {r.status:<6} "
            f"{_fmt(None if m is None else m.max_abs):>10} "
            f"{_fmt(None if m is None else m.mean_abs):>10} "
            f"{_fmt(None if m is None else m.rmse):>10} "
            f"{_fmt(None if m is None else m.cosine):>9} "
            f"{_fmt(None if m is None else m.rel):>10}"
        )
        if r.skip_reason:
            tag = "skip" if r.status == "skip" else r.status
            lines.append(f"        {tag}: {r.skip_reason}")
    return "\n".join(lines)


def write_csv(path: str | Path, rows: Sequence[Row]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "leg",
                "surface",
                "kind",
                "source",
                "M",
                "K",
                "N",
                "status",
                "max_abs",
                "mean_abs",
                "rmse",
                "cosine",
                "rel",
                "skip_reason",
            ]
        )
        for r in rows:
            m = r.metrics
            w.writerow(
                [
                    r.leg,
                    r.surface,
                    r.kind,
                    r.source,
                    r.M,
                    r.K,
                    r.N,
                    r.status,
                    "" if m is None else m.max_abs,
                    "" if m is None else m.mean_abs,
                    "" if m is None else m.rmse,
                    "" if m is None else m.cosine,
                    "" if m is None else m.rel,
                    r.skip_reason,
                ]
            )


def _parse_ns(text: str, *, max_n: int = LIVE_MAX_N) -> list[int]:
    ns = [int(p) for p in str(text).split(",") if p.strip()]
    if not ns:
        raise ValueError("need at least one N")
    for n in ns:
        if not 1 <= n <= int(max_n):
            raise ValueError(
                f"N={n} not in 1..{max_n} (live TokenLoop chunks above "
                f"{LIVE_MAX_N}; pass --plan-n for the 17..{PLAN_MAX_N} oracle)"
            )
    return ns


def _parse_kinds(text: str | None) -> list[tuple[str, int, int]]:
    catalog = {name: (name, m, k) for name, m, k in QWEN25_3B_LINEARS}
    if not text:
        return list(QWEN25_3B_LINEARS)
    out = []
    for raw in text.split(","):
        name = raw.strip()
        if name not in catalog:
            raise ValueError(f"unknown kind {name!r}; want {', '.join(catalog)}")
        out.append(catalog[name])
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU NF4 numerics: quantization vs kernel, never mixed"
    )
    p.add_argument("--tiny", action="store_true", help="cap M at 64 (CPU tests / smoke)")
    p.add_argument("--max-rows", type=int, default=0, help="cap M; 0 = full 3B rows")
    p.add_argument(
        "--n",
        default="1",
        help=f"comma-separated N (default 1; live 1..{LIVE_MAX_N}, --plan-n up to {PLAN_MAX_N})",
    )
    p.add_argument("--kinds", default="", help="comma-separated slots; default all 7")
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--row-chunk", type=int, default=256)
    p.add_argument("--layer", type=int, default=0, help="layer index for --chr / --model-dir")
    p.add_argument("--chr", default="", help="optional CHR0 path (CPU decode, no GPU)")
    p.add_argument("--model-dir", default="", help="optional HF dir for original BF16 W")
    p.add_argument("--no-random", action="store_true", help="skip synthetic matrices")
    p.add_argument(
        "--cuda",
        action="store_true",
        help="kernel leg: import chr_nf4_gemm only if nvidia-smi says free",
    )
    p.add_argument(
        "--cuda-force",
        action="store_true",
        help="after the lab: run --cuda even if smi still looks high (Windows sticky reserved)",
    )
    p.add_argument(
        "--plan-n",
        action="store_true",
        help=f"allow N up to {PLAN_MAX_N} for the kernel oracle (does not change TokenLoop)",
    )
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--csv", default="", help="optional CSV path (does not write docs/runs/qwen25-3b)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    max_n = PLAN_MAX_N if args.plan_n else LIVE_MAX_N
    ns = _parse_ns(args.n, max_n=max_n)
    kinds = _parse_kinds(args.kinds or None)
    max_rows = 64 if args.tiny and args.max_rows <= 0 else int(args.max_rows)
    kernel_reason = cuda_launch_reason(
        want_cuda=bool(args.cuda or args.cuda_force),
        force=bool(args.cuda_force),
        index=int(args.gpu_index),
    )
    print(LEGEND, end="")
    if kernel_reason:
        print(f"kernel leg: SKIP ({kernel_reason})")
    else:
        print("kernel leg: CUDA chr_nf4_gemm vs CPU NF4 GEMM")
        if max_n > LIVE_MAX_N:
            print(f"plan-n oracle: N up to {max_n}; TokenLoop still chunks at {LIVE_MAX_N}")
    print()

    rows: list[Row] = []
    if not args.no_random:
        rows.extend(
            run_random(
                kinds,
                ns=ns,
                seed=int(args.seed),
                max_rows=max_rows,
                row_chunk=int(args.row_chunk),
                kernel_reason=kernel_reason,
                max_n=max_n,
            )
        )
    chr_path = args.chr.strip() or None
    model_dir = args.model_dir.strip() or None
    if chr_path or model_dir:
        rows.extend(
            run_real(
                kinds,
                ns=ns,
                seed=int(args.seed),
                layer=int(args.layer),
                chr_path=chr_path,
                model_dir=model_dir,
                max_rows=max_rows,
                row_chunk=int(args.row_chunk),
                kernel_reason=kernel_reason,
                max_n=max_n,
            )
        )
    print(format_table(rows))
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nwrote {args.csv}")
    n_ok = sum(1 for r in rows if r.status == "ok")
    n_skip = sum(1 for r in rows if r.status == "skip")
    n_fail = sum(1 for r in rows if r.status == "fail")
    print(
        f"\nok={n_ok} skip={n_skip} fail={n_fail}  "
        "golden SS9.4 mae/rmse still in gpu.tests.nf4_oracle"
    )
    if not rows:
        return 2
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
