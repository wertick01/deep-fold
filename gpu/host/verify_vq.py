"""Acceptance for the VQ 2x8 half of ``gpu/host`` (wave 3).

    python gpu/host/verify_vq.py

Checks (docs/spec/vq.md, plus host invariants):

  V1  golden 1x16 layout: reconstruct == vq.md §6.2/§7.3 by value, and the
      kernel agrees with it
  V2  synthetic CompressedVqLinear == CPU ``C1[i1] + C2[i2] @ x``, maxabs <= 0.05
      (the required check; run at the real gate_proj shape 11008x2048)
  V3  K not a multiple of 8: junk in the pad columns cannot reach y
  V4  N != 1 raises instead of computing one column
  V5  no BF16 ``W`` [M, K]: ``weight`` allocates nothing and the resident bytes
      are 2 bits/weight plus one 8 KiB book
  V6  CHR0 round trip on a tiny ``chr.exe --codec vq`` file: the bytes in VRAM
      are the bytes on disk, and ``load_chr_vq`` -> kernel matches the oracle
  V7  a real ``.vq2.chr`` if one exists: one gate_proj, same threshold (bonus)

Exit code: 0 PASS, 2 FAIL, 3 BLOCKED (a required check could not run).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpu.chr0 import load_header  # noqa: E402
from gpu.chr0._fixtures import write_safetensors  # noqa: E402
from gpu.host.vq_blobs import (  # noqa: E402
    CODEBOOK_NBYTES,
    CODEBOOK_SIZE,
    N_CODEBOOKS,
    VQ_GROUP_SIZE,
    VqMatrix,
    iter_vq,
    k_pad_vq,
    materialize_vq,
    reconstruct_vq,
)
from gpu.host.vq_linear import CompressedVqLinear, load_chr_vq  # noqa: E402

MIB = 1024 * 1024
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
CHR_EXE = os.path.join(REPO, "chr.exe")
MODELS_DIR = r"C:\dev\models"
# Real Qwen2.5-3B gate_proj: [11008, 2048]. The synthetic case runs at that
# shape so the numbers mean something next to the NF4 H1 check.
GATE_M, GATE_K = 11008, 2048
GATE = "model.layers.0.mlp.gate_proj"


@dataclass
class Check:
    ident: str
    what: str
    status: str
    detail: str = ""
    required: bool = True


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def synthetic_vq(
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    *,
    seed: int,
    scale: float = 0.02,
    device: str = "cuda",
    name: str = "synthetic",
) -> VqMatrix:
    """A random but LLM-shaped VQ matrix: book ~ N(0, scale), indices uniform.

    ``scale=0.02`` puts ``C1[i1] + C2[i2]`` at the std of a real Qwen2.5
    projection, which is what makes the 0.05 maxabs threshold a statement about
    the kernel rather than about how big the numbers happen to be.
    """
    g = torch.Generator().manual_seed(seed)
    book = (torch.randn(N_CODEBOOKS, CODEBOOK_SIZE, VQ_GROUP_SIZE, generator=g) * scale)
    K_pad = k_pad_vq(K)
    G = K_pad // VQ_GROUP_SIZE
    index = torch.randint(0, 256, (M, G, N_CODEBOOKS), generator=g, dtype=torch.uint8)
    dev = torch.device(device)
    return VqMatrix(
        name=name,
        M=M,
        K=K,
        K_pad=K_pad,
        index=index.to(dev),
        book=book.to(torch.float16).to(dev),
    )


def golden_1x16() -> tuple[VqMatrix, torch.Tensor]:
    """vq.md §6.2 indices ``00 07 01 00`` over the §7.3 hand-written book."""
    book = torch.zeros(N_CODEBOOKS, CODEBOOK_SIZE, VQ_GROUP_SIZE, dtype=torch.float16)
    book[0, 0, 0] = 1.0
    book[0, 1, 1] = 1.0
    book[1, 7, 2] = 0.5
    index = torch.tensor([[[0, 7], [1, 0]]], dtype=torch.uint8)
    want = torch.tensor(
        [[1, 0, 0.5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32
    )
    return VqMatrix(name="golden", M=1, K=16, K_pad=16, index=index, book=book), want


def rand_x(K: int, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: N803
    """``[1, 1, K]`` bf16 activations with rms ~ 1, plus the host copy."""
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn((1, 1, K), generator=g, dtype=torch.float32).to(torch.bfloat16)
    return ref.to(device), ref


def oracle_y(matrix: VqMatrix, x_ref: torch.Tensor) -> np.ndarray:
    """``reconstruct_vq(...) @ x`` on the host, accumulated in float64."""
    w = reconstruct_vq(
        matrix.index.cpu(), matrix.book.cpu(), matrix.M, matrix.K, matrix.K_pad
    ).numpy()
    xh = x_ref.to(torch.float32).numpy().reshape(matrix.K, 1)
    return w.astype(np.float64) @ xh.astype(np.float64)


def err_stats(y: torch.Tensor, y_ref: np.ndarray) -> tuple[float, float]:
    got = y.to(torch.float32).cpu().numpy().reshape(-1).astype(np.float64)
    err = got - y_ref.reshape(-1)
    return float(np.abs(err).max()), float(np.sqrt((err * err).mean()))


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


class Verify:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.checks: list[Check] = []
        self.notes: dict[str, object] = {}

    def add(self, ident, what, ok, detail="", required=True) -> None:  # noqa: ANN001
        self.checks.append(
            Check(ident, what, SKIP if ok is None else (PASS if ok else FAIL), detail, required)
        )

    # --- V1 ---------------------------------------------------------------
    def v1_golden(self) -> None:
        matrix, want = golden_1x16()
        got = reconstruct_vq(matrix.index, matrix.book, 1, 16, 16)
        exact = bool(torch.equal(got, want))
        self.add(
            "V1",
            "golden 1x16 reconstruct == vq.md §6.2/§7.3",
            exact,
            f"bytes 00 07 01 00 -> {got.reshape(-1).tolist()} (bit-exact={exact})",
        )
        dev = synthetic_to_device(matrix, self.args.device)
        layer = attach(dev)
        x, x_ref = rand_x(16, self.args.seed, self.args.device)
        y = layer(x)
        torch.cuda.synchronize()
        maxabs, rmse = err_stats(y, oracle_y(dev, x_ref))
        self.add(
            "V1b",
            "kernel agrees with the golden reconstruct",
            maxabs <= self.args.tol,
            f"y{tuple(y.shape)} {y.dtype} maxabs={maxabs:.3e} rmse={rmse:.3e} tol={self.args.tol}",
        )

    # --- V2 / V3 -----------------------------------------------------------
    def v2_synthetic(self) -> None:
        cases = [
            ("V2", f"synthetic {GATE_M}x{GATE_K} (gate_proj shape)", GATE_M, GATE_K, True),
            ("V2b", "synthetic 2048x11008 (down_proj shape)", 2048, 11008, True),
            ("V2c", "synthetic 128x256 (toy)", 128, 256, True),
        ]
        for ident, what, m, k, required in cases:
            self.one_synthetic(ident, what, m, k, required=required)

    def one_synthetic(self, ident, what, m, k, *, required=True) -> None:  # noqa: ANN001
        matrix = synthetic_vq(m, k, seed=self.args.seed + m + k, device=self.args.device)
        layer = attach(matrix)
        x, x_ref = rand_x(k, self.args.seed + k, self.args.device)
        y = layer(x)
        torch.cuda.synchronize()
        ref = oracle_y(matrix, x_ref)
        maxabs, rmse = err_stats(y, ref)
        shape_ok = tuple(y.shape) == (1, 1, m) and y.dtype is torch.bfloat16
        self.add(
            ident,
            what,
            maxabs <= self.args.tol and shape_ok,
            f"[{m},{k}] K_pad={matrix.K_pad} |y|max={np.abs(ref).max():.4f} "
            f"y{tuple(y.shape)} {y.dtype} maxabs={maxabs:.6f} rmse={rmse:.6f} "
            f"tol={self.args.tol}",
            required=required,
        )
        if ident == "V2":
            self.notes["gate_maxabs"] = maxabs
            self.notes["gate_rmse"] = rmse
            self.v4_prefill(layer, k)
            self.v5_no_bf16_w(layer, matrix, x)

    def v3_tails(self) -> None:
        """K % 8 != 0: the pad group is stored (vq.md §6.3) but must not reach y."""
        m, k = 130, 65
        matrix = synthetic_vq(m, k, seed=self.args.seed + 3, device=self.args.device)
        if matrix.K_pad != 72:
            self.add("V3", "K=65 pads to 72", False, f"K_pad={matrix.K_pad}")
            return
        layer = attach(matrix)
        x, x_ref = rand_x(k, self.args.seed + 5, self.args.device)
        y_before = layer(x)
        torch.cuda.synchronize()
        maxabs, rmse = err_stats(y_before, oracle_y(matrix, x_ref))

        # Rewrite the indices of the last (mixed) group. Columns 64..71 are pad
        # for K=65, so only k=64 stays live -- but the oracle recomputes it too,
        # and y must still match. Bytes are shared with the module on purpose.
        g_last = matrix.K_pad // VQ_GROUP_SIZE - 1
        matrix.index[:, g_last, :] = torch.randint(
            0, 256, (m, N_CODEBOOKS), dtype=torch.uint8, device=matrix.index.device
        )
        y_after = layer(x)
        torch.cuda.synchronize()
        maxabs2, _ = err_stats(y_after, oracle_y(matrix, x_ref))
        self.add(
            "V3",
            "K=65 -> K_pad=72: pad columns never reach y",
            maxabs <= self.args.tol and maxabs2 <= self.args.tol and tuple(y_after.shape) == (1, 1, m),
            f"[{m},{k}] K_pad=72 maxabs={maxabs:.6f} rmse={rmse:.6f}; "
            f"after rewriting the tail group maxabs={maxabs2:.6f}; y{tuple(y_after.shape)}",
        )

    # --- V4 ---------------------------------------------------------------
    def v4_prefill(self, layer: CompressedVqLinear, k: int) -> None:
        rows, ok = [], True
        for shape in ((1, 4, k), (2, 1, k), (8, k)):
            x = torch.zeros(shape, dtype=torch.bfloat16, device=self.args.device)
            try:
                layer(x)
                rows.append(f"{shape}: ACCEPTED")
                ok = False
            except NotImplementedError as exc:
                rows.append(f"{shape}: NotImplementedError({str(exc)[:32]}...)")
            except Exception as exc:  # noqa: BLE001
                rows.append(f"{shape}: {type(exc).__name__}")
                ok = False
        try:
            layer(torch.zeros((1, 1, k), dtype=torch.bfloat16, device=self.args.device))
            rows.append("N=1 still fine")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"N=1 broke: {type(exc).__name__}")
            ok = False
        self.add("V4", "N != 1 raises (no silent one-column prefill)", ok, "; ".join(rows))

    # --- V5 ---------------------------------------------------------------
    def v5_no_bf16_w(self, layer: CompressedVqLinear, matrix: VqMatrix, x: torch.Tensor) -> None:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        peak_before = torch.cuda.max_memory_allocated()
        shapes = set()
        for _ in range(256):
            w = layer.weight
            shapes.add((tuple(w.shape), w.numel(), str(w.dtype), w.device.type))
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - before
        peak_delta = torch.cuda.max_memory_allocated() - peak_before
        want_bytes = matrix.M * matrix.G * N_CODEBOOKS + CODEBOOK_NBYTES
        bf16_bytes = matrix.M * matrix.K * 2
        ok = (
            shapes == {((0,), 0, "torch.bfloat16", "cuda")}
            and delta == 0
            and peak_delta == 0
            and layer.nbytes == want_bytes
        )
        self.add(
            "V5",
            "no BF16 [M, K]: weight allocates nothing",
            ok,
            f"256 reads -> {sorted(shapes)}; alloc delta={delta} B peak delta={peak_delta} B; "
            f"resident {layer.nbytes / MIB:.2f} MiB = {8 * layer.nbytes / (matrix.M * matrix.K):.3f} "
            f"bits/weight (BF16 W would be {bf16_bytes / MIB:.1f} MiB)",
        )
        try:
            layer.weight = torch.zeros(1)
            self.add("V5b", "weight is not assignable", False, "assignment silently accepted")
        except AttributeError:
            self.add("V5b", "weight is not assignable", True, "AttributeError, as intended")

        # A forward must not grow the allocator either: no dequantized W, no
        # per-call scratch that scales with M*K.
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        for _ in range(8):
            layer(x)
        torch.cuda.synchronize()
        grew = torch.cuda.memory_allocated() - base
        self.add(
            "V5c",
            "8 forwards do not dequantize W into HBM",
            grew < matrix.M * 4,
            f"allocator grew {grew} B over 8 calls (one BF16 W = {bf16_bytes} B)",
        )

    # --- V6 ---------------------------------------------------------------
    def v6_chr_roundtrip(self) -> None:
        ident, what = "V6", "CHR0 round trip: chr.exe --codec vq -> materialize_vq"
        if not os.path.isfile(CHR_EXE):
            self.add(ident, what, None, f"{CHR_EXE} missing; go build -o chr.exe ./cmd/chr")
            return
        try:
            path = build_vq_fixture()
        except Exception as exc:  # noqa: BLE001
            self.add(ident, what, False, f"{type(exc).__name__}: {exc}")
            return

        hdr = load_header(path)
        names = list(iter_vq(hdr))
        if not names:
            self.add(ident, what, False, f"fixture has no codec=vq tensors: {list(hdr.tensors)}")
            return

        details, ok = [], True
        for name in names:
            info = hdr.tensor(name)
            matrix = materialize_vq(path, name, self.args.device, header=hdr)
            with open(path, "rb") as f:
                f.seek(info.blobs["index"].start)
                raw_index = f.read(info.blobs["index"].nbytes)
                f.seek(info.blobs["codebook"].start)
                raw_book = f.read(info.blobs["codebook"].nbytes)
            same_index = bytes(matrix.index.reshape(-1).cpu().numpy().tobytes()) == raw_index
            same_book = bytes(matrix.book.reshape(-1).cpu().view(torch.uint8).numpy().tobytes()) == raw_book
            ok = ok and same_index and same_book
            details.append(
                f"{name} [{info.M},{info.K}] K_pad={info.K_pad} "
                f"index {len(raw_index)}B bit-exact={same_index} book bit-exact={same_book}"
            )
        self.add(ident, what, ok, "; ".join(details))

        name = max(names, key=lambda n: hdr.tensor(n).M * hdr.tensor(n).K)
        info = hdr.tensor(name)
        layer = load_chr_vq(path, name, self.args.device, header=hdr)
        x, x_ref = rand_x(info.K, self.args.seed + 11, self.args.device)
        y = layer(x)
        torch.cuda.synchronize()
        matrix = VqMatrix(
            name=name, M=info.M, K=info.K, K_pad=info.K_pad,
            index=layer.index, book=layer.book,
        )
        maxabs, rmse = err_stats(y, oracle_y(matrix, x_ref))
        self.add(
            "V6b",
            "load_chr_vq -> kernel == CPU reconstruct",
            maxabs <= self.args.tol,
            f"{name} [{info.M},{info.K}] maxabs={maxabs:.6f} rmse={rmse:.6f} "
            f"tol={self.args.tol}; {layer.nbytes} B resident",
        )

    # --- V7 ---------------------------------------------------------------
    def v7_real(self) -> None:
        ident, what = "V7", "real .vq2.chr: one projection vs CPU reconstruct"
        path = self.args.chr_path
        if not path or not os.path.isfile(path):
            self.add(ident, what, None, f"no {path} yet (compress still running?)", required=False)
            return
        try:
            hdr = load_header(path)
        except Exception as exc:  # noqa: BLE001
            self.add(ident, what, False, f"header: {type(exc).__name__}: {exc}", required=False)
            return
        names = list(iter_vq(hdr))
        if not names:
            self.add(ident, what, None, f"{path}: no codec=vq tensors", required=False)
            return
        name = self.args.name if self.args.name in hdr.tensors else names[0]
        info = hdr.tensor(name)
        layer = load_chr_vq(path, name, self.args.device, header=hdr)
        x, x_ref = rand_x(info.K, self.args.seed, self.args.device)
        y = layer(x)
        torch.cuda.synchronize()
        matrix = VqMatrix(
            name=name, M=info.M, K=info.K, K_pad=info.K_pad,
            index=layer.index, book=layer.book,
        )
        maxabs, rmse = err_stats(y, oracle_y(matrix, x_ref))
        self.notes["real_file"] = f"{path} ({os.path.getsize(path) / MIB:.1f} MiB)"
        self.notes["real_maxabs"] = maxabs
        self.add(
            ident,
            what,
            maxabs <= self.args.tol,
            f"{name} [{info.M},{info.K}] K_pad={info.K_pad} maxabs={maxabs:.6f} "
            f"rmse={rmse:.6f} tol={self.args.tol}; {len(names)} vq tensors in the file; "
            f"{layer.nbytes / MIB:.2f} MiB resident vs {info.M * info.K * 2 / MIB:.1f} MiB BF16",
            required=False,
        )

    # --- driver ------------------------------------------------------------
    def run(self) -> int:
        if not torch.cuda.is_available():
            self.add("GPU", "cuda available", False, "torch.cuda.is_available() is False")
            return self.report()
        cap = "".join(str(v) for v in torch.cuda.get_device_capability(0))
        self.add(
            "V-", "environment", True,
            f"{torch.cuda.get_device_name(0)} sm_{cap} torch={torch.__version__}",
        )
        for step in (self.v1_golden, self.v2_synthetic, self.v3_tails, self.v6_chr_roundtrip, self.v7_real):
            try:
                step()
            except Exception as exc:  # noqa: BLE001
                self.add(step.__name__, f"{step.__name__} raised", False, f"{type(exc).__name__}: {exc}")
        return self.report()

    def report(self) -> int:
        print("")
        print("=" * 78)
        print("deep-fold wave 3 -- gpu/host VQ 2x8 acceptance (agent 8)")
        print("=" * 78)
        print(f"chr : {self.args.chr_path}")
        print("")
        print(f"{'ID':<5} {'STATUS':<6} {'CHECK':<46} DETAIL")
        print("-" * 78)
        for c in self.checks:
            print(f"{c.ident:<5} {c.status:<6} {c.what:<46} {c.detail}")
        print("-" * 78)
        for key, value in self.notes.items():
            print(f"{key:<16}: {value}")
        counts = {s: sum(1 for c in self.checks if c.status == s) for s in (PASS, FAIL, SKIP)}
        verdict = (
            "FAIL"
            if any(c.status == FAIL and c.required for c in self.checks)
            else "BLOCKED"
            if any(c.status == SKIP and c.required for c in self.checks)
            else "PASS"
        )
        soft = [c.ident for c in self.checks if c.status == FAIL and not c.required]
        print("")
        print(
            f"{verdict}  (pass={counts[PASS]} fail={counts[FAIL]} skip={counts[SKIP]})"
            + (f"   non-blocking failures: {soft}" if soft else "")
        )
        return {"PASS": 0, "FAIL": 2, "BLOCKED": 3}[verdict]


def synthetic_to_device(matrix: VqMatrix, device: str) -> VqMatrix:
    dev = torch.device(device)
    return VqMatrix(
        name=matrix.name, M=matrix.M, K=matrix.K, K_pad=matrix.K_pad,
        index=matrix.index.to(dev), book=matrix.book.to(dev),
    )


def attach(matrix: VqMatrix) -> CompressedVqLinear:
    layer = CompressedVqLinear(in_features=matrix.K, out_features=matrix.M, bias=False)
    layer.attach(matrix)
    return layer


_FIXTURE: str | None = None


def build_vq_fixture() -> str:
    """A tiny two-matrix ``codec=vq`` CHR0 file, written by the Go codec.

    64x128 (aligned) and 2x65 (K_pad=72, so the padding rule of vq.md §6.3 is
    exercised on a file the Python side did not write). Weights are scaled to
    the magnitude of a real projection so the bf16 output of the kernel is
    compared at a realistic dynamic range.
    """
    global _FIXTURE
    if _FIXTURE is not None:
        return _FIXTURE
    tmp = os.path.join(tempfile.gettempdir(), "deep-fold-vq-fixture")
    src = os.path.join(tmp, "toy")
    os.makedirs(src, exist_ok=True)

    def ramp(*shape: int) -> np.ndarray:
        n = int(np.prod(shape))
        return ((((np.arange(n, dtype=np.float32) * 7) % 37.0) - 18.0) / 340.0).reshape(shape)

    write_safetensors(
        os.path.join(src, "model.safetensors"),
        {
            "model.layers.0.mlp.gate_proj.weight": ramp(64, 128),
            "model.layers.0.mlp.down_proj.weight": ramp(2, 65),
            "model.layers.0.input_layernorm.weight": ramp(128),
        },
    )
    out = os.path.join(tmp, "toy.vq2.chr")
    proc = subprocess.run(
        [
            CHR_EXE, "compress",
            "--in", src,
            "--out", out,
            "--codec", "vq",
            "--seed", "0",
            "--iters", "20",
            "--arch", "toy",
            "--hidden-size", "128",
            "--intermediate-size", "65",
            "--num-layers", "1",
            "--vocab-size", "0",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"chr compress failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    _FIXTURE = out
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gpu/host VQ acceptance")
    p.add_argument("--chr", dest="chr_path", default=os.path.join(MODELS_DIR, "qwen25-3b.vq2.chr"))
    p.add_argument("--name", default=GATE, help="CHR0 tensor for the V7 oracle")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tol", type=float, default=0.05, help="floor-1 maxabs threshold")
    p.add_argument("--seed", type=int, default=20260912)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return Verify(parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
