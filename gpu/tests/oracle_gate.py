"""Wave-2 floor-1 gate: GPU NF4 linear vs CPU decode of OUR codec, plus VRAM safety.

    python gpu/tests/oracle_gate.py --chr C:\\dev\\models\\qwen25-3b.nf4.chr

Protocol, thresholds and what a FAIL means: docs/spec/gpu-safety.md.

What this gate does NOT do (docs/spec/gpu-safety.md):
  * no comparison against original BF16 safetensors (that is floor 2, KL);
  * no `chr decode` of the 3B model (the F32 dump is ~12 GiB -- forbidden here);
  * no kernel tuning, no generate, no WikiText.

Exit codes (like `chr verify`): 0 = PASS, 2 = FAIL, 3 = BLOCKED (a required
check could not run, e.g. gpu/nf4 is not built yet).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import backend  # noqa: E402
import chr0_min  # noqa: E402
import nf4_oracle as oracle  # noqa: E402
import gpu_probe as probe  # noqa: E402

MIB = 1024 * 1024
DEFAULT_NAME = "model.layers.0.mlp.gate_proj"
# S8: the gate must never materialize the whole model. A crosscheck against an
# already existing F32 dump may read at most one tensor slice.
CROSSCHECK_MAX_BYTES = 256 * MIB
PROCESS_SPAWN_ATTRS = {"system", "popen", "startfile", "spawnl", "spawnv", "spawnvp", "execv", "execvp"}
# torch reserves large device blocks in ~20 MiB segments; below that, an smi
# delta says nothing about whether a tensor of a given size was allocated.
SMI_RESOLUTION_MIB = 24.0

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Result:
    ident: str
    what: str
    status: str
    detail: str = ""
    required: bool = True


@dataclass
class Numbers:
    maxabs: float | None = None
    rmse: float | None = None
    metric_source: str = "n/a"
    smi_boot_mib: int | None = None
    smi_ctx_mib: int | None = None
    smi_before_mib: int | None = None
    smi_after_mib: int | None = None
    smi_delta_mib: int | None = None
    display_reserved_mib: float | None = None
    cuda_ctx_mib: int | None = None
    peak_alloc_delta_mib: float | None = None
    budget: dict[str, float] = field(default_factory=dict)
    toys: dict[str, dict[str, float]] = field(default_factory=dict)
    oracle_f32_vs_f64_maxabs: float | None = None
    load_s: float | None = None
    gemm_ms: float | None = None


def digest(t: Any) -> str:
    return hashlib.blake2b(t.detach().cpu().numpy().tobytes(), digest_size=16).hexdigest()


def compare(y_gpu_f32: np.ndarray, y_cpu_f32: np.ndarray) -> tuple[float, float]:
    e = (y_gpu_f32.astype(np.float64) - y_cpu_f32.astype(np.float64)).ravel()
    return float(np.abs(e).max()), float(np.sqrt((e * e).mean()))


class Gate:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: list[Result] = []
        self.num = Numbers()
        self.torch: Any = None
        self.loader = backend.probe_loader()
        self.gemm = backend.probe_gemm()

    # --- bookkeeping ---------------------------------------------------------
    def add(self, ident: str, what: str, ok: bool | None, detail: str = "", required: bool = True) -> None:
        status = SKIP if ok is None else (PASS if ok else FAIL)
        self.results.append(Result(ident, what, status, detail, required))

    def skip_all(self, idents: list[tuple[str, str]], reason: str) -> None:
        for ident, what in idents:
            self.add(ident, what, None, reason)

    # --- inputs -------------------------------------------------------------
    def make_x(self, k: int, n: int, seed: int) -> tuple[Any, np.ndarray]:
        """x: bf16 [K, N] on device (stitch-gpu.md) + the same values as f32 host.

        The oracle consumes the bf16-rounded x, so x quantization is not charged
        to the kernel: floor 1 is about the kernel, not about rounding inputs.
        """
        torch = self.torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        x_f32 = torch.randn((k, n), generator=g, dtype=torch.float32)
        x_bf16 = x_f32.to(torch.bfloat16)
        x_host = x_bf16.to(torch.float32).numpy().astype(np.float32)
        return x_bf16.contiguous().to(self.args.device), x_host

    # --- phase 1: no GPU, no agents ----------------------------------------
    def run_selfchecks(self) -> None:
        for c in oracle.selfcheck():
            self.add(c.ident, "nf4.md golden (LUT/pack/decode/encode)", c.ok, c.detail)
        ok, detail = chr0_min.selfcheck_writer()
        self.add("C0", "toy CHR0 writer vs chr0.md SS2.4 byte-exact header", ok, detail)
        self.check_s8()

    def check_s8(self) -> None:
        """S8: the gate itself must not be able to dump the whole model.

        Enforced, not merely promised: the only child process the test tree may
        spawn is nvidia-smi, so no code path can reach `chr decode`.
        """
        notes = []
        ok = True
        readme = os.path.join(HERE, "README.md")
        if os.path.isfile(readme):
            text = open(readme, "r", encoding="utf-8").read()
            has = "chr decode" in text and "forbidden" in text
            ok = ok and has
            notes.append(f"README states the prohibition={has}")
        else:
            ok = False
            notes.append("gpu/tests/README.md missing")

        spawn_sites: list[str] = []
        importers: list[str] = []
        for fn in sorted(f for f in os.listdir(HERE) if f.endswith(".py")):
            src = open(os.path.join(HERE, fn), "r", encoding="utf-8").read()
            for node in ast.walk(ast.parse(src, filename=fn)):
                if isinstance(node, ast.Import) and any(a.name == "subprocess" for a in node.names):
                    importers.append(fn)
                elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
                    importers.append(fn)
                elif isinstance(node, ast.Call):
                    f = node.func
                    hit = None
                    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                        if f.value.id == "subprocess":
                            hit = f"subprocess.{f.attr}"
                        elif f.value.id == "os" and f.attr in PROCESS_SPAWN_ATTRS:
                            hit = f"os.{f.attr}"
                    elif isinstance(f, ast.Name) and f.id in ("Popen", "run", "check_output"):
                        hit = f.id
                    if hit:
                        spawn_sites.append(f"{fn}:{node.lineno}:{hit}")
        rogue_import = sorted({fn for fn in importers if fn != "gpu_probe.py"})
        rogue_spawn = [s for s in spawn_sites if s.split(":")[0] != "gpu_probe.py"]
        smi_only = "nvidia-smi" in open(os.path.join(HERE, "gpu_probe.py"), "r", encoding="utf-8").read()
        ok = ok and not rogue_import and not rogue_spawn and smi_only
        notes.append(
            f"child processes: only gpu_probe.py spawns (nvidia-smi={smi_only}), "
            f"rogue imports={rogue_import or 'none'} rogue spawns={rogue_spawn or 'none'}"
        )
        notes.append(f"crosscheck read cap={CROSSCHECK_MAX_BYTES // MIB} MiB")
        self.add("S8", "no full-model decode inside the gate", ok, "; ".join(notes))

    # --- torch / device -----------------------------------------------------
    def setup_torch(self) -> bool:
        self.num.smi_boot_mib = self.try_smi()
        try:
            import torch
        except Exception as exc:
            self.add("GPU", "torch import", False, f"{type(exc).__name__}: {exc}")
            return False
        self.torch = torch
        if self.args.device.startswith("cuda") and not torch.cuda.is_available():
            self.add("GPU", "cuda available", False, "torch.cuda.is_available() is False")
            return False
        if self.args.device.startswith("cuda"):
            torch.zeros(1, device=self.args.device).sum().item()  # create the context
            torch.cuda.synchronize()
            self.num.smi_ctx_mib = self.try_smi()
            alloc_mib = torch.cuda.memory_allocated() / MIB
            if self.num.smi_ctx_mib is not None:
                self.num.display_reserved_mib = round(self.num.smi_ctx_mib - alloc_mib, 2)
                if self.num.smi_boot_mib is not None:
                    self.num.cuda_ctx_mib = self.num.smi_ctx_mib - self.num.smi_boot_mib
            name = torch.cuda.get_device_name(0)
            cap = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
            self.add(
                "S7",
                "display/context VRAM logged",
                self.num.display_reserved_mib is not None,
                f"gpu={name} sm_{cap.replace('.', '')} torch={torch.__version__} "
                f"smi_boot={self.num.smi_boot_mib} smi_after_ctx={self.num.smi_ctx_mib} "
                f"torch_allocated_mib={alloc_mib:.2f} display_reserved_mib={self.num.display_reserved_mib}",
            )
        return True

    def try_smi(self) -> int | None:
        try:
            return probe.smi_used_mib(self.args.gpu_index)
        except Exception:
            return None

    # --- phase 2: toys ------------------------------------------------------
    def warmup_kernel(self) -> bool:
        """Load/JIT the extension before any measured check.

        Without this, the first real check pays for the build and can trip over a
        cold or concurrently rebuilt JIT cache (this box has no ninja, so the
        module must already be built). Retried, then reported as K0.
        """
        torch = self.torch
        packed, scale = oracle.toy_nf4(64, 64, seed=1, pad_garbage=False)
        w = backend.make_matrix("warmup", packed, scale, 64, 64, self.args.device)
        x, _ = self.make_x(64, 1, self.args.seed)
        last = ""
        for attempt in range(3):
            try:
                self.gemm.obj(w, x, 1)
                torch.cuda.synchronize()
                self.add("K0", "kernel loads and launches", True, f"{self.gemm.detail}, attempt {attempt + 1}")
                self.check_binary_fresh()
                return True
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(2.0)
        self.add("K0", "kernel loads and launches", False, last)
        return False

    def check_binary_fresh(self) -> None:
        """K1: the loaded kernel binary must be newer than gpu/nf4 sources.

        torch's JIT caches build versions per process: when a rebuild fails, a
        retry can quietly load the previous .pyd, and every number below would
        then describe code nobody wrote. Rebuild with MSVC + ninja on PATH:
        `vcvars64.bat` then `python gpu/nf4/setup.py build_ext --inplace`.
        """
        binary, sources = backend.loaded_kernel_binary()
        if binary is None or not os.path.isfile(binary):
            self.add("K1", "kernel binary newer than its sources", None, f"binary not identified ({binary})")
            return
        b = os.path.getmtime(binary)
        newer = [(os.path.basename(s), os.path.getmtime(s)) for s in sources if os.path.getmtime(s) > b]
        stamp = lambda t: time.strftime("%H:%M:%S", time.localtime(t))  # noqa: E731
        self.add(
            "K1",
            "kernel binary newer than its sources",
            not newer,
            f"{os.path.basename(binary)} built {stamp(b)}; "
            + (
                "STALE, rebuild needed: " + ", ".join(f"{n} touched {stamp(t)}" for n, t in newer)
                if newer
                else "all of " + ", ".join(os.path.basename(s) for s in sources) + " are older"
            ),
        )

    def run_toys(self) -> None:
        toys = [("F1", 128, 256), ("F2", 130, 65)]
        if not self.gemm.ok or self.torch is None:
            reason = f"gpu.nf4 not importable: {self.gemm.detail}" if self.torch else "no torch/cuda"
            self.add("K0", "kernel loads and launches", None, reason)
            self.add("K1", "kernel binary newer than its sources", None, reason)
            self.skip_all([(i, f"toy {m}x{k} vs CPU oracle") for i, m, k in toys], reason)
            return
        if not self.warmup_kernel():
            self.add("K1", "kernel binary newer than its sources", None, "kernel never loaded")
            self.gemm = backend.Probe(False, self.gemm.obj, self.gemm.detail + " (launch failed, see K0)")
            self.skip_all([(i, f"toy {m}x{k} vs CPU oracle") for i, m, k in toys], "kernel launch failed (K0)")
            return
        for ident, m, k in toys:
            self.run_toy(ident, m, k)

    def run_toy(self, ident: str, m: int, k: int) -> None:
        torch = self.torch
        kp = oracle.k_pad(k)
        n_groups = kp // 64
        rng = np.random.default_rng(self.args.seed + m + k)
        nib = rng.integers(0, 16, size=(m, kp), dtype=np.uint8)
        scale = oracle.toy_scale(m, n_groups, rng)

        # Padding columns get garbage on purpose: a real .chr writes nibble 7
        # there, so garbage is the harder case. Variant B differs only in pad.
        nib_a = nib.copy()
        nib_b = nib.copy()
        if kp > k:
            nib_a[:, k:] = rng.integers(0, 16, size=(m, kp - k), dtype=np.uint8)
            nib_b[:, k:] = 7
        packed_a = oracle.pack_nibbles(nib_a)
        packed_b = oracle.pack_nibbles(nib_b)

        w_hat = oracle.decode_nf4(packed_a, scale, m, k)
        x_dev, x_host = self.make_x(k, 1, self.args.seed + 1)
        y_cpu = oracle.matmul_f32(w_hat, x_host)

        try:
            w_a = backend.make_matrix(f"toy_{ident}_a", packed_a, scale, m, k, self.args.device)
            y_gpu = self.gemm.obj(w_a, x_dev, 1).clone()   # clone: y may be a reused buffer
            torch.cuda.synchronize()
        except Exception as exc:
            self.add(ident, f"toy {m}x{k} vs CPU oracle", False, f"{type(exc).__name__}: {exc}")
            return

        y_host = y_gpu.to(torch.float32).cpu().numpy()
        maxabs, rmse = compare(y_host, y_cpu)
        self.num.toys[ident] = {"maxabs": maxabs, "rmse": rmse, "M": m, "K": k, "K_pad": kp}
        ok = maxabs <= self.args.tol
        detail = f"M={m} K={k} K_pad={kp} N=1 maxabs={maxabs:.6f} rmse={rmse:.6f} tol={self.args.tol}"

        if kp > k:  # pad must not reach y: same logical K, different pad nibbles
            w_b = backend.make_matrix(f"toy_{ident}_b", packed_b, scale, m, k, self.args.device)
            y_b = self.gemm.obj(w_b, x_dev, 1)
            torch.cuda.synchronize()
            same = bool(torch.equal(y_gpu, y_b))
            maxabs_b, _ = compare(y_b.to(torch.float32).cpu().numpy(), y_cpu)
            ok = ok and same and maxabs_b <= self.args.tol
            detail += f"; pad-garbage vs pad-7 y bit-identical={same} maxabs_pad7={maxabs_b:.6f}"

        self.add(ident, f"toy {m}x{k} vs CPU oracle", ok, detail)

    # --- phase 3: the real matrix ------------------------------------------
    def run_3b(self) -> None:
        idents = [
            ("F3", "3B gate_proj vs CPU oracle"),
            ("F4", "second x seed, packed untouched"),
            ("F5", "two identical calls bit-stable"),
            ("X1", "loader device bytes == .chr bytes"),
            ("S1", "smi delta << BF16 W"),
            ("S2", "no [M,K] fp16/bf16/fp32 dequant W"),
            ("S3", ".chr read-only, mtime unchanged"),
            ("S6", "packed immutable after HtoD"),
        ]
        if not self.args.chr_path:
            self.skip_all(idents, "no --chr given")
            return
        if not os.path.isfile(self.args.chr_path):
            self.skip_all(idents, f"missing file: {self.args.chr_path}")
            return
        if self.torch is None:
            self.skip_all(idents, "no torch/cuda")
            return
        if not self.gemm.ok:
            # The loader half can still be judged without a kernel.
            self.skip_all(
                [(i, w) for i, w in idents if i in ("F3", "F4", "F5", "S2")],
                f"gpu.nf4 not importable: {self.gemm.detail}",
            )
            self.run_3b_loader_only()
            return
        self.run_3b_full()

    def read_reference(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        header = chr0_min.read_header(self.args.chr_path)
        packed, scale, m, k = chr0_min.read_nf4(header, self.args.name)
        return packed, scale, m, k

    def materialize(self, guard_report: list[str]) -> Any:
        """Device matrix via gpu.chr0 if present, else via our own reader."""
        packed_np, scale_np, m, k = self.read_reference()
        if self.loader.ok:
            w = self.loader.obj(self.args.chr_path, self.args.name, device=self.args.device)
            guard_report.append(f"loader={self.loader.detail}")
            same_packed = bool(
                (w.packed.detach().cpu().numpy().reshape(-1) == packed_np.reshape(-1)).all()
            )
            same_scale = bool(
                (
                    w.scale.detach().cpu().numpy().view(np.uint16).reshape(-1)
                    == scale_np.view(np.uint16).reshape(-1)
                ).all()
            )
            shape_ok = int(w.M) == m and int(w.K) == k and int(w.K_pad) == oracle.k_pad(k)
            self.add(
                "X1",
                "loader device bytes == .chr bytes",
                same_packed and same_scale and shape_ok,
                f"packed=={same_packed} scale=={same_scale} "
                f"M/K/K_pad=({w.M},{w.K},{w.K_pad}) ok={shape_ok}",
            )
            return w
        self.add("X1", "loader device bytes == .chr bytes", None, f"gpu.chr0 missing: {self.loader.detail}")
        guard_report.append("loader=gpu/tests fallback upload (gpu.chr0 missing)")
        return backend.make_matrix(self.args.name, packed_np, scale_np, m, k, self.args.device)

    def run_3b_loader_only(self) -> None:
        torch = self.torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.num.smi_before_mib = self.try_smi()
        notes: list[str] = []
        t0 = time.perf_counter()
        with probe.ReadOnlyFileGuard(self.args.chr_path) as guard:
            w = self.materialize(notes)
        self.num.load_s = round(time.perf_counter() - t0, 3)
        torch.cuda.synchronize()
        self.num.smi_after_mib = self.try_smi()
        self.judge_vram(w, notes + ["kernel absent: materialize only"])
        self.judge_file_guard(guard)
        self.add("S6", "packed immutable after HtoD", None, "no kernel call to make")

    def run_3b_full(self) -> None:
        torch = self.torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated()
        self.num.smi_before_mib = self.try_smi()

        packed_np, scale_np, m, k = self.read_reference()
        w_hat = oracle.decode_nf4(packed_np, scale_np, m, k)
        x_dev, x_host = self.make_x(k, 1, self.args.seed + 1)
        y_cpu = oracle.matmul_f32(w_hat, x_host)
        y_cpu64 = oracle.matmul_f64_chunked(w_hat, x_host)
        self.num.oracle_f32_vs_f64_maxabs = float(np.abs(y_cpu.astype(np.float64) - y_cpu64).max())

        notes: list[str] = []
        watcher = probe.DequantWatcher(min_numel=m * k // 2)
        with probe.ReadOnlyFileGuard(self.args.chr_path) as guard:
            t0 = time.perf_counter()
            with watcher:
                w = self.materialize(notes)
                torch.cuda.synchronize()
                self.num.load_s = round(time.perf_counter() - t0, 3)
                d_packed, d_scale = digest(w.packed), digest(w.scale)
                ptr_before = (w.packed.data_ptr(), w.scale.data_ptr())
                t1 = time.perf_counter()
                y1 = self.gemm.obj(w, x_dev, 1).clone()  # clone: y may be a reused buffer
                torch.cuda.synchronize()
                self.num.gemm_ms = round((time.perf_counter() - t1) * 1e3, 3)
                y2 = self.gemm.obj(w, x_dev, 1).clone()  # F5: same x, twice
                torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        self.num.peak_alloc_delta_mib = round((peak - alloc_before) / MIB, 2)
        self.num.smi_after_mib = self.try_smi()

        y_host = y1.to(torch.float32).cpu().numpy()
        maxabs, rmse = compare(y_host, y_cpu)
        self.num.maxabs, self.num.rmse = maxabs, rmse
        self.num.metric_source = f"{self.args.name} [{m},{k}] N=1"
        self.add(
            "F3",
            "3B gate_proj vs CPU oracle",
            maxabs <= self.args.tol,
            f"M={m} K={k} maxabs={maxabs:.6f} rmse={rmse:.6f} tol={self.args.tol} "
            f"oracle_f32_vs_f64_maxabs={self.num.oracle_f32_vs_f64_maxabs:.3e} "
            f"load_s={self.num.load_s} gemm_ms={self.num.gemm_ms}",
        )

        bit_stable = bool(torch.equal(y1, y2))
        self.add(
            "F5",
            "two identical calls bit-stable",
            bit_stable,
            f"y(call1) == y(call2) bitwise={bit_stable}",
        )

        x2_dev, x2_host = self.make_x(k, 1, self.args.seed + 777)
        y3 = self.gemm.obj(w, x2_dev, 1)
        torch.cuda.synchronize()
        maxabs2, rmse2 = compare(y3.to(torch.float32).cpu().numpy(), oracle.matmul_f32(w_hat, x2_host))
        same_packed = digest(w.packed) == d_packed and digest(w.scale) == d_scale
        self.add(
            "F4",
            "second x seed, packed untouched",
            maxabs2 <= self.args.tol and same_packed,
            f"maxabs={maxabs2:.6f} rmse={rmse2:.6f} packed+scale digest unchanged={same_packed}",
        )
        ptr_ok = (w.packed.data_ptr(), w.scale.data_ptr()) == ptr_before
        self.add(
            "S6",
            "packed immutable after HtoD",
            same_packed and ptr_ok and w.packed.dtype == torch.uint8,
            f"digest unchanged={same_packed} data_ptr stable={ptr_ok} "
            f"packed.dtype={w.packed.dtype} contiguous={w.packed.is_contiguous()} "
            f"requires_grad={w.packed.requires_grad}",
        )

        self.judge_vram(w, notes)
        self.judge_dequant(watcher, m, k)
        self.judge_file_guard(guard)

        if self.args.crosscheck_f32:
            self.crosscheck_f32(w_hat, m, k)

    def judge_vram(self, w: Any, notes: list[str]) -> None:
        m, k = int(w.M), int(w.K)
        budget = probe.nf4_vram_budget(m, k, 1)
        self.num.budget = {kk: round(v, 3) for kk, v in budget.items()}
        before, after = self.num.smi_before_mib, self.num.smi_after_mib
        if before is None or after is None:
            self.add("S1", "smi delta << BF16 W", None, "nvidia-smi unavailable")
            return
        delta = after - before
        self.num.smi_delta_mib = delta
        limit = budget["legit_mib"] + self.args.vram_extra_mib
        peak = self.num.peak_alloc_delta_mib
        ok = peak is None or peak <= limit

        # smi reports *reserved* VRAM, and torch reserves large blocks in ~20 MiB
        # segments, so a matrix whose BF16 W would fit inside that granularity
        # cannot be judged by smi at all: q_proj [2048,2048] shows a 22 MiB smi
        # delta while the allocator used 2.1 MiB. For those the allocator counter
        # governs; smi stays in the log. The acceptance matrix (gate_proj, BF16 W
        # = 43 MiB) is far above the granularity, so there smi is the gate.
        if budget["bf16_w_mib"] >= SMI_RESOLUTION_MIB:
            ok = ok and delta < budget["bf16_w_mib"] and delta <= limit
            verdict_src = "smi is the gate"
        else:
            verdict_src = (
                f"smi cannot resolve a BF16 W of {budget['bf16_w_mib']:.2f} MiB "
                f"(allocator segment granularity ~{SMI_RESOLUTION_MIB:.0f} MiB); "
                "torch peak governs"
            )
        self.add(
            "S1",
            "smi delta << BF16 W",
            ok,
            f"smi_before={before} smi_after={after} delta_mib={delta} "
            f"legit(packed+scale+x+y)={budget['legit_mib']:.2f} limit={limit:.2f} "
            f"bf16_W={budget['bf16_w_mib']:.2f} fp32_W={budget['fp32_w_mib']:.2f} "
            f"torch_peak_delta_mib={peak}; {verdict_src}; " + "; ".join(notes),
        )

    def judge_dequant(self, watcher: probe.DequantWatcher, m: int, k: int) -> None:
        if watcher.error:
            self.add("S2", "no [M,K] fp16/bf16/fp32 dequant W", None, f"watcher unavailable: {watcher.error}")
            return
        big = [h for h in watcher.hits if h.nbytes >= m * k * 2]
        src = self.scan_kernel_sources()
        ok = not big and not src["violations"]
        self.add(
            "S2",
            "no [M,K] fp16/bf16/fp32 dequant W",
            ok,
            f"cuda float tensors >= M*K/2 elems: {len(watcher.hits)} "
            f"(>= bf16 W size: {[(h.op, h.shape, h.dtype) for h in big]}); "
            f"source scan: {src['detail']}",
        )

    def scan_kernel_sources(self) -> dict[str, Any]:
        """Read-only scan of gpu/nf4 for on-token allocations (agent 2 invariant)."""
        root = os.path.join(REPO, "gpu", "nf4")
        if not os.path.isdir(root):
            return {"violations": [], "detail": "gpu/nf4 absent (nothing to scan)"}
        needles = ("cudaMalloc", "cudaMallocAsync", "cudaHostAlloc", "malloc(")
        violations = []
        files = 0
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith((".cu", ".cuh", ".cpp", ".h", ".hpp")):
                    continue
                files += 1
                path = os.path.join(dirpath, fn)
                for lineno, line in enumerate(open(path, "r", encoding="utf-8", errors="replace"), 1):
                    stripped = line.strip()
                    if stripped.startswith(("//", "*", "/*")):
                        continue
                    for needle in needles:
                        if needle in stripped:
                            violations.append(f"{fn}:{lineno}: {needle}")
        return {
            "violations": violations,
            "detail": f"{files} kernel sources, allocation calls: {violations or 'none'}",
        }

    def judge_file_guard(self, guard: probe.ReadOnlyFileGuard) -> None:
        self.add(
            "S3",
            ".chr read-only, mtime unchanged",
            guard.ok,
            f"open modes={guard.modes or ['(none via builtins.open)']} violations={guard.violations or 'none'}",
        )

    def crosscheck_f32(self, w_hat: np.ndarray, m: int, k: int) -> None:
        """Optional: our numpy decode vs an already existing `chr decode` F32 dump.

        Reads ONE tensor slice (capped at CROSSCHECK_MAX_BYTES). It never creates
        a dump: producing the ~12 GiB F32 dump is forbidden in this gate (S8).
        """
        path = self.args.crosscheck_f32
        try:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                head = json.loads(f.read(n).decode("utf-8"))
                info = head.get(self.args.name) or head.get(self.args.name + ".weight")
                if info is None:
                    raise KeyError(f"{self.args.name} not in {os.path.basename(path)}")
                if info["dtype"] != "F32":
                    raise ValueError(f"dtype {info['dtype']} != F32")
                start, end = info["data_offsets"]
                nbytes = end - start
                if nbytes > CROSSCHECK_MAX_BYTES:
                    raise ValueError(f"slice {nbytes} B over the {CROSSCHECK_MAX_BYTES} B cap")
                f.seek(8 + n + start)
                raw = f.read(nbytes)
            ref = np.frombuffer(raw, dtype="<f4").reshape(tuple(info["shape"]))
            if ref.shape != (m, k):
                raise ValueError(f"shape {ref.shape} != {(m, k)}")
            mine = np.ascontiguousarray(w_hat)
            bitwise = bool((ref.view(np.uint32) == mine.view(np.uint32)).all())
            maxabs = float(np.abs(ref.astype(np.float64) - mine.astype(np.float64)).max())
            self.add(
                "X2",
                "python decode == chr decode F32 dump (opt-in)",
                bitwise,
                f"bitwise equal={bitwise} maxabs={maxabs:.3e} slice={nbytes / MIB:.1f} MiB from {os.path.basename(path)}",
                required=False,
            )
        except Exception as exc:
            self.add("X2", "python decode == chr decode F32 dump (opt-in)", False, f"{type(exc).__name__}: {exc}", required=False)

    # --- phase 4: malformed header -----------------------------------------
    def run_malformed(self) -> None:
        if self.torch is None:
            self.add("S4", "malformed header -> exception, 0 launches", None, "no torch/cuda")
            return
        if not self.loader.ok:
            self.add(
                "S4",
                "malformed header -> exception, 0 launches",
                None,
                f"gpu.chr0 missing, cannot test loader rejection: {self.loader.detail}",
            )
            return
        counter = probe.LaunchCounter()
        original = self.gemm.obj
        if self.gemm.ok:
            self.gemm.obj = counter.wrap(original)       # any launch here is a bug
        rows: list[str] = []
        ok = True
        try:
            with tempfile.TemporaryDirectory(prefix="chr0_bad_") as tmp:
                for mode in chr0_min.CORRUPTIONS:
                    packed, scale = oracle.toy_nf4(64, 64, seed=1)
                    t = chr0_min.ToyNF4("model.layers.0.self_attn.q_proj", "q", 0, packed, scale, 64, 64)
                    path = os.path.join(tmp, f"bad_{mode}.chr")
                    chr0_min.write_toy_chr(path, [t], corrupt=mode, hidden_size=64, intermediate_size=64)
                    mine = "reject"
                    try:
                        chr0_min.read_nf4(chr0_min.read_header(path), t.name)
                        mine = "ACCEPT"
                    except Exception:
                        pass
                    theirs = "reject"
                    try:
                        self.loader.obj(path, t.name, device=self.args.device)
                        theirs = "ACCEPT"
                    except Exception:
                        pass
                    ok = ok and mine == "reject" and theirs == "reject"
                    rows.append(f"{mode}: gate_reader={mine} loader={theirs}")
        finally:
            self.gemm.obj = original
        launches = counter.count
        ok = ok and launches == 0
        self.add("S4", "malformed header -> exception, 0 launches", ok, f"{'; '.join(rows)}; gemm launches={launches}")

    # --- phase 4b: ABI boundary (S9) ---------------------------------------
    def run_arg_validation(self) -> None:
        """S9: lying about M/K/K_pad must raise, not read past the blobs.

        Found while auditing gpu/nf4: undersized packed/scale used to launch and
        return plausible garbage (an OOB device read). The fix is host-side
        validation in gpu/nf4/bindings.cpp; this check keeps it honest.
        """
        if self.torch is None or not self.gemm.ok:
            self.add("S9", "bad M/K/K_pad rejected, no OOB launch", None, "no kernel to probe")
            return
        torch = self.torch
        m, k = 128, 256
        packed, scale = oracle.toy_nf4(m, k, seed=self.args.seed + 9, pad_garbage=False)
        w = backend.make_matrix("argcheck", packed, scale, m, k, self.args.device)
        x, _ = self.make_x(k, 1, self.args.seed + 10)
        half_rows = m // 2

        cases: list[tuple[str, Any]] = [
            ("packed too small (half the rows)", lambda: self.gemm.obj(
                backend.ChrMatrixFallback("bad", m, k, k, w.packed[:half_rows].contiguous(), w.scale), x, 1)),
            ("scale too small (half the rows)", lambda: self.gemm.obj(
                backend.ChrMatrixFallback("bad", m, k, k, w.packed, w.scale[:half_rows].contiguous()), x, 1)),
            ("K_pad != 64*ceil(K/64)", lambda: self.gemm.obj(
                backend.ChrMatrixFallback("bad", m, k, k + 64, w.packed, w.scale), x, 1)),
            ("x shorter than K", lambda: self.gemm.obj(w, x[: k // 2].contiguous(), 1)),
            ("x on cpu", lambda: self.gemm.obj(w, x.cpu(), 1)),
            ("x fp16 instead of bf16", lambda: self.gemm.obj(w, x.to(torch.float16), 1)),
        ]
        rows, ok = [], True
        for what, call in cases:
            try:
                call()
                torch.cuda.synchronize()
                rows.append(f"{what}: ACCEPTED")
                ok = False
            except Exception as exc:
                rows.append(f"{what}: {type(exc).__name__}")
        try:  # the context must still be healthy afterwards
            self.gemm.obj(w, x, 1)
            torch.cuda.synchronize()
            rows.append("honest call after the batch: ok")
        except Exception as exc:
            rows.append(f"honest call after the batch: {type(exc).__name__}: {exc}")
            ok = False
        self.add("S9", "bad M/K/K_pad rejected, no OOB launch", ok, "; ".join(rows))

    # --- phase 5: NaN behaviour (S5) ---------------------------------------
    def run_nan(self) -> None:
        if self.torch is None or not self.gemm.ok:
            self.add("S5", "NaN behaviour documented, no cross-row bleed", None, "no kernel to probe")
            return
        torch = self.torch
        m, k = 128, 256
        packed, scale = oracle.toy_nf4(m, k, seed=self.args.seed + 5, pad_garbage=False)
        w = backend.make_matrix("nan_probe", packed, scale, m, k, self.args.device)
        x_dev, _ = self.make_x(k, 1, self.args.seed + 6)
        notes = []
        ok = True

        x_nan = x_dev.clone()
        x_nan[3, 0] = float("nan")
        y = self.gemm.obj(w, x_nan, 1).to(torch.float32).cpu().numpy()
        bad_rows = int((~np.isfinite(y)).sum())
        notes.append(f"NaN in x[3]: non-finite y entries={bad_rows}/{m} (expected all: every row sums over all K)")
        ok = ok and bad_rows > 0                     # silent swallowing would be worse

        scale_nan = scale.copy()
        scale_nan[0, 0] = np.float16("nan")
        w_nan = backend.make_matrix("nan_scale", packed, scale_nan, m, k, self.args.device)
        y2 = self.gemm.obj(w_nan, x_dev, 1).to(torch.float32).cpu().numpy()
        row0_bad = not np.isfinite(y2[0, 0])
        others_ok = bool(np.isfinite(y2[1:]).all())
        ok = ok and row0_bad and others_ok
        notes.append(
            f"NaN scale in row 0 group 0: row0 non-finite={row0_bad}, rows 1..M-1 all finite={others_ok} "
            "(a real .chr cannot contain this: nf4.md SS6.3 rejects non-finite scales at encode)"
        )
        self.add("S5", "NaN behaviour documented, no cross-row bleed", ok, "; ".join(notes))

    # --- report -------------------------------------------------------------
    def report(self) -> None:
        n = self.num
        if n.maxabs is None and "F1" in n.toys:
            n.maxabs = n.toys["F1"]["maxabs"]
            n.rmse = n.toys["F1"]["rmse"]
            n.metric_source = "toy F1 128x256 N=1 (3B path did not run)"
        print("")
        print("=" * 78)
        print("deep-fold wave 2 -- floor 1 oracle + safety gate")
        print("=" * 78)
        print(f"chr           : {self.args.chr_path or '(none)'}")
        print(f"tensor        : {self.args.name}")
        print(f"loader        : {'gpu.chr0 -> ' + self.loader.detail if self.loader.ok else 'MISSING (' + self.loader.detail + ')'}")
        print(f"kernel        : {'gpu.nf4 -> ' + self.gemm.detail if self.gemm.ok else 'MISSING (' + self.gemm.detail + ')'}")
        print("")
        print(f"{'ID':<4} {'STATUS':<6} {'CHECK':<44} DETAIL")
        print("-" * 78)
        for r in self.results:
            print(f"{r.ident:<4} {r.status:<6} {r.what:<44} {r.detail}")
        print("-" * 78)
        fmt = lambda v: "n/a" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))  # noqa: E731
        print(f"maxabs               : {fmt(n.maxabs)}   (threshold {self.args.tol}, source: {n.metric_source})")
        print(f"rmse                 : {fmt(n.rmse)}")
        print(f"smi_before           : {fmt(n.smi_before_mib)} MiB")
        print(f"smi_after            : {fmt(n.smi_after_mib)} MiB")
        print(f"smi_delta            : {fmt(n.smi_delta_mib)} MiB")
        print(
            f"display_reserved_mib : {fmt(n.display_reserved_mib)}"
            f"   (smi_boot={fmt(n.smi_boot_mib)} MiB incl. display + other processes,"
            f" cuda_ctx={fmt(n.cuda_ctx_mib)} MiB)"
        )
        print(f"peak_alloc_delta_mib : {fmt(n.peak_alloc_delta_mib)}")
        if n.budget:
            print(
                "vram_budget          : packed={packed_mib:.2f} scale={scale_mib:.2f} x={x_mib:.4f} "
                "y={y_mib:.4f} legit={legit_mib:.2f} bf16_W={bf16_w_mib:.2f}".format(**n.budget)
            )
        counts = {s: sum(1 for r in self.results if r.status == s) for s in (PASS, FAIL, SKIP)}
        verdict = self.verdict()
        print("")
        print(f"{verdict}  (pass={counts[PASS]} fail={counts[FAIL]} skip={counts[SKIP]})")
        if verdict == "BLOCKED":
            for r in self.results:
                if r.status == SKIP and r.required:
                    print(f"  blocked {r.ident}: {r.detail}")
        if self.args.json:
            with open(self.args.json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "verdict": verdict,
                        "numbers": asdict(n),
                        "checks": [asdict(r) for r in self.results],
                        "args": {k: str(v) for k, v in vars(self.args).items()},
                    },
                    f,
                    indent=2,
                )
            print(f"json written: {self.args.json}")

    def verdict(self) -> str:
        if any(r.status == FAIL for r in self.results):
            return "FAIL"
        if any(r.status == SKIP and r.required for r in self.results):
            return "BLOCKED"
        return "PASS"

    def exit_code(self) -> int:
        return {"PASS": 0, "FAIL": 2, "BLOCKED": 3}[self.verdict()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="floor-1 GPU NF4 oracle + VRAM/safety gate")
    p.add_argument("--chr", dest="chr_path", default=None, help="path to a .chr (e.g. qwen25-3b.nf4.chr)")
    p.add_argument("--name", default=DEFAULT_NAME, help="CHR0 tensor name (no .weight)")
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--tol", type=float, default=0.05, help="floor-1 maxabs threshold")
    p.add_argument("--vram-extra-mib", type=float, default=20.0, dest="vram_extra_mib")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-index", type=int, default=0, dest="gpu_index")
    p.add_argument("--crosscheck-f32", default=None, dest="crosscheck_f32",
                   help="opt-in: existing chr decode F32 dump; ONE tensor slice is read")
    p.add_argument("--json", default=None, help="write the full report as JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = Gate(args)
    gate.run_selfchecks()
    if gate.setup_torch():
        gate.run_toys()
        gate.run_3b()
        gate.run_malformed()
        gate.run_arg_validation()
        gate.run_nan()
    else:
        gate.skip_all(
            [
                ("K0", "kernel loads and launches"),
                ("K1", "kernel binary newer than its sources"),
                ("F1", "toy 128x256 vs CPU oracle"),
                ("F2", "toy 130x65 vs CPU oracle"),
                ("F3", "3B gate_proj vs CPU oracle"),
                ("F4", "second x seed, packed untouched"),
                ("F5", "two identical calls bit-stable"),
                ("X1", "loader device bytes == .chr bytes"),
                ("S1", "smi delta << BF16 W"),
                ("S2", "no [M,K] fp16/bf16/fp32 dequant W"),
                ("S3", ".chr read-only, mtime unchanged"),
                ("S4", "malformed header -> exception, 0 launches"),
                ("S5", "NaN behaviour documented, no cross-row bleed"),
                ("S6", "packed immutable after HtoD"),
                ("S7", "display/context VRAM logged"),
                ("S9", "bad M/K/K_pad rejected, no OOB launch"),
            ],
            "torch/cuda unavailable",
        )
    gate.report()
    return gate.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
