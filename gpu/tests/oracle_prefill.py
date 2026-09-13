"""Wave-3 floor-1 oracle for prefill: the same CPU decode, now for N > 1.

    python gpu/tests/oracle_prefill.py --chr C:\\dev\\models\\qwen25-3b.nf4.chr

Protocol and thresholds: docs/spec/gpu-safety.md. Floor 1 only -- the
reference is *our own* CPU decode:

    W_hat = decode_nf4(packed, scale)        # nf4_oracle.py, float32
    Y_cpu = W_hat @ X                        # float32, X is [K, N]
    PASS  = maxabs(Y_gpu - Y_cpu) <= 0.05    at rms(x) ~ 1

What this file does NOT do: no BF16 safetensors, no KL, no `chr decode`, no
kernel tuning. It never spawns a child process (gpu-safety.md S8 forbids it for
everything in gpu/tests except nvidia-smi in gpu_probe.py), so P0 runs the
wave-2 gate by *importing* oracle_gate and calling its main() in-process.

Exit codes: 0 = PASS, 2 = FAIL, 3 = BLOCKED (a required check could not run --
including "the kernel still rejects N != 1", unless --allow-skip is given).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import backend  # noqa: E402
import chr0_min  # noqa: E402
import gpu_probe as probe  # noqa: E402
import nf4_oracle as oracle  # noqa: E402
import oracle_gate  # noqa: E402

MIB = oracle_gate.MIB
DEFAULT_NAME = oracle_gate.DEFAULT_NAME
SMI_RESOLUTION_MIB = oracle_gate.SMI_RESOLUTION_MIB
PASS, FAIL, SKIP = oracle_gate.PASS, oracle_gate.FAIL, oracle_gate.SKIP
Result = oracle_gate.Result
compare = oracle_gate.compare

# Reused, not re-implemented: x must be bit-identical to what the N=1 gate feeds
# the kernel, otherwise P4 could not claim "the same number as F1/F2/F3".
make_x = oracle_gate.Gate.make_x

# Shapes are the gate's toys on purpose: F1 has no K padding, F2 has K_pad=128
# for K=65. Reusing them makes P4 a literal regression of F1/F2/F3.
TOY_P1 = (128, 256)
TOY_P2 = (130, 65)


@dataclass
class Numbers:
    n: int = 16
    n_tail: int = 3
    gate_exit: int | None = None
    gate_verdict: str | None = None
    gate_n1: dict[str, float] = field(default_factory=dict)
    prefill_supported: bool | None = None
    prefill_detail: str = "not probed"
    maxabs: float | None = None
    rmse: float | None = None
    metric_source: str = "n/a"
    cases: dict[str, dict[str, float]] = field(default_factory=dict)
    smi_before_mib: int | None = None
    smi_after_mib: int | None = None
    smi_delta_mib: int | None = None
    peak_alloc_delta_mib: float | None = None
    budget: dict[str, float] = field(default_factory=dict)
    vram_mode: str = "n/a"
    prefill_ms: float | None = None
    per_column_ms: float | None = None
    loop_tok_s: float | None = None
    loop_prefill_ms: float | None = None
    loop_vram_mib: int | None = None
    loop_verdict: str | None = None


def short_exc(exc: BaseException) -> str:
    """First line only: a TORCH_CHECK failure drags a 20-frame C++ backtrace."""
    lines = [ln for ln in str(exc).strip().splitlines() if ln.strip()]
    return f"{type(exc).__name__}: {lines[0].strip() if lines else '(no message)'}"


def toy_pair(m: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The gate's toy blobs, drawn in the gate's exact order (oracle_gate.run_toy).

    Returns (packed_pad_garbage, packed_pad_seven, scale, K_pad). Both packed
    variants share every live nibble and differ only in columns K..K_pad-1, so a
    y that moves between them means the kernel read padding as live K.
    """
    kp = oracle.k_pad(k)
    n_groups = kp // 64
    rng = np.random.default_rng(seed + m + k)
    nib = rng.integers(0, 16, size=(m, kp), dtype=np.uint8)
    scale = oracle.toy_scale(m, n_groups, rng)
    nib_a, nib_b = nib.copy(), nib.copy()
    if kp > k:
        nib_a[:, k:] = rng.integers(0, 16, size=(m, kp - k), dtype=np.uint8)
        nib_b[:, k:] = 7
    return oracle.pack_nibbles(nib_a), oracle.pack_nibbles(nib_b), scale, kp


class Prefill:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: list[Result] = []
        self.num = Numbers(n=args.n, n_tail=args.n_tail)
        self.torch: Any = None
        self.loader = backend.probe_loader()
        self.gemm = backend.probe_gemm()
        self.prefill_ok = False
        self._ref3b: tuple[np.ndarray, np.ndarray, int, int, np.ndarray] | None = None
        self.lbl = self.labels()

    def labels(self) -> dict[str, str]:
        """One name per check ID, so a crashed phase still reports its own rows."""
        a = self.args
        return {
            "P0": "wave-2 gate (N=1) still exits 0",
            "K2": f"kernel N range: N={a.n} runs, N>ceiling refused",
            "P1": f"toy {TOY_P1[0]}x{TOY_P1[1]} N={a.n} vs CPU oracle",
            "P2": f"toy {TOY_P2[0]}x{TOY_P2[1]} N={a.n_tail} tails: no pad in y",
            "P3": f"3B {a.name.split('.')[-1]} N={a.n} vs CPU oracle",
            "S": f"smi delta for one N={a.n} gemm << BF16 W",
            "P4": "N=1 unchanged after prefill work",
            "L1": "gpu/loop/smoke.py tok/s",
        }

    def guarded(self, idents: list[str], fn: Any) -> None:
        """Run a phase; an unexpected exception becomes FAIL, not a stack trace.

        A gate that dies on the first surprise reports nothing about the checks
        after it, and "crashed" would be indistinguishable from "not run".
        """
        try:
            fn()
        except Exception as exc:
            have = {r.ident for r in self.results}
            for ident in idents:
                if ident not in have:
                    self.add(ident, self.lbl[ident], False, f"check crashed: {short_exc(exc)}")

    # --- bookkeeping --------------------------------------------------------
    def add(self, ident: str, what: str, ok: bool | None, detail: str = "", required: bool = True) -> None:
        status = SKIP if ok is None else (PASS if ok else FAIL)
        self.results.append(Result(ident, what, status, detail, required))

    def skip_all(self, idents: list[str], reason: str, required: bool = True) -> None:
        for ident in idents:
            self.add(ident, self.lbl[ident], None, reason, required)

    @property
    def prefill_required(self) -> bool:
        """Without --allow-skip, a missing prefill path is BLOCKED, not a pass."""
        return not self.args.allow_skip

    # --- P0: the wave-2 gate must still be green ----------------------------
    def phase_p0(self) -> None:
        """P0: oracle_gate.py (N=1) unchanged, imported rather than spawned.

        A subprocess here would break the gate's own S8 (only gpu_probe.py may
        spawn, and only nvidia-smi), so main() is called in-process and its JSON
        report is read back for the N=1 reference numbers used by P4.
        """
        argv = [
            "--name", self.args.name,
            "--seed", str(self.args.seed),
            "--tol", str(self.args.tol),
            "--device", self.args.device,
            "--gpu-index", str(self.args.gpu_index),
            "--vram-extra-mib", str(self.args.vram_extra_mib),
        ]
        if self.args.chr_path:
            argv += ["--chr", self.args.chr_path]
        tmp = tempfile.mkdtemp(prefix="prefill_p0_")
        report = os.path.join(tmp, "gate.json")
        argv += ["--json", report]
        print("[P0] wave-2 gate, in-process (no child process: gpu-safety.md S8)")
        try:
            code = oracle_gate.main(argv)
            self.num.gate_exit = int(code)
            counts, note = {}, ""
            if os.path.isfile(report):
                with open(report, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.num.gate_verdict = data.get("verdict")
                checks = data.get("checks", [])
                counts = {s: sum(1 for c in checks if c["status"] == s) for s in (PASS, FAIL, SKIP)}
                nums = data.get("numbers", {})
                n1 = {}
                if nums.get("maxabs") is not None:
                    n1["F3"] = float(nums["maxabs"])
                for ident, toy in (nums.get("toys") or {}).items():
                    n1[ident] = float(toy["maxabs"])
                self.num.gate_n1 = n1
                note = f" checks(pass/fail/skip)={counts.get(PASS)}/{counts.get(FAIL)}/{counts.get(SKIP)}"
                note += " N=1 maxabs " + ", ".join(f"{k}={v:.6f}" for k, v in sorted(n1.items()))
            self.add(
                "P0",
                self.lbl["P0"],
                code == 0,
                f"oracle_gate.main() exit={code} verdict={self.num.gate_verdict}{note}",
            )
        except Exception as exc:
            self.add("P0", self.lbl["P0"], False, short_exc(exc))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # --- torch --------------------------------------------------------------
    def setup_torch(self) -> bool:
        try:
            import torch
        except Exception as exc:
            self.add("GPU", "torch import", False, f"{type(exc).__name__}: {exc}")
            return False
        self.torch = torch
        if self.args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                self.add("GPU", "cuda available", False, "torch.cuda.is_available() is False")
                return False
            torch.zeros(1, device=self.args.device).sum().item()
            torch.cuda.synchronize()
        return True

    # --- K2: can the kernel do N > 1 at all? --------------------------------
    def probe_capability(self) -> None:
        """K2: probe both the Python wrapper and the raw extension with N=16.

        Two probes, because they can disagree: gpu/nf4/__init__.py raises on
        N != 1 before the extension is reached, so a kernel that grew prefill
        support would still look dead until that guard is lifted. Saying which
        of the two refused is the difference between "wave 3 landed, wrapper
        forgotten" and "the .cu still returns -2".
        """
        if self.torch is None or not self.gemm.ok:
            self.num.prefill_detail = f"no kernel: {self.gemm.detail}"
            self.add("K2", self.lbl["K2"], None, self.num.prefill_detail, self.prefill_required)
            return
        torch = self.torch
        m, k, n = 64, 64, self.args.n
        packed, scale = oracle.toy_nf4(m, k, seed=self.args.seed, pad_garbage=False)
        w = backend.make_matrix("cap_probe", packed, scale, m, k, self.args.device)
        x, _ = make_x(self, k, n, self.args.seed + 1)

        notes: list[str] = []
        wrapper_ok = False
        try:
            y = self.gemm.obj(w, x, n)
            torch.cuda.synchronize()
            wrapper_ok = tuple(y.shape) == (m, n)
            notes.append(f"{self.gemm.detail}(N={n}) -> y{tuple(y.shape)} {y.dtype}")
        except Exception as exc:
            notes.append(f"{self.gemm.detail}(N={n}) -> {short_exc(exc)}")

        ext_ok: bool | None = None
        try:  # private on purpose: this is a capability probe, not a code path
            mod = __import__("gpu.nf4", fromlist=["_load_ext"])
            ext = mod._load_ext()
            y2 = ext.nf4_gemm(w.packed, w.scale, x, int(w.M), int(w.K), int(w.K_pad))
            torch.cuda.synchronize()
            ext_ok = tuple(y2.shape) == (m, n)
            notes.append(f"chr_nf4_ext.nf4_gemm(N={n}) -> y{tuple(y2.shape)}")
        except Exception as exc:
            ext_ok = False
            notes.append(f"chr_nf4_ext.nf4_gemm(N={n}) -> {short_exc(exc)}")

        self.prefill_ok = bool(wrapper_ok)
        self.num.prefill_supported = self.prefill_ok
        if not self.prefill_ok:
            where = "python wrapper only (gpu/nf4/__init__.py); the .cu accepted N>1" if ext_ok else "wrapper and .cu"
            notes.append(f"prefill refused by: {where}")

        status: bool | None = None
        if self.prefill_ok:
            status = self.probe_ceiling(w, packed, scale, m, k, notes)
        self.num.prefill_detail = "; ".join(notes)
        self.add("K2", self.lbl["K2"], status, self.num.prefill_detail, self.prefill_required)

    def probe_ceiling(self, w: Any, packed: np.ndarray, scale: np.ndarray, m: int, k: int,
                      notes: list[str]) -> bool:
        """One N above the kernel's advertised ceiling must refuse or be right.

        nf4_gemm.cu caps a launch at N=16 and expects the host to chunk wider
        prefills. Either answer is acceptable -- an exception (chunk it) or a
        correct y (ceiling lifted). The third outcome, a plausible y computed
        from a tile that only covers 16 columns, is the one that would poison a
        prefill silently, so it is a FAIL.
        """
        torch = self.torch
        n = self.args.n_over
        x, x_host = make_x(self, k, n, self.args.seed + 2)
        try:
            y = self.gemm.obj(w, x, n)
            torch.cuda.synchronize()
        except Exception as exc:
            notes.append(f"N={n} refused: {short_exc(exc)} (host chunks N>ceiling)")
            return True
        w_hat = oracle.decode_nf4(packed, scale, m, k)
        maxabs, _ = compare(y.to(torch.float32).cpu().numpy(), oracle.matmul_f32(w_hat, x_host))
        ok = maxabs <= self.args.tol
        notes.append(f"N={n} accepted (ceiling lifted): maxabs={maxabs:.6f} tol={self.args.tol} ok={ok}")
        return ok

    # --- prefill vs column loop --------------------------------------------
    def gemm_prefill(self, w: Any, x: Any, n: int) -> tuple[Any, float]:
        """One prefill call, y [M, N]. Timed on the device."""
        torch = self.torch
        t0 = time.perf_counter()
        y = self.gemm.obj(w, x, n).clone()  # clone: y may be a reused buffer
        torch.cuda.synchronize()
        return y, (time.perf_counter() - t0) * 1e3

    def gemm_columns(self, w: Any, x: Any, n: int) -> tuple[Any, float]:
        """N separate N=1 calls stitched into [M, N] -- the reference path.

        This is NOT prefill and is never used to pass P1/P2/P3. It exists so the
        N>1 oracle itself is exercised (and its numbers published) while the
        kernel is still decode-only.
        """
        torch = self.torch
        t0 = time.perf_counter()
        cols = [self.gemm.obj(w, x[:, j : j + 1].contiguous(), 1).reshape(-1, 1) for j in range(n)]
        y = torch.cat(cols, dim=1)
        torch.cuda.synchronize()
        return y, (time.perf_counter() - t0) * 1e3

    def gemm_n(self, w: Any, x: Any, n: int) -> tuple[Any, float, str]:
        if self.prefill_ok:
            y, ms = self.gemm_prefill(w, x, n)
            return y, ms, "prefill"
        y, ms = self.gemm_columns(w, x, n)
        return y, ms, "column-loop"

    def record(self, ident: str, m: int, k: int, n: int, maxabs: float, rmse: float) -> None:
        self.num.cases[ident] = {"M": m, "K": k, "N": n, "maxabs": maxabs, "rmse": rmse}

    # --- P1: toy N = 16 -----------------------------------------------------
    def phase_p1(self) -> None:
        m, k = TOY_P1
        n = self.args.n
        what = self.lbl["P1"]
        if not self.ready("P1"):
            return
        torch = self.torch
        packed_a, _, scale, kp = toy_pair(m, k, self.args.seed)
        w_hat = oracle.decode_nf4(packed_a, scale, m, k)
        w = backend.make_matrix("p1", packed_a, scale, m, k, self.args.device)
        x_dev, x_host = make_x(self, k, n, self.args.seed + 1)
        y_cpu = oracle.matmul_f32(w_hat, x_host)

        y, ms, mode = self.gemm_n(w, x_dev, n)
        maxabs, rmse = compare(y.to(torch.float32).cpu().numpy(), y_cpu)
        self.record("P1", m, k, n, maxabs, rmse)

        # Every column must also match the decode path (N=1) it will replace.
        y1, _ = self.gemm_columns(w, x_dev, n)
        col_max, _ = compare(y.to(torch.float32).cpu().numpy(), y1.to(torch.float32).cpu().numpy())
        detail = (
            f"M={m} K={k} K_pad={kp} N={n} maxabs={maxabs:.6f} rmse={rmse:.6f} tol={self.args.tol} "
            f"shape={tuple(y.shape)} dtype={y.dtype} vs N=1 column loop maxabs={col_max:.6f} "
            f"[{mode}, {ms:.3f} ms]"
        )
        ok = maxabs <= self.args.tol and tuple(y.shape) == (m, n) and y.dtype == torch.bfloat16
        if self.prefill_ok:
            self.add("P1", what, ok, detail)
        else:
            self.add("P1", what, None, "kernel is N=1 only (K2); numbers below are the column loop, not prefill",
                     self.prefill_required)
            self.add("R1", f"oracle N={n} vs N=1 column loop (reference)", ok, detail, required=False)

    # --- P2: N = 3, both tails ---------------------------------------------
    def phase_p2(self) -> None:
        m, k = TOY_P2
        n = self.args.n_tail
        what = self.lbl["P2"]
        if not self.ready("P2"):
            return
        torch = self.torch
        packed_a, packed_b, scale, kp = toy_pair(m, k, self.args.seed)
        w_hat = oracle.decode_nf4(packed_a, scale, m, k)
        w_a = backend.make_matrix("p2_pad_garbage", packed_a, scale, m, k, self.args.device)
        w_b = backend.make_matrix("p2_pad_seven", packed_b, scale, m, k, self.args.device)
        x_dev, x_host = make_x(self, k, n, self.args.seed + 1)
        y_cpu = oracle.matmul_f32(w_hat, x_host)

        y_a, ms, mode = self.gemm_n(w_a, x_dev, n)
        maxabs, rmse = compare(y_a.to(torch.float32).cpu().numpy(), y_cpu)
        self.record("P2", m, k, n, maxabs, rmse)

        # K tail: columns K..K_pad-1 hold garbage in w_a and nibble 7 in w_b.
        y_b, _, _ = self.gemm_n(w_b, x_dev, n)
        pad_same = bool(torch.equal(y_a, y_b))
        maxabs_b, _ = compare(y_b.to(torch.float32).cpu().numpy(), y_cpu)

        # N tail: widen x by one wild column. The live columns must not move, or
        # the N-dimension mask leaks the pad column into y.
        junk = torch.full((k, 1), 137.0, dtype=torch.bfloat16, device=x_dev.device)
        x_wide = torch.cat([x_dev, junk], dim=1).contiguous()
        y_wide, _, _ = self.gemm_n(w_a, x_wide, n + 1)
        n_tail_same = bool(torch.equal(y_wide[:, :n].contiguous(), y_a))
        n_tail_max, _ = compare(y_wide[:, :n].to(torch.float32).cpu().numpy(), y_cpu)

        ok = (
            maxabs <= self.args.tol
            and pad_same
            and maxabs_b <= self.args.tol
            and n_tail_max <= self.args.tol
            and tuple(y_a.shape) == (m, n)
        )
        detail = (
            f"M={m} K={k} K_pad={kp} N={n} maxabs={maxabs:.6f} rmse={rmse:.6f} shape={tuple(y_a.shape)}; "
            f"K pad-garbage vs pad-7 bit-identical={pad_same} maxabs_pad7={maxabs_b:.6f}; "
            f"N tail: y[:, :{n}] of an N={n + 1} call bit-identical={n_tail_same} maxabs={n_tail_max:.6f} "
            f"[{mode}, {ms:.3f} ms]"
        )
        if self.prefill_ok:
            self.add("P2", what, ok, detail)
        else:
            self.add("P2", what, None, "kernel is N=1 only (K2); numbers below are the column loop, not prefill",
                     self.prefill_required)
            self.add("R2", f"oracle N={n} tails via N=1 column loop (reference)", ok, detail, required=False)

    # --- P3 + S: the real matrix -------------------------------------------
    def reference_3b(self) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
        """packed/scale/M/K read from the .chr by our own parser, plus W_hat."""
        if self._ref3b is None:
            header = chr0_min.read_header(self.args.chr_path)
            packed, scale, m, k = chr0_min.read_nf4(header, self.args.name)
            self._ref3b = (packed, scale, m, k, oracle.decode_nf4(packed, scale, m, k))
        return self._ref3b

    def materialize(self, notes: list[str]) -> Any:
        packed, scale, m, k, _ = self.reference_3b()
        if self.loader.ok:
            notes.append(f"loader={self.loader.detail}")
            return self.loader.obj(self.args.chr_path, self.args.name, device=self.args.device)
        notes.append("loader=gpu/tests fallback upload (gpu.chr0 missing)")
        return backend.make_matrix(self.args.name, packed, scale, m, k, self.args.device)

    def phase_p3(self) -> None:
        n = self.args.n
        idents = ["P3", "S"]
        if not self.args.chr_path:
            self.skip_all(idents, "no --chr given")
            return
        if not os.path.isfile(self.args.chr_path):
            self.skip_all(idents, f"missing file: {self.args.chr_path}")
            return
        if self.torch is None or not self.gemm.ok:
            self.skip_all(idents, f"no kernel: {self.gemm.detail}")
            return
        torch = self.torch
        _, _, m, k, w_hat = self.reference_3b()
        x_dev, x_host = make_x(self, k, n, self.args.seed + 1)
        y_cpu = oracle.matmul_f32(w_hat, x_host)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated()
        self.num.smi_before_mib = self.try_smi()

        notes: list[str] = []
        watcher = probe.DequantWatcher(min_numel=m * k // 2)
        with probe.ReadOnlyFileGuard(self.args.chr_path) as guard:
            with watcher:
                w = self.materialize(notes)
                torch.cuda.synchronize()
                y, ms, mode = self.gemm_n(w, x_dev, n)
        peak = torch.cuda.max_memory_allocated()
        self.num.peak_alloc_delta_mib = round((peak - alloc_before) / MIB, 2)
        self.num.smi_after_mib = self.try_smi()
        self.num.prefill_ms = round(ms, 3)
        self.num.per_column_ms = round(ms / n, 3)

        maxabs, rmse = compare(y.to(torch.float32).cpu().numpy(), y_cpu)
        self.record("P3", m, k, n, maxabs, rmse)
        self.num.maxabs, self.num.rmse = maxabs, rmse
        self.num.metric_source = f"{self.args.name} [{m},{k}] N={n} ({mode})"
        detail = (
            f"M={m} K={k} N={n} maxabs={maxabs:.6f} rmse={rmse:.6f} tol={self.args.tol} "
            f"shape={tuple(y.shape)} [{mode}, {ms:.3f} ms, {ms / n:.3f} ms/column]"
        )
        if self.prefill_ok:
            self.add("P3", self.lbl["P3"], maxabs <= self.args.tol and tuple(y.shape) == (m, n), detail)
        else:
            self.add("P3", self.lbl["P3"], None,
                     "kernel is N=1 only (K2); numbers below are the column loop, not prefill",
                     self.prefill_required)
            self.add("R3", f"3B oracle N={n} via N=1 column loop (reference)",
                     maxabs <= self.args.tol, detail, required=False)
        self.judge_vram(m, k, n, mode, watcher, guard, notes)

    def judge_vram(
        self,
        m: int,
        k: int,
        n: int,
        mode: str,
        watcher: probe.DequantWatcher,
        guard: probe.ReadOnlyFileGuard,
        notes: list[str],
    ) -> None:
        """S: a prefill must not birth a BF16 W, and x/y grow only by N."""
        budget = probe.nf4_vram_budget(m, k, n)
        self.num.budget = {kk: round(v, 3) for kk, v in budget.items()}
        self.num.vram_mode = mode
        before, after = self.num.smi_before_mib, self.num.smi_after_mib
        big = [h for h in watcher.hits if h.nbytes >= m * k * 2]
        extra = [f"aten cuda float tensors >= BF16 W: {[(h.op, h.shape, h.dtype) for h in big]}"]
        extra.append(f".chr open modes={guard.modes or ['(none)']} violations={guard.violations or 'none'}")
        if before is None or after is None:
            self.add("S", self.lbl["S"], None, "nvidia-smi unavailable")
            return
        delta = after - before
        self.num.smi_delta_mib = delta
        limit = budget["legit_mib"] + self.args.vram_extra_mib
        peak = self.num.peak_alloc_delta_mib
        ok = (peak is None or peak <= limit) and not big and guard.ok
        # Same resolution rule as gpu-safety.md SS4: smi reports *reserved* VRAM in
        # ~20 MiB segments, so it can only arbitrate matrices whose BF16 W is
        # bigger than that granularity. gate_proj (43 MiB) is.
        if budget["bf16_w_mib"] >= SMI_RESOLUTION_MIB:
            ok = ok and delta < budget["bf16_w_mib"] and delta <= limit
            src = "smi is the gate"
        else:
            src = (
                f"smi cannot resolve a BF16 W of {budget['bf16_w_mib']:.2f} MiB "
                f"(segment granularity ~{SMI_RESOLUTION_MIB:.0f} MiB); torch peak governs"
            )
        self.add(
            "S",
            self.lbl["S"],
            ok,
            f"mode={mode} smi_before={before} smi_after={after} delta_mib={delta} "
            f"legit(packed+scale+x+y at N={n})={budget['legit_mib']:.2f} limit={limit:.2f} "
            f"bf16_W={budget['bf16_w_mib']:.2f} fp32_W={budget['fp32_w_mib']:.2f} "
            f"torch_peak_delta_mib={peak}; {src}; " + "; ".join(notes + extra),
        )

    # --- P4: N = 1 must not regress ----------------------------------------
    def phase_p4(self) -> None:
        what = self.lbl["P4"]
        if not self.ready("P4"):
            return
        torch = self.torch
        rows: list[str] = []
        ok = True
        for ident, (m, k) in (("F1", TOY_P1), ("F2", TOY_P2)):
            packed_a, _, scale, _ = toy_pair(m, k, self.args.seed)
            w_hat = oracle.decode_nf4(packed_a, scale, m, k)
            w = backend.make_matrix(f"p4_{ident}", packed_a, scale, m, k, self.args.device)
            x_dev, x_host = make_x(self, k, 1, self.args.seed + 1)
            y = self.gemm.obj(w, x_dev, 1)
            torch.cuda.synchronize()
            maxabs, rmse = compare(y.to(torch.float32).cpu().numpy(), oracle.matmul_f32(w_hat, x_host))
            self.record(f"P4/{ident}", m, k, 1, maxabs, rmse)
            same = self.same_as_gate(ident, maxabs)
            ok = ok and maxabs <= self.args.tol and same is not False
            rows.append(f"{ident} {m}x{k}: maxabs={maxabs:.6f} rmse={rmse:.6f} gate_match={self.fmt_match(same)}")

        if self.args.chr_path and os.path.isfile(self.args.chr_path):
            _, _, m, k, w_hat = self.reference_3b()
            notes: list[str] = []
            w = self.materialize(notes)
            x_dev, x_host = make_x(self, k, 1, self.args.seed + 1)
            y = self.gemm.obj(w, x_dev, 1)
            torch.cuda.synchronize()
            maxabs, rmse = compare(y.to(torch.float32).cpu().numpy(), oracle.matmul_f32(w_hat, x_host))
            self.record("P4/F3", m, k, 1, maxabs, rmse)
            same = self.same_as_gate("F3", maxabs)
            ok = ok and maxabs <= self.args.tol and same is not False
            rows.append(f"F3 {m}x{k}: maxabs={maxabs:.6f} rmse={rmse:.6f} gate_match={self.fmt_match(same)}")
        else:
            rows.append("F3 3B: skipped (no --chr)")
        rows.append(f"tol={self.args.tol}")
        self.add("P4", what, ok, "; ".join(rows))

    def same_as_gate(self, ident: str, maxabs: float) -> bool | None:
        """Exact equality against the same check inside oracle_gate (P0).

        Same seed -> same x, same W, same kernel, so the number must be *equal*,
        not merely under tolerance. This is the regression tripwire: a prefill
        patch that quietly changes the N=1 path shows up here even if 0.05 still
        holds.
        """
        ref = self.num.gate_n1.get(ident)
        if ref is None:
            return None
        return maxabs == ref

    @staticmethod
    def fmt_match(same: bool | None) -> str:
        return "n/a" if same is None else ("exact" if same else "DIFFERS")

    # --- L1: token loop smoke ----------------------------------------------
    def phase_l1(self) -> None:
        """L1: run the token-loop smoke if it exists and record tok/s + smi.

        Imported, not spawned (S8). Its main() reads sys.argv and calls
        sys.exit(), so argv is replaced for the call and SystemExit is caught --
        otherwise somebody else's argparse would kill this gate before it ever
        printed a verdict. Runs last: the smoke loads the whole model, so it
        must not be inside the S measurement.
        """
        what = self.lbl["L1"]
        path = os.path.join(REPO, "gpu", "loop", "smoke.py")
        rel = os.path.relpath(path, REPO)
        if self.args.no_l1:
            self.add("L1", what, None, "--no-l1 given", required=False)
            return
        if not os.path.isfile(path):
            self.add("L1", what, None, f"absent: {rel} (token loop not landed yet)", required=False)
            return
        try:
            spec = importlib.util.spec_from_file_location("gpu_loop_smoke", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = next((getattr(mod, a) for a in ("main", "smoke", "measure") if callable(getattr(mod, a, None))), None)
            if fn is None:
                self.add("L1", what, None, f"{rel} has no main()/smoke()/measure()", required=False)
                return
        except Exception as exc:
            self.add("L1", what, None, f"{rel} import raised: {short_exc(exc)}", required=False)
            return

        argv, buf = list(sys.argv), io.StringIO()
        sys.argv = [path] + (["--chr", self.args.chr_path] if self.args.chr_path else [])
        smi_before = self.try_smi()
        t0 = time.perf_counter()
        rc: Any = None
        err = ""
        try:
            with contextlib.redirect_stdout(buf):
                rc = fn()
        except SystemExit as exc:  # argparse and friends
            rc = exc.code
        except BaseException as exc:  # noqa: BLE001 -- the smoke must not kill the gate
            err = short_exc(exc)
        finally:
            sys.argv = argv
        dt = time.perf_counter() - t0
        smi_after = self.try_smi()
        text = buf.getvalue()
        print(f"[L1] {rel} output ({dt:.1f}s):")
        print(text.rstrip() or "  (no output)")

        grab = lambda pat: (re.search(pat, text).group(1) if re.search(pat, text) else None)  # noqa: E731
        tok_s = grab(r"decode_tok_s=([0-9.]+)")
        prefill_ms = grab(r"prefill_\d+_ms=([0-9.]+)")
        vram = grab(r"vram_decode_mb=(\d+)")
        self.num.loop_tok_s = float(tok_s) if tok_s else None
        self.num.loop_prefill_ms = float(prefill_ms) if prefill_ms else None
        self.num.loop_vram_mib = int(vram) if vram else None
        self.num.loop_verdict = grab(r"SMOKE: (PASS|FAIL)") or (f"error: {err}" if err else f"rc={rc}")
        gates = grab(r"\n((?:\w+=(?:PASS|FAIL)  ?)+)") or ""
        # wave3-safety.md L1 is "Paris or skip": the greedy English answer plus a
        # tok/s number. The smoke's own rc also folds in gates this oracle does
        # not own (the RU chat template, for one), so it is reported, never
        # gated -- otherwise agent 7's tokenizer could redden a prefill result.
        paris = grab(r"greedy_en_paris=(PASS|FAIL)")
        ok = True if paris == "PASS" else None
        self.add(
            "L1",
            what,
            ok,
            f"{rel} rc={rc} in {dt:.1f}s smoke={self.num.loop_verdict} greedy_en_paris={paris} "
            f"decode_tok_s={self.num.loop_tok_s} prefill_ms={self.num.loop_prefill_ms} "
            f"smi {smi_before} -> {smi_after} MiB (vram_decode_mb={self.num.loop_vram_mib}); "
            f"{gates.strip() or err or 'no gate line parsed'}",
            required=False,
        )

    # --- helpers ------------------------------------------------------------
    def ready(self, ident: str) -> bool:
        if self.torch is None or not self.gemm.ok:
            self.add(ident, self.lbl[ident], None, f"no kernel: {self.gemm.detail}")
            return False
        return True

    def try_smi(self) -> int | None:
        try:
            return probe.smi_used_mib(self.args.gpu_index)
        except Exception:
            return None

    # --- report -------------------------------------------------------------
    def report(self) -> None:
        n = self.num
        print("")
        print("=" * 78)
        print("deep-fold wave 3 -- floor 1 prefill oracle (N > 1)")
        print("=" * 78)
        print(f"chr           : {self.args.chr_path or '(none)'}")
        print(f"tensor        : {self.args.name}")
        print(f"N / N_tail    : {n.n} / {n.n_tail}")
        print(f"kernel        : {'gpu.nf4 -> ' + self.gemm.detail if self.gemm.ok else 'MISSING (' + self.gemm.detail + ')'}")
        print(f"prefill path  : {'kernel N>1' if self.prefill_ok else 'NOT AVAILABLE (kernel is N=1 only)'}")
        print("")
        print(f"{'ID':<4} {'STATUS':<6} {'CHECK':<44} DETAIL")
        print("-" * 78)
        for r in self.results:
            print(f"{r.ident:<4} {r.status:<6} {r.what:<44} {r.detail}")
        print("-" * 78)
        fmt = lambda v: "n/a" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))  # noqa: E731
        print(f"maxabs               : {fmt(n.maxabs)}   (threshold {self.args.tol}, source: {n.metric_source})")
        print(f"rmse                 : {fmt(n.rmse)}")
        for ident, c in n.cases.items():
            print(f"  {ident:<7} M={c['M']:<6} K={c['K']:<5} N={c['N']:<3} maxabs={c['maxabs']:.6f} rmse={c['rmse']:.6f}")
        print(f"smi_before           : {fmt(n.smi_before_mib)} MiB")
        print(f"smi_after            : {fmt(n.smi_after_mib)} MiB")
        print(f"smi_delta            : {fmt(n.smi_delta_mib)} MiB   (mode: {n.vram_mode})")
        print(f"peak_alloc_delta_mib : {fmt(n.peak_alloc_delta_mib)}")
        if n.budget:
            print(
                "vram_budget          : packed={packed_mib:.2f} scale={scale_mib:.2f} x={x_mib:.4f} "
                "y={y_mib:.4f} legit={legit_mib:.2f} bf16_W={bf16_w_mib:.2f}".format(**n.budget)
            )
        print(f"gemm                 : {fmt(n.prefill_ms)} ms total, {fmt(n.per_column_ms)} ms/column")
        print(
            f"loop (L1)            : decode_tok_s={fmt(n.loop_tok_s)} prefill_ms={fmt(n.loop_prefill_ms)} "
            f"vram_decode_mb={fmt(n.loop_vram_mib)} smoke={n.loop_verdict}"
        )
        counts = {s: sum(1 for r in self.results if r.status == s) for s in (PASS, FAIL, SKIP)}
        verdict = self.verdict()
        print("")
        print(f"{verdict}  (pass={counts[PASS]} fail={counts[FAIL]} skip={counts[SKIP]})")
        if verdict == "BLOCKED":
            for r in self.results:
                if r.status == SKIP and r.required:
                    print(f"  blocked {r.ident}: {r.detail}")
            if self.gemm.ok and not self.prefill_ok:
                print("")
                print(f"  kernel refused N={n.n}: prefill at this width is not available.")
                print(f"    {self.num.prefill_detail}")
                print("    P1/P2/P3 cannot run. Re-run with --allow-skip to gate P0/P4/S only")
                print("    (N=1 regression plus VRAM), or with an N the kernel accepts.")
                print("    See docs/spec/gpu-safety.md SS11.5.")
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
    p = argparse.ArgumentParser(description="floor-1 NF4 prefill (N>1) oracle; wave 3")
    p.add_argument("--chr", dest="chr_path", default=None, help="path to a .chr (e.g. qwen25-3b.nf4.chr)")
    p.add_argument("--name", default=DEFAULT_NAME, help="CHR0 tensor name (no .weight)")
    p.add_argument("--n", type=int, default=16, help="prefill width for P1/P3")
    p.add_argument("--n-tail", type=int, default=3, dest="n_tail", help="tail width for P2")
    p.add_argument("--n-over", type=int, default=17, dest="n_over",
                   help="N just above the kernel's ceiling; must refuse or be correct (K2)")
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--tol", type=float, default=0.05, help="floor-1 maxabs threshold")
    p.add_argument("--vram-extra-mib", type=float, default=20.0, dest="vram_extra_mib")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-index", type=int, default=0, dest="gpu_index")
    p.add_argument(
        "--allow-skip",
        action="store_true",
        dest="allow_skip",
        help="a kernel that still rejects N!=1 skips P1-P3 instead of BLOCKING the run",
    )
    p.add_argument("--no-l1", action="store_true", dest="no_l1",
                   help="skip the token-loop smoke (L1); it loads the whole model")
    p.add_argument("--json", default=None, help="write the full report as JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 2:
        print(f"--n must be >= 2 (this is the prefill oracle), got {args.n}")
        return 3
    gate = Prefill(args)
    plan: list[tuple[list[str], Any]] = [
        (["K2"], gate.probe_capability),
        (["P1"], gate.phase_p1),
        (["P2"], gate.phase_p2),
        (["P3", "S"], gate.phase_p3),
        (["P4"], gate.phase_p4),
    ]
    gate.guarded(["P0"], gate.phase_p0)
    if gate.setup_torch():
        for idents, phase in plan:
            gate.guarded(idents, phase)
    else:
        for idents, _ in plan:
            gate.skip_all(idents, "torch/cuda unavailable")
    gate.guarded(["L1"], gate.phase_l1)
    gate.report()
    return gate.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
