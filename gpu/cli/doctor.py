"""``deepfold doctor``: can ``deepfold run`` succeed on *this* machine?

Doctor is the only public chooser of prebuilt-versus-JIT and of whether this
GPU may generate at all. Shipped generate is **sm_86 only**. sm_80 and sm_89
are refused unless ``DEEPFOLD_ALLOW_UNMEASURED_ARCH=1`` after a rebuild --
that override is not a documented feature.

:func:`probe` is the only function that touches torch, the driver, or the
filesystem; :func:`verdict` / :func:`checks` / :func:`exit_code` are pure
functions of a :class:`Machine`.

Exit codes: **0** generate is possible (or ``--compress-only`` and ``chr``
works); **2** broken install on a card that could run; **3** generate refused
by class while compress still works (**3 is not green generate**); **1**
neither.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import messages
from .paths import REPO, find_chr_bin

SHIP_CAPABILITY = (8, 6)
# Same .cu, same MMA + cp.async; only an extra -gencode away from loading.
UNMEASURED_CAPABILITIES = ((8, 0), (8, 9))
OVERRIDE_ENV = "DEEPFOLD_ALLOW_UNMEASURED_ARCH"
KERNEL_GENCODE = "compute_86,sm_86; no PTX"

# An arch class whose only problem is the install, not the hardware.
FIXABLE = frozenset({"ship", "experimental", "cpu-torch", "no-torch"})

_NF4_DIR = REPO / "gpu" / "nf4"
_KERNEL_SOURCES = ("nf4_gemm.cu", "bindings.cpp")


# --------------------------------------------------------------------------- #
# the machine
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Machine:
    """Everything doctor learned. Built by :func:`probe`, faked in tests."""

    system: str = "Windows"
    python: tuple[int, int, int] = (3, 11, 0)
    torch: str | None = None
    torch_cuda: str | None = None
    torch_hip: str | None = None
    cuda_available: bool = False
    capability: tuple[int, int] | None = None
    device_name: str | None = None
    vram_total_mib: int | None = None
    smi_used_mib: int | None = None
    mps: bool = False
    nf4_ext: Path | None = None
    nf4_ext_stale: bool = False
    nf4_ext_abi_ok: bool = True
    host_cc: str | None = None
    nvcc: str | None = None
    chr_bin: Path | None = None
    chr_runs: bool = False
    transformers: bool = False
    safetensors: bool = False
    hf_hub: bool = False
    ollama: str | None = None
    extras_missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def sm(self) -> str:
        if self.capability is None:
            return "unknown"
        return f"sm_{self.capability[0]}{self.capability[1]}"


def _interpreter_tag() -> str:
    return f"cp{sys.version_info[0]}{sys.version_info[1]}"


def _nf4_artifact() -> tuple[Path | None, bool, bool]:
    """In-tree extension, whether the sources are newer, whether this Python can load it.

    Globs both suffixes: ``.pyd`` on Windows, ``.so`` on Linux
    (``gpu.ext_bin.find_ext`` -- the generate wrapper used to look only for ``.pyd``).

    The ABI tag matters as much as the file's existence: a
    ``chr_nf4_ext.cp311-win_amd64.pyd`` is not importable from Python 3.12, and
    reporting it as a green kernel is exactly the "file exists => this box"
    assumption D8 forbids.
    """
    matches = sorted(_NF4_DIR.glob("chr_nf4_ext*.pyd")) + sorted(
        _NF4_DIR.glob("chr_nf4_ext*.so")
    )
    if not matches:
        return None, False, True
    tag = _interpreter_tag()
    usable = [p for p in matches if "cp3" not in p.name or tag in p.name]
    artifact = (usable or matches)[0]
    built = artifact.stat().st_mtime
    stale = any(
        (_NF4_DIR / name).is_file() and (_NF4_DIR / name).stat().st_mtime > built
        for name in _KERNEL_SOURCES
    )
    return artifact, stale, bool(usable)


def _host_compiler(*, inject: bool) -> str | None:
    """``cl.exe`` on Windows (after ``vcvars64.bat`` if asked), else ``g++``.

    The VS environment dump costs about a second, so it only runs when there is
    no prebuilt extension and the answer therefore matters.
    """
    if os.name == "nt":
        from gpu.win_toolchain import inject_msvc_env, which_cl

        hit = which_cl()
        if hit is None and inject:
            inject_msvc_env()
            hit = which_cl()
        return hit
    return shutil.which("g++") or shutil.which("c++")


def _smi_used_mib() -> int | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _chr_runs(chr_bin: Path) -> bool:
    """``chr -h`` exits 0 and names its three subcommands."""
    try:
        out = subprocess.run(
            [str(chr_bin), "-h"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "compress" in (out.stdout + out.stderr)


def _installed(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def probe(*, chr_bin: str | None = None, extras_for: str | None = None) -> Machine:
    """Read the machine. The only impure function in this module."""
    torch_version = torch_cuda = torch_hip = None
    cuda_available = False
    capability = device_name = None
    vram_total = None
    mps = False

    try:
        import torch
    except Exception:  # noqa: BLE001 - a broken torch install is "no torch"
        torch = None
    if torch is not None:
        torch_version = str(torch.__version__)
        torch_cuda = getattr(torch.version, "cuda", None)
        torch_hip = getattr(torch.version, "hip", None)
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            cuda_available = False
        if cuda_available:
            try:
                capability = tuple(torch.cuda.get_device_capability(0))  # type: ignore[assignment]
                device_name = torch.cuda.get_device_name(0)
                vram_total = int(
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                )
            except Exception:  # noqa: BLE001
                capability = None
        try:
            mps = bool(torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            mps = False

    nf4_ext, stale, abi_ok = _nf4_artifact()
    resolved_chr = find_chr_bin(chr_bin)
    if resolved_chr is not None:
        resolved_chr = resolved_chr.resolve()

    extras: tuple[str, ...] = ()
    if extras_for == "internlm2":
        from .arch import missing_internlm_extras

        extras = tuple(missing_internlm_extras())

    return Machine(
        system=platform.system(),
        python=sys.version_info[:3],
        torch=torch_version,
        torch_cuda=torch_cuda,
        torch_hip=torch_hip,
        cuda_available=cuda_available,
        capability=capability,
        device_name=device_name,
        vram_total_mib=vram_total,
        smi_used_mib=_smi_used_mib(),
        mps=mps,
        nf4_ext=nf4_ext,
        nf4_ext_stale=stale,
        nf4_ext_abi_ok=abi_ok,
        host_cc=_host_compiler(inject=nf4_ext is None or not abi_ok),
        nvcc=shutil.which("nvcc"),
        chr_bin=resolved_chr,
        chr_runs=_chr_runs(resolved_chr) if resolved_chr else False,
        transformers=_installed("transformers"),
        safetensors=_installed("safetensors"),
        hf_hub=_installed("huggingface_hub"),
        ollama=shutil.which("ollama"),
        extras_missing=extras,
    )


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Verdict:
    """Can this box generate, and the copy that says why not."""

    generate: str  # "yes" | "experimental" | "no"
    arch: str  # ship | experimental | unmeasured | turing | rocm | darwin | ...
    line: str  # the one-liner from wave8-runtime.md §2.1
    refusal: str = ""  # the multi-line block from §8, for run's stderr

    @property
    def allowed(self) -> bool:
        return self.generate in ("yes", "experimental")


def allow_unmeasured(env: dict[str, str] | None = None) -> bool:
    """wave8-runtime.md D10. A developer switch, never a documented feature."""
    source = os.environ if env is None else env
    return source.get(OVERRIDE_ENV, "") not in ("", "0", "false", "False")


def verdict(m: Machine, *, override: bool | None = None) -> Verdict:
    """Map a :class:`Machine` onto the support matrix. Fails closed.

    Order matters. macOS is decided before torch is even considered: there is no
    CUDA kernel there whatever the wheel says, so "install a CUDA torch" would
    be the wrong next action (D5). ROCm reports ``torch.cuda.is_available() ==
    True``, so HIP is checked before CUDA.
    """
    unmeasured_ok = allow_unmeasured() if override is None else override

    if m.system == "Darwin":
        return Verdict("no", "darwin", messages.GENERATE_APPLE, messages.MACOS_RUN)

    if m.torch is None:
        return Verdict("no", "no-torch", messages.GENERATE_NO_TORCH, messages.NO_TORCH)

    if m.torch_hip:
        return Verdict("no", "rocm", messages.GENERATE_ROCM, messages.GENERATE_ROCM)

    if not m.cuda_available:
        if m.mps:
            return Verdict("no", "darwin", messages.GENERATE_APPLE, messages.MACOS_RUN)
        if m.torch_cuda is None:
            return Verdict("no", "cpu-torch", messages.GENERATE_CPU, messages.CPU_TORCH)
        return Verdict(
            "no",
            "cpu-only",
            messages.GENERATE_CPU,
            messages.GENERATE_CPU
            + "\nThis build of torch has CUDA, but no usable NVIDIA device was "
            "found.\nDeepfold does not install the NVIDIA driver.",
        )

    if m.capability is None:
        return Verdict(
            "no",
            "unsupported",
            "generate: no -- CUDA is available but the device capability could "
            "not be read.",
            messages.wrong_capability(None),
        )

    cap = tuple(m.capability)
    if cap == SHIP_CAPABILITY:
        return Verdict("yes", "ship", messages.GENERATE_SHIP)

    if cap in UNMEASURED_CAPABILITIES:
        if unmeasured_ok:
            return Verdict("experimental", "experimental", messages.GENERATE_EXPERIMENTAL)
        return Verdict(
            "no",
            "unmeasured",
            messages.generate_unmeasured(cap),
            messages.wrong_capability(cap),
        )

    if cap == (7, 5):
        return Verdict(
            "no", "turing", messages.GENERATE_TURING, messages.wrong_capability(cap)
        )

    return Verdict(
        "no",
        "unsupported",
        messages.generate_unsupported(cap),
        messages.wrong_capability(cap),
    )


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Check:
    tag: str  # ok | fail | warn | skip
    name: str
    detail: str = ""

    def __str__(self) -> str:
        line = f"[{self.tag}]".ljust(7) + self.name
        return f"{line} {self.detail}".rstrip()


MIN_PYTHON = (3, 11)


def compress_ok(m: Machine) -> bool:
    """``chr compress`` is the CPU product; it needs the Go binary and nothing else."""
    return m.chr_bin is not None and m.chr_runs


def checks(m: Machine, v: Verdict) -> list[Check]:
    """One tagged line per requirement of ``deepfold run`` (§6.2)."""
    out: list[Check] = []
    py = ".".join(str(p) for p in m.python)
    out.append(
        Check(
            "ok" if m.python[:2] >= MIN_PYTHON else "fail",
            f"python {py}",
            "" if m.python[:2] >= MIN_PYTHON else "3.11 is the measured line",
        )
    )

    if m.torch is None:
        out.append(Check("fail", "torch", "not importable"))
    else:
        cuda = m.torch_cuda or ("hip " + m.torch_hip if m.torch_hip else "cpu build")
        out.append(
            Check("ok" if m.torch_cuda else "fail", f"torch {m.torch}", f"({cuda})")
        )

    if m.torch_hip:
        out.append(Check("fail", "cuda available", f"torch.version.hip={m.torch_hip}"))
    else:
        out.append(
            Check("ok" if m.cuda_available else "fail", "cuda available", "")
        )

    if m.capability is not None:
        detail = m.device_name or ""
        if m.vram_total_mib:
            detail = f"{detail} ({m.vram_total_mib} MiB)".strip()
        tag = "ok" if v.arch in ("ship", "experimental") else "fail"
        out.append(Check(tag, f"GPU {m.sm}", detail))
    else:
        out.append(Check("fail", "GPU capability", "no CUDA device to ask"))
    out.append(Check("ok", "kernel target sm_86", f"(gencode {KERNEL_GENCODE})"))

    if m.chr_bin is None:
        out.append(Check("fail", "chr", "not found"))
    elif not m.chr_runs:
        out.append(Check("fail", "chr", f"{m.chr_bin} does not run"))
    else:
        out.append(Check("ok", "chr", str(m.chr_bin)))

    compiler = "cl.exe" if os.name == "nt" else "g++"
    if m.nf4_ext is not None and not m.nf4_ext_abi_ok:
        # The file is there but built for another interpreter; only a compiler
        # can turn this box green.
        detail = (
            f"{m.nf4_ext.name} is not importable from python "
            f"{m.python[0]}.{m.python[1]} ({_interpreter_tag()})"
        )
        if m.host_cc:
            out.append(Check("warn", "nf4 kernel", detail + "; first run will JIT"))
            out.append(Check("ok", "host compiler", m.host_cc))
        else:
            out.append(Check("fail", "nf4 kernel", detail + ", and no host compiler"))
    elif m.nf4_ext is not None:
        rel = m.nf4_ext.relative_to(REPO) if m.nf4_ext.is_relative_to(REPO) else m.nf4_ext
        out.append(
            Check(
                "warn" if m.nf4_ext_stale else "ok",
                "nf4 kernel",
                f"{rel}" + (" (stale: sources are newer)" if m.nf4_ext_stale else ""),
            )
        )
        out.append(
            Check("skip", compiler, "(not needed; prebuilt extension present)")
        )
    elif m.host_cc:
        out.append(Check("warn", "nf4 kernel", "not built; first run will JIT (~1 min)"))
        out.append(Check("ok", "host compiler", m.host_cc))
    else:
        out.append(Check("fail", "nf4 kernel", "no extension and no host compiler"))
    if os.name != "nt":
        out.append(
            Check("ok" if m.nvcc else "warn", "nvcc", m.nvcc or "not on PATH")
        )

    out.append(Check("ok" if m.transformers else "fail", "transformers", ""))
    out.append(Check("ok" if m.safetensors else "fail", "safetensors", ""))
    out.append(
        Check(
            "ok" if m.hf_hub else "warn",
            "huggingface_hub",
            "" if m.hf_hub else "(only needed to download a model)",
        )
    )
    out.append(
        Check(
            "ok" if m.ollama else "warn",
            "ollama",
            m.ollama or "not on PATH (only needed for from-ollama)",
        )
    )
    if m.extras_missing:
        out.append(
            Check("fail", "internlm extra", "missing " + ", ".join(m.extras_missing))
        )
    if m.smi_used_mib is not None:
        out.append(
            Check(
                "ok",
                "nvidia-smi used",
                f"{m.smi_used_mib} MiB (desktop and other processes included)",
            )
        )
    return out


def exit_code(m: Machine, v: Verdict, report: list[Check]) -> int:
    """§3.3. A refusal is never folded into 0, and 3 is never green generate."""
    failed = [c for c in report if c.tag == "fail"]
    if v.allowed and not failed:
        return 0
    if v.arch in FIXABLE:
        return 2
    return 3 if compress_ok(m) else 1


def render(m: Machine, v: Verdict, report: list[Check]) -> str:
    lines = [str(c) for c in report]
    lines.append("")
    lines.append(v.line)
    code = exit_code(m, v, report)
    if code == 0:
        lines.append(messages.DOCTOR_OK)
    else:
        if v.refusal:
            lines.append("")
            lines.append(v.refusal)
        first = next((c for c in report if c.tag == "fail"), None)
        if first is not None and not v.refusal:
            lines.append("")
            lines.append(_fail_copy(m, first))
        if code == 2:
            # "generate: yes" plus a failed row would otherwise read as green.
            lines.append("")
            lines.append(
                "doctor: this machine could run, but the install is incomplete "
                "(see the [fail] rows)"
            )
        elif compress_ok(m) and not v.allowed:
            lines.append("")
            lines.append(messages.DOCTOR_COMPRESS_ONLY)
    return "\n".join(lines)


def _fail_copy(m: Machine, check: Check) -> str:
    """The human block for the first failing row (§8), not a bare tag."""
    name = check.name
    if name.startswith("chr"):
        return messages.missing_chr()
    if name.startswith("nf4 kernel"):
        return messages.NO_COMPILER if os.name == "nt" else messages.NO_COMPILER_POSIX
    if name.startswith("torch"):
        return messages.CPU_TORCH if m.torch else messages.NO_TORCH
    if name.startswith("internlm"):
        return messages.INTERNLM_EXTRAS
    if name in ("transformers", "safetensors"):
        return f"{name} is missing: pip install {name}"
    return f"{name}: {check.detail}".rstrip()


def doctor(args) -> int:
    """``deepfold doctor``. Writes the report to stdout (§6.1: pick one, keep it)."""
    extras_for = None
    if args.model:
        from .arch import gate

        checked = gate(args.model)
        if not checked.ok:
            print(checked.reason, file=sys.stderr)
            return 1
        extras_for = checked.model_type

    m = probe(chr_bin=args.chr_bin, extras_for=extras_for)
    v = verdict(m)

    if args.compress_only:
        ok = compress_ok(m)
        chr_line = str(m.chr_bin) if m.chr_bin else "not found"
        print(str(Check("ok" if ok else "fail", "chr", chr_line)))
        print(f"compress: {'yes' if ok else 'no'}")
        if not ok:
            print("", file=sys.stderr)
            print(messages.missing_chr(), file=sys.stderr)
        return 0 if ok else 1

    report = checks(m, v)
    print(render(m, v, report))
    return exit_code(m, v, report)
