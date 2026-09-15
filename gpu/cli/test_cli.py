"""Acceptance for ``gpu/cli`` with no GPU, no torch and no model on disk.

The doctor verdict is a pure function of a hand-built :class:`Machine`, so
"this GPU is sm_90, refuse" is a test and not a story:

    python gpu/cli/test_cli.py
    python -m pytest gpu/cli/test_cli.py

The point of this file is that every refusal is reachable on a laptop. The
doctor verdict is a pure function of a hand-built :class:`Machine`:

1. the architecture gate accepts the graphs the loop can drive (qwen2 /
   internlm2 / llama / mistral without a live sliding window), fails closed on a
   family it can name (gemma_gelu, phi3_concat, MoE, vision), and *defers* an
   unrecognised model_type to the walker instead of guessing either way;
2. GGUF and Ollama blob paths are refused **without being opened**;
3. generate is *experimental* on sm_80 / sm_89 (Ampere-family, unmeasured plate),
   refused on sm_75 (no BF16 tensor cores), Hopper, ROCm, macOS, CPU torch and a
   missing torch -- each with its own copy;
4. sm_89 is never ``ship``; the 3080 plate stays sm_86;
5. doctor exit codes separate a broken install (2) from a machine whose class
   refuses generate (3); 3 is not green;
6. ``run`` does not call ``chr compress`` when a ``.chr`` was given;
7. the argparse surface is setup / doctor / pull / compress / run / chat /
   test / from-ollama (K5 user CLI; WAVE 7 freeze plus those commands).
"""

from __future__ import annotations

import builtins
import io
import json
import os
import struct
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.cli import chat as chat_mod  # noqa: E402
from gpu.cli import from_ollama as from_ollama_mod  # noqa: E402
from gpu.cli import hub as hub_mod  # noqa: E402
from gpu.cli import messages, paths, run as run_mod  # noqa: E402
from gpu.cli import selftest as selftest_mod  # noqa: E402
from gpu.cli import setup_env as setup_mod  # noqa: E402
from gpu.cli.ollama_map import ResolveError, hf_id_list, resolve  # noqa: E402
from gpu.cli.arch import gate  # noqa: E402
from gpu.cli.doctor import (  # noqa: E402
    Machine,
    checks,
    compress_ok,
    exit_code,
    probe,
    render,
    verdict,
)
from gpu.cli.main import build_parser, main  # noqa: E402
from gpu.tests.skips import Skip, requires_sm86  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _fake_model(
    tmp: Path,
    model_type: str | None,
    *,
    config: bool = True,
    name: str | None = None,
    **fields: object,
) -> Path:
    """A HuggingFace directory with nothing in it but a config and a tokenizer."""
    d = tmp / (name or model_type or "noconfig")
    d.mkdir(parents=True, exist_ok=True)
    if config:
        payload: dict[str, object] = {"hidden_size": 8, "num_hidden_layers": 1}
        if model_type:
            payload["model_type"] = model_type
        payload.update(fields)
        (d / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return d


def _write_chr(path: Path, *, hidden: int, layers: int, vocab: int) -> Path:
    """A minimal but valid CHR0 file with the root fields the matcher compares.

    Hand-rolled from stdlib on purpose: this suite must run with no torch and
    no numpy. One 64x128 NF4 tensor is the smallest header ``load_header``
    accepts (an empty ``tensors`` map is rejected).
    """
    data, scale, end = 4096, 4096 + 4096, 4096 + 4096 + 256
    root = {
        "magic": "CHR0",
        "version": 1,
        "arch": "qwen2",
        "hidden_size": hidden,
        "intermediate_size": hidden * 4,
        "num_layers": layers,
        "vocab_size": vocab,
        "tile": {"row": 64, "col_group": 8},
        "tensors": {
            "model.layers.0.self_attn.q_proj": {
                "kind": "q",
                "codec": "nf4",
                "layer": 0,
                "shape": [64, 128],
                "group_size": 64,
                "data": [data, scale],
                "scale": [scale, end],
            }
        },
    }
    js = json.dumps(root, separators=(",", ":")).encode("utf-8")
    body = struct.pack("<Q", len(js)) + js
    path.write_bytes(body + b"\x00" * (end - len(body)))
    return path


def _write_vq_chr(path: Path, *, hidden: int, layers: int, vocab: int) -> Path:
    """Same as :func:`_write_chr` but the one tensor is ``codec=vq``."""
    codebook_end = 4096 + 8192
    index_end = codebook_end + 64 * (128 // 8) * 2
    root = {
        "magic": "CHR0",
        "version": 1,
        "arch": "qwen2",
        "hidden_size": hidden,
        "intermediate_size": hidden * 4,
        "num_layers": layers,
        "vocab_size": vocab,
        "tile": {"row": 64, "col_group": 8},
        "tensors": {
            "model.layers.0.self_attn.q_proj": {
                "kind": "q",
                "codec": "vq",
                "layer": 0,
                "shape": [64, 128],
                "group_size": 8,
                "n_codebooks": 2,
                "codebook_bits": 8,
                "codebook": [4096, codebook_end],
                "index": [codebook_end, index_end],
            }
        },
    }
    js = json.dumps(root, separators=(",", ":")).encode("utf-8")
    body = struct.pack("<Q", len(js)) + js
    path.write_bytes(body + b"\x00" * (index_end - len(body)))
    return path


def _config(model: Path, **fields) -> Path:
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps(fields), encoding="utf-8")
    return model


class _CompressArgs:
    """What argparse would have produced for ``deepfold compress``."""

    def __init__(self, **over) -> None:
        self.__dict__.update(
            inp=None, out=None, chr_bin=None, force=False, quiet=True, codec="auto",
            vram_mib=None,
        )
        self.__dict__.update(over)


def _ship(**over) -> Machine:
    """The machine of record: Windows, RTX 3080, prebuilt extension, chr present."""
    base = dict(
        system="Windows",
        python=(3, 11, 14),
        torch="2.5.1+cu124",
        torch_cuda="12.4",
        cuda_available=True,
        capability=(8, 6),
        device_name="NVIDIA GeForce RTX 3080",
        vram_total_mib=12288,
        nf4_ext=Path("gpu/nf4/chr_nf4_ext.cp311-win_amd64.pyd"),
        host_cc=None,
        chr_bin=Path("C:/dev/deep-fold/chr.exe"),
        chr_runs=True,
        transformers=True,
        safetensors=True,
        hf_hub=True,
    )
    base.update(over)
    return Machine(**base)  # type: ignore[arg-type]


class _Spy:
    """Records calls instead of making them."""

    def __init__(self, result: object = 0) -> None:
        self.calls: list[tuple] = []
        self.result = result

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return self.result


class _RunArgs:
    """What argparse would have produced for ``deepfold run``."""

    def __init__(self, **over) -> None:
        self.__dict__.update(
            model=None,
            chr=None,
            chr_bin=None,
            prompt=None,
            max_new_tokens=8,
            max_seq=128,
            raw=False,
            warmup=False,
            no_compress=False,
            quiet=True,
            debug=False,
            codec="auto",
            max_resident_mib=None,
        )
        self.__dict__.update(over)


# --------------------------------------------------------------------------- #
# 1. architecture gate
# --------------------------------------------------------------------------- #


def test_gate_accepts_wired_layouts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qwen = gate(str(_fake_model(root, "qwen2")))
        internlm = gate(str(_fake_model(root, "internlm2")))
    assert qwen.ok and qwen.model_type == "qwen2"
    assert not qwen.trust_remote_code, "qwen2 must not need remote code"
    assert internlm.ok and internlm.trust_remote_code, "internlm2 needs remote code"


def test_gate_accepts_llama_without_a_sliding_window() -> None:
    """wave10 P1: Llama and Mistral ride the Qwen2 split-SwiGLU family."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        llama = gate(str(_fake_model(root, "llama", hidden_act="silu")))
        mistral = gate(
            str(_fake_model(root, "mistral", hidden_act="silu", sliding_window=None))
        )
    assert llama.ok and llama.family == "llama_swiglu", llama.reason
    assert not llama.trust_remote_code
    assert mistral.ok and mistral.family == "llama_swiglu", mistral.reason


def test_gate_accepts_qwen25s_inert_sliding_window() -> None:
    """The measured 3B config: sliding_window=32768 with the switch off."""
    with tempfile.TemporaryDirectory() as tmp:
        g = gate(
            str(
                _fake_model(
                    Path(tmp), "qwen2", sliding_window=32768, use_sliding_window=False
                )
            )
        )
    assert g.ok and g.family == "llama_swiglu", g.reason


def test_gate_refuses_a_live_sliding_window() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        g = gate(str(_fake_model(Path(tmp), "mistral", sliding_window=4096)))
    assert not g.ok
    assert "sliding_window=4096" in g.reason and "SWA" in g.reason
    assert "model_type=mistral" in g.reason


def test_gate_refuses_named_families_by_family_id() -> None:
    """The copy points at the missing glue family, not at a model_type allowlist."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gemma = gate(str(_fake_model(root, "gemma", hidden_act="gelu_pytorch_tanh")))
        phi3 = gate(str(_fake_model(root, "phi3", hidden_act="silu")))
        moe = gate(str(_fake_model(root, "qwen2_moe")))
        experts = gate(
            str(_fake_model(root, "qwen2", name="experts", num_experts=60))
        )
    assert not gemma.ok and "gemma_gelu" in gemma.reason
    assert "llama_swiglu and internlm_gqa" in gemma.reason
    assert not phi3.ok and "phi3_concat" in phi3.reason
    assert not moe.ok and "MoE experts are not in this wave" in moe.reason
    # A wired model_type does not buy a pass: the expert count refuses too.
    assert not experts.ok and "num_experts=60" in experts.reason


def test_gate_refuses_a_vision_tower() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        g = gate(str(_fake_model(Path(tmp), "qwen2_vl", name="vl")))
    assert not g.ok and "vision" in g.reason.lower()


def test_gate_defers_an_unknown_model_type_to_the_walker() -> None:
    """model_type is not evidence of "no": the module tree decides at load time."""
    with tempfile.TemporaryDirectory() as tmp:
        g = gate(str(_fake_model(Path(tmp), "olmo", hidden_act="silu")))
    assert g.ok, g.reason
    assert g.family is None
    assert "attach() will walk the module tree" in g.note
    assert "model_type=olmo" in g.note


def test_gate_refuses_missing_config_and_missing_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        no_cfg = gate(str(_fake_model(root, None, config=False)))
        absent = gate(str(root / "nope"))
    assert not no_cfg.ok and "config.json" in no_cfg.reason
    assert not absent.ok and "does not exist" in absent.reason


def test_gate_refuses_bare_chr_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "qwen25-3b.nf4.chr"
        f.write_bytes(b"CHR0")
        g = gate(str(f))
    assert not g.ok
    assert "weights only" in g.reason


# --------------------------------------------------------------------------- #
# 2. GGUF / Ollama: refused, and never opened
# --------------------------------------------------------------------------- #


def test_gguf_paths_are_refused_without_being_opened() -> None:
    suspects = [
        r"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf",
        r"C:\Users\Professional\.ollama\models\blobs\sha256-abc123",
        "/home/p/.ollama/models/blobs/sha256-deadbeef",
        "/var/lib/llama/models/blobs/sha256-1",
    ]
    real_open = builtins.open
    opened: list[str] = []

    def watched(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    builtins.open = watched  # type: ignore[assignment]
    try:
        for suspect in suspects:
            assert paths.looks_like_gguf(suspect), suspect
            g = gate(suspect)
            assert not g.ok, suspect
            assert "cannot load GGUF" in g.reason, suspect
    finally:
        builtins.open = real_open  # type: ignore[assignment]

    leaked = [p for p in opened if any(k in p.lower() for k in (".gguf", "blobs", "sha256-"))]
    assert not leaked, f"the gate opened a blob path: {leaked}"


def test_a_normal_directory_is_not_mistaken_for_a_blob() -> None:
    assert not paths.looks_like_gguf(r"C:\dev\models\Qwen2.5-3B-Instruct")
    assert not paths.looks_like_gguf("/models/blobstore-notollama/x")


# --------------------------------------------------------------------------- #
# 3 + 4. the generate verdict
# --------------------------------------------------------------------------- #


def test_sm86_ships() -> None:
    v = verdict(_ship(), override=False)
    assert v.generate == "yes" and v.arch == "ship"
    assert v.line == messages.GENERATE_SHIP


def test_ada_is_experimental_and_exit_zero() -> None:
    m = _ship(capability=(8, 9), device_name="RTX 4070")
    v = verdict(m, override=False)
    assert v.allowed and v.arch == "experimental"
    assert _code(m) == 0


def test_ampere_family_is_experimental_not_ship() -> None:
    """sm_80 / sm_87 / sm_89 generate without an env override; never 'ship'."""
    from gpu.ampere_gencode import FAMILY_CAPABILITIES, MEASURED_CAPABILITY

    for cap in sorted(FAMILY_CAPABILITIES - {MEASURED_CAPABILITY}):
        v = verdict(_ship(capability=cap, device_name="family"), override=False)
        assert v.generate == "experimental", cap
        assert v.arch == "experimental", cap
        assert v.allowed, cap
        assert messages.GENERATE_SHIP not in v.line, cap
        assert "experimental" in v.line, cap


def test_experimental_never_becomes_ship() -> None:
    v = verdict(_ship(capability=(8, 9)), override=True)
    assert v.generate == "experimental" and v.arch == "experimental"
    assert v.generate != "yes", "Ada must not claim the 3080 plate"


def test_turing_is_refused_for_the_right_reason() -> None:
    v = verdict(_ship(capability=(7, 5), device_name="RTX 2080"), override=False)
    assert v.generate == "no" and v.arch == "turing"
    assert "BF16 tensor cores" in v.line and "cp.async" in v.line


def test_rocm_is_refused_even_though_cuda_looks_available() -> None:
    v = verdict(
        _ship(torch_hip="6.0", torch_cuda=None, capability=(9, 0)), override=False
    )
    assert v.generate == "no" and v.arch == "rocm"
    assert "ROCm is not implemented" in v.line


def test_apple_is_compress_only() -> None:
    v = verdict(
        Machine(
            system="Darwin",
            torch="2.5.1",
            torch_cuda=None,
            cuda_available=False,
            mps=True,
            chr_bin=Path("/usr/local/bin/chr"),
            chr_runs=True,
        ),
        override=False,
    )
    assert v.generate == "no" and v.arch == "darwin"
    assert "does not run on M2" in v.line
    assert "no CUDA kernel on macOS" in v.refusal


def test_cpu_torch_points_at_the_cu124_index() -> None:
    v = verdict(_ship(torch="2.5.1+cpu", torch_cuda=None, cuda_available=False,
                      capability=None), override=False)
    assert v.arch == "cpu-torch"
    assert "download.pytorch.org/whl/cu124" in v.refusal


def test_missing_torch_is_a_verdict_not_a_traceback() -> None:
    v = verdict(Machine(torch=None), override=False)
    assert v.generate == "no" and v.arch == "no-torch"
    assert "not importable" in v.line


def test_newer_nvidia_arch_is_out_of_the_matrix() -> None:
    v = verdict(_ship(capability=(9, 0), device_name="H100"), override=False)
    assert v.generate == "no" and v.arch == "unsupported"
    assert "sm_90" in v.line


def test_kernel_gencode_is_ampere_family_fatbinary() -> None:
    from gpu.ampere_gencode import KERNEL_GENCODE, NVCC_GENCODE_FLAGS, nvcc_cflags

    flags = " ".join(NVCC_GENCODE_FLAGS)
    assert "sm_80" in flags and "sm_86" in flags and "sm_89" in flags
    assert "compute_80" in flags
    assert "sm_90" not in flags and "sm_75" not in flags
    assert "PTX" in KERNEL_GENCODE
    cflags = " ".join(nvcc_cflags())
    assert flags in cflags


# --------------------------------------------------------------------------- #
# 5. doctor exit codes and report
# --------------------------------------------------------------------------- #


def _code(m: Machine, *, override: bool = False) -> int:
    v = verdict(m, override=override)
    return exit_code(m, v, checks(m, v))


def test_doctor_green_only_on_the_machine_of_record() -> None:
    assert _code(_ship()) == 0


def test_doctor_two_is_a_broken_install_on_a_card_that_could_run() -> None:
    assert _code(_ship(chr_bin=None, chr_runs=False)) == 2, "sm_86 without chr"
    assert _code(_ship(nf4_ext=None, host_cc=None)) == 2, "sm_86 without kernel"
    assert _code(_ship(torch=None)) == 2, "no torch at all"
    assert _code(_ship(torch="2.5.1+cpu", torch_cuda=None, cuda_available=False,
                       capability=None)) == 2, "CPU torch"


def test_doctor_three_is_refused_generate_with_working_compress() -> None:
    hopper = _ship(capability=(9, 0), device_name="H100")
    assert _code(hopper) == 3
    assert compress_ok(hopper), "compress is still the CPU product"
    mac = Machine(system="Darwin", torch="2.5.1", cuda_available=False, mps=True,
                  chr_bin=Path("/usr/local/bin/chr"), chr_runs=True)
    assert _code(mac) == 3


def test_doctor_one_when_nothing_works() -> None:
    assert _code(Machine(system="Darwin", torch=None)) == 1


def test_doctor_report_never_calls_a_refusal_green() -> None:
    m = _ship(capability=(9, 0), device_name="H100")
    v = verdict(m, override=False)
    text = render(m, v, checks(m, v))
    assert messages.DOCTOR_OK not in text
    assert "sm_90" in text
    assert messages.DOCTOR_COMPRESS_ONLY in text


def test_doctor_skips_the_compiler_when_a_prebuilt_extension_exists() -> None:
    m = _ship()
    rows = {c.name.split()[0]: c for c in checks(m, verdict(m, override=False))}
    compiler = "cl.exe" if os.name == "nt" else "g++"
    assert rows[compiler].tag == "skip"
    assert rows["nf4"].tag == "ok"


def test_stale_extension_warns_and_stays_loadable() -> None:
    m = _ship(nf4_ext_stale=True)
    v = verdict(m, override=False)
    rows = [c for c in checks(m, v) if c.name.startswith("nf4")]
    assert rows and rows[0].tag == "warn", "stale is a warning, not a failure"
    assert exit_code(m, v, checks(m, v)) == 0


def test_doctor_names_the_chr_binary_that_won() -> None:
    m = _ship()
    line = next(c for c in checks(m, verdict(m, override=False)) if c.name == "chr")
    assert "chr.exe" in line.detail


# --------------------------------------------------------------------------- #
# 6. run wiring
# --------------------------------------------------------------------------- #


def test_run_does_not_compress_when_chr_is_given() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2")
        packed = root / "given.nf4.chr"
        packed.write_bytes(b"CHR0")

        spy = _Spy(0)
        original = run_mod.compress_to
        run_mod.compress_to = spy  # type: ignore[assignment]
        try:
            resolved, code = run_mod._resolve_weights(
                _RunArgs(chr=str(packed)), str(model)
            )
        finally:
            run_mod.compress_to = original  # type: ignore[assignment]

    assert code == 0 and resolved == packed
    assert spy.calls == [], "chr compress was invoked although --chr existed"


def test_run_compresses_once_when_no_chr_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2")
        home = root / "home"
        spy = _Spy(0)
        original = run_mod.compress_to
        run_mod.compress_to = spy  # type: ignore[assignment]
        os.environ["DEEPFOLD_HOME"] = str(home)
        os.environ["DEEPFOLD_CHR_BIN"] = str(_REPO / "chr.exe")
        try:
            resolved, code = run_mod._resolve_weights(_RunArgs(), str(model))
        finally:
            run_mod.compress_to = original  # type: ignore[assignment]
            os.environ.pop("DEEPFOLD_HOME", None)
            os.environ.pop("DEEPFOLD_CHR_BIN", None)

    assert len(spy.calls) == 1, "first run must pack exactly once"
    assert code == 0
    assert resolved is not None and resolved.name.endswith(".nf4.chr")


def test_run_auto_refuses_when_nf4_would_not_fit() -> None:
    """H2-1: 32B-shaped config on 12 GB packs NF4 overflow, never VQ."""
    qwen32 = dict(
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        vocab_size=152064,
        tie_word_embeddings=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2", **qwen32)
        home = root / "home"
        spy = _Spy(0)
        original = run_mod.compress_to
        run_mod.compress_to = spy  # type: ignore[assignment]
        os.environ["DEEPFOLD_HOME"] = str(home)
        os.environ["DEEPFOLD_CHR_BIN"] = str(_REPO / "chr.exe")
        err = io.StringIO()
        try:
            with redirect_stderr(err):
                resolved, code = run_mod._resolve_weights(
                    _RunArgs(), str(model), vram_mib=12288
                )
        finally:
            run_mod.compress_to = original  # type: ignore[assignment]
            os.environ.pop("DEEPFOLD_HOME", None)
            os.environ.pop("DEEPFOLD_CHR_BIN", None)

    text = err.getvalue()
    assert code == 0 and resolved is not None
    assert resolved.name.endswith(".nf4.chr"), resolved.name
    assert len(spy.calls) == 1, "32B auto must pack NF4 overflow once"
    assert spy.calls[0][1].get("codec") == "nf4"
    assert "overflow" in text and "H2" in text
    assert "vq" not in (spy.calls[0][1].get("codec") or "")


def test_run_codec_vq_flag_packs_vq_on_a_small_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2")
        home = root / "home"
        spy = _Spy(0)
        original = run_mod.compress_to
        run_mod.compress_to = spy
        os.environ["DEEPFOLD_HOME"] = str(home)
        os.environ["DEEPFOLD_CHR_BIN"] = str(_REPO / "chr.exe")
        try:
            resolved, code = run_mod._resolve_weights(
                _RunArgs(codec="vq"), str(model), vram_mib=12288
            )
        finally:
            run_mod.compress_to = original
            os.environ.pop("DEEPFOLD_HOME", None)
            os.environ.pop("DEEPFOLD_CHR_BIN", None)
    assert code == 0 and resolved is not None and resolved.name.endswith(".vq2.chr")
    assert spy.calls[0][1].get("codec") == "vq"


def test_run_codec_vq_uses_existing_vq2_without_packing() -> None:
    """``--codec vq`` may load a sibling ``.vq2.chr``; auto must not."""
    qwen32 = dict(
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        vocab_size=152064,
        tie_word_embeddings=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2", name="Qwen2.5-32B", **qwen32)
        mine = _write_vq_chr(root / "qwen25-32b.vq2.chr", hidden=5120, layers=64, vocab=152064)
        spy = _Spy(0)
        original = run_mod.compress_to
        run_mod.compress_to = spy  # type: ignore[assignment]
        os.environ["DEEPFOLD_HOME"] = str(root / "home")
        os.environ.pop("DEEPFOLD_CHR", None)
        try:
            resolved, code = run_mod._resolve_weights(
                _RunArgs(codec="vq"), str(model), vram_mib=12288
            )
        finally:
            run_mod.compress_to = original  # type: ignore[assignment]
            os.environ.pop("DEEPFOLD_HOME", None)
    assert code == 0 and resolved == mine, f"picked {resolved}"
    assert spy.calls == [], "matching .vq2.chr was on disk; nothing should pack"


def test_run_refuses_a_gguf_chr_flag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "qwen2")
        err = io.StringIO()
        with redirect_stderr(err):
            resolved, code = run_mod._resolve_weights(
                _RunArgs(chr="C:/models/x.gguf"), str(model)
            )
    assert resolved is None and code == 1
    assert "cannot load GGUF" in err.getvalue()


def test_run_refuses_unknown_architecture_before_touching_the_gpu() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "mixtral")
        err = io.StringIO()
        probed = _Spy(_ship())
        original = run_mod.probe
        run_mod.probe = probed  # type: ignore[assignment]
        try:
            with redirect_stderr(err):
                code = run_mod.run(_RunArgs(model=str(model)))
        finally:
            run_mod.probe = original  # type: ignore[assignment]
    assert code == 1
    assert "model_type=mixtral" in err.getvalue()
    assert probed.calls == [], "the gate must lose before the device is probed"


def test_run_refuses_when_doctor_refuses() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "qwen2")
        err = io.StringIO()
        original = run_mod.probe
        run_mod.probe = _Spy(_ship(capability=(9, 0), device_name="H100"))  # type: ignore[assignment]
        try:
            with redirect_stderr(err):
                code = run_mod.run(_RunArgs(model=str(model)))
        finally:
            run_mod.probe = original  # type: ignore[assignment]
    text = err.getvalue()
    assert code == 1
    assert "sm_90" in text and "deepfold doctor" in text


def test_run_allows_ada_experimental() -> None:
    """Ada is generate-allowed; doctor exit 3 is Hopper, not sm_89."""
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "qwen2")
        fake_chr = Path(tmp) / "x.nf4.chr"
        fake_chr.write_bytes(b"CHR0")
        err = io.StringIO()
        original_probe = run_mod.probe
        original_resolve = run_mod._resolve_weights
        original_generate = run_mod._generate
        resolve_spy = _Spy((fake_chr, 0))
        generate_spy = _Spy(0)
        run_mod.probe = _Spy(_ship(capability=(8, 9), device_name="RTX 4070"))  # type: ignore[assignment]
        run_mod._resolve_weights = resolve_spy  # type: ignore[assignment]
        run_mod._generate = generate_spy  # type: ignore[assignment]
        try:
            with redirect_stderr(err):
                code = run_mod.run(_RunArgs(model=str(model)))
        finally:
            run_mod.probe = original_probe  # type: ignore[assignment]
            run_mod._resolve_weights = original_resolve  # type: ignore[assignment]
            run_mod._generate = original_generate  # type: ignore[assignment]
    assert code == 0
    assert "experimental" in err.getvalue()
    assert generate_spy.calls, "Ada must reach generate, not the class-3 refuse"


def test_both_fit_only_when_a_bf16_copy_would_also_fit() -> None:
    # 3B: 1,563 MiB packed -> ~5,886 MiB dense, plus overhead, under 12,288.
    assert run_mod._both_fit(1563.0, 12288)
    # 14B: 7,483 MiB packed -> ~28,172 MiB dense. It does not fit.
    assert not run_mod._both_fit(7483.0, 12288)
    assert not run_mod._both_fit(1563.0, None)


# --------------------------------------------------------------------------- #
# 7. argparse surface
# --------------------------------------------------------------------------- #


def test_parser_has_the_frozen_subcommands() -> None:
    ap = build_parser()
    action = next(a for a in ap._actions if a.dest == "command")
    assert {
        "setup",
        "doctor",
        "pull",
        "compress",
        "run",
        "chat",
        "test",
        "from-ollama",
    } <= set(action.choices)


def test_parser_run_flags() -> None:
    args = build_parser().parse_args(
        ["run", "--model", "D:/m", "--chr", "D:/m.chr", "--prompt", "hi",
         "--max-new-tokens", "16"]
    )
    assert args.command == "run"
    assert (args.model, args.chr, args.prompt, args.max_new_tokens) == (
        "D:/m", "D:/m.chr", "hi", 16,
    )
    assert args.warmup is True and args.raw is False
    assert args.codec == "auto"
    assert args.max_resident_mib is None
    vq = build_parser().parse_args(["run", "--model", "D:/m", "--codec", "vq"])
    assert vq.codec == "vq"
    cap = build_parser().parse_args(
        ["run", "--model", "D:/m", "--max-resident-mib", "2048"]
    )
    assert cap.max_resident_mib == 2048
    packed = build_parser().parse_args(["compress", "--in", "D:/m"])
    assert packed.codec == "auto"


_QWEN3B_FIT = dict(
    hidden_size=2048,
    intermediate_size=11008,
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=2,
    vocab_size=151936,
    tie_word_embeddings=True,
)
_QWEN32B_OVERFLOW = dict(
    hidden_size=5120,
    intermediate_size=27648,
    num_hidden_layers=64,
    num_attention_heads=40,
    num_key_value_heads=8,
    vocab_size=152064,
    tie_word_embeddings=False,
)


def test_max_resident_mib_flag_is_bytes() -> None:
    cap = run_mod._max_resident_bytes(
        _RunArgs(max_resident_mib=512), "unused", Path("x.chr"), 12288
    )
    assert cap == 512 * 1024 * 1024


def test_3b_auto_without_flag_does_not_pass_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "qwen2", **_QWEN3B_FIT)
        dummy = Path(tmp) / "unused.nf4.chr"
        cap = run_mod._max_resident_bytes(
            _RunArgs(codec="auto", max_seq=512), str(model), dummy, 12288
        )
    assert cap is None


def test_codec_vq_no_auto_overflow_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "qwen2", name="Qwen2.5-32B", **_QWEN32B_OVERFLOW)
        cap = run_mod._max_resident_bytes(
            _RunArgs(codec="vq", max_seq=2048), str(model), Path("x.vq2.chr"), 12288
        )
    assert cap is None


def test_32b_auto_overflow_computes_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "qwen2", name="Qwen2.5-32B", **_QWEN32B_OVERFLOW)
        original = run_mod._overflow_cap_from_chr
        run_mod._overflow_cap_from_chr = lambda *a, **k: 4242  # type: ignore[assignment]
        try:
            cap = run_mod._max_resident_bytes(
                _RunArgs(codec="auto", max_seq=2048),
                str(model),
                Path("x.nf4.chr"),
                12288,
            )
        finally:
            run_mod._overflow_cap_from_chr = original  # type: ignore[assignment]
    assert cap == 4242


def test_parser_doctor_flags() -> None:
    args = build_parser().parse_args(["doctor", "--compress-only"])
    assert args.command == "doctor" and args.compress_only is True
    assert build_parser().parse_args(["doctor"]).model is None


def test_parser_compress_requires_in() -> None:
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            build_parser().parse_args(["compress"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("compress without --in should not parse")
    assert "--in" in err.getvalue()


def test_no_subcommand_prints_help_and_fails() -> None:
    err = io.StringIO()
    with redirect_stderr(err):
        code = main([])
    assert code == 1
    assert "doctor" in err.getvalue()


def _watch_open() -> tuple[list[str], object]:
    opened: list[str] = []
    real_open = builtins.open

    def watched(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    builtins.open = watched  # type: ignore[assignment]
    return opened, real_open


def test_from_ollama_refuses_without_reading_blobs() -> None:
    opened, real_open = _watch_open()
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = main(["from-ollama", "llama3.1:8b"])
    finally:
        builtins.open = real_open  # type: ignore[assignment]
    text = err.getvalue()
    assert code == 1
    assert "unknown tag 'llama3.1:8b'" in text
    assert "never reads ~/.ollama" in text
    assert "qwen2.5:3b" in text
    assert not [p for p in opened if ".ollama" in p.lower()]
    assert not [p for p in opened if "blobs" in p.lower() or p.lower().endswith(".gguf")]


def test_from_ollama_qwen_yes_stub_snapshot_download() -> None:
    called: list[dict[str, str]] = []

    def stub(*, repo_id: str, local_dir: str) -> str:
        called.append({"repo_id": repo_id, "local_dir": local_dir})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        return local_dir

    opened, real_open = _watch_open()
    original = from_ollama_mod._snapshot_download
    from_ollama_mod._snapshot_download = stub  # type: ignore[assignment]
    err, out = io.StringIO(), io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "hf"
            with redirect_stderr(err), redirect_stdout(out):
                code = main(["from-ollama", "qwen2.5:3b", "--yes", "--dir", str(dest)])
    finally:
        from_ollama_mod._snapshot_download = original  # type: ignore[assignment]
        builtins.open = real_open  # type: ignore[assignment]
    assert code == 0, err.getvalue()
    assert called == [{"repo_id": "Qwen/Qwen2.5-3B-Instruct", "local_dir": str(dest)}]
    assert f"deepfold run --model {dest}" in out.getvalue()
    assert "Qwen/Qwen2.5-3B-Instruct" in err.getvalue()
    assert not [p for p in opened if ".ollama" in p.lower()]


def test_from_ollama_without_yes_and_no_tty_downloads_nothing() -> None:
    called: list[dict[str, str]] = []

    def stub(*, repo_id: str, local_dir: str) -> str:
        called.append({"repo_id": repo_id, "local_dir": local_dir})
        return local_dir

    original = from_ollama_mod._snapshot_download
    from_ollama_mod._snapshot_download = stub  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "hf"
            stdin = sys.stdin
            sys.stdin = io.StringIO("")
            try:
                with redirect_stderr(err), redirect_stdout(io.StringIO()):
                    code = main(["from-ollama", "qwen2.5:3b", "--dir", str(dest)])
            finally:
                sys.stdin = stdin
    finally:
        from_ollama_mod._snapshot_download = original  # type: ignore[assignment]
    assert code == 1
    assert called == []
    assert "No TTY and no --yes" in err.getvalue()


def test_from_ollama_blob_path_is_gguf_copy_and_not_opened() -> None:
    opened, real_open = _watch_open()
    blob = r"C:\Users\x\.ollama\models\blobs\sha256-dead"
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = main(["from-ollama", blob])
    finally:
        builtins.open = real_open  # type: ignore[assignment]
    assert code == 1
    assert "cannot load GGUF" in err.getvalue()
    assert "unknown tag" not in err.getvalue()
    assert blob not in opened
    assert not [p for p in opened if ".ollama" in p.lower() or "sha256-dead" in p.lower()]


def test_from_ollama_hf_must_match_the_tag() -> None:
    err = io.StringIO()
    with redirect_stderr(err):
        code = main(
            ["from-ollama", "qwen2.5:3b", "--hf", "internlm/internlm2_5-20b-chat"]
        )
    assert code == 1
    assert "does not match tag 'qwen2.5:3b'" in err.getvalue()
    assert "Qwen/Qwen2.5-3B-Instruct" in err.getvalue()


def test_from_ollama_unknown_tag_hf_internlm_is_allowed() -> None:
    called: list[dict[str, str]] = []

    def stub(*, repo_id: str, local_dir: str) -> str:
        called.append({"repo_id": repo_id, "local_dir": local_dir})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        return local_dir

    original = from_ollama_mod._snapshot_download
    from_ollama_mod._snapshot_download = stub  # type: ignore[assignment]
    err, out = io.StringIO(), io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "internlm"
            with redirect_stderr(err), redirect_stdout(out):
                code = main(
                    [
                        "from-ollama",
                        "unknown",
                        "--hf",
                        "internlm/internlm2_5-20b-chat",
                        "--yes",
                        "--dir",
                        str(dest),
                    ]
                )
    finally:
        from_ollama_mod._snapshot_download = original  # type: ignore[assignment]
    assert code == 0, err.getvalue()
    assert called == [{"repo_id": "internlm/internlm2_5-20b-chat", "local_dir": str(dest)}]


def test_from_ollama_missing_hub_names_the_extra() -> None:
    original_missing = from_ollama_mod._hub_missing
    original_snap = from_ollama_mod._snapshot_download
    from_ollama_mod._hub_missing = lambda: True  # type: ignore[assignment]
    called: list[object] = []

    def boom(**kw: object) -> str:
        called.append(kw)
        raise AssertionError("snapshot_download must not run")

    from_ollama_mod._snapshot_download = boom  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stderr(err):
                code = main(
                    ["from-ollama", "qwen2.5:3b", "--yes", "--dir", str(Path(tmp) / "hf")]
                )
    finally:
        from_ollama_mod._hub_missing = original_missing  # type: ignore[assignment]
        from_ollama_mod._snapshot_download = original_snap  # type: ignore[assignment]
    assert code == 1
    assert called == []
    assert 'pip install "deepfold[hub]"' in err.getvalue() or "deepfold[hub]" in err.getvalue()


def test_from_ollama_existing_tree_skips_snapshot() -> None:
    called: list[object] = []

    def boom(**kw: object) -> str:
        called.append(kw)
        raise AssertionError("already on disk")

    original = from_ollama_mod._snapshot_download
    from_ollama_mod._snapshot_download = boom  # type: ignore[assignment]
    err, out = io.StringIO(), io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "qwen"
            dest.mkdir()
            (dest / "config.json").write_text("{}", encoding="utf-8")
            with redirect_stderr(err), redirect_stdout(out):
                code = main(["from-ollama", "qwen2.5:3b", "--dir", str(dest)])
    finally:
        from_ollama_mod._snapshot_download = original  # type: ignore[assignment]
    assert code == 0, err.getvalue()
    assert called == []
    assert "Already on disk" in err.getvalue()
    assert f"deepfold run --model {dest}" in out.getvalue()


def test_ollama_map_resolve_strips_library_prefix() -> None:
    mapped = resolve("library/qwen2.5:3b")
    assert mapped.hf_id == "Qwen/Qwen2.5-3B-Instruct"
    try:
        resolve(r"C:\Users\x\.ollama\models\blobs\sha256-dead")
    except ResolveError as exc:
        assert exc.kind == "gguf"
    else:  # pragma: no cover
        raise AssertionError("a blob path must be GGUF, not an unknown tag")


def test_parser_from_ollama_flags() -> None:
    args = build_parser().parse_args(
        ["from-ollama", "qwen2.5:3b", "--hf", "Qwen/Qwen2.5-3B-Instruct", "--yes", "--run"]
    )
    assert args.command == "from-ollama"
    assert args.tag == "qwen2.5:3b"
    assert args.hf == "Qwen/Qwen2.5-3B-Instruct"
    assert args.yes is True and args.run is True


def test_parser_k5_commands() -> None:
    pull = build_parser().parse_args(
        ["pull", "Qwen/Qwen2.5-3B-Instruct", "--yes", "--dir", "D:/hf"]
    )
    assert pull.command == "pull" and pull.yes is True and pull.dir == "D:/hf"
    chat = build_parser().parse_args(["chat", "--model", "D:/m", "--max-new-tokens", "8"])
    assert chat.command == "chat" and chat.model == "D:/m" and chat.max_new_tokens == 8
    assert not hasattr(chat, "prompt")
    setup = build_parser().parse_args(["setup", "--dry-run"])
    assert setup.dry_run is True
    live = build_parser().parse_args(["test", "--live"])
    assert live.live is True


def test_hf_allowlist_includes_32b() -> None:
    ids = hf_id_list()
    assert ids[0] == "Qwen/Qwen2.5-3B-Instruct"
    assert "Qwen/Qwen2.5-32B-Instruct" in ids
    assert "internlm/internlm2_5-20b-chat" in ids


def test_slash_commands_are_not_prompts() -> None:
    assert chat_mod.classify_slash("hello") is None
    assert chat_mod.classify_slash("/quit") == "quit"
    assert chat_mod.classify_slash("/exit") == "quit"
    assert chat_mod.classify_slash("/clear") == "clear"
    assert chat_mod.classify_slash("/help") == "help"
    assert chat_mod.classify_slash("/stats") == "stats"
    assert chat_mod.classify_slash("/rm") == "unknown"


def test_chat_without_tty_points_at_run_prompt() -> None:
    stdin = sys.stdin
    sys.stdin = io.StringIO("hello\n")
    err = io.StringIO()
    try:
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(["chat", "--model", "D:/m"])
    finally:
        sys.stdin = stdin
    assert code == 1
    assert "run --prompt" in err.getvalue()


def test_setup_dry_run_does_not_pip() -> None:
    called: list[object] = []
    original = setup_mod._run
    setup_mod._run = lambda *a, **k: called.append((a, k)) or 0  # type: ignore[assignment]
    err, out = io.StringIO(), io.StringIO()
    try:
        with redirect_stderr(err), redirect_stdout(out):
            code = main(["setup", "--dry-run"])
    finally:
        setup_mod._run = original  # type: ignore[assignment]
    assert code == 0
    assert called == []
    text = out.getvalue() + err.getvalue()
    assert "torch" in text and "download.pytorch.org/whl/cu124" in text


def test_setup_refuses_torch_gpu_without_pip() -> None:
    called: list[object] = []
    original_prot = setup_mod.prefix_is_protected
    original_run = setup_mod._run
    setup_mod.prefix_is_protected = lambda prefix=None: True  # type: ignore[assignment]
    setup_mod._run = lambda *a, **k: called.append(1) or 0  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(["setup"])
    finally:
        setup_mod.prefix_is_protected = original_prot  # type: ignore[assignment]
        setup_mod._run = original_run  # type: ignore[assignment]
    assert code == 1
    assert called == []
    assert "torch-gpu" in err.getvalue()
    assert "scripts/setup.ps1" in err.getvalue()


def test_pull_unknown_id_does_not_open_files() -> None:
    opened, real_open = _watch_open()
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = main(["pull", "meta-llama/Llama-3.1-8B-Instruct"])
    finally:
        builtins.open = real_open  # type: ignore[assignment]
    assert code == 1
    assert "unknown HuggingFace id" in err.getvalue()
    assert "Qwen/Qwen2.5-3B-Instruct" in err.getvalue()
    assert "not a general HuggingFace runtime" in err.getvalue()
    assert not [p for p in opened if "Llama" in p]


def test_pull_gguf_is_refused_without_being_opened() -> None:
    opened, real_open = _watch_open()
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = main(["pull", r"C:\Users\x\.ollama\models\blobs\sha256-dead"])
    finally:
        builtins.open = real_open  # type: ignore[assignment]
    assert code == 1
    assert "GGUF" in err.getvalue()
    assert not [p for p in opened if "sha256-dead" in p.lower() or ".gguf" in p.lower()]


def test_pull_yes_stub_snapshot_download() -> None:
    called: list[dict[str, str]] = []

    def stub(*, repo_id: str, local_dir: str) -> str:
        called.append({"repo_id": repo_id, "local_dir": local_dir})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        return local_dir

    original = hub_mod.snapshot_download
    hub_mod.snapshot_download = stub  # type: ignore[assignment]
    err, out = io.StringIO(), io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "hf"
            with redirect_stderr(err), redirect_stdout(out):
                code = main(
                    ["pull", "Qwen/Qwen2.5-3B-Instruct", "--yes", "--dir", str(dest)]
                )
    finally:
        hub_mod.snapshot_download = original  # type: ignore[assignment]
    assert code == 0, err.getvalue()
    assert called == [{"repo_id": "Qwen/Qwen2.5-3B-Instruct", "local_dir": str(dest)}]
    assert f"deepfold chat --model {dest}" in out.getvalue()
    assert f"deepfold run --model {dest}" in out.getvalue()


def test_pull_32b_is_on_the_table() -> None:
    called: list[str] = []

    def stub(*, repo_id: str, local_dir: str) -> str:
        called.append(repo_id)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        return local_dir

    original = hub_mod.snapshot_download
    hub_mod.snapshot_download = stub  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "hf"
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(
                    ["pull", "Qwen/Qwen2.5-32B-Instruct", "--yes", "--dir", str(dest)]
                )
    finally:
        hub_mod.snapshot_download = original  # type: ignore[assignment]
    assert code == 0
    assert called == ["Qwen/Qwen2.5-32B-Instruct"]


def test_selftest_live_skips_without_3b() -> None:
    import gpu.cli.doctor as doctor_mod

    original_root = selftest_mod.models_root
    original_probe = doctor_mod.probe
    doctor_mod.probe = lambda **k: _ship()  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            selftest_mod.models_root = lambda: Path(tmp)  # type: ignore[assignment]
            with redirect_stderr(err):
                code = selftest_mod.selftest(SimpleNamespace(live=True, chr_bin=None))
    finally:
        selftest_mod.models_root = original_root  # type: ignore[assignment]
        doctor_mod.probe = original_probe  # type: ignore[assignment]
    assert code == 0
    assert "SKIP:" in err.getvalue()
    assert "deepfold pull Qwen/Qwen2.5-3B-Instruct" in err.getvalue()


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #


def test_slug_for_a_directory_and_for_a_hub_id() -> None:
    assert paths.slug(r"C:\dev\models\Qwen2.5-3B-Instruct") == "Qwen2.5-3B-Instruct"
    assert paths.slug("Qwen/Qwen2.5-3B-Instruct") == "Qwen_Qwen2.5-3B-Instruct"


def test_models_root_honours_env() -> None:
    previous_models = os.environ.get("DEEPFOLD_MODELS")
    previous_runs = os.environ.get("DEEPFOLD_RUNS")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DEEPFOLD_MODELS"] = tmp
        os.environ["DEEPFOLD_RUNS"] = str(Path(tmp) / "runs")
        try:
            assert paths.models_root() == Path(tmp)
            assert paths.runs_root() == Path(tmp) / "runs"
        finally:
            if previous_models is None:
                os.environ.pop("DEEPFOLD_MODELS", None)
            else:
                os.environ["DEEPFOLD_MODELS"] = previous_models
            if previous_runs is None:
                os.environ.pop("DEEPFOLD_RUNS", None)
            else:
                os.environ["DEEPFOLD_RUNS"] = previous_runs


def test_chr_bin_search_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / ("chr.exe" if os.name == "nt" else "chr")
        fake.write_bytes(b"")
        os.environ["DEEPFOLD_CHR_BIN"] = str(fake)
        try:
            assert paths.find_chr_bin() == fake, "DEEPFOLD_CHR_BIN beats PATH"
            explicit = Path(tmp) / "other"
            explicit.write_bytes(b"")
            assert paths.find_chr_bin(str(explicit)) == explicit, "the flag wins"
        finally:
            os.environ.pop("DEEPFOLD_CHR_BIN", None)


def test_env_chr_is_ignored_when_the_file_is_gone() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = _fake_model(Path(tmp), "qwen2")
        os.environ["DEEPFOLD_CHR"] = str(Path(tmp) / "absent.nf4.chr")
        os.environ["DEEPFOLD_HOME"] = str(Path(tmp) / "home")
        try:
            assert paths.find_chr_file(model) is None
        finally:
            os.environ.pop("DEEPFOLD_CHR", None)
            os.environ.pop("DEEPFOLD_HOME", None)


def test_ambiguous_siblings_are_never_resolved_by_sort_order() -> None:
    """A flat models folder must not hand the 20B .chr to a 3B skeleton."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2")
        for name in ("internlm2_5-20b.nf4.chr", "qwen25-14b.nf4.chr", "qwen25-3b.nf4.chr"):
            (root / name).write_bytes(b"CHR0")
        os.environ["DEEPFOLD_HOME"] = str(root / "home")
        os.environ.pop("DEEPFOLD_CHR", None)
        try:
            assert len(paths.chr_candidates(model)) == 3
            assert paths.find_chr_file(model) is None, "three candidates is not an answer"
            picked = paths.find_chr_file(
                model, accept=lambda p: p.name == "qwen25-3b.nf4.chr"
            )
            assert picked is not None and picked.name == "qwen25-3b.nf4.chr"
            # An explicit --chr is the user's call and is not re-checked.
            given = root / "internlm2_5-20b.nf4.chr"
            assert paths.find_chr_file(model, str(given), accept=lambda p: False) == given
        finally:
            os.environ.pop("DEEPFOLD_HOME", None)


def test_env_chr_is_dropped_when_it_does_not_match_the_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2")
        wrong = root / "some-other.nf4.chr"
        wrong.write_bytes(b"CHR0")
        right = model / "right.nf4.chr"
        right.write_bytes(b"CHR0")
        os.environ["DEEPFOLD_CHR"] = str(wrong)
        os.environ["DEEPFOLD_HOME"] = str(root / "home")
        try:
            picked = paths.find_chr_file(model, accept=lambda p: p == right)
            assert picked == right, "DEEPFOLD_CHR counts only if it matches this model"
        finally:
            os.environ.pop("DEEPFOLD_CHR", None)
            os.environ.pop("DEEPFOLD_HOME", None)


def test_sibling_chr_is_found() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = _fake_model(root, "qwen2")
        sibling = root / "qwen25-3b.nf4.chr"
        sibling.write_bytes(b"CHR0")
        os.environ["DEEPFOLD_HOME"] = str(root / "home")
        os.environ.pop("DEEPFOLD_CHR", None)
        try:
            assert paths.find_chr_file(model) == sibling
        finally:
            os.environ.pop("DEEPFOLD_HOME", None)


# --------------------------------------------------------------------------- #
# which .chr belongs to this model: the CHR0 header, never the glob order
# --------------------------------------------------------------------------- #

_QWEN3B = dict(hidden=2048, layers=36, vocab=151936)
_QWEN14B = dict(hidden=5120, layers=48, vocab=152064)
_INTERNLM20B = dict(hidden=6144, layers=48, vocab=92544)


def _flat_models(root: Path) -> tuple[Path, Path]:
    """A 3B skeleton next to three sibling ``.chr`` files, 3B sorting last."""
    model = _config(
        root / "Qwen2.5-3B-Instruct",
        model_type="qwen2",
        hidden_size=2048,
        num_hidden_layers=36,
        vocab_size=151936,
    )
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    _write_chr(root / "internlm2_5-20b.nf4.chr", **_INTERNLM20B)
    _write_chr(root / "qwen25-14b.nf4.chr", **_QWEN14B)
    mine = _write_chr(root / "qwen25-3b.nf4.chr", **_QWEN3B)
    return model, mine


def test_header_matcher_picks_the_chr_packed_from_this_model() -> None:
    """sorted() would hand the 20B file to the 3B skeleton; the header does not."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, mine = _flat_models(root)
        assert paths.chr_candidates(model)[0].name.startswith("internlm")

        accept = run_mod._header_matcher(model)
        assert accept is not None, "a complete config must produce a matcher"
        assert accept(mine)
        assert not accept(root / "qwen25-14b.nf4.chr")
        assert not accept(root / "internlm2_5-20b.nf4.chr")


def test_run_resolves_the_header_matched_sibling_without_compressing() -> None:
    spy = _Spy(0)
    original = run_mod.compress_to
    run_mod.compress_to = spy  # type: ignore[assignment]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, mine = _flat_models(root)
        os.environ["DEEPFOLD_HOME"] = str(root / "home")
        os.environ.pop("DEEPFOLD_CHR", None)
        try:
            resolved, code = run_mod._resolve_weights(_RunArgs(), str(model))
        finally:
            run_mod.compress_to = original  # type: ignore[assignment]
            os.environ.pop("DEEPFOLD_HOME", None)
    assert code == 0 and resolved == mine, f"picked {resolved}"
    assert spy.calls == [], "a matching .chr was on disk; nothing should repack"


def test_header_matcher_says_no_to_a_foreign_or_absent_file() -> None:
    """A truncated or non-CHR0 sibling is a no, not a traceback out of run."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, _ = _flat_models(root)
        accept = run_mod._header_matcher(model)
        assert accept is not None

        junk = root / "junk.nf4.chr"
        junk.write_bytes(b"CHR0 but not really")
        assert not accept(junk)
        assert not accept(root / "never-written.nf4.chr")
        truncated = root / "cut.nf4.chr"
        truncated.write_bytes(_write_chr(root / "src.nf4.chr", **_QWEN3B).read_bytes()[:200])
        assert not accept(truncated)


def test_header_matcher_refuses_to_guess_from_a_thin_config() -> None:
    """No vocab_size in config.json -> no predicate, so run compresses instead."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        thin = _fake_model(root, "qwen2")  # hidden_size + num_hidden_layers only
        assert run_mod._header_matcher(thin) is None
        assert run_mod._header_matcher(root / "not-a-model") is None

        _write_chr(root / "a.nf4.chr", **_QWEN3B)
        _write_chr(root / "b.nf4.chr", **_QWEN14B)
        os.environ["DEEPFOLD_HOME"] = str(root / "home")
        os.environ.pop("DEEPFOLD_CHR", None)
        try:
            assert paths.find_chr_file(thin, accept=None) is None
        finally:
            os.environ.pop("DEEPFOLD_HOME", None)


def test_header_matched_env_chr_beats_an_ambiguous_directory() -> None:
    """DEEPFOLD_CHR counts only when its header matches this model."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, mine = _flat_models(root)
        os.environ["DEEPFOLD_CHR"] = str(root / "internlm2_5-20b.nf4.chr")
        os.environ["DEEPFOLD_HOME"] = str(root / "home")
        try:
            picked = paths.find_chr_file(model, accept=run_mod._header_matcher(model))
        finally:
            os.environ.pop("DEEPFOLD_CHR", None)
            os.environ.pop("DEEPFOLD_HOME", None)
    assert picked == mine, "the 20B file in DEEPFOLD_CHR must lose to the header"


# --------------------------------------------------------------------------- #
# compress: the same refusals as run, before chr is ever started
# --------------------------------------------------------------------------- #


def test_compress_refuses_a_gguf_in_without_opening_it() -> None:
    real_open = builtins.open
    opened: list[str] = []

    def watched(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    spy = _Spy(0)
    original = run_mod.compress_to
    run_mod.compress_to = spy  # type: ignore[assignment]
    builtins.open = watched  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = run_mod.compress(
                _CompressArgs(inp=r"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf")
            )
    finally:
        builtins.open = real_open  # type: ignore[assignment]
        run_mod.compress_to = original  # type: ignore[assignment]

    assert code == 1
    assert "cannot load GGUF" in err.getvalue()
    assert spy.calls == [], "chr must not be started on a GGUF path"
    assert not [p for p in opened if ".gguf" in p.lower()]


def test_compress_refuses_an_unwired_arch_before_starting_chr() -> None:
    spy = _Spy(0)
    original = run_mod.compress_to
    run_mod.compress_to = spy  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_model(Path(tmp), "gemma", hidden_act="gelu_pytorch_tanh")
            with redirect_stderr(err):
                code = run_mod.compress(_CompressArgs(inp=str(model)))
    finally:
        run_mod.compress_to = original  # type: ignore[assignment]
    assert code == 1
    assert "gemma_gelu" in err.getvalue()
    assert "model_type=gemma" in err.getvalue()
    assert spy.calls == [], "the config gate must lose before chr is started"


def test_compress_will_not_silently_overwrite_an_existing_chr() -> None:
    spy = _Spy(0)
    original = run_mod.compress_to
    run_mod.compress_to = spy  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _fake_model(root, "qwen2")
            out = root / "already.nf4.chr"
            out.write_bytes(b"CHR0")
            os.environ["DEEPFOLD_CHR_BIN"] = str(_REPO / "chr.exe")
            try:
                with redirect_stderr(err):
                    code = run_mod.compress(
                        _CompressArgs(inp=str(model), out=str(out))
                    )
            finally:
                os.environ.pop("DEEPFOLD_CHR_BIN", None)
    finally:
        run_mod.compress_to = original  # type: ignore[assignment]
    assert code == 1 and "--force" in err.getvalue()
    assert spy.calls == [], "an existing .chr must not be repacked by accident"


def test_compress_auto_refuses_vq_when_nf4_would_not_fit() -> None:
    """H2-1: 32B compress --codec auto packs NF4 overflow, never VQ."""
    qwen32 = dict(
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        vocab_size=152064,
        tie_word_embeddings=False,
    )
    spy = _Spy(0)
    original = run_mod.compress_to
    run_mod.compress_to = spy  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_model(Path(tmp), "qwen2", name="Qwen2.5-32B", **qwen32)
            os.environ["DEEPFOLD_CHR_BIN"] = str(_REPO / "chr.exe")
            os.environ["DEEPFOLD_HOME"] = str(Path(tmp) / "home")
            try:
                with redirect_stderr(err):
                    code = run_mod.compress(
                        _CompressArgs(inp=str(model), vram_mib=12288)
                    )
            finally:
                os.environ.pop("DEEPFOLD_CHR_BIN", None)
                os.environ.pop("DEEPFOLD_HOME", None)
    finally:
        run_mod.compress_to = original  # type: ignore[assignment]
    text = err.getvalue()
    assert code == 0
    assert len(spy.calls) == 1 and spy.calls[0][1].get("codec") == "nf4"
    assert "overflow" in text and "H2" in text


# --------------------------------------------------------------------------- #
# doctor exit codes, continued
# --------------------------------------------------------------------------- #


def test_override_is_green_but_never_claims_ship() -> None:
    m = _ship(capability=(8, 9))
    v = verdict(m, override=True)
    assert exit_code(m, v, checks(m, v)) == 0
    text = render(m, v, checks(m, v))
    assert "experimental" in text
    assert messages.GENERATE_SHIP not in text, "the override must not print 'ship'"


def test_override_does_not_paper_over_a_broken_install() -> None:
    """sm_89 + no chr is still 2: the arch is fixable, the missing binary is not ok."""
    m = _ship(capability=(8, 9), chr_bin=None, chr_runs=False)
    assert exit_code(m, verdict(m, override=True), checks(m, verdict(m, override=True))) == 2


def test_a_refused_arch_with_no_chr_is_one_not_three() -> None:
    """3 means 'compress still works here'. Without chr, nothing works: 1."""
    m = _ship(capability=(7, 5), chr_bin=None, chr_runs=False)
    v = verdict(m, override=False)
    assert not compress_ok(m)
    assert exit_code(m, v, checks(m, v)) == 1


def test_the_machine_of_record_probes_as_ship() -> None:
    """The only check here that touches real hardware; elsewhere it skips (D12).

    Every other test hand-builds a :class:`Machine`, so nothing would notice if
    ``probe()`` stopped reading the capability at all. On an sm_86 box this
    pins the live probe to the ship verdict; on anything else it is a skip and
    never a pass.
    """
    requires_sm86()
    m = probe()
    v = verdict(m, override=False)
    assert m.capability == (8, 6) and m.cuda_available
    assert v.generate == "yes" and v.arch == "ship"
    assert v.line == messages.GENERATE_SHIP
    code = exit_code(m, v, checks(m, v))
    assert code in (0, 2), f"sm_86 must never be refused by class, got {code}"


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/cli acceptance, {len(TESTS)} tests, no GPU required\n")
    for fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
        except Skip as exc:
            # A check that could not be made is never folded into the pass count.
            SKIPPED.append((fn.__name__, str(exc)))
            print(f"  SKIP  {fn.__name__}  -- {exc}")
        except AssertionError as exc:
            check(fn.__name__, False, str(exc).splitlines()[0] if str(exc) else "assert")
        except Exception as exc:  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
        else:
            check(fn.__name__, True)

    failed = [name for name, ok, _ in CHECKS if not ok]
    tail = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed{tail}")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())
