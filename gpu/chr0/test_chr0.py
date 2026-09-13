"""Verification for the CHR0 GPU loader.

Runs under pytest, or standalone:

    python gpu/chr0/test_chr0.py

The happy-path fixture is produced by the real Go compressor (``chr.exe
compress --codec nf4``) so the loader is checked against bytes it did not
write. Malformed cases are hand-built, since the writer cannot emit them.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # standalone: put gpu/ on sys.path, import chr0
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chr0 import (
        AlignmentError,
        CodecError,
        GroupSizeError,
        HeaderError,
        OverlapError,
        SizeMismatchError,
        TensorNotFoundError,
        TruncatedError,
        iter_linears,
        load_header,
        materialize_nf4,
    )
    from chr0._fixtures import build_chr, nf4_blob_sizes, write_safetensors
else:
    from . import (
        AlignmentError,
        CodecError,
        GroupSizeError,
        HeaderError,
        OverlapError,
        SizeMismatchError,
        TensorNotFoundError,
        TruncatedError,
        iter_linears,
        load_header,
        materialize_nf4,
    )
    from ._fixtures import build_chr, nf4_blob_sizes, write_safetensors

REPO = Path(__file__).resolve().parents[2]
CHR_EXE = REPO / "chr.exe"
QWEN = Path(r"C:\dev\models\qwen25-3b.nf4.chr")

GATE_PROJ = "model.layers.0.mlp.gate_proj"
Q_PROJ = "model.layers.0.self_attn.q_proj"
DOWN_PROJ = "model.layers.0.mlp.down_proj"
NORM = "model.norm"


try:  # let pytest report these as skips rather than errors
    import pytest

    Skip = pytest.skip.Exception
except ImportError:

    class Skip(Exception):
        """Raised instead of a test that needs absent hardware or files."""


# --------------------------------------------------------------------------- #
# Session fixture: one tiny .chr built by the Go compressor
# --------------------------------------------------------------------------- #

_TMP: Path | None = None
_FIXTURE: Path | None = None


def _tmp() -> Path:
    global _TMP
    if _TMP is None:
        _TMP = Path(tempfile.mkdtemp(prefix="chr0-test-"))
        atexit.register(shutil.rmtree, _TMP, True)
    return _TMP


def _ramp(*shape: int) -> np.ndarray:
    """Deterministic, finite, non-degenerate weights in roughly [-1, 1]."""
    n = int(np.prod(shape))
    return (((np.arange(n, dtype=np.float32) % 37.0) - 18.0) / 17.0).reshape(shape)


def fixture_chr() -> Path:
    """64x128 and 2x65 NF4 matrices plus a BF16 norm, compressed by chr.exe."""
    global _FIXTURE
    if _FIXTURE is not None:
        return _FIXTURE
    if not CHR_EXE.is_file():
        raise Skip(f"{CHR_EXE} missing; build with: go build -o chr.exe ./cmd/chr")

    src = _tmp() / "toy"
    src.mkdir(exist_ok=True)
    write_safetensors(
        src / "model.safetensors",
        {
            f"{Q_PROJ}.weight": _ramp(64, 128),
            f"{DOWN_PROJ}.weight": _ramp(2, 65),
            f"{NORM}.weight": _ramp(128),
        },
    )
    out = _tmp() / "toy.nf4.chr"
    proc = subprocess.run(
        [
            str(CHR_EXE), "compress",
            "--in", str(src),
            "--out", str(out),
            "--codec", "nf4",
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
        raise AssertionError(
            f"chr compress failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    _FIXTURE = out
    return out


def _file_slice(path: Path, start: int, end: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(end - start)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _smi_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None


def _tensor_bytes(t: torch.Tensor) -> bytes:
    flat = t.reshape(-1).contiguous().cpu()
    return flat.view(torch.uint8).numpy().tobytes()


def _nf4_entry(m: int, k: int, data_start: int, *, kind="q", layer=0, group_size=64) -> dict:
    data_n, scale_n = nf4_blob_sizes(m, k)
    entry: dict = {
        "kind": kind,
        "codec": "nf4",
        "shape": [m, k],
        "group_size": group_size,
        "data": [data_start, data_start + data_n],
        "scale": [data_start + data_n, data_start + data_n + scale_n],
    }
    if layer is not None:
        entry["layer"] = layer
    return entry


def _expect(exc_type, fn, *args, **kwargs):
    wanted = exc_type if isinstance(exc_type, tuple) else (exc_type,)
    label = "/".join(e.__name__ for e in wanted)
    try:
        fn(*args, **kwargs)
    except wanted as exc:
        return exc
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"expected {label}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {label}, nothing raised")


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_header_matches_fixture():
    hdr = load_header(str(fixture_chr()))
    assert hdr.magic == "CHR0" and hdr.version == 1
    assert (hdr.tile_row, hdr.tile_col_group) == (64, 8)
    assert set(hdr.tensors) == {Q_PROJ, DOWN_PROJ, NORM}

    q = hdr.tensor(Q_PROJ)
    assert (q.codec, q.kind, q.layer, q.group_size) == ("nf4", "q", 0, 64)
    assert (q.M, q.K, q.K_pad, q.n_groups) == (64, 128, 128, 2)
    assert q.blobs["data"].nbytes == 4096  # 64 * 128 / 2
    assert q.blobs["scale"].nbytes == 256  # 64 * 2 * 2

    assert hdr.tensor(NORM).codec == "bf16"
    assert hdr.tensor(NORM).layer is None
    _expect(TensorNotFoundError, hdr.tensor, "model.layers.0.mlp.up_proj")


def test_iter_linears_is_nf4_only():
    hdr = load_header(str(fixture_chr()))
    assert sorted(iter_linears(hdr)) == sorted([DOWN_PROJ, Q_PROJ])


def test_packed_and_scale_equal_file_bytes():
    """The whole point: bytes on disk are the bytes in the tensor."""
    path = fixture_chr()
    hdr = load_header(str(path))
    for name in iter_linears(hdr):
        info = hdr.tensor(name)
        m = materialize_nf4(str(path), name, device="cpu", header=hdr)

        assert (m.name, m.M, m.K, m.K_pad) == (name, info.M, info.K, info.K_pad)
        assert m.packed.dtype == torch.uint8
        assert m.scale.dtype == torch.float16
        assert tuple(m.packed.shape) == (info.M, info.K_pad // 2)
        assert tuple(m.scale.shape) == (info.M, info.K_pad // 64)
        assert m.packed.is_contiguous() and m.scale.is_contiguous()

        data, scale = info.blobs["data"], info.blobs["scale"]
        assert _tensor_bytes(m.packed) == _file_slice(path, data.start, data.end)
        assert _tensor_bytes(m.scale) == _file_slice(path, scale.start, scale.end)
        assert m.nbytes == data.nbytes + scale.nbytes


def test_k65_pads_to_128():
    path = fixture_chr()
    hdr = load_header(str(path))
    info = hdr.tensor(DOWN_PROJ)
    assert (info.M, info.K, info.K_pad) == (2, 65, 128)
    assert info.blobs["data"].nbytes == info.M * 64 == 128
    assert info.blobs["scale"].nbytes == info.M * 2 * 2 == 8

    m = materialize_nf4(str(path), DOWN_PROJ, device="cpu", header=hdr)
    assert tuple(m.packed.shape) == (2, 64)
    assert tuple(m.scale.shape) == (2, 2)
    # Tail columns 65..127 are the codec's zero padding, never exposed as K.
    assert m.K == 65 and m.packed.numel() == 2 * 64


def test_default_header_path_needs_no_preparsed_header():
    path = fixture_chr()
    m = materialize_nf4(str(path), Q_PROJ, device="cpu")
    assert (m.M, m.K, m.K_pad) == (64, 128, 128)


def test_file_is_never_written():
    path = fixture_chr()
    before = (_sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
    hdr = load_header(str(path))
    for name in iter_linears(hdr):
        materialize_nf4(str(path), name, device="cpu", header=hdr)
    after = (_sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
    assert before == after


# --------------------------------------------------------------------------- #
# Rejections (gpu-abi.md §5)
# --------------------------------------------------------------------------- #


def test_reject_end_past_eof_before_any_device_touch():
    """A short file must raise without allocating or copying on the device."""
    path = _tmp() / "past-eof.chr"
    full = build_chr(path, {Q_PROJ: _nf4_entry(64, 128, 4096)})
    with open(path, "r+b") as f:  # chop the last 64 bytes off the payload
        f.truncate(full - 64)

    exc = _expect(TruncatedError, load_header, str(path))
    assert "truncated" in str(exc)

    original_empty = torch.empty

    def no_device_memory(*args, **kwargs):
        raise AssertionError("device allocation happened before header validation")

    torch.empty = no_device_memory
    try:
        _expect(TruncatedError, materialize_nf4, str(path), Q_PROJ, "cuda")
    finally:
        torch.empty = original_empty


def test_reject_truncated_payload_with_valid_header():
    """Header is self-consistent but the read comes up short: still no copy."""
    path = _tmp() / "short-read.chr"
    build_chr(path, {Q_PROJ: _nf4_entry(64, 128, 4096)})
    hdr = load_header(str(path))
    with open(path, "r+b") as f:
        f.truncate(hdr.tensor(Q_PROJ).blobs["data"].start + 16)

    original_empty = torch.empty

    def no_device_memory(*args, **kwargs):
        raise AssertionError("device allocation happened before the payload was read")

    torch.empty = no_device_memory
    try:
        _expect(TruncatedError, materialize_nf4, str(path), Q_PROJ, "cuda", header=hdr)
    finally:
        torch.empty = original_empty


def test_reject_size_mismatch():
    path = _tmp() / "size-mismatch.chr"
    entry = _nf4_entry(64, 128, 4096)
    entry["shape"] = [64, 64]  # blobs still sized for K=128
    build_chr(path, {Q_PROJ: entry})
    exc = _expect(SizeMismatchError, load_header, str(path))
    assert "data" in str(exc)


def test_reject_overlap():
    path = _tmp() / "overlap.chr"
    a = _nf4_entry(64, 128, 4096)
    b = _nf4_entry(64, 128, 4096 + 4096)  # starts inside a's scale blob
    build_chr(path, {Q_PROJ: a, "model.layers.0.self_attn.k_proj": dict(b, kind="k")})
    _expect(OverlapError, load_header, str(path))


def test_reject_unaligned_start():
    path = _tmp() / "unaligned.chr"
    build_chr(path, {Q_PROJ: _nf4_entry(64, 128, 4096 + 32)})
    _expect(AlignmentError, load_header, str(path))


def test_reject_group_size_not_64():
    path = _tmp() / "group32.chr"
    build_chr(path, {Q_PROJ: _nf4_entry(64, 128, 4096, group_size=32)})
    _expect(GroupSizeError, load_header, str(path))


def test_reject_materialize_non_nf4():
    path = fixture_chr()
    _expect(CodecError, materialize_nf4, str(path), NORM, "cpu")


def test_reject_vq_materialize_but_parse_header():
    """VQ slots parse fine; wave 2.0 refuses to materialize them."""
    path = _tmp() / "vq.chr"
    index_n = 64 * (128 // 8) * 2
    build_chr(
        path,
        {
            Q_PROJ: {
                "layer": 0,
                "kind": "q",
                "codec": "vq",
                "shape": [64, 128],
                "group_size": 8,
                "n_codebooks": 2,
                "codebook_bits": 8,
                "codebook": [4096, 4096 + 8192],
                "index": [4096 + 8192, 4096 + 8192 + index_n],
            }
        },
    )
    hdr = load_header(str(path))
    assert hdr.tensor(Q_PROJ).codec == "vq"
    assert list(iter_linears(hdr)) == []
    _expect(CodecError, materialize_nf4, str(path), Q_PROJ, "cpu", header=hdr)


def test_reject_bad_root_fields():
    base = {Q_PROJ: _nf4_entry(64, 128, 4096)}
    for label, override in [
        ("magic", {"magic": "chr0"}),
        ("version", {"version": 2}),
        ("tile", {"tile": {"row": 32, "col_group": 8}}),
        ("tensors", {"tensors": {}}),
    ]:
        path = _tmp() / f"root-{label}.chr"
        build_chr(path, base, **override)
        _expect(HeaderError, load_header, str(path))


def test_reject_broken_header_bytes():
    cases = {
        "empty": b"",
        "short": b"\x01\x02\x03",
        "nbytes_past_eof": struct.pack("<Q", 4096) + b"{}",
        "not_json": struct.pack("<Q", 4) + b"nope",
        "bom": struct.pack("<Q", 5) + b"\xef\xbb\xbf{}",
        "array_root": struct.pack("<Q", 2) + b"[]",
        "trailing": struct.pack("<Q", 5) + b"{} {}",
    }
    for label, blob in cases.items():
        path = _tmp() / f"broken-{label}.chr"
        path.write_bytes(blob)
        _expect((TruncatedError, HeaderError), load_header, str(path))


def test_reject_float_offsets():
    root = {
        "magic": "CHR0", "version": 1, "arch": "toy", "hidden_size": 128,
        "intermediate_size": 128, "num_layers": 1, "vocab_size": 0,
        "tile": {"row": 64, "col_group": 8},
        "tensors": {Q_PROJ: _nf4_entry(64, 128, 4096)},
    }
    js = json.dumps(root, separators=(",", ":")).replace('"hidden_size":128', '"hidden_size":128.0')
    raw = js.encode()
    path = _tmp() / "float-field.chr"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 65536)
    _expect(HeaderError, load_header, str(path))


# --------------------------------------------------------------------------- #
# Real model
# --------------------------------------------------------------------------- #


def test_qwen3b_gate_proj_on_cuda():
    if not QWEN.is_file():
        raise Skip(f"{QWEN} not present")
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")

    hdr = load_header(str(QWEN))
    info = hdr.tensor(GATE_PROJ)
    assert (info.M, info.K, info.K_pad) == (11008, 2048, 2048)
    expected_bytes = info.blobs["data"].nbytes + info.blobs["scale"].nbytes
    assert expected_bytes == 11008 * 1024 + 11008 * 32 * 2

    torch.zeros(1, device="cuda")  # pay for the CUDA context before measuring
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before, smi_before = torch.cuda.memory_allocated(), _smi_used_mib()
    m = materialize_nf4(str(QWEN), GATE_PROJ, device="cuda", header=hdr)
    torch.cuda.synchronize()
    delta, smi_after = torch.cuda.memory_allocated() - before, _smi_used_mib()

    assert m.packed.device.type == "cuda" and m.scale.device.type == "cuda"
    assert tuple(m.packed.shape) == (11008, 1024)
    assert tuple(m.scale.shape) == (11008, 32)
    assert m.nbytes == expected_bytes
    # One allocation for the whole matrix, plus the caching allocator's 2 MiB
    # block rounding. A BF16 draft of W would be 43 MiB, F32 86 MiB.
    assert delta <= expected_bytes + 2 * 2**20, f"{delta} bytes for {expected_bytes}"
    assert delta < 11008 * 2048 * 2 // 2

    # Same bytes as the file, and still no dequantisation anywhere.
    data = info.blobs["data"]
    assert _tensor_bytes(m.packed[:4]) == _file_slice(QWEN, data.start, data.start + 4 * 1024)
    scale = info.blobs["scale"]
    assert _tensor_bytes(m.scale[:4]) == _file_slice(QWEN, scale.start, scale.start + 4 * 64)

    smi = "n/a" if smi_before is None or smi_after is None else f"{smi_after - smi_before} MiB"
    print(
        f"    qwen3b {GATE_PROJ}: M={m.M} K={m.K} K_pad={m.K_pad} "
        f"blobs={expected_bytes / 2**20:.2f} MiB torch_delta={delta / 2**20:.2f} MiB "
        f"smi_delta={smi} (bf16 W would be {11008 * 2048 * 2 / 2**20:.0f} MiB)"
    )
    del m
    torch.cuda.empty_cache()


def test_qwen3b_header_survey():
    if not QWEN.is_file():
        raise Skip(f"{QWEN} not present")
    hdr = load_header(str(QWEN))
    assert hdr.arch == "qwen2" and hdr.num_layers == 36
    names = list(iter_linears(hdr))
    assert GATE_PROJ in names
    assert all(hdr.tensor(n).group_size == 64 for n in names)
    assert all(hdr.tensor(n).codec == "nf4" for n in names)
    # Norms stay BF16 and never show up as linears.
    assert "model.layers.0.input_layernorm" not in names
    print(f"    qwen3b: {len(hdr.tensors)} tensors, {len(names)} nf4")


# --------------------------------------------------------------------------- #


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in tests:
        try:
            fn()
        except Skip as exc:
            print(f"SKIP {fn.__name__}: {exc}")
            skipped += 1
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(tests) - failed - skipped} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
