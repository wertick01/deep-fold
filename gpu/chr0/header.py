"""CHR0 header parsing and validation (read-only).

Implements the reader half of ``docs/spec/chr0.md`` §1-§2 plus the blob size
formulas of ``docs/spec/stitch-gpu.md``. Everything is checked eagerly in
:func:`load_header`, so a malformed file is rejected before any device memory
is touched.

The file is opened ``"rb"`` and closed before this module returns. No blob
payload is read here.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .errors import (
    AlignmentError,
    CodecError,
    GroupSizeError,
    HeaderError,
    OverlapError,
    SizeMismatchError,
    TensorNotFoundError,
    TruncatedError,
)

__all__ = ["Blob", "TensorInfo", "Header", "load_header", "iter_linears", "align64"]

MAX_HEADER_NBYTES = 100_000_000
MAX_TENSORS = 1_000_000
MAX_AXIS = (1 << 24) - 1
MAX_NAME_BYTES = 1024

NF4_GROUP_SIZE = 64
VQ_GROUP_SIZE = 8

_ROOT_KEYS = (
    "magic",
    "version",
    "arch",
    "hidden_size",
    "intermediate_size",
    "num_layers",
    "vocab_size",
    "tile",
    "tensors",
)

_KINDS = frozenset(
    ("q", "k", "v", "o", "qkv", "gate", "up", "down", "embed", "lm_head", "norm", "other")
)

# kinds that take the file-wide codec ("one codec per file", chr0.md §2.3)
_QUANTIZABLE_KINDS = frozenset(
    ("q", "k", "v", "o", "qkv", "gate", "up", "down", "embed", "lm_head")
)

_BLOB_KEYS = frozenset(("data", "scale", "zero", "codebook", "index"))

# codec -> (required keys, optional keys). Anything else is rejected.
_CODEC_KEYS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "bf16": (frozenset(("data",)), frozenset()),
    "nf4": (frozenset(("group_size", "data", "scale")), frozenset()),
    "int4": (frozenset(("group_size", "data", "scale")), frozenset(("zero",))),
    "vq": (
        frozenset(("group_size", "n_codebooks", "codebook_bits", "codebook", "index")),
        frozenset(),
    ),
}

_ALWAYS_KEYS = frozenset(("kind", "codec", "shape"))


def align64(x: int) -> int:
    """Round ``x`` up to a multiple of 64 (chr0.md §1.1)."""
    return (x + 63) & ~63


def _pad_to(n_in: int, group: int) -> int:
    return ((n_in + group - 1) // group) * group


# --------------------------------------------------------------------------- #
# JSON primitives
# --------------------------------------------------------------------------- #


def _reject_float(literal: str) -> Any:
    raise HeaderError(f"json_invalid: non-integer number {literal!r} in header")


def _reject_constant(literal: str) -> Any:
    raise HeaderError(f"json_invalid: JSON constant {literal!r} in header")


def _object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise HeaderError(f"json_invalid: duplicate key {key!r}")
        out[key] = value
    return out


def _as_int(value: Any, what: str) -> int:
    # bool is an int subclass; `true` is not a JSON integer.
    if isinstance(value, bool) or not isinstance(value, int):
        raise HeaderError(f"{what}: expected integer, got {type(value).__name__}")
    return value


def _as_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise HeaderError(f"{what}: expected string, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Blob:
    """A ``[start, end)`` byte range measured from the start of the file."""

    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TensorInfo:
    """One entry of the ``tensors`` object, already validated."""

    name: str
    kind: str
    codec: str
    shape: tuple[int, ...]
    layer: int | None
    group_size: int | None
    n_codebooks: int | None
    codebook_bits: int | None
    blobs: Mapping[str, Blob]

    @property
    def M(self) -> int:
        """Logical ``n_out`` = ``shape[0]``."""
        return self.shape[0]

    @property
    def K(self) -> int:
        """Logical ``n_in`` = ``shape[1]``; only meaningful for rank 2."""
        if len(self.shape) != 2:
            raise SizeMismatchError(f"{self.name}: rank {len(self.shape)} has no K")
        return self.shape[1]

    @property
    def K_pad(self) -> int:
        """``group_size * ceil(K / group_size)`` for quantized codecs."""
        group = self.group_size or NF4_GROUP_SIZE
        return _pad_to(self.K, group)

    @property
    def n_groups(self) -> int:
        return self.K_pad // (self.group_size or NF4_GROUP_SIZE)


@dataclass(frozen=True)
class Header:
    """A fully validated CHR0 header. Holds no payload bytes and no file handle."""

    path: str
    header_nbytes: int
    payload_start: int
    file_size: int
    magic: str
    version: int
    arch: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    vocab_size: int
    tile_row: int
    tile_col_group: int
    tensors: Mapping[str, TensorInfo]

    def tensor(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise TensorNotFoundError(f"not_found: {name!r} not in {self.path}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.tensors


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _blob_sizes(name: str, codec: str, shape: tuple[int, ...], group_size: int | None) -> dict[str, int]:
    """Expected ``end - start`` per blob key (gpu-abi.md §3)."""
    if codec == "bf16":
        n = 1
        for axis in shape:
            n *= axis
        return {"data": 2 * n}

    n_out, n_in = shape[0], shape[1]
    if codec in ("nf4", "int4"):
        k_pad = _pad_to(n_in, NF4_GROUP_SIZE)
        n_groups = k_pad // NF4_GROUP_SIZE
        scale_bytes = n_out * n_groups * 2
        return {
            "data": n_out * k_pad // 2,
            "scale": scale_bytes,
            "zero": scale_bytes,
        }
    if codec == "vq":
        k_pad = _pad_to(n_in, VQ_GROUP_SIZE)
        return {
            "codebook": 2 * 256 * 8 * 2,
            "index": n_out * (k_pad // VQ_GROUP_SIZE) * 2,
        }
    raise CodecError(f"{name}: unknown codec {codec!r}")


def _layer_from_name(name: str) -> int | None:
    """First ``layers.<n>`` dotted pair, left to right (chr0.md §6.4)."""
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "layers":
            candidate = parts[i + 1]
            if candidate.isdigit() and candidate.isascii():
                return int(candidate)
            return None
    return None


def _check_name(name: str) -> None:
    raw = name.encode("utf-8")
    if not 1 <= len(raw) <= MAX_NAME_BYTES:
        raise HeaderError(f"tensor name length {len(raw)} out of range 1..{MAX_NAME_BYTES}")
    if any(ch < 0x20 for ch in raw):
        raise HeaderError(f"tensor name {name!r} contains an ASCII control byte")


def _parse_blob(name: str, key: str, value: Any, expected: int, file_size: int) -> Blob:
    if not isinstance(value, list) or len(value) != 2:
        raise HeaderError(f"{name}.{key}: expected [start, end)")
    start = _as_int(value[0], f"{name}.{key}.start")
    end = _as_int(value[1], f"{name}.{key}.end")
    if start < 0 or end > (1 << 63) - 1:
        raise HeaderError(f"{name}.{key}: offsets out of range")
    if end <= start:
        raise SizeMismatchError(f"{name}.{key}: end {end} <= start {start}")
    if start % 64 != 0:
        raise AlignmentError(f"{name}.{key}: start {start} is not 64-aligned")
    if end - start != expected:
        raise SizeMismatchError(
            f"{name}.{key}: {end - start} bytes on disk, formula says {expected}"
        )
    if end > file_size:
        raise TruncatedError(
            f"truncated: {name}.{key} ends at {end}, file is {file_size} bytes"
        )
    return Blob(start, end)


def _parse_tensor(
    name: str, obj: Any, num_layers: int, file_size: int
) -> TensorInfo:
    if not isinstance(obj, dict):
        raise HeaderError(f"{name}: tensor entry is not an object")
    _check_name(name)

    codec = _as_str(obj.get("codec"), f"{name}.codec")
    if codec not in _CODEC_KEYS:
        raise CodecError(f"{name}: unknown codec {codec!r}")
    required, optional = _CODEC_KEYS[codec]

    allowed = _ALWAYS_KEYS | {"layer"} | required | optional
    unexpected = set(obj) - allowed
    if unexpected:
        raise HeaderError(f"{name}: unexpected key(s) {sorted(unexpected)} for codec {codec}")
    missing = (_ALWAYS_KEYS | required) - set(obj)
    if missing:
        raise HeaderError(f"{name}: missing key(s) {sorted(missing)} for codec {codec}")

    kind = _as_str(obj["kind"], f"{name}.kind")
    if kind not in _KINDS:
        raise HeaderError(f"{name}: bad kind {kind!r}")

    raw_shape = obj["shape"]
    if not isinstance(raw_shape, list) or not 1 <= len(raw_shape) <= 2:
        raise HeaderError(f"{name}.shape: rank must be 1 or 2")
    shape = tuple(_as_int(a, f"{name}.shape") for a in raw_shape)
    if any(a < 1 or a > MAX_AXIS for a in shape):
        raise HeaderError(f"{name}.shape: axis out of range 1..{MAX_AXIS}")
    numel = 1
    for axis in shape:
        numel *= axis
    if numel * 4 > (1 << 32):
        raise HeaderError(f"{name}.shape: {numel} elements exceeds the 4 GiB tensor limit")
    if codec != "bf16" and len(shape) != 2:
        raise HeaderError(f"{name}: codec {codec} requires rank 2, got {shape}")

    want_layer = _layer_from_name(name)
    if want_layer is None:
        if "layer" in obj:
            raise HeaderError(f"{name}: has 'layer' but no 'layers.<n>' segment")
        layer = None
    else:
        if "layer" not in obj:
            raise HeaderError(f"{name}: missing 'layer' for 'layers.{want_layer}'")
        layer = _as_int(obj["layer"], f"{name}.layer")
        if layer != want_layer:
            raise HeaderError(f"{name}: layer {layer} disagrees with name segment {want_layer}")
        if num_layers > 0 and not 0 <= layer < num_layers:
            raise HeaderError(f"{name}: layer {layer} outside 0..{num_layers - 1}")

    group_size = None
    if "group_size" in obj:
        group_size = _as_int(obj["group_size"], f"{name}.group_size")
        canonical = VQ_GROUP_SIZE if codec == "vq" else NF4_GROUP_SIZE
        if group_size != canonical:
            raise GroupSizeError(
                f"{name}: group_size {group_size} for codec {codec}, must be {canonical}"
            )
    n_codebooks = None
    if "n_codebooks" in obj:
        n_codebooks = _as_int(obj["n_codebooks"], f"{name}.n_codebooks")
        if n_codebooks != 2:
            raise HeaderError(f"{name}: n_codebooks must be 2, got {n_codebooks}")
    codebook_bits = None
    if "codebook_bits" in obj:
        codebook_bits = _as_int(obj["codebook_bits"], f"{name}.codebook_bits")
        if codebook_bits != 8:
            raise HeaderError(f"{name}: codebook_bits must be 8, got {codebook_bits}")

    expected = _blob_sizes(name, codec, shape, group_size)
    blobs = {
        key: _parse_blob(name, key, obj[key], expected[key], file_size)
        for key in sorted(_BLOB_KEYS & set(obj))
    }

    return TensorInfo(
        name=name,
        kind=kind,
        codec=codec,
        shape=shape,
        layer=layer,
        group_size=group_size,
        n_codebooks=n_codebooks,
        codebook_bits=codebook_bits,
        blobs=blobs,
    )


def _check_overlaps(tensors: Mapping[str, TensorInfo]) -> None:
    ranges: list[tuple[int, int, str]] = [
        (blob.start, blob.end, f"{info.name}.{key}")
        for info in tensors.values()
        for key, blob in info.blobs.items()
    ]
    ranges.sort()
    for (a_start, a_end, a_name), (b_start, b_end, b_name) in zip(ranges, ranges[1:]):
        if b_start < a_end:
            raise OverlapError(
                f"overlap: {a_name} [{a_start},{a_end}) and {b_name} [{b_start},{b_end})"
            )


def _check_single_codec(tensors: Mapping[str, TensorInfo]) -> None:
    quantized = {
        info.codec
        for info in tensors.values()
        if info.kind in _QUANTIZABLE_KINDS and not info.name.endswith(".bias")
    }
    if len(quantized) > 1:
        raise CodecError(f"mixed codecs in one file: {sorted(quantized)}")
    if quantized and quantized <= {"bf16"}:
        raise CodecError("quantizable tensors carry codec bf16; expected nf4/vq/int4")


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def load_header(path: str) -> Header:
    """Read and fully validate the header of a CHR0 file.

    Opens ``path`` read-only, reads ``8 + header_nbytes`` bytes, closes it.
    Every blob range in the file is checked here -- alignment, size formula,
    overlap, EOF -- so that a later :func:`materialize_nf4` cannot start a
    host-to-device copy against a bad range.
    """
    path = str(path)
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size < 8:
            raise TruncatedError(f"truncated: {path} is {file_size} bytes, need at least 8")
        f.seek(0)
        (header_nbytes,) = struct.unpack("<Q", f.read(8))
        if header_nbytes < 2:
            raise HeaderError(f"header_nbytes: {header_nbytes} cannot hold a JSON object")
        if header_nbytes > MAX_HEADER_NBYTES:
            raise HeaderError(f"header_too_large: header_nbytes={header_nbytes}")
        if 8 + header_nbytes > file_size:
            raise TruncatedError(
                f"truncated: header_nbytes={header_nbytes} runs past EOF at {file_size}"
            )
        raw = f.read(header_nbytes)
    if len(raw) != header_nbytes:
        raise TruncatedError(f"truncated: short read of header at {path}")
    if raw[0] != 0x7B:
        raise HeaderError("json_invalid: header does not start with '{' (BOM? padding?)")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HeaderError(f"json_invalid: header is not valid UTF-8 ({exc})") from None
    try:
        root = json.loads(
            text,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_pairs,
        )
    except json.JSONDecodeError as exc:
        raise HeaderError(f"json_invalid: {exc}") from None
    if not isinstance(root, dict):
        raise HeaderError("json_invalid: header root is not an object")

    if set(root) != set(_ROOT_KEYS):
        missing = sorted(set(_ROOT_KEYS) - set(root))
        extra = sorted(set(root) - set(_ROOT_KEYS))
        raise HeaderError(f"header root keys: missing={missing} unexpected={extra}")

    magic = _as_str(root["magic"], "magic")
    if magic != "CHR0":
        raise HeaderError(f"magic: expected 'CHR0', got {magic!r}")
    version = _as_int(root["version"], "version")
    if version != 1:
        raise HeaderError(f"unsupported version: {version}")
    arch = _as_str(root["arch"], "arch")
    if not arch:
        raise HeaderError("arch: must be non-empty")
    hidden_size = _as_int(root["hidden_size"], "hidden_size")
    intermediate_size = _as_int(root["intermediate_size"], "intermediate_size")
    num_layers = _as_int(root["num_layers"], "num_layers")
    vocab_size = _as_int(root["vocab_size"], "vocab_size")
    if hidden_size < 1 or min(intermediate_size, num_layers, vocab_size) < 0:
        raise HeaderError("header root: negative or zero size field")

    tile = root["tile"]
    if not isinstance(tile, dict) or set(tile) != {"row", "col_group"}:
        raise HeaderError("tile: expected exactly {'row', 'col_group'}")
    tile_row = _as_int(tile["row"], "tile.row")
    tile_col_group = _as_int(tile["col_group"], "tile.col_group")
    if (tile_row, tile_col_group) != (64, 8):
        raise HeaderError(f"tile: expected row=64 col_group=8, got {tile_row}/{tile_col_group}")

    raw_tensors = root["tensors"]
    if not isinstance(raw_tensors, dict) or not raw_tensors:
        raise HeaderError("no tensors: 'tensors' must be a non-empty object")
    if len(raw_tensors) > MAX_TENSORS:
        raise HeaderError(f"too many tensors: {len(raw_tensors)}")

    tensors = {
        name: _parse_tensor(name, obj, num_layers, file_size)
        for name, obj in raw_tensors.items()
    }
    _check_overlaps(tensors)
    _check_single_codec(tensors)

    return Header(
        path=path,
        header_nbytes=header_nbytes,
        payload_start=align64(8 + header_nbytes),
        file_size=file_size,
        magic=magic,
        version=version,
        arch=arch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        tile_row=tile_row,
        tile_col_group=tile_col_group,
        tensors=tensors,
    )


def iter_linears(header: Header) -> Iterator[str]:
    """Yield CHR0 names whose ``codec`` is ``nf4``, in header key order.

    ``embed_tokens`` and ``lm_head`` are nf4 in the file and therefore appear
    here; deciding which of them actually go through the GEMM kernel is the
    host's job, not the loader's.
    """
    for name, info in header.tensors.items():
        if info.codec == "nf4":
            yield name
