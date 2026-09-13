"""Minimal read-only CHR0 access + toy .chr fixtures, for gpu/tests only.

Why a second reader exists: the oracle must be able to say "the bytes the GPU
holds are the bytes on disk" without borrowing the loader it is judging
(gpu/chr0, agent 1). So this module parses the container itself, strictly
read-only ("rb", no mmap, no writes), per docs/spec/chr0.md SS1-SS2 and SS7.2.

It is deliberately partial: only what the gate needs (header, nf4 blobs) plus a
toy writer used to build small fixtures and the malformed-header cases of S4.
This is NOT a second implementation of the container for production use.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from nf4_oracle import GROUP_SIZE, nf4_blob_sizes

ALIGN = 64
MAX_HEADER_NBYTES = 100_000_000

ROOT_KEY_ORDER = (
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
TENSOR_KEY_ORDER = (
    "layer",
    "kind",
    "codec",
    "shape",
    "group_size",
    "n_codebooks",
    "codebook_bits",
    "data",
    "scale",
    "zero",
    "codebook",
    "index",
)


class Chr0Error(Exception):
    """Container-level rejection (truncated / size mismatch / overlap / codec)."""


def align64(x: int) -> int:
    return (x + ALIGN - 1) & ~(ALIGN - 1)


@dataclass(frozen=True)
class Header:
    path: str
    nbytes: int
    filesize: int
    root: dict[str, Any]

    @property
    def tensors(self) -> dict[str, Any]:
        return self.root["tensors"]


def read_header(path: str) -> Header:
    filesize = os.path.getsize(path)
    if filesize < 8:
        raise Chr0Error(f"truncated: {filesize} bytes")
    with open(path, "rb") as f:                      # read-only, always
        n = struct.unpack("<Q", f.read(8))[0]
        if n < 2 or n > MAX_HEADER_NBYTES:
            raise Chr0Error(f"header_nbytes out of range: {n}")
        if 8 + n > filesize:
            raise Chr0Error(f"header_nbytes lies: 8+{n} > filesize {filesize}")
        raw = f.read(n)
    if raw[:1] != b"{":
        raise Chr0Error("json_invalid: does not start with '{'")
    root = json.loads(raw.decode("utf-8"))
    if not isinstance(root, dict):
        raise Chr0Error("json_invalid: root is not an object")
    if root.get("magic") != "CHR0":
        raise Chr0Error(f"magic != CHR0: {root.get('magic')!r}")
    if root.get("version") != 1:
        raise Chr0Error(f"unsupported version: {root.get('version')!r}")
    if root.get("tile") != {"row": 64, "col_group": 8}:
        raise Chr0Error(f"tile != row 64 / col_group 8: {root.get('tile')!r}")
    if not isinstance(root.get("tensors"), dict) or not root["tensors"]:
        raise Chr0Error("no tensors")
    return Header(path=path, nbytes=n, filesize=filesize, root=root)


def _check_range(name: str, key: str, rng: Any, want_nbytes: int, filesize: int) -> tuple[int, int]:
    if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, int) for v in rng)):
        raise Chr0Error(f"{name}.{key}: not a pair of integers: {rng!r}")
    start, end = rng
    if start % ALIGN != 0:
        raise Chr0Error(f"{name}.{key}: start {start} not 64-aligned")
    if end <= start:
        raise Chr0Error(f"{name}.{key}: end {end} <= start {start}")
    if end - start != want_nbytes:
        raise Chr0Error(f"{name}.{key}: {end - start} bytes, formula says {want_nbytes}")
    if end > filesize:
        raise Chr0Error(f"{name}.{key}: end {end} > filesize {filesize} (truncated)")
    return start, end


def nf4_ranges(header: Header, name: str) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    """Validated (data, scale) byte ranges plus logical (M, K)."""
    t = header.tensors.get(name)
    if t is None:
        raise Chr0Error(f"not_found: {name}")
    if t.get("codec") != "nf4":
        raise Chr0Error(f"{name}: codec {t.get('codec')!r} != nf4 (wave 2.0 is nf4 only)")
    if t.get("group_size") != GROUP_SIZE:
        raise Chr0Error(f"{name}: group_size {t.get('group_size')!r} != 64")
    for forbidden in ("zero", "codebook", "index", "n_codebooks", "codebook_bits"):
        if forbidden in t:
            raise Chr0Error(f"{name}: key {forbidden!r} forbidden for codec nf4")
    shape = t.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2 and all(isinstance(v, int) and v >= 1 for v in shape)):
        raise Chr0Error(f"{name}: shape {shape!r} must be rank 2, each >= 1")
    m, k = shape
    data_nbytes, scale_nbytes, _, _ = nf4_blob_sizes(m, k)
    data = _check_range(name, "data", t.get("data"), data_nbytes, header.filesize)
    scale = _check_range(name, "scale", t.get("scale"), scale_nbytes, header.filesize)
    lo, hi = (data, scale) if data[0] <= scale[0] else (scale, data)
    if lo[1] > hi[0]:
        raise Chr0Error(f"{name}: blobs overlap: {lo} vs {hi}")
    return data, scale, m, k


def read_nf4(header: Header, name: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(packed uint8 [M, K_pad/2], scale float16 [M, n_groups], M, K) straight from disk."""
    (d0, d1), (s0, s1), m, k = nf4_ranges(header, name)
    _, _, kp, n_groups = nf4_blob_sizes(m, k)
    with open(header.path, "rb") as f:                # read-only, always
        f.seek(d0)
        data = f.read(d1 - d0)
        f.seek(s0)
        sc = f.read(s1 - s0)
    if len(data) != d1 - d0 or len(sc) != s1 - s0:
        raise Chr0Error(f"{name}: short read (truncated file)")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(m, kp // 2)
    scale = np.frombuffer(sc, dtype="<f2").reshape(m, n_groups)
    return packed, scale, m, k


def iter_nf4_names(header: Header) -> list[str]:
    return [n for n, t in header.tensors.items() if t.get("codec") == "nf4"]


# --- toy fixtures ------------------------------------------------------------
@dataclass(frozen=True)
class ToyNF4:
    name: str
    kind: str
    layer: int | None
    packed: np.ndarray
    scale: np.ndarray
    m: int
    k: int


@dataclass(frozen=True)
class ToyBF16:
    name: str
    kind: str
    layer: int | None
    shape: tuple[int, ...]
    data: bytes


CORRUPTIONS = ("truncate", "unaligned_start", "overlap", "group_size", "data_len")


def _compact(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _tensor_obj(t: ToyNF4 | ToyBF16, blobs: dict[str, tuple[int, int]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    if t.layer is not None:
        obj["layer"] = t.layer
    obj["kind"] = t.kind
    if isinstance(t, ToyNF4):
        obj["codec"] = "nf4"
        obj["shape"] = [t.m, t.k]
        obj["group_size"] = GROUP_SIZE
        obj["data"] = list(blobs["data"])
        obj["scale"] = list(blobs["scale"])
    else:
        obj["codec"] = "bf16"
        obj["shape"] = list(t.shape)
        obj["data"] = list(blobs["data"])
    return {k: obj[k] for k in TENSOR_KEY_ORDER if k in obj}


def _blob_bytes(t: ToyNF4 | ToyBF16) -> dict[str, bytes]:
    if isinstance(t, ToyNF4):
        data_nbytes, scale_nbytes, _, _ = nf4_blob_sizes(t.m, t.k)
        data = t.packed.tobytes()
        scale = t.scale.astype("<f2").tobytes()
        if len(data) != data_nbytes or len(scale) != scale_nbytes:
            raise ValueError(f"{t.name}: toy blob length != formula")
        return {"data": data, "scale": scale}
    return {"data": t.data}


def plan_toy_header(
    tensors: Iterable[ToyNF4 | ToyBF16],
    *,
    arch: str = "toy",
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_layers: int = 1,
    vocab_size: int = 0,
    corrupt: str | None = None,
) -> tuple[bytes, dict[str, dict[str, tuple[int, int]]], int]:
    """Stabilized (header_json, blob plan, filesize) per docs/spec/chr0.md SS3.3.

    `corrupt` mutates the emitted JSON *inside* the stabilization loop, so the
    malformed file is otherwise well-formed: same N, same blob placement, one
    deliberate lie.
    """
    tensors = list(tensors)
    blob_bytes = {t.name: _blob_bytes(t) for t in tensors}
    n = 0
    for _ in range(8):
        pos = align64(8 + n)
        plan: dict[str, dict[str, tuple[int, int]]] = {}
        for t in tensors:
            plan[t.name] = {}
            for key, payload in blob_bytes[t.name].items():
                start = pos
                end = start + len(payload)
                plan[t.name][key] = (start, end)
                pos = align64(end)
        filesize = pos
        objs = {t.name: _tensor_obj(t, plan[t.name]) for t in tensors}
        if corrupt == "unaligned_start":
            first = tensors[0].name
            s, e = objs[first]["data"]
            objs[first]["data"] = [s - 8, e - 8]
        elif corrupt == "overlap":
            first = tensors[0].name
            ds, _ = objs[first]["data"]
            ss, se = objs[first]["scale"]
            objs[first]["scale"] = [ds, ds + (se - ss)]
        elif corrupt == "group_size":
            objs[tensors[0].name]["group_size"] = 32
        elif corrupt == "data_len":
            first = tensors[0].name
            s, e = objs[first]["data"]
            objs[first]["data"] = [s, e + 64]
        root = {
            "magic": "CHR0",
            "version": 1,
            "arch": arch,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "tile": {"row": 64, "col_group": 8},
            "tensors": {k: objs[k] for k in sorted(objs, key=lambda s: s.encode("utf-8"))},
        }
        root = {k: root[k] for k in ROOT_KEY_ORDER}
        js = _compact(root)
        if len(js) == n:
            return js, plan, filesize
        n = len(js)
    raise Chr0Error("header_not_stable")


def write_toy_chr(
    path: str,
    tensors: Iterable[ToyNF4 | ToyBF16],
    *,
    corrupt: str | None = None,
    **root_kw: Any,
) -> str:
    """Write a toy .chr. `corrupt` in CORRUPTIONS builds an S4 fixture."""
    if corrupt is not None and corrupt not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {corrupt!r}")
    tensors = list(tensors)
    js, plan, filesize = plan_toy_header(tensors, corrupt=corrupt, **root_kw)
    blob_bytes = {t.name: _blob_bytes(t) for t in tensors}
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)))
        f.write(js)
        f.write(b"\x00" * (align64(8 + len(js)) - (8 + len(js))))
        for t in tensors:
            for key, payload in blob_bytes[t.name].items():
                start, end = plan[t.name][key]
                f.seek(start)
                f.write(payload)
                f.write(b"\x00" * (align64(end) - end))
        f.truncate(filesize)
    if corrupt == "truncate":
        with open(path, "r+b") as f:                  # fixture only, never a real .chr
            f.truncate(filesize - ALIGN)
    return path


# docs/spec/chr0.md SS2.4: byte-exact golden header of the 3-tensor toy model.
GOLDEN_TOY_HEADER_NBYTES = 516
GOLDEN_TOY_HEADER_JSON = (
    '{"magic":"CHR0","version":1,"arch":"toy","hidden_size":64,"intermediate_size":128,'
    '"num_layers":1,"vocab_size":0,"tile":{"row":64,"col_group":8},"tensors":'
    '{"model.layers.0.mlp.down_proj":{"layer":0,"kind":"down","codec":"nf4","shape":[32,128],'
    '"group_size":64,"data":[2752,4800],"scale":[4800,4928]},'
    '"model.layers.0.self_attn.q_proj":{"layer":0,"kind":"q","codec":"nf4","shape":[64,64],'
    '"group_size":64,"data":[576,2624],"scale":[2624,2752]},'
    '"model.norm":{"kind":"norm","codec":"bf16","shape":[64],"data":[4928,5056]}}}'
)


def golden_toy_tensors() -> list[ToyNF4 | ToyBF16]:
    """The SS2.4 toy model, in safetensors traversal order (q, down, norm)."""
    def zeros(m: int, k: int) -> ToyNF4:
        data_nbytes, scale_nbytes, kp, n_groups = nf4_blob_sizes(m, k)
        del data_nbytes, scale_nbytes
        return ToyNF4(
            name="",
            kind="",
            layer=0,
            packed=np.zeros((m, kp // 2), dtype=np.uint8),
            scale=np.ones((m, n_groups), dtype=np.float16),
            m=m,
            k=k,
        )

    q = zeros(64, 64)
    d = zeros(32, 128)
    return [
        ToyNF4("model.layers.0.self_attn.q_proj", "q", 0, q.packed, q.scale, 64, 64),
        ToyNF4("model.layers.0.mlp.down_proj", "down", 0, d.packed, d.scale, 32, 128),
        ToyBF16("model.norm", "norm", None, (64,), b"\x00" * 128),
    ]


def selfcheck_writer() -> tuple[bool, str]:
    """Validate the toy writer against the byte-exact golden header of SS2.4."""
    js, plan, filesize = plan_toy_header(golden_toy_tensors())
    want = GOLDEN_TOY_HEADER_JSON.encode("utf-8")
    if js != want:
        for i, (a, b) in enumerate(zip(js, want)):
            if a != b:
                return False, f"json differs at byte {i}: got {js[i:i+40]!r} want {want[i:i+40]!r}"
        return False, f"json length {len(js)} != {len(want)}"
    ok = (
        len(js) == GOLDEN_TOY_HEADER_NBYTES
        and plan["model.layers.0.self_attn.q_proj"]["data"] == (576, 2624)
        and plan["model.layers.0.mlp.down_proj"]["scale"] == (4800, 4928)
        and filesize == 5056
    )
    return ok, f"N={len(js)} offsets/filesize({filesize}) match chr0.md SS2.4 golden={ok}"
