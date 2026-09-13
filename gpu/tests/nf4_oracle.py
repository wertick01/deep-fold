"""CPU oracle for CHR0 NF4 -- floor 1 of docs/token-loop.md SS7.3.

Normative source: docs/spec/nf4.md -- LUT literals (SS1), nibble packing (SS5),
decode (SS3), encode (SS2). This module decodes *packed bytes* only. It never
opens original safetensors and never shells out to `chr decode`.

Everything is float32 by contract:

    s     = fp16_to_float32(scale[r, g])       # exact widening
    W_hat = float32(LUT[nib]) * s              # float32 product
    Y_cpu = W_hat @ x                          # float32 matmul
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# --- SS1: the 16 levels, literal-for-literal from get_4bit_type("nf4") -------
NF4_LEVELS: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

# binary32 bit patterns, column "binary32 bits" of docs/spec/nf4.md SS1.
NF4_BITS: tuple[int, ...] = (
    0xBF800000,
    0xBF3239B1,
    0xBF066B30,
    0xBECA32A0,
    0xBE91A24D,
    0xBE3D353F,
    0xBDBA7871,
    0x00000000,
    0x3DA2FAFF,
    0x3E24CAE3,
    0x3E7C04DD,
    0x3EAD033A,
    0x3EE1A4B8,
    0x3F1007AB,
    0x3F3913B3,
    0x3F800000,
)

LUT = np.array(NF4_LEVELS, dtype=np.float32)

GROUP_SIZE = 64

# docs/spec/nf4.md SS8 golden row: W[i] = float32(i) / float32(63), one group.
GOLDEN64_HEX = "778788889999A9AAAABABBBBCBCCCCCCDDDDDDDDEDEEEEEEEEEEEEFEFFFFFFFF"
GOLDEN64_IDX = (
    7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15,
)
# docs/spec/nf4.md SS9.4, row 1: W[i] = float32(2*i - 63) / float32(63).
GOLDEN2X64_ROW1_HEX = "00001011111121222233434454556676878899AABABBCCCCDDDDEEEEEEFEFFFF"
GOLDEN2X64_METRICS = {
    "rmse": (0.05184147829055911, 0.05185),
    "maxabs": (0.14507704973220825, 0.14508),
    "mae": (0.03979102736047935, 0.03980),
}


def f32_bits(x: float) -> int:
    """binary32 bit pattern of a Python float rounded to float32 (RNE)."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def k_pad(k: int) -> int:
    """64 * ceil(K / 64) -- stitch-gpu.md size formulas."""
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")
    return GROUP_SIZE * ((k + GROUP_SIZE - 1) // GROUP_SIZE)


def nf4_blob_sizes(m: int, k: int) -> tuple[int, int, int, int]:
    """(data_nbytes, scale_nbytes, K_pad, n_groups) for a logical [M, K]."""
    if m < 1:
        raise ValueError(f"M must be >= 1, got {m}")
    kp = k_pad(k)
    n_groups = kp // GROUP_SIZE
    return m * kp // 2, m * n_groups * 2, kp, n_groups


# --- SS3: decode -------------------------------------------------------------
def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """[M, K_pad/2] uint8 -> [M, K_pad] uint8 nibble indices.

    Low nibble = even K (docs/spec/nf4.md SS5.1) -- NOT the bitsandbytes CUDA
    packing, where the even weight lands in the high nibble.
    """
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise TypeError(f"packed must be uint8 [M, K_pad/2], got {packed.dtype} {packed.shape}")
    m, half = packed.shape
    nib = np.empty((m, half * 2), dtype=np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = packed >> 4
    return nib


def decode_nf4(packed: np.ndarray, scale: np.ndarray, m: int, k: int) -> np.ndarray:
    """Decode CHR0 NF4 blobs to W_hat float32 [M, K] (padding columns dropped)."""
    data_nbytes, scale_nbytes, kp, n_groups = nf4_blob_sizes(m, k)
    if packed.shape != (m, kp // 2):
        raise ValueError(f"packed shape {packed.shape} != {(m, kp // 2)}")
    if scale.shape != (m, n_groups):
        raise ValueError(f"scale shape {scale.shape} != {(m, n_groups)}")
    if scale.dtype != np.float16:
        raise TypeError(f"scale must be float16 (IEEE binary16), got {scale.dtype}")
    del data_nbytes, scale_nbytes

    w = LUT[unpack_nibbles(packed)]                 # float32 [M, K_pad]
    s = scale.astype(np.float32)                    # exact fp16 -> f32
    w = w.reshape(m, n_groups, GROUP_SIZE)
    w *= s[:, :, None]                              # float32 product, in place
    w = w.reshape(m, kp)
    return w[:, :k]                                 # logical K only


def matmul_f32(w_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Y = W_hat @ x in float32. x is [K, N] float32, y is [M, N] float32."""
    if w_hat.dtype != np.float32 or x.dtype != np.float32:
        raise TypeError("oracle matmul is float32 only")
    return w_hat @ x


def matmul_f64_chunked(w_hat: np.ndarray, x: np.ndarray, rows: int = 2048) -> np.ndarray:
    """Same product accumulated in float64, for reporting the oracle's own noise."""
    m = w_hat.shape[0]
    out = np.empty((m, x.shape[1]), dtype=np.float64)
    xd = x.astype(np.float64)
    for lo in range(0, m, rows):
        hi = min(lo + rows, m)
        out[lo:hi] = w_hat[lo:hi].astype(np.float64) @ xd
    return out


# --- SS2: encode (only used by the golden self-checks and toy fixtures) ------
def encode_nf4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode float32 [M, K] -> (packed uint8 [M, K_pad/2], scale float16 [M, n_groups]).

    Steps are exactly docs/spec/nf4.md SS2.2: max-abs -> FP16 scale -> divide by
    float32(s16) -> clip -> nearest level by L2 -> tie to the lower index.
    """
    if w.dtype != np.float32 or w.ndim != 2:
        raise TypeError(f"encode wants float32 [M, K], got {w.dtype} {w.shape}")
    m, k = w.shape
    _, _, kp, n_groups = nf4_blob_sizes(m, k)
    if not np.isfinite(w).all():
        raise ValueError("NaN/Inf in encode input (docs/spec/nf4.md SS6.3)")

    g = np.zeros((m, kp), dtype=np.float32)
    g[:, :k] = w
    g = g.reshape(m, n_groups, GROUP_SIZE)

    s32 = np.abs(g).max(axis=2)
    s32 = np.where(s32 == 0, np.float32(1.0), s32).astype(np.float32)
    s16 = s32.astype(np.float16)
    if not np.isfinite(s16).all() or (s16 <= 0).any():
        raise ValueError("scale not finite-positive in fp16 (docs/spec/nf4.md SS2.2 step 3)")
    s = s16.astype(np.float32)

    u = np.clip(g / s[:, :, None], np.float32(-1.0), np.float32(1.0)).astype(np.float32)
    d = u[:, :, :, None] - LUT[None, None, None, :]
    idx = np.argmin((d * d).astype(np.float32), axis=3).astype(np.uint8)  # first min == lower index

    idx = idx.reshape(m, kp)
    packed = (idx[:, 1::2] << 4) | idx[:, 0::2]
    return packed.astype(np.uint8), s16


@dataclass(frozen=True)
class Check:
    ident: str
    ok: bool
    detail: str


def _hex(b: np.ndarray) -> str:
    return b.tobytes().hex().upper()


def selfcheck() -> list[Check]:
    """LUT / packing / decode self-checks against docs/spec/nf4.md goldens.

    These must pass with no GPU, no loader and no kernel: if they fail, every
    downstream number in the gate is meaningless.
    """
    out: list[Check] = []

    bad = [
        (i, hex(f32_bits(v)), hex(b))
        for i, (v, b) in enumerate(zip(NF4_LEVELS, NF4_BITS))
        if f32_bits(v) != b
    ]
    out.append(Check("L0", not bad, "16 LUT literals == nf4.md SS1 bits" if not bad else f"mismatch {bad}"))

    plus_zero = f32_bits(NF4_LEVELS[7]) == 0x00000000
    increasing = all(NF4_LEVELS[i] < NF4_LEVELS[i + 1] for i in range(15))
    lut_dtype = LUT.dtype == np.float32 and all(f32_bits(float(LUT[i])) == NF4_BITS[i] for i in range(16))
    ok = plus_zero and increasing and lut_dtype
    out.append(
        Check(
            "L1",
            ok,
            f"idx7 is +0.0={plus_zero}, strictly increasing={increasing}, numpy LUT bits ok={lut_dtype}",
        )
    )

    # SS5.2: packing is a pure function of two indices; low nibble = even column.
    pairs = {(7, 15): 0xF7, (15, 7): 0x7F, (3, 12): 0xC3}
    got = {}
    for (i0, i1), want in pairs.items():
        idx = np.array([[i0, i1]], dtype=np.uint8)
        got[(i0, i1)] = int((idx[:, 1::2] << 4 | idx[:, 0::2])[0, 0])
    ok = all(got[p] == w for p, w in pairs.items())
    out.append(Check("L2", ok, f"nibble pack pairs -> {[hex(v) for v in got.values()]} want F7/7F/C3"))

    # SS8 golden: encode(W[i]=i/63) and decode round trip.
    w8 = (np.arange(64, dtype=np.float32) / np.float32(63.0)).reshape(1, 64)
    p8, s8 = encode_nf4(w8)
    idx8 = unpack_nibbles(p8)[0]
    enc_ok = _hex(p8) == GOLDEN64_HEX and int(s8.view(np.uint16)[0, 0]) == 0x3C00
    idx_ok = bool((idx8 == np.array(GOLDEN64_IDX, dtype=np.uint8)).all())
    dec = decode_nf4(p8, s8, 1, 64)
    want = LUT[np.array(GOLDEN64_IDX, dtype=np.uint8)].reshape(1, 64)
    bit_ok = bool((dec.view(np.uint32) == want.view(np.uint32)).all())
    out.append(
        Check(
            "L3",
            enc_ok and idx_ok and bit_ok,
            f"SS8 packed hex ok={enc_ok}, idx ok={idx_ok}, decode==LUT bitwise={bit_ok}, scale=0x3C00",
        )
    )

    # SS9.4 golden 2x64 + hard metric thresholds.
    i = np.arange(64, dtype=np.float32)
    w2 = np.stack([i / np.float32(63.0), (np.float32(2.0) * i - np.float32(63.0)) / np.float32(63.0)])
    p2, s2 = encode_nf4(w2.astype(np.float32))
    hex_ok = _hex(p2[0]) == GOLDEN64_HEX and _hex(p2[1]) == GOLDEN2X64_ROW1_HEX
    scale_ok = bool((s2.view(np.uint16) == 0x3C00).all())
    err = (decode_nf4(p2, s2, 2, 64) - w2).astype(np.float32)
    e64 = err.astype(np.float64)
    metrics = {
        "rmse": float(np.sqrt((e64**2).mean())),
        "maxabs": float(np.abs(e64).max()),
        "mae": float(np.abs(e64).mean()),
    }
    m_ok = all(metrics[k2] <= lim for k2, (_, lim) in GOLDEN2X64_METRICS.items())
    out.append(
        Check(
            "L4",
            hex_ok and scale_ok and m_ok,
            "SS9.4 packed hex ok={} scale ok={} rmse={:.10f} maxabs={:.10f} mae={:.10f}".format(
                hex_ok, scale_ok, metrics["rmse"], metrics["maxabs"], metrics["mae"]
            ),
        )
    )

    # SS6.1 padding: [1,2] = [0,1] -> one group, 62 tail zeros with nibble 7.
    p5, s5 = encode_nf4(np.array([[0.0, 1.0]], dtype=np.float32))
    pad_ok = _hex(p5) == "F7" + "77" * 31 and int(s5.view(np.uint16)[0, 0]) == 0x3C00
    dec5 = decode_nf4(p5, s5, 1, 2)
    pad_ok = pad_ok and dec5.shape == (1, 2) and float(dec5[0, 0]) == 0.0 and float(dec5[0, 1]) == 1.0
    out.append(Check("L5", pad_ok, f"SS6.1 pad-to-64 packed/scale/decode shape ok={pad_ok}"))

    return out


def pack_nibbles(nib: np.ndarray) -> np.ndarray:
    """[M, K_pad] nibble indices -> [M, K_pad/2] packed bytes (low = even K)."""
    if nib.dtype != np.uint8 or nib.shape[1] % 2:
        raise TypeError("nibbles must be uint8 with an even K_pad")
    return ((nib[:, 1::2] << 4) | nib[:, 0::2]).astype(np.uint8)


def toy_scale(m: int, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    """Weight-like FP16 group scales, so |y| stays O(1) at rms(x) ~ 1."""
    scale = rng.uniform(0.012, 0.055, size=(m, n_groups)).astype(np.float16)
    if (scale <= 0).any() or not np.isfinite(scale).all():
        raise AssertionError("toy scale must be finite positive")
    return scale


def toy_nf4(m: int, k: int, seed: int, *, pad_garbage: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Random packed/scale blobs for a toy [M, K].

    Nibbles are uniform over 0..15 (all 16 levels exercised). Scales are drawn
    in a weight-like range so that |y| stays O(1) at rms(x) ~ 1, which is what
    the 0.05 floor-1 threshold is calibrated for.

    pad_garbage: fill the padding columns (K .. K_pad-1) with non-zero nibbles.
    A real .chr writes nibble 7 there, so garbage is the strictly harder test:
    if the kernel reads pad as live K, y drifts and F2 fails.
    """
    _, _, kp, n_groups = nf4_blob_sizes(m, k)
    rng = np.random.default_rng(seed)
    nib = rng.integers(0, 16, size=(m, kp), dtype=np.uint8)
    if not pad_garbage and kp > k:
        nib[:, k:] = 7
    return pack_nibbles(nib), toy_scale(m, n_groups, rng)
