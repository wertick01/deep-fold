// Fused NF4 dequant + HMMA.16816 GEMM for sm_86.
// y = dequant_nf4(W) @ x. Dequant in registers; no BF16 W in HBM.
//
// Specs: docs/spec/stitch-gpu.md (wins), docs/spec/nf4.md §1, docs/kernel-ampere.md
// Decode N=1:    BM=128, BN=8 pad,  BK=256, block=256, stages=3, split_k=1.
// Prefill N=2..16: BM=64, BN=16 pad, BK=128, block=256, stages=3, epilogue mask.
// MMA: only mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// N>16 is not launched here: return -2 and let the host chunk.

#include "chr_gpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kBM = 128;
constexpr int kBN = 8;
constexpr int kBK = 256;
constexpr int kBlock = 256;
constexpr int kStages = 3;
constexpr int kGroup = 64;
constexpr int kPackedStageBytes = kBM * (kBK / 2); // 16384
constexpr int kXStageElems = kBK;                   // 256 bf16
constexpr int kSmemBytes =
    kStages * kPackedStageBytes + kStages * kXStageElems * 2;

// Prefill tile (kernel-ampere.md §2 / §4). Scales stay in __ldg, not smem.
constexpr int pBM = 64;
constexpr int pBN = 16;
constexpr int pBK = 128;
constexpr int pBlock = 256;
constexpr int pStages = 3;
constexpr int pPackedStageBytes = pBM * (pBK / 2); // 4096
constexpr int pXStageElems = pBK * pBN;             // 2048 bf16, layout [BK, 16]
constexpr int pSmemBytes =
    pStages * pPackedStageBytes + pStages * pXStageElems * 2; // 24576

static_assert(pBN == 16, "prefill pads N to 16 for two m16n8 along N");
static_assert(pPackedStageBytes == 4096, "prefill packed stage");
static_assert(pXStageElems == 2048, "prefill x stage");
static_assert(pSmemBytes == 24576, "prefill ring");
static_assert(pBlock == kBlock, "both launches are 256 threads");

// Canonical NF4 LUT, docs/spec/nf4.md §1, binary32 bits (not recomputed quantiles).
__constant__ uint32_t kNf4LutBits[16] = {
    0xBF800000u, // -1.0
    0xBF3239B1u, // -0.6961928009986877
    0xBF066B30u, // -0.5250730514526367
    0xBECA32A0u, // -0.39491748809814453
    0xBE91A24Du, // -0.28444138169288635
    0xBE3D353Fu, // -0.18477343022823334
    0xBDBA7871u, // -0.09105003625154495
    0x00000000u, // 0.0
    0x3DA2FAFFu, // 0.07958029955625534
    0x3E24CAE3u, // 0.16093020141124725
    0x3E7C04DDu, // 0.24611230194568634
    0x3EAD033Au, // 0.33791524171829224
    0x3EE1A4B8u, // 0.44070982933044434
    0x3F1007ABu, // 0.5626170039176941
    0x3F3913B3u, // 0.7229568362236023
    0x3F800000u, // 1.0
};

__device__ __forceinline__ float nf4_lut(unsigned nib) {
  return __uint_as_float(kNf4LutBits[nib]);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float w0, float w1) {
  const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(w0));
  const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(w1));
  return lo | (hi << 16);
}

__device__ __forceinline__ uint32_t pack_bf16x2_bits(__nv_bfloat16 a,
                                                    __nv_bfloat16 b) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(a)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(b)) << 16);
}

__device__ __forceinline__ void cp_async_cg_16(void *smem, const void *gmem,
                                                int src_bytes) {
  unsigned sm = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  // Immediate srcSize: 0/4/8/16. srcSize < 16 zero-fills the rest.
  if (src_bytes >= 16) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                     sm),
                 "l"(gmem));
  } else if (src_bytes >= 8) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 8;\n" ::"r"(sm),
                 "l"(gmem));
  } else if (src_bytes >= 4) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 4;\n" ::"r"(sm),
                 "l"(gmem));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(sm),
                 "l"(gmem));
  }
}

__device__ __forceinline__ void cp_async_ca_16(void *smem, const void *gmem,
                                                int src_bytes) {
  unsigned sm = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  if (src_bytes >= 16) {
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                     sm),
                 "l"(gmem));
  } else if (src_bytes >= 8) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 8;\n" ::"r"(sm),
                 "l"(gmem));
  } else if (src_bytes >= 4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 4;\n" ::"r"(sm),
                 "l"(gmem));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n" ::"r"(sm),
                 "l"(gmem));
  }
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mma_m16n8k16(uint32_t a0, uint32_t a1,
                                               uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1,
                                               float &d0, float &d1, float &d2,
                                               float &d3) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// PTX m16n8k16 A fragment (elements a0..a7 low-to-high, 4x .b32):
//   groupID = lane>>2, t = lane%4
//   a0,a1: (row=g,   k=2t+{0,1})     -> reg a0
//   a2,a3: (row=g+8, k=2t+{0,1})     -> reg a1
//   a4,a5: (row=g,   k=2t+{8,9})     -> reg a2
//   a6,a7: (row=g+8, k=2t+{8,9})     -> reg a3
// B: b0,b1 (k=2t+{0,1}, n=g); b2,b3 (k=2t+{8,9}, n=g)
// D: d0=(g, 2t), d1=(g, 2t+1), d2=(g+8, 2t), d3=(g+8, 2t+1)

__device__ void issue_packed(uint8_t *dst, const uint8_t *packed,
                             int m0, int M, int k0, int K_pad,
                             int packed_stride) {
  constexpr int kChunksPerRow = kBK / 32; // 8 * 16B = 128 packed bytes / row
  constexpr int kNChunks = kBM * kChunksPerRow;
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kNChunks; i += kBlock) {
    const int row = i / kChunksPerRow;
    const int chunk = i % kChunksPerRow;
    const int gm = m0 + row;
    const int col = (k0 >> 1) + chunk * 16;
    uint8_t *out = dst + row * (kBK / 2) + chunk * 16;
    int src_bytes = 0;
    const uint8_t *src = packed; // aligned dummy when OOB
    if (gm < M && col < packed_stride) {
      src = packed + static_cast<size_t>(gm) * static_cast<size_t>(packed_stride) +
            static_cast<size_t>(col);
      const int remain = packed_stride - col;
      src_bytes = remain >= 16 ? 16 : 0;
    }
    cp_async_cg_16(out, src, src_bytes);
  }
}

__device__ void issue_x(__nv_bfloat16 *dst, const __nv_bfloat16 *x, int k0,
                        int K) {
  constexpr int kChunks = kBK / 8; // 32 * 16B = 256 bf16
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kChunks; i += kBlock) {
    const int k = k0 + i * 8;
    __nv_bfloat16 *out = dst + i * 8;
    int src_bytes = 0;
    const __nv_bfloat16 *src = x;
    if (k < K) {
      const int remain = (K - k) * static_cast<int>(sizeof(__nv_bfloat16));
      src = x + k;
      src_bytes = remain >= 16 ? 16 : 0;
    }
    cp_async_ca_16(out, src, src_bytes);
  }
}

__device__ void fill_x_tail(__nv_bfloat16 *xs, const __nv_bfloat16 *x, int k0,
                            int K) {
  const int tid = static_cast<int>(threadIdx.x);
  const int valid = K > k0 ? min(kBK, K - k0) : 0;
  const int n16 = (valid / 8) * 8;
  const int tail = valid - n16;
  if (tid < tail) {
    xs[n16 + tid] = x[k0 + n16 + tid];
  }
}

__device__ uint32_t dequant_pair(const uint8_t *row_pk, int k0, int k, int K,
                                float s) {
  if (k >= K) {
    return 0u;
  }
  const uint8_t byte = row_pk[(k - k0) >> 1];
  const float w0 = nf4_lut(byte & 0xFu) * s;
  const float w1 = (k + 1 < K) ? nf4_lut(byte >> 4) * s : 0.f;
  return pack_bf16x2(w0, w1);
}

__device__ void compute_tile(
    const uint8_t *pk, const __nv_bfloat16 *xs, const uint16_t *scale,
    int m0, int M, int K, int k0, int K_pad, int n_groups, float &d0, float &d1,
    float &d2, float &d3) {
  (void)K_pad;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row0 = m0 + (warp << 4) + g;
  const int row1 = row0 + 8;
  const int local0 = (warp << 4) + g;
  const int local1 = local0 + 8;

#pragma unroll 1
  for (int ki = 0; ki < kBK / 16; ++ki) {
    const int k_tile = k0 + ki * 16;
    if (k_tile >= K) {
      break;
    }
    const int k_lo = k_tile + (t << 1);
    const int k_hi = k_lo + 8;

    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (row0 < M) {
      const int grp = k_tile / kGroup;
      const uint16_t s16 = __ldg(scale + static_cast<size_t>(row0) * n_groups +
                                 grp);
      const float s = __half2float(__ushort_as_half(s16));
      const uint8_t *row_pk = pk + local0 * (kBK / 2);
      a0 = dequant_pair(row_pk, k0, k_lo, K, s);
      a2 = dequant_pair(row_pk, k0, k_hi, K, s);
    }
    if (row1 < M) {
      const int grp = k_tile / kGroup;
      const uint16_t s16 = __ldg(scale + static_cast<size_t>(row1) * n_groups +
                                 grp);
      const float s = __half2float(__ushort_as_half(s16));
      const uint8_t *row_pk = pk + local1 * (kBK / 2);
      a1 = dequant_pair(row_pk, k0, k_lo, K, s);
      a3 = dequant_pair(row_pk, k0, k_hi, K, s);
    }

    uint32_t b0 = 0, b1 = 0;
    if (g == 0) {
      // n = groupID; only column 0 is live for decode N=1.
      const __nv_bfloat16 z = __float2bfloat16(0.f);
      const int off0 = k_lo - k0;
      const int off1 = k_hi - k0;
      const __nv_bfloat16 x0 = (k_lo < K) ? xs[off0] : z;
      const __nv_bfloat16 x1 = (k_lo + 1 < K) ? xs[off0 + 1] : z;
      const __nv_bfloat16 x8 = (k_hi < K) ? xs[off1] : z;
      const __nv_bfloat16 x9 = (k_hi + 1 < K) ? xs[off1 + 1] : z;
      b0 = pack_bf16x2_bits(x0, x1);
      b1 = pack_bf16x2_bits(x8, x9);
    }

    mma_m16n8k16(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
  }
}

__global__ void __launch_bounds__(256, 2)
    chr_nf4_gemm_decode_n1(const uint8_t *__restrict__ packed,
                           const uint16_t *__restrict__ scale,
                           const __nv_bfloat16 *__restrict__ x,
                           __nv_bfloat16 *__restrict__ y, int M, int K,
                           int K_pad) {
  const int m0 = static_cast<int>(blockIdx.x) * kBM;
  const int n_groups = K_pad / kGroup;
  const int packed_stride = K_pad / 2;
  const int n_tiles = (K_pad + kBK - 1) / kBK;

  extern __shared__ char smem[];
  uint8_t *pk_base = reinterpret_cast<uint8_t *>(smem);
  __nv_bfloat16 *x_base = reinterpret_cast<__nv_bfloat16 *>(
      pk_base + kStages * kPackedStageBytes);

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < n_tiles) {
      const int k0 = s * kBK;
      issue_packed(pk_base + s * kPackedStageBytes, packed, m0, M, k0, K_pad,
                   packed_stride);
      issue_x(x_base + s * kXStageElems, x, k0, K);
    }
    cp_async_commit();
  }

  int smem_write = kStages - 1;
  int smem_read = 0;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

  for (int tile = 0; tile < n_tiles; ++tile) {
    if (tile + kStages - 1 < n_tiles) {
      const int k0 = (tile + kStages - 1) * kBK;
      issue_packed(pk_base + smem_write * kPackedStageBytes, packed, m0, M, k0,
                   K_pad, packed_stride);
      issue_x(x_base + smem_write * kXStageElems, x, k0, K);
    }
    cp_async_commit();
    cp_async_wait<kStages - 2>();
    __syncthreads();

    fill_x_tail(x_base + smem_read * kXStageElems, x, tile * kBK, K);
    __syncthreads();

    compute_tile(pk_base + smem_read * kPackedStageBytes,
                 x_base + smem_read * kXStageElems, scale, m0, M, K,
                 tile * kBK, K_pad, n_groups, d0, d1, d2, d3);

    __syncthreads();
    smem_write = (smem_write + 1) % kStages;
    smem_read = (smem_read + 1) % kStages;
  }
  cp_async_wait<0>();

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if ((lane & 3) == 0) {
    const int r0 = m0 + (warp << 4) + (lane >> 2);
    const int r1 = r0 + 8;
    if (r0 < M) {
      y[r0] = __float2bfloat16(d0);
    }
    if (r1 < M) {
      y[r1] = __float2bfloat16(d2);
    }
  }
}

// --- prefill N=2..16: BM=64, BN=16, BK=128, 4 warps along M x 2 along N -----

__device__ void issue_packed_prefill(uint8_t *dst, const uint8_t *packed,
                                     int m0, int M, int k0, int K_pad,
                                     int packed_stride) {
  constexpr int kChunksPerRow = pBK / 32; // 4 * 16B = 64 packed bytes / row
  constexpr int kNChunks = pBM * kChunksPerRow;
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kNChunks; i += pBlock) {
    const int row = i / kChunksPerRow;
    const int chunk = i % kChunksPerRow;
    const int gm = m0 + row;
    const int col = (k0 >> 1) + chunk * 16;
    uint8_t *out = dst + row * (pBK / 2) + chunk * 16;
    int src_bytes = 0;
    const uint8_t *src = packed;
    if (gm < M && col < packed_stride) {
      src = packed + static_cast<size_t>(gm) * static_cast<size_t>(packed_stride) +
            static_cast<size_t>(col);
      const int remain = packed_stride - col;
      src_bytes = remain >= 16 ? 16 : 0;
    }
    cp_async_cg_16(out, src, src_bytes);
  }
}

// x is [K, 16] row-major: each K-row is 32 B, two 16 B cp.async per row.
__device__ void issue_x_prefill_n16(__nv_bfloat16 *dst, const __nv_bfloat16 *x,
                                     int k0, int K) {
  const int tid = static_cast<int>(threadIdx.x);
  const int row = tid >> 1; // 0..127
  const int half = tid & 1;
  __nv_bfloat16 *out = dst + row * pBN + half * 8;
  int src_bytes = 0;
  const __nv_bfloat16 *src = x;
  const int k = k0 + row;
  if (k < K) {
    src = x + static_cast<size_t>(k) * pBN + half * 8;
    src_bytes = 16;
  }
  cp_async_ca_16(out, src, src_bytes);
}

// Scalar copy for N=2..15: global rows are N*2 bytes, not a 16 B granule.
// Writes every smem element so pad columns and K tails are exact zeros.
__device__ void fill_x_prefill(__nv_bfloat16 *dst, const __nv_bfloat16 *x,
                                int k0, int K, int N) {
  const int tid = static_cast<int>(threadIdx.x);
  constexpr int kElems = pBK * pBN;
  const __nv_bfloat16 z = __float2bfloat16(0.f);
  for (int i = tid; i < kElems; i += pBlock) {
    const int local_k = i / pBN;
    const int n = i - local_k * pBN;
    const int k = k0 + local_k;
    __nv_bfloat16 v = z;
    if (n < N && k < K) {
      v = x[static_cast<size_t>(k) * static_cast<size_t>(N) + n];
    }
    dst[i] = v;
  }
}

__device__ void compute_tile_prefill(const uint8_t *pk, const __nv_bfloat16 *xs,
                                     const uint16_t *scale, int m0, int M,
                                     int K, int k0, int n_groups, float &d0,
                                     float &d1, float &d2, float &d3) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int wm = warp >> 1; // 0..3 along M
  const int wn = warp & 1;  // 0..1 along N
  const int row0 = m0 + (wm << 4) + g;
  const int row1 = row0 + 8;
  const int local0 = (wm << 4) + g;
  const int local1 = local0 + 8;
  const int n = (wn << 3) + g;

#pragma unroll 1
  for (int ki = 0; ki < pBK / 16; ++ki) {
    const int k_tile = k0 + ki * 16;
    if (k_tile >= K) {
      break;
    }
    const int k_lo = k_tile + (t << 1);
    const int k_hi = k_lo + 8;

    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (row0 < M) {
      const int grp = k_tile / kGroup;
      const uint16_t s16 =
          __ldg(scale + static_cast<size_t>(row0) * n_groups + grp);
      const float s = __half2float(__ushort_as_half(s16));
      const uint8_t *row_pk = pk + local0 * (pBK / 2);
      a0 = dequant_pair(row_pk, k0, k_lo, K, s);
      a2 = dequant_pair(row_pk, k0, k_hi, K, s);
    }
    if (row1 < M) {
      const int grp = k_tile / kGroup;
      const uint16_t s16 =
          __ldg(scale + static_cast<size_t>(row1) * n_groups + grp);
      const float s = __half2float(__ushort_as_half(s16));
      const uint8_t *row_pk = pk + local1 * (pBK / 2);
      a1 = dequant_pair(row_pk, k0, k_lo, K, s);
      a3 = dequant_pair(row_pk, k0, k_hi, K, s);
    }

    uint32_t b0 = 0, b1 = 0;
    {
      const __nv_bfloat16 z = __float2bfloat16(0.f);
      const __nv_bfloat16 x0 =
          (k_lo < K) ? xs[(k_lo - k0) * pBN + n] : z;
      const __nv_bfloat16 x1 =
          (k_lo + 1 < K) ? xs[(k_lo + 1 - k0) * pBN + n] : z;
      const __nv_bfloat16 x8 =
          (k_hi < K) ? xs[(k_hi - k0) * pBN + n] : z;
      const __nv_bfloat16 x9 =
          (k_hi + 1 < K) ? xs[(k_hi + 1 - k0) * pBN + n] : z;
      b0 = pack_bf16x2_bits(x0, x1);
      b1 = pack_bf16x2_bits(x8, x9);
    }

    mma_m16n8k16(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
  }
}

__global__ void __launch_bounds__(256, 2)
    chr_nf4_gemm_prefill_n16(const uint8_t *__restrict__ packed,
                              const uint16_t *__restrict__ scale,
                              const __nv_bfloat16 *__restrict__ x,
                              __nv_bfloat16 *__restrict__ y, int M, int K,
                              int K_pad, int N) {
  const int m0 = static_cast<int>(blockIdx.x) * pBM;
  const int n_groups = K_pad / kGroup;
  const int packed_stride = K_pad / 2;
  const int n_tiles = (K_pad + pBK - 1) / pBK;
  const bool x_async = (N == 16);

  extern __shared__ char smem[];
  uint8_t *pk_base = reinterpret_cast<uint8_t *>(smem);
  __nv_bfloat16 *x_base = reinterpret_cast<__nv_bfloat16 *>(
      pk_base + pStages * pPackedStageBytes);

#pragma unroll
  for (int s = 0; s < pStages - 1; ++s) {
    if (s < n_tiles) {
      const int k0 = s * pBK;
      issue_packed_prefill(pk_base + s * pPackedStageBytes, packed, m0, M, k0,
                           K_pad, packed_stride);
      if (x_async) {
        issue_x_prefill_n16(x_base + s * pXStageElems, x, k0, K);
      }
    }
    cp_async_commit();
  }

  int smem_write = pStages - 1;
  int smem_read = 0;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

  for (int tile = 0; tile < n_tiles; ++tile) {
    if (tile + pStages - 1 < n_tiles) {
      const int k0 = (tile + pStages - 1) * pBK;
      issue_packed_prefill(pk_base + smem_write * pPackedStageBytes, packed, m0,
                           M, k0, K_pad, packed_stride);
      if (x_async) {
        issue_x_prefill_n16(x_base + smem_write * pXStageElems, x, k0, K);
      }
    }
    cp_async_commit();
    cp_async_wait<pStages - 2>();
    __syncthreads();

    if (!x_async) {
      fill_x_prefill(x_base + smem_read * pXStageElems, x, tile * pBK, K, N);
      __syncthreads();
    }

    compute_tile_prefill(pk_base + smem_read * pPackedStageBytes,
                          x_base + smem_read * pXStageElems, scale, m0, M, K,
                          tile * pBK, n_groups, d0, d1, d2, d3);

    __syncthreads();
    smem_write = (smem_write + 1) % pStages;
    smem_read = (smem_read + 1) % pStages;
  }
  cp_async_wait<0>();

  // MMA D: d0=(g, 2t), d1=(g, 2t+1), d2=(g+8, 2t), d3=(g+8, 2t+1).
  // Warp covers n in [8*wn, 8*wn+7]; mask n >= N so pad does not store.
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int wm = warp >> 1;
  const int wn = warp & 1;
  const int r0 = m0 + (wm << 4) + g;
  const int r1 = r0 + 8;
  const int n0 = (wn << 3) + (t << 1);
  const int n1 = n0 + 1;
  if (r0 < M) {
    if (n0 < N) {
      y[static_cast<size_t>(r0) * N + n0] = __float2bfloat16(d0);
    }
    if (n1 < N) {
      y[static_cast<size_t>(r0) * N + n1] = __float2bfloat16(d1);
    }
  }
  if (r1 < M) {
    if (n0 < N) {
      y[static_cast<size_t>(r1) * N + n0] = __float2bfloat16(d2);
    }
    if (n1 < N) {
      y[static_cast<size_t>(r1) * N + n1] = __float2bfloat16(d3);
    }
  }
}

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

int k_pad_from_k(int K) { return kGroup * ((K + kGroup - 1) / kGroup); }

} // namespace

extern "C" int chr_nf4_gemm(const chr_nf4_dev_t *w, const void *x, void *y,
                            int32_t N, void *stream) {
  if (!w || !x || !y) {
    return -1;
  }
  // N in [1, 16]. N=1 keeps the decode tile. N=2..16 is prefill (pad to 16).
  // N>16: host chunks into slices of at most 16; this entry does not slice.
  if (N < 1 || N > 16) {
    return -2;
  }
  const chr_nf4_dev_t h = *w;
  if (h.M < 1 || h.K < 1 || !h.packed || !h.scale) {
    return -3;
  }
  if (h.K_pad != k_pad_from_k(h.K) || h.K_pad < h.K) {
    return -4;
  }
  if (!aligned16(h.packed) || !aligned16(x) ||
      (reinterpret_cast<uintptr_t>(h.scale) & 1u) != 0u) {
    return -5;
  }

  static_assert(kBN == 8, "decode N=1 pads BN to 8 for m16n8");
  const dim3 block(kBlock);
  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  const __nv_bfloat16 *x_bf = reinterpret_cast<const __nv_bfloat16 *>(x);
  __nv_bfloat16 *y_bf = reinterpret_cast<__nv_bfloat16 *>(y);

  if (N == 1) {
    const dim3 grid(static_cast<unsigned>((h.M + kBM - 1) / kBM));
    cudaError_t attr = cudaFuncSetAttribute(
        chr_nf4_gemm_decode_n1, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemBytes);
    if (attr != cudaSuccess) {
      return -6;
    }
    chr_nf4_gemm_decode_n1<<<grid, block, kSmemBytes, s>>>(
        h.packed, h.scale, x_bf, y_bf, h.M, h.K, h.K_pad);
  } else {
    const dim3 grid(static_cast<unsigned>((h.M + pBM - 1) / pBM));
    cudaError_t attr = cudaFuncSetAttribute(
        chr_nf4_gemm_prefill_n16, cudaFuncAttributeMaxDynamicSharedMemorySize,
        pSmemBytes);
    if (attr != cudaSuccess) {
      return -6;
    }
    chr_nf4_gemm_prefill_n16<<<grid, block, pSmemBytes, s>>>(
        h.packed, h.scale, x_bf, y_bf, h.M, h.K, h.K_pad, N);
  }

  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -6;
}
