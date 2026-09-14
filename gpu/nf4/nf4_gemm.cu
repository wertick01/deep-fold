// Fused NF4 dequant + HMMA.16816 GEMM for sm_86.
// y = dequant_nf4(W) @ x. Dequant in registers; no BF16 W in HBM.
//
// Specs: docs/spec/stitch-gpu.md (wins), docs/spec/nf4.md §1, docs/kernel-ampere.md
// Decode N=1, M large:  BM=128, BN=8 pad,  BK=256, block=256, stages=3, split_k=1.
// Decode N=1, M small:  BM=64,  BN=8 pad,  BK=128, block=128, stages=3, split_k>1.
// Prefill N=2..8:       BM=64,  BN=8 pad,  BK=128, block=256, stages=3, split_k>=1.
// Prefill N=9..16:      BM=64,  BN=16 pad, BK=128, block=256, stages=3, split_k>=1.
// MMA: only mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// Prefill N=17..32 / 33..64: planned BN=32 / BN=64, same BM/BK as n16.
// Live launch is N in [1, 16]. N>16: chr_nf4_gemm_ws returns -2; host chunks.
// Planners describe N=17..64 (path 3/4) without raising kLiveMaxN: 167 vs
// 52 ms TTFT is the next floor; ncu showed ~5% DRAM (MMA per dequant, not HBM).
//
// Occupancy (docs/tz/wave9-review.md §3, RTX 3080 = 70 SMs). Wave 2 launched
// grid = (ceil(M / BM), 1, 1) only, so a 3B decode step ran 16 CTAs on q/o_proj
// and 2 on the GQA k/v_proj: a 3.77x cut in weight traffic could not show up
// because the card was idle. grid.y is now split_k over the K tiles, and small
// M also drops to a 64-row tile. split_k > 1 writes FP32 partials into a
// caller-owned workspace and a second kernel reduces them into BF16 y; the
// splits read disjoint K ranges, so packed bytes are still read exactly once
// and nothing dense ever lands in HBM.

#include "chr_gpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

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

// Small decode tile (wave 2 occupancy work, docs/tz/wave9-review.md §3).
// Half the rows per CTA and half the K per stage, 4 warps, and grid.y = split_k
// over the K tiles. 13056 B of smem is under the 48 KiB static limit, so this
// launch needs no cudaFuncSetAttribute -- one less one-time call inside a CUDA
// graph capture than the BM=128 tile.
constexpr int sBM = 64;
constexpr int sBK = 128;
constexpr int sBlock = 128;
constexpr int sStages = 3;
constexpr int sPackedStageBytes = sBM * (sBK / 2); // 4096
constexpr int sXStageElems = sBK;                   // 128 bf16
constexpr int sSmemBytes =
    sStages * sPackedStageBytes + sStages * sXStageElems * 2; // 13056

static_assert(sBM % 16 == 0, "one m16n8k16 A fragment per warp needs 16 rows");
static_assert(sBM / 16 == sBlock / 32, "4 warps cover sBM rows, 16 each");
static_assert(sBK % kGroup == 0, "a stage holds whole 64-wide scale groups");
static_assert(sSmemBytes == 13056, "small decode ring");
static_assert(sSmemBytes <= 48 * 1024, "small tile must not need opt-in smem");

// GA102-200 (RTX 3080 12 GB) has 70 SMs. This is the unit the split-K target is
// expressed in, not a tile switch: the planner raises split_k until grid.x *
// split_k reaches 2 CTAs per SM, then stops because a split with no K tile to
// walk only writes zeros.
constexpr int kDefaultOneWave = 70;
constexpr int kTargetCtas = 140;
// Live chr_nf4_gemm_ws / TokenLoop ceiling. Planner describes up to
// kPlanMaxN. n64 stays behind this cap. Dispatch is by tile width
// (N<=8/16/32/64), not by kLiveMaxN.
constexpr int kLiveMaxN = 32;
constexpr int kPlanMaxN = 64;

// Prefill tile (kernel-ampere.md §2 / §4). Scales stay in __ldg, not smem.
// N<=8 uses BN=8 so a tail chunk does not pad a second m16n8; N=9..16 keep
// BN=16. x is cp.async'd as a compact [BK, N] blob (16 B aligned at k0*N) and
// expanded to [BK, BN] after the wait -- per-row cp.async cannot do odd N
// (2-byte row starts) and is misaligned for many even N as well.
constexpr int pBM = 64;
constexpr int pBN8 = 8;
constexpr int pBN16 = 16;
constexpr int pBK = 128;
constexpr int pBlock = 256;
constexpr int pStages = 3;
constexpr int pPackedStageBytes = pBM * (pBK / 2); // 4096
constexpr int pXStageElems8 = pBK * pBN8;           // 1024 bf16, layout [BK, 8]
constexpr int pXStageElems16 = pBK * pBN16;         // 2048 bf16, layout [BK, 16]
constexpr int pSmemBytes8 =
    pStages * pPackedStageBytes + pStages * pXStageElems8 * 2; // 18432
constexpr int pSmemBytes16 =
    pStages * pPackedStageBytes + pStages * pXStageElems16 * 2; // 24576

// Prefill ring: BN=8 for N=2..8, BN=16 for N=9..16. Both stay on path=2.

// How the 8 warps are cut. Wave 2 cut them 4 along M x 2 along N, which made
// the two N warps of a row block reconstruct the *same* 16x16 W fragment: a
// prefill column cost twice the dequant of a decode column for identical
// weight traffic. They are now cut 4 along M x 2 along K, and each warp issues
// both m16n8 MMAs (n < 8 and n >= 8) from one A fragment. BM, BK, and block
// stay put; smem follows BN (8 vs 16) -- see compute_tile_prefill.
constexpr int pWarpsM = 4;
constexpr int pWarpsK = 2;
//: 8 FP32 accumulators per lane, padded to 9 so the one cross-warp reduce at
//: the end of the CTA lands on 32 distinct banks (gcd(9, 32) == 1).
constexpr int pRedStride = 9;
constexpr int pRedFloats = pWarpsM * 32 * pRedStride; // 1152

static_assert(pPackedStageBytes == 4096, "prefill packed stage");
static_assert(pXStageElems8 == 1024, "prefill x stage BN=8");
static_assert(pXStageElems16 == 2048, "prefill x stage BN=16");
static_assert(pSmemBytes8 == 18432, "prefill ring BN=8");
static_assert(pSmemBytes16 == 24576, "prefill ring BN=16");
static_assert(pSmemBytes16 <= 99 * 1024, "must not blow the sm_86 99 KiB cap");

// N=17..64: same BM/BK/stages as n16 so 3B split-K occupancy (128/64
// CTAs on q/k_proj) is unchanged. Only the x-stage grows. n32 is live
// when kLiveMaxN >= 32. n64 stays plan-only.
constexpr int pBN32 = 32;
constexpr int pBN64 = 64;
constexpr int pXStageElems32 = pBK * pBN32; // 4096 bf16
constexpr int pXStageElems64 = pBK * pBN64; // 8192 bf16
constexpr int pSmemBytes32 =
    pStages * pPackedStageBytes + pStages * pXStageElems32 * 2; // 36864
constexpr int pSmemBytes64 =
    pStages * pPackedStageBytes + pStages * pXStageElems64 * 2; // 61440
static_assert(pSmemBytes32 == 36864, "prefill ring BN=32");
static_assert(pSmemBytes64 == 61440, "prefill ring BN=64");
static_assert(pSmemBytes64 <= 99 * 1024, "BN=64 must not blow the sm_86 99 KiB cap");
// Cross-warp reduce: (BN/8)*4 accums + 1 so gcd(stride, 32)==1. Fits in the ring.
static_assert(pWarpsM * 32 * (32 / 8 * 4 + 1) * 4 <= pSmemBytes32,
              "BN=32 reduce reuses the staging ring");
static_assert(pWarpsM * 32 * (64 / 8 * 4 + 1) * 4 <= pSmemBytes64,
              "BN=64 reduce reuses the staging ring");
static_assert(pBlock == kBlock, "both launches are 256 threads");
static_assert(pWarpsM * pWarpsK == pBlock / 32, "8 warps, 4 along M x 2 along K");
static_assert(pWarpsM * 16 == pBM, "each M warp owns one m16n8k16 A fragment");
static_assert(pWarpsK * kGroup == pBK, "each K warp owns one 64-wide scale group");
static_assert(static_cast<int>(pRedFloats * sizeof(float)) <= pSmemBytes8,
              "the cross-warp reduce reuses the staging ring");

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

// Templated on the tile so the decode tiles (BM=128/BK=256 and BM=64/BK=128)
// share one staging body; the generated code is what each tile had inline.
template <int BM, int BK, int BLOCK>
__device__ void issue_packed_t(uint8_t *dst, const uint8_t *packed, int m0,
                               int M, int k0, int packed_stride) {
  constexpr int kChunksPerRow = BK / 32; // 16B chunks of packed nibbles per row
  constexpr int kNChunks = BM * kChunksPerRow;
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kNChunks; i += BLOCK) {
    const int row = i / kChunksPerRow;
    const int chunk = i % kChunksPerRow;
    const int gm = m0 + row;
    const int col = (k0 >> 1) + chunk * 16;
    uint8_t *out = dst + row * (BK / 2) + chunk * 16;
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

template <int BK, int BLOCK>
__device__ void issue_x_t(__nv_bfloat16 *dst, const __nv_bfloat16 *x, int k0,
                          int K) {
  constexpr int kChunks = BK / 8; // 16B = 8 bf16
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kChunks; i += BLOCK) {
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

template <int BK>
__device__ void fill_x_tail_t(__nv_bfloat16 *xs, const __nv_bfloat16 *x, int k0,
                              int K) {
  const int tid = static_cast<int>(threadIdx.x);
  const int valid = K > k0 ? min(BK, K - k0) : 0;
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

// Same reconstruct with the K-tail branches dropped. Only called when the whole
// 64-wide scale group sits inside K, so both nibbles are live by construction.
__device__ __forceinline__ uint32_t dequant_pair_live(const uint8_t *row_pk,
                                                      int k0, int k, float s) {
  const uint8_t byte = row_pk[(k - k0) >> 1];
  return pack_bf16x2(nf4_lut(byte & 0xFu) * s, nf4_lut(byte >> 4) * s);
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
      issue_packed_t<kBM, kBK, kBlock>(pk_base + s * kPackedStageBytes, packed,
                                       m0, M, k0, packed_stride);
      issue_x_t<kBK, kBlock>(x_base + s * kXStageElems, x, k0, K);
    }
    cp_async_commit();
  }

  int smem_write = kStages - 1;
  int smem_read = 0;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

  for (int tile = 0; tile < n_tiles; ++tile) {
    if (tile + kStages - 1 < n_tiles) {
      const int k0 = (tile + kStages - 1) * kBK;
      issue_packed_t<kBM, kBK, kBlock>(pk_base + smem_write * kPackedStageBytes,
                                       packed, m0, M, k0, packed_stride);
      issue_x_t<kBK, kBlock>(x_base + smem_write * kXStageElems, x, k0, K);
    }
    cp_async_commit();
    cp_async_wait<kStages - 2>();
    __syncthreads();

    fill_x_tail_t<kBK>(x_base + smem_read * kXStageElems, x, tile * kBK, K);
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

// --- decode N=1, small tile + split-K: BM=64, BK=128, 4 warps, grid.y --------
//
// Why a second decode tile. The tile above is correct and its 128 rows amortise
// staging well, but grid.x = ceil(M / 128) is the only parallelism it has: a 3B
// q_proj (M=2048) launches 16 CTAs and a GQA k_proj (M=256) launches 2, on a
// card with 70 SMs. Halving BM doubles the CTAs; splitting the K loop across
// grid.y multiplies them again, at the cost of FP32 partials the host reduces.
// Weight traffic is unchanged -- the splits read disjoint K ranges, so every
// packed byte is still read exactly once per GEMM.
//
// Same contract as the tile above: packed stays packed in HBM, one 64-row x
// BK/2-byte window lands in smem, and W is reconstructed into MMA A registers.

__device__ __forceinline__ void compute_tile_small(
    const uint8_t *pk, const __nv_bfloat16 *xs, const uint16_t *scale, int m0,
    int M, int K, int k0, int n_groups, float &d0, float &d1, float &d2,
    float &d3) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5; // 0..3, 16 rows each -> sBM
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int local0 = (warp << 4) + g;
  const int local1 = local0 + 8;
  const int row0 = m0 + local0;
  const int row1 = m0 + local1;
  const uint8_t *row_pk0 = pk + local0 * (sBK / 2);
  const uint8_t *row_pk1 = pk + local1 * (sBK / 2);
  const bool live0 = row0 < M;
  const bool live1 = row1 < M;
  // B: n = groupID, so only the lane group holding n == 0 is live for N == 1.
  const bool has_b = (g == 0);

  // Outer loop is the 64-wide scale group, not the 16-wide MMA step: the four
  // MMA steps inside a group share one FP16 scale per row, which turns four
  // __ldg per row into one and lets the inner four steps unroll.
#pragma unroll 1
  for (int gi = 0; gi < sBK / kGroup; ++gi) {
    const int k_grp = k0 + gi * kGroup;
    if (k_grp >= K) {
      break;
    }
    const int grp = k_grp / kGroup;
    float s0 = 0.f, s1 = 0.f;
    if (live0) {
      s0 = __half2float(__ushort_as_half(
          __ldg(scale + static_cast<size_t>(row0) * n_groups + grp)));
    }
    if (live1) {
      s1 = __half2float(__ushort_as_half(
          __ldg(scale + static_cast<size_t>(row1) * n_groups + grp)));
    }
    const bool full = (k_grp + kGroup <= K);

    if (full) {
#pragma unroll
      for (int ki = 0; ki < kGroup / 16; ++ki) {
        const int k_lo = k_grp + ki * 16 + (t << 1);
        const int k_hi = k_lo + 8;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (live0) {
          a0 = dequant_pair_live(row_pk0, k0, k_lo, s0);
          a2 = dequant_pair_live(row_pk0, k0, k_hi, s0);
        }
        if (live1) {
          a1 = dequant_pair_live(row_pk1, k0, k_lo, s1);
          a3 = dequant_pair_live(row_pk1, k0, k_hi, s1);
        }
        uint32_t b0 = 0, b1 = 0;
        if (has_b) {
          const int off0 = k_lo - k0;
          const int off1 = k_hi - k0;
          b0 = pack_bf16x2_bits(xs[off0], xs[off0 + 1]);
          b1 = pack_bf16x2_bits(xs[off1], xs[off1 + 1]);
        }
        mma_m16n8k16(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
      }
      continue;
    }

    // Tail group: K ends inside it, so every nibble and every x element is
    // bounds-checked exactly like the BM=128 tile does.
#pragma unroll 1
    for (int ki = 0; ki < kGroup / 16; ++ki) {
      const int k_tile = k_grp + ki * 16;
      if (k_tile >= K) {
        break;
      }
      const int k_lo = k_tile + (t << 1);
      const int k_hi = k_lo + 8;
      uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      if (live0) {
        a0 = dequant_pair(row_pk0, k0, k_lo, K, s0);
        a2 = dequant_pair(row_pk0, k0, k_hi, K, s0);
      }
      if (live1) {
        a1 = dequant_pair(row_pk1, k0, k_lo, K, s1);
        a3 = dequant_pair(row_pk1, k0, k_hi, K, s1);
      }
      uint32_t b0 = 0, b1 = 0;
      if (has_b) {
        const __nv_bfloat16 z = __float2bfloat16(0.f);
        const int off0 = k_lo - k0;
        const int off1 = k_hi - k0;
        b0 = pack_bf16x2_bits((k_lo < K) ? xs[off0] : z,
                              (k_lo + 1 < K) ? xs[off0 + 1] : z);
        b1 = pack_bf16x2_bits((k_hi < K) ? xs[off1] : z,
                              (k_hi + 1 < K) ? xs[off1 + 1] : z);
      }
      mma_m16n8k16(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
  }
}

// partial == nullptr means split_k == 1: accumulate straight into y and skip
// the reduce launch entirely.
__global__ void __launch_bounds__(128, 4)
    chr_nf4_gemm_decode_small(const uint8_t *__restrict__ packed,
                              const uint16_t *__restrict__ scale,
                              const __nv_bfloat16 *__restrict__ x,
                              __nv_bfloat16 *__restrict__ y,
                              float *__restrict__ partial, int M, int K,
                              int K_pad, int n_ktiles, int tiles_per_split) {
  const int m0 = static_cast<int>(blockIdx.x) * sBM;
  const int split = static_cast<int>(blockIdx.y);
  const int tile0 = split * tiles_per_split;
  // A split past the end contributes an exact zero rather than returning, so a
  // grid.y the planner over-estimated can never leave a partial uninitialised.
  const int n_local = min(tiles_per_split, n_ktiles - tile0);
  const int n_groups = K_pad / kGroup;
  const int packed_stride = K_pad / 2;

  extern __shared__ char smem[];
  uint8_t *pk_base = reinterpret_cast<uint8_t *>(smem);
  __nv_bfloat16 *x_base = reinterpret_cast<__nv_bfloat16 *>(
      pk_base + sStages * sPackedStageBytes);

#pragma unroll
  for (int s = 0; s < sStages - 1; ++s) {
    if (s < n_local) {
      const int k0 = (tile0 + s) * sBK;
      issue_packed_t<sBM, sBK, sBlock>(pk_base + s * sPackedStageBytes, packed,
                                       m0, M, k0, packed_stride);
      issue_x_t<sBK, sBlock>(x_base + s * sXStageElems, x, k0, K);
    }
    cp_async_commit();
  }

  int smem_write = sStages - 1;
  int smem_read = 0;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

  for (int tile = 0; tile < n_local; ++tile) {
    if (tile + sStages - 1 < n_local) {
      const int k0 = (tile0 + tile + sStages - 1) * sBK;
      issue_packed_t<sBM, sBK, sBlock>(pk_base + smem_write * sPackedStageBytes,
                                       packed, m0, M, k0, packed_stride);
      issue_x_t<sBK, sBlock>(x_base + smem_write * sXStageElems, x, k0, K);
    }
    cp_async_commit();
    cp_async_wait<sStages - 2>();
    __syncthreads();

    const int k0 = (tile0 + tile) * sBK;
    fill_x_tail_t<sBK>(x_base + smem_read * sXStageElems, x, k0, K);
    __syncthreads();

    compute_tile_small(pk_base + smem_read * sPackedStageBytes,
                       x_base + smem_read * sXStageElems, scale, m0, M, K, k0,
                       n_groups, d0, d1, d2, d3);

    __syncthreads();
    smem_write = (smem_write + 1) % sStages;
    smem_read = (smem_read + 1) % sStages;
  }
  cp_async_wait<0>();
  (void)d1;
  (void)d3;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if ((lane & 3) == 0) {
    const int r0 = m0 + (warp << 4) + (lane >> 2);
    const int r1 = r0 + 8;
    if (partial != nullptr) {
      float *p = partial + static_cast<size_t>(split) * M;
      if (r0 < M) {
        p[r0] = d0;
      }
      if (r1 < M) {
        p[r1] = d2;
      }
    } else {
      if (r0 < M) {
        y[r0] = __float2bfloat16(d0);
      }
      if (r1 < M) {
        y[r1] = __float2bfloat16(d2);
      }
    }
  }
}

// FP32 partials -> BF16 y. Partials are [split, M, N] and y is [M, N], both
// row-major, so one linear index covers decode and prefill. Two launches
// instead of a BF16 atomicAdd: atomics would round every split into y and make
// the result depend on CTA order, which the oracle in gpu/nf4/verify.py checks.
__global__ void chr_nf4_reduce_splitk(const float *__restrict__ partial,
                                      __nv_bfloat16 *__restrict__ y, int rows,
                                      int split) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
  if (i >= rows) {
    return;
  }
  float acc = 0.f;
  for (int s = 0; s < split; ++s) {
    acc += partial[static_cast<size_t>(s) * rows + i];
  }
  y[i] = __float2bfloat16(acc);
}

// --- prefill N=2..16: BM=64, BN=8 or 16, BK=128, 4 along M x 2 along K -----

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

// Compact [min(BK, K-k0), N] into the x stage via 16 B cp.async. k0 is a
// multiple of BK, so x + k0*N is 16-byte aligned whenever x is, and the copy
// does not depend on the per-row alignment of a [K, N] layout (N=3 rows
// start 6 B apart). srcSize 0/4/8/16 zero-fills the rest of each 16 B dest;
// leftover 2 B (odd n_k and odd N) are patched after the wait.
template <int BN>
__device__ void issue_x_prefill(__nv_bfloat16 *dst, const __nv_bfloat16 *x,
                                 int k0, int K, int N) {
  static_assert(BN == 8 || BN == 16, "prefill BN is one or two m16n8");
  constexpr int kStageBytes = pBK * BN * 2;
  const int tid = static_cast<int>(threadIdx.x);
  const int n_k = K > k0 ? min(pBK, K - k0) : 0;
  const int n_bytes = n_k * N * 2;
  const int aligned = n_bytes & ~15;
  const int rem = n_bytes - aligned;
  const char *gbase = reinterpret_cast<const char *>(
      x + static_cast<size_t>(k0) * static_cast<size_t>(N));
  char *sbase = reinterpret_cast<char *>(dst);
  const int off = tid * 16;
  if (off >= kStageBytes) {
    return;
  }
  int src_bytes = 0;
  const void *src = x;
  if (off + 16 <= n_bytes) {
    src = gbase + off;
    src_bytes = 16;
  } else if (off == aligned && rem > 0) {
    src = gbase + off;
    src_bytes = rem >= 8 ? 8 : rem >= 4 ? 4 : 0;
  } else if (off >= aligned + (rem > 0 ? 16 : 0)) {
    src_bytes = 0;
  }
  cp_async_ca_16(sbase + off, src, src_bytes);
}

template <int BN>
__device__ void patch_x_prefill_tail(__nv_bfloat16 *dst, const __nv_bfloat16 *x,
                                     int k0, int K, int N) {
  const int n_k = K > k0 ? min(pBK, K - k0) : 0;
  const int n_bytes = n_k * N * 2;
  const int rem = n_bytes & 15;
  if (rem == 0) {
    return;
  }
  const int src_bytes = rem >= 8 ? 8 : rem >= 4 ? 4 : 0;
  const int n_left = (rem - src_bytes) / 2;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < n_left) {
    const int elem = (n_bytes / 16) * 8 + src_bytes / 2 + tid;
    dst[elem] = x[static_cast<size_t>(k0) * static_cast<size_t>(N) +
                  static_cast<size_t>(elem)];
  }
}

// Compact [BK, N] in the first pBK*N slots -> padded [BK, BN] the MMA walks.
template <int BN>
__device__ void expand_x_prefill(__nv_bfloat16 *xs, int N) {
  constexpr int kRegs = (pBK * BN) / pBlock;
  __nv_bfloat16 r[kRegs];
  const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int i = 0; i < kRegs; ++i) {
    r[i] = xs[tid + i * pBlock];
  }
  __syncthreads();
  const __nv_bfloat16 z = __float2bfloat16(0.f);
#pragma unroll
  for (int i = 0; i < kRegs; ++i) {
    xs[tid + i * pBlock] = z;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kRegs; ++i) {
    const int src = tid + i * pBlock;
    if (src < pBK * N) {
      const int row = src / N;
      const int col = src - row * N;
      xs[row * BN + col] = r[i];
    }
  }
}

template <int BN>
__device__ void prepare_x_prefill(__nv_bfloat16 *xs, const __nv_bfloat16 *x,
                                   int k0, int K, int N) {
  if (N == BN) {
    return;
  }
  const int n_k = K > k0 ? min(pBK, K - k0) : 0;
  if ((n_k * N * 2) & 15) {
    patch_x_prefill_tail<BN>(xs, x, k0, K, N);
    __syncthreads();
  }
  expand_x_prefill<BN>(xs, N);
  __syncthreads();
}

// One warp = 16 rows x one 64-wide scale group x both m16n8 columns blocks.
//
// Three things this does that the wave-2 body did not, in the order they cost:
//
//  1. ``A`` is reconstructed once and fed to both MMAs. Under the old 4xM-by-2xN
//     warp cut, the ``wn=0`` and ``wn=1`` warps of a row block dequantized the
//     same nibbles, so N=16 burned twice the reconstruct of N=1 while reading
//     the same bytes.
//  2. The k loop is ``#pragma unroll``ed inside a 64-wide scale group instead
//     of ``#pragma unroll 1`` over all 8 steps of the K tile. The MMAs chain on
//     the accumulator, so with no unroll a warp had exactly one dequant->MMA
//     dependency in flight and ~10 warps per SM could not cover it. This is the
//     shape ``compute_tile_small`` already had; prefill never got it.
//  3. One ``__ldg`` per row per group, not one per row per 16-wide step: the
//     four steps inside a group share a scale by construction.
//
// ``WIDE`` is the BN=16 tile (N=9..16). BN=8 never forms the n >= 8 fragment.
template <int BN, bool WIDE>
__device__ __forceinline__ void compute_tile_prefill(
    const uint8_t *pk, const __nv_bfloat16 *xs, const uint16_t *scale, int m0,
    int M, int K, int k0, int n_groups, float &d0, float &d1, float &d2,
    float &d3, float &e0, float &e1, float &e2, float &e3) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp & (pWarpsM - 1); // 0..3 along M
  const int wk = warp >> 2;            // 0..1 along K
  const int g = lane >> 2;
  const int t = lane & 3;
  const int local0 = (wm << 4) + g;
  const int local1 = local0 + 8;
  const int row0 = m0 + local0;
  const int row1 = m0 + local1;
  const bool live0 = row0 < M;
  const bool live1 = row1 < M;
  const uint8_t *row_pk0 = pk + local0 * (pBK / 2);
  const uint8_t *row_pk1 = pk + local1 * (pBK / 2);
  // B: n = groupID, so this lane holds column g of the low m16n8 and column
  // g + 8 of the high one. Both walk k with stride BN through the x stage.
  static_assert(BN == 8 || BN == 16, "prefill BN is one or two m16n8");
  static_assert(!WIDE || BN == 16, "high m16n8 only exists on the BN=16 tile");
  const __nv_bfloat16 *x_lo = xs + g;
  const __nv_bfloat16 *x_hi = xs + g + 8;

  // A split whose group starts past K contributes an exact zero: the reduce
  // below still adds it, so the partner warp's sum stays the whole K tile.
  const int k_grp = k0 + wk * kGroup;
  if (k_grp >= K) {
    return;
  }

  const int grp = k_grp / kGroup;
  float s0 = 0.f, s1 = 0.f;
  if (live0) {
    s0 = __half2float(__ushort_as_half(
        __ldg(scale + static_cast<size_t>(row0) * n_groups + grp)));
  }
  if (live1) {
    s1 = __half2float(__ushort_as_half(
        __ldg(scale + static_cast<size_t>(row1) * n_groups + grp)));
  }

  if (k_grp + kGroup <= K) {
#pragma unroll
    for (int ki = 0; ki < kGroup / 16; ++ki) {
      const int k_lo = k_grp + ki * 16 + (t << 1);
      const int k_hi = k_lo + 8;
      uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      if (live0) {
        a0 = dequant_pair_live(row_pk0, k0, k_lo, s0);
        a2 = dequant_pair_live(row_pk0, k0, k_hi, s0);
      }
      if (live1) {
        a1 = dequant_pair_live(row_pk1, k0, k_lo, s1);
        a3 = dequant_pair_live(row_pk1, k0, k_hi, s1);
      }
      const int o_lo = (k_lo - k0) * BN;
      const int o_hi = (k_hi - k0) * BN;
      const uint32_t b0 = pack_bf16x2_bits(x_lo[o_lo], x_lo[o_lo + BN]);
      const uint32_t b1 = pack_bf16x2_bits(x_lo[o_hi], x_lo[o_hi + BN]);
      mma_m16n8k16(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
      if (WIDE) {
        const uint32_t c0 = pack_bf16x2_bits(x_hi[o_lo], x_hi[o_lo + BN]);
        const uint32_t c1 = pack_bf16x2_bits(x_hi[o_hi], x_hi[o_hi + BN]);
        mma_m16n8k16(a0, a1, a2, a3, c0, c1, e0, e1, e2, e3);
      }
    }
    return;
  }

  // Tail group: K ends inside it, so every nibble and every x element is
  // bounds-checked exactly like the decode tiles do.
#pragma unroll 1
  for (int ki = 0; ki < kGroup / 16; ++ki) {
    const int k_tile = k_grp + ki * 16;
    if (k_tile >= K) {
      break;
    }
    const int k_lo = k_tile + (t << 1);
    const int k_hi = k_lo + 8;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (live0) {
      a0 = dequant_pair(row_pk0, k0, k_lo, K, s0);
      a2 = dequant_pair(row_pk0, k0, k_hi, K, s0);
    }
    if (live1) {
      a1 = dequant_pair(row_pk1, k0, k_lo, K, s1);
      a3 = dequant_pair(row_pk1, k0, k_hi, K, s1);
    }
    const __nv_bfloat16 z = __float2bfloat16(0.f);
    const int o_lo = (k_lo - k0) * BN;
    const int o_hi = (k_hi - k0) * BN;
    const bool lo0 = k_lo < K, lo1 = k_lo + 1 < K;
    const bool hi0 = k_hi < K, hi1 = k_hi + 1 < K;
    const uint32_t b0 = pack_bf16x2_bits(lo0 ? x_lo[o_lo] : z,
                                         lo1 ? x_lo[o_lo + BN] : z);
    const uint32_t b1 = pack_bf16x2_bits(hi0 ? x_lo[o_hi] : z,
                                         hi1 ? x_lo[o_hi + BN] : z);
    mma_m16n8k16(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    if (WIDE) {
      const uint32_t c0 = pack_bf16x2_bits(lo0 ? x_hi[o_lo] : z,
                                           lo1 ? x_hi[o_lo + BN] : z);
      const uint32_t c1 = pack_bf16x2_bits(hi0 ? x_hi[o_hi] : z,
                                           hi1 ? x_hi[o_hi + BN] : z);
      mma_m16n8k16(a0, a1, a2, a3, c0, c1, e0, e1, e2, e3);
    }
  }
}

template <int BN>
__device__ void prefill_gemm_body(
    const uint8_t *__restrict__ packed, const uint16_t *__restrict__ scale,
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y,
    float *__restrict__ partial, int M, int K, int K_pad, int N,
    int n_ktiles, int tiles_per_split) {
  static_assert(BN == 8 || BN == 16, "prefill BN is one or two m16n8");
  constexpr int kXStageElems = pBK * BN;
  const int m0 = static_cast<int>(blockIdx.x) * pBM;
  // grid.y is the K split (TTFT is the second occupancy floor: 3B prefill
  // chunks are N<=16, so grid.x = ceil(M / 64) is 32 CTAs on q_proj).
  const int split = static_cast<int>(blockIdx.y);
  const int tile0 = split * tiles_per_split;
  const int n_tiles_local = min(tiles_per_split, n_ktiles - tile0);
  const int n_groups = K_pad / kGroup;
  const int packed_stride = K_pad / 2;

  extern __shared__ char smem[];
  uint8_t *pk_base = reinterpret_cast<uint8_t *>(smem);
  __nv_bfloat16 *x_base = reinterpret_cast<__nv_bfloat16 *>(
      pk_base + pStages * pPackedStageBytes);

#pragma unroll
  for (int s = 0; s < pStages - 1; ++s) {
    if (s < n_tiles_local) {
      const int k0 = (tile0 + s) * pBK;
      issue_packed_prefill(pk_base + s * pPackedStageBytes, packed, m0, M, k0,
                           K_pad, packed_stride);
      issue_x_prefill<BN>(x_base + s * kXStageElems, x, k0, K, N);
    }
    cp_async_commit();
  }

  int smem_write = pStages - 1;
  int smem_read = 0;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;

  for (int tile = 0; tile < n_tiles_local; ++tile) {
    if (tile + pStages - 1 < n_tiles_local) {
      const int k0 = (tile0 + tile + pStages - 1) * pBK;
      issue_packed_prefill(pk_base + smem_write * pPackedStageBytes, packed, m0,
                           M, k0, K_pad, packed_stride);
      issue_x_prefill<BN>(x_base + smem_write * kXStageElems, x, k0, K, N);
    }
    cp_async_commit();
    cp_async_wait<pStages - 2>();
    __syncthreads();

    const int k0 = (tile0 + tile) * pBK;
    prepare_x_prefill<BN>(x_base + smem_read * kXStageElems, x, k0, K, N);

    const uint8_t *pk = pk_base + smem_read * pPackedStageBytes;
    const __nv_bfloat16 *xs = x_base + smem_read * kXStageElems;
    compute_tile_prefill<BN, (BN == 16)>(pk, xs, scale, m0, M, K, k0, n_groups,
                                          d0, d1, d2, d3, e0, e1, e2, e3);

    __syncthreads();
    smem_write = (smem_write + 1) % pStages;
    smem_read = (smem_read + 1) % pStages;
  }
  cp_async_wait<0>();

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp & (pWarpsM - 1);
  const int wk = warp >> 2;

  // The two K warps of a row block hold FP32 halves of the same accumulator.
  // Sum them once per CTA through the staging ring, which is dead after the
  // last tile: this is the price of reconstructing A once instead of twice,
  // and it is paid per CTA, not per K tile. FP32 in, FP32 out -- the store
  // below is still the only rounding, so the oracle in gpu/nf4/verify.py sees
  // the same answer the one-warp-per-column layout produced.
  __syncthreads();
  float *red = reinterpret_cast<float *>(smem);
  float *slot = red + static_cast<size_t>(wm * 32 + lane) * pRedStride;
  if (wk == 1) {
    slot[0] = d0;
    slot[1] = d1;
    slot[2] = d2;
    slot[3] = d3;
    slot[4] = e0;
    slot[5] = e1;
    slot[6] = e2;
    slot[7] = e3;
  }
  __syncthreads();
  if (wk != 0) {
    return;
  }
  d0 += slot[0];
  d1 += slot[1];
  d2 += slot[2];
  d3 += slot[3];
  e0 += slot[4];
  e1 += slot[5];
  e2 += slot[6];
  e3 += slot[7];

  // MMA D: d0=(g, 2t), d1=(g, 2t+1), d2=(g+8, 2t), d3=(g+8, 2t+1). d is the
  // n < 8 block and e the n >= 8 one; mask n >= N so pad does not store.
  const int g = lane >> 2;
  const int t = lane & 3;
  const int r0 = m0 + (wm << 4) + g;
  const int r1 = r0 + 8;
  const int n0 = t << 1;
  const int n1 = n0 + 1;
  if (partial != nullptr) {
    float *p = partial + static_cast<size_t>(split) * M * N;
    if (r0 < M) {
      float *p0 = p + static_cast<size_t>(r0) * N;
      if (n0 < N) {
        p0[n0] = d0;
      }
      if (n1 < N) {
        p0[n1] = d1;
      }
      if (n0 + 8 < N) {
        p0[n0 + 8] = e0;
      }
      if (n1 + 8 < N) {
        p0[n1 + 8] = e1;
      }
    }
    if (r1 < M) {
      float *p1 = p + static_cast<size_t>(r1) * N;
      if (n0 < N) {
        p1[n0] = d2;
      }
      if (n1 < N) {
        p1[n1] = d3;
      }
      if (n0 + 8 < N) {
        p1[n0 + 8] = e2;
      }
      if (n1 + 8 < N) {
        p1[n1 + 8] = e3;
      }
    }
    return;
  }
  if (r0 < M) {
    __nv_bfloat16 *y0 = y + static_cast<size_t>(r0) * N;
    if (n0 < N) {
      y0[n0] = __float2bfloat16(d0);
    }
    if (n1 < N) {
      y0[n1] = __float2bfloat16(d1);
    }
    if (n0 + 8 < N) {
      y0[n0 + 8] = __float2bfloat16(e0);
    }
    if (n1 + 8 < N) {
      y0[n1 + 8] = __float2bfloat16(e1);
    }
  }
  if (r1 < M) {
    __nv_bfloat16 *y1 = y + static_cast<size_t>(r1) * N;
    if (n0 < N) {
      y1[n0] = __float2bfloat16(d2);
    }
    if (n1 < N) {
      y1[n1] = __float2bfloat16(d3);
    }
    if (n0 + 8 < N) {
      y1[n0 + 8] = __float2bfloat16(e2);
    }
    if (n1 + 8 < N) {
      y1[n1 + 8] = __float2bfloat16(e3);
    }
  }
}

__global__ void __launch_bounds__(256, 2)
    chr_nf4_gemm_prefill_n16(const uint8_t *__restrict__ packed,
                              const uint16_t *__restrict__ scale,
                              const __nv_bfloat16 *__restrict__ x,
                              __nv_bfloat16 *__restrict__ y,
                              float *__restrict__ partial, int M, int K,
                              int K_pad, int N, int n_ktiles,
                              int tiles_per_split) {
  prefill_gemm_body<16>(packed, scale, x, y, partial, M, K, K_pad, N, n_ktiles,
                         tiles_per_split);
}

__global__ void __launch_bounds__(256, 2)
    chr_nf4_gemm_prefill_n8(const uint8_t *__restrict__ packed,
                             const uint16_t *__restrict__ scale,
                             const __nv_bfloat16 *__restrict__ x,
                             __nv_bfloat16 *__restrict__ y,
                             float *__restrict__ partial, int M, int K,
                             int K_pad, int N, int n_ktiles,
                             int tiles_per_split) {
  prefill_gemm_body<8>(packed, scale, x, y, partial, M, K, K_pad, N, n_ktiles,
                        tiles_per_split);
}

// --- planned prefill N=17..64: BN=32 / BN=64, same BM/BK as n16 -----------
// Compiled so a later GPU-free rebuild can type-check them. chr_nf4_gemm_ws
// still refuses N>kLiveMaxN, so TokenLoop cannot reach these. Remaining work
// after unfreeze: oracle at N=17/32/64, ncu vs 2x n16, register pressure on
// n64 (8 m16n8 per warp), then raise kLiveMaxN / LIVE_MAX_N together.

template <int BN>
__device__ void issue_x_prefill_wide(__nv_bfloat16 *dst, const __nv_bfloat16 *x,
                                      int k0, int K, int N) {
  static_assert(BN == 32 || BN == 64, "planned wide BN");
  constexpr int kStageBytes = pBK * BN * 2;
  const int tid = static_cast<int>(threadIdx.x);
  const int n_k = K > k0 ? min(pBK, K - k0) : 0;
  const int n_bytes = n_k * N * 2;
  const int aligned = n_bytes & ~15;
  const int rem = n_bytes - aligned;
  const char *gbase = reinterpret_cast<const char *>(
      x + static_cast<size_t>(k0) * static_cast<size_t>(N));
  char *sbase = reinterpret_cast<char *>(dst);
  const int n_chunks = kStageBytes / 16;
  for (int i = tid; i < n_chunks; i += pBlock) {
    const int off = i * 16;
    int src_bytes = 0;
    const void *src = x;
    if (off + 16 <= n_bytes) {
      src = gbase + off;
      src_bytes = 16;
    } else if (off == aligned && rem > 0) {
      src = gbase + off;
      src_bytes = rem >= 8 ? 8 : rem >= 4 ? 4 : 0;
    } else if (off >= aligned + (rem > 0 ? 16 : 0)) {
      src_bytes = 0;
    }
    cp_async_ca_16(sbase + off, src, src_bytes);
  }
}

template <int BN>
__device__ __forceinline__ void compute_tile_prefill_wide(
    const uint8_t *pk, const __nv_bfloat16 *xs, const uint16_t *scale, int m0,
    int M, int K, int k0, int n_groups, float acc[][4]) {
  static_assert(BN == 32 || BN == 64, "planned wide BN");
  constexpr int kFrags = BN / 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp & (pWarpsM - 1);
  const int wk = warp >> 2;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int local0 = (wm << 4) + g;
  const int local1 = local0 + 8;
  const int row0 = m0 + local0;
  const int row1 = m0 + local1;
  const bool live0 = row0 < M;
  const bool live1 = row1 < M;
  const uint8_t *row_pk0 = pk + local0 * (pBK / 2);
  const uint8_t *row_pk1 = pk + local1 * (pBK / 2);

  const int k_grp = k0 + wk * kGroup;
  if (k_grp >= K) {
    return;
  }

  const int grp = k_grp / kGroup;
  float s0 = 0.f, s1 = 0.f;
  if (live0) {
    s0 = __half2float(__ushort_as_half(
        __ldg(scale + static_cast<size_t>(row0) * n_groups + grp)));
  }
  if (live1) {
    s1 = __half2float(__ushort_as_half(
        __ldg(scale + static_cast<size_t>(row1) * n_groups + grp)));
  }

  if (k_grp + kGroup <= K) {
#pragma unroll
    for (int ki = 0; ki < kGroup / 16; ++ki) {
      const int k_lo = k_grp + ki * 16 + (t << 1);
      const int k_hi = k_lo + 8;
      uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      if (live0) {
        a0 = dequant_pair_live(row_pk0, k0, k_lo, s0);
        a2 = dequant_pair_live(row_pk0, k0, k_hi, s0);
      }
      if (live1) {
        a1 = dequant_pair_live(row_pk1, k0, k_lo, s1);
        a3 = dequant_pair_live(row_pk1, k0, k_hi, s1);
      }
      const int o_lo = (k_lo - k0) * BN;
      const int o_hi = (k_hi - k0) * BN;
#pragma unroll
      for (int f = 0; f < kFrags; ++f) {
        const __nv_bfloat16 *xf = xs + g + f * 8;
        const uint32_t b0 = pack_bf16x2_bits(xf[o_lo], xf[o_lo + BN]);
        const uint32_t b1 = pack_bf16x2_bits(xf[o_hi], xf[o_hi + BN]);
        mma_m16n8k16(a0, a1, a2, a3, b0, b1, acc[f][0], acc[f][1], acc[f][2],
                     acc[f][3]);
      }
    }
    return;
  }

#pragma unroll 1
  for (int ki = 0; ki < kGroup / 16; ++ki) {
    const int k_tile = k_grp + ki * 16;
    if (k_tile >= K) {
      break;
    }
    const int k_lo = k_tile + (t << 1);
    const int k_hi = k_lo + 8;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (live0) {
      a0 = dequant_pair(row_pk0, k0, k_lo, K, s0);
      a2 = dequant_pair(row_pk0, k0, k_hi, K, s0);
    }
    if (live1) {
      a1 = dequant_pair(row_pk1, k0, k_lo, K, s1);
      a3 = dequant_pair(row_pk1, k0, k_hi, K, s1);
    }
    const __nv_bfloat16 z = __float2bfloat16(0.f);
    const int o_lo = (k_lo - k0) * BN;
    const int o_hi = (k_hi - k0) * BN;
    const bool lo0 = k_lo < K, lo1 = k_lo + 1 < K;
    const bool hi0 = k_hi < K, hi1 = k_hi + 1 < K;
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      const __nv_bfloat16 *xf = xs + g + f * 8;
      const uint32_t b0 = pack_bf16x2_bits(lo0 ? xf[o_lo] : z,
                                           lo1 ? xf[o_lo + BN] : z);
      const uint32_t b1 = pack_bf16x2_bits(hi0 ? xf[o_hi] : z,
                                           hi1 ? xf[o_hi + BN] : z);
      mma_m16n8k16(a0, a1, a2, a3, b0, b1, acc[f][0], acc[f][1], acc[f][2],
                   acc[f][3]);
    }
  }
}

template <int BN>
__device__ void prefill_gemm_body_wide(
    const uint8_t *__restrict__ packed, const uint16_t *__restrict__ scale,
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y,
    float *__restrict__ partial, int M, int K, int K_pad, int N,
    int n_ktiles, int tiles_per_split) {
  static_assert(BN == 32 || BN == 64, "planned wide BN");
  constexpr int kFrags = BN / 8;
  constexpr int kRedStride = kFrags * 4 + 1;
  constexpr int kXStageElems = pBK * BN;
  const int m0 = static_cast<int>(blockIdx.x) * pBM;
  const int split = static_cast<int>(blockIdx.y);
  const int tile0 = split * tiles_per_split;
  const int n_tiles_local = min(tiles_per_split, n_ktiles - tile0);
  const int n_groups = K_pad / kGroup;
  const int packed_stride = K_pad / 2;

  extern __shared__ char smem[];
  uint8_t *pk_base = reinterpret_cast<uint8_t *>(smem);
  __nv_bfloat16 *x_base = reinterpret_cast<__nv_bfloat16 *>(
      pk_base + pStages * pPackedStageBytes);

#pragma unroll
  for (int s = 0; s < pStages - 1; ++s) {
    if (s < n_tiles_local) {
      const int k0 = (tile0 + s) * pBK;
      issue_packed_prefill(pk_base + s * pPackedStageBytes, packed, m0, M, k0,
                           K_pad, packed_stride);
      issue_x_prefill_wide<BN>(x_base + s * kXStageElems, x, k0, K, N);
    }
    cp_async_commit();
  }

  int smem_write = pStages - 1;
  int smem_read = 0;
  float acc[kFrags][4];
#pragma unroll
  for (int f = 0; f < kFrags; ++f) {
    acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f;
  }

  for (int tile = 0; tile < n_tiles_local; ++tile) {
    if (tile + pStages - 1 < n_tiles_local) {
      const int k0 = (tile0 + tile + pStages - 1) * pBK;
      issue_packed_prefill(pk_base + smem_write * pPackedStageBytes, packed, m0,
                           M, k0, K_pad, packed_stride);
      issue_x_prefill_wide<BN>(x_base + smem_write * kXStageElems, x, k0, K, N);
    }
    cp_async_commit();
    cp_async_wait<pStages - 2>();
    __syncthreads();

    const int k0 = (tile0 + tile) * pBK;
    prepare_x_prefill<BN>(x_base + smem_read * kXStageElems, x, k0, K, N);

    const uint8_t *pk = pk_base + smem_read * pPackedStageBytes;
    const __nv_bfloat16 *xs = x_base + smem_read * kXStageElems;
    compute_tile_prefill_wide<BN>(pk, xs, scale, m0, M, K, k0, n_groups, acc);

    __syncthreads();
    smem_write = (smem_write + 1) % pStages;
    smem_read = (smem_read + 1) % pStages;
  }
  cp_async_wait<0>();

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp & (pWarpsM - 1);
  const int wk = warp >> 2;

  __syncthreads();
  float *red = reinterpret_cast<float *>(smem);
  float *slot = red + static_cast<size_t>(wm * 32 + lane) * kRedStride;
  if (wk == 1) {
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      slot[f * 4 + 0] = acc[f][0];
      slot[f * 4 + 1] = acc[f][1];
      slot[f * 4 + 2] = acc[f][2];
      slot[f * 4 + 3] = acc[f][3];
    }
  }
  __syncthreads();
  if (wk != 0) {
    return;
  }
#pragma unroll
  for (int f = 0; f < kFrags; ++f) {
    acc[f][0] += slot[f * 4 + 0];
    acc[f][1] += slot[f * 4 + 1];
    acc[f][2] += slot[f * 4 + 2];
    acc[f][3] += slot[f * 4 + 3];
  }

  const int g = lane >> 2;
  const int t = lane & 3;
  const int r0 = m0 + (wm << 4) + g;
  const int r1 = r0 + 8;
  const int n_pair = t << 1;
  if (partial != nullptr) {
    float *p = partial + static_cast<size_t>(split) * M * N;
    if (r0 < M) {
      float *p0 = p + static_cast<size_t>(r0) * N;
#pragma unroll
      for (int f = 0; f < kFrags; ++f) {
        const int n0 = n_pair + f * 8;
        if (n0 < N) {
          p0[n0] = acc[f][0];
        }
        if (n0 + 1 < N) {
          p0[n0 + 1] = acc[f][1];
        }
      }
    }
    if (r1 < M) {
      float *p1 = p + static_cast<size_t>(r1) * N;
#pragma unroll
      for (int f = 0; f < kFrags; ++f) {
        const int n0 = n_pair + f * 8;
        if (n0 < N) {
          p1[n0] = acc[f][2];
        }
        if (n0 + 1 < N) {
          p1[n0 + 1] = acc[f][3];
        }
      }
    }
    return;
  }
  if (r0 < M) {
    __nv_bfloat16 *y0 = y + static_cast<size_t>(r0) * N;
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      const int n0 = n_pair + f * 8;
      if (n0 < N) {
        y0[n0] = __float2bfloat16(acc[f][0]);
      }
      if (n0 + 1 < N) {
        y0[n0 + 1] = __float2bfloat16(acc[f][1]);
      }
    }
  }
  if (r1 < M) {
    __nv_bfloat16 *y1 = y + static_cast<size_t>(r1) * N;
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      const int n0 = n_pair + f * 8;
      if (n0 < N) {
        y1[n0] = __float2bfloat16(acc[f][2]);
      }
      if (n0 + 1 < N) {
        y1[n0 + 1] = __float2bfloat16(acc[f][3]);
      }
    }
  }
}

__global__ void __launch_bounds__(256, 2)
    chr_nf4_gemm_prefill_n32(const uint8_t *__restrict__ packed,
                              const uint16_t *__restrict__ scale,
                              const __nv_bfloat16 *__restrict__ x,
                              __nv_bfloat16 *__restrict__ y,
                              float *__restrict__ partial, int M, int K,
                              int K_pad, int N, int n_ktiles,
                              int tiles_per_split) {
  prefill_gemm_body_wide<32>(packed, scale, x, y, partial, M, K, K_pad, N,
                              n_ktiles, tiles_per_split);
}

__global__ void __launch_bounds__(256, 2)
    chr_nf4_gemm_prefill_n64(const uint8_t *__restrict__ packed,
                              const uint16_t *__restrict__ scale,
                              const __nv_bfloat16 *__restrict__ x,
                              __nv_bfloat16 *__restrict__ y,
                              float *__restrict__ partial, int M, int K,
                              int K_pad, int N, int n_ktiles,
                              int tiles_per_split) {
  prefill_gemm_body_wide<64>(packed, scale, x, y, partial, M, K, K_pad, N,
                              n_ktiles, tiles_per_split);
}

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

int k_pad_from_k(int K) { return kGroup * ((K + kGroup - 1) / kGroup); }

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// --- tuning knobs -----------------------------------------------------------
// Defaults are the product behaviour. The overrides exist so gpu/nf4/bench.py
// can time both grids inside one process, and so a bad planner decision can be
// pinned from the outside without a rebuild.

struct Tuning {
  int path;     // 0 auto, 1 force classic decode tile, 2 force small tile
  int split_k;  // 0 auto, >0 forced
  int one_wave; // CTAs that count as "the card is busy"
};

int env_int(const char *name, int fallback) {
  const char *v = std::getenv(name);
  if (!v || !*v) {
    return fallback;
  }
  char *end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

Tuning &tuning() {
  static Tuning t = [] {
    Tuning init{};
    init.path = env_int("CHR_NF4_PATH", 0);
    init.split_k = env_int("CHR_NF4_SPLIT_K", 0);
    init.one_wave = env_int("CHR_NF4_ONE_WAVE", kDefaultOneWave);
    if (init.path < 0 || init.path > 2) {
      init.path = 0;
    }
    if (init.split_k < 0) {
      init.split_k = 0;
    }
    if (init.one_wave < 1) {
      init.one_wave = kDefaultOneWave;
    }
    return init;
  }();
  return t;
}

// split_k so that grid_x * split_k reaches kTargetCtas, capped at n_ktiles
// (a split with no K tiles to walk is a CTA that only writes zeros). The
// returned split is recomputed from tiles_per_split so the two always agree.
void pick_split(int grid_x, int n_ktiles, int have_ws, int *split,
                int *tiles_per_split) {
  const Tuning &t = tuning();
  int want = 1;
  if (have_ws) {
    const int target = t.one_wave * 2 > kTargetCtas ? t.one_wave * 2 : kTargetCtas;
    want = t.split_k > 0 ? t.split_k : ceil_div(target, grid_x);
  }
  if (want < 1) {
    want = 1;
  }
  if (want > n_ktiles) {
    want = n_ktiles;
  }
  const int tps = ceil_div(n_ktiles, want);
  *tiles_per_split = tps;
  *split = ceil_div(n_ktiles, tps);
}

int plan_impl(const chr_nf4_dev_t &h, int32_t N, int32_t have_ws,
              chr_nf4_plan_t *out) {
  const Tuning &t = tuning();
  chr_nf4_plan_t p{};
  if (N == 1) {
    const int classic_x = ceil_div(h.M, kBM);
    // Measured on the 3080 with gpu/nf4/bench.py at N=1, Qwen2.5-3B shapes
    // (us per launch, weights rotated so L2 cannot hold them):
    //
    //   matrix      BM=128 grid   us   |  BM=64 + split_k   us
    //   q/o_proj      (16,1)     180   |     (32,4)         46
    //   k/v_proj       (2,1)     159   |     (4,16)          22
    //   gate/up_proj  (86,1)     321   |    (172,1)         224
    //   down_proj     (16,1)     846   |     (32,5)         235
    //   lm_head     (1187,1)    2768   |   (2374,1)        2646
    //
    // The 64-row tile is ahead at every shape, so auto takes it for all of
    // decode and the 128-row tile stays reachable for the comparison
    // (CHR_NF4_PATH=1 / nf4_set_tuning(path=1)) rather than as the default.
    const bool small = (t.path != 1);
    if (!small) {
      p.path = 0;
      p.bm = kBM;
      p.bk = kBK;
      p.block = kBlock;
      p.grid_x = classic_x;
      p.grid_y = 1;
      p.n_ktiles = ceil_div(h.K_pad, kBK);
      p.tiles_per_split = p.n_ktiles;
      p.smem_bytes = kSmemBytes;
    } else {
      p.path = 1;
      p.bm = sBM;
      p.bk = sBK;
      p.block = sBlock;
      p.grid_x = ceil_div(h.M, sBM);
      p.n_ktiles = ceil_div(h.K_pad, sBK);
      pick_split(p.grid_x, p.n_ktiles, have_ws, &p.grid_y, &p.tiles_per_split);
      p.smem_bytes = sSmemBytes;
      if (p.grid_y > 1) {
        p.ws_floats = static_cast<int64_t>(p.grid_y) * h.M;
      }
    }
  } else if (N <= 16) {
    // Tile width, not kLiveMaxN. Raising kLiveMaxN to 32 must still send
    // N=17..32 to BN=32, not reuse the n16 smem ring with N=32.
    p.path = 2;
    p.bm = pBM;
    p.bk = pBK;
    p.block = pBlock;
    p.grid_x = ceil_div(h.M, pBM);
    p.n_ktiles = ceil_div(h.K_pad, pBK);
    pick_split(p.grid_x, p.n_ktiles, have_ws, &p.grid_y, &p.tiles_per_split);
    p.smem_bytes = N <= 8 ? pSmemBytes8 : pSmemBytes16;
    if (p.grid_y > 1) {
      p.ws_floats = static_cast<int64_t>(p.grid_y) * h.M * N;
    }
  } else {
    // BN=32 / BN=64. Same BM/BK as n16. Live refuses N > kLiveMaxN.
    p.path = N <= 32 ? 3 : 4;
    p.bm = pBM;
    p.bk = pBK;
    p.block = pBlock;
    p.grid_x = ceil_div(h.M, pBM);
    p.n_ktiles = ceil_div(h.K_pad, pBK);
    pick_split(p.grid_x, p.n_ktiles, have_ws, &p.grid_y, &p.tiles_per_split);
    p.smem_bytes = N <= 32 ? pSmemBytes32 : pSmemBytes64;
    if (p.grid_y > 1) {
      p.ws_floats = static_cast<int64_t>(p.grid_y) * h.M * N;
    }
  }
  p.ctas = p.grid_x * p.grid_y;
  *out = p;
  return 0;
}

int check_args(const chr_nf4_dev_t *w, int32_t N, bool need_ptrs,
               chr_nf4_dev_t *h_out, int32_t max_n) {
  if (!w) {
    return -1;
  }
  // Launch (max_n = kLiveMaxN): N in [1, 32]. N=1 decode; N=2..8 pad-8;
  // N=9..16 pad-16; N=17..32 pad-32. Plan (max_n = kPlanMaxN) also
  // describes N=33..64. TokenLoop chunks at kLiveMaxN; this does not slice.
  if (N < 1 || N > max_n) {
    return -2;
  }
  const chr_nf4_dev_t h = *w;
  if (h.M < 1 || h.K < 1) {
    return -3;
  }
  if (need_ptrs && (!h.packed || !h.scale)) {
    return -3;
  }
  if (h.K_pad != k_pad_from_k(h.K) || h.K_pad < h.K) {
    return -4;
  }
  *h_out = h;
  return 0;
}

} // namespace

extern "C" void chr_nf4_set_tuning(int32_t path, int32_t split_k,
                                   int32_t one_wave) {
  Tuning &t = tuning();
  t.path = (path >= 0 && path <= 2) ? path : 0;
  t.split_k = split_k > 0 ? split_k : 0;
  t.one_wave = one_wave > 0 ? one_wave : kDefaultOneWave;
}

extern "C" int chr_nf4_gemm_plan(const chr_nf4_dev_t *w, int32_t N,
                                 int32_t have_ws, chr_nf4_plan_t *out) {
  if (!out) {
    return -1;
  }
  chr_nf4_dev_t h{};
  const int rc = check_args(w, N, /*need_ptrs=*/false, &h, kPlanMaxN);
  if (rc != 0) {
    return rc;
  }
  return plan_impl(h, N, have_ws, out);
}

extern "C" int chr_nf4_gemm_ws_max(const chr_nf4_dev_t *w, const void *x, void *y,
                                   int32_t N, float *ws, int64_t ws_floats,
                                   void *stream, int32_t max_n) {
  if (!x || !y) {
    return -1;
  }
  if (max_n < 1 || max_n > kPlanMaxN) {
    return -2;
  }
  chr_nf4_dev_t h{};
  const int rc = check_args(w, N, /*need_ptrs=*/true, &h, max_n);
  if (rc != 0) {
    return rc;
  }
  if (!aligned16(h.packed) || !aligned16(x) ||
      (reinterpret_cast<uintptr_t>(h.scale) & 1u) != 0u) {
    return -5;
  }

  static_assert(kBN == 8, "decode N=1 pads BN to 8 for m16n8");
  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  const __nv_bfloat16 *x_bf = reinterpret_cast<const __nv_bfloat16 *>(x);
  __nv_bfloat16 *y_bf = reinterpret_cast<__nv_bfloat16 *>(y);

  const int have_ws = (ws != nullptr && ws_floats > 0) ? 1 : 0;
  chr_nf4_plan_t p{};
  plan_impl(h, N, have_ws, &p);
  if (p.ws_floats > 0 && (!have_ws || ws_floats < p.ws_floats)) {
    return -7;
  }
  float *partial = p.grid_y > 1 ? ws : nullptr;
  const dim3 grid(static_cast<unsigned>(p.grid_x),
                  static_cast<unsigned>(p.grid_y));
  const dim3 block(static_cast<unsigned>(p.block));

  if (p.path == 0) {
    cudaError_t attr = cudaFuncSetAttribute(
        chr_nf4_gemm_decode_n1, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemBytes);
    if (attr != cudaSuccess) {
      return -6;
    }
    chr_nf4_gemm_decode_n1<<<grid, block, kSmemBytes, s>>>(
        h.packed, h.scale, x_bf, y_bf, h.M, h.K, h.K_pad);
  } else if (p.path == 1) {
    chr_nf4_gemm_decode_small<<<grid, block, sSmemBytes, s>>>(
        h.packed, h.scale, x_bf, y_bf, partial, h.M, h.K, h.K_pad, p.n_ktiles,
        p.tiles_per_split);
  } else if (N <= 8) {
    cudaError_t attr = cudaFuncSetAttribute(
        chr_nf4_gemm_prefill_n8, cudaFuncAttributeMaxDynamicSharedMemorySize,
        pSmemBytes8);
    if (attr != cudaSuccess) {
      return -6;
    }
    chr_nf4_gemm_prefill_n8<<<grid, block, pSmemBytes8, s>>>(
        h.packed, h.scale, x_bf, y_bf, partial, h.M, h.K, h.K_pad, N,
        p.n_ktiles, p.tiles_per_split);
  } else if (N <= 16) {
    cudaError_t attr = cudaFuncSetAttribute(
        chr_nf4_gemm_prefill_n16, cudaFuncAttributeMaxDynamicSharedMemorySize,
        pSmemBytes16);
    if (attr != cudaSuccess) {
      return -6;
    }
    chr_nf4_gemm_prefill_n16<<<grid, block, pSmemBytes16, s>>>(
        h.packed, h.scale, x_bf, y_bf, partial, h.M, h.K, h.K_pad, N,
        p.n_ktiles, p.tiles_per_split);
  } else if (N <= 32) {
    // Live when kLiveMaxN >= 32. Do not key this branch on kLiveMaxN:
    // N<=kLiveMaxN with a cap of 32 would otherwise launch n16.
    cudaError_t attr = cudaFuncSetAttribute(
        chr_nf4_gemm_prefill_n32, cudaFuncAttributeMaxDynamicSharedMemorySize,
        pSmemBytes32);
    if (attr != cudaSuccess) {
      return -6;
    }
    chr_nf4_gemm_prefill_n32<<<grid, block, pSmemBytes32, s>>>(
        h.packed, h.scale, x_bf, y_bf, partial, h.M, h.K, h.K_pad, N,
        p.n_ktiles, p.tiles_per_split);
  } else {
    cudaError_t attr = cudaFuncSetAttribute(
        chr_nf4_gemm_prefill_n64, cudaFuncAttributeMaxDynamicSharedMemorySize,
        pSmemBytes64);
    if (attr != cudaSuccess) {
      return -6;
    }
    chr_nf4_gemm_prefill_n64<<<grid, block, pSmemBytes64, s>>>(
        h.packed, h.scale, x_bf, y_bf, partial, h.M, h.K, h.K_pad, N,
        p.n_ktiles, p.tiles_per_split);
  }

  if (partial != nullptr) {
    const int rows = h.M * N;
    const int rblock = 256;
    chr_nf4_reduce_splitk<<<dim3(static_cast<unsigned>(ceil_div(rows, rblock))),
                            dim3(rblock), 0, s>>>(partial, y_bf, rows,
                                                  p.grid_y);
  }

  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -6;
}

extern "C" int chr_nf4_gemm_ws(const chr_nf4_dev_t *w, const void *x, void *y,
                                int32_t N, float *ws, int64_t ws_floats,
                                void *stream) {
  return chr_nf4_gemm_ws_max(w, x, y, N, ws, ws_floats, stream, kLiveMaxN);
}

extern "C" int chr_nf4_gemm(const chr_nf4_dev_t *w, const void *x, void *y,
                            int32_t N, void *stream) {
  // No workspace: the planner pins split_k to 1, so this keeps the wave-2
  // meaning of the entry point for any caller that has not been taught to
  // allocate FP32 partials.
  return chr_nf4_gemm_ws(w, x, y, N, nullptr, 0, stream);
}
