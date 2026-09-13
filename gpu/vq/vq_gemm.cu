// Fused VQ 2×8 dequant + HMMA.16816 GEMM for sm_86.
// y = dequant_vq(W) @ x. Reconstruct in registers; no BF16 W in HBM.
// Book [2,256,8] FP16 lives in smem (8 KiB) once before the K-loop.
// We never gather a 2^16 book.
//
// Specs: docs/spec/stitch-gpu.md (wins), docs/spec/vq.md §6–7, docs/kernel-ampere.md
// Tile: BM=128, BN=8 pad, BK=256, block=256, stages=3, split_k=1.
// MMA: only mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32

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
constexpr int kGroup = 8;
constexpr int kNBooks = 2;
constexpr int kNCentroids = 256;
constexpr int kBookElems = kNBooks * kNCentroids * kGroup; // 4096 fp16
constexpr int kBookBytes = kBookElems * 2;                 // 8192
constexpr int kIndexStageBytes = kBM * (kBK / kGroup) * 2;  // 8192
constexpr int kXStageElems = kBK;                           // 256 bf16
constexpr int kSmemBytes =
    kBookBytes + kStages * kIndexStageBytes + kStages * kXStageElems * 2;

static_assert(kBookBytes == 8192, "two 256x8 fp16 books = 8 KiB");
static_assert(kBookBytes <= 8192, "book must stay in smem, never a 1 MiB gather");
static_assert(kIndexStageBytes == 8192, "index tile BM*(BK/8)*2");
static_assert(kBN == 8, "decode N=1 pads BN to 8 for m16n8");

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

__device__ void issue_book(uint16_t *dst, const uint16_t *book) {
  constexpr int kChunks = kBookBytes / 16;
  const int tid = static_cast<int>(threadIdx.x);
  const uint8_t *src8 = reinterpret_cast<const uint8_t *>(book);
  uint8_t *dst8 = reinterpret_cast<uint8_t *>(dst);
  for (int i = tid; i < kChunks; i += kBlock) {
    cp_async_cg_16(dst8 + i * 16, src8 + i * 16, 16);
  }
}

__device__ void issue_index(uint8_t *dst, const uint8_t *index, int m0, int M,
                            int k0, int K_pad) {
  constexpr int kBytesPerRow = (kBK / kGroup) * 2;
  constexpr int kChunksPerRow = kBytesPerRow / 16;
  constexpr int kNChunks = kBM * kChunksPerRow;
  const int index_stride = (K_pad / kGroup) * 2;
  const int col0 = (k0 / kGroup) * 2;
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kNChunks; i += kBlock) {
    const int row = i / kChunksPerRow;
    const int chunk = i % kChunksPerRow;
    const int gm = m0 + row;
    const int col = col0 + chunk * 16;
    uint8_t *out = dst + row * kBytesPerRow + chunk * 16;
    int src_bytes = 0;
    const uint8_t *src = index;
    if (gm < M && col < index_stride) {
      src = index + static_cast<size_t>(gm) * static_cast<size_t>(index_stride) +
            static_cast<size_t>(col);
      const int remain = index_stride - col;
      if (remain >= 16 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u) {
        src_bytes = 16;
      }
    }
    cp_async_cg_16(out, src, src_bytes);
  }
}

__device__ void issue_x(__nv_bfloat16 *dst, const __nv_bfloat16 *x, int k0,
                        int K) {
  constexpr int kChunks = kBK / 8;
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

// K_pad is only 8-aligned, so a row's index stride need not be 16B. cp.async
// covers the aligned 16B prefix; this copies the rest after wait_group.
__device__ void fill_index_tail(uint8_t *dst, const uint8_t *index, int m0,
                               int M, int k0, int K_pad) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= kBM) {
    return;
  }
  constexpr int kBytesPerRow = (kBK / kGroup) * 2;
  const int index_stride = (K_pad / kGroup) * 2;
  const int col0 = (k0 / kGroup) * 2;
  const int gm = m0 + tid;
  uint8_t *row = dst + tid * kBytesPerRow;
  if (gm >= M || col0 >= index_stride) {
    return;
  }
  const int remain = index_stride - col0;
  const int bytes_copy = remain < kBytesPerRow ? remain : kBytesPerRow;
  const uint8_t *src =
      index + static_cast<size_t>(gm) * static_cast<size_t>(index_stride) +
      static_cast<size_t>(col0);
  int async_end = 0;
  while (async_end + 16 <= bytes_copy) {
    const uint8_t *s = src + async_end;
    if ((reinterpret_cast<uintptr_t>(s) & 15u) != 0u) {
      break;
    }
    async_end += 16;
  }
  for (int b = async_end; b < bytes_copy; ++b) {
    row[b] = src[b];
  }
  for (int b = bytes_copy; b < kBytesPerRow; ++b) {
    row[b] = 0;
  }
}

__device__ uint32_t dequant_vq_pair(const uint8_t *row_idx,
                                    const uint16_t *book, int k0, int k,
                                    int K) {
  if (k >= K) {
    return 0u;
  }
  const int j = (k - k0) >> 3;
  const int d = k & 7;
  const uint8_t *pair = row_idx + (j << 1);
  const unsigned i1 = pair[0];
  const unsigned i2 = pair[1];
  const uint16_t *c1 = book + (i1 << 3) + d;
  const uint16_t *c2 = book + (kNCentroids << 3) + (i2 << 3) + d;
  const float w0 = __half2float(__ushort_as_half(c1[0])) +
                   __half2float(__ushort_as_half(c2[0]));
  const float w1 =
      (k + 1 < K) ? (__half2float(__ushort_as_half(c1[1])) +
                      __half2float(__ushort_as_half(c2[1])))
                  : 0.f;
  return pack_bf16x2(w0, w1);
}

__device__ void compute_tile(const uint8_t *idx, const __nv_bfloat16 *xs,
                             const uint16_t *book, int m0, int M, int K,
                             int k0, float &d0, float &d1, float &d2,
                             float &d3) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row0 = m0 + (warp << 4) + g;
  const int row1 = row0 + 8;
  const int local0 = (warp << 4) + g;
  const int local1 = local0 + 8;
  constexpr int kIdxRow = (kBK / kGroup) * 2;

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
      const uint8_t *row_idx = idx + local0 * kIdxRow;
      a0 = dequant_vq_pair(row_idx, book, k0, k_lo, K);
      a2 = dequant_vq_pair(row_idx, book, k0, k_hi, K);
    }
    if (row1 < M) {
      const uint8_t *row_idx = idx + local1 * kIdxRow;
      a1 = dequant_vq_pair(row_idx, book, k0, k_lo, K);
      a3 = dequant_vq_pair(row_idx, book, k0, k_hi, K);
    }

    uint32_t b0 = 0, b1 = 0;
    if (g == 0) {
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
    chr_vq_gemm_decode_n1(const uint8_t *__restrict__ index,
                          const uint16_t *__restrict__ book,
                          const __nv_bfloat16 *__restrict__ x,
                          __nv_bfloat16 *__restrict__ y, int M, int K,
                          int K_pad) {
  const int m0 = static_cast<int>(blockIdx.x) * kBM;
  const int n_tiles = (K_pad + kBK - 1) / kBK;

  extern __shared__ char smem[];
  uint16_t *book_s = reinterpret_cast<uint16_t *>(smem);
  uint8_t *idx_base = reinterpret_cast<uint8_t *>(book_s + kBookElems);
  __nv_bfloat16 *x_base = reinterpret_cast<__nv_bfloat16 *>(
      idx_base + kStages * kIndexStageBytes);

  issue_book(book_s, book);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < n_tiles) {
      const int k0 = s * kBK;
      issue_index(idx_base + s * kIndexStageBytes, index, m0, M, k0, K_pad);
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
      issue_index(idx_base + smem_write * kIndexStageBytes, index, m0, M, k0,
                  K_pad);
      issue_x(x_base + smem_write * kXStageElems, x, k0, K);
    }
    cp_async_commit();
    cp_async_wait<kStages - 2>();
    __syncthreads();

    fill_x_tail(x_base + smem_read * kXStageElems, x, tile * kBK, K);
    fill_index_tail(idx_base + smem_read * kIndexStageBytes, index, m0, M,
                   tile * kBK, K_pad);
    __syncthreads();

    compute_tile(idx_base + smem_read * kIndexStageBytes,
                 x_base + smem_read * kXStageElems, book_s, m0, M, K,
                 tile * kBK, d0, d1, d2, d3);

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

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

int k_pad_from_k(int K) { return kGroup * ((K + kGroup - 1) / kGroup); }

} // namespace

extern "C" int chr_vq_gemm(const chr_vq_dev_t *w, const void *x, void *y,
                            int32_t N, void *stream) {
  if (!w || !x || !y) {
    return -1;
  }
  if (N != 1) {
    return -2;
  }
  const chr_vq_dev_t h = *w;
  if (h.M < 1 || h.K < 1 || !h.index || !h.book) {
    return -3;
  }
  if (h.K_pad != k_pad_from_k(h.K) || h.K_pad < h.K) {
    return -4;
  }
  if (!aligned16(h.index) || !aligned16(h.book) || !aligned16(x)) {
    return -5;
  }

  const dim3 grid(static_cast<unsigned>((h.M + kBM - 1) / kBM));
  const dim3 block(kBlock);
  cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  cudaError_t attr = cudaFuncSetAttribute(
      chr_vq_gemm_decode_n1, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemBytes);
  if (attr != cudaSuccess) {
    return -6;
  }

  chr_vq_gemm_decode_n1<<<grid, block, kSmemBytes, s>>>(
      h.index, h.book, reinterpret_cast<const __nv_bfloat16 *>(x),
      reinterpret_cast<__nv_bfloat16 *>(y), h.M, h.K, h.K_pad);

  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : -6;
}
