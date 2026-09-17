// N=1 CUDA-core NF4 GEMV. y[M] = dequant(W) @ x[K].
// Fused dequant×vector (same job as llama.cpp mmvq). Does not replace HMMA gemm.
// Packing: docs/spec/nf4.md low nibble = even K. LUT bits match nf4.md §1.
//
// Four warps split K (mmvq ncols=1). LUT lives in smem: a __constant__ table
// serializes when every lane hits a different nibble. Large M packs several
// output rows per CTA so x is loaded once per K-tile (lm_head / SwiGLU).
// M < 4096 stays one row: NR>1 on 3B q/o/down (M=2048) underfills the SMs.
// Same-tensor A/B: lm_head 152k×2048 is ~20% faster at NR=4 than NR=1;
// 3B down prefers NR=1. Override: CHR_NF4_GEMV_NR=1|2|4|8.
//
// chr_nf4_gemv_qkv: one grid over q+k+v rows so the tiny KV projections
// run with Q, not as two 256-CTA afterthoughts.
// chr_nf4_gemv_swiglu: each CTA owns NR gate+up pairs, shares x, writes
// silu(gate)*up. No second copy of the packed weights.
// Optional rms_w: each CTA RMSNorms x into dynamic smem (K bf16) then GEMVs.

#include "chr_gpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kGroup = 64;
constexpr int kWarps = 4;
constexpr int kBlock = kWarps * 32;
constexpr int kAttnWarps = 4;
constexpr int kAttnBlock = kAttnWarps * 32;

__constant__ uint32_t kNf4LutBits[16] = {
    0xBF800000u, 0xBF3239B1u, 0xBF066B30u, 0xBECA32A0u, 0xBE91A24Du, 0xBE3D353Fu,
    0xBDBA7871u, 0x00000000u, 0x3DA2FAFFu, 0x3E24CAE3u, 0x3E7C04DDu, 0x3EAD033Au,
    0x3EE1A4B8u, 0x3F1007ABu, 0x3F3913B3u, 0x3F800000u,
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  return v;
}

__device__ __forceinline__ float warp_max_all(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ float bf16_pair(unsigned bits, float w0, float w1) {
  const float x0 = __bfloat162float(
      __ushort_as_bfloat16(static_cast<unsigned short>(bits & 0xFFFFu)));
  const float x1 = __bfloat162float(
      __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16)));
  return w0 * x0 + w1 * x1;
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.f + expf(-x));
}

__device__ __forceinline__ float dot8(uint32_t pk, float s, uint4 xv,
                                      const float *lut) {
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned bits = i == 0 ? xv.x : i == 1 ? xv.y : i == 2 ? xv.z : xv.w;
    const unsigned byte = (pk >> (8 * i)) & 0xFFu;
    acc += bf16_pair(bits, lut[byte & 0xFu] * s, lut[(byte >> 4) & 0xFu] * s);
  }
  return acc;
}

__device__ __forceinline__ float tile16_pre(const uint8_t *prow,
                                            const uint16_t *srow, int k0,
                                            uint4 x0, uint4 x1,
                                            const float *lut) {
  const int grp = k0 / kGroup;
  const float s = __half2float(__ushort_as_half(__ldg(srow + grp)));
  const uint2 pk = __ldg(reinterpret_cast<const uint2 *>(prow + (k0 >> 1)));
  return dot8(pk.x, s, x0, lut) + dot8(pk.y, s, x1, lut);
}

__device__ void load_x16(const __nv_bfloat16 *x, int k0, int x_gmem, uint4 *x0,
                         uint4 *x1) {
  if (x_gmem) {
    *x0 = __ldg(reinterpret_cast<const uint4 *>(x + k0));
    *x1 = __ldg(reinterpret_cast<const uint4 *>(x + k0 + 8));
  } else {
    *x0 = *reinterpret_cast<const uint4 *>(x + k0);
    *x1 = *reinterpret_cast<const uint4 *>(x + k0 + 8);
  }
}

__device__ __forceinline__ float tile16(const uint8_t *prow, const uint16_t *srow,
                                       const __nv_bfloat16 *x, int k0,
                                       const float *lut, int x_gmem) {
  uint4 x0;
  uint4 x1;
  load_x16(x, k0, x_gmem, &x0, &x1);
  return tile16_pre(prow, srow, k0, x0, x1, lut);
}

__device__ __forceinline__ void tile16_pair_pre(const uint8_t *p0,
                                               const uint16_t *s0,
                                               const uint8_t *p1,
                                               const uint16_t *s1, int k0,
                                               uint4 xv0, uint4 xv1,
                                               const float *lut, float *a0,
                                               float *a1) {
  const int grp = k0 / kGroup;
  const float sc0 = __half2float(__ushort_as_half(__ldg(s0 + grp)));
  const float sc1 = __half2float(__ushort_as_half(__ldg(s1 + grp)));
  const uint2 pk0 = __ldg(reinterpret_cast<const uint2 *>(p0 + (k0 >> 1)));
  const uint2 pk1 = __ldg(reinterpret_cast<const uint2 *>(p1 + (k0 >> 1)));
  *a0 += dot8(pk0.x, sc0, xv0, lut) + dot8(pk0.y, sc0, xv1, lut);
  *a1 += dot8(pk1.x, sc1, xv0, lut) + dot8(pk1.y, sc1, xv1, lut);
}

__device__ void load_lut(float *lut, int tid) {
  if (tid < 16) {
    lut[tid] = __uint_as_float(kNf4LutBits[tid]);
  }
  __syncthreads();
}

__device__ float row_partial(const uint8_t *pr, const uint16_t *sr,
                             const __nv_bfloat16 *x, int K, const float *lut,
                             int tid, int x_gmem) {
  float acc = 0.f;
  const int vec_end = (K / 16) * 16;
  const int stride = kWarps * 512;
  int k0 = tid * 16;
  for (; k0 + 15 < vec_end; k0 += stride) {
    acc += tile16(pr, sr, x, k0, lut, x_gmem);
  }
  for (int k = vec_end + tid; k < K; k += kBlock) {
    const unsigned byte = __ldg(pr + (k >> 1));
    const unsigned nib = (k & 1) ? (byte >> 4) : (byte & 0xFu);
    const float s =
        __half2float(__ushort_as_half(__ldg(sr + (k / kGroup))));
    const float xv = x_gmem ? __bfloat162float(__ldg(x + k))
                            : __bfloat162float(x[k]);
    acc += lut[nib] * s * xv;
  }
  return acc;
}

template <int NR>
__device__ void acc_rows_k16(const uint8_t *packed, const uint16_t *scale,
                             int packed_stride, int n_groups, int row0, int n,
                             int k0, uint4 x0, uint4 x1, const float *lut,
                             float acc[NR]) {
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    if (r < n) {
      const uint8_t *pr =
          packed + static_cast<size_t>(row0 + r) * packed_stride;
      const uint16_t *sr =
          scale + static_cast<size_t>(row0 + r) * n_groups;
      acc[r] += tile16_pre(pr, sr, k0, x0, x1, lut);
    }
  }
}

template <int NR>
__device__ void acc_rows_tail(const uint8_t *packed, const uint16_t *scale,
                              int packed_stride, int n_groups, int row0, int n,
                              int k, float xv, const float *lut, float acc[NR]) {
  const int grp = k / kGroup;
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    if (r < n) {
      const uint8_t *pr =
          packed + static_cast<size_t>(row0 + r) * packed_stride;
      const uint16_t *sr =
          scale + static_cast<size_t>(row0 + r) * n_groups;
      const unsigned byte = __ldg(pr + (k >> 1));
      const unsigned nib = (k & 1) ? (byte >> 4) : (byte & 0xFu);
      const float s = __half2float(__ushort_as_half(__ldg(sr + grp)));
      acc[r] += lut[nib] * s * xv;
    }
  }
}

template <int NR>
__device__ void row_partial_n(const uint8_t *packed, const uint16_t *scale,
                              int packed_stride, int n_groups, int row0, int n,
                              const __nv_bfloat16 *x, int K, const float *lut,
                              int tid, int x_gmem, float acc[NR]) {
  const int vec_end = (K / 16) * 16;
  const int stride = kWarps * 512;
  int k0 = tid * 16;
  for (; k0 + 15 < vec_end; k0 += stride) {
    uint4 x0;
    uint4 x1;
    load_x16(x, k0, x_gmem, &x0, &x1);
    acc_rows_k16<NR>(packed, scale, packed_stride, n_groups, row0, n, k0, x0,
                     x1, lut, acc);
  }
  for (int k = vec_end + tid; k < K; k += kBlock) {
    const float xv = x_gmem ? __bfloat162float(__ldg(x + k))
                            : __bfloat162float(x[k]);
    acc_rows_tail<NR>(packed, scale, packed_stride, n_groups, row0, n, k, xv,
                      lut, acc);
  }
}

template <int NR>
__device__ void finish_n(float acc[NR], int n, int lane, int warp, int tid,
                         float *part, __nv_bfloat16 *y, int row0,
                         const __nv_bfloat16 *add) {
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    acc[r] = warp_sum(acc[r]);
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      part[warp * NR + r] = acc[r];
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int r = 0; r < n; ++r) {
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        sum += part[w * NR + r];
      }
      if (add != nullptr) {
        sum += __bfloat162float(add[row0 + r]);
      }
      y[row0 + r] = __float2bfloat16(sum);
    }
  }
}

__device__ float finish_row(float acc, int lane, int warp, int tid, float *part) {
  acc = warp_sum(acc);
  if (lane == 0) {
    part[warp] = acc;
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      sum += part[w];
    }
    return sum;
  }
  return 0.f;
}

__device__ const __nv_bfloat16 *prep_rms(const __nv_bfloat16 *x,
                                         const __nv_bfloat16 *rms_w, int K,
                                         float eps, int tid, int lane, int warp,
                                         float *part, float *inv,
                                         __nv_bfloat16 *xh) {
  if (rms_w == nullptr) {
    return x;
  }
  float acc = 0.f;
  for (int i = tid; i < K; i += kBlock) {
    const float v = __bfloat162float(x[i]);
    acc += v * v;
  }
  const float sum = finish_row(acc, lane, warp, tid, part);
  if (tid == 0) {
    *inv = rsqrtf(sum / static_cast<float>(K) + eps);
  }
  __syncthreads();
  const float s = *inv;
  for (int i = tid; i < K; i += kBlock) {
    xh[i] = __float2bfloat16(__bfloat162float(rms_w[i]) * __bfloat162float(x[i]) *
                            s);
  }
  __syncthreads();
  return xh;
}

template <int NR>
__global__ void gemv_splitk(const uint8_t *__restrict__ packed,
                            const uint16_t *__restrict__ scale,
                            const __nv_bfloat16 *__restrict__ x,
                            __nv_bfloat16 *__restrict__ y,
                            const __nv_bfloat16 *__restrict__ add,
                            const __nv_bfloat16 *__restrict__ rms_w, int M,
                            int K, int K_pad, float rms_eps) {
  extern __shared__ __nv_bfloat16 xh[];
  const int row0 = static_cast<int>(blockIdx.x) * NR;
  if (row0 >= M) {
    return;
  }
  int n = M - row0;
  if (n > NR) {
    n = NR;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int tid = warp * 32 + lane;
  __shared__ float lut[16];
  __shared__ float part[kWarps * NR];
  __shared__ float inv;
  load_lut(lut, tid);
  const __nv_bfloat16 *xv =
      prep_rms(x, rms_w, K, rms_eps, tid, lane, warp, part, &inv, xh);
  const int x_gmem = rms_w == nullptr ? 1 : 0;
  const int packed_stride = K_pad / 2;
  const int n_groups = K_pad / kGroup;
  float acc[NR] = {};
  row_partial_n<NR>(packed, scale, packed_stride, n_groups, row0, n, xv, K, lut,
                    tid, x_gmem, acc);
  finish_n<NR>(acc, n, lane, warp, tid, part, y, row0, add);
}

__global__ void gemv_qkv(const uint8_t *__restrict__ pq,
                         const uint16_t *__restrict__ sq,
                         const uint8_t *__restrict__ pk,
                         const uint16_t *__restrict__ sk,
                         const uint8_t *__restrict__ pv,
                         const uint16_t *__restrict__ sv,
                         const __nv_bfloat16 *__restrict__ x,
                         __nv_bfloat16 *__restrict__ yq,
                         __nv_bfloat16 *__restrict__ yk,
                         __nv_bfloat16 *__restrict__ yv,
                         const __nv_bfloat16 *__restrict__ bq,
                         const __nv_bfloat16 *__restrict__ bk,
                         const __nv_bfloat16 *__restrict__ bv,
                         const __nv_bfloat16 *__restrict__ rms_w, int Mq, int Mk,
                         int Mv, int K, int K_pad, float rms_eps) {
  extern __shared__ __nv_bfloat16 xh[];
  const int id = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int tid = warp * 32 + lane;
  __shared__ float lut[16];
  __shared__ float part[kWarps];
  __shared__ float inv;
  load_lut(lut, tid);
  const __nv_bfloat16 *xv =
      prep_rms(x, rms_w, K, rms_eps, tid, lane, warp, part, &inv, xh);
  const int x_gmem = rms_w == nullptr ? 1 : 0;
  const int packed_stride = K_pad / 2;
  const int n_groups = K_pad / kGroup;
  const uint8_t *pr;
  const uint16_t *sr;
  const __nv_bfloat16 *bias;
  __nv_bfloat16 *y;
  int row;
  if (id < Mq) {
    pr = pq;
    sr = sq;
    y = yq;
    bias = bq;
    row = id;
  } else if (id < Mq + Mk) {
    pr = pk;
    sr = sk;
    y = yk;
    bias = bk;
    row = id - Mq;
  } else {
    pr = pv;
    sr = sv;
    y = yv;
    bias = bv;
    row = id - Mq - Mk;
  }
  pr += static_cast<size_t>(row) * packed_stride;
  sr += static_cast<size_t>(row) * n_groups;
  const float acc = row_partial(pr, sr, xv, K, lut, tid, x_gmem);
  const float sum = finish_row(acc, lane, warp, tid, part);
  if (tid == 0) {
    float out = sum;
    if (bias != nullptr) {
      out += __bfloat162float(bias[row]);
    }
    y[row] = __float2bfloat16(out);
  }
}

template <int NR>
__global__ void gemv_swiglu(const uint8_t *__restrict__ pg,
                            const uint16_t *__restrict__ sg,
                            const uint8_t *__restrict__ pu,
                            const uint16_t *__restrict__ su,
                            const __nv_bfloat16 *__restrict__ x,
                            __nv_bfloat16 *__restrict__ y,
                            const __nv_bfloat16 *__restrict__ rms_w, int M,
                            int K, int K_pad, float rms_eps) {
  extern __shared__ __nv_bfloat16 xh[];
  const int row0 = static_cast<int>(blockIdx.x) * NR;
  if (row0 >= M) {
    return;
  }
  int n = M - row0;
  if (n > NR) {
    n = NR;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int tid = warp * 32 + lane;
  __shared__ float lut[16];
  __shared__ float part_g[kWarps * NR];
  __shared__ float part_u[kWarps * NR];
  __shared__ float inv;
  load_lut(lut, tid);
  const __nv_bfloat16 *xv =
      prep_rms(x, rms_w, K, rms_eps, tid, lane, warp, part_g, &inv, xh);
  const int x_gmem = rms_w == nullptr ? 1 : 0;
  const int packed_stride = K_pad / 2;
  const int n_groups = K_pad / kGroup;
  float acc_g[NR] = {};
  float acc_u[NR] = {};
  const int vec_end = (K / 16) * 16;
  const int stride = kWarps * 512;
  int k0 = tid * 16;
  for (; k0 + 15 < vec_end; k0 += stride) {
    uint4 x0;
    uint4 x1;
    load_x16(xv, k0, x_gmem, &x0, &x1);
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      if (r < n) {
        const uint8_t *g_row =
            pg + static_cast<size_t>(row0 + r) * packed_stride;
        const uint8_t *u_row =
            pu + static_cast<size_t>(row0 + r) * packed_stride;
        const uint16_t *gs = sg + static_cast<size_t>(row0 + r) * n_groups;
        const uint16_t *us = su + static_cast<size_t>(row0 + r) * n_groups;
        tile16_pair_pre(g_row, gs, u_row, us, k0, x0, x1, lut, &acc_g[r],
                        &acc_u[r]);
      }
    }
  }
  for (int k = vec_end + tid; k < K; k += kBlock) {
    const float xvv = x_gmem ? __bfloat162float(__ldg(xv + k))
                             : __bfloat162float(xv[k]);
    const int grp = k / kGroup;
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      if (r < n) {
        const uint8_t *g_row =
            pg + static_cast<size_t>(row0 + r) * packed_stride;
        const uint8_t *u_row =
            pu + static_cast<size_t>(row0 + r) * packed_stride;
        const uint16_t *gs = sg + static_cast<size_t>(row0 + r) * n_groups;
        const uint16_t *us = su + static_cast<size_t>(row0 + r) * n_groups;
        const unsigned bg = __ldg(g_row + (k >> 1));
        const unsigned bu = __ldg(u_row + (k >> 1));
        const unsigned ng = (k & 1) ? (bg >> 4) : (bg & 0xFu);
        const unsigned nu = (k & 1) ? (bu >> 4) : (bu & 0xFu);
        const float scg = __half2float(__ushort_as_half(__ldg(gs + grp)));
        const float scu = __half2float(__ushort_as_half(__ldg(us + grp)));
        acc_g[r] += lut[ng] * scg * xvv;
        acc_u[r] += lut[nu] * scu * xvv;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    acc_g[r] = warp_sum(acc_g[r]);
    acc_u[r] = warp_sum(acc_u[r]);
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      part_g[warp * NR + r] = acc_g[r];
      part_u[warp * NR + r] = acc_u[r];
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int r = 0; r < n; ++r) {
      float g = 0.f;
      float u = 0.f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        g += part_g[w * NR + r];
        u += part_u[w * NR + r];
      }
      y[row0 + r] = __float2bfloat16(silu(g) * u);
    }
  }
}

__global__ void rope_kv(__nv_bfloat16 *__restrict__ q, __nv_bfloat16 *__restrict__ k,
                        const __nv_bfloat16 *__restrict__ v,
                        __nv_bfloat16 *__restrict__ k_cache,
                        __nv_bfloat16 *__restrict__ v_cache,
                        const __nv_bfloat16 *__restrict__ cos,
                        const __nv_bfloat16 *__restrict__ sin,
                        const int64_t *__restrict__ position, int n_q, int n_kv,
                        int hd, int max_seq) {
  const int hid = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const int pos = static_cast<int>(*position);
  if (pos < 0 || pos >= max_seq) {
    return;
  }
  const int d2 = hd / 2;
  const __nv_bfloat16 *c = cos + static_cast<size_t>(pos) * hd;
  const __nv_bfloat16 *s = sin + static_cast<size_t>(pos) * hd;
  const bool is_q = hid < n_q;
  const int head = is_q ? hid : hid - n_q;
  __nv_bfloat16 *vec =
      is_q ? q + static_cast<size_t>(head) * hd : k + static_cast<size_t>(head) * hd;
  for (int d = lane; d < d2; d += static_cast<int>(blockDim.x)) {
    const float t0 = __bfloat162float(vec[d]);
    const float t1 = __bfloat162float(vec[d + d2]);
    const float c0 = __bfloat162float(c[d]);
    const float c1 = __bfloat162float(c[d + d2]);
    const float s0 = __bfloat162float(s[d]);
    const float s1 = __bfloat162float(s[d + d2]);
    vec[d] = __float2bfloat16(t0 * c0 + (-t1) * s0);
    vec[d + d2] = __float2bfloat16(t1 * c1 + t0 * s1);
  }
  if (!is_q) {
    __syncthreads();
    const size_t off =
        (static_cast<size_t>(pos) * n_kv + static_cast<size_t>(head)) * hd;
    for (int d = lane; d < hd; d += static_cast<int>(blockDim.x)) {
      k_cache[off + d] = vec[d];
      v_cache[off + d] = v[static_cast<size_t>(head) * hd + d];
    }
  }
}

__device__ float attn_block_sum(float v, int lane, int warp, int tid,
                                float *part) {
  v = warp_sum(v);
  if (lane == 0) {
    part[warp] = v;
  }
  __syncthreads();
  if (tid == 0) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) {
      s += part[w];
    }
    part[0] = s;
  }
  __syncthreads();
  return part[0];
}

__device__ void merge_one_head(const float *m_s, const float *l_s, const float *o_s,
                               __nv_bfloat16 *out, int head, int hd, int n_split,
                               int lane) {
  const float *mh = m_s + static_cast<size_t>(head) * n_split;
  const float *lh = l_s + static_cast<size_t>(head) * n_split;
  float m = -1.0e30f;
  for (int s = lane; s < n_split; s += 32) {
    m = fmaxf(m, mh[s]);
  }
  m = warp_max_all(m);
  float l = 0.f;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.f;
  }
  for (int s = 0; s < n_split; ++s) {
    const float a = __expf(mh[s] - m);
    l += a * lh[s];
    const float *oh = o_s + (static_cast<size_t>(head) * n_split + s) * hd;
    int i = 0;
    for (int d = lane; d < hd; d += 32) {
      acc[i] += a * oh[d];
      ++i;
    }
  }
  __nv_bfloat16 *dst = out + static_cast<size_t>(head) * hd;
  if (l <= 0.f) {
    for (int d = lane; d < hd; d += 32) {
      dst[d] = __float2bfloat16(0.f);
    }
    return;
  }
  const float inv = 1.f / l;
  int i = 0;
  for (int d = lane; d < hd; d += 32) {
    dst[d] = __float2bfloat16(acc[i] * inv);
    ++i;
  }
}

__global__ void attn_flash(const __nv_bfloat16 *__restrict__ q,
                           const __nv_bfloat16 *__restrict__ k_cache,
                           const __nv_bfloat16 *__restrict__ v_cache,
                           __nv_bfloat16 *__restrict__ out, float *m_s, float *l_s,
                           float *o_s, const int32_t *__restrict__ valid_len,
                           const __nv_bfloat16 *__restrict__ k_act,
                           const __nv_bfloat16 *__restrict__ v_act,
                           const __nv_bfloat16 *__restrict__ cos,
                           const __nv_bfloat16 *__restrict__ sin,
                           const int64_t *__restrict__ position, int n_q, int n_kv,
                           int hd, int n_split, int max_seq, float scale) {
  const int head = static_cast<int>(blockIdx.x) / n_split;
  const int split = static_cast<int>(blockIdx.x) % n_split;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int n_rep = n_q / n_kv;
  const int kv_head = head / n_rep;
  const int vl = *valid_len;
  const int d2 = hd / 2;
  (void)out;
  __shared__ float qrot[256];
  __shared__ float krot[256];
  __shared__ float vcur[256];
  __shared__ float part[kAttnWarps];
  const __nv_bfloat16 *qh = q + static_cast<size_t>(head) * hd;
  int pos_tok = -1;
  int fuse_kv = 0;
  if (cos != nullptr && sin != nullptr && position != nullptr) {
    pos_tok = static_cast<int>(*position);
    if (pos_tok >= 0 && pos_tok < max_seq) {
      const __nv_bfloat16 *c = cos + static_cast<size_t>(pos_tok) * hd;
      const __nv_bfloat16 *s = sin + static_cast<size_t>(pos_tok) * hd;
      for (int d = tid; d < d2; d += kAttnBlock) {
        const float t0 = __bfloat162float(qh[d]);
        const float t1 = __bfloat162float(qh[d + d2]);
        qrot[d] = t0 * __bfloat162float(c[d]) + (-t1) * __bfloat162float(s[d]);
        qrot[d + d2] = t1 * __bfloat162float(c[d + d2]) + t0 * __bfloat162float(s[d + d2]);
      }
      if (k_act != nullptr && v_act != nullptr && k_cache != nullptr &&
          v_cache != nullptr) {
        fuse_kv = 1;
        const __nv_bfloat16 *ka = k_act + static_cast<size_t>(kv_head) * hd;
        const __nv_bfloat16 *va = v_act + static_cast<size_t>(kv_head) * hd;
        for (int d = tid; d < d2; d += kAttnBlock) {
          const float t0 = __bfloat162float(ka[d]);
          const float t1 = __bfloat162float(ka[d + d2]);
          krot[d] = t0 * __bfloat162float(c[d]) + (-t1) * __bfloat162float(s[d]);
          krot[d + d2] =
              t1 * __bfloat162float(c[d + d2]) + t0 * __bfloat162float(s[d + d2]);
        }
        for (int d = tid; d < hd; d += kAttnBlock) {
          vcur[d] = __bfloat162float(va[d]);
        }
        __syncthreads();
        if (head == kv_head * n_rep && split == pos_tok % n_split) {
          __nv_bfloat16 *kc = const_cast<__nv_bfloat16 *>(k_cache) +
                              (static_cast<size_t>(pos_tok) * n_kv + kv_head) * hd;
          __nv_bfloat16 *vc = const_cast<__nv_bfloat16 *>(v_cache) +
                              (static_cast<size_t>(pos_tok) * n_kv + kv_head) * hd;
          for (int d = tid; d < hd; d += kAttnBlock) {
            kc[d] = __float2bfloat16(krot[d]);
            vc[d] = va[d];
          }
        }
      }
    } else {
      for (int d = tid; d < hd; d += kAttnBlock) {
        qrot[d] = __bfloat162float(qh[d]);
      }
    }
  } else {
    for (int d = tid; d < hd; d += kAttnBlock) {
      qrot[d] = __bfloat162float(qh[d]);
    }
  }
  __syncthreads();

  const size_t slot = static_cast<size_t>(head) * n_split + split;
  float m = -1.0e30f;
  float l = 0.f;
  float acc[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    acc[i] = 0.f;
  }
  for (int t = split; t < vl; t += n_split) {
    const int use_act = (fuse_kv != 0 && t == pos_tok) ? 1 : 0;
    const size_t row = (static_cast<size_t>(t) * n_kv + kv_head) * hd;
    const __nv_bfloat16 *kh = k_cache + row;
    const __nv_bfloat16 *vh = v_cache + row;
    float dot = 0.f;
    for (int d = tid; d < hd; d += kAttnBlock) {
      const float k = use_act ? krot[d] : __bfloat162float(__ldg(kh + d));
      dot += qrot[d] * k;
    }
    dot = attn_block_sum(dot, lane, warp, tid, part) * scale;
    const float m2 = fmaxf(m, dot);
    const float alpha = __expf(m - m2);
    const float e = __expf(dot - m2);
    l = l * alpha + e;
    int i = 0;
    for (int d = tid; d < hd; d += kAttnBlock) {
      const float vv = use_act ? vcur[d] : __bfloat162float(__ldg(vh + d));
      acc[i] = acc[i] * alpha + e * vv;
      ++i;
    }
    m = m2;
  }
  if (tid == 0) {
    m_s[slot] = m;
    l_s[slot] = l;
  }
  float *oh = o_s + slot * hd;
  int i = 0;
  for (int d = tid; d < hd; d += kAttnBlock) {
    oh[d] = acc[i];
    ++i;
  }
}

__global__ void attn_merge(const float *__restrict__ m_s,
                           const float *__restrict__ l_s,
                           const float *__restrict__ o_s,
                           __nv_bfloat16 *__restrict__ out, int hd,
                           int n_split) {
  merge_one_head(m_s, l_s, o_s, out, static_cast<int>(blockIdx.x), hd, n_split,
                 static_cast<int>(threadIdx.x));
}

__global__ void rms_vec(const __nv_bfloat16 *__restrict__ x,
                        const __nv_bfloat16 *__restrict__ w,
                        __nv_bfloat16 *__restrict__ y, int n, float eps) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int tid = warp * 32 + lane;
  __shared__ float part[kWarps];
  __shared__ float inv;
  float acc = 0.f;
  for (int i = tid; i < n; i += kBlock) {
    const float v = __bfloat162float(x[i]);
    acc += v * v;
  }
  const float sum = finish_row(acc, lane, warp, tid, part);
  if (tid == 0) {
    inv = rsqrtf(sum / static_cast<float>(n) + eps);
  }
  __syncthreads();
  const float s = inv;
  for (int i = tid; i < n; i += kBlock) {
    y[i] = __float2bfloat16(__bfloat162float(w[i]) * __bfloat162float(x[i]) * s);
  }
}

int k_pad_ok(int K, int K_pad) {
  if (K < 1 || K_pad < K) {
    return 0;
  }
  const int want = kGroup * ((K + kGroup - 1) / kGroup);
  return K_pad == want;
}

int aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

size_t rms_dyn(const void *rms_w, int K) {
  return rms_w == nullptr ? 0
                          : static_cast<size_t>(K) * sizeof(__nv_bfloat16);
}

int pick_nr(int M) {
  const char *e = std::getenv("CHR_NF4_GEMV_NR");
  if (e != nullptr && e[0] != '\0') {
    const int v = std::atoi(e);
    if (v == 1 || v == 2 || v == 4 || v == 8) {
      return v;
    }
  }
  if (M >= 4096) {
    return 4;
  }
  return 1;
}

unsigned grid_nr(int M, int nr) {
  return static_cast<unsigned>((M + nr - 1) / nr);
}

} // namespace

extern "C" int chr_nf4_gemv(const chr_nf4_dev_t *w, const void *x, void *y,
                            void *stream, const void *add, const void *rms_w,
                            float rms_eps) {
  if (w == nullptr || x == nullptr || y == nullptr) {
    return -1;
  }
  if (w->M < 1 || w->K < 1 || w->packed == nullptr || w->scale == nullptr) {
    return -3;
  }
  if (!k_pad_ok(w->K, w->K_pad)) {
    return -4;
  }
  if (!aligned16(w->packed) || !aligned16(x)) {
    return -5;
  }
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const auto *xp = static_cast<const __nv_bfloat16 *>(x);
  auto *yp = static_cast<__nv_bfloat16 *>(y);
  const int nr = pick_nr(w->M);
  const dim3 grid(grid_nr(w->M, nr));
  const dim3 block(static_cast<unsigned>(kBlock));
  const size_t dyn = rms_dyn(rms_w, w->K);
  if (nr == 8) {
    gemv_splitk<8><<<grid, block, dyn, st>>>(
        w->packed, w->scale, xp, yp, static_cast<const __nv_bfloat16 *>(add),
        static_cast<const __nv_bfloat16 *>(rms_w), w->M, w->K, w->K_pad,
        rms_eps);
  } else if (nr == 4) {
    gemv_splitk<4><<<grid, block, dyn, st>>>(
        w->packed, w->scale, xp, yp, static_cast<const __nv_bfloat16 *>(add),
        static_cast<const __nv_bfloat16 *>(rms_w), w->M, w->K, w->K_pad,
        rms_eps);
  } else if (nr == 2) {
    gemv_splitk<2><<<grid, block, dyn, st>>>(
        w->packed, w->scale, xp, yp, static_cast<const __nv_bfloat16 *>(add),
        static_cast<const __nv_bfloat16 *>(rms_w), w->M, w->K, w->K_pad,
        rms_eps);
  } else {
    gemv_splitk<1><<<grid, block, dyn, st>>>(
        w->packed, w->scale, xp, yp, static_cast<const __nv_bfloat16 *>(add),
        static_cast<const __nv_bfloat16 *>(rms_w), w->M, w->K, w->K_pad,
        rms_eps);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}

extern "C" int chr_nf4_gemv_qkv(const chr_nf4_dev_t *q, const chr_nf4_dev_t *k,
                                const chr_nf4_dev_t *v, const void *x, void *yq,
                                void *yk, void *yv, const void *bq,
                                const void *bk, const void *bv, void *stream,
                                const void *rms_w, float rms_eps) {
  if (q == nullptr || k == nullptr || v == nullptr || x == nullptr ||
      yq == nullptr || yk == nullptr || yv == nullptr) {
    return -1;
  }
  if (q->M < 1 || k->M < 1 || v->M < 1 || q->K < 1) {
    return -3;
  }
  if (k->K != q->K || v->K != q->K || k->K_pad != q->K_pad ||
      v->K_pad != q->K_pad) {
    return -3;
  }
  if (!k_pad_ok(q->K, q->K_pad)) {
    return -4;
  }
  if (!aligned16(q->packed) || !aligned16(k->packed) || !aligned16(v->packed) ||
      !aligned16(x)) {
    return -5;
  }
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const auto *xp = static_cast<const __nv_bfloat16 *>(x);
  const dim3 grid(static_cast<unsigned>(q->M + k->M + v->M));
  const dim3 block(static_cast<unsigned>(kBlock));
  const size_t dyn = rms_dyn(rms_w, q->K);
  gemv_qkv<<<grid, block, dyn, st>>>(
      q->packed, q->scale, k->packed, k->scale, v->packed, v->scale, xp,
      static_cast<__nv_bfloat16 *>(yq), static_cast<__nv_bfloat16 *>(yk),
      static_cast<__nv_bfloat16 *>(yv),
      static_cast<const __nv_bfloat16 *>(bq),
      static_cast<const __nv_bfloat16 *>(bk),
      static_cast<const __nv_bfloat16 *>(bv),
      static_cast<const __nv_bfloat16 *>(rms_w), q->M, k->M, v->M, q->K,
      q->K_pad, rms_eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}

extern "C" int chr_nf4_gemv_swiglu(const chr_nf4_dev_t *gate,
                                   const chr_nf4_dev_t *up, const void *x,
                                   void *y, void *stream, const void *rms_w,
                                   float rms_eps) {
  if (gate == nullptr || up == nullptr || x == nullptr || y == nullptr) {
    return -1;
  }
  if (gate->M < 1 || gate->K < 1 || up->M != gate->M || up->K != gate->K ||
      up->K_pad != gate->K_pad) {
    return -3;
  }
  if (!k_pad_ok(gate->K, gate->K_pad)) {
    return -4;
  }
  if (!aligned16(gate->packed) || !aligned16(up->packed) || !aligned16(x)) {
    return -5;
  }
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int nr = pick_nr(gate->M);
  const dim3 grid(grid_nr(gate->M, nr));
  const dim3 block(static_cast<unsigned>(kBlock));
  const size_t dyn = rms_dyn(rms_w, gate->K);
  if (nr == 8) {
    gemv_swiglu<8><<<grid, block, dyn, st>>>(
        gate->packed, gate->scale, up->packed, up->scale,
        static_cast<const __nv_bfloat16 *>(x), static_cast<__nv_bfloat16 *>(y),
        static_cast<const __nv_bfloat16 *>(rms_w), gate->M, gate->K,
        gate->K_pad, rms_eps);
  } else if (nr == 4) {
    gemv_swiglu<4><<<grid, block, dyn, st>>>(
        gate->packed, gate->scale, up->packed, up->scale,
        static_cast<const __nv_bfloat16 *>(x), static_cast<__nv_bfloat16 *>(y),
        static_cast<const __nv_bfloat16 *>(rms_w), gate->M, gate->K,
        gate->K_pad, rms_eps);
  } else if (nr == 2) {
    gemv_swiglu<2><<<grid, block, dyn, st>>>(
        gate->packed, gate->scale, up->packed, up->scale,
        static_cast<const __nv_bfloat16 *>(x), static_cast<__nv_bfloat16 *>(y),
        static_cast<const __nv_bfloat16 *>(rms_w), gate->M, gate->K,
        gate->K_pad, rms_eps);
  } else {
    gemv_swiglu<1><<<grid, block, dyn, st>>>(
        gate->packed, gate->scale, up->packed, up->scale,
        static_cast<const __nv_bfloat16 *>(x), static_cast<__nv_bfloat16 *>(y),
        static_cast<const __nv_bfloat16 *>(rms_w), gate->M, gate->K,
        gate->K_pad, rms_eps);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}

extern "C" int chr_nf4_rope_kv(void *q, void *k, const void *v, void *k_cache,
                               void *v_cache, const void *cos, const void *sin,
                               const void *position, int32_t n_q, int32_t n_kv,
                               int32_t hd, int32_t max_seq, void *stream) {
  if (q == nullptr || k == nullptr || v == nullptr || k_cache == nullptr ||
      v_cache == nullptr || cos == nullptr || sin == nullptr ||
      position == nullptr) {
    return -1;
  }
  if (n_q < 1 || n_kv < 1 || hd < 2 || (hd & 1) != 0 || max_seq < 1) {
    return -3;
  }
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const dim3 grid(static_cast<unsigned>(n_q + n_kv));
  const dim3 block(32);
  rope_kv<<<grid, block, 0, st>>>(
      static_cast<__nv_bfloat16 *>(q), static_cast<__nv_bfloat16 *>(k),
      static_cast<const __nv_bfloat16 *>(v),
      static_cast<__nv_bfloat16 *>(k_cache),
      static_cast<__nv_bfloat16 *>(v_cache),
      static_cast<const __nv_bfloat16 *>(cos),
      static_cast<const __nv_bfloat16 *>(sin),
      static_cast<const int64_t *>(position), n_q, n_kv, hd, max_seq);
  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}

extern "C" int chr_nf4_attn(const void *q, const void *k, const void *v,
                            void *out, const void *valid_len, float *ws,
                            int32_t n_q, int32_t n_kv, int32_t hd,
                            int32_t n_split, float scale, void *stream,
                            const void *k_act, const void *v_act, const void *cos,
                            const void *sin, const void *position,
                            int32_t max_seq) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      valid_len == nullptr || ws == nullptr) {
    return -1;
  }
  if (n_q < 1 || n_kv < 1 || hd < 1 || hd > 256 || (n_q % n_kv) != 0 ||
      n_split < 1 || n_split > 64) {
    return -3;
  }
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int nml = n_q * n_split;
  float *m_s = ws;
  float *l_s = ws + nml;
  float *o_s = ws + 2 * nml;
  auto *qp = static_cast<const __nv_bfloat16 *>(q);
  auto *kp = static_cast<const __nv_bfloat16 *>(k);
  auto *vp = static_cast<const __nv_bfloat16 *>(v);
  auto *op = static_cast<__nv_bfloat16 *>(out);
  auto *vl = static_cast<const int32_t *>(valid_len);
  auto *ka = static_cast<const __nv_bfloat16 *>(k_act);
  auto *va = static_cast<const __nv_bfloat16 *>(v_act);
  auto *cp = static_cast<const __nv_bfloat16 *>(cos);
  auto *sp = static_cast<const __nv_bfloat16 *>(sin);
  auto *pp = static_cast<const int64_t *>(position);
  const dim3 grid(static_cast<unsigned>(nml));
  const dim3 flash_block(static_cast<unsigned>(kAttnBlock));
  const dim3 merge_block(32);
  // Cooperative this_grid().sync() merge returns wrong greedy ids on WDDM
  // (medium llama). Two-launch merge is the correct path; occupancy comes
  // from 4-warp flash CTAs (hd=128 is one pass), not from dropping merge.
  attn_flash<<<grid, flash_block, 0, st>>>(qp, kp, vp, op, m_s, l_s, o_s, vl, ka,
                                          va, cp, sp, pp, n_q, n_kv, hd, n_split,
                                          max_seq, scale);
  attn_merge<<<dim3(static_cast<unsigned>(n_q)), merge_block, 0, st>>>(
      m_s, l_s, o_s, op, hd, n_split);
  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}

extern "C" int chr_nf4_rms(const void *x, const void *w, void *y, int32_t n,
                           float eps, void *stream) {
  if (x == nullptr || w == nullptr || y == nullptr) {
    return -1;
  }
  if (n < 1) {
    return -3;
  }
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const dim3 grid(1);
  const dim3 block(static_cast<unsigned>(kBlock));
  rms_vec<<<grid, block, 0, st>>>(static_cast<const __nv_bfloat16 *>(x),
                                  static_cast<const __nv_bfloat16 *>(w),
                                  static_cast<__nv_bfloat16 *>(y), n, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}
