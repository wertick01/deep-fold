/* Frozen ABI for wave 2. Field layout is the contract between loader and kernel.
 * Do not reorder or widen fields. Agent 2 implements chr_nf4_gemm. */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t M;             /* logical out_features */
  int32_t K;             /* logical in_features */
  int32_t K_pad;         /* 64 * ceil(K / 64) */
  const uint8_t *packed;  /* device, [M, K_pad/2], row-major uint8 */
  const uint16_t *scale; /* device, [M, K_pad/64], IEEE binary16 bits */
} chr_nf4_dev_t;

/* x: __nv_bfloat16 [K, N] row-major; y: __nv_bfloat16 [M, N].
 * N in [1, 16]: N=1 decode tile (BM=128, BN=8, BK=256); N=2..16 prefill
 * (BM=64, BN=16 pad, BK=128, epilogue masks n>=N). No CUDA-core GEMV.
 * N>16 returns -2 — this call does not slice; the host chunks into N<=16.
 * (-2 is no longer "any N!=1".)
 * stream is cudaStream_t (0 / nullptr = default stream).
 * Returns 0 on success. Negative: -1 null, -2 N not in [1,16], -3 bad dims,
 * -4 K_pad, -5 alignment (packed needs 16B), -6 CUDA launch. */
int chr_nf4_gemm(const chr_nf4_dev_t *w, const void *x, void *y, int32_t N,
                 void *stream);

/* VQ 2×8. Do not reorder. K_pad = 8 * ceil(K / 8).
 * index: uint8 [M, K_pad/8, 2], last axis = codebook (i1, i2).
 * book: fp16 bits [2, 256, 8]. Reconstruct g = C1[i1] + C2[i2] in float32. */
typedef struct {
  int32_t M;
  int32_t K;
  int32_t K_pad;
  const uint8_t *index;  /* device */
  const uint16_t *book; /* device, IEEE binary16 bits */
} chr_vq_dev_t;

int chr_vq_gemm(const chr_vq_dev_t *w, const void *x, void *y, int32_t N,
                 void *stream);

#ifdef __cplusplus
}
#endif
