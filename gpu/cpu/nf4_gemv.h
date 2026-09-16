/* CPU fused NF4 GEMV. Same CHR0 layout as the CUDA kernel / nf4_oracle:

   group 64, low nibble = even K, decode = float32(LUT[nib]) * float32(scale).
   Never writes W_hat. x is float32 [N, K] (N=1 may be [K]); y is float32 [M, N].

   Returns 0 on success. Negative: -1 null, -2 N not in 1..32, -3 bad M/K,
   -4 K_pad != 64*ceil(K/64). */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int chr_nf4_gemv_cpu(const uint8_t *packed, const uint16_t *scale, const float *x,
                     float *y, int32_t M, int32_t K, int32_t K_pad, int32_t N,
                     int32_t nthreads);

/* Affine INT4 CPU GEMV (docs/spec/i4c.md). Group 64, K_pad = 64*ceil(K/64).
   N=1: dequant int4 → float, FMA with float x. Same return codes; -4 if K_pad
   is not 64*ceil(K/64). */
int chr_i4c_gemv_cpu(const uint8_t *packed, const uint16_t *scale, const float *x,
                     float *y, int32_t M, int32_t K, int32_t K_pad, int32_t N,
                     int32_t nthreads);

/* "avx2" or "scalar". Compile-time ISA, not a tok/s claim. */
const char *chr_nf4_cpu_isa(void);

#ifdef __cplusplus
}
#endif
