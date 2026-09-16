#include "nf4_gemv.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define CHR_INLINE __forceinline
#else
#define CHR_INLINE inline __attribute__((always_inline))
#endif
#if defined(__AVX2__) || defined(CHR_NF4_FORCE_AVX2)
#include <immintrin.h>
#define CHR_NF4_AVX2 1
#else
#define CHR_NF4_AVX2 0
#endif

namespace {

constexpr int kGroup = 64;
constexpr int kI4cGroup = 64;
constexpr int kLiveMaxN = 32;
// 5950X: L1D 32 KiB / L2 512 KiB / L3 32 MiB per CCD. N=1 down_proj x is
// 108 KiB (L2). A 4-row packed panel is 4×13.5 KiB; plus x fits in L2.
// Sequential K inside the panel so packed streams; x stays in ymm/L2.

// docs/spec/nf4.md SS1, binary32 bits (same table as gpu.tests.nf4_oracle).
alignas(32) constexpr uint32_t kLutBits[16] = {
    0xBF800000u, 0xBF3239B1u, 0xBF066B30u, 0xBECA32A0u, 0xBE91A24Du, 0xBE3D353Fu,
    0xBDBA7871u, 0x00000000u, 0x3DA2FAFFu, 0x3E24CAE3u, 0x3E7C04DDu, 0x3EAD033Au,
    0x3EE1A4B8u, 0x3F1007ABu, 0x3F3913B3u, 0x3F800000u,
};

inline float lut_f32(int nib) {
  float v;
  std::memcpy(&v, &kLutBits[nib & 15], sizeof(v));
  return v;
}

inline float fp16_to_f32(uint16_t h) {
#if CHR_NF4_AVX2
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
#else
  const uint32_t s = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  const uint32_t e = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
  const uint32_t m = static_cast<uint32_t>(h) & 0x3FFu;
  uint32_t bits;
  if (e == 0) {
    if (m == 0) {
      bits = s;
    } else {
      uint32_t mm = m;
      uint32_t ee = 127 - 15 + 1;
      while ((mm & 0x400u) == 0) {
        mm <<= 1;
        ee--;
      }
      bits = s | (ee << 23) | ((mm & 0x3FFu) << 13);
    }
  } else if (e == 31) {
    bits = s | 0x7F800000u | (m << 13);
  } else {
    bits = s | ((e + (127 - 15)) << 23) | (m << 13);
  }
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
#endif
}

inline int nibble_at(const uint8_t *row, int k) {
  const uint8_t b = row[k >> 1];
  return (k & 1) ? (b >> 4) : (b & 0x0F);
}

#if !CHR_NF4_AVX2
void gemv_rows_scalar(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                      int n_groups, const float *x, float *y, int M, int K, int N,
                      int m0, int m1) {
  (void)M;
  for (int m = m0; m < m1; ++m) {
    const uint8_t *row = packed + static_cast<int64_t>(m) * packed_stride;
    const uint16_t *sc = scale + static_cast<int64_t>(m) * n_groups;
    for (int n = 0; n < N; ++n) {
      const float *xn = x + static_cast<int64_t>(n) * K;
      float acc = 0.f;
      int k = 0;
      for (int g = 0; g < n_groups && k < K; ++g) {
        const float s = fp16_to_f32(sc[g]);
        const int k_end = (k + kGroup <= K) ? k + kGroup : K;
        for (; k < k_end; ++k) {
          acc += lut_f32(nibble_at(row, k)) * s * xn[k];
        }
      }
      y[static_cast<int64_t>(m) * N + n] = acc;
    }
  }
}
#endif

#if CHR_NF4_AVX2

CHR_INLINE __m256 lut_load_lo() {
  return _mm256_load_ps(reinterpret_cast<const float *>(kLutBits));
}

CHR_INLINE __m256 lut_load_hi() {
  return _mm256_load_ps(reinterpret_cast<const float *>(kLutBits) + 8);
}

CHR_INLINE __m256 lut_lookup(__m256i idx, __m256 lut_lo, __m256 lut_hi) {
  const __m256i seven = _mm256_set1_epi32(7);
  const __m256i sel = _mm256_and_si256(idx, seven);
  const __m256 a = _mm256_permutevar8x32_ps(lut_lo, sel);
  const __m256 b = _mm256_permutevar8x32_ps(lut_hi, sel);
  const __m256 mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx, seven));
  return _mm256_blendv_ps(a, b, mask);
}

CHR_INLINE float hsum256(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

CHR_INLINE void dequant16(const uint8_t *p, __m256 vscale, __m256 lut_lo, __m256 lut_hi,
                          __m256 *w0, __m256 *w1) {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  const __m128i mask_hi = _mm_set1_epi8(static_cast<char>(0xF0));
  const __m128i lo = _mm_and_si128(raw, mask_lo);
  const __m128i hi =
      _mm_and_si128(_mm_srli_epi16(_mm_and_si128(raw, mask_hi), 4), mask_lo);
  const __m128i nib = _mm_unpacklo_epi8(lo, hi);
  *w0 = _mm256_mul_ps(lut_lookup(_mm256_cvtepu8_epi32(nib), lut_lo, lut_hi), vscale);
  *w1 = _mm256_mul_ps(
      lut_lookup(_mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)), lut_lo, lut_hi), vscale);
}

// 4 rows × 16 K. x0/x1 stay in ymm; 8 independent FMA chains (Zen3 FMA lat=4).
CHR_INLINE void acc4_16(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2,
                        const uint8_t *p3, __m256 vs0, __m256 vs1, __m256 vs2, __m256 vs3,
                        __m256 x0, __m256 x1, __m256 lut_lo, __m256 lut_hi, __m256 &a0l,
                        __m256 &a0h, __m256 &a1l, __m256 &a1h, __m256 &a2l, __m256 &a2h,
                        __m256 &a3l, __m256 &a3h) {
  __m256 w0, w1;
  dequant16(p0, vs0, lut_lo, lut_hi, &w0, &w1);
  a0l = _mm256_fmadd_ps(w0, x0, a0l);
  a0h = _mm256_fmadd_ps(w1, x1, a0h);
  dequant16(p1, vs1, lut_lo, lut_hi, &w0, &w1);
  a1l = _mm256_fmadd_ps(w0, x0, a1l);
  a1h = _mm256_fmadd_ps(w1, x1, a1h);
  dequant16(p2, vs2, lut_lo, lut_hi, &w0, &w1);
  a2l = _mm256_fmadd_ps(w0, x0, a2l);
  a2h = _mm256_fmadd_ps(w1, x1, a2h);
  dequant16(p3, vs3, lut_lo, lut_hi, &w0, &w1);
  a3l = _mm256_fmadd_ps(w0, x0, a3l);
  a3h = _mm256_fmadd_ps(w1, x1, a3h);
}

CHR_INLINE void acc4_group(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2,
                           const uint8_t *p3, __m256 vs0, __m256 vs1, __m256 vs2,
                           __m256 vs3, const float *xk, __m256 lut_lo, __m256 lut_hi,
                           __m256 &a0l, __m256 &a0h, __m256 &a1l, __m256 &a1h, __m256 &a2l,
                           __m256 &a2h, __m256 &a3l, __m256 &a3h) {
  _mm_prefetch(reinterpret_cast<const char *>(p0 + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char *>(p1 + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char *>(p2 + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char *>(p3 + 64), _MM_HINT_T0);
  for (int off = 0; off < 32; off += 8) {
    const __m256 x0 = _mm256_loadu_ps(xk + off * 2);
    const __m256 x1 = _mm256_loadu_ps(xk + off * 2 + 8);
    acc4_16(p0 + off, p1 + off, p2 + off, p3 + off, vs0, vs1, vs2, vs3, x0, x1, lut_lo,
            lut_hi, a0l, a0h, a1l, a1h, a2l, a2h, a3l, a3h);
  }
}

CHR_INLINE void acc1_group(const uint8_t *p, __m256 vscale, const float *xk, __m256 lut_lo,
                           __m256 lut_hi, __m256 &a0, __m256 &a1, __m256 &a2, __m256 &a3) {
  _mm_prefetch(reinterpret_cast<const char *>(p + 64), _MM_HINT_T0);
  __m256 w0, w1;
  dequant16(p + 0, vscale, lut_lo, lut_hi, &w0, &w1);
  a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 0), a0);
  a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 8), a1);
  dequant16(p + 8, vscale, lut_lo, lut_hi, &w0, &w1);
  a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 16), a2);
  a3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 24), a3);
  dequant16(p + 16, vscale, lut_lo, lut_hi, &w0, &w1);
  a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 32), a0);
  a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 40), a1);
  dequant16(p + 24, vscale, lut_lo, lut_hi, &w0, &w1);
  a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 48), a2);
  a3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 56), a3);
}

void gemv_n1_avx2(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                  int n_groups, const float *x, float *y, int K, int m0, int m1) {
  const __m256 lut_lo = lut_load_lo();
  const __m256 lut_hi = lut_load_hi();
  const int k_full = (K / kGroup) * kGroup;
  int m = m0;
  for (; m + 3 < m1; m += 4) {
    const uint8_t *r0 = packed + static_cast<int64_t>(m) * packed_stride;
    const uint8_t *r1 = r0 + packed_stride;
    const uint8_t *r2 = r1 + packed_stride;
    const uint8_t *r3 = r2 + packed_stride;
    const uint16_t *s0 = scale + static_cast<int64_t>(m) * n_groups;
    const uint16_t *s1 = s0 + n_groups;
    const uint16_t *s2 = s1 + n_groups;
    const uint16_t *s3 = s2 + n_groups;
    if (m + 7 < m1) {
      _mm_prefetch(reinterpret_cast<const char *>(r3 + packed_stride), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char *>(r3 + 2 * packed_stride), _MM_HINT_T0);
    }
    __m256 a0l = _mm256_setzero_ps();
    __m256 a0h = _mm256_setzero_ps();
    __m256 a1l = _mm256_setzero_ps();
    __m256 a1h = _mm256_setzero_ps();
    __m256 a2l = _mm256_setzero_ps();
    __m256 a2h = _mm256_setzero_ps();
    __m256 a3l = _mm256_setzero_ps();
    __m256 a3h = _mm256_setzero_ps();
    int k = 0;
    int g = 0;
    for (; k < k_full; k += kGroup, ++g) {
      acc4_group(r0 + (k >> 1), r1 + (k >> 1), r2 + (k >> 1), r3 + (k >> 1),
                 _mm256_set1_ps(fp16_to_f32(s0[g])), _mm256_set1_ps(fp16_to_f32(s1[g])),
                 _mm256_set1_ps(fp16_to_f32(s2[g])), _mm256_set1_ps(fp16_to_f32(s3[g])),
                 x + k, lut_lo, lut_hi, a0l, a0h, a1l, a1h, a2l, a2h, a3l, a3h);
    }
    float t0 = hsum256(_mm256_add_ps(a0l, a0h));
    float t1 = hsum256(_mm256_add_ps(a1l, a1h));
    float t2 = hsum256(_mm256_add_ps(a2l, a2h));
    float t3 = hsum256(_mm256_add_ps(a3l, a3h));
    if (k < K) {
      const float g0 = fp16_to_f32(s0[k / kGroup]);
      const float g1 = fp16_to_f32(s1[k / kGroup]);
      const float g2 = fp16_to_f32(s2[k / kGroup]);
      const float g3 = fp16_to_f32(s3[k / kGroup]);
      for (int kt = k; kt < K; ++kt) {
        t0 += lut_f32(nibble_at(r0, kt)) * g0 * x[kt];
        t1 += lut_f32(nibble_at(r1, kt)) * g1 * x[kt];
        t2 += lut_f32(nibble_at(r2, kt)) * g2 * x[kt];
        t3 += lut_f32(nibble_at(r3, kt)) * g3 * x[kt];
      }
    }
    y[m] = t0;
    y[m + 1] = t1;
    y[m + 2] = t2;
    y[m + 3] = t3;
  }
  for (; m < m1; ++m) {
    const uint8_t *row = packed + static_cast<int64_t>(m) * packed_stride;
    const uint16_t *sc = scale + static_cast<int64_t>(m) * n_groups;
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int k = 0;
    int g = 0;
    for (; k < k_full; k += kGroup, ++g) {
      acc1_group(row + (k >> 1), _mm256_set1_ps(fp16_to_f32(sc[g])), x + k, lut_lo, lut_hi,
                 a0, a1, a2, a3);
    }
    float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    if (k < K) {
      const float sg = fp16_to_f32(sc[k / kGroup]);
      for (; k < K; ++k) {
        s += lut_f32(nibble_at(row, k)) * sg * x[k];
      }
    }
    y[m] = s;
  }
}

void gemv_n_avx2(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                 int n_groups, const float *x, float *y, int K, int N, int m0, int m1) {
  const __m256 lut_lo = lut_load_lo();
  const __m256 lut_hi = lut_load_hi();
  const int k_full = (K / kGroup) * kGroup;
  alignas(32) __m256 accv[kLiveMaxN];
  for (int m = m0; m < m1; ++m) {
    const uint8_t *row = packed + static_cast<int64_t>(m) * packed_stride;
    const uint16_t *sc = scale + static_cast<int64_t>(m) * n_groups;
    for (int n = 0; n < N; ++n) {
      accv[n] = _mm256_setzero_ps();
    }
    int k = 0;
    int g = 0;
    for (; k < k_full; k += kGroup, ++g) {
      const __m256 vscale = _mm256_set1_ps(fp16_to_f32(sc[g]));
      const uint8_t *p = row + (k >> 1);
      for (int off = 0; off < 32; off += 8) {
        __m256 w0, w1;
        dequant16(p + off, vscale, lut_lo, lut_hi, &w0, &w1);
        const int kk = k + off * 2;
        for (int n = 0; n < N; ++n) {
          const float *xn = x + static_cast<int64_t>(n) * K;
          accv[n] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xn + kk), accv[n]);
          accv[n] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xn + kk + 8), accv[n]);
        }
      }
    }
    for (int n = 0; n < N; ++n) {
      float s = hsum256(accv[n]);
      if (k < K) {
        const float *xn = x + static_cast<int64_t>(n) * K;
        const float scale_g = fp16_to_f32(sc[k / kGroup]);
        for (int kt = k; kt < K; ++kt) {
          s += lut_f32(nibble_at(row, kt)) * scale_g * xn[kt];
        }
      }
      y[static_cast<int64_t>(m) * N + n] = s;
    }
    (void)n_groups;
  }
}

#endif  // CHR_NF4_AVX2

void gemv_rows(const uint8_t *packed, int packed_stride, const uint16_t *scale,
               int n_groups, const float *x, float *y, int M, int K, int N, int m0,
               int m1) {
#if CHR_NF4_AVX2
  if (N == 1) {
    gemv_n1_avx2(packed, packed_stride, scale, n_groups, x, y, K, m0, m1);
    return;
  }
  gemv_n_avx2(packed, packed_stride, scale, n_groups, x, y, K, N, m0, m1);
#else
  gemv_rows_scalar(packed, packed_stride, scale, n_groups, x, y, M, K, N, m0, m1);
#endif
}

#if CHR_NF4_AVX2
// Affine INT4: same FMA skeleton as NF4, cheaper dequant (cvtepi8, no LUT).
// Accumulators stay in ymm until the row ends — hsum-per-group was the 2x loss.
CHR_INLINE void i4c_dequant16(const uint8_t *p, __m256 vs, __m256 *w0, __m256 *w1) {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
  const __m128i nib = _mm_set1_epi8(0x0f);
  const __m128i lo = _mm_and_si128(raw, nib);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nib);
  const __m128i seq = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
  *w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(seq)), vs);
  *w1 = _mm256_mul_ps(
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(seq, 8))), vs);
}

CHR_INLINE void i4c_acc4_16(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2,
                            const uint8_t *p3, __m256 vs0, __m256 vs1, __m256 vs2,
                            __m256 vs3, __m256 x0, __m256 x1, __m256 &a0l, __m256 &a0h,
                            __m256 &a1l, __m256 &a1h, __m256 &a2l, __m256 &a2h,
                            __m256 &a3l, __m256 &a3h) {
  __m256 w0, w1;
  i4c_dequant16(p0, vs0, &w0, &w1);
  a0l = _mm256_fmadd_ps(w0, x0, a0l);
  a0h = _mm256_fmadd_ps(w1, x1, a0h);
  i4c_dequant16(p1, vs1, &w0, &w1);
  a1l = _mm256_fmadd_ps(w0, x0, a1l);
  a1h = _mm256_fmadd_ps(w1, x1, a1h);
  i4c_dequant16(p2, vs2, &w0, &w1);
  a2l = _mm256_fmadd_ps(w0, x0, a2l);
  a2h = _mm256_fmadd_ps(w1, x1, a2h);
  i4c_dequant16(p3, vs3, &w0, &w1);
  a3l = _mm256_fmadd_ps(w0, x0, a3l);
  a3h = _mm256_fmadd_ps(w1, x1, a3h);
}

CHR_INLINE void i4c_acc4_group(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2,
                               const uint8_t *p3, __m256 vs0, __m256 vs1, __m256 vs2,
                               __m256 vs3, const float *xk, __m256 &a0l, __m256 &a0h,
                               __m256 &a1l, __m256 &a1h, __m256 &a2l, __m256 &a2h,
                               __m256 &a3l, __m256 &a3h) {
  _mm_prefetch(reinterpret_cast<const char *>(p0 + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char *>(p1 + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char *>(p2 + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char *>(p3 + 64), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char *>(xk + 64), _MM_HINT_T0);
  __m256 x0 = _mm256_loadu_ps(xk + 0);
  __m256 x1 = _mm256_loadu_ps(xk + 8);
  i4c_acc4_16(p0 + 0, p1 + 0, p2 + 0, p3 + 0, vs0, vs1, vs2, vs3, x0, x1, a0l, a0h, a1l,
              a1h, a2l, a2h, a3l, a3h);
  x0 = _mm256_loadu_ps(xk + 16);
  x1 = _mm256_loadu_ps(xk + 24);
  i4c_acc4_16(p0 + 8, p1 + 8, p2 + 8, p3 + 8, vs0, vs1, vs2, vs3, x0, x1, a0l, a0h, a1l,
              a1h, a2l, a2h, a3l, a3h);
  x0 = _mm256_loadu_ps(xk + 32);
  x1 = _mm256_loadu_ps(xk + 40);
  i4c_acc4_16(p0 + 16, p1 + 16, p2 + 16, p3 + 16, vs0, vs1, vs2, vs3, x0, x1, a0l, a0h,
              a1l, a1h, a2l, a2h, a3l, a3h);
  x0 = _mm256_loadu_ps(xk + 48);
  x1 = _mm256_loadu_ps(xk + 56);
  i4c_acc4_16(p0 + 24, p1 + 24, p2 + 24, p3 + 24, vs0, vs1, vs2, vs3, x0, x1, a0l, a0h,
              a1l, a1h, a2l, a2h, a3l, a3h);
}

void gemv_i4c_n1_avx2(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                      int n_groups, const float *x, float *y, int K, int m0, int m1) {
  const int k_full = (K / kI4cGroup) * kI4cGroup;
  int m = m0;
  for (; m + 3 < m1; m += 4) {
    const uint8_t *r0 = packed + static_cast<int64_t>(m) * packed_stride;
    const uint8_t *r1 = r0 + packed_stride;
    const uint8_t *r2 = r1 + packed_stride;
    const uint8_t *r3 = r2 + packed_stride;
    const uint16_t *s0 = scale + static_cast<int64_t>(m) * n_groups;
    const uint16_t *s1 = s0 + n_groups;
    const uint16_t *s2 = s1 + n_groups;
    const uint16_t *s3 = s2 + n_groups;
    if (m + 7 < m1) {
      _mm_prefetch(reinterpret_cast<const char *>(r3 + packed_stride), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char *>(r3 + 2 * packed_stride), _MM_HINT_T0);
    }
    __m256 a0l = _mm256_setzero_ps();
    __m256 a0h = _mm256_setzero_ps();
    __m256 a1l = _mm256_setzero_ps();
    __m256 a1h = _mm256_setzero_ps();
    __m256 a2l = _mm256_setzero_ps();
    __m256 a2h = _mm256_setzero_ps();
    __m256 a3l = _mm256_setzero_ps();
    __m256 a3h = _mm256_setzero_ps();
    int k = 0;
    int g = 0;
    for (; k < k_full; k += kI4cGroup, ++g) {
      i4c_acc4_group(r0 + (k >> 1), r1 + (k >> 1), r2 + (k >> 1), r3 + (k >> 1),
                     _mm256_set1_ps(fp16_to_f32(s0[g])), _mm256_set1_ps(fp16_to_f32(s1[g])),
                     _mm256_set1_ps(fp16_to_f32(s2[g])), _mm256_set1_ps(fp16_to_f32(s3[g])),
                     x + k, a0l, a0h, a1l, a1h, a2l, a2h, a3l, a3h);
    }
    float t0 = hsum256(_mm256_add_ps(a0l, a0h));
    float t1 = hsum256(_mm256_add_ps(a1l, a1h));
    float t2 = hsum256(_mm256_add_ps(a2l, a2h));
    float t3 = hsum256(_mm256_add_ps(a3l, a3h));
    if (k < K) {
      const float g0 = fp16_to_f32(s0[k / kI4cGroup]);
      const float g1 = fp16_to_f32(s1[k / kI4cGroup]);
      const float g2 = fp16_to_f32(s2[k / kI4cGroup]);
      const float g3 = fp16_to_f32(s3[k / kI4cGroup]);
      for (int kt = k; kt < K; ++kt) {
        const int byte = kt >> 1;
        const int odd = kt & 1;
        const uint8_t b0 = r0[byte], b1 = r1[byte], b2 = r2[byte], b3 = r3[byte];
        const int sft = odd ? 4 : 0;
        const int msk = 0x0f << sft;
        t0 += static_cast<float>(((b0 & msk) >> sft) - 8) * g0 * x[kt];
        t1 += static_cast<float>(((b1 & msk) >> sft) - 8) * g1 * x[kt];
        t2 += static_cast<float>(((b2 & msk) >> sft) - 8) * g2 * x[kt];
        t3 += static_cast<float>(((b3 & msk) >> sft) - 8) * g3 * x[kt];
      }
    }
    y[m] = t0;
    y[m + 1] = t1;
    y[m + 2] = t2;
    y[m + 3] = t3;
  }
  for (; m < m1; ++m) {
    const uint8_t *row = packed + static_cast<int64_t>(m) * packed_stride;
    const uint16_t *sc = scale + static_cast<int64_t>(m) * n_groups;
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int k = 0;
    int g = 0;
    for (; k < k_full; k += kI4cGroup, ++g) {
      const __m256 vs = _mm256_set1_ps(fp16_to_f32(sc[g]));
      const uint8_t *pp = row + (k >> 1);
      const float *xk = x + k;
      __m256 w0, w1;
      i4c_dequant16(pp + 0, vs, &w0, &w1);
      a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 0), a0);
      a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 8), a1);
      i4c_dequant16(pp + 8, vs, &w0, &w1);
      a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 16), a2);
      a3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 24), a3);
      i4c_dequant16(pp + 16, vs, &w0, &w1);
      a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 32), a0);
      a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 40), a1);
      i4c_dequant16(pp + 24, vs, &w0, &w1);
      a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xk + 48), a2);
      a3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xk + 56), a3);
    }
    float t = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    if (k < K) {
      const float sg = fp16_to_f32(sc[k / kI4cGroup]);
      for (; k < K; ++k) {
        const uint8_t b = row[k >> 1];
        const int q = ((k & 1) ? (b >> 4) : (b & 0x0f)) - 8;
        t += static_cast<float>(q) * sg * x[k];
      }
    }
    y[m] = t;
  }
}
#endif

void gemv_i4c_rows_scalar(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                          int n_groups, const float *x, float *y, int K, int m0, int m1) {
  (void)n_groups;
  for (int m = m0; m < m1; ++m) {
    const uint8_t *row = packed + static_cast<int64_t>(m) * packed_stride;
    const uint16_t *sc = scale + static_cast<int64_t>(m) * ((K + kI4cGroup - 1) / kI4cGroup);
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      const uint8_t b = row[k >> 1];
      const int q = ((k & 1) ? (b >> 4) : (b & 0x0f)) - 8;
      acc += static_cast<float>(q) * fp16_to_f32(sc[k / kI4cGroup]) * x[k];
    }
    y[m] = acc;
  }
}

void gemv_i4c_rows(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                   int n_groups, const float *x, float *y, int K, int m0, int m1) {
#if CHR_NF4_AVX2
  gemv_i4c_n1_avx2(packed, packed_stride, scale, n_groups, x, y, K, m0, m1);
#else
  gemv_i4c_rows_scalar(packed, packed_stride, scale, n_groups, x, y, K, m0, m1);
#endif
}

}  // namespace

namespace {

struct GemvJob {
  const uint8_t *packed;
  const uint16_t *scale;
  const float *x;
  float *y;
  int packed_stride;
  int n_groups;
  int M;
  int K;
  int N;
  int chunk;
  int nthr;
  int kind;  // 0 nf4, 1 i4c
};

std::mutex g_mu;
std::condition_variable g_cv_work;
std::condition_variable g_cv_done;
GemvJob g_job{};
std::atomic<int> g_epoch{0};
std::atomic<int> g_done{0};
std::atomic<bool> g_stop{false};
int g_pool_nthr = 0;
std::vector<std::thread> g_workers;

void worker_loop(int tid) {
  int seen = 0;
  for (;;) {
    std::unique_lock<std::mutex> lk(g_mu);
    g_cv_work.wait(lk, [&] {
      return g_stop.load(std::memory_order_acquire) ||
             g_epoch.load(std::memory_order_acquire) != seen;
    });
    if (g_stop.load(std::memory_order_acquire) &&
        g_epoch.load(std::memory_order_acquire) == seen) {
      return;
    }
    seen = g_epoch.load(std::memory_order_relaxed);
    const GemvJob job = g_job;
    lk.unlock();
    if (tid < job.nthr) {
      const int lo = tid * job.chunk;
      const int hi = std::min(job.M, lo + job.chunk);
      if (lo < hi) {
        if (job.kind == 1) {
          gemv_i4c_rows(job.packed, job.packed_stride, job.scale, job.n_groups, job.x,
                        job.y, job.K, lo, hi);
        } else {
          gemv_rows(job.packed, job.packed_stride, job.scale, job.n_groups, job.x, job.y,
                    job.M, job.K, job.N, lo, hi);
        }
      }
    }
    if (g_done.fetch_add(1, std::memory_order_acq_rel) + 1 == g_pool_nthr) {
      std::lock_guard<std::mutex> done_lk(g_mu);
      g_cv_done.notify_one();
    }
  }
}

void stop_pool() {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_stop.store(true, std::memory_order_release);
  }
  g_cv_work.notify_all();
  for (auto &th : g_workers) {
    th.join();
  }
  g_workers.clear();
  g_pool_nthr = 0;
  g_stop.store(false, std::memory_order_release);
}

void ensure_pool(int nworkers) {
  if (g_pool_nthr == nworkers) {
    return;
  }
  if (g_pool_nthr != 0) {
    stop_pool();
  }
  if (nworkers < 1) {
    return;
  }
  static std::once_flag atexit_once;
  std::call_once(atexit_once, [] { std::atexit(stop_pool); });
  g_workers.reserve(static_cast<size_t>(nworkers));
  for (int tid = 1; tid <= nworkers; ++tid) {
    g_workers.emplace_back(worker_loop, tid);
  }
  g_pool_nthr = nworkers;
}

void run_parallel_gemv(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                       int n_groups, const float *x, float *y, int M, int K, int N,
                       int nthr, int chunk) {
  if (nthr <= 1) {
    gemv_rows(packed, packed_stride, scale, n_groups, x, y, M, K, N, 0, M);
    return;
  }
  ensure_pool(nthr - 1);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_job = GemvJob{packed, scale, x, y, packed_stride, n_groups, M, K, N, chunk, nthr, 0};
    g_done.store(0, std::memory_order_relaxed);
    g_epoch.fetch_add(1, std::memory_order_acq_rel);
  }
  g_cv_work.notify_all();
  gemv_rows(packed, packed_stride, scale, n_groups, x, y, M, K, N, 0, std::min(M, chunk));
  std::unique_lock<std::mutex> lk(g_mu);
  g_cv_done.wait(lk, [&] { return g_done.load(std::memory_order_acquire) >= nthr - 1; });
}

void run_parallel_i4c(const uint8_t *packed, int packed_stride, const uint16_t *scale,
                      int n_groups, const float *x, float *y, int M, int K, int nthr,
                      int chunk) {
  if (nthr <= 1) {
    gemv_i4c_rows(packed, packed_stride, scale, n_groups, x, y, K, 0, M);
    return;
  }
  ensure_pool(nthr - 1);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_job = GemvJob{packed, scale, x, y, packed_stride, n_groups, M, K, 1, chunk, nthr, 1};
    g_done.store(0, std::memory_order_relaxed);
    g_epoch.fetch_add(1, std::memory_order_acq_rel);
  }
  g_cv_work.notify_all();
  gemv_i4c_rows(packed, packed_stride, scale, n_groups, x, y, K, 0, std::min(M, chunk));
  std::unique_lock<std::mutex> lk(g_mu);
  g_cv_done.wait(lk, [&] { return g_done.load(std::memory_order_acquire) >= nthr - 1; });
}

}  // namespace

extern "C" int chr_nf4_gemv_cpu(const uint8_t *packed, const uint16_t *scale,
                                const float *x, float *y, int32_t M, int32_t K,
                                int32_t K_pad, int32_t N, int32_t nthreads) {
  if (packed == nullptr || scale == nullptr || x == nullptr || y == nullptr) {
    return -1;
  }
  if (N < 1 || N > kLiveMaxN) {
    return -2;
  }
  if (M < 1 || K < 1) {
    return -3;
  }
  const int32_t k_pad_want = kGroup * ((K + kGroup - 1) / kGroup);
  if (K_pad != k_pad_want) {
    return -4;
  }
  const int packed_stride = K_pad / 2;
  const int n_groups = K_pad / kGroup;
  // ATen parallel_for inside this .pyd stayed serial on Windows (1==16 threads).
  // std::thread over row blocks; cap 16 so we do not fight a live CUDA prefix.
  int nthr = static_cast<int>(nthreads);
  if (nthr < 1) {
    nthr = 1;
  }
  if (nthr > 16) {
    nthr = 16;
  }
  if (nthr > M) {
    nthr = M;
  }
  int chunk = (M + nthr - 1) / nthr;
  if (M >= nthr * 4) {
    chunk = (chunk + 3) & ~3;
  }
  run_parallel_gemv(packed, packed_stride, scale, n_groups, x, y, M, K, N, nthr, chunk);
  return 0;
}

extern "C" int chr_i4c_gemv_cpu(const uint8_t *packed, const uint16_t *scale,
                                const float *x, float *y, int32_t M, int32_t K,
                                int32_t K_pad, int32_t N, int32_t nthreads) {
  if (packed == nullptr || scale == nullptr || x == nullptr || y == nullptr) {
    return -1;
  }
  if (N != 1) {
    return -2;
  }
  if (M < 1 || K < 1) {
    return -3;
  }
  const int32_t k_pad_want = kI4cGroup * ((K + kI4cGroup - 1) / kI4cGroup);
  if (K_pad != k_pad_want) {
    return -4;
  }
  int nthr = static_cast<int>(nthreads);
  if (nthr < 1) {
    nthr = 1;
  }
  if (nthr > 16) {
    nthr = 16;
  }
  if (nthr > M) {
    nthr = M;
  }
  int chunk = (M + nthr - 1) / nthr;
  if (M >= nthr * 4) {
    chunk = (chunk + 3) & ~3;
  }
  const int packed_stride = K_pad / 2;
  const int n_groups = K_pad / kI4cGroup;
  run_parallel_i4c(packed, packed_stride, scale, n_groups, x, y, M, K, nthr, chunk);
  return 0;
}

extern "C" const char *chr_nf4_cpu_isa(void) {
#if CHR_NF4_AVX2
  return "avx2";
#else
  return "scalar";
#endif
}
