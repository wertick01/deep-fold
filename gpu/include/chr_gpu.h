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
 * Live launch N in [1, 16]: N=1 decode tile (BM=128, BN=8, BK=256); N=2..8
 * prefill (BM=64, BN=8 pad, BK=128); N=9..16 prefill (BM=64, BN=16 pad,
 * BK=128, epilogue masks n>=N). No CUDA-core GEMV.
 * chr_nf4_gemm / chr_nf4_gemm_ws: N>16 returns -2 — this call does not slice;
 * the host chunks into N<=16 (TokenLoop.prefill_chunk). That ceiling is the
 * WAVE freeze (decode ~31.6 tok/s stays the default path).
 * chr_nf4_gemm_plan also describes N=17..64 (path 3 BN=32, path 4 BN=64):
 * next TTFT floor, 167 vs 52 ms; ncu showed ~5% DRAM so this is MMA/dequant.
 * (-2 is no longer "any N!=1".)
 * stream is cudaStream_t (0 / nullptr = default stream).
 * Returns 0 on success. Negative: -1 null, -2 N out of range (live [1,16],
 * plan [1,64]), -3 bad dims,
 * -4 K_pad, -5 alignment (packed needs 16B), -6 CUDA launch, -7 workspace too
 * small for the plan. */
int chr_nf4_gemm(const chr_nf4_dev_t *w, const void *x, void *y, int32_t N,
                 void *stream);

/* --- launch plan + split-K workspace (additive; chr_nf4_dev_t stays frozen) --
 * Wave 2 shipped grid = (ceil(M / BM), 1, 1), split_k = 1. On this card that is
 * 16 CTAs for a 3B q/o_proj and 2 for k/v_proj against 70 SMs, so the decode
 * GEMM cannot use the machine (docs/tz/wave9-review.md §3). The plan below adds
 * grid.y = split_k over the K tiles plus a smaller decode tile, and needs FP32
 * partials because split_k > 1 cannot accumulate in y (BF16 atomics would both
 * round every partial and force y to be pre-zeroed).
 *
 * The workspace is the *caller's*: nothing in the .cu allocates. Ask
 * chr_nf4_gemm_plan for ws_floats, hand that many floats to chr_nf4_gemm_ws.
 * chr_nf4_gemm above is chr_nf4_gemm_ws with no workspace, i.e. split_k == 1. */
typedef struct {
  int32_t path;            /* 0 classic, 1 small decode, 2 prefill N<=16,
                            * 3 planned BN=32, 4 planned BN=64 */
  int32_t grid_x;          /* ceil(M / bm) */
  int32_t grid_y;          /* split_k */
  int32_t block;           /* threads per CTA */
  int32_t bm;              /* rows of M per CTA */
  int32_t bk;              /* K per pipeline stage */
  int32_t n_ktiles;        /* ceil(K_pad / bk) */
  int32_t tiles_per_split; /* K tiles one CTA walks */
  int32_t ctas;            /* grid_x * grid_y -- the occupancy number */
  int32_t smem_bytes;      /* dynamic smem per CTA */
  int64_t ws_floats;       /* FP32 partials; 0 when grid_y == 1 */
} chr_nf4_plan_t;

/* Pure launch math: only M/K/K_pad are read, packed/scale may be NULL.
 * have_ws == 0 reports the plan for a caller that cannot supply partials
 * (split_k pinned to 1). N in [1, 64]. Same error codes as chr_nf4_gemm. */
int chr_nf4_gemm_plan(const chr_nf4_dev_t *w, int32_t N, int32_t have_ws,
                      chr_nf4_plan_t *out);

/* ws must hold at least plan.ws_floats floats when plan.ws_floats > 0.
 * ws == NULL is legal and means "no split-K". Returns -7 if ws is too small.
 * Live TokenLoop uses this entry (N in [1, 16]). */
int chr_nf4_gemm_ws(const chr_nf4_dev_t *w, const void *x, void *y, int32_t N,
                    float *ws, int64_t ws_floats, void *stream);

/* Same launch as chr_nf4_gemm_ws, but N may go up to max_n (clamped to [1, 64]).
 * Oracle / numerics only. TokenLoop must keep calling chr_nf4_gemm_ws. */
int chr_nf4_gemm_ws_max(const chr_nf4_dev_t *w, const void *x, void *y, int32_t N,
                        float *ws, int64_t ws_floats, void *stream, int32_t max_n);

/* Tuning override for the microbench, so one process can time old vs new.
 * path: 0 auto, 1 force classic decode tile, 2 force small decode tile.
 * split_k: 0 auto, >0 force that many K splits (clamped to n_ktiles).
 * one_wave: SMs the split-K target is expressed in; 0 = this GPU's
 *           multiProcessorCount (fallback 70, which matches 3080 and 5070 Ti).
 * Initial values come from CHR_NF4_PATH / CHR_NF4_SPLIT_K / CHR_NF4_ONE_WAVE. */
void chr_nf4_set_tuning(int32_t path, int32_t split_k, int32_t one_wave);

/* N=1 CUDA-core GEMV. Does not replace chr_nf4_gemm. x is bf16 [K], y is bf16 [M].
 * Same packed/scale layout. FP32 LUT*scale*x, fused dequant×vector (mmvq-style).
 * Large M: several output rows per CTA so x is loaded once per K-tile.
 * add is optional bf16 [M] (NULL = overwrite y). y and add may alias (residual).
 * rms_w is optional bf16 [K] (NULL = use x as-is). Dynamic smem is K bf16 when set.
 * Error codes match chr_nf4_gemm (-1 null, -3 dims, -4 K_pad, -5 align, -6 launch). */
int chr_nf4_gemv(const chr_nf4_dev_t *w, const void *x, void *y, void *stream,
                 const void *add, const void *rms_w, float rms_eps);

/* One launch over q.M+k.M+v.M rows. q/k/v share K and K_pad. x is bf16 [K].
 * yq/yk/yv are bf16 [M]. bq/bk/bv are optional bf16 [M] (NULL = no bias).
 * Qwen2.5 q/k/v are biased; omitting them wrong-ids from token 0.
 * rms_w is optional bf16 [K]: apply RMSNorm(x, rms_w, rms_eps) before the dots. */
int chr_nf4_gemv_qkv(const chr_nf4_dev_t *q, const chr_nf4_dev_t *k,
                     const chr_nf4_dev_t *v, const void *x, void *yq, void *yk,
                     void *yv, const void *bq, const void *bk, const void *bv,
                     void *stream, const void *rms_w, float rms_eps);

/* Several gate+up pairs per CTA when M is large: silu(dequant(gate)@x) *
 * (dequant(up)@x). gate.M == up.M, same K. y is bf16 [M]. Optional rms_w as in
 * chr_nf4_gemv_qkv. */
int chr_nf4_gemv_swiglu(const chr_nf4_dev_t *gate, const chr_nf4_dev_t *up,
                        const void *x, void *y, void *stream, const void *rms_w,
                        float rms_eps);

/* In-place rotate-half RoPE on q [n_q,hd] and k [n_kv,hd], then write k and v
 * into cache [max_seq, n_kv, hd] at device *position (int64). cos/sin are
 * bf16 [max_seq, hd]. */
int chr_nf4_rope_kv(void *q, void *k, const void *v, void *k_cache, void *v_cache,
                    const void *cos, const void *sin, const void *position,
                    int32_t n_q, int32_t n_kv, int32_t hd, int32_t max_seq,
                    void *stream);

/* GQA flash-decode. k/v are cache [max_seq, n_kv, hd]. Optional k_act/v_act +
 * cos/sin/position: rotate-half Q, write rotated K and V into cache at
 * *position (one CTA per kv head), then attend. Partial splits merge in a
 * second launch (WDDM cooperative grid sync returned wrong greedy ids). */
int chr_nf4_attn(const void *q, const void *k, const void *v, void *out,
                 const void *valid_len, float *ws, int32_t n_q, int32_t n_kv,
                 int32_t hd, int32_t n_split, float scale, void *stream,
                 const void *k_act, const void *v_act, const void *cos,
                 const void *sin, const void *position, int32_t max_seq);

/* y[i] = w[i] * x[i] * rsqrt(mean(x^2) + eps). x/w/y are bf16 [n]. */
int chr_nf4_rms(const void *x, const void *w, void *y, int32_t n, float eps,
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
