#include "chr_gpu.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

const char *gemm_err(int rc) {
  switch (rc) {
  case 0:
    return "ok";
  case -1:
    return "null pointer";
  case -2:
    return "N out of range (live 1..16, plan 1..64)";
  case -3:
    return "invalid M/K or null packed/scale";
  case -4:
    return "K_pad != 64*ceil(K/64)";
  case -5:
    return "packed or x not 16-byte aligned";
  case -6:
    return "CUDA launch / func attribute failed";
  case -7:
    return "split-K workspace smaller than the plan asked for";
  default:
    return "unknown error";
  }
}

} // namespace

torch::Tensor nf4_gemm(torch::Tensor packed, torch::Tensor scale, torch::Tensor x,
                        int64_t M, int64_t K, int64_t K_pad, int64_t max_n) {
  TORCH_CHECK(packed.is_cuda() && scale.is_cuda() && x.is_cuda(),
              "packed, scale, and x must be CUDA tensors");
  TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat16, "scale must be float16");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(M >= 1 && K >= 1, "M and K must be >= 1");

  packed = packed.contiguous();
  scale = scale.contiguous();
  x = x.contiguous();
  if (x.dim() == 1) {
    x = x.unsqueeze(1);
  }
  TORCH_CHECK(x.dim() == 2, "x must be [K, N] or [K]");
  TORCH_CHECK(x.size(0) == K, "x.size(0) must equal K");

  // The kernel indexes packed with stride K_pad/2 up to row M-1, and scale with
  // stride K_pad/64. Blobs smaller than the declared M/K_pad therefore make it
  // read past the allocation: before these checks an undersized packed/scale
  // launched anyway and returned plausible-looking garbage instead of failing.
  // Added by agent 4 for docs/spec/gpu-safety.md S9 (OOB fix, not a kernel change).
  const int64_t k_pad_want = 64 * ((K + 63) / 64);
  TORCH_CHECK(K_pad == k_pad_want, "K_pad must be 64*ceil(K/64) = ", k_pad_want,
              ", got ", K_pad);
  TORCH_CHECK(packed.numel() >= M * (K_pad / 2), "packed has ", packed.numel(),
              " bytes, the kernel reads M*K_pad/2 = ", M * (K_pad / 2));
  TORCH_CHECK(scale.numel() >= M * (K_pad / 64), "scale has ", scale.numel(),
              " elements, the kernel reads M*K_pad/64 = ", M * (K_pad / 64));

  const int32_t N = static_cast<int32_t>(x.size(1));
  auto y = torch::empty({M, x.size(1)}, x.options());

  chr_nf4_dev_t w{};
  w.M = static_cast<int32_t>(M);
  w.K = static_cast<int32_t>(K);
  w.K_pad = static_cast<int32_t>(K_pad);
  w.packed = packed.data_ptr<uint8_t>();
  w.scale = reinterpret_cast<const uint16_t *>(scale.data_ptr<at::Half>());

  // Ask the planner what grid it wants *before* launching, because split_k > 1
  // needs FP32 partials and nothing inside the .cu is allowed to allocate.
  // Torch's caching allocator hands these back for free after the first call,
  // and under CUDA graph capture they come from the graph's private pool.
  chr_nf4_plan_t plan{};
  const int prc = chr_nf4_gemm_plan(&w, N, /*have_ws=*/1, &plan);
  TORCH_CHECK(prc == 0, "chr_nf4_gemm_plan failed (", prc, "): ", gemm_err(prc));

  torch::Tensor ws;
  float *ws_ptr = nullptr;
  if (plan.ws_floats > 0) {
    ws = torch::empty({plan.ws_floats}, x.options().dtype(torch::kFloat32));
    ws_ptr = ws.data_ptr<float>();
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc = chr_nf4_gemm_ws_max(&w, x.data_ptr(), y.data_ptr(), N, ws_ptr,
                                      plan.ws_floats, stream,
                                      static_cast<int32_t>(max_n));
  TORCH_CHECK(rc == 0, "chr_nf4_gemm failed (", rc, "): ", gemm_err(rc));
  return y;
}

// Launch math with no device memory and no launch: the occupancy claim in
// docs/tz/wave9-review.md §3 is a grid size, so it should be assertable.
py::dict nf4_plan(int64_t M, int64_t K, int64_t K_pad, int64_t N,
                  bool have_ws) {
  chr_nf4_dev_t w{};
  w.M = static_cast<int32_t>(M);
  w.K = static_cast<int32_t>(K);
  w.K_pad = static_cast<int32_t>(K_pad);
  chr_nf4_plan_t p{};
  const int rc = chr_nf4_gemm_plan(&w, static_cast<int32_t>(N),
                                   have_ws ? 1 : 0, &p);
  TORCH_CHECK(rc == 0, "chr_nf4_gemm_plan failed (", rc, "): ", gemm_err(rc));
  py::dict d;
  d["path"] = p.path;
  d["grid_x"] = p.grid_x;
  d["grid_y"] = p.grid_y;
  d["block"] = p.block;
  d["bm"] = p.bm;
  d["bk"] = p.bk;
  d["n_ktiles"] = p.n_ktiles;
  d["tiles_per_split"] = p.tiles_per_split;
  d["ctas"] = p.ctas;
  d["smem_bytes"] = p.smem_bytes;
  d["ws_floats"] = p.ws_floats;
  return d;
}

torch::Tensor nf4_gemv(torch::Tensor packed, torch::Tensor scale, torch::Tensor x,
                       int64_t M, int64_t K, int64_t K_pad,
                       c10::optional<torch::Tensor> out,
                       c10::optional<torch::Tensor> add,
                       c10::optional<torch::Tensor> rms_w, double rms_eps) {
  TORCH_CHECK(packed.is_cuda() && scale.is_cuda() && x.is_cuda(),
              "packed, scale, and x must be CUDA tensors");
  TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat16, "scale must be float16");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(M >= 1 && K >= 1, "M and K must be >= 1");

  packed = packed.contiguous();
  scale = scale.contiguous();
  x = x.contiguous();
  if (x.dim() == 1) {
    x = x.unsqueeze(1);
  }
  TORCH_CHECK(x.dim() == 2, "x must be [K, 1] or [K]");
  TORCH_CHECK(x.size(0) == K, "x.size(0) must equal K");
  TORCH_CHECK(x.size(1) == 1, "chr_nf4_gemv is N=1 only");

  const int64_t k_pad_want = 64 * ((K + 63) / 64);
  TORCH_CHECK(K_pad == k_pad_want, "K_pad must be 64*ceil(K/64) = ", k_pad_want,
              ", got ", K_pad);
  TORCH_CHECK(packed.numel() >= M * (K_pad / 2), "packed has ", packed.numel(),
              " bytes, the kernel reads M*K_pad/2 = ", M * (K_pad / 2));
  TORCH_CHECK(scale.numel() >= M * (K_pad / 64), "scale has ", scale.numel(),
              " elements, the kernel reads M*K_pad/64 = ", M * (K_pad / 64));

  torch::Tensor y;
  if (out.has_value() && out->defined()) {
    TORCH_CHECK(out->is_cuda() && out->scalar_type() == torch::kBFloat16,
                "out must be CUDA bfloat16");
    TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out->numel() >= M, "out too small");
    y = *out;
  } else {
    y = torch::empty({M, 1}, x.options());
  }

  torch::Tensor add_keep;
  const void *add_ptr = nullptr;
  if (add.has_value() && add->defined() && add->numel() > 0) {
    TORCH_CHECK(add->is_cuda(), "add must be CUDA");
    TORCH_CHECK(add->numel() >= M, "add too small");
    add_keep = add->contiguous();
    if (add_keep.scalar_type() != torch::kBFloat16) {
      add_keep = add_keep.to(torch::kBFloat16);
    }
    add_ptr = add_keep.data_ptr();
  }

  torch::Tensor rms_keep;
  const void *rms_p = nullptr;
  if (rms_w.has_value() && rms_w->defined() && rms_w->numel() > 0) {
    TORCH_CHECK(rms_w->is_cuda(), "rms_w must be CUDA");
    TORCH_CHECK(rms_w->numel() == K, "rms_w numel must equal K");
    rms_keep = rms_w->contiguous();
    if (rms_keep.scalar_type() != torch::kBFloat16) {
      rms_keep = rms_keep.to(torch::kBFloat16);
    }
    rms_p = rms_keep.data_ptr();
  }

  chr_nf4_dev_t w{};
  w.M = static_cast<int32_t>(M);
  w.K = static_cast<int32_t>(K);
  w.K_pad = static_cast<int32_t>(K_pad);
  w.packed = packed.data_ptr<uint8_t>();
  w.scale = reinterpret_cast<const uint16_t *>(scale.data_ptr<at::Half>());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc =
      chr_nf4_gemv(&w, x.data_ptr(), y.data_ptr(), stream, add_ptr, rms_p,
                   static_cast<float>(rms_eps));
  TORCH_CHECK(rc == 0, "chr_nf4_gemv failed (", rc, "): ", gemm_err(rc));
  return y;
}

namespace {

void check_packed(const torch::Tensor &packed, const torch::Tensor &scale,
                  int64_t M, int64_t K, int64_t K_pad, const char *name) {
  TORCH_CHECK(packed.is_cuda() && scale.is_cuda(), name, " packed/scale must be CUDA");
  TORCH_CHECK(packed.scalar_type() == torch::kByte, name, " packed must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat16, name, " scale must be float16");
  const int64_t k_pad_want = 64 * ((K + 63) / 64);
  TORCH_CHECK(K_pad == k_pad_want, name, " K_pad must be 64*ceil(K/64)");
  TORCH_CHECK(packed.numel() >= M * (K_pad / 2), name, " packed too small");
  TORCH_CHECK(scale.numel() >= M * (K_pad / 64), name, " scale too small");
}

torch::Tensor as_kvec(torch::Tensor x, int64_t K) {
  x = x.contiguous();
  if (x.dim() == 1) {
    TORCH_CHECK(x.size(0) == K, "x.size(0) must equal K");
    return x;
  }
  TORCH_CHECK(x.dim() == 2, "x must be [K], [K,1], or [1,K]");
  if (x.size(0) == K && x.size(1) == 1) {
    return x.view(K);
  }
  if (x.size(0) == 1 && x.size(1) == K) {
    return x.view(K);
  }
  TORCH_CHECK(false, "x must be [K], [K,1], or [1,K]");
  return x;
}

chr_nf4_dev_t make_dev(const torch::Tensor &packed, const torch::Tensor &scale,
                       int64_t M, int64_t K, int64_t K_pad) {
  chr_nf4_dev_t w{};
  w.M = static_cast<int32_t>(M);
  w.K = static_cast<int32_t>(K);
  w.K_pad = static_cast<int32_t>(K_pad);
  w.packed = packed.data_ptr<uint8_t>();
  w.scale = reinterpret_cast<const uint16_t *>(scale.data_ptr<at::Half>());
  return w;
}

torch::Tensor out_vec(torch::Tensor out, int64_t M, const torch::Tensor &like) {
  TORCH_CHECK(out.defined(), "out must be defined");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16,
              "out must be CUDA bfloat16");
  TORCH_CHECK(out.numel() >= M, "out too small");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  return out;
}

const void *rms_ptr(c10::optional<torch::Tensor> rms_w, int64_t K,
                    torch::Tensor &keep) {
  if (!rms_w.has_value() || !rms_w->defined() || rms_w->numel() == 0) {
    return nullptr;
  }
  TORCH_CHECK(rms_w->is_cuda(), "rms_w must be CUDA");
  TORCH_CHECK(rms_w->numel() == K, "rms_w numel must equal K");
  keep = rms_w->contiguous();
  if (keep.scalar_type() != torch::kBFloat16) {
    keep = keep.to(torch::kBFloat16);
  }
  return keep.data_ptr();
}

const void *bias_ptr(c10::optional<torch::Tensor> bias, int64_t M,
                     torch::Tensor &keep) {
  if (!bias.has_value() || !bias->defined() || bias->numel() == 0) {
    return nullptr;
  }
  TORCH_CHECK(bias->is_cuda(), "bias must be CUDA");
  TORCH_CHECK(bias->numel() == M, "bias numel must equal M");
  keep = bias->contiguous();
  if (keep.scalar_type() != torch::kBFloat16) {
    keep = keep.to(torch::kBFloat16);
  }
  return keep.data_ptr();
}

} // namespace

std::vector<torch::Tensor> nf4_qkv(
    torch::Tensor q_packed, torch::Tensor q_scale, torch::Tensor k_packed,
    torch::Tensor k_scale, torch::Tensor v_packed, torch::Tensor v_scale,
    torch::Tensor x, int64_t Mq, int64_t Mk, int64_t Mv, int64_t K,
    int64_t K_pad, torch::Tensor out_q, torch::Tensor out_k,
    torch::Tensor out_v, c10::optional<torch::Tensor> q_bias,
    c10::optional<torch::Tensor> k_bias, c10::optional<torch::Tensor> v_bias,
    c10::optional<torch::Tensor> rms_w, double rms_eps) {
  check_packed(q_packed, q_scale, Mq, K, K_pad, "q");
  check_packed(k_packed, k_scale, Mk, K, K_pad, "k");
  check_packed(v_packed, v_scale, Mv, K, K_pad, "v");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
  x = as_kvec(x, K);
  auto yq = out_vec(out_q, Mq, x);
  auto yk = out_vec(out_k, Mk, x);
  auto yv = out_vec(out_v, Mv, x);
  q_packed = q_packed.contiguous();
  q_scale = q_scale.contiguous();
  k_packed = k_packed.contiguous();
  k_scale = k_scale.contiguous();
  v_packed = v_packed.contiguous();
  v_scale = v_scale.contiguous();
  torch::Tensor qb_keep, kb_keep, vb_keep, rms_keep;
  const void *bq = bias_ptr(q_bias, Mq, qb_keep);
  const void *bk = bias_ptr(k_bias, Mk, kb_keep);
  const void *bv = bias_ptr(v_bias, Mv, vb_keep);
  const void *rms_p = rms_ptr(rms_w, K, rms_keep);
  auto q = make_dev(q_packed, q_scale, Mq, K, K_pad);
  auto k = make_dev(k_packed, k_scale, Mk, K, K_pad);
  auto v = make_dev(v_packed, v_scale, Mv, K, K_pad);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc = chr_nf4_gemv_qkv(&q, &k, &v, x.data_ptr(), yq.data_ptr(),
                                  yk.data_ptr(), yv.data_ptr(), bq, bk, bv,
                                  stream, rms_p, static_cast<float>(rms_eps));
  TORCH_CHECK(rc == 0, "chr_nf4_gemv_qkv failed (", rc, "): ", gemm_err(rc));
  return {yq, yk, yv};
}

torch::Tensor nf4_swiglu(torch::Tensor g_packed, torch::Tensor g_scale,
                         torch::Tensor u_packed, torch::Tensor u_scale,
                         torch::Tensor x, int64_t M, int64_t K, int64_t K_pad,
                         torch::Tensor out, c10::optional<torch::Tensor> rms_w,
                         double rms_eps) {
  check_packed(g_packed, g_scale, M, K, K_pad, "gate");
  check_packed(u_packed, u_scale, M, K, K_pad, "up");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
  x = as_kvec(x, K);
  auto y = out_vec(out, M, x);
  g_packed = g_packed.contiguous();
  g_scale = g_scale.contiguous();
  u_packed = u_packed.contiguous();
  u_scale = u_scale.contiguous();
  auto g = make_dev(g_packed, g_scale, M, K, K_pad);
  auto u = make_dev(u_packed, u_scale, M, K, K_pad);
  torch::Tensor rms_keep;
  const void *rms_p = rms_ptr(rms_w, K, rms_keep);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc =
      chr_nf4_gemv_swiglu(&g, &u, x.data_ptr(), y.data_ptr(), stream, rms_p,
                         static_cast<float>(rms_eps));
  TORCH_CHECK(rc == 0, "chr_nf4_gemv_swiglu failed (", rc, "): ", gemm_err(rc));
  return y;
}

void nf4_rope_kv(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos,
                 torch::Tensor sin, torch::Tensor position) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  cos = cos.contiguous();
  sin = sin.contiguous();
  position = position.contiguous();
  TORCH_CHECK(position.is_cuda(), "position must be CUDA");
  TORCH_CHECK(position.numel() == 1, "position must be a scalar");
  TORCH_CHECK(position.scalar_type() == torch::kLong, "position must be int64");
  TORCH_CHECK(k_cache.dim() == 3 && v_cache.dim() == 3, "cache must be [max_seq,n_kv,hd]");
  const int64_t max_seq = k_cache.size(0);
  const int64_t n_kv = k_cache.size(1);
  const int64_t hd = k_cache.size(2);
  TORCH_CHECK(v_cache.size(0) == max_seq && v_cache.size(1) == n_kv &&
                  v_cache.size(2) == hd,
              "v_cache shape mismatch");
  TORCH_CHECK(cos.size(0) == max_seq && cos.size(1) == hd, "cos shape");
  TORCH_CHECK(sin.size(0) == max_seq && sin.size(1) == hd, "sin shape");
  int64_t n_q = 0;
  if (q.dim() == 3) {
    TORCH_CHECK(q.size(0) == 1, "q batch must be 1");
    n_q = q.size(1);
    TORCH_CHECK(q.size(2) == hd, "q head_dim");
  } else {
    TORCH_CHECK(q.dim() == 2, "q must be [n_q,hd] or [1,n_q,hd]");
    n_q = q.size(0);
    TORCH_CHECK(q.size(1) == hd, "q head_dim");
  }
  auto squeeze_kv = [&](torch::Tensor t, const char *name) {
    if (t.dim() == 3) {
      TORCH_CHECK(t.size(0) == 1 && t.size(1) == n_kv && t.size(2) == hd, name);
      return t;
    }
    TORCH_CHECK(t.dim() == 2 && t.size(0) == n_kv && t.size(1) == hd, name);
    return t;
  };
  k = squeeze_kv(k, "k");
  v = squeeze_kv(v, "v");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc = chr_nf4_rope_kv(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), cos.data_ptr(), sin.data_ptr(),
      position.data_ptr<int64_t>(),
      static_cast<int32_t>(n_q), static_cast<int32_t>(n_kv),
      static_cast<int32_t>(hd), static_cast<int32_t>(max_seq), stream);
  TORCH_CHECK(rc == 0, "chr_nf4_rope_kv failed (", rc, "): ", gemm_err(rc));
}

void nf4_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
              torch::Tensor valid_len, double scale, torch::Tensor ws,
              int64_t n_split, c10::optional<torch::Tensor> k_act,
              c10::optional<torch::Tensor> v_act, c10::optional<torch::Tensor> cos,
              c10::optional<torch::Tensor> sin,
              c10::optional<torch::Tensor> position) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
              "q/k/v/out must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  valid_len = valid_len.contiguous();
  TORCH_CHECK(valid_len.is_cuda(), "valid_len must be CUDA");
  TORCH_CHECK(valid_len.numel() == 1, "valid_len must be a scalar");
  TORCH_CHECK(valid_len.scalar_type() == torch::kInt, "valid_len must be int32");
  TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "cache must be [max_seq,n_kv,hd]");
  const int64_t max_seq = k.size(0);
  const int64_t n_kv = k.size(1);
  const int64_t hd = k.size(2);
  TORCH_CHECK(v.size(0) == max_seq && v.size(1) == n_kv && v.size(2) == hd,
              "v cache shape mismatch");
  int64_t n_q = 0;
  if (q.dim() == 3) {
    TORCH_CHECK(q.size(0) == 1, "q batch must be 1");
    n_q = q.size(1);
    TORCH_CHECK(q.size(2) == hd, "q head_dim");
  } else {
    TORCH_CHECK(q.dim() == 2, "q must be [n_q,hd] or [1,n_q,hd]");
    n_q = q.size(0);
    TORCH_CHECK(q.size(1) == hd, "q head_dim");
  }
  TORCH_CHECK(out.numel() >= n_q * hd, "out too small");
  TORCH_CHECK(n_split >= 1 && n_split <= 64, "n_split must be 1..64");
  TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == torch::kFloat32, "ws must be CUDA fp32");
  ws = ws.contiguous();
  const int64_t need = n_q * n_split * (hd + 2);
  TORCH_CHECK(ws.numel() >= need, "attn workspace too small");
  torch::Tensor ka_keep, va_keep, c_keep, s_keep, p_keep;
  const void *ka = nullptr;
  const void *va = nullptr;
  const void *cp = nullptr;
  const void *sp = nullptr;
  const void *pp = nullptr;
  if (k_act.has_value() && k_act->defined() && k_act->numel() > 0) {
    ka_keep = k_act->contiguous();
    ka = ka_keep.data_ptr();
  }
  if (v_act.has_value() && v_act->defined() && v_act->numel() > 0) {
    va_keep = v_act->contiguous();
    va = va_keep.data_ptr();
  }
  if (cos.has_value() && cos->defined() && cos->numel() > 0) {
    c_keep = cos->contiguous();
    cp = c_keep.data_ptr();
  }
  if (sin.has_value() && sin->defined() && sin->numel() > 0) {
    s_keep = sin->contiguous();
    sp = s_keep.data_ptr();
  }
  if (position.has_value() && position->defined() && position->numel() == 1) {
    TORCH_CHECK(position->is_cuda() && position->scalar_type() == torch::kLong,
                "position must be CUDA int64");
    p_keep = position->contiguous();
    pp = p_keep.data_ptr<int64_t>();
  }
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc = chr_nf4_attn(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
      valid_len.data_ptr<int32_t>(), ws.data_ptr<float>(),
      static_cast<int32_t>(n_q), static_cast<int32_t>(n_kv),
      static_cast<int32_t>(hd), static_cast<int32_t>(n_split),
      static_cast<float>(scale), stream, ka, va, cp, sp, pp,
      static_cast<int32_t>(max_seq));
  TORCH_CHECK(rc == 0, "chr_nf4_attn failed (", rc, "): ", gemm_err(rc));
}

void nf4_rms(torch::Tensor x, torch::Tensor w, torch::Tensor y, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda(), "x/w/y must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  x = x.contiguous();
  w = w.contiguous();
  TORCH_CHECK(y.is_contiguous() && y.scalar_type() == torch::kBFloat16,
              "y must be contiguous bf16");
  const int64_t n = x.numel();
  TORCH_CHECK(w.numel() == n && y.numel() == n, "rms numel mismatch");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc =
      chr_nf4_rms(x.data_ptr(), w.data_ptr(), y.data_ptr(),
                  static_cast<int32_t>(n), static_cast<float>(eps), stream);
  TORCH_CHECK(rc == 0, "chr_nf4_rms failed (", rc, "): ", gemm_err(rc));
}

void nf4_set_tuning(int64_t path, int64_t split_k, int64_t one_wave) {
  chr_nf4_set_tuning(static_cast<int32_t>(path), static_cast<int32_t>(split_k),
                     static_cast<int32_t>(one_wave));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nf4_gemm", &nf4_gemm,
        "chr_nf4_gemm: y[M,N] = dequant_nf4(packed,scale) @ x[K,N]; max_n is 16 for live generate",
        py::arg("packed"), py::arg("scale"), py::arg("x"), py::arg("M"),
        py::arg("K"), py::arg("K_pad"), py::arg("max_n") = 16);
  m.def("nf4_plan", &nf4_plan,
        "launch plan for (M, K, K_pad, N): grid, tile, CTAs, workspace floats",
        py::arg("M"), py::arg("K"), py::arg("K_pad"), py::arg("N"),
        py::arg("have_ws") = true);
  m.def("nf4_gemv", &nf4_gemv,
        "chr_nf4_gemv: y[M,1] = dequant_nf4(packed,scale) @ x[K,1] CUDA-core N=1",
        py::arg("packed"), py::arg("scale"), py::arg("x"), py::arg("M"),
        py::arg("K"), py::arg("K_pad"), py::arg("out") = py::none(),
        py::arg("add") = py::none(), py::arg("rms_w") = py::none(),
        py::arg("rms_eps") = 1e-6);
  m.def("nf4_qkv", &nf4_qkv,
        "one-launch q/k/v GEMV into out_q/out_k/out_v", py::arg("q_packed"),
        py::arg("q_scale"), py::arg("k_packed"), py::arg("k_scale"),
        py::arg("v_packed"), py::arg("v_scale"), py::arg("x"), py::arg("Mq"),
        py::arg("Mk"), py::arg("Mv"), py::arg("K"), py::arg("K_pad"),
        py::arg("out_q"), py::arg("out_k"), py::arg("out_v"),
        py::arg("q_bias") = py::none(), py::arg("k_bias") = py::none(),
        py::arg("v_bias") = py::none(), py::arg("rms_w") = py::none(),
        py::arg("rms_eps") = 1e-6);
  m.def("nf4_swiglu", &nf4_swiglu, "one-launch silu(gate@x)*(up@x)",
        py::arg("g_packed"), py::arg("g_scale"), py::arg("u_packed"),
        py::arg("u_scale"), py::arg("x"), py::arg("M"), py::arg("K"),
        py::arg("K_pad"), py::arg("out"), py::arg("rms_w") = py::none(),
        py::arg("rms_eps") = 1e-6);
  m.def("nf4_rope_kv", &nf4_rope_kv,
        "in-place RoPE on q,k then write k,v into cache[position]", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("k_cache"), py::arg("v_cache"),
        py::arg("cos"), py::arg("sin"), py::arg("position"));
  m.def("nf4_attn", &nf4_attn,
        "flash-decode GQA attention, optional fused RoPE+KV write", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("out"), py::arg("valid_len"),
        py::arg("scale"), py::arg("ws"), py::arg("n_split") = 32,
        py::arg("k_act") = py::none(), py::arg("v_act") = py::none(),
        py::arg("cos") = py::none(), py::arg("sin") = py::none(),
        py::arg("position") = py::none());
  m.def("nf4_rms", &nf4_rms, "y = w * x * rsqrt(mean(x^2)+eps) into y",
        py::arg("x"), py::arg("w"), py::arg("y"), py::arg("eps"));
  m.def("nf4_set_tuning", &nf4_set_tuning,
        "path (0 auto / 1 classic decode / 2 small decode), split_k (0 auto), "
        "one_wave (0 default 70)",
        py::arg("path") = 0, py::arg("split_k") = 0, py::arg("one_wave") = 0);
}
