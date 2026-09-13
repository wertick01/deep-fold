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
  m.def("nf4_set_tuning", &nf4_set_tuning,
        "path (0 auto / 1 classic decode / 2 small decode), split_k (0 auto), "
        "one_wave (0 default 70)",
        py::arg("path") = 0, py::arg("split_k") = 0, py::arg("one_wave") = 0);
}
