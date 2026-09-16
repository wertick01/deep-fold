#include "nf4_gemv.h"

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

const char *gemv_err(int rc) {
  switch (rc) {
  case 0:
    return "ok";
  case -1:
    return "null pointer";
  case -2:
    return "N out of range (i4c wants N=1; nf4 1..32)";
  case -3:
    return "invalid M/K";
  case -4:
    return "K_pad does not match codec group";
  default:
    return "unknown error";
  }
}

}  // namespace

torch::Tensor nf4_gemm(torch::Tensor packed, torch::Tensor scale, torch::Tensor x,
                       int64_t M, int64_t K, int64_t K_pad) {
  TORCH_CHECK(!packed.is_cuda() && !scale.is_cuda() && !x.is_cuda(),
              "chr_nf4_cpu: packed/scale/x must be CPU; refuse silent H2D");
  TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat16, "scale must be float16");
  TORCH_CHECK(M >= 1 && K >= 1, "M and K must be >= 1");
  TORCH_CHECK(packed.dim() == 2, "packed must be [M, K_pad/2]");

  packed = packed.contiguous();
  scale = scale.contiguous();
  if (x.dim() == 1) {
    x = x.unsqueeze(1);
  }
  TORCH_CHECK(x.dim() == 2, "x must be [K, N] or [K]");
  TORCH_CHECK(x.size(0) == K, "x.size(0) must equal K");

  const int64_t k_pad_want = 64 * ((K + 63) / 64);
  TORCH_CHECK(K_pad == k_pad_want, "K_pad must be 64*ceil(K/64) = ", k_pad_want,
              ", got ", K_pad);
  TORCH_CHECK(packed.size(0) == M, "packed.size(0) must equal M");
  TORCH_CHECK(packed.numel() >= M * (K_pad / 2), "packed has ", packed.numel(),
              " bytes, kernel reads M*K_pad/2 = ", M * (K_pad / 2));
  TORCH_CHECK(scale.numel() >= M * (K_pad / 64), "scale has ", scale.numel(),
              " elements, kernel reads M*K_pad/64 = ", M * (K_pad / 64));

  const int32_t N = static_cast<int32_t>(x.size(1));
  TORCH_CHECK(N >= 1 && N <= 32, "N=", N, " not in 1..32");

  auto x32 = x.contiguous();
  if (x32.scalar_type() != torch::kFloat32) {
    x32 = x32.to(torch::kFloat32);
  }
  // N=1: [K,1] is already contiguous K. N>1: kernel wants [N,K] so each dot streams x.
  torch::Tensor xt = (N == 1) ? x32.reshape({K}) : x32.transpose(0, 1).contiguous();
  auto y = torch::empty({M, static_cast<int64_t>(N)}, x32.options());

  // Read TLS on this thread before the kernel. ATen parallel_for inside the
  // .pyd was staying serial (1 and 16 intra-op threads both ~25 ms).
  const int32_t nthr = static_cast<int32_t>(std::max<int64_t>(1, at::get_num_threads()));
  int rc = 0;
  {
    py::gil_scoped_release release;
    rc = chr_nf4_gemv_cpu(packed.data_ptr<uint8_t>(),
                          reinterpret_cast<const uint16_t *>(scale.data_ptr<at::Half>()),
                          xt.data_ptr<float>(), y.data_ptr<float>(), static_cast<int32_t>(M),
                          static_cast<int32_t>(K), static_cast<int32_t>(K_pad), N, nthr);
  }
  TORCH_CHECK(rc == 0, "chr_nf4_gemv_cpu failed (", rc, "): ", gemv_err(rc));
  return y;
}

torch::Tensor i4c_gemm(torch::Tensor packed, torch::Tensor scale, torch::Tensor x,
                       int64_t M, int64_t K, int64_t K_pad) {
  TORCH_CHECK(!packed.is_cuda() && !scale.is_cuda() && !x.is_cuda(),
              "chr_i4c_cpu: packed/scale/x must be CPU; refuse silent H2D");
  TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat16, "scale must be float16");
  TORCH_CHECK(M >= 1 && K >= 1, "M and K must be >= 1");
  TORCH_CHECK(packed.dim() == 2, "packed must be [M, K_pad/2]");

  packed = packed.contiguous();
  scale = scale.contiguous();
  if (x.dim() == 1) {
    x = x.unsqueeze(1);
  }
  TORCH_CHECK(x.dim() == 2, "x must be [K, N] or [K]");
  TORCH_CHECK(x.size(0) == K, "x.size(0) must equal K");
  TORCH_CHECK(x.size(1) == 1, "i4c decode is N=1, got N=", x.size(1));

  const int64_t k_pad_want = 64 * ((K + 63) / 64);
  TORCH_CHECK(K_pad == k_pad_want, "K_pad must be 64*ceil(K/64) = ", k_pad_want,
              ", got ", K_pad);
  TORCH_CHECK(packed.size(0) == M, "packed.size(0) must equal M");
  TORCH_CHECK(packed.numel() >= M * (K_pad / 2), "packed has ", packed.numel(),
              " bytes, kernel reads M*K_pad/2 = ", M * (K_pad / 2));
  TORCH_CHECK(scale.numel() >= M * (K_pad / 64), "scale has ", scale.numel(),
              " elements, kernel reads M*K_pad/64 = ", M * (K_pad / 64));

  auto x32 = x.contiguous();
  if (x32.scalar_type() != torch::kFloat32) {
    x32 = x32.to(torch::kFloat32);
  }
  torch::Tensor xt = x32.reshape({K});
  auto y = torch::empty({M, static_cast<int64_t>(1)}, x32.options());
  const int32_t nthr = static_cast<int32_t>(std::max<int64_t>(1, at::get_num_threads()));
  int rc = 0;
  {
    py::gil_scoped_release release;
    rc = chr_i4c_gemv_cpu(packed.data_ptr<uint8_t>(),
                          reinterpret_cast<const uint16_t *>(scale.data_ptr<at::Half>()),
                          xt.data_ptr<float>(), y.data_ptr<float>(), static_cast<int32_t>(M),
                          static_cast<int32_t>(K), static_cast<int32_t>(K_pad), 1, nthr);
  }
  TORCH_CHECK(rc == 0, "chr_i4c_gemv_cpu failed (", rc, "): ", gemv_err(rc));
  return y;
}

std::string nf4_cpu_isa() { return std::string(chr_nf4_cpu_isa()); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nf4_gemm", &nf4_gemm,
        "fused CPU NF4 GEMV: y[M,N] = dequant_nf4(packed,scale) @ x[K,N]",
        py::arg("packed"), py::arg("scale"), py::arg("x"), py::arg("M"), py::arg("K"),
        py::arg("K_pad"));
  m.def("i4c_gemm", &i4c_gemm,
        "fused CPU i4c GEMV: y[M,1] = dequant_i4c(packed,scale) @ x[K,1]",
        py::arg("packed"), py::arg("scale"), py::arg("x"), py::arg("M"), py::arg("K"),
        py::arg("K_pad"));
  m.def("isa", &nf4_cpu_isa, "avx2 or scalar");
}
