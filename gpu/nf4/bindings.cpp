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
    return "N not in [1,16] (host must chunk N>16)";
  case -3:
    return "invalid M/K or null packed/scale";
  case -4:
    return "K_pad != 64*ceil(K/64)";
  case -5:
    return "packed or x not 16-byte aligned";
  case -6:
    return "CUDA launch / func attribute failed";
  default:
    return "unknown error";
  }
}

} // namespace

torch::Tensor nf4_gemm(torch::Tensor packed, torch::Tensor scale, torch::Tensor x,
                        int64_t M, int64_t K, int64_t K_pad) {
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

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc =
      chr_nf4_gemm(&w, x.data_ptr(), y.data_ptr(), N, stream);
  TORCH_CHECK(rc == 0, "chr_nf4_gemm failed (", rc, "): ", gemm_err(rc));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nf4_gemm", &nf4_gemm,
        "chr_nf4_gemm: y[M,N] = dequant_nf4(packed,scale) @ x[K,N], N in 1..16");
}
