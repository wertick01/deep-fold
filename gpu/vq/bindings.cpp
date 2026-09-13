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
    return "N != 1 (prefill not implemented)";
  case -3:
    return "invalid M/K or null index/book";
  case -4:
    return "K_pad != 8*ceil(K/8)";
  case -5:
    return "index, book, or x not 16-byte aligned";
  case -6:
    return "CUDA launch / func attribute failed";
  default:
    return "unknown error";
  }
}

} // namespace

torch::Tensor vq_gemm(torch::Tensor index, torch::Tensor book, torch::Tensor x,
                       int64_t M, int64_t K, int64_t K_pad) {
  TORCH_CHECK(index.is_cuda() && book.is_cuda() && x.is_cuda(),
              "index, book, and x must be CUDA tensors");
  TORCH_CHECK(index.scalar_type() == torch::kByte, "index must be uint8");
  TORCH_CHECK(book.scalar_type() == torch::kFloat16, "book must be float16");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(M >= 1 && K >= 1, "M and K must be >= 1");

  index = index.contiguous();
  book = book.contiguous();
  x = x.contiguous();
  if (x.dim() == 1) {
    x = x.unsqueeze(1);
  }
  TORCH_CHECK(x.dim() == 2, "x must be [K, N] or [K]");
  TORCH_CHECK(x.size(0) == K, "x.size(0) must equal K");

  const int64_t k_pad_want = 8 * ((K + 7) / 8);
  TORCH_CHECK(K_pad == k_pad_want, "K_pad must be 8*ceil(K/8) = ", k_pad_want,
              ", got ", K_pad);
  TORCH_CHECK(index.numel() >= M * (K_pad / 8) * 2, "index has ", index.numel(),
              " bytes, the kernel reads M*(K_pad/8)*2 = ",
              M * (K_pad / 8) * 2);
  TORCH_CHECK(book.numel() == 2 * 256 * 8, "book has ", book.numel(),
              " elements, want 2*256*8 = 4096");

  const int32_t N = static_cast<int32_t>(x.size(1));
  auto y = torch::empty({M, x.size(1)}, x.options());

  chr_vq_dev_t w{};
  w.M = static_cast<int32_t>(M);
  w.K = static_cast<int32_t>(K);
  w.K_pad = static_cast<int32_t>(K_pad);
  w.index = index.data_ptr<uint8_t>();
  w.book = reinterpret_cast<const uint16_t *>(book.data_ptr<at::Half>());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int rc = chr_vq_gemm(&w, x.data_ptr(), y.data_ptr(), N, stream);
  TORCH_CHECK(rc == 0, "chr_vq_gemm failed (", rc, "): ", gemm_err(rc));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_gemm", &vq_gemm,
        "chr_vq_gemm: y[M,N] = dequant_vq(index,book) @ x[K,N], N=1");
}
