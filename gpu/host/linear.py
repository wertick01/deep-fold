"""``CompressedLinear``: an ``nn.Linear`` seat for the NF4 kernel.

The module holds the file's bytes (``packed``, ``scale``) and nothing else. No
``[out, in]`` parameter is ever registered, so ``model.to("cuda")`` cannot
resurrect a BF16 weight (token-loop.md §3) and nothing dequantizes a draft W
into HBM (stitch-gpu.md).

Transposition lives here, as gpu-abi.md §6 requires: HuggingFace hands us
``[..., K]``, the kernel wants ``[K, N]`` and answers ``[M, N]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._deps import ChrMatrix, nf4_gemm
from .host_image import HostImage, cpu_is_pinned, maybe_pin
from gpu.nf4.plan import LIVE_MAX_N

__all__ = ["CompressedLinear", "nf4_linear", "k_pad"]

GROUP_SIZE = 64


def _pageable_cpu(t: torch.Tensor) -> torch.Tensor:
    """CPU tensor that is never pinned (clone of a pinned tensor can stay pinned)."""
    src = t.detach()
    if src.device.type != "cpu":
        src = src.cpu()
    if not cpu_is_pinned(src):
        return src
    out = torch.empty(src.shape, dtype=src.dtype, device="cpu")
    out.copy_(src)
    return out


def k_pad(k: int) -> int:
    """``64 * ceil(K / 64)`` -- stitch-gpu.md size formulas."""
    return GROUP_SIZE * ((int(k) + GROUP_SIZE - 1) // GROUP_SIZE)


def nf4_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = dequant_nf4(packed, scale) @ x`` for activations shaped ``[..., K]``.

    HuggingFace hands ``[..., K]``; the kernel wants ``[K, N]`` and returns
    ``[M, N]``. ``N = prod(lead)``: decode launch when ``N==1`` (a view, no
    transpose copy); ``N<=LIVE_MAX_N`` (32) is one prefill GEMM; above that
    the host chunks. The host does not pad tails to 16. n64 is plan-only.
    """
    if x.shape[-1] != K:
        raise ValueError(f"x has {x.shape[-1]} features, this linear takes K={K}")
    lead = tuple(x.shape[:-1])
    n = 1
    for d in lead:
        n *= int(d)
    if n < 1:
        raise ValueError(f"CompressedLinear: empty leading dims x{tuple(x.shape)}")
    if x.dtype is not torch.bfloat16:
        x = x.to(torch.bfloat16)

    if n == 1:
        # [..., K] with prod(lead) == 1 -> [K, 1]: same bytes, decode layout.
        xk = x.reshape(K, 1)
        y = nf4_gemm(packed, scale, xk, M, K, K_pad)  # bf16 [M, 1]
        out = y.view(M)
        if bias is not None:
            out = out + bias
        return out.view(*lead, M)

    x2 = x.contiguous().view(n, K)
    if n <= LIVE_MAX_N:
        xk = x2.transpose(0, 1).contiguous()  # [K, N]
        y = nf4_gemm(packed, scale, xk, M, K, K_pad)  # bf16 [M, N]
    else:
        cols = []
        for start in range(0, n, LIVE_MAX_N):
            sl = x2[start : start + LIVE_MAX_N]
            xk = sl.transpose(0, 1).contiguous()
            cols.append(nf4_gemm(packed, scale, xk, M, K, K_pad))
        y = torch.cat(cols, dim=1)
    out = y.transpose(0, 1).contiguous()  # [N, M]
    if bias is not None:
        out = out + bias
    return out.view(*lead, M)


class CompressedLinear(nn.Module):
    """Drop-in for ``nn.Linear`` whose weight lives as CHR0 NF4 bytes.

    Buffers are 0-numel until :meth:`attach` gets a ``ChrMatrix``, so the
    replacement pass costs nothing and a skeleton can be built on ``meta``
    first. ``weight`` is a property returning an empty tensor: utilities that
    poke ``module.weight.dtype`` keep working without ``M*K`` ever existing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features < 1 or self.out_features < 1:
            raise ValueError(f"bad shape: in={in_features} out={out_features}")
        self.K = self.in_features
        self.M = self.out_features
        self.K_pad = k_pad(self.K)
        self.n_groups = self.K_pad // GROUP_SIZE
        self.compute_dtype = dtype

        dev = torch.device(device) if device is not None else torch.device("cpu")
        # persistent=False: these are file bytes, not a checkpoint of ours.
        self.register_buffer("packed", torch.empty(0, dtype=torch.uint8, device=dev), persistent=False)
        self.register_buffer("scale", torch.empty(0, dtype=torch.float16, device=dev), persistent=False)
        if bias:
            self.register_buffer("bias", torch.empty(0, dtype=dtype, device=dev), persistent=False)
        else:
            self.register_buffer("bias", None)
        self.chr_name: str | None = None
        self.host_image: HostImage | None = None
        self._home: str | None = None
        self.cpu_codec: str = "nf4"

    # --- the weight that is not there ------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """A 0-numel stand-in with the right dtype/device. Never ``[M, K]``.

        Assigning to it raises (``property`` has no setter) -- that is the
        point: a stray ``module.weight = ...`` from a generic utility must fail
        loudly instead of allocating ``out x in`` BF16.
        """
        return torch.empty(0, dtype=self.compute_dtype, device=self.packed.device)

    @property
    def is_loaded(self) -> bool:
        return int(self.packed.numel()) > 0 or self.host_image is not None

    @property
    def nbytes(self) -> int:
        """Device bytes held by this layer's weight. HOST overflow is 0."""
        return int(self.packed.numel()) + int(self.scale.numel()) * 2

    @property
    def home(self) -> str:
        """``device`` / ``host`` (CopyRing) / ``cpu`` (pageable suffix) / ``empty``."""
        if self._home is not None:
            return self._home
        if int(self.packed.numel()) > 0:
            return "device"
        if self.host_image is not None:
            return "host"
        return "empty"

    # --- loading ----------------------------------------------------------
    def attach(self, matrix: ChrMatrix) -> None:
        """Point this layer at an already materialized CHR0 matrix.

        The tensors are shared, not copied: two tied layers (``lm_head`` and
        ``embed_tokens`` on Qwen2.5-3B) can hold the same ``ChrMatrix`` without
        a second copy in VRAM.
        """
        if (int(matrix.M), int(matrix.K)) != (self.M, self.K):
            raise ValueError(
                f"{matrix.name}: matrix is [{matrix.M},{matrix.K}], "
                f"this linear is [{self.M},{self.K}]"
            )
        if int(matrix.K_pad) != self.K_pad:
            raise ValueError(f"{matrix.name}: K_pad {matrix.K_pad} != {self.K_pad}")
        if matrix.packed.dtype is not torch.uint8 or matrix.scale.dtype is not torch.float16:
            raise TypeError(
                f"{matrix.name}: expected uint8 packed / float16 scale, "
                f"got {matrix.packed.dtype} / {matrix.scale.dtype}"
            )
        self.packed = matrix.packed
        self.scale = matrix.scale
        self.chr_name = matrix.name
        self.host_image = None
        self._home = "device"
        self.cpu_codec = "nf4"

    def attach_host(self, image: HostImage) -> None:
        """Keep packed/scale empty; the matrix lives on pinned host (CopyRing is H2-4).

        No H2D. ``maybe_pin`` runs here if the arena is still pageable.
        """
        if (int(image.M), int(image.K)) != (self.M, self.K):
            raise ValueError(
                f"HostImage is [{image.M},{image.K}], this linear is [{self.M},{self.K}]"
            )
        if int(image.K_pad) != self.K_pad:
            raise ValueError(f"HostImage K_pad {image.K_pad} != {self.K_pad}")
        arena = image.arena
        if arena.is_cpu and not cpu_is_pinned(arena):
            pinned = maybe_pin(arena)
            if pinned is not arena:
                image = HostImage(pinned, image.M, image.K, image.K_pad)
        if int(self.packed.numel()) != 0:
            dev = self.packed.device
            self.packed = torch.empty(0, dtype=torch.uint8, device=dev)
            self.scale = torch.empty(0, dtype=torch.float16, device=dev)
        self.host_image = image
        self._home = "host"
        self.cpu_codec = "nf4"

    def attach_cpu(self, matrix: ChrMatrix, *, codec: str = "nf4") -> None:
        """Packed+scale on pageable CPU. No pin, no H2D, not CopyRing.

        ``home="cpu"``. A CUDA ``ChrMatrix`` is copied to a new pageable
        allocation so attach never leaves a pinned or device buffer.
        ``codec="i4c"`` transcodes the NF4 table in RAM (sidecar; not ``.chr``).
        """
        if (int(matrix.M), int(matrix.K)) != (self.M, self.K):
            raise ValueError(
                f"{matrix.name}: matrix is [{matrix.M},{matrix.K}], "
                f"this linear is [{self.M},{self.K}]"
            )
        if int(matrix.K_pad) != self.K_pad:
            raise ValueError(f"{matrix.name}: K_pad {matrix.K_pad} != {self.K_pad}")
        if matrix.packed.dtype is not torch.uint8 or matrix.scale.dtype is not torch.float16:
            raise TypeError(
                f"{matrix.name}: expected uint8 packed / float16 scale, "
                f"got {matrix.packed.dtype} / {matrix.scale.dtype}"
            )
        packed = _pageable_cpu(matrix.packed)
        scale = _pageable_cpu(matrix.scale)
        if cpu_is_pinned(packed) or cpu_is_pinned(scale):
            raise RuntimeError(
                f"{matrix.name}: attach_cpu must not pin packed/scale "
                "(CPU suffix is pageable; pin fights the gpu-mode ring)"
            )
        self.packed = packed
        self.scale = scale
        self.chr_name = matrix.name
        self.host_image = None
        self._home = "cpu"
        self.cpu_codec = "nf4"
        codec = str(codec or "nf4")
        if codec not in ("nf4", "i4c"):
            raise ValueError(f"attach_cpu codec={codec!r}; expected nf4 or i4c")
        if codec == "i4c":
            from .i4c import pack_i4c_from_nf4

            p, s, kp = pack_i4c_from_nf4(self.packed, self.scale, self.M, self.K)
            if int(kp) != int(self.K_pad):
                raise ValueError(
                    f"{matrix.name}: i4c K_pad {kp} != NF4 K_pad {self.K_pad}"
                )
            self.packed = p
            self.scale = s
            self.cpu_codec = "i4c"
            import gc

            gc.collect()

    def set_bias(self, bias: torch.Tensor) -> None:
        if self.bias is None:
            raise ValueError(f"{self.chr_name or self}: built with bias=False")
        if bias.shape != (self.M,):
            raise ValueError(f"bias {tuple(bias.shape)} != {(self.M,)}")
        dest = self.packed.device if self.home == "cpu" else bias.device
        self.bias = bias.to(device=dest, dtype=self.compute_dtype)

    # --- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.home == "host" or (
            self.host_image is not None and int(self.packed.numel()) == 0
        ):
            raise RuntimeError(
                f"CompressedLinear[{self.M},{self.K}] "
                f"({self.chr_name or 'unnamed'}) is host-resident; "
                "CopyRing (H2-4) is required, refuse silent H2D in forward"
            )
        if not self.is_loaded:
            raise RuntimeError(
                f"CompressedLinear[{self.M},{self.K}] "
                f"({self.chr_name or 'unnamed'}) has no weights: call load_chr_nf4 first"
            )
        if self.home == "cpu":
            if x.device.type == "cuda":
                raise RuntimeError(
                    f"CompressedLinear[{self.M},{self.K}] "
                    f"({self.chr_name or 'unnamed'}) is CPU-resident; "
                    "refuse silent H2D in forward (TokenLoop bounces the hidden state)"
                )
            if getattr(self, "cpu_codec", "nf4") == "i4c":
                from .i4c import i4c_linear_cpu

                return i4c_linear_cpu(
                    x, self.packed, self.scale, self.M, self.K, self.K_pad, self.bias
                )
            from .cpu_linear import nf4_linear_cpu

            return nf4_linear_cpu(
                x, self.packed, self.scale, self.M, self.K, self.K_pad, self.bias
            )
        return nf4_linear(x, self.packed, self.scale, self.M, self.K, self.K_pad, self.bias)

    def extra_repr(self) -> str:
        codec = "nf4"
        if self.home == "cpu":
            codec = getattr(self, "cpu_codec", "nf4")
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, codec={codec}-g{GROUP_SIZE}, "
            f"K_pad={self.K_pad}, home={self.home}, loaded={self.is_loaded}"
        )
