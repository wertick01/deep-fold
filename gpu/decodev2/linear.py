"""NF4 linear + uploaded synthetic weights.

CUDA backends: ``mma`` (default ``chr_nf4_gemm``) or ``gemv`` (``chr_nf4_gemv``).
``CHR_NF4_DECODE=gemv`` selects the CUDA-core path. TokenLoop is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from gpu.tests.nf4_oracle import decode_nf4, matmul_f32

from .embed import PackedEmbed, bind_embed
from .plan import ArchSpec
from .synth import Nf4Matrix, SynthLayer, SynthModel

__all__ = [
    "DeviceLinear",
    "DeviceLayer",
    "DeviceWeights",
    "PackedEmbed",
    "linear_backend",
    "nf4_linear",
    "set_linear_backend",
]

_BACKEND = os.environ.get("CHR_NF4_DECODE", "mma").strip().lower() or "mma"


def linear_backend() -> str:
    return _BACKEND


def set_linear_backend(name: str) -> None:
    global _BACKEND
    key = str(name).strip().lower()
    if key not in {"mma", "gemv"}:
        raise ValueError(f"linear backend {name!r} not in mma|gemv")
    _BACKEND = key


@dataclass
class DeviceLinear:
    packed: torch.Tensor
    scale: torch.Tensor
    M: int
    K: int
    K_pad: int
    bias: torch.Tensor | None
    _w_cpu: np.ndarray | None = None

    @classmethod
    def from_nf4(cls, mat: Nf4Matrix, device: torch.device, dtype: torch.dtype) -> "DeviceLinear":
        packed = torch.from_numpy(np.ascontiguousarray(mat.packed)).to(
            device=device, dtype=torch.uint8
        )
        scale = torch.from_numpy(np.ascontiguousarray(mat.scale)).to(
            device=device, dtype=torch.float16
        )
        bias = None
        if mat.bias is not None:
            bias = torch.from_numpy(np.ascontiguousarray(mat.bias)).to(device=device, dtype=dtype)
        w_cpu = None
        if device.type != "cuda":
            w_cpu = decode_nf4(mat.packed, mat.scale, mat.M, mat.K)
        return cls(
            packed=packed,
            scale=scale,
            M=mat.M,
            K=mat.K,
            K_pad=mat.K_pad,
            bias=bias,
            _w_cpu=w_cpu,
        )

    @classmethod
    def from_compressed(cls, lin) -> "DeviceLinear":
        """Alias a loaded ``CompressedLinear``; no copy of packed bytes."""
        packed = lin.packed
        scale = lin.scale
        if packed is None or int(packed.numel()) == 0:
            raise RuntimeError("CompressedLinear has no packed weights")
        bias = getattr(lin, "bias", None)
        if bias is not None and int(bias.numel()) == 0:
            bias = None
        return cls(
            packed=packed,
            scale=scale,
            M=int(lin.M),
            K=int(lin.K),
            K_pad=int(lin.K_pad),
            bias=bias,
            _w_cpu=None,
        )


def nf4_linear(mat: DeviceLinear, x: torch.Tensor) -> torch.Tensor:
    """``x`` is ``[N, K]`` → ``[N, M]`` on ``x.device``."""
    if x.shape[-1] != mat.K:
        raise ValueError(f"x K={x.shape[-1]} != {mat.K}")
    if x.device.type == "cuda":
        row = x.reshape(-1, mat.K)
        n = int(row.size(0))
        # N=1 row-major [1,K] is already the K-vector the kernels want. Avoid
        # .t().contiguous(), which copied every decode linear.
        if n == 1 and row.is_contiguous():
            xk = row.reshape(mat.K)
        else:
            xk = row.t().contiguous()
        if _BACKEND == "gemv" and n == 1:
            from gpu.nf4 import nf4_gemv

            col = nf4_gemv(mat.packed, mat.scale, xk, mat.M, mat.K, mat.K_pad)
        else:
            from gpu.nf4 import nf4_gemm

            col = nf4_gemm(mat.packed, mat.scale, xk, mat.M, mat.K, mat.K_pad)
        if n == 1:
            y = col.view(mat.M).unsqueeze(0)
        else:
            y = col.t().contiguous()
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
    else:
        if mat._w_cpu is None:
            raise RuntimeError("CPU linear missing decoded W")
        xn = x.detach().float().cpu().numpy()
        y_np = matmul_f32(mat._w_cpu, xn.T.copy()).T
        y = torch.from_numpy(y_np).to(device=x.device, dtype=x.dtype)
    if mat.bias is not None:
        y = y + mat.bias.to(dtype=y.dtype)
    return y


@dataclass
class DeviceLayer:
    norm1: torch.Tensor
    norm2: torch.Tensor
    q: DeviceLinear | None
    k: DeviceLinear | None
    v: DeviceLinear | None
    wqkv: DeviceLinear | None
    o: DeviceLinear
    gate: DeviceLinear
    up: DeviceLinear
    down: DeviceLinear


@dataclass
class DeviceWeights:
    spec: ArchSpec
    embed: torch.Tensor | PackedEmbed
    final_norm: torch.Tensor
    lm_head: DeviceLinear
    layers: list[DeviceLayer]

    @classmethod
    def from_synth(
        cls,
        model: SynthModel,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "DeviceWeights":
        device = torch.device(device)

        def upload(m: Nf4Matrix) -> DeviceLinear:
            return DeviceLinear.from_nf4(m, device, dtype)

        def layer(ly: SynthLayer) -> DeviceLayer:
            return DeviceLayer(
                norm1=torch.from_numpy(ly.norm1.copy()).to(device=device, dtype=dtype),
                norm2=torch.from_numpy(ly.norm2.copy()).to(device=device, dtype=dtype),
                q=None if ly.q is None else upload(ly.q),
                k=None if ly.k is None else upload(ly.k),
                v=None if ly.v is None else upload(ly.v),
                wqkv=None if ly.wqkv is None else upload(ly.wqkv),
                o=upload(ly.o),
                gate=upload(ly.gate),
                up=upload(ly.up),
                down=upload(ly.down),
            )

        return cls(
            spec=model.spec,
            embed=torch.from_numpy(model.embed.copy()).to(device=device, dtype=dtype),
            final_norm=torch.from_numpy(model.final_norm.copy()).to(device=device, dtype=dtype),
            lm_head=upload(model.lm_head),
            layers=[layer(ly) for ly in model.layers],
        )

    @classmethod
    def from_loaded(cls, model, spec: ArchSpec, dtype: torch.dtype = torch.bfloat16) -> "DeviceWeights":
        """Bind Decode V2 linears to a ``load_model`` tree. Packed buffers are views.

        Embed stays NF4 rows (``PackedEmbed``). There is no dense vocab table.
        """
        plan = getattr(model, "deepfold_plan", None)
        if plan is None:
            raise RuntimeError("model has no deepfold_plan; call gpu.host.load_model first")
        get = model.get_submodule
        fused = plan.attn == "fused"

        def lin(name: str) -> DeviceLinear:
            return DeviceLinear.from_compressed(get(name))

        layers: list[DeviceLayer] = []
        for lp in plan.layers:
            g = lp.gemms
            layers.append(
                DeviceLayer(
                    norm1=get(lp.norm1).weight.to(dtype=dtype),
                    norm2=get(lp.norm2).weight.to(dtype=dtype),
                    q=None if fused else lin(g["q"]),
                    k=None if fused else lin(g["k"]),
                    v=None if fused else lin(g["v"]),
                    wqkv=lin(g["qkv"]) if fused else None,
                    o=lin(g["o"]),
                    gate=lin(g["gate"]),
                    up=lin(g["up"]),
                    down=lin(g["down"]),
                )
            )
        embed = bind_embed(get(plan.embed), spec, dtype)
        final_norm = get(plan.final_norm).weight.to(dtype=dtype)
        return cls(
            spec=spec,
            embed=embed,
            final_norm=final_norm,
            lm_head=lin(plan.lm_head),
            layers=layers,
        )
