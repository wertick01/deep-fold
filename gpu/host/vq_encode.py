"""GPU residual VQ 2x8 encode. Same CHR0 layout as ``docs/spec/vq.md``, not the Go RNG.

The Go package is the bit-exact compressor. This path exists because that
MSE-only k-means (no scales, no activations) destroys chat on 3B: cosine ~0.94
vs NF4 ~0.996. We keep M=2, k=256, B=8 so the Ampere kernel can load the file.
Optional ``channel_w`` is an imatrix (per-input-dim second moment). Optional
column scales are SmoothQuant: encode ``W * s``, decode as ``hat / s`` at the
previous RMSNorm or by dividing ``x``.
"""

from __future__ import annotations

import torch

from .vq_blobs import (
    CODEBOOK_SIZE,
    N_CODEBOOKS,
    VQ_GROUP_SIZE,
    k_pad_vq,
    reconstruct_vq,
)

__all__ = ["encode_vq", "groups_from_weight", "weight_from_groups"]


def groups_from_weight(weight: torch.Tensor) -> torch.Tensor:
    """``[M, K]`` -> ``[M, G, 8]`` float32, zero-pad along ``K``."""
    if weight.dim() != 2:
        raise ValueError(f"weight must be [M, K], got {tuple(weight.shape)}")
    m, k = int(weight.shape[0]), int(weight.shape[1])
    g = k_pad_vq(k) // VQ_GROUP_SIZE
    out = torch.zeros(m, g, VQ_GROUP_SIZE, dtype=torch.float32, device=weight.device)
    flat = out.reshape(m, g * VQ_GROUP_SIZE)
    flat[:, :k] = weight.float()
    return out


def weight_from_groups(groups: torch.Tensor, k: int) -> torch.Tensor:
    m, g, b = groups.shape
    if b != VQ_GROUP_SIZE:
        raise ValueError(f"groups last dim {b} != {VQ_GROUP_SIZE}")
    return groups.reshape(m, g * VQ_GROUP_SIZE)[:, :k]


def _assign(
    x: torch.Tensor,
    code: torch.Tensor,
    w: torch.Tensor | None,
    chunk: int = 262144,
) -> torch.Tensor:
    """Argmin L2 (optionally weighted) of ``x [N,8]`` against ``code [256,8]``.

    Weighted distance is ``sum_d w_d (x_d - c_d)^2`` = ``w @ c^2 - 2 (w*x) @ c``
    (the ``||x||`` term does not affect argmin). Chunked so ``[T, 256]`` stays
    off the 12 GB card while a 3B ``down_proj`` is encoded.
    """
    n = x.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=x.device)
    c2 = (code * code)
    if w is None:
        c2_sum = c2.sum(-1)
        for off in range(0, n, chunk):
            sl = slice(off, min(off + chunk, n))
            dist = c2_sum.unsqueeze(0) - 2.0 * (x[sl] @ code.T)
            out[sl] = dist.argmin(-1)
        return out
    c2_t = c2.T
    for off in range(0, n, chunk):
        sl = slice(off, min(off + chunk, n))
        ww = w[sl]
        xw = x[sl] * ww
        dist = ww @ c2_t - 2.0 * (xw @ code.T)
        out[sl] = dist.argmin(-1)
    return out


def _update(x: torch.Tensor, assign: torch.Tensor, w: torch.Tensor | None, k: int) -> torch.Tensor:
    d = x.shape[1]
    idx = assign.unsqueeze(1).expand(-1, d)
    if w is None:
        sums = torch.zeros(k, d, dtype=torch.float32, device=x.device)
        sums.scatter_add_(0, idx, x)
        counts = torch.zeros(k, dtype=torch.float32, device=x.device)
        counts.scatter_add_(0, assign, torch.ones_like(assign, dtype=torch.float32))
        live = counts > 0
        code = torch.zeros(k, d, dtype=torch.float32, device=x.device)
        code[live] = sums[live] / counts[live].unsqueeze(1)
        return code, counts
    wx = x * w
    sums = torch.zeros(k, d, dtype=torch.float32, device=x.device)
    sums.scatter_add_(0, idx, wx)
    wsum = torch.zeros(k, d, dtype=torch.float32, device=x.device)
    wsum.scatter_add_(0, idx, w)
    counts = torch.zeros(k, dtype=torch.float32, device=x.device)
    counts.scatter_add_(0, assign, torch.ones_like(assign, dtype=torch.float32))
    live = (wsum.abs() > 1e-12).any(-1)
    code = torch.zeros(k, d, dtype=torch.float32, device=x.device)
    code[live] = sums[live] / wsum[live].clamp_min(1e-12)
    return code, counts


def _resplit(code: torch.Tensor, counts: torch.Tensor) -> None:
    empty = counts <= 0
    if not bool(empty.any()):
        return
    fat = int(counts.argmax().item())
    noise = (torch.rand_like(code[empty]) * 2.0 - 1.0) * 1e-5
    code[empty] = code[fat].unsqueeze(0) + noise


def _kmeans(
    x: torch.Tensor,
    *,
    k: int,
    iters: int,
    w: torch.Tensor | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    n_sub = min(n, 65536)
    if n_sub == n:
        sub = x
    else:
        pick = torch.randperm(n, generator=g)[:n_sub]
        sub = x[pick.to(x.device)]
    # k-means++ on the subset (unweighted init; Lloyd uses ``w``).
    sub_cpu = sub.detach().float()
    chosen = torch.empty(k, sub_cpu.shape[1], dtype=torch.float32, device=x.device)
    j0 = int(torch.randint(0, n_sub, (1,), generator=g).item())
    chosen[0] = sub_cpu[j0]
    dist = torch.full((n_sub,), 1e30, dtype=torch.float32, device=x.device)
    for t in range(1, k):
        d = ((sub_cpu - chosen[t - 1]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        total = float(dist.sum().item())
        if total <= 0:
            chosen[t] = sub_cpu[int(torch.randint(0, n_sub, (1,), generator=g).item())]
            continue
        r = float(torch.rand(1, generator=g).item()) * total
        acc = dist.cumsum(0)
        pick_i = int(torch.searchsorted(acc, torch.tensor(r, device=x.device)).item())
        pick_i = min(max(pick_i, 0), n_sub - 1)
        chosen[t] = sub_cpu[pick_i]
    code = chosen
    assign = _assign(x, code, w)
    for _ in range(max(0, int(iters))):
        code, counts = _update(x, assign, w, k)
        _resplit(code, counts)
        assign = _assign(x, code, w)
    return code, assign


def encode_vq(
    weight: torch.Tensor,
    *,
    iters: int = 20,
    seed: int = 0,
    channel_w: torch.Tensor | None = None,
    column_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``weight [M,K]`` to ``index uint8 [M,G,2]`` and ``book fp16 [2,256,8]``.

    ``channel_w`` is a length-``K`` imatrix (non-negative). ``column_scale`` is
    SmoothQuant ``s`` of length ``K``: we encode ``W * s`` (columns). The caller
    must apply ``1/s`` on the activation side so GEMM stays correct.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be [M, K], got {tuple(weight.shape)}")
    m, k = int(weight.shape[0]), int(weight.shape[1])
    work = weight.float()
    if column_scale is not None:
        if tuple(column_scale.shape) != (k,):
            raise ValueError(f"column_scale {tuple(column_scale.shape)} != ({k},)")
        work = work * column_scale.float().to(work.device).clamp_min(1e-8)
    groups = groups_from_weight(work)
    x = groups.reshape(-1, VQ_GROUP_SIZE)
    w_vec = None
    if channel_w is not None:
        if tuple(channel_w.shape) != (k,):
            raise ValueError(f"channel_w {tuple(channel_w.shape)} != ({k},)")
        g = groups.shape[1]
        ww = torch.ones(g * VQ_GROUP_SIZE, dtype=torch.float32, device=work.device)
        ww[:k] = channel_w.float().to(work.device).clamp_min(0)
        w_g = ww.view(g, VQ_GROUP_SIZE)
        w_vec = w_g.unsqueeze(0).expand(m, -1, -1).reshape(-1, VQ_GROUP_SIZE)

    residual = x.clone()
    codes = []
    assigns = []
    rng = int(seed)
    for book_i in range(N_CODEBOOKS):
        code, assign = _kmeans(
            residual, k=CODEBOOK_SIZE, iters=iters, w=w_vec, seed=rng + book_i * 997
        )
        code_fp16 = code.to(torch.float16)
        if not torch.isfinite(code_fp16.float()).all():
            code_fp16 = code.clamp(-65504, 65504).to(torch.float16)
        codes.append(code_fp16)
        assigns.append(assign.to(torch.uint8))
        residual = residual - code_fp16.float()[assign]

    g = groups.shape[1]
    index = torch.stack(assigns, dim=-1).view(m, g, N_CODEBOOKS).contiguous()
    book = torch.stack(codes, dim=0).contiguous()
    return index, book


def reconstruct_encoded(
    index: torch.Tensor,
    book: torch.Tensor,
    k: int,
    column_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Float32 ``[M,K]``. If ``column_scale`` was used at encode, divide it back."""
    k_pad = index.shape[1] * VQ_GROUP_SIZE
    hat = reconstruct_vq(index, book, index.shape[0], k, k_pad)
    if column_scale is not None:
        hat = hat / column_scale.float().to(hat.device).clamp_min(1e-8).unsqueeze(0)
    return hat
