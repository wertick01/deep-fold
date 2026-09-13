"""Our own token loop: preallocated KV, one forward per step, greedy sampling.

``transformers.generate`` is not used and not patched. It grows its cache with
``torch.cat``, allocates freely on the token path and occasionally moves logits
through the CPU -- all three are exactly what ``docs/token-loop.md`` §1 and §5
say to avoid on a 12 GB card. So the layer body is spelled out here, in the
order of §2, with every long-lived tensor owned by Python:

    h = rms(x)                      # PyTorch
    q, k, v = qkv(h)                # NF4 kernel
    q, k = rope(q, k, pos)          # PyTorch, from a precomputed table
    kv[layer, pos] = k, v           # slot assignment, no cat
    a = sdpa(q, kv[:seq])           # PyTorch
    x += o(a)                       # NF4 kernel
    h = rms(x); x += down(silu(gate(h)) * up(h))

Activations are ``[N, hidden]``; the kernel's ``[K, N]`` is one ``.t()`` away
(:class:`gpu.loop.graph.GemmGroup`). ``N == 1`` for decode, and also for prefill
while the wave-2 GEMM is decode-only -- :func:`gpu.loop.graph.nf4_max_n` asks the
kernel instead of assuming, and :attr:`TokenLoop.prefill_chunk` reports the
answer. Either way the prompt is walked *once*: the KV cache is filled slot by
slot and never recomputed, which is the difference that matters for tok/s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .graph import Gemm, GemmGroup, GraphedGemmGroup, capture, nf4_max_n
from .kv_cache import KVCache

__all__ = ["TokenLoop", "Generation", "rms_norm_exact", "split_internlm_wqkv"]

MIB = 1024 * 1024


# --------------------------------------------------------------------------- #
# the glue PyTorch keeps
# --------------------------------------------------------------------------- #


def rms_norm_exact(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``Qwen2RMSNorm``, operation for operation: accumulate in fp32, scale in bf16."""
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return w * v.to(x.dtype)


def _rms_norm_fast(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """One fused ATen call. Same formula, one rounding step fewer."""
    return F.rms_norm(x, w.shape, w, eps)


def split_internlm_wqkv(
    y: torch.Tensor, n_q: int, n_kv: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undo InternLM2's fused ``wqkv`` packing.

    The GEMM writes ``[N, (n_q + 2*n_kv)*head_dim]``. InternLM2 stores, for
    each KV head: ``n_q/n_kv`` query heads, then K, then V. See
    ``modeling_internlm2.py`` (``rearrange(..., gs=2+num_key_value_groups)``).
    """
    n = int(y.shape[0])
    n_rep = n_q // n_kv
    packed = y.view(n, n_kv, 2 + n_rep, head_dim)
    q = packed[:, :, :n_rep, :].reshape(n, n_q, head_dim)
    k = packed[:, :, -2, :].contiguous()
    v = packed[:, :, -1, :].contiguous()
    return q, k, v


def _rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``t`` is ``[N, heads, head_dim]``; ``cos``/``sin`` are ``[N, 1, head_dim]``.

    ``rotate_half`` + fused multiply-add, i.e. HuggingFace's
    ``apply_rotary_pos_emb`` with the head axis already in the middle.
    """
    d = t.shape[-1] // 2
    rot = torch.cat((-t[..., d:], t[..., :d]), dim=-1)
    return t * cos + rot * sin


def _rope_tables(
    model, max_seq: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos``/``sin`` for positions ``0..max_seq-1``, computed once.

    RoPE is position-wise, so a table is bit-identical to calling the module per
    step and costs 512 KiB at ``max_seq=1024``. It also removes the classic
    "``inv_freq`` was built on CPU and only decode disagrees" bug
    (token-loop.md §6.3): the buffer is whatever ``load_chr_nf4`` rebuilt on the
    device, used here and nowhere else.

    Qwen keeps one ``model.rotary_emb``. InternLM2 keeps one per layer; they
    share the same dim/base, so layer 0's module is the table.
    """
    base = model.model
    rot = getattr(base, "rotary_emb", None)
    if rot is None:
        layer0 = base.layers[0]
        attn = getattr(layer0, "self_attn", None) or layer0.attention
        rot = attn.rotary_emb
    pos = torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0)
    ref = torch.zeros((1, 1, 1), dtype=dtype, device=device)
    cos, sin = rot(ref, pos)
    cos, sin = cos[0].to(dtype).contiguous(), sin[0].to(dtype).contiguous()
    if cos.shape[0] != max_seq:
        raise RuntimeError(f"rope table is {tuple(cos.shape)}, expected [{max_seq}, head_dim]")
    return cos, sin


def _sdpa_has_gqa() -> bool:
    q = torch.zeros((1, 2, 1, 4))
    k = torch.zeros((1, 1, 2, 4))
    try:
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
    except TypeError:
        return False
    return True


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


@dataclass
class Generation:
    """One greedy request. Times are kept apart on purpose (token-loop.md §6.2)."""

    prompt_len: int
    tokens: list[int] = field(default_factory=list)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    decode_steps: int = 0
    stop_token: int | None = None
    prefill_chunk: int = 1
    graph: str = "off"

    @property
    def decode_tok_s(self) -> float:
        return self.decode_steps / (self.decode_ms / 1000.0) if self.decode_ms > 0 else 0.0

    @property
    def ms_per_token(self) -> float:
        return self.decode_ms / self.decode_steps if self.decode_steps else 0.0


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


class TokenLoop:
    """Drives a loaded ``gpu.host`` skeleton. Allocates once, in ``__init__``.

    ``model`` must already be through ``load_chr_nf4`` -- the loop reads the NF4
    buffers out of the modules once (:class:`~gpu.loop.graph.Gemm`) and then never
    touches ``nn.Module`` attribute lookup on the token path again.
    """

    def __init__(
        self,
        model,
        *,
        max_seq: int = 1024,
        norm: str = "exact",
        overlap: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        cfg = model.config
        base = model.model
        self.model = model
        self.max_seq = int(max_seq)
        self.n_layers = int(cfg.num_hidden_layers)
        self.hidden = int(cfg.hidden_size)
        self.n_q = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.eps = float(cfg.rms_norm_eps)
        self.vocab = int(cfg.vocab_size)

        layer0 = base.layers[0]
        self._fused_qkv = hasattr(layer0, "attention") and hasattr(layer0.attention, "wqkv")
        if self._fused_qkv:
            attn0 = layer0.attention
            self.embed = base.tok_embeddings
            self._lm = model.output
            self._norm1 = [layer.attention_norm.weight for layer in base.layers]
            self._norm2 = [layer.ffn_norm.weight for layer in base.layers]
        else:
            attn0 = layer0.self_attn
            self.embed = base.embed_tokens
            self._lm = model.lm_head
            self._norm1 = [layer.input_layernorm.weight for layer in base.layers]
            self._norm2 = [layer.post_attention_layernorm.weight for layer in base.layers]

        self.head_dim = int(getattr(attn0, "head_dim", self.hidden // self.n_q))
        self.scaling = float(getattr(attn0, "scaling", self.head_dim**-0.5))
        self.q_dim = self.n_q * self.head_dim
        self.n_rep = self.n_q // self.n_kv
        if self.n_q % self.n_kv:
            raise ValueError(f"{self.n_q} q heads do not group into {self.n_kv} kv heads")
        self.qkv_dim = (self.n_q + 2 * self.n_kv) * self.head_dim

        self.final_norm = base.norm.weight
        if self.final_norm.is_meta:
            raise RuntimeError("model.norm.weight is still on meta: load the .chr first")
        self.device = torch.device(device) if device is not None else self.final_norm.device

        if norm not in ("exact", "fast"):
            raise ValueError(f"norm={norm!r}; expected 'exact' or 'fast'")
        self.norm_mode = norm
        self.rms = rms_norm_exact if norm == "exact" else _rms_norm_fast

        # --- weights, flattened out of the module tree ----------------------
        # Two side streams, made once and shared by every group: q/k/v and
        # gate/up are independent and each leaves most of the card idle on its
        # own (see GemmGroup). Set overlap=False for the strictly serial loop.
        self.overlap = bool(overlap) and torch.cuda.is_available()
        self._streams = tuple(torch.cuda.Stream() for _ in range(2)) if self.overlap else ()
        self._groups = self._build_groups()
        self._apply(self._groups)
        self.graph_mode = "off"
        self.graph_error: str | None = None

        # --- everything long-lived, allocated here and nowhere else ---------
        self.kv = KVCache(
            self.n_layers,
            self.max_seq,
            self.n_kv,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.cos, self.sin = _rope_tables(model, self.max_seq, self.device, torch.bfloat16)
        self._tok = torch.zeros(1, dtype=torch.long, device=self.device)
        self._gqa = _sdpa_has_gqa()
        self.prefill_chunk = min(nf4_max_n(16), self.max_seq)

    # --- setup ------------------------------------------------------------
    def _build_groups(self) -> list[GemmGroup]:
        """Contiguous runs of NF4 GEMMs, in execution order (see graph.py)."""
        groups: list[GemmGroup] = []
        st = self._streams
        for li, layer in enumerate(self.model.model.layers):
            if self._fused_qkv:
                sa, mlp = layer.attention, layer.feed_forward
                groups.append(GemmGroup(f"L{li}.qkv", [Gemm.of(sa.wqkv, f"L{li}.wqkv")]))
                groups.append(GemmGroup(f"L{li}.o", [Gemm.of(sa.wo, f"L{li}.o")]))
                groups.append(
                    GemmGroup(
                        f"L{li}.gateup",
                        [
                            Gemm.of(mlp.w1, f"L{li}.gate"),
                            Gemm.of(mlp.w3, f"L{li}.up"),
                        ],
                        st[:1],
                    )
                )
                groups.append(GemmGroup(f"L{li}.down", [Gemm.of(mlp.w2, f"L{li}.down")]))
            else:
                sa, mlp = layer.self_attn, layer.mlp
                groups.append(
                    GemmGroup(
                        f"L{li}.qkv",
                        [
                            Gemm.of(sa.q_proj, f"L{li}.q"),
                            Gemm.of(sa.k_proj, f"L{li}.k"),
                            Gemm.of(sa.v_proj, f"L{li}.v"),
                        ],
                        st,
                    )
                )
                groups.append(GemmGroup(f"L{li}.o", [Gemm.of(sa.o_proj, f"L{li}.o")]))
                groups.append(
                    GemmGroup(
                        f"L{li}.gateup",
                        [
                            Gemm.of(mlp.gate_proj, f"L{li}.gate"),
                            Gemm.of(mlp.up_proj, f"L{li}.up"),
                        ],
                        st[:1],
                    )
                )
                groups.append(GemmGroup(f"L{li}.down", [Gemm.of(mlp.down_proj, f"L{li}.down")]))
        groups.append(GemmGroup("lm_head", [Gemm.of(self._lm, "lm_head")]))
        return groups

    def _apply(self, runners: Sequence[GemmGroup | GraphedGemmGroup]) -> None:
        """Bind one runner per group; four per layer plus the head."""
        if len(runners) != 4 * self.n_layers + 1:
            raise ValueError(f"expected {4 * self.n_layers + 1} runners, got {len(runners)}")
        self._layer = [tuple(runners[4 * li : 4 * li + 4]) for li in range(self.n_layers)]
        self._head = runners[-1]

    @property
    def weight_bytes(self) -> int:
        """Device bytes of NF4 weights, deduplicated by pointer.

        Qwen2.5-3B ties ``lm_head`` to ``embed_tokens``: one ``[151936, 2048]``
        blob serves both, and counting it twice would invent 156 MiB.
        """
        seen = {m.packed.data_ptr(): m.nbytes for g in self._groups for m in g.gemms}
        packed = getattr(self.embed, "packed", None)
        if packed is not None:
            seen[packed.data_ptr()] = self.embed.nbytes
        elif getattr(self.embed, "weight", None) is not None:
            w = self.embed.weight
            seen[w.data_ptr()] = w.numel() * w.element_size()
        return sum(seen.values())

    def reset(self) -> None:
        """Forget the conversation, keep the memory."""
        self.kv.reset()

    @torch.no_grad()
    def capture_graphs(self) -> str:
        """CUDA graph plan A. Idempotent; returns the resulting mode.

        Must be called after :meth:`warmup`. On failure the loop keeps running
        eager and :attr:`graph_error` says why -- eager is the contract, the graph
        is the optimization.
        """
        if self.graph_mode == "linears":
            return self.graph_mode
        runners, mode, err = capture(self._groups)
        self._apply(runners)
        self.graph_mode, self.graph_error = mode, err
        return mode

    def drop_graphs(self) -> str:
        """Back to eager launches (used to check replay against eager)."""
        self._apply(self._groups)
        self.graph_mode, self.graph_error = "off", None
        return self.graph_mode

    # --- the forward -------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start_pos: int, *, logits: bool = True):
        """``ids`` is ``[N]``; writes KV slots ``start_pos..start_pos+N-1``.

        Returns the ``[vocab]`` logits of the *last* position, or ``None`` when
        ``logits=False`` (a prefill chunk that is not the last one -- the point of
        prefill is the cache, not the distribution).
        """
        n = int(ids.numel())
        seq = start_pos + n
        if seq > self.max_seq:
            raise ValueError(f"position {seq} past max_seq={self.max_seq}")

        x = self.embed(ids)  # [n, hidden] bf16
        cos = self.cos[start_pos:seq].unsqueeze(1)  # [n, 1, head_dim]
        sin = self.sin[start_pos:seq].unsqueeze(1)
        mask = None if n == 1 else self._causal_mask(start_pos, seq)
        kv, eps = self.kv, self.eps
        rms, hd, n_kv = self.rms, self.head_dim, self.n_kv

        for li in range(self.n_layers):
            g_qkv, g_o, g_gu, g_down = self._layer[li]

            h = rms(x, self._norm1[li], eps)
            if self._fused_qkv:
                q, k, v = split_internlm_wqkv(
                    g_qkv.run(h)[0], self.n_q, n_kv, hd
                )
            else:
                q, k, v = g_qkv.run(h)
                q = q.reshape(n, self.n_q, hd)
                k = k.reshape(n, n_kv, hd)
                v = v.reshape(n, n_kv, hd)
            q = _rope(q, cos, sin)
            k = _rope(k, cos, sin)
            kv.write(li, start_pos, k, v)
            k_all, v_all = kv.view(li, seq)
            x = x.add_(g_o.run(self._attend(q, k_all, v_all, n, mask))[0])

            h = rms(x, self._norm2[li], eps)
            gate, up = g_gu.run(h)
            x = x.add_(g_down.run(F.silu(gate) * up)[0])

        kv.seq_len = seq
        if not logits:
            return None
        return self._head.run(rms(x[-1:], self.final_norm, eps))[0].view(-1)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """SDPA on ``[N, n_q, head_dim]`` queries against ``[1, n_kv, seq, hd]`` cache.

        Decode takes the cheap road: ``q.view(1, n_kv, n_rep, hd)`` puts the
        ``n_rep`` queries that share a KV head on the *query* axis, so grouped
        attention needs no ``repeat_kv`` copy of the cache and no ``enable_gqa``.
        Head ``j`` reads KV head ``j // n_rep``, exactly HuggingFace's mapping.
        ``is_causal`` stays off: everything in ``[:seq]`` is in the past by
        construction, and torch would align a causal mask top-left and mask it all
        away (token-loop.md §6.3).
        """
        if n == 1:
            a = F.scaled_dot_product_attention(
                q.view(1, self.n_kv, self.n_rep, self.head_dim), k, v, scale=self.scaling
            )
            return a.reshape(1, self.q_dim)
        if not self._gqa:
            raise RuntimeError("prefill with N>1 needs torch>=2.5 (enable_gqa) for GQA")
        a = F.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0), k, v, attn_mask=mask, scale=self.scaling, enable_gqa=True
        )
        return a.squeeze(0).permute(1, 0, 2).reshape(n, self.q_dim)

    def _causal_mask(self, start_pos: int, seq: int) -> torch.Tensor:
        """``[1, 1, n, seq]`` bool, ``True`` = attend. Only built for ``N > 1``."""
        q_pos = torch.arange(start_pos, seq, device=self.device).unsqueeze(1)
        k_pos = torch.arange(seq, device=self.device).unsqueeze(0)
        return (k_pos <= q_pos)[None, None]

    # --- phases ------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Walk the prompt once, filling KV slots ``0..len-1``.

        Chunked by :attr:`prefill_chunk`, which is ``1`` while the kernel is
        decode-only. One pass either way: no prompt token is ever recomputed.
        """
        ids = ids.reshape(-1).to(self.device, torch.long)
        n = int(ids.numel())
        if n == 0:
            raise ValueError("empty prompt")
        step = max(1, self.prefill_chunk)
        out = None
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            out = self.forward(ids[lo:hi], lo, logits=(hi == n))
        assert out is not None
        return out

    @torch.no_grad()
    def step(self, token_id: int) -> torch.Tensor:
        """One decode token at position ``seq_len``. ``N == 1``, RoPE at ``seq-1``."""
        self._tok[0] = token_id
        return self.forward(self._tok, self.kv.seq_len)

    @torch.no_grad()
    def warmup(self, *, prompt: int = 8, tokens: int = 16) -> float:
        """Eager pass over both phases, then reset. Capture *after* this.

        Kernels, SDPA backends, the RoPE table and the allocator all pay their
        one-time cost here so that neither the graph capture nor the measurement
        includes it (token-loop.md §6.2, §6.3).
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.reset()
        ids = torch.arange(1, prompt + 1, device=self.device, dtype=torch.long)
        logits = self.prefill(ids)
        for _ in range(tokens):
            logits = self.step(int(logits.argmax()))
        torch.cuda.synchronize()
        self.reset()
        return (time.perf_counter() - t0) * 1000.0

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        *,
        stop: Sequence[int] = (),
        on_token: Callable[[int], None] | None = None,
    ) -> Generation:
        """Greedy. Prefill and decode are timed separately and never averaged."""
        self.reset()
        ids = prompt_ids.reshape(-1).to(self.device, torch.long)
        stop_set = frozenset(int(s) for s in stop)
        out = Generation(
            prompt_len=int(ids.numel()),
            prefill_chunk=self.prefill_chunk,
            graph=self.graph_mode,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.prefill(ids)
        token = int(logits.argmax())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.prefill_ms = (t1 - t0) * 1000.0

        for i in range(max_new_tokens):
            out.tokens.append(token)
            if on_token is not None:
                on_token(token)
            if token in stop_set:
                out.stop_token = token
                break
            if i + 1 == max_new_tokens or self.kv.seq_len >= self.max_seq:
                break
            logits = self.step(token)
            out.decode_steps += 1
            token = int(logits.argmax())
        torch.cuda.synchronize()
        out.decode_ms = (time.perf_counter() - t1) * 1000.0
        return out

    def __repr__(self) -> str:
        return (
            f"TokenLoop(layers={self.n_layers}, hidden={self.hidden}, "
            f"q/kv={self.n_q}/{self.n_kv}x{self.head_dim}, max_seq={self.max_seq}, "
            f"groups={len(self._groups)}, prefill_chunk={self.prefill_chunk}, "
            f"norm={self.norm_mode}, overlap={self.overlap}, graph={self.graph_mode})"
        )
