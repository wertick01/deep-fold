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
while a GEMM is decode-only -- :func:`gpu.loop.graph.linear_max_n` asks the
loaded codec instead of assuming, and :attr:`TokenLoop.prefill_chunk` reports the
answer. Either way the prompt is walked *once*: the KV cache is filled slot by
slot and never recomputed, which is the difference that matters for tok/s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import torch
import torch.nn.functional as F

from .graph import Gemm, GemmGroup, GraphedGemmGroup, capture, linear_max_n
from .kv_cache import KVCache
from .ring import CopyRing
from gpu.nf4.plan import LIVE_MAX_N

__all__ = [
    "TokenLoop",
    "Generation",
    "PACKERS",
    "PREFILL_HOLD_SUPERCHUNK",
    "rms_norm_exact",
    "repeat_kv",
    "split_concat_qkv",
    "split_internlm_wqkv",
    "split_neox_qkv",
]

MIB = 1024 * 1024
#: Superchunk width for ``prefill_mode="hold"``. See docs/plan-h2-accel.md §B.
PREFILL_HOLD_SUPERCHUNK = 256


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
    # Views. KVCache.write copies into the slot; a contiguous here is a second copy.
    k = packed[:, :, -2, :]
    v = packed[:, :, -1, :]
    return q, k, v


def split_concat_qkv(
    y: torch.Tensor, n_q: int, n_kv: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phi-3 style fused ``qkv_proj``: ``[Q | K | V]`` blocks, in that order.

    Not interchangeable with :func:`split_internlm_wqkv` -- that is the whole
    reason the packer is chosen from the module's name and never from "the
    matrix is fused, so it must be InternLM". Feeding one layout to the other
    routine produces a model that decodes fluent nonsense.
    """
    n = int(y.shape[0])
    q_dim, kv_dim = n_q * head_dim, n_kv * head_dim
    q = y[:, :q_dim].reshape(n, n_q, head_dim)
    k = y[:, q_dim : q_dim + kv_dim].reshape(n, n_kv, head_dim)
    v = y[:, q_dim + kv_dim : q_dim + 2 * kv_dim].reshape(n, n_kv, head_dim)
    return q, k, v


def split_neox_qkv(
    y: torch.Tensor, n_q: int, n_kv: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPT-NeoX ``query_key_value``: per head ``[q, k, v]``, interleaved.

    MHA only (``n_kv == n_q``), which is what the families using this name ship.
    Present so the "no packer" refusal is honest about what is known; the NeoX
    *glue* (LayerNorm, partial RoPE) is still a named refusal.
    """
    if n_kv != n_q:
        raise ValueError(f"neox_interleaved is MHA only; got n_q={n_q}, n_kv={n_kv}")
    n = int(y.shape[0])
    packed = y.view(n, n_q, 3, head_dim)
    q = packed[:, :, 0, :]
    k = packed[:, :, 1, :]
    v = packed[:, :, 2, :]
    return q, k, v


#: Packer id -> function. The id comes from :mod:`gpu.graphs` (the fused
#: matrix's own last component), so ``kind=qkv`` never implies InternLM.
PACKERS = {
    "internlm_gqa": split_internlm_wqkv,
    "concat": split_concat_qkv,
    "neox_interleaved": split_neox_qkv,
}


def _rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``t`` is ``[N, heads, head_dim]``; ``cos``/``sin`` are ``[N, 1, head_dim]``.

    ``rotate_half`` + fused multiply-add, i.e. HuggingFace's
    ``apply_rotary_pos_emb`` with the head axis already in the middle.
    """
    d = t.shape[-1] // 2
    rot = torch.cat((-t[..., d:], t[..., :d]), dim=-1)
    return t * cos + rot * sin


class _RopePairGraph:
    """One CUDA graph for decode ``N==1`` RoPE of Q and K.

    Eager RoPE is four ATen launches (cat, mul, mul, add) times two heads.
    On WDDM that is milliseconds across 64 layers; a captured pair plus two
    ``copy_`` into static buffers is the same math at one replay. Inputs are
    device activations, never H2D or host packed. Prefill ``N!=1`` stays eager.
    """

    __slots__ = ("graph", "q", "k", "cos", "sin", "q_out", "k_out", "q_live", "k_live")

    def __init__(self, n_q: int, n_kv: int, head_dim: int, device, dtype) -> None:
        self.q = torch.empty((1, n_q, head_dim), device=device, dtype=dtype)
        self.k = torch.empty((1, n_kv, head_dim), device=device, dtype=dtype)
        self.cos = torch.empty((1, 1, head_dim), device=device, dtype=dtype)
        self.sin = torch.empty((1, 1, head_dim), device=device, dtype=dtype)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _rope(self.q, self.cos, self.sin)
                _rope(self.k, self.cos, self.sin)
            side.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            self.graph.capture_begin()
            try:
                self.q_out = _rope(self.q, self.cos, self.sin)
                self.k_out = _rope(self.k, self.cos, self.sin)
            finally:
                self.graph.capture_end()
        torch.cuda.current_stream().wait_stream(side)
        # Graph-pool tensors cannot be ``select``/``view``-sliced on the token
        # path (resident 3B: Offset increment outside graph capture).
        self.q_live = torch.empty_like(self.q)
        self.k_live = torch.empty_like(self.k)

    def load_pos(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.q.copy_(q)
        self.k.copy_(k)
        self.graph.replay()
        self.q_live.copy_(self.q_out)
        self.k_live.copy_(self.k_out)
        return self.q_live, self.k_live


def _rope_tables(
    model, plan, max_seq: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos``/``sin`` for positions ``0..max_seq-1``, computed once.

    RoPE is position-wise, so a table is bit-identical to calling the module per
    step and costs 512 KiB at ``max_seq=1024``. It also removes the classic
    "``inv_freq`` was built on CPU and only decode disagrees" bug
    (token-loop.md §6.3): the buffer is whatever ``load_chr_nf4`` rebuilt on the
    device, used here and nowhere else.

    Qwen keeps one ``model.rotary_emb``. InternLM2 keeps one per layer; they
    share the same dim/base, so layer 0's module is the table. Which of the two
    it is comes from the plan, not from an attribute guess.
    """
    base = model.get_submodule(plan.backbone) if plan.backbone else model
    rot = getattr(base, "rotary_emb", None)
    if rot is None:
        rot = getattr(model.get_submodule(plan.layers[0].attn), "rotary_emb", None)
    if rot is None:
        raise RuntimeError(
            f"attach: no rotary module on {plan.backbone or '<model>'} or on "
            f"{plan.layers[0].attn}, and plan.rope is {plan.rope!r}. "
            "The loop does not reimplement RoPE init."
        )
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


def repeat_kv(t: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA K/V along the head axis. ``t`` is ``[1, n_kv, seq, hd]``."""
    if int(n_rep) == 1:
        return t
    return t.repeat_interleave(int(n_rep), dim=1)


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
    interrupted: bool = False
    prefill_chunk: int = 1
    graph: str = "off"
    h2d_bytes: int = 0
    h2d_copies: int = 0
    h2d_copy_ms: float = 0.0
    h2d_forwards: int = 0
    spec_verifies: int = 0
    spec_skips: int = 0
    spec_draft_accepted: int = 0
    spec_draft_ms: float = 0.0

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

    ``model`` must already be through ``load_model`` -- the loop reads the packed
    buffers out of the modules once (:class:`~gpu.loop.graph.Gemm`) and then never
    touches ``nn.Module`` attribute lookup on the token path again.

    Which modules those are comes from a :class:`~gpu.host.attach.DriverPlan`.
    ``load_model`` leaves one on the model; otherwise ``attach(model)`` walks the
    tree here. There is no ``hasattr(layer0.attention, "wqkv")``: InternLM2 is a
    *packer id* in :data:`PACKERS`, not a boolean, and a graph whose family is
    not implemented never reaches this constructor.
    """

    def __init__(
        self,
        model,
        *,
        max_seq: int = 1024,
        norm: str = "exact",
        overlap: bool = True,
        device: torch.device | str | None = None,
        plan=None,
        slots=None,
        ring_timing: bool = False,
        prefill_mode: Literal["chunk", "hold"] = "chunk",
    ) -> None:
        from gpu.graphs import IMPLEMENTED_FAMILIES, refuse

        cfg = model.config
        self.model = model

        if plan is None:
            plan = getattr(model, "deepfold_plan", None)
        if plan is None:
            from gpu.host.attach import attach

            plan = attach(model)
        if plan.family not in IMPLEMENTED_FAMILIES:
            raise RuntimeError(
                refuse.family(
                    plan.family, attn=plan.attn, mlp=plan.mlp, act=plan.act, norm=plan.norm
                )
            )
        self.plan = plan

        self.max_seq = int(max_seq)
        self.n_layers = int(cfg.num_hidden_layers)
        self.hidden = int(cfg.hidden_size)
        self.n_q = int(cfg.num_attention_heads)
        # Missing num_key_value_heads is legal and means MHA (wave8-arch §2.4).
        self.n_kv = int(getattr(cfg, "num_key_value_heads", None) or self.n_q)
        self.eps = float(cfg.rms_norm_eps)
        self.vocab = int(cfg.vocab_size)
        if self.n_layers != plan.n_layers:
            raise ValueError(
                f"config says {self.n_layers} layers, the plan walked {plan.n_layers}"
            )
        cplan = getattr(model, "deepfold_compute", None)
        self.compute_plan = cplan
        self.compute = str(getattr(cplan, "compute", "gpu"))
        self.n_gpu = int(getattr(cplan, "n_gpu", self.n_layers))
        self.n_cpu = int(getattr(cplan, "n_cpu", 0))
        if self.n_gpu + self.n_cpu != self.n_layers:
            raise ValueError(
                f"compute split n_gpu={self.n_gpu} + n_cpu={self.n_cpu} "
                f"!= n_layers={self.n_layers}"
            )
        if not 0 <= self.n_gpu <= self.n_layers:
            raise ValueError(f"n_gpu={self.n_gpu} outside 0..{self.n_layers}")

        get = model.get_submodule
        attn0 = get(plan.layers[0].attn)
        self.embed = get(plan.embed)
        self._lm = get(plan.lm_head)
        self._norm1 = [get(p.norm1).weight for p in plan.layers]
        self._norm2 = [get(p.norm2).weight for p in plan.layers]
        #: None -> three GEMMs and a reshape; otherwise one GEMM and this packer.
        self._pack = PACKERS[plan.qkv_pack] if plan.qkv_pack else None

        self.head_dim = int(getattr(attn0, "head_dim", None) or self.hidden // self.n_q)
        self.scaling = float(getattr(attn0, "scaling", self.head_dim**-0.5))
        self.q_dim = self.n_q * self.head_dim
        self.n_rep = self.n_q // self.n_kv
        if self.n_q % self.n_kv:
            raise ValueError(f"{self.n_q} q heads do not group into {self.n_kv} kv heads")
        self.qkv_dim = (self.n_q + 2 * self.n_kv) * self.head_dim

        self.final_norm = get(plan.final_norm).weight
        if self.final_norm.is_meta:
            raise RuntimeError(
                f"{plan.final_norm}.weight is still on meta: load the .chr first"
            )
        self.device = torch.device(device) if device is not None else self.final_norm.device

        if norm not in ("exact", "fast"):
            raise ValueError(f"norm={norm!r}; expected 'exact' or 'fast'")
        self.norm_mode = norm
        self.rms = rms_norm_exact if norm == "exact" else _rms_norm_fast
        self._rms_shape = (self.hidden,)

        # --- weights, flattened out of the module tree ----------------------
        # Two side streams, made once and shared by every group: q/k/v and
        # gate/up are independent and each leaves most of the card idle on its
        # own (see GemmGroup). Set overlap=False for the strictly serial loop.
        self.overlap = bool(overlap) and torch.cuda.is_available()
        self._streams = tuple(torch.cuda.Stream() for _ in range(2)) if self.overlap else ()
        if slots is None:
            slots = getattr(model, "deepfold_slots", None)
        if self.n_cpu > 0:
            # v1: do not mix CopyRing with a CPU suffix (WDDM depth, two taxes).
            slots = None
        self.slots = slots
        self._ring = CopyRing(slots, timing=ring_timing) if slots is not None else None
        self._compute = None
        if (
            self._ring is not None
            and torch.cuda.is_available()
            and torch.device(slots.device).type == "cuda"
        ):
            # Overflow: compute on an explicit stream, not the legacy default.
            self._compute = torch.cuda.Stream()
        self._groups = self._build_groups()
        if self._ring is not None:
            for grp in self._groups:
                grp.ring = self._ring
        self._host_tape = tuple(
            g for grp in self._groups for g in grp.gemms if g.home == "host"
        )
        self._apply(self._groups)
        self.graph_mode = "off"
        self.graph_error: str | None = None

        # --- everything long-lived, allocated here and nowhere else ---------
        if self.n_gpu > 0:
            self.kv = KVCache(
                self.n_gpu,
                self.max_seq,
                self.n_kv,
                self.head_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
        else:
            self.kv = None
        self.kv_cpu = None
        if self.n_cpu > 0:
            from gpu.host.cpu_linear import ensure_cpu_threads

            ensure_cpu_threads()
            self.kv_cpu = KVCache(
                self.n_cpu,
                self.max_seq,
                self.n_kv,
                self.head_dim,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
        self.cos, self.sin = _rope_tables(
            model, plan, self.max_seq, self.device, torch.bfloat16
        )
        # [max_seq, 1, head_dim] so a token slice is already broadcast-ready.
        self.cos = self.cos.unsqueeze(1)
        self.sin = self.sin.unsqueeze(1)
        if self.n_cpu > 0:
            self.cos_cpu = self.cos.to("cpu")
            self.sin_cpu = self.sin.to("cpu")
            for li in range(self.n_gpu, self.n_layers):
                self._norm1[li] = self._norm1[li].detach().to("cpu")
                self._norm2[li] = self._norm2[li].detach().to("cpu")
        else:
            self.cos_cpu = None
            self.sin_cpu = None
        self._tok = torch.zeros(1, dtype=torch.long, device=self.device)
        self._gqa = _sdpa_has_gqa()
        self._rope_pair: _RopePairGraph | None = None
        self.prefill_chunk = min(
            linear_max_n(self._groups[0].gemms[0].codec, LIVE_MAX_N), self.max_seq
        )
        if prefill_mode not in ("chunk", "hold"):
            raise ValueError(f"prefill_mode={prefill_mode!r}; expected 'chunk' or 'hold'")
        if prefill_mode == "hold" and self.n_cpu > 0:
            raise ValueError(
                "prefill_mode='hold' is --compute gpu only; hybrid/cpu-suffix use chunk"
            )
        self.prefill_mode = prefill_mode

    # --- setup ------------------------------------------------------------
    def _build_groups(self) -> list[GemmGroup]:
        """Contiguous runs of NF4 GEMMs, in execution order (see graph.py).

        Four groups per layer plus the head, exactly as before -- the only change
        is that the modules come from ``plan.layers[i].gemms`` (qualified names)
        instead of from two hardcoded attribute spellings.
        """
        groups: list[GemmGroup] = []
        st = self._streams
        get = self.model.get_submodule
        fused = self.plan.attn == "fused"
        for lp in self.plan.layers:
            li, slots = lp.index, lp.gemms
            if fused:
                groups.append(
                    GemmGroup(f"L{li}.qkv", [Gemm.of(get(slots["qkv"]), f"L{li}.qkv")])
                )
            else:
                groups.append(
                    GemmGroup(
                        f"L{li}.qkv",
                        [
                            Gemm.of(get(slots["q"]), f"L{li}.q"),
                            Gemm.of(get(slots["k"]), f"L{li}.k"),
                            Gemm.of(get(slots["v"]), f"L{li}.v"),
                        ],
                        st,
                    )
                )
            groups.append(GemmGroup(f"L{li}.o", [Gemm.of(get(slots["o"]), f"L{li}.o")]))
            groups.append(
                GemmGroup(
                    f"L{li}.gateup",
                    [
                        Gemm.of(get(slots["gate"]), f"L{li}.gate"),
                        Gemm.of(get(slots["up"]), f"L{li}.up"),
                    ],
                    st[:1],
                )
            )
            groups.append(
                GemmGroup(f"L{li}.down", [Gemm.of(get(slots["down"]), f"L{li}.down")])
            )
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
        seen: dict[int, int] = {}
        for g in self._groups:
            for m in g.gemms:
                if m.home == "host" or m.home == "cpu":
                    continue
                t = m.packed if m.codec != "vq" else m.index
                if t is None or int(t.numel()) == 0:
                    continue
                seen[t.data_ptr()] = m.nbytes
        packed = getattr(self.embed, "packed", None)
        index = getattr(self.embed, "index", None)
        if packed is not None and packed.numel() > 0:
            seen[packed.data_ptr()] = self.embed.nbytes
        elif index is not None and index.numel() > 0:
            seen[index.data_ptr()] = self.embed.nbytes
        elif getattr(self.embed, "weight", None) is not None:
            w = self.embed.weight
            if w.numel() > 0:
                seen[w.data_ptr()] = w.numel() * w.element_size()
        return sum(seen.values())

    def reset(self) -> None:
        """Forget the conversation, keep the memory."""
        if self.kv is not None:
            self.kv.reset()
        if self.kv_cpu is not None:
            self.kv_cpu.reset()

    def _seq_len(self) -> int:
        if self.kv is not None:
            return self.kv.seq_len
        if self.kv_cpu is not None:
            return self.kv_cpu.seq_len
        return 0

    def _set_seq_len(self, seq: int) -> None:
        if self.kv is not None:
            self.kv.seq_len = seq
        if self.kv_cpu is not None:
            self.kv_cpu.seq_len = seq

    def _bounce_to_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """10 KiB (N=1) D2H after the GPU prefix. Compute stream, not copy_stream."""
        if x.device.type != "cuda":
            return x
        torch.cuda.synchronize()
        return x.to("cpu")

    def _bounce_to_device(self, h: torch.Tensor) -> torch.Tensor:
        if h.device == self.device:
            return h
        return h.to(self.device)

    @torch.no_grad()
    def capture_graphs(self) -> str:
        """CUDA graph plan A on all-DEVICE groups. Idempotent; returns the mode.

        Must be called after :meth:`warmup`. HOST overflow groups stay eager.
        ``graph_mode`` is ``"linears"`` if at least one group is captured. On a
        DEVICE capture failure the loop keeps running eager and
        :attr:`graph_error` says why -- eager is the contract, the graph is
        the optimization. Mixed sessions do not use ``graph_error="overflow"``.
        """
        if self.graph_mode == "linears":
            return self.graph_mode
        # Mixed overflow: DEVICE groups graph, HOST groups stay eager. capture()
        # skips HOST warmup (no CopyRing in that path). Not a full "overflow" off.
        runners, mode, err = capture(self._groups)
        self._apply(runners)
        self.graph_mode, self.graph_error = mode, err
        self._capture_decode_glue()
        return mode

    def drop_graphs(self) -> str:
        """Back to eager launches (used to check replay against eager)."""
        self._apply(self._groups)
        self.graph_mode, self.graph_error = "off", None
        return self.graph_mode

    # --- the forward -------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        ids: torch.Tensor,
        start_pos: int,
        *,
        logits: bool = True,
        all_positions: bool = False,
    ):
        """``ids`` is ``[N]``; writes KV slots ``start_pos..start_pos+N-1``.

        Returns the ``[vocab]`` logits of the *last* position, or ``None`` when
        ``logits=False`` (a prefill chunk that is not the last one -- the point of
        prefill is the cache, not the distribution). ``all_positions=True`` is
        teacher-forced NLL: ``[N, vocab]`` at every consumed token, still eager
        for ``N > 1`` (the CUDA graph is decode ``N == 1`` only).
        """
        n = int(ids.numel())
        seq = start_pos + n
        if seq > self.max_seq:
            raise ValueError(f"position {seq} past max_seq={self.max_seq}")

        compute = self._compute
        if compute is None:
            return self._forward_body(ids, start_pos, n, seq, logits, all_positions)
        outer = torch.cuda.current_stream()
        compute.wait_stream(outer)
        with torch.cuda.stream(compute):
            out = self._forward_body(ids, start_pos, n, seq, logits, all_positions)
        if out is not None:
            out.record_stream(outer)
        outer.wait_stream(compute)
        return out

    def _capture_decode_glue(self) -> None:
        """Capture decode-N=1 RoPE. Idle GPU only; not on the token path.

        GEMM plan A stays :func:`capture`. This graph is device activations
        only -- no H2D, no host packed, no CopyRing.
        """
        if self._rope_pair is not None:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
            self._rope_pair = _RopePairGraph(
                self.n_q, self.n_kv, self.head_dim, self.device, torch.bfloat16
            )
        except Exception:
            self._rope_pair = None

    def _forward_body(
        self,
        ids: torch.Tensor,
        start_pos: int,
        n: int,
        seq: int,
        logits: bool,
        all_positions: bool,
    ):
        ring = self._ring
        if ring is not None:
            ring.arm(self._host_tape)
            ring.prefetch()  # first overflow H2D overlaps embed
        x = self.embed(ids)  # [n, hidden] bf16
        cos = self.cos[start_pos:seq]  # [n, 1, head_dim]
        sin = self.sin[start_pos:seq]
        if n == 1:
            x = self._decode_layers(x, start_pos, seq, cos, sin)
        else:
            x = self._prefill_layers(
                x, start_pos, n, seq, cos, sin, self._causal_mask(start_pos, seq)
            )

        self._set_seq_len(seq)
        if all_positions:
            src = x
        elif not logits:
            return None
        else:
            src = x if n == 1 else x[-1:]
        if n == 1:
            hidden = F.rms_norm(src, self._rms_shape, self.final_norm, self.eps)
        else:
            hidden = self.rms(src, self.final_norm, self.eps)
        self._prefetch_next_token(n)
        y = self._head.run(hidden)[0]
        if all_positions:
            return y if y.dim() == 2 else y.view(1, -1)
        return y.view(-1)

    def _decode_layers(
        self,
        x: torch.Tensor,
        pos: int,
        seq: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Layer body for ``N == 1``. Locals, GQA view, fused RMS, optional RoPE graph.

        ``norm="exact"`` stays the TokenLoop default and the prefill ``N!=1``
        path. Decode uses :func:`torch.nn.functional.rms_norm` (same formula,
        one fused kernel). CUDA bf16 can differ by ~1 ULP from
        :func:`rms_norm_exact`; greedy needles are quality checks, not bitwise
        logits. Do not ``torch.compile`` this method: it would break CUDA graphs
        and CopyRing. CPU suffix is a second walk after one 10 KiB bounce.
        """
        n_gpu = self.n_gpu
        if n_gpu:
            kv = self.kv
            write, view = kv.write, kv.view
            eps = self.eps
            hd = self.head_dim
            n_q, n_kv, n_rep = self.n_q, self.n_kv, self.n_rep
            q_dim, scaling = self.q_dim, self.scaling
            pack = self._pack
            sdpa = F.scaled_dot_product_attention
            silu = F.silu
            pair = self._rope_pair
            if pair is not None:
                pair.load_pos(cos, sin)
            shape = self._rms_shape
            norm1, norm2, layers = self._norm1, self._norm2, self._layer
            ring = self._ring

            for li in range(n_gpu):
                g_qkv, g_o, g_gu, g_down = layers[li]
                if ring is not None:
                    ring.prefetch()  # first HOST of this layer overlaps qkv
                h = F.rms_norm(x, shape, norm1[li], eps)
                if pack is not None:
                    q, k, v = pack(g_qkv.run(h)[0], n_q, n_kv, hd)
                else:
                    q, k, v = g_qkv.run(h)
                    q = q.view(1, n_q, hd)
                    k = k.view(1, n_kv, hd)
                    v = v.view(1, n_kv, hd)
                if pair is not None:
                    q, k = pair.apply(q, k)
                else:
                    q = _rope(q, cos, sin)
                    k = _rope(k, cos, sin)
                write(li, pos, k, v)
                k_all, v_all = view(li, seq)
                a = sdpa(q.view(1, n_kv, n_rep, hd), k_all, v_all, scale=scaling)
                x = x.add_(g_o.run(a.reshape(1, q_dim))[0])

                if ring is not None:
                    ring.prefetch()  # HOST gate/up (tail D) overlaps remaining DEVICE
                h = F.rms_norm(x, shape, norm2[li], eps)
                gate, up = g_gu.run(h)
                if ring is not None:
                    ring.prefetch()  # down if not already in flight
                x = x.add_(g_down.run(silu(gate) * up)[0])
        if self.n_cpu == 0:
            return x
        return self._decode_cpu_suffix(x, pos, seq)

    def _decode_cpu_suffix(self, x: torch.Tensor, pos: int, seq: int) -> torch.Tensor:
        """CPU repeating layers after one D2H. Never capture; never CopyRing."""
        x = self._bounce_to_cpu(x)
        kv = self.kv_cpu
        write, view = kv.write, kv.view
        eps = self.eps
        hd = self.head_dim
        n_q, n_kv, n_rep = self.n_q, self.n_kv, self.n_rep
        q_dim, scaling = self.q_dim, self.scaling
        pack = self._pack
        sdpa = F.scaled_dot_product_attention
        silu = F.silu
        shape = self._rms_shape
        norm1, norm2, layers = self._norm1, self._norm2, self._layer
        cos = self.cos_cpu[pos:seq]
        sin = self.sin_cpu[pos:seq]
        n_gpu = self.n_gpu
        for li in range(n_gpu, self.n_layers):
            g_qkv, g_o, g_gu, g_down = layers[li]
            cpu_li = li - n_gpu
            h = F.rms_norm(x, shape, norm1[li], eps)
            if pack is not None:
                q, k, v = pack(g_qkv.run(h)[0], n_q, n_kv, hd)
            else:
                q, k, v = g_qkv.run(h)
                q = q.view(1, n_q, hd)
                k = k.view(1, n_kv, hd)
                v = v.view(1, n_kv, hd)
            q = _rope(q, cos, sin)
            k = _rope(k, cos, sin)
            write(cpu_li, pos, k, v)
            k_all, v_all = view(cpu_li, seq)
            a = sdpa(q.view(1, n_kv, n_rep, hd), k_all, v_all, scale=scaling)
            x = x.add_(g_o.run(a.reshape(1, q_dim))[0])
            h = F.rms_norm(x, shape, norm2[li], eps)
            gate, up = g_gu.run(h)
            x = x.add_(g_down.run(silu(gate) * up)[0])
        return self._bounce_to_device(x)

    def _prefill_layers(
        self,
        x: torch.Tensor,
        start_pos: int,
        n: int,
        seq: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.prefill_mode == "hold":
            return self._prefill_layers_hold(x, start_pos, n, seq, cos, sin, mask)
        n_gpu = self.n_gpu
        if n_gpu:
            kv, eps = self.kv, self.eps
            rms, hd, n_kv = self.rms, self.head_dim, self.n_kv
            pack = self._pack
            n_q = self.n_q
            attend = self._attend
            silu = F.silu
            norm1, norm2, layers = self._norm1, self._norm2, self._layer
            ring = self._ring

            for li in range(n_gpu):
                g_qkv, g_o, g_gu, g_down = layers[li]
                if ring is not None:
                    ring.prefetch()
                h = rms(x, norm1[li], eps)
                if pack is not None:
                    q, k, v = pack(g_qkv.run(h)[0], n_q, n_kv, hd)
                else:
                    q, k, v = g_qkv.run(h)
                    q = q.reshape(n, n_q, hd)
                    k = k.reshape(n, n_kv, hd)
                    v = v.reshape(n, n_kv, hd)
                q = _rope(q, cos, sin)
                k = _rope(k, cos, sin)
                kv.write(li, start_pos, k, v)
                k_all, v_all = kv.view(li, seq)
                x = x.add_(g_o.run(attend(q, k_all, v_all, n, mask))[0])

                if ring is not None:
                    ring.prefetch()
                h = rms(x, norm2[li], eps)
                gate, up = g_gu.run(h)
                if ring is not None:
                    ring.prefetch()
                x = x.add_(g_down.run(silu(gate) * up)[0])
        if self.n_cpu == 0:
            return x
        return self._prefill_cpu_suffix(x, start_pos, n, seq)

    def _prefill_cpu_suffix(
        self,
        x: torch.Tensor,
        start_pos: int,
        n: int,
        seq: int,
    ) -> torch.Tensor:
        x = self._bounce_to_cpu(x)
        kv, eps = self.kv_cpu, self.eps
        rms, hd, n_kv = self.rms, self.head_dim, self.n_kv
        pack = self._pack
        n_q = self.n_q
        attend = self._attend_cpu
        silu = F.silu
        norm1, norm2, layers = self._norm1, self._norm2, self._layer
        cos = self.cos_cpu[start_pos:seq]
        sin = self.sin_cpu[start_pos:seq]
        mask = self._causal_mask(start_pos, seq, device=x.device)
        n_gpu = self.n_gpu
        for li in range(n_gpu, self.n_layers):
            g_qkv, g_o, g_gu, g_down = layers[li]
            cpu_li = li - n_gpu
            h = rms(x, norm1[li], eps)
            if pack is not None:
                q, k, v = pack(g_qkv.run(h)[0], n_q, n_kv, hd)
            else:
                q, k, v = g_qkv.run(h)
                q = q.reshape(n, n_q, hd)
                k = k.reshape(n, n_kv, hd)
                v = v.reshape(n, n_kv, hd)
            q = _rope(q, cos, sin)
            k = _rope(k, cos, sin)
            kv.write(cpu_li, start_pos, k, v)
            k_all, v_all = kv.view(cpu_li, seq)
            x = x.add_(g_o.run(attend(q, k_all, v_all, n, mask))[0])
            h = rms(x, norm2[li], eps)
            gate, up = g_gu.run(h)
            x = x.add_(g_down.run(silu(gate) * up)[0])
        return self._bounce_to_device(x)

    def _prefill_layers_hold(
        self,
        x: torch.Tensor,
        start_pos: int,
        n: int,
        seq: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention LTR per ``prefill_chunk``; one H2D per HOST MLP matrix.

        docs/plan-h2-accel.md §B: do not keep a full ``[T, intermediate]`` —
        the caller walks superchunks of :data:`PREFILL_HOLD_SUPERCHUNK`.
        """
        from .graph import nf4_gemm

        kv, eps = self.kv, self.eps
        rms, hd, n_kv = self.rms, self.head_dim, self.n_kv
        pack = self._pack
        n_q = self.n_q
        attend = self._attend
        silu = F.silu
        norm1, norm2, layers = self._norm1, self._norm2, self._layer
        ring = self._ring
        gemm_n = max(1, self.prefill_chunk)

        def _eager(grp):
            return getattr(grp, "eager", grp)

        def _ranges() -> list[tuple[int, int]]:
            return [(lo, min(lo + gemm_n, n)) for lo in range(0, n, gemm_n)]

        def _nf4(g: Gemm, packed: torch.Tensor, scale: torch.Tensor, x_n: torch.Tensor):
            y = nf4_gemm(packed, scale, x_n.t(), g.M, g.K, g.K_pad).t()
            if g.bias is not None:
                y = y.add_(g.bias)
            return y

        def _hold_member(g: Gemm, xs: list[torch.Tensor]) -> list[torch.Tensor]:
            if g.home != "host":
                return [_nf4(g, g.packed, g.scale, xc) for xc in xs]
            if ring is None:
                raise RuntimeError(f"{g.name}: host-resident GEMM needs CopyRing")
            packed, scale = ring.bind_hold(g)
            outs: list[torch.Tensor] = []
            for xc in xs:
                ring.gemm_hold()
                outs.append(_nf4(g, packed, scale, xc))
            ring.release_hold(g)
            return outs

        for li, (g_qkv, g_o, g_gu, g_down) in enumerate(layers):
            if ring is not None:
                ring.prefetch()
            for lo, hi in _ranges():
                cn = hi - lo
                x_c = x[lo:hi]
                h = rms(x_c, norm1[li], eps)
                if pack is not None:
                    q, k, v = pack(g_qkv.run(h)[0], n_q, n_kv, hd)
                else:
                    q, k, v = g_qkv.run(h)
                    q = q.reshape(cn, n_q, hd)
                    k = k.reshape(cn, n_kv, hd)
                    v = v.reshape(cn, n_kv, hd)
                q = _rope(q, cos[lo:hi], sin[lo:hi])
                k = _rope(k, cos[lo:hi], sin[lo:hi])
                kv.write(li, start_pos + lo, k, v)
                seq_c = start_pos + hi
                k_all, v_all = kv.view(li, seq_c)
                mask_c = mask[:, :, lo:hi, :seq_c]
                x_c.add_(g_o.run(attend(q, k_all, v_all, cn, mask_c))[0])

            if ring is not None:
                ring.prefetch()
            h = rms(x, norm2[li], eps)
            h_chunks = [h[lo:hi] for lo, hi in _ranges()]
            gu = _eager(g_gu)
            dn = _eager(g_down)
            gate_g, up_g = gu.gemms
            down_g = dn.gemms[0]
            if gate_g.home != "host" and up_g.home != "host":
                down_ins = []
                for hc in h_chunks:
                    gate, up = g_gu.run(hc)
                    down_ins.append(silu(gate) * up)
            else:
                gate_outs = _hold_member(gate_g, h_chunks)
                up_outs = _hold_member(up_g, h_chunks)
                down_ins = [silu(ga) * ua for ga, ua in zip(gate_outs, up_outs)]

            if ring is not None:
                ring.prefetch()
            if down_g.home == "host":
                downs = _hold_member(down_g, down_ins)
            else:
                downs = [g_down.run(d)[0] for d in down_ins]
            for (lo, hi), d in zip(_ranges(), downs):
                x[lo:hi].add_(d)
        return x

    def _prefetch_next_token(self, n: int) -> None:
        """H2D the first overflow matrix of token t+1 during this token's lm_head.

        Decode-only (``N == 1``). Prefill chunks with ``N != 1`` leave the copy
        engine idle at the boundary so the next ``arm()`` issues as before.
        H2D stays on ``copy_stream`` (Python, never inside a CUDA graph); it
        overlaps DEVICE ``lm_head`` / graph replay. Bytes/token unchanged.
        """
        ring = self._ring
        if ring is None or n != 1:
            return
        ring.prefetch_next()

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
            return a.view(1, self.q_dim)
        if not self._gqa:
            raise RuntimeError("prefill with N>1 needs torch>=2.5 (enable_gqa) for GQA")
        a = F.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0), k, v, attn_mask=mask, scale=self.scaling, enable_gqa=True
        )
        return a.squeeze(0).permute(1, 0, 2).reshape(n, self.q_dim)

    def _attend_cpu(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """CPU SDPA. If this build lacks ``enable_gqa``, expand K/V by ``n_rep``."""
        if n == 1:
            a = F.scaled_dot_product_attention(
                q.view(1, self.n_kv, self.n_rep, self.head_dim), k, v, scale=self.scaling
            )
            return a.view(1, self.q_dim)
        qh = q.permute(1, 0, 2).unsqueeze(0)
        if self._gqa:
            a = F.scaled_dot_product_attention(
                qh, k, v, attn_mask=mask, scale=self.scaling, enable_gqa=True
            )
        else:
            a = F.scaled_dot_product_attention(
                qh,
                repeat_kv(k, self.n_rep),
                repeat_kv(v, self.n_rep),
                attn_mask=mask,
                scale=self.scaling,
            )
        return a.squeeze(0).permute(1, 0, 2).reshape(n, self.q_dim)

    def _causal_mask(
        self, start_pos: int, seq: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        """``[1, 1, n, seq]`` bool, ``True`` = attend. Only built for ``N > 1``."""
        dev = self.device if device is None else device
        q_pos = torch.arange(start_pos, seq, device=dev).unsqueeze(1)
        k_pos = torch.arange(seq, device=dev).unsqueeze(0)
        return (k_pos <= q_pos)[None, None]

    # --- phases ------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Walk the prompt once, filling KV slots ``0..len-1``.

        Chunked by :attr:`prefill_chunk`, which is ``1`` while the kernel is
        decode-only. ``prefill_mode="hold"`` walks
        :data:`PREFILL_HOLD_SUPERCHUNK` so each HOST MLP matrix is H2D once
        per superchunk. One pass either way: no prompt token is ever recomputed.
        """
        ids = ids.reshape(-1).to(self.device, torch.long)
        n = int(ids.numel())
        if n == 0:
            raise ValueError("empty prompt")
        if self.prefill_mode == "hold":
            step = max(1, min(PREFILL_HOLD_SUPERCHUNK, self.max_seq))
        else:
            step = max(1, self.prefill_chunk)
        out = None
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            out = self.forward(ids[lo:hi], lo, logits=(hi == n))
        assert out is not None
        return out

    @torch.no_grad()
    def loglikelihood(
        self, prompt_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Teacher-forced logits at every prompt position. Does not sample.

        Walks the prompt in :attr:`prefill_chunk` pieces so N stays inside the
        live GEMM cap, but runs ``lm_head`` on every token of each chunk. Returns
        ``(logits [T, vocab], ids [T], elapsed_ms)``. The lab adapter scores
        ``ids[1:]`` against ``logits[:-1]``. Empty or ``T > max_seq`` raise
        ``ValueError`` so the adapter can turn that into an empty cell rather
        than truncating.
        """
        self.reset()
        ids = prompt_ids.reshape(-1).to(self.device, torch.long)
        n = int(ids.numel())
        if n == 0:
            raise ValueError("empty prompt")
        if n > self.max_seq:
            raise ValueError(f"position {n} past max_seq={self.max_seq}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        parts: list[torch.Tensor] = []
        step = max(1, self.prefill_chunk)
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            part = self.forward(ids[lo:hi], lo, all_positions=True)
            parts.append(part if part.dim() == 2 else part.view(1, -1))
        logits = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        torch.cuda.synchronize()
        return logits, ids, (time.perf_counter() - t0) * 1000.0

    @torch.no_grad()
    def step(self, token_id: int) -> torch.Tensor:
        """One decode token at position ``seq_len``. ``N == 1``, RoPE at ``seq-1``."""
        self._tok[0] = token_id
        return self.forward(self._tok, self._seq_len())

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
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.reset()
        self._capture_decode_glue()
        return elapsed

    def h2_snapshot(self) -> dict:
        """DEVICE vs HOST topology and ring counters. Safe on CPU."""
        from .graph import GraphedGemmGroup

        runners = [g for row in self._layer for g in row] + [self._head]
        groups = []
        n_graph = n_eager = n_host = n_dev = n_cpu = 0
        for g in runners:
            graphed = isinstance(g, GraphedGemmGroup)
            eager = g.eager if graphed else g
            homes = [m.home for m in eager.gemms]
            if graphed:
                n_graph += 1
            else:
                n_eager += 1
            n_host += sum(1 for h in homes if h == "host")
            n_cpu += sum(1 for h in homes if h == "cpu")
            n_dev += sum(1 for h in homes if h == "device")
            groups.append(
                {
                    "name": g.name,
                    "graphed": graphed,
                    "homes": homes,
                    "nbytes": [int(m.nbytes) for m in eager.gemms],
                    "gemms": [m.name for m in eager.gemms],
                    "streams": len(getattr(eager, "streams", ())),
                }
            )
        kv_gpu = 0.0 if self.kv is None else float(self.kv.mib)
        kv_cpu = 0.0 if self.kv_cpu is None else float(self.kv_cpu.mib)
        return {
            "max_seq": self.max_seq,
            "n_layers": self.n_layers,
            "n_gpu": self.n_gpu,
            "n_cpu": self.n_cpu,
            "compute": self.compute,
            "hidden": self.hidden,
            "n_q": self.n_q,
            "n_kv": self.n_kv,
            "prefill_chunk": self.prefill_chunk,
            "prefill_mode": self.prefill_mode,
            "graph_mode": self.graph_mode,
            "graph_error": self.graph_error,
            "overlap": self.overlap,
            "n_groups": len(runners),
            "n_graphed_groups": n_graph,
            "n_eager_groups": n_eager,
            "n_host_gemms": n_host,
            "n_cpu_gemms": n_cpu,
            "n_device_gemms": n_dev,
            "weight_bytes_device": int(self.weight_bytes),
            "kv_mib": kv_gpu + kv_cpu,
            "kv_gpu_mib": kv_gpu,
            "kv_cpu_mib": kv_cpu,
            "host_tape": [g.name for g in self._host_tape],
            "ring": None if self._ring is None else self._ring.snapshot(),
            "groups": groups,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        *,
        stop: Sequence[int] = (),
        on_token: Callable[[int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        speculate: int = 1,
        draft: str = "none",
        oracle_ids: torch.Tensor | Sequence[int] | None = None,
        drafter=None,
    ) -> Generation:
        """Greedy. Prefill and decode are timed separately and never averaged.

        ``speculate <= 1`` or ``draft == "none"`` is the ``step()`` loop.
        ``draft == "lookup"`` and ``speculate >= 2`` is n-gram draft + greedy
        verify (:mod:`gpu.loop.speculate`); no second model.
        ``draft == "oracle"`` is lab-only: the teacher sequence (prompt +
        greedy tokens) is proposed as the draft so T_verify-bound tok/s can
        be measured.
        ``draft == "cpu"`` is lab-only: ``drafter(known_ids, k)`` proposes
        tokens from a CPU model (RAM, not VRAM). Same-GPU draft stays
        forbidden. Not a product default; CLI does not grow ``--draft``.

        ``should_stop`` is polled between decode tokens, not inside a CUDA
        kernel. When it returns true, :attr:`Generation.interrupted` is set and
        already emitted tokens are kept.
        """
        if draft not in ("none", "lookup", "oracle", "cpu"):
            raise ValueError(
                f"draft={draft!r}; expected 'none', 'lookup', 'oracle', or 'cpu'"
            )
        if draft == "oracle" and oracle_ids is None:
            raise ValueError("draft='oracle' requires oracle_ids (prompt + greedy)")
        if draft == "cpu" and drafter is None:
            raise ValueError("draft='cpu' requires drafter (CPU model, not VRAM)")
        self.reset()
        self._capture_decode_glue()
        ids = prompt_ids.reshape(-1).to(self.device, torch.long)
        stop_set = frozenset(int(s) for s in stop)
        out = Generation(
            prompt_len=int(ids.numel()),
            prefill_chunk=self.prefill_chunk,
            graph=self.graph_mode,
        )
        ring = self._ring
        b0 = 0 if ring is None else ring.total_bytes
        c0 = 0 if ring is None else ring.total_copies
        ms0 = 0.0 if ring is None else ring.total_copy_ms
        f0 = 0 if ring is None else ring.total_forwards

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.prefill(ids)
        token = int(logits.argmax())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.prefill_ms = (t1 - t0) * 1000.0

        if int(speculate) <= 1 or draft == "none":
            for i in range(max_new_tokens):
                if should_stop is not None and should_stop():
                    out.interrupted = True
                    break
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
        else:
            from .speculate import lookup_draft, oracle_draft, spec_generate

            if draft == "lookup":
                draft_fn = lookup_draft
            elif draft == "oracle":
                teacher = oracle_ids

                def draft_fn(known, k, _teacher=teacher):
                    return oracle_draft(known, k, _teacher)

            else:
                reset = getattr(drafter, "reset", None)
                if callable(reset):
                    reset()
                draft_fn = drafter

            spec_generate(
                self,
                out,
                ids,
                logits,
                max_new_tokens,
                stop_set,
                on_token,
                int(speculate),
                draft_fn,
                should_stop,
            )
        torch.cuda.synchronize()
        out.decode_ms = (time.perf_counter() - t1) * 1000.0
        if ring is not None:
            out.h2d_bytes = int(ring.total_bytes - b0)
            out.h2d_copies = int(ring.total_copies - c0)
            out.h2d_copy_ms = float(ring.total_copy_ms - ms0)
            out.h2d_forwards = int(ring.total_forwards - f0)
        return out

    def __repr__(self) -> str:
        return (
            f"TokenLoop({self.plan.family}, pack={self.plan.qkv_pack}, "
            f"layers={self.n_layers}, hidden={self.hidden}, "
            f"q/kv={self.n_q}/{self.n_kv}x{self.head_dim}, max_seq={self.max_seq}, "
            f"groups={len(self._groups)}, prefill_chunk={self.prefill_chunk}, "
            f"norm={self.norm_mode}, overlap={self.overlap}, graph={self.graph_mode})"
        )
