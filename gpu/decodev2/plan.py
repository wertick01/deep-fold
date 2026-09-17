"""Frozen synthetic architecture for Decode V2. No checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ArchSpec",
    "TINY_INTERNLM",
    "TINY_LLAMA",
    "MEDIUM_LLAMA",
    "spec_from_loaded",
]


@dataclass(frozen=True)
class ArchSpec:
    """Tiny transformer the synthetic stack can drive.

    ``family`` is the glue, not a HuggingFace ``model_type``:
    ``llama_swiglu`` has split q/k/v; ``internlm_gqa`` has fused wqkv with
    InternLM packing (``split_internlm_wqkv``).
    """

    name: str
    family: str
    n_layers: int
    hidden: int
    n_q: int
    n_kv: int
    head_dim: int
    intermediate: int
    vocab: int
    max_seq: int
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    eos_id: int = 2
    tied_embed: bool = False

    def __post_init__(self) -> None:
        if self.family not in {"llama_swiglu", "internlm_gqa"}:
            raise ValueError(f"unsupported family {self.family!r}")
        if self.n_layers < 1 or self.max_seq < 2:
            raise ValueError("need at least one layer and max_seq>=2")
        if self.hidden != self.n_q * self.head_dim:
            raise ValueError(
                f"hidden {self.hidden} != n_q*head_dim {self.n_q * self.head_dim}"
            )
        if self.n_q % self.n_kv != 0:
            raise ValueError(f"n_q={self.n_q} is not a multiple of n_kv={self.n_kv}")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE needs even head_dim")
        if self.hidden % 64 != 0 or self.intermediate % 64 != 0:
            raise ValueError("hidden and intermediate must be multiples of NF4 group 64")
        if not 0 <= self.eos_id < self.vocab:
            raise ValueError(f"eos_id {self.eos_id} out of vocab {self.vocab}")

    @property
    def n_rep(self) -> int:
        return self.n_q // self.n_kv

    @property
    def q_dim(self) -> int:
        return self.n_q * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv * self.head_dim

    @property
    def wqkv_out(self) -> int:
        """Rows of fused wqkv (InternLM packing)."""
        return (self.n_q + 2 * self.n_kv) * self.head_dim

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5


def spec_from_loaded(model, plan, *, max_seq: int) -> ArchSpec:
    """ArchSpec from a ``load_model`` skeleton and its DriverPlan."""
    cfg = model.config
    hidden = int(cfg.hidden_size)
    n_q = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", None) or n_q)
    attn0 = model.get_submodule(plan.layers[0].attn)
    head_dim = int(getattr(attn0, "head_dim", None) or hidden // n_q)
    eos = int(getattr(cfg, "eos_token_id", 0) or 0)
    family = str(plan.family)
    return ArchSpec(
        name=str(getattr(cfg, "_name_or_path", None) or family),
        family=family,
        n_layers=int(cfg.num_hidden_layers),
        hidden=hidden,
        n_q=n_q,
        n_kv=n_kv,
        head_dim=head_dim,
        intermediate=int(cfg.intermediate_size),
        vocab=int(cfg.vocab_size),
        max_seq=int(max_seq),
        rms_eps=float(cfg.rms_norm_eps),
        rope_theta=float(getattr(cfg, "rope_theta", 10000.0)),
        eos_id=eos,
        tied_embed=bool(getattr(cfg, "tie_word_embeddings", False)),
    )


TINY_LLAMA = ArchSpec(
    name="tiny-llama",
    family="llama_swiglu",
    n_layers=2,
    hidden=64,
    n_q=4,
    n_kv=2,
    head_dim=16,
    intermediate=128,
    vocab=32,
    max_seq=16,
)

TINY_INTERNLM = ArchSpec(
    name="tiny-internlm",
    family="internlm_gqa",
    n_layers=2,
    hidden=64,
    n_q=4,
    n_kv=2,
    head_dim=16,
    intermediate=128,
    vocab=32,
    max_seq=16,
)

# Timing-only toy: NF4-legal, still no checkpoint. 4 layers, hidden 512.
MEDIUM_LLAMA = ArchSpec(
    name="medium-llama",
    family="llama_swiglu",
    n_layers=4,
    hidden=512,
    n_q=8,
    n_kv=2,
    head_dim=64,
    intermediate=1024,
    vocab=128,
    max_seq=32,
)
