"""Which transformer graphs this driver can drive, named from `config.json` alone.

Torch-free on purpose. ``gpu/cli/arch.py`` runs on a laptop with no torch and
must still refuse a Gemma tree *before* ``chr compress`` spends twenty minutes
packing a graph the loop cannot drive; ``gpu/host/attach.py`` imports the same
tables and strings after it has a module tree to walk. One copy of the refuse
copy, two callers (``docs/tz/wave8-arch.md`` §2.3).

Two rules this module exists to enforce:

* the **family** is the unit of support, not ``model_type``. Qwen2, Llama and
  Mistral-without-SWA are one implemented family (``llama_swiglu``); InternLM2
  is the same body with a fused-QKV packer (``internlm_gqa``). A future
  ``olmo`` that is literally split-SwiGLU attaches without a catalog row, and a
  ``llama`` with a live sliding window does not.
* the **packer** comes from the fused matrix's own name (``wqkv`` vs
  ``qkv_proj`` vs ``query_key_value``), never from ``model_type`` and never
  from "it is fused, so it must be InternLM". Feeding a Phi-3 concat layout to
  InternLM's GQA split would generate fluent garbage.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "IMPLEMENTED_FAMILIES",
    "INTERNLM_EXTRA_MODULES",
    "LEGAL_SLOT_SETS",
    "QKV_PACKERS",
    "SLOT_ALIASES",
    "cfg_get",
    "config_family",
    "config_refusal",
    "effective_sliding_window",
    "fused_qkv_packer",
    "is_embed_name",
    "is_final_norm_name",
    "is_lm_head_name",
    "is_router",
    "last_component",
    "looks_fused_qkv",
    "needs_internlm_extras",
    "needs_remote_code",
    "norm_slot",
    "refuse",
    "slot_for",
    "vision_prefix",
]

#: Glue families ``TokenLoop`` actually implements. Everything else is a *named*
#: refusal: the walker says which graph it found and stops.
IMPLEMENTED_FAMILIES = ("llama_swiglu", "internlm_gqa")

# --------------------------------------------------------------------------- #
# name -> slot (wave8-arch.md §1.1 role table)
# --------------------------------------------------------------------------- #

#: Fused attention-in matrices: last component -> packer id. The packer is a
#: runtime property of the graph, not a CHR0 header field; ``kind`` is ``qkv``
#: for all three (wave8-arch §1.4).
QKV_PACKERS = {
    "wqkv": "internlm_gqa",  # InternLM2: per KV head, n_rep queries then K then V
    "qkv_proj": "concat",  # Phi-3: [Q | K | V] on axis 0
    "query_key_value": "neox_interleaved",  # GPT-NeoX / Falcon: per head [q,k,v]
}

#: Last component -> GEMM slot, for the graphs this walker can *name*. Slots
#: ``gate_up`` and ``fc`` are named so the refusal can say ``phi3_concat``
#: instead of "unclassified Linear"; they are not CHR0 kinds in this wave.
SLOT_ALIASES = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "wo": "o",
    "gate_proj": "gate",
    "w1": "gate",
    "up_proj": "up",
    "w3": "up",
    "down_proj": "down",
    "w2": "down",
    "gate_up_proj": "gate_up",
    "fc1": "fc",
    "dense_h_to_4h": "fc",
    "fc2": "down",
    "dense_4h_to_h": "down",
}

#: ``dense`` is attention-out in Phi-2 / GPT-NeoX and MLP-out elsewhere, so the
#: last component alone is not enough: the parent decides (wave8-arch §1.3).
_ATTN_PARENTS = frozenset({"attention", "self_attn", "attn", "self_attention"})

#: ``mlp.gate`` is a MoE router, not ``gate_proj``.
_ROUTER_PARENTS = frozenset({"mlp", "moe", "block_sparse_moe", "ffn", "feed_forward"})

#: Per-layer slot sets this wave can name. Value is ``(attn, mlp)``.
LEGAL_SLOT_SETS: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"q", "k", "v", "o", "gate", "up", "down"}): ("split", "swiglu_split"),
    frozenset({"qkv", "o", "gate", "up", "down"}): ("fused", "swiglu_split"),
    frozenset({"qkv", "o", "gate_up", "down"}): ("fused", "swiglu_fused_gate_up"),
    frozenset({"q", "k", "v", "o", "fc", "down"}): ("split", "gelu_fc"),
    frozenset({"qkv", "o", "fc", "down"}): ("fused", "gelu_fc"),
}

_EMBED_NAMES = ("embed_tokens", "tok_embeddings", "wte", "embed_in")
_LM_HEAD_NAMES = ("lm_head", "output", "embed_out")
_FINAL_NORM_NAMES = ("norm", "final_layernorm", "ln_f", "final_layer_norm")
_NORM1_NAMES = ("input_layernorm", "attention_norm")
_NORM2_NAMES = ("post_attention_layernorm", "ffn_norm")

_VISION_PREFIXES = ("visual", "vision_tower", "vision_model", "multi_modal", "image_newline")


def last_component(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _parent_component(name: str) -> str:
    parts = name.split(".")
    return parts[-2] if len(parts) >= 2 else ""


def slot_for(qualified: str) -> str | None:
    """GEMM slot for one qualified module name, or ``None`` when unclassified.

    Path-sensitive on purpose: ``attention.dense`` is ``o`` while ``mlp.dense``
    is nothing we can name.
    """
    last = last_component(qualified)
    if last in QKV_PACKERS:
        return "qkv"
    hit = SLOT_ALIASES.get(last)
    if hit is not None:
        return hit
    if last == "dense" and _parent_component(qualified) in _ATTN_PARENTS:
        return "o"
    return None


def fused_qkv_packer(qualified: str) -> str | None:
    """Packer id for a fused attention-in matrix, from its own name."""
    return QKV_PACKERS.get(last_component(qualified))


def looks_fused_qkv(qualified: str) -> bool:
    """Does this name smell like one-matrix-in, three-heads-out?

    Used only to turn an unknown fused name (``Wqkv``, ``attn.qkv``) into the
    "no packer" refusal instead of the generic "unclassified Linear" one.
    """
    last = last_component(qualified)
    return "qkv" in last.lower() or last == "query_key_value"


def is_router(qualified: str) -> bool:
    return (
        last_component(qualified) == "gate"
        and _parent_component(qualified) in _ROUTER_PARENTS
    )


def norm_slot(qualified: str) -> str | None:
    last = last_component(qualified)
    if last in _NORM1_NAMES:
        return "norm1"
    if last in _NORM2_NAMES:
        return "norm2"
    if last == "q_norm":
        return "q_norm"
    if last == "k_norm":
        return "k_norm"
    return None


def is_embed_name(qualified: str) -> bool:
    return last_component(qualified) in _EMBED_NAMES


def is_lm_head_name(qualified: str) -> bool:
    last = last_component(qualified)
    return last in _LM_HEAD_NAMES and "norm" not in qualified


def is_final_norm_name(qualified: str) -> bool:
    return last_component(qualified) in _FINAL_NORM_NAMES


def vision_prefix(qualified: str) -> str | None:
    """The vision-tower component of this name, or ``None``.

    Every component is checked, not just the first: Qwen2-VL hangs its tower at
    ``model.visual``, so looking only at the head would miss it and the decoder
    would be packed with the tower silently skipped.
    """
    for part in qualified.split("."):
        if part in _VISION_PREFIXES:
            return part
    return None


# --------------------------------------------------------------------------- #
# config.json, read the same way whether it is a dict or a PretrainedConfig
# --------------------------------------------------------------------------- #


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_MOE_TYPES = frozenset(
    {
        "qwen2_moe",
        "qwen3_moe",
        "mixtral",
        "mixtral_moe",
        "dbrx",
        "deepseek",
        "deepseek_v2",
        "deepseek_v3",
        "olmoe",
        "granitemoe",
        "phimoe",
        "jamba",
    }
)
_MOE_COUNTS = ("num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok")

#: Activations whose glue (GeGLU, ungated GELU) TokenLoop does not implement.
_GELU_ACTS = frozenset(
    {"gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh", "quick_gelu", "relu", "relu2"}
)

#: ``model_type`` -> family, for the graphs whose glue is a *named* refusal.
#: A ``model_type`` that is not here is **not** a refusal by itself: the walker
#: gets to look at the module tree (wave8-arch §2.2).
_NAMED_FAMILIES = {
    "qwen2": "llama_swiglu",
    "llama": "llama_swiglu",
    "mistral": "llama_swiglu",
    "internlm2": "internlm_gqa",
    "gemma": "gemma_gelu",
    "gemma2": "gemma_gelu",
    "gemma3": "gemma_gelu",
    "gemma3_text": "gemma_gelu",
    "phi": "phi_gelu",
    "phi3": "phi3_concat",
    "phi4": "phi3_concat",
    "gpt_neox": "neox_interleaved",
    "falcon": "neox_interleaved",
    "gpt2": "gpt2_conv1d",
}

#: Remote code is a fact about the checkpoint, not about ``model_type``:
#: ``auto_map`` decides. This table is only the fallback for a config that
#: shipped without one (hand-written fixtures, and older InternLM2 trees).
_REMOTE_CODE_TYPES = frozenset({"internlm2"})

#: InternLM2's own Python needs these at *install* time; not a walker fact
#: (wave7-ux.md §4.1).
INTERNLM_EXTRA_MODULES = ("einops", "sentencepiece")


def effective_sliding_window(cfg: Any) -> int | None:
    """The window that is actually applied, or ``None`` for full causal.

    Qwen2.5 ships ``"sliding_window": 32768`` next to
    ``"use_sliding_window": false`` -- the number is inert and the measured 3B
    is full causal. Reading ``sliding_window`` alone would refuse the one model
    this driver has the most numbers for. Mistral sets the window and no
    switch: that one really is SWA and really is refused.
    """
    window = cfg_get(cfg, "sliding_window")
    if window in (None, 0, False):
        return None
    if cfg_get(cfg, "use_sliding_window", None) is False:
        return None
    try:
        return int(window)
    except (TypeError, ValueError):
        return None


def needs_remote_code(cfg: Any) -> bool:
    """``auto_map`` first; ``model_type`` only as a fallback."""
    if cfg_get(cfg, "auto_map"):
        return True
    return str(cfg_get(cfg, "model_type", "") or "") in _REMOTE_CODE_TYPES


def needs_internlm_extras(cfg: Any) -> bool:
    model_type = str(cfg_get(cfg, "model_type", "") or "")
    auto_map = cfg_get(cfg, "auto_map") or {}
    text = " ".join(str(v) for v in auto_map.values()) if isinstance(auto_map, Mapping) else ""
    return model_type == "internlm2" or "internlm" in text.lower()


def config_family(cfg: Any) -> str | None:
    """Family id from ``config.json`` alone, or ``None`` when only a walk can tell."""
    model_type = str(cfg_get(cfg, "model_type", "") or "").lower()
    named = _NAMED_FAMILIES.get(model_type)
    act = str(cfg_get(cfg, "hidden_act", "") or cfg_get(cfg, "hidden_activation", "") or "")
    if act in _GELU_ACTS:
        # An activation we have no glue for outranks the table: a `llama` fork
        # with gelu is not the measured llama_swiglu body.
        if named in IMPLEMENTED_FAMILIES or named is None:
            if model_type.startswith("gemma") or act == "gelu_pytorch_tanh":
                return "gemma_gelu"
            return f"{act}_mlp"
    return named


def _moe_evidence(cfg: Any) -> str | None:
    model_type = str(cfg_get(cfg, "model_type", "") or "").lower()
    if model_type in _MOE_TYPES or "moe" in model_type.split("_"):
        return f"config.json model_type={model_type}"
    for key in _MOE_COUNTS:
        value = cfg_get(cfg, key)
        try:
            if value is not None and int(value) > 0:
                return f"config.json {key}={int(value)}"
        except (TypeError, ValueError):
            continue
    return None


def _vision_evidence(cfg: Any) -> str | None:
    model_type = str(cfg_get(cfg, "model_type", "") or "").lower()
    parts = model_type.split("_")
    if "vl" in parts or "llava" in model_type or "vision" in model_type or "internvl" in model_type:
        return f"config.json model_type={model_type}"
    if cfg_get(cfg, "vision_config"):
        return "config.json vision_config"
    return None


def _partial_rope(cfg: Any) -> float | None:
    for key in ("partial_rotary_factor", "rotary_pct", "rotary_percentage"):
        value = cfg_get(cfg, key)
        if value is None:
            continue
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            continue
        if fraction not in (1.0,):
            return fraction
    return None


def config_refusal(cfg: Any) -> str | None:
    """Cheap pre-refuse from ``config.json``: MoE, vision, SWA, glue, partial RoPE.

    Runs before any skeleton is built and before ``chr compress`` is started,
    so a graph the loop cannot drive costs a JSON parse rather than twenty
    minutes of packing (wave8-arch §2.2, step 0).
    """
    found = _moe_evidence(cfg)
    if found:
        return refuse.moe(found)

    found = _vision_evidence(cfg)
    if found:
        return refuse.vision(found)

    window = effective_sliding_window(cfg)
    if window is not None:
        return refuse.sliding_window(window)

    fraction = _partial_rope(cfg)
    if fraction is not None:
        return refuse.partial_rope(fraction)

    family = config_family(cfg)
    if family is not None and family not in IMPLEMENTED_FAMILIES:
        act = str(cfg_get(cfg, "hidden_act", "") or "?")
        return refuse.family(family, act=act, source="config.json")
    return None


# --------------------------------------------------------------------------- #
# refuse copy (wave8-arch.md §2.3 -- implement these strings, do not improvise)
# --------------------------------------------------------------------------- #


class refuse:  # noqa: N801 - a namespace, used as `refuse.moe(...)`
    """Every refusal string in one place so tests can assert the wording."""

    @staticmethod
    def unclassified(names: Sequence[str]) -> str:
        listed = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
        last = last_component(names[0]) if names else "?"
        return (
            f"attach: unclassified Linear(s): {listed} (last={last}).\n"
            "CHR0 kind aliases cover q/k/v/o/qkv/gate/up/down (and listed fused names).\n"
            "This is not a silent skip; the graph is not wired."
        )

    @staticmethod
    def moe(found: str) -> str:
        return (
            f"attach: MoE experts are not in this wave (found {found}).\n"
            "Attention GEMMs may already classify; the loop cannot dispatch experts\n"
            "in one kernel. Refuse. See docs/tz/wave8-arch.md section 4."
        )

    @staticmethod
    def no_packer(name: str) -> str:
        return (
            f"attach: fused QKV at {name} (kind=qkv) has no packer.\n"
            "Known packers: internlm_gqa (wqkv), concat (qkv_proj), neox_interleaved\n"
            "(query_key_value). Refusing rather than applying InternLM's split."
        )

    @staticmethod
    def family(
        name: str,
        *,
        attn: str = "?",
        mlp: str = "?",
        act: str = "?",
        norm: str = "?",
        source: str = "",
    ) -> str:
        where = f" from {source}" if source else ""
        return (
            f"attach: graph is {name} (attn={attn}, mlp={mlp}, act={act}, "
            f"norm={norm}){where}.\n"
            f"TokenLoop implements {' and '.join(IMPLEMENTED_FAMILIES)} only."
        )

    @staticmethod
    def sliding_window(window: int) -> str:
        return (
            f"attach: sliding_window={window} is sliding window attention (SWA), "
            "not full causal.\n"
            "SWA glue is not in this wave: refusing rather than attending past the "
            "window.\n"
            "A config with sliding_window null, or use_sliding_window false "
            "(Qwen2.5), attaches."
        )

    @staticmethod
    def vision(found: str) -> str:
        return (
            f"attach: vision tower / multimodal graph ({found}).\n"
            "This wave drives text-only causal LMs. Refusing the whole model rather\n"
            "than packing the decoder and skipping the tower."
        )

    @staticmethod
    def qk_norm(found: str) -> str:
        return (
            f"attach: graph is llama_swiglu with qk-norm (found {found}).\n"
            "TokenLoop's forward does not apply those two RMS calls, and attaching\n"
            "without them is a silent quality bug. Refuse until the glue lands."
        )

    @staticmethod
    def partial_rope(fraction: float) -> str:
        return (
            f"attach: partial RoPE (rotary fraction {fraction}) is not in this wave.\n"
            "The loop rotates the whole head dimension. Refuse."
        )

    @staticmethod
    def non_uniform(first: Iterable[str], other: Iterable[str], index: int) -> str:
        return (
            "attach: non-uniform decoder -- layers do not share one slot set.\n"
            f"layer 0: {{{', '.join(sorted(first))}}}\n"
            f"layer {index}: {{{', '.join(sorted(other))}}}\n"
            "Refusing rather than driving a mixed graph."
        )

    @staticmethod
    def mixed_packers(packers: Iterable[str]) -> str:
        return (
            f"attach: layers disagree on the fused-QKV packer "
            f"({', '.join(sorted(packers))}).\n"
            "One decoder, one packing convention. Refuse."
        )

    @staticmethod
    def slot_set(slots: Iterable[str]) -> str:
        legal = "\n".join(
            f"  {{{', '.join(sorted(s))}}} -> attn={a}, mlp={m}"
            for s, (a, m) in LEGAL_SLOT_SETS.items()
        )
        return (
            f"attach: layer slot set {{{', '.join(sorted(slots))}}} is not a graph "
            "this wave can name.\nLegal sets:\n" + legal
        )

    @staticmethod
    def no_decoder(kind: str) -> str:
        return (
            f"attach: no decoder layer list ({kind}).\n"
            "Expected model.model.layers. Refuse."
        )

    @staticmethod
    def gpt2_conv1d(found: str) -> str:
        return (
            f"attach: graph is gpt2_conv1d (found {found}).\n"
            "GPT-2 stores attention as Conv1D with a transposed weight, under h.<n>\n"
            "rather than layers.<n>. The kernel contract is nn.Linear W[M,K]. Refuse."
        )

    @staticmethod
    def missing_norm(layer: str, which: str) -> str:
        return (
            f"attach: {layer} has no {which} (looked for "
            f"{', '.join(_NORM1_NAMES if which == 'norm1' else _NORM2_NAMES)}).\n"
            "TokenLoop needs both pre-norms per layer. Refuse."
        )

    @staticmethod
    def missing_piece(what: str, tried: Iterable[str]) -> str:
        return (
            f"attach: no {what} in the module tree (looked for "
            f"{', '.join(tried)}).\nRefuse."
        )

    @staticmethod
    def duplicate_slot(slot: str, first: str, second: str) -> str:
        return (
            f"attach: two Linears claim slot {slot} ({first}, {second}).\n"
            "Refusing rather than picking one by module order."
        )

    @staticmethod
    def load_violations(what: Iterable[str]) -> str:
        listed = "\n".join(f"  {line}" for line in what)
        return (
            "load_chr_nf4: the plan is not wired after loading.\n"
            f"{listed}\n"
            "Refusing to generate: a missing or skipped plan slot is a wrong answer,\n"
            "not a slow one."
        )
