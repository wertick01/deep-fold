"""One attach path: walk the module tree, return a :class:`DriverPlan` or raise.

This replaces the InternLM fork. There is no ``hasattr(layer0.attention,
"wqkv")`` anywhere downstream of this module: the walker maps every
``nn.Linear`` (or :class:`~gpu.host.linear.CompressedLinear` seat) onto a GEMM
slot by name, checks that every layer has the *same* slot set, picks the QKV
packer from the fused matrix's own last component, and then either names an
implemented glue family or refuses with the offending names printed.

Two graphs are implemented (``docs/tz/wave10-product.md`` §2.1):

* ``llama_swiglu`` -- split ``q/k/v/o`` + SwiGLU + Llama RMS + full causal +
  full RoPE. Qwen2 (measured 3B/14B), Llama 3.x without qk-norm, and Mistral
  **only** when the config's sliding window is inert.
* ``internlm_gqa`` -- the same body with one fused ``wqkv`` and the InternLM
  GQA packer (measured 20B).

Everything else is a *named* refusal: ``phi3_concat``, ``gemma_gelu``,
``neox_interleaved``, MoE, vision, SWA, GPT-2 Conv1D, leftover Linears. Fail
closed, never ``report.skipped``.

    from gpu.host.attach import attach
    plan = attach(r"C:\\dev\\models\\Qwen2.5-3B-Instruct")   # meta skeleton, no shards
    plan.family, plan.qkv_pack
    ('llama_swiglu', None)

No weights are read here and no device is touched: a directory is walked as a
``torch.device("meta")`` skeleton built from ``config.json`` only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch.nn as nn

from gpu.graphs import (
    IMPLEMENTED_FAMILIES,
    LEGAL_SLOT_SETS,
    cfg_get,
    config_refusal,
    effective_sliding_window,
    fused_qkv_packer,
    is_embed_name,
    is_final_norm_name,
    is_lm_head_name,
    is_router,
    looks_fused_qkv,
    needs_remote_code,
    norm_slot,
    refuse,
    slot_for,
    vision_prefix,
)

__all__ = [
    "AttachError",
    "DriverPlan",
    "LayerPlan",
    "attach",
    "attach_dir",
    "attach_module",
    "plan_violations",
]

_GGUF_MARKERS = (".gguf", ".ollama", "sha256-")


class AttachError(ValueError):
    """The graph is not wired. The message names what was found."""


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LayerPlan:
    """One decoder layer, as qualified names. No modules, no tensors."""

    index: int
    gemms: Mapping[str, str]  # slot -> qualified name
    norm1: str
    norm2: str
    attn: str  # the attention submodule (head_dim / scaling / rotary live here)
    mlp: str
    q_norm: str | None = None
    k_norm: str | None = None

    @property
    def slots(self) -> frozenset[str]:
        return frozenset(self.gemms)


@dataclass(frozen=True)
class DriverPlan:
    """What ``TokenLoop`` binds against. Produced only by :func:`attach`."""

    family: str
    attn: str  # "split" | "fused"
    qkv_pack: str | None  # None | "internlm_gqa" | "concat" | "neox_interleaved"
    mlp: str  # "swiglu_split" | "swiglu_fused_gate_up" | "gelu_fc"
    act: str
    norm: str  # "rms_llama" | "rms_gemma" | "layer"
    qk_norm: bool
    sliding_window: int | None
    rope: str  # "full" | "partial"
    trust_remote: bool
    backbone: str  # "" or the attribute holding `layers`, e.g. "model"
    layers: tuple[LayerPlan, ...]
    embed: str
    lm_head: str
    final_norm: str
    model_type: str = ""
    extras: tuple[str, ...] = field(default_factory=tuple)


    @property
    def gemm_names(self) -> tuple[str, ...]:
        """Every qualified name ``replace_linears`` is allowed to swap."""
        names: list[str] = []
        for layer in self.layers:
            names.extend(layer.gemms[slot] for slot in sorted(layer.gemms))
        names.append(self.lm_head)
        return tuple(names)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"DriverPlan({self.family}, attn={self.attn}, pack={self.qkv_pack}, "
            f"mlp={self.mlp}, act={self.act}, norm={self.norm}, "
            f"layers={len(self.layers)}, model_type={self.model_type})"
        )


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #


def _is_linear_seat(mod: nn.Module) -> bool:
    """``nn.Linear``, or one of our compressed seats standing in for one.

    Duck-typed rather than imported so this module stays cheap and so a VQ seat
    counts too: ``CompressedLinear`` is an ``nn.Module``, not an ``nn.Linear``,
    because it must never own an ``[out, in]`` parameter.
    """
    if isinstance(mod, nn.Linear):
        return True
    if isinstance(mod, nn.Embedding):
        return False
    return all(hasattr(mod, attr) for attr in ("in_features", "out_features", "K", "M"))


def _is_conv1d(mod: nn.Module) -> bool:
    return type(mod).__name__ == "Conv1D"


def _decoder_layers(model: nn.Module) -> tuple[str, Sequence[nn.Module]]:
    """``("model", layers)``, or a named refusal for the backbones we do not drive."""
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None) if base is not None else None
    if layers is not None and len(layers) > 0:
        return "model", layers

    neox = getattr(model, "gpt_neox", None)
    if neox is not None and getattr(neox, "layers", None) is not None:
        raise AttachError(
            refuse.family("neox_interleaved", attn="fused", mlp="gelu_fc", act="gelu")
        )

    transformer = getattr(model, "transformer", None)
    if transformer is not None and getattr(transformer, "h", None) is not None:
        raise AttachError(refuse.gpt2_conv1d("transformer.h"))

    own = getattr(model, "layers", None)
    if own is not None and len(own) > 0:
        return "", own

    if layers is not None:
        raise AttachError(refuse.no_decoder("model.model.layers is empty"))
    raise AttachError(refuse.no_decoder(f"{type(model).__name__} has no .model.layers"))


@dataclass
class _LayerWalk:
    gemms: dict[str, str] = field(default_factory=dict)
    packs: dict[str, str] = field(default_factory=dict)
    norms: dict[str, str] = field(default_factory=dict)
    extras: list[str] = field(default_factory=list)


def _walk_layer(layer: nn.Module, prefix: str) -> _LayerWalk:
    out = _LayerWalk()
    for sub, mod in layer.named_modules():
        if not sub:
            continue
        qualified = f"{prefix}.{sub}"

        if ".experts." in f".{sub}." or sub.endswith(".experts") or sub == "experts":
            raise AttachError(refuse.moe(qualified))
        if _is_conv1d(mod):
            raise AttachError(refuse.gpt2_conv1d(qualified))

        which = norm_slot(qualified)
        if which is not None and not _is_linear_seat(mod):
            out.norms.setdefault(which, qualified)
            continue

        if not _is_linear_seat(mod):
            continue

        if is_router(qualified):
            raise AttachError(refuse.moe(f"{qualified} (MoE router)"))

        slot = slot_for(qualified)
        if slot is None:
            if looks_fused_qkv(qualified):
                raise AttachError(refuse.no_packer(qualified))
            out.extras.append(qualified)
            continue
        if slot == "qkv":
            packer = fused_qkv_packer(qualified)
            if packer is None:  # pragma: no cover - slot_for only says qkv for known names
                raise AttachError(refuse.no_packer(qualified))
            out.packs[qualified] = packer
        if slot in out.gemms:
            raise AttachError(refuse.duplicate_slot(slot, out.gemms[slot], qualified))
        out.gemms[slot] = qualified
    return out


def _norm_flavour(cfg: Any, model_type: str) -> str:
    if model_type.startswith("gemma"):
        return "rms_gemma"
    if cfg_get(cfg, "rms_norm_eps") is not None:
        return "rms_llama"
    if cfg_get(cfg, "layer_norm_eps") is not None or cfg_get(cfg, "layer_norm_epsilon") is not None:
        return "layer"
    return "rms_llama"


def _family_id(*, mlp: str, act: str, norm: str, pack: str | None, model_type: str) -> str:
    """The glue family this graph *is*, implemented or not.

    Named from the graph, never from ``model_type``: a fused ``qkv_proj`` is
    ``phi3_concat`` whatever the config calls itself, and a split-SwiGLU
    ``olmo`` is ``llama_swiglu`` without a catalog row.
    """
    if mlp == "swiglu_fused_gate_up":
        return "phi3_concat"
    if mlp == "gelu_fc":
        if pack == "neox_interleaved" or model_type.startswith("gpt_neox"):
            return "neox_interleaved"
        return "phi_gelu"
    if norm == "rms_gemma":
        return "gemma_gelu"
    if act != "silu":
        if model_type.startswith("gemma") or act == "gelu_pytorch_tanh":
            return "gemma_gelu"
        return f"{act}_mlp"
    if norm == "layer":
        return "layer_swiglu"
    if pack is None:
        return "llama_swiglu"
    if pack == "internlm_gqa":
        return "internlm_gqa"
    return f"{pack}_swiglu"


def _find_one(model: nn.Module, predicate, *, want_linear: bool | None = None) -> str | None:
    for name, mod in model.named_modules():
        if not name or ".layers." in name:
            continue
        if want_linear is True and not _is_linear_seat(mod):
            continue
        if want_linear is False and _is_linear_seat(mod):
            continue
        if predicate(name):
            return name
    return None


def attach_module(
    model: nn.Module,
    *,
    config: Any = None,
    trust_remote_code: bool | None = None,
) -> DriverPlan:
    """Walk an existing module tree (meta skeleton or loaded model) into a plan."""
    cfg = config if config is not None else getattr(model, "config", None)
    if cfg is None:
        raise AttachError("attach: model has no .config and none was passed")

    refusal = config_refusal(cfg)
    if refusal is not None:
        raise AttachError(refusal)

    model_type = str(cfg_get(cfg, "model_type", "") or "").lower()
    backbone, layers = _decoder_layers(model)

    for name, _ in model.named_modules():
        head = vision_prefix(name)
        if head is not None:
            raise AttachError(refuse.vision(name))

    prefix = f"{backbone}.layers" if backbone else "layers"
    walks = [_walk_layer(layer, f"{prefix}.{i}") for i, layer in enumerate(layers)]

    first = frozenset(walks[0].gemms)
    for index, walk in enumerate(walks[1:], start=1):
        if frozenset(walk.gemms) != first:
            raise AttachError(refuse.non_uniform(first, frozenset(walk.gemms), index))
    packs = {p for walk in walks for p in walk.packs.values()}
    if len(packs) > 1:
        raise AttachError(refuse.mixed_packers(packs))

    graph = LEGAL_SLOT_SETS.get(first)
    if graph is None:
        raise AttachError(refuse.slot_set(first))
    attn_kind, mlp_kind = graph
    pack = next(iter(packs)) if packs else None

    act = str(cfg_get(cfg, "hidden_act", None) or cfg_get(cfg, "hidden_activation", None) or "silu")
    norm = _norm_flavour(cfg, model_type)
    family = _family_id(mlp=mlp_kind, act=act, norm=norm, pack=pack, model_type=model_type)

    # --- leftovers, before we bother naming the family --------------------
    slot_names = {name for walk in walks for name in walk.gemms.values()}
    lm_head = _find_one(model, is_lm_head_name, want_linear=True)
    embed = _find_one(model, is_embed_name, want_linear=False)
    final_norm = _find_one(model, is_final_norm_name, want_linear=False)

    leftover = [
        name
        for name, mod in model.named_modules()
        if name and _is_linear_seat(mod) and name not in slot_names and name != lm_head
    ]
    leftover += [name for walk in walks for name in walk.extras if name not in slot_names]
    if leftover:
        raise AttachError(refuse.unclassified(sorted(set(leftover))))

    if lm_head is None:
        raise AttachError(refuse.missing_piece("lm_head", ("lm_head", "output")))
    if embed is None:
        raise AttachError(refuse.missing_piece("embedding", ("embed_tokens", "tok_embeddings")))
    if final_norm is None:
        raise AttachError(refuse.missing_piece("final norm", ("norm", "ln_f")))

    layer_plans: list[LayerPlan] = []
    for index, walk in enumerate(walks):
        for which in ("norm1", "norm2"):
            if which not in walk.norms:
                raise AttachError(refuse.missing_norm(f"{prefix}.{index}", which))
        o_name = walk.gemms["o"]
        gate_name = walk.gemms.get("gate") or walk.gemms.get("gate_up") or walk.gemms["down"]
        layer_plans.append(
            LayerPlan(
                index=index,
                gemms=dict(walk.gemms),
                norm1=walk.norms["norm1"],
                norm2=walk.norms["norm2"],
                attn=o_name.rsplit(".", 1)[0],
                mlp=gate_name.rsplit(".", 1)[0],
                q_norm=walk.norms.get("q_norm"),
                k_norm=walk.norms.get("k_norm"),
            )
        )

    qk_norm = any(p.q_norm or p.k_norm for p in layer_plans)

    # --- the two gates that keep a wrong answer off the card --------------
    if family not in IMPLEMENTED_FAMILIES:
        raise AttachError(
            refuse.family(family, attn=attn_kind, mlp=mlp_kind, act=act, norm=norm)
        )
    if qk_norm:
        found = next(p.q_norm or p.k_norm for p in layer_plans if p.q_norm or p.k_norm)
        raise AttachError(refuse.qk_norm(str(found)))
    if norm == "rms_llama" and cfg_get(cfg, "rms_norm_eps") is None:
        raise AttachError(
            "attach: rms_llama plan needs config.json rms_norm_eps; it is missing."
        )

    trust = needs_remote_code(cfg) if trust_remote_code is None else bool(trust_remote_code)
    return DriverPlan(
        family=family,
        attn=attn_kind,
        qkv_pack=pack,
        mlp=mlp_kind,
        act=act,
        norm=norm,
        qk_norm=False,
        sliding_window=effective_sliding_window(cfg),
        rope="full",
        trust_remote=trust,
        backbone=backbone,
        layers=tuple(layer_plans),
        embed=embed,
        lm_head=lm_head,
        final_norm=final_norm,
        model_type=model_type,
        extras=(),
    )


def _read_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise AttachError(f"attach: {model_dir / 'config.json'} is not a JSON object")
    return cfg


def attach_dir(
    model_dir: str | Path,
    *,
    trust_remote_code: bool | None = None,
    dtype: Any = None,
) -> DriverPlan:
    """``config.json`` pre-refuse, then a meta skeleton, then the walk.

    No shard is opened and no device is touched: ``build_skeleton`` builds the
    HuggingFace tree under ``torch.device("meta")`` from the config alone.
    """
    text = str(model_dir).replace("\\", "/").lower()
    if any(marker in text for marker in _GGUF_MARKERS) or "/blobs/" in text:
        raise AttachError(
            f"attach: {model_dir} is a GGUF / Ollama blob path. CHR0 reads "
            "HuggingFace safetensors only; the file was not opened."
        )

    path = Path(model_dir)
    if not (path / "config.json").is_file():
        raise AttachError(f"attach: no config.json in {path}")
    cfg = _read_config(path)

    refusal = config_refusal(cfg)
    if refusal is not None:
        raise AttachError(refusal)

    trust = needs_remote_code(cfg) if trust_remote_code is None else bool(trust_remote_code)
    from .model import build_skeleton

    kwargs: dict[str, Any] = {"trust_remote_code": trust}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = build_skeleton(str(path), **kwargs)
    return attach_module(model, config=cfg, trust_remote_code=trust)


def attach(
    model_or_dir: Any,
    *,
    trust_remote_code: bool | None = None,
) -> DriverPlan:
    """``attach(model)`` or ``attach(directory)`` -> :class:`DriverPlan`, or raise."""
    if isinstance(model_or_dir, (str, Path)):
        return attach_dir(model_or_dir, trust_remote_code=trust_remote_code)
    return attach_module(model_or_dir, trust_remote_code=trust_remote_code)


# --------------------------------------------------------------------------- #
# after the load: the plan has to be *wired*, not merely replaced
# --------------------------------------------------------------------------- #


def plan_violations(plan: DriverPlan, report: Any) -> list[str]:
    """Reasons this loaded model must not generate. Empty means go.

    ``load_chr_nf4`` reports rather than raises so the lab can print a partial
    load. The product path (``deepfold run`` / ``TokenLoop``) turns the same
    report into a refusal: a plan slot that stayed BF16, stayed on meta, or was
    skipped as ``kind=other`` is a wrong answer, not a slow one
    (wave8-arch §2.5).
    """
    slots = set(plan.gemm_names)
    out: list[str] = []
    for name in getattr(report, "missing", ()) or ():
        if name in slots or name == plan.embed:
            out.append(f"missing from the .chr: {name}")
    for entry in getattr(report, "skipped", ()) or ():
        head = str(entry).split(" ", 1)[0]
        if head in slots:
            out.append(f"skipped a plan slot: {entry}")
    for name in getattr(report, "leftover_meta", ()) or ():
        out.append(f"still on meta: {name}")
    return out
