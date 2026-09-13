"""Meta skeleton -> ``CompressedLinear`` -> CHR0 bytes.

The order is the whole point (token-loop.md §3): build the HuggingFace module
tree on ``torch.device("meta")`` from ``config.json`` only, swap every
``nn.Linear`` for a :class:`~gpu.host.linear.CompressedLinear` *before* any
device is touched, then hand each module the bytes of the ``.chr``.

Never called here: ``from_pretrained`` on the original shards,
``model.cuda()``, ``model.to(dtype)``, ``load_state_dict``, ``tie_weights``,
``device_map``. Meta weights are read for their ``shape`` only -- no ``.cpu()``,
no ``.float()``, no ``.cuda()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from ._deps import Header, load_header, materialize_nf4
from .blobs import iter_bf16, load_bf16
from .embedding import Nf4Embedding, dequant_table
from .linear import CompressedLinear

__all__ = [
    "LINEAR_KINDS",
    "LoadReport",
    "build_skeleton",
    "replace_linears",
    "load_chr_nf4",
    "load_model",
    "linear_modules",
]

# CHR0 `kind`s that are a Linear in the module tree. `embed` is a lookup, not a
# GEMM (wave2-gpu.md non-goals), so it never reaches the kernel even though
# `iter_linears` yields it.
LINEAR_KINDS = frozenset({"q", "k", "v", "o", "qkv", "gate", "up", "down", "lm_head"})

MIB = 1024 * 1024


@dataclass
class LoadReport:
    """What ``load_chr_nf4`` actually put on the device."""

    linears: int = 0
    linear_bytes: int = 0
    bf16_tensors: int = 0
    bf16_bytes: int = 0
    embed_mode: str = "none"
    embed_bytes: int = 0
    tied_lm_head: bool = False
    seconds: float = 0.0
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    leftover_meta: list[str] = field(default_factory=list)

    @property
    def device_mib(self) -> float:
        return (self.linear_bytes + self.bf16_bytes + self.embed_bytes) / MIB

    def __str__(self) -> str:
        return (
            f"{self.linears} nf4 linears ({self.linear_bytes / MIB:.1f} MiB), "
            f"{self.bf16_tensors} bf16 tensors ({self.bf16_bytes / MIB:.2f} MiB), "
            f"embed={self.embed_mode} ({self.embed_bytes / MIB:.1f} MiB), "
            f"tied_lm_head={self.tied_lm_head}, total={self.device_mib:.1f} MiB, "
            f"{self.seconds:.1f}s"
            + (f", missing={self.missing}" if self.missing else "")
            + (f", leftover_meta={self.leftover_meta}" if self.leftover_meta else "")
        )


# --------------------------------------------------------------------------- #
# skeleton
# --------------------------------------------------------------------------- #


def build_skeleton(
    model_id: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = "sdpa",
    trust_remote_code: bool = False,
):
    """``AutoConfig`` + ``from_config`` on ``meta``. Reads ``config.json`` only.

    ``model_id`` is a directory with a config; its weight shards are never
    opened. Zero bytes of parameters exist after this call: every tensor is a
    meta tensor with a shape and no storage.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(
        model_id, local_files_only=True, trust_remote_code=trust_remote_code
    )
    attn = attn_implementation
    if trust_remote_code:
        attn = getattr(cfg, "attn_implementation", None) or attn_implementation
    kwargs = {}
    if attn:
        kwargs["attn_implementation"] = attn
    with torch.device("meta"):
        try:
            model = AutoModelForCausalLM.from_config(
                cfg, dtype=dtype, trust_remote_code=trust_remote_code, **kwargs
            )
        except TypeError:  # transformers < 5 spelled it torch_dtype
            try:
                model = AutoModelForCausalLM.from_config(
                    cfg, torch_dtype=dtype, trust_remote_code=trust_remote_code, **kwargs
                )
            except TypeError:
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, **kwargs)
    model.eval()
    return model


def replace_linears(
    model: nn.Module,
    *,
    skip: Iterable[str] = (),
    dtype: torch.dtype = torch.bfloat16,
) -> list[str]:
    """Swap every ``nn.Linear`` for a ``CompressedLinear``, in place.

    Returns the qualified names that were replaced. ``nn.Embedding`` is not a
    ``Linear`` and is left alone (see ``load_chr_nf4``'s ``embed`` argument).
    Only ``in_features`` / ``out_features`` / ``bias is not None`` are read off
    the meta child -- its storage is never touched.
    """
    skip_set = {s for s in skip}
    replaced: list[str] = []

    def walk(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            qualified = f"{prefix}{name}"
            if isinstance(child, CompressedLinear):
                continue
            if isinstance(child, nn.Linear) and name not in skip_set and qualified not in skip_set:
                setattr(
                    module,
                    name,
                    CompressedLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        dtype=dtype,
                    ),
                )
                replaced.append(qualified)
            else:
                walk(child, f"{qualified}.")

    walk(model, "")
    return replaced


def linear_modules(model: nn.Module) -> dict[str, CompressedLinear]:
    """Qualified name -> ``CompressedLinear``, in module order."""
    return {n: m for n, m in model.named_modules() if isinstance(m, CompressedLinear)}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _embedding_name(model: nn.Module) -> str | None:
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Embedding, Nf4Embedding)):
            return name
    return None


def _assign_weight(module: nn.Module, attr: str, tensor: torch.Tensor) -> None:
    """Put a real tensor where a meta parameter/buffer was."""
    if attr in module._parameters:
        module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
    elif attr in module._buffers:
        module._buffers[attr] = tensor
    else:
        raise AttributeError(f"{type(module).__name__} has no parameter/buffer {attr!r}")


def _in_scope(layer: int | None, layers: Sequence[int] | None) -> bool:
    return layers is None or layer is None or layer in layers


def _rebuild_rotary(model: nn.Module, device: torch.device) -> list[str]:
    """RoPE ``inv_freq`` is computed, not stored: recompute it off ``meta``.

    ``from_config`` built it under ``torch.device("meta")``, so the buffer has a
    shape and no values. Rebuilding the module on the real device is cheaper and
    safer than reimplementing the rope init here.
    """
    rebuilt: list[str] = []
    for name, mod in list(model.named_modules()):
        buffers = [(bn, b) for bn, b in mod.named_buffers(recurse=False) if b is not None]
        if not any(b.is_meta for _, b in buffers):
            continue
        if not hasattr(mod, "rope_init_fn") and "rotary" not in type(mod).__name__.lower():
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        fresh = None
        try:
            fresh = type(mod)(model.config).to(device)
        except TypeError:
            try:
                fresh = type(mod)(config=model.config).to(device)
            except TypeError:
                dim = int(getattr(mod, "dim", 0))
                if dim <= 0:
                    cfg = getattr(model, "config", None)
                    heads = int(getattr(cfg, "num_attention_heads", 0) or 1)
                    dim = int(getattr(cfg, "hidden_size", 0)) // heads
                kwargs = {}
                if hasattr(mod, "max_position_embeddings"):
                    kwargs["max_position_embeddings"] = int(mod.max_position_embeddings)
                if hasattr(mod, "base"):
                    kwargs["base"] = float(mod.base)
                if hasattr(mod, "scaling_factor"):
                    kwargs["scaling_factor"] = float(mod.scaling_factor)
                fresh = type(mod)(dim, **kwargs).to(device)
        setattr(parent, child, fresh)
        rebuilt.append(name)
    return rebuilt


def _leftover_meta(model: nn.Module, layers: Sequence[int] | None) -> list[str]:
    """Tensors still on ``meta`` that were in scope, i.e. that we meant to fill."""
    out = []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t is None or not t.is_meta:
            continue
        if not _in_scope(_layer_of(name), layers):
            continue
        out.append(name)
    return out


def _layer_of(name: str) -> int | None:
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def load_chr_nf4(
    model: nn.Module,
    path: str,
    *,
    device: str | torch.device = "cuda",
    header: Header | None = None,
    embed: str = "rows",
    layers: Sequence[int] | None = None,
    verbose: bool = False,
) -> LoadReport:
    """Fill a replaced skeleton from ``path``. The ``.chr`` is the only file read.

    ``embed``:

    * ``"rows"``    -- ``Nf4Embedding``, rows decoded per step (default).
    * ``"bf16"``    -- one BF16 ``[vocab, hidden]`` table for ``nn.Embedding``
                       (594 MiB on 3B). Permitted for the embedding only.
    * ``"skip"``    -- leave the embedding on ``meta``; a forward from
                       ``input_ids`` will then fail, ``inputs_embeds`` still works.

    ``layers`` restricts loading to those decoder layers (for a cheap
    layer-0-only session); ``None`` loads everything.

    The header is parsed once and passed down, so 300+ matrices do not reparse
    65 KiB of JSON each (gpu-abi.md §2).
    """
    t0 = time.perf_counter()
    path = str(path)
    hdr = header if header is not None else load_header(path)
    dev = torch.device(device)
    report = LoadReport()
    layer_set = None if layers is None else sorted({int(v) for v in layers})

    modules = dict(model.named_modules())
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))

    # --- the embedding table, materialized at most once -------------------
    embed_name = _embedding_name(model)
    embed_info = hdr.tensors.get(embed_name) if embed_name else None
    embed_matrix = None
    lm_head_names = [
        n
        for n in linear_modules(model)
        if n.rsplit(".", 1)[-1] in ("lm_head", "output")
    ]
    need_embed_bytes = embed_info is not None and embed_info.codec == "nf4" and (
        embed != "skip" or (tied and any(n not in hdr.tensors for n in lm_head_names))
    )
    if need_embed_bytes:
        embed_matrix = materialize_nf4(path, embed_name, dev, header=hdr)

    # --- NF4 linears -------------------------------------------------------
    for name, mod in linear_modules(model).items():
        info = hdr.tensors.get(name)
        if info is None:
            # Qwen2.5-3B ties lm_head to the embedding and writes no lm_head
            # tensor. Tied means *the same rows*: share the ChrMatrix, do not
            # load a second [vocab, hidden] copy.
            if (
                name in lm_head_names
                and tied
                and embed_matrix is not None
                and (embed_matrix.M, embed_matrix.K) == (mod.M, mod.K)
            ):
                mod.attach(embed_matrix)
                report.tied_lm_head = True
                if verbose:
                    print(f"  {name}: tied to {embed_name} (shared packed)")
            else:
                report.missing.append(name)
            continue
        if info.codec != "nf4" or info.kind not in LINEAR_KINDS:
            report.skipped.append(f"{name} (kind={info.kind}, codec={info.codec})")
            continue
        if not _in_scope(info.layer, layer_set):
            report.skipped.append(f"{name} (layer {info.layer} out of scope)")
            continue
        matrix = materialize_nf4(path, name, dev, header=hdr)
        mod.attach(matrix)
        report.linears += 1
        report.linear_bytes += matrix.nbytes
        if verbose:
            print(f"  {name}: [{matrix.M},{matrix.K}] {matrix.nbytes / MIB:.2f} MiB")

    # --- bf16 payload: norms and biases ------------------------------------
    for name in iter_bf16(hdr):
        info = hdr.tensors[name]
        if not _in_scope(info.layer, layer_set):
            continue
        target, attr = None, None
        if name in modules:
            target, attr = modules[name], "weight"
        elif name.endswith(".bias") and name[: -len(".bias")] in modules:
            target, attr = modules[name[: -len(".bias")]], "bias"
        if target is None:
            report.skipped.append(f"{name} (no module)")
            continue
        tensor = load_bf16(path, name, dev, header=hdr)
        if isinstance(target, CompressedLinear) and attr == "bias":
            target.set_bias(tensor)
        else:
            _assign_weight(target, attr, tensor)
        report.bf16_tensors += 1
        report.bf16_bytes += tensor.numel() * tensor.element_size()

    # --- embed_tokens ------------------------------------------------------
    if embed_name is not None and embed_info is not None:
        report.embed_mode = _load_embedding(
            model, embed_name, embed_info, embed_matrix, embed, path, hdr, dev, report
        )
    elif embed_name is not None:
        report.missing.append(embed_name)

    _rebuild_rotary(model, dev)
    report.leftover_meta = _leftover_meta(model, layer_set)
    report.seconds = time.perf_counter() - t0
    return report


def _load_embedding(
    model: nn.Module,
    embed_name: str,
    embed_info,
    embed_matrix,
    embed: str,
    path: str,
    hdr: Header,
    dev: torch.device,
    report: LoadReport,
) -> str:
    parent_name, _, child = embed_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    current = getattr(parent, child)

    if embed_info.codec == "bf16":
        table = load_bf16(path, embed_name, dev, header=hdr)
        _assign_weight(current, "weight", table)
        report.embed_bytes += table.numel() * table.element_size()
        return "bf16-blob"

    if embed == "skip":
        return "skip (left on meta)"

    if embed == "rows":
        module = Nf4Embedding(
            embed_matrix.M,
            embed_matrix.K,
            padding_idx=getattr(current, "padding_idx", None),
        )
        module.attach(embed_matrix)
        setattr(parent, child, module)
        report.embed_bytes += embed_matrix.nbytes
        return "nf4-rows"

    if embed == "bf16":
        table = dequant_table(embed_matrix)
        _assign_weight(current, "weight", table)
        report.embed_bytes += table.numel() * table.element_size()
        return "nf4-dequant-table"

    raise ValueError(f"embed={embed!r}; expected 'rows', 'bf16' or 'skip'")


def load_model(
    model_id: str,
    chr_path: str,
    *,
    device: str | torch.device = "cuda",
    embed: str = "rows",
    layers: Sequence[int] | None = None,
    skip: Iterable[str] = (),
    verbose: bool = False,
    trust_remote_code: bool = False,
):
    """``build_skeleton`` + ``replace_linears`` + ``load_chr_nf4``, in that order."""
    model = build_skeleton(model_id, trust_remote_code=trust_remote_code)
    replace_linears(model, skip=skip)
    report = load_chr_nf4(
        model, chr_path, device=device, embed=embed, layers=layers, verbose=verbose
    )
    return model, report
