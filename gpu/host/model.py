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

from ._deps import Header, load_header, materialize_nf4, quantized_codec
from .blobs import iter_bf16, load_bf16
from .embedding import Nf4Embedding, dequant_table
from .host_image import HostImage
from .linear import CompressedLinear
from .residency import (
    DEFAULT_POLICY,
    KV_BYTES_PER_TOKEN_32B,
    descs_from_header,
    overflow_resident_cap,
    plan_residency,
)
from .slots import OVERFLOW_SLOT_COUNT, SlotPair
from .vq_blobs import materialize_vq, reconstruct_vq
from .vq_linear import CompressedVqLinear, VqEmbedding

__all__ = [
    "LINEAR_KINDS",
    "LoadReport",
    "build_skeleton",
    "replace_linears",
    "load_chr_nf4",
    "load_chr_vq_model",
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
    codec: str = "nf4"
    overflow: bool = False
    streamed: int = 0
    streamed_bytes: int = 0
    resident_bytes: int = 0
    streamed_tape: tuple[str, ...] = ()
    slot_nbytes: int = 0
    slots: SlotPair | None = None
    compute: str = "gpu"
    n_gpu: int = 0
    n_cpu: int = 0
    cpu_linears: int = 0
    cpu_bytes: int = 0
    cpu_codec: str = "nf4"

    @property
    def device_mib(self) -> float:
        embed = 0 if str(self.embed_mode).endswith("-cpu") else self.embed_bytes
        return (self.linear_bytes + self.bf16_bytes + embed) / MIB

    def __str__(self) -> str:
        overflow = ""
        if self.overflow:
            overflow = (
                f", overflow streamed={self.streamed} "
                f"({self.streamed_bytes / MIB:.1f} MiB host) "
                f"resident_plan={self.resident_bytes / MIB:.1f} MiB "
                f"slot={self.slot_nbytes / MIB:.2f} MiB"
                f"×{self.slots.count if self.slots is not None else OVERFLOW_SLOT_COUNT}"
            )
        cpu = ""
        if self.n_cpu:
            cpu = (
                f", compute={self.compute} gpu_layers={self.n_gpu} "
                f"cpu_layers={self.n_cpu} "
                f"({self.cpu_bytes / MIB:.1f} MiB pageable)"
            )
        return (
            f"{self.linears} {self.codec} linears ({self.linear_bytes / MIB:.1f} MiB), "
            f"{self.bf16_tensors} bf16 tensors ({self.bf16_bytes / MIB:.2f} MiB), "
            f"embed={self.embed_mode} ({self.embed_bytes / MIB:.1f} MiB), "
            f"tied_lm_head={self.tied_lm_head}, total={self.device_mib:.1f} MiB, "
            f"{self.seconds:.1f}s"
            + overflow
            + cpu
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
    slots: Iterable[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seat: type = CompressedLinear,
) -> list[str]:
    """Swap ``nn.Linear`` for ``seat`` (NF4 or VQ), in place.

    With ``slots`` (a :class:`~gpu.host.attach.DriverPlan`'s ``gemm_names``) only
    those qualified names are swapped **and any other ``nn.Linear`` is an error**
    listing every name: swapping a router or a vision projection and hoping
    ``report.leftover_meta`` stays empty is the silence wave8-arch §2.5 forbids.

    Without ``slots`` every ``nn.Linear`` is swapped, which is what the
    lower-level verify scripts and notebooks have always done.

    Returns the qualified names that were replaced. ``nn.Embedding`` is not a
    ``Linear`` and is left alone (see ``load_chr_nf4``'s ``embed`` argument).
    Only ``in_features`` / ``out_features`` / ``bias is not None`` are read off
    the meta child -- its storage is never touched.
    """
    skip_set = {s for s in skip}
    wanted = None if slots is None else {str(s) for s in slots}
    replaced: list[str] = []
    leftover: list[str] = []

    def walk(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            qualified = f"{prefix}{name}"
            if isinstance(child, (CompressedLinear, CompressedVqLinear)):
                continue
            if isinstance(child, nn.Linear) and name not in skip_set and qualified not in skip_set:
                if wanted is not None and qualified not in wanted:
                    leftover.append(qualified)
                    continue
                setattr(
                    module,
                    name,
                    seat(
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
    if leftover:
        from gpu.graphs import refuse

        raise RuntimeError(refuse.unclassified(sorted(leftover)))
    if wanted is not None:
        unseen = sorted(wanted - set(replaced) - skip_set)
        if unseen:
            raise RuntimeError(
                "replace_linears: the plan named GEMM slots that are not "
                f"nn.Linear in this tree: {', '.join(unseen)}"
            )
    return replaced


def linear_modules(model: nn.Module) -> dict[str, CompressedLinear | CompressedVqLinear]:
    """Qualified name -> packed linear seat, in module order."""
    return {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, (CompressedLinear, CompressedVqLinear))
    }


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _embedding_name(model: nn.Module) -> str | None:
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Embedding, Nf4Embedding, VqEmbedding)):
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
    max_resident_bytes: int | None = None,
    residency_policy: str = DEFAULT_POLICY,
    pin_embed: bool = True,
    refill_embed: bool = True,
    compute_plan=None,
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

    ``max_resident_bytes``: ``None`` keeps every NF4 matrix on ``device`` (the
    3B/14B/20B path). An ``int`` runs ``residency_policy`` (default ``D``):
    allocate :class:`SlotPair` first, materialize resident NF4 on ``device``,
    streamed linears as :class:`~gpu.host.host_image.HostImage` on CPU. Embed
    stays DEVICE unless ``pin_embed=False`` or ``residency_policy="D_host_embed"``
    (CPU packed rows, not CopyRing tape). Tied embeddings refuse host-embed.

    ``compute_plan`` with ``n_cpu>0`` is the layer split (cpu-suffix / hybrid):
    suffix matrices use :meth:`CompressedLinear.attach_cpu` (pageable, no pin).
    ``cpu_codec=i4c`` transcodes those tables in RAM from NF4 (not written
    into the ``.chr``). When the GPU prefix does not fit as whole layers,
    ``ring=D`` streams prefix MLP through :class:`SlotPair` (pinned RAM, GPU
    compute). Join that tape before the hidden bounce; do not prefetch during
    the CPU suffix.

    The header is parsed once and passed down, so 300+ matrices do not reparse
    65 KiB of JSON each (gpu-abi.md §2).
    """
    t0 = time.perf_counter()
    path = str(path)
    hdr = header if header is not None else load_header(path)
    dev = torch.device(device)
    report = LoadReport(codec="nf4")
    layer_set = None if layers is None else sorted({int(v) for v in layers})
    host_names: set[str] = set()
    cpu_embed: set[str] = set()
    cpu_layer_ids: set[int] = set()
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    host_embed = (residency_policy == "D_host_embed") or (not pin_embed)
    if host_embed and tied:
        raise RuntimeError(
            "host-embed: tied embeddings share one packed table; refusing to "
            "put embed on CPU while lm_head stays DEVICE (3B)."
        )

    mixed_ring = False
    cpu_codec = "nf4"
    if compute_plan is not None:
        report.compute = compute_plan.compute
        report.n_gpu = int(compute_plan.n_gpu)
        report.n_cpu = int(compute_plan.n_cpu)
        cpu_codec = str(getattr(compute_plan, "cpu_codec", "nf4") or "nf4")
        report.cpu_codec = cpu_codec
        model.deepfold_compute = compute_plan
        if compute_plan.n_cpu > 0:
            cpu_layer_ids = set(compute_plan.cpu_layer_ids)
            if compute_plan.lm_head != "device" or compute_plan.embed != "device":
                raise RuntimeError("v1: embed and lm_head stay DEVICE")
            mixed_ring = str(compute_plan.ring) != "none"
            if not mixed_ring:
                max_resident_bytes = None

    if mixed_ring:
        from .compute import prefix_weight_descs

        descs = descs_from_header(hdr)
        prefix = prefix_weight_descs(descs, compute_plan.n_gpu)
        kv_tok = max(
            1,
            KV_BYTES_PER_TOKEN_32B
            * int(compute_plan.n_gpu)
            // max(1, int(compute_plan.n_layers)),
        )
        cap = overflow_resident_cap(
            int(compute_plan.vram_mib),
            int(compute_plan.max_seq),
            prefix,
            kv_bytes_per_token=kv_tok,
        )
        plan = plan_residency(
            prefix,
            cap,
            policy=residency_policy,
            pin_embed=not host_embed,
            refill_embed=refill_embed,
        )
        report.slots = SlotPair(plan.slot_nbytes, dev, count=OVERFLOW_SLOT_COUNT)
        report.slot_nbytes = plan.slot_nbytes
        report.streamed = len(plan.streamed)
        report.streamed_bytes = plan.streamed_bytes
        report.resident_bytes = plan.resident_bytes
        report.streamed_tape = plan.streamed
        report.overflow = bool(plan.streamed)
        model.deepfold_residency = plan
        host_names = set(plan.streamed)
        cpu_embed = set(plan.cpu)
        embed_tape = [d.name for d in prefix if d.kind == "embed" and d.name in host_names]
        if embed_tape:
            raise RuntimeError(
                f"embed must not be on CopyRing tape, plan streamed {embed_tape}"
            )
    elif max_resident_bytes is not None:
        descs = descs_from_header(hdr)
        plan = plan_residency(
            descs,
            max_resident_bytes,
            policy=residency_policy,
            pin_embed=not host_embed,
            refill_embed=refill_embed,
        )
        # Slots first: addresses must not move when resident matrices scatter.
        report.slots = SlotPair(plan.slot_nbytes, dev, count=OVERFLOW_SLOT_COUNT)
        report.slot_nbytes = plan.slot_nbytes
        report.streamed = len(plan.streamed)
        report.streamed_bytes = plan.streamed_bytes
        report.resident_bytes = plan.resident_bytes
        report.streamed_tape = plan.streamed
        report.overflow = bool(plan.streamed)
        model.deepfold_residency = plan
        host_names = set(plan.streamed)
        cpu_embed = set(plan.cpu)
        embed_tape = [d.name for d in descs if d.kind == "embed" and d.name in host_names]
        if embed_tape:
            raise RuntimeError(
                f"embed must not be on CopyRing tape, plan streamed {embed_tape}"
            )

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
    if host_embed and embed_name:
        cpu_embed.add(embed_name)
    need_embed_bytes = embed_info is not None and embed_info.codec == "nf4" and (
        embed != "skip" or (tied and any(n not in hdr.tensors for n in lm_head_names))
    )
    if need_embed_bytes:
        embed_dev = torch.device("cpu") if embed_name in cpu_embed else dev
        embed_matrix = materialize_nf4(path, embed_name, embed_dev, header=hdr)

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
        if info.layer is not None and int(info.layer) in cpu_layer_ids:
            if not isinstance(mod, CompressedLinear):
                raise TypeError(f"{name}: CPU suffix requires CompressedLinear")
            cpu_mat = materialize_nf4(path, name, torch.device("cpu"), header=hdr)
            mod.attach_cpu(cpu_mat, codec=cpu_codec)
            report.linears += 1
            report.cpu_linears += 1
            report.cpu_bytes += int(mod.nbytes)
            if verbose or cpu_codec == "i4c":
                tag = "i4c sidecar" if cpu_codec == "i4c" else "CPU"
                print(
                    f"  {name}: {tag} [{mod.M},{mod.K}] "
                    f"{mod.nbytes / MIB:.2f} MiB pageable",
                    flush=True,
                )
            continue
        if name in host_names:
            if not isinstance(mod, CompressedLinear):
                raise TypeError(f"{name}: HOST overflow requires CompressedLinear")
            cpu_mat = materialize_nf4(path, name, torch.device("cpu"), header=hdr)
            image = HostImage.from_blobs(cpu_mat.packed, cpu_mat.scale, cpu_mat.K)
            mod.attach_host(image)
            mod.chr_name = name
            report.linears += 1
            if verbose:
                print(f"  {name}: HOST [{cpu_mat.M},{cpu_mat.K}] {cpu_mat.nbytes / MIB:.2f} MiB")
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
        bf16_dev = (
            torch.device("cpu")
            if info.layer is not None and int(info.layer) in cpu_layer_ids
            else dev
        )
        tensor = load_bf16(path, name, bf16_dev, header=hdr)
        if isinstance(target, (CompressedLinear, CompressedVqLinear)) and attr == "bias":
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
    model.deepfold_slots = report.slots
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

    if embed_info.codec == "vq":
        if embed == "rows":
            module = VqEmbedding(
                embed_matrix.M,
                embed_matrix.K,
                padding_idx=getattr(current, "padding_idx", None),
            )
            module.attach(embed_matrix)
            setattr(parent, child, module)
            report.embed_bytes += embed_matrix.nbytes
            return "vq-rows"
        if embed == "bf16":
            table = reconstruct_vq(
                embed_matrix.index,
                embed_matrix.book,
                embed_matrix.M,
                embed_matrix.K,
                embed_matrix.K_pad,
            ).to(torch.bfloat16)
            _assign_weight(current, "weight", table)
            report.embed_bytes += table.numel() * table.element_size()
            return "vq-dequant-table"
        raise ValueError(f"embed={embed!r}; expected 'rows', 'bf16' or 'skip'")

    if embed == "rows":
        module = Nf4Embedding(
            embed_matrix.M,
            embed_matrix.K,
            padding_idx=getattr(current, "padding_idx", None),
        )
        module.attach(embed_matrix)
        setattr(parent, child, module)
        report.embed_bytes += embed_matrix.nbytes
        if embed_matrix.packed.device.type == "cpu":
            return "nf4-rows-cpu"
        return "nf4-rows"

    if embed == "bf16":
        table = dequant_table(embed_matrix)
        _assign_weight(current, "weight", table)
        report.embed_bytes += table.numel() * table.element_size()
        return "nf4-dequant-table"

    raise ValueError(f"embed={embed!r}; expected 'rows', 'bf16' or 'skip'")


def load_chr_vq_model(
    model: nn.Module,
    path: str,
    *,
    device: str | torch.device = "cuda",
    header: Header | None = None,
    embed: str = "rows",
    layers: Sequence[int] | None = None,
    verbose: bool = False,
) -> LoadReport:
    """Fill a VQ-replaced skeleton from ``path``. Twin of :func:`load_chr_nf4`."""
    t0 = time.perf_counter()
    path = str(path)
    hdr = header if header is not None else load_header(path)
    dev = torch.device(device)
    report = LoadReport(codec="vq")
    layer_set = None if layers is None else sorted({int(v) for v in layers})

    modules = dict(model.named_modules())
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))

    embed_name = _embedding_name(model)
    embed_info = hdr.tensors.get(embed_name) if embed_name else None
    embed_matrix = None
    lm_head_names = [
        n
        for n in linear_modules(model)
        if n.rsplit(".", 1)[-1] in ("lm_head", "output")
    ]
    need_embed_bytes = embed_info is not None and embed_info.codec == "vq" and (
        embed != "skip" or (tied and any(n not in hdr.tensors for n in lm_head_names))
    )
    if need_embed_bytes:
        embed_matrix = materialize_vq(path, embed_name, dev, header=hdr)

    for name, mod in linear_modules(model).items():
        info = hdr.tensors.get(name)
        if info is None:
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
        if info.codec != "vq" or info.kind not in LINEAR_KINDS:
            report.skipped.append(f"{name} (kind={info.kind}, codec={info.codec})")
            continue
        if not _in_scope(info.layer, layer_set):
            report.skipped.append(f"{name} (layer {info.layer} out of scope)")
            continue
        matrix = materialize_vq(path, name, dev, header=hdr)
        mod.attach(matrix)
        report.linears += 1
        report.linear_bytes += matrix.nbytes
        if verbose:
            print(f"  {name}: [{matrix.M},{matrix.K}] {matrix.nbytes / MIB:.2f} MiB")

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
        if isinstance(target, (CompressedLinear, CompressedVqLinear)) and attr == "bias":
            target.set_bias(tensor)
        else:
            _assign_weight(target, attr, tensor)
        report.bf16_tensors += 1
        report.bf16_bytes += tensor.numel() * tensor.element_size()

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
    strict: bool = True,
    max_resident_bytes: int | None = None,
    residency_policy: str = DEFAULT_POLICY,
    pin_embed: bool = True,
    refill_embed: bool = True,
    compute_plan=None,
):
    """``build_skeleton`` + ``attach`` + packed seats + load from the ``.chr``.

    The file's quantized codec picks the seat: ``nf4`` → ``CompressedLinear``,
    ``vq`` → ``CompressedVqLinear``. ``attach`` still runs on the meta skeleton
    before any packed byte reaches the device.

    ``strict`` (the product default) turns a partial load into a refusal: a plan
    slot that is missing from the ``.chr``, was skipped as ``kind=other``, or
    stayed on ``meta`` means wrong answers, not slow ones. The lab passes
    ``strict=False`` so it can *print* an incomplete report instead.

    ``max_resident_bytes`` is NF4 overflow (default policy D). ``None`` keeps
    every matrix on ``device``. Ignored for VQ. ``residency_policy`` selects
    WHO (default ``D``); ``pin_embed=False`` / ``D_host_embed`` puts packed
    embed on CPU, not the CopyRing tape. ``compute_plan`` with ``n_cpu>0``
    loads the suffix with ``attach_cpu``. A GPU prefix that does not fit as
    whole layers uses policy D CopyRing on the prefix only.
    """
    from .attach import attach_module, plan_violations

    hdr = load_header(chr_path)
    codec = quantized_codec(hdr)
    if codec not in ("nf4", "vq"):
        raise RuntimeError(
            f"{chr_path}: quantized codec {codec!r}; load_model drives nf4 or vq"
        )
    if codec == "vq" and compute_plan is not None and getattr(compute_plan, "n_cpu", 0):
        raise RuntimeError("cpu-suffix/hybrid is NF4 only; --codec vq refuses a CPU suffix")
    seat = CompressedVqLinear if codec == "vq" else CompressedLinear
    filler = load_chr_vq_model if codec == "vq" else load_chr_nf4

    model = build_skeleton(model_id, trust_remote_code=trust_remote_code)
    plan = attach_module(model, trust_remote_code=trust_remote_code)
    replace_linears(model, skip=skip, slots=plan.gemm_names, seat=seat)
    fill_kw = dict(
        device=device, embed=embed, layers=layers, verbose=verbose, header=hdr
    )
    if codec == "nf4":
        fill_kw["max_resident_bytes"] = max_resident_bytes
        fill_kw["residency_policy"] = residency_policy
        fill_kw["pin_embed"] = pin_embed
        fill_kw["refill_embed"] = refill_embed
        fill_kw["compute_plan"] = compute_plan
    report = filler(model, chr_path, **fill_kw)
    model.deepfold_plan = plan
    if strict and layers is None:
        violations = plan_violations(plan, report)
        if violations:
            from gpu.graphs import refuse

            raise RuntimeError(refuse.load_violations(violations))
    return model, report
