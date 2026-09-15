"""Who stays in HBM (DEVICE) vs who streams from pinned host (HOST).

Policy D (docs/plan-h2-ring.md): not a resident prefix of whole blocks (A),
not LRU. Embed, lm_head and every attention matrix (q/k/v/o/qkv) are pinned.
Overflow is every ``down``, then tail ``gate+up`` pairs of the same layer.

Named extras (``plan_residency(..., policy=)``) are CPU comparison unless
load passes ``residency_policy``. Load / CLI keep default ``D`` (embed DEVICE).
``pairs_stride`` and ``D_host_embed`` are Accel-3 loadable policies. Tape order
is TokenLoop consume order of HOST names, not eviction order. Embed on CPU
(host-embed) is not a tape slot.

``cap_bytes`` is explicit (``max_resident_bytes``). Do not pass codec leftover:
that budget has no slots. The optional helper subtracts runtime overhead, KV
and two slot arenas — never the 978 MiB 4.5-bit table error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_POLICY",
    "GROUP_SIZE",
    "HOST_EMBED_POLICY",
    "KV_BYTES_PER_TOKEN_32B",
    "MIB",
    "PIN_KINDS",
    "POLICIES",
    "RUNTIME_OVERHEAD_MIB",
    "CapTooSmallError",
    "ResidencyPlan",
    "WeightDesc",
    "descs_from_header",
    "descs_from_qwen",
    "embeddings_tied",
    "nf4_nbytes",
    "pin_kinds",
    "pin_nbytes",
    "mlp_slot_nbytes",
    "overflow_cap_from_chr",
    "overflow_resident_cap",
    "plan_residency",
    "resident_cap_bytes",
    "stride_pair_layer_ids",
    "stride_pair_layers",
    "summarize_residency",
]

GROUP_SIZE = 64  # linear.py / ChrMatrix; 64 * ceil(K / 64)
MIB = 1024 * 1024
# Same number as gpu.cli.codec.RUNTIME_OVERHEAD_MIB. Host must not import CLI.
RUNTIME_OVERHEAD_MIB = 1800
# docs/vram-3080.md Qwen2.5-32B FP16 KV: 256 KiB / token.
KV_BYTES_PER_TOKEN_32B = 256 * 1024

PIN_KINDS = frozenset({"embed", "lm_head", "q", "k", "v", "o", "qkv"})
DEFAULT_POLICY = "D"
HOST_EMBED_POLICY = "D_host_embed"
# D / downs_only: downs then pairs if still over. interleaved_down: head-first
# downs (WHO differs only when some downs stay DEVICE). pairs_first: negative
# control. all_mlp: every gate+up+down HOST (more bytes, not a candidate).
# pairs_stride: D downs, then gate+up on layers 3,7,11,... (every 4th).
# D_host_embed: D WHO then embed on CPU (not tape) and refill 2 pairs + 1 down.
POLICIES = frozenset(
    {
        "D",
        "downs_only",
        "pairs_first",
        "interleaved_down",
        "all_mlp",
        "pairs_stride",
        HOST_EMBED_POLICY,
    }
)
# After unpinning embed, put this much MLP back on DEVICE (whole pairs).
_REFILL_PAIRS = 2
_REFILL_DOWNS = 1
_EVICT_DOWN = "down"
_PAIR_KINDS = ("gate", "up")
_ALL_KINDS = PIN_KINDS | frozenset({"gate", "up", "down"})
# TokenLoop consume order inside a layer. Only HOST names go on the tape.
_CONSUME_KINDS = ("q", "k", "v", "qkv", "o", "gate", "up", "down")

_QWEN_ATTN = "model.layers.{i}.self_attn"
_QWEN_MLP = "model.layers.{i}.mlp"


class CapTooSmallError(ValueError):
    """``max_resident_bytes`` is below the pin set; qkvo/embed/lm_head stay DEVICE."""


@dataclass(frozen=True)
class WeightDesc:
    """One NF4 matrix the planner knows about. Norms and bias are not listed."""

    name: str
    kind: str
    layer: int | None
    M: int
    K: int
    nbytes: int

    @classmethod
    def nf4(
        cls,
        name: str,
        kind: str,
        layer: int | None,
        m: int,
        k: int,
    ) -> WeightDesc:
        return cls(name, kind, layer, int(m), int(k), nf4_nbytes(m, k))


@dataclass(frozen=True)
class ResidencyPlan:
    """DEVICE vs HOST names after a residency policy. ``streamed`` is the tape.

    ``cpu`` is packed-on-CPU, not CopyRing (host-embed). Empty for policy D.
    """

    resident: frozenset[str]
    streamed: tuple[str, ...]
    slot_nbytes: int
    resident_bytes: int
    streamed_bytes: int
    cpu: frozenset[str] = frozenset()


def k_pad(k: int) -> int:
    """``64 * ceil(K / 64)`` — same as ``gpu.host.linear.k_pad`` / ChrMatrix."""
    return GROUP_SIZE * ((int(k) + GROUP_SIZE - 1) // GROUP_SIZE)


def nf4_nbytes(m: int, k: int) -> int:
    """packed + scale bytes. Matches ``ChrMatrix.nbytes``.

    ``K_pad = 64 * ceil(K / 64)``, packed = ``M * K_pad / 2``,
    scale = ``M * (K_pad / 64) * 2``.
    """
    m = int(m)
    k_pad_ = k_pad(k)
    packed = m * k_pad_ // 2
    scale = m * (k_pad_ // GROUP_SIZE) * 2
    return packed + scale


def pin_kinds(*, pin_embed: bool = True) -> frozenset[str]:
    """DEVICE pin set. Host-embed drops ``embed`` (CPU, not tape)."""
    return PIN_KINDS if pin_embed else (PIN_KINDS - {"embed"})


def pin_nbytes(descs: Sequence[WeightDesc], *, pin_embed: bool = True) -> int:
    """Bytes that stay DEVICE: lm_head + qkvo, plus embed unless host-embed."""
    kinds = pin_kinds(pin_embed=pin_embed)
    return sum(int(d.nbytes) for d in descs if d.kind in kinds)


def embeddings_tied(descs: Sequence[WeightDesc]) -> bool:
    """No separate ``lm_head`` tensor: packed embed is shared (Qwen2.5-3B)."""
    kinds = {d.kind for d in descs}
    return "embed" in kinds and "lm_head" not in kinds


def stride_pair_layer_ids(descs: Sequence[WeightDesc]) -> list[int]:
    """Every 4th layer ending at 3,7,11,...,63 (``layer % 4 == 3``)."""
    layers = sorted(
        {int(d.layer) for d in descs if d.kind in _PAIR_KINDS and d.layer is not None}
    )
    return [i for i in layers if i % 4 == 3]


stride_pair_layers = stride_pair_layer_ids


def mlp_slot_nbytes(descs: Sequence[WeightDesc]) -> int:
    """Worst packed+scale among ``gate``/``up``/``down`` — H2 slot before a plan."""
    return max((int(d.nbytes) for d in descs if d.kind in _PAIR_KINDS or d.kind == _EVICT_DOWN), default=0)


def overflow_cap_from_chr(
    path: str,
    vram_mib: int,
    max_seq: int,
    *,
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_32B,
) -> int:
    """``overflow_resident_cap`` from a packed ``.chr`` header (slot = max MLP)."""
    from gpu.chr0 import load_header

    return overflow_resident_cap(
        int(vram_mib),
        int(max_seq),
        descs_from_header(load_header(str(path))),
        kv_bytes_per_token=kv_bytes_per_token,
    )


def overflow_resident_cap(
    vram_mib: int,
    max_seq: int,
    descs: Sequence[WeightDesc],
    *,
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_32B,
) -> int:
    """``resident_cap_bytes`` using the MLP slot from ``descs``.

    CLI auto-overflow: cap is below full packed NF4 and above the pin set
    on 32B/12 GB. Does not run a trial plan.
    """
    return resident_cap_bytes(
        vram_mib,
        max_seq,
        mlp_slot_nbytes(descs),
        kv_bytes_per_token=kv_bytes_per_token,
    )


def resident_cap_bytes(
    vram_mib: int,
    max_seq: int,
    slot_bytes: int,
    *,
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_32B,
) -> int:
    """HBM left for resident NF4 after overhead, KV and two slot arenas.

    ``(vram - 1800) * MiB - kv - 2 * slot``. Does not subtract the 978 MiB
    4.5-bit table error. Codec leftover is the wrong cap (no slots).
    """
    return (
        (int(vram_mib) - RUNTIME_OVERHEAD_MIB) * MIB
        - int(max_seq) * int(kv_bytes_per_token)
        - 2 * int(slot_bytes)
    )


def descs_from_header(hdr) -> tuple[WeightDesc, ...]:
    """NF4 descriptors from a parsed CHR0 header.

    Names are the file's tensor keys. ``kind`` / ``layer`` come from
    ``TensorInfo``. ``nbytes`` is ``nf4_nbytes(M, K)`` and must match the
    ``data``+``scale`` blob sizes.
    """
    out: list[WeightDesc] = []
    for name, info in hdr.tensors.items():
        if info.codec != "nf4" or info.kind not in _ALL_KINDS:
            continue
        desc = WeightDesc.nf4(name, info.kind, info.layer, info.M, info.K)
        blob_n = int(info.blobs["data"].nbytes) + int(info.blobs["scale"].nbytes)
        if blob_n != desc.nbytes:
            raise ValueError(
                f"{name}: nf4 blobs {blob_n} bytes != nf4_nbytes({desc.M},{desc.K})"
                f"={desc.nbytes}"
            )
        out.append(desc)
    return tuple(out)


def descs_from_qwen(cfg: Mapping[str, object]) -> tuple[WeightDesc, ...]:
    """NF4 descriptors from HuggingFace Qwen2/Qwen2.5 ``config.json`` shapes.

    Split attention: per layer q/k/v/o + gate/up/down, plus embed, plus
    ``lm_head`` when not tied. InternLM fused ``qkv`` is not emitted (v1);
    kind ``qkv`` is still pinned if a caller builds that desc by hand.
    """
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    vocab = int(cfg["vocab_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads") or n_q)
    head_dim = int(cfg["head_dim"]) if cfg.get("head_dim") else hidden // n_q
    tied = bool(cfg.get("tie_word_embeddings", False))

    q_m = n_q * head_dim
    kv_m = n_kv * head_dim
    out: list[WeightDesc] = [
        WeightDesc.nf4("model.embed_tokens", "embed", None, vocab, hidden),
    ]
    for i in range(n_layers):
        attn = _QWEN_ATTN.format(i=i)
        mlp = _QWEN_MLP.format(i=i)
        out.extend(
            (
                WeightDesc.nf4(f"{attn}.q_proj", "q", i, q_m, hidden),
                WeightDesc.nf4(f"{attn}.k_proj", "k", i, kv_m, hidden),
                WeightDesc.nf4(f"{attn}.v_proj", "v", i, kv_m, hidden),
                WeightDesc.nf4(f"{attn}.o_proj", "o", i, hidden, q_m),
                WeightDesc.nf4(f"{mlp}.gate_proj", "gate", i, intermediate, hidden),
                WeightDesc.nf4(f"{mlp}.up_proj", "up", i, intermediate, hidden),
                WeightDesc.nf4(f"{mlp}.down_proj", "down", i, hidden, intermediate),
            )
        )
    if not tied:
        out.append(WeightDesc.nf4("lm_head", "lm_head", None, vocab, hidden))
    return tuple(out)


def _tail_first(items: Iterable[WeightDesc]) -> list[WeightDesc]:
    """High layer index first. ``layer is None`` sorts after numbered layers."""
    return sorted(
        items,
        key=lambda d: d.layer if d.layer is not None else -1,
        reverse=True,
    )


def _head_first(items: Iterable[WeightDesc]) -> list[WeightDesc]:
    """Low layer index first. ``layer is None`` sorts after numbered layers."""
    return sorted(
        items,
        key=lambda d: d.layer if d.layer is not None else 1 << 30,
    )


def _layer_ids(pair_of: Mapping[int | None, object], *, tail_first: bool) -> list[int | None]:
    """Numbered layers in eviction order; ``None`` last either direction."""

    def key(layer: int | None) -> int:
        if layer is None:
            return -1 if tail_first else 1 << 30
        return int(layer)

    return sorted(pair_of, key=key, reverse=tail_first)


def _evict_downs(
    descs: Sequence[WeightDesc],
    host: set[str],
    resident_bytes: int,
    cap: int,
    *,
    tail_first: bool,
    stop_at_cap: bool = True,
) -> int:
    order = _tail_first if tail_first else _head_first
    for d in order(d for d in descs if d.kind == _EVICT_DOWN):
        if stop_at_cap and resident_bytes <= cap:
            break
        if d.name in host:
            continue
        host.add(d.name)
        resident_bytes -= int(d.nbytes)
    return resident_bytes


def _evict_pairs(
    descs: Sequence[WeightDesc],
    host: set[str],
    resident_bytes: int,
    cap: int,
    *,
    tail_first: bool,
    stop_at_cap: bool = True,
) -> int:
    """Whole ``gate+up`` of one layer, never one side only."""
    pair_of: dict[int | None, list[WeightDesc]] = {}
    for d in descs:
        if d.kind in _PAIR_KINDS:
            pair_of.setdefault(d.layer, []).append(d)
    for layer in _layer_ids(pair_of, tail_first=tail_first):
        if stop_at_cap and resident_bytes <= cap:
            break
        for d in pair_of[layer]:
            if d.name in host:
                continue
            host.add(d.name)
            resident_bytes -= int(d.nbytes)
    return resident_bytes


def _pair_of(descs: Sequence[WeightDesc]) -> dict[int | None, list[WeightDesc]]:
    pair_of: dict[int | None, list[WeightDesc]] = {}
    for d in descs:
        if d.kind in _PAIR_KINDS:
            pair_of.setdefault(d.layer, []).append(d)
    return pair_of


def _evict_pairs_stride(
    descs: Sequence[WeightDesc],
    host: set[str],
    resident_bytes: int,
    cap: int,
    *,
    stop_at_cap: bool = True,
) -> int:
    """Whole ``gate+up`` on stride layers 3,7,... then leftover tail-first."""
    pair_of = _pair_of(descs)
    stride = [L for L in stride_pair_layer_ids(descs) if L in pair_of]
    stride_set = set(stride)
    leftover = [L for L in _layer_ids(pair_of, tail_first=True) if L not in stride_set]
    for layer in stride + leftover:
        if stop_at_cap and resident_bytes <= cap:
            break
        for d in pair_of[layer]:
            if d.name in host:
                continue
            host.add(d.name)
            resident_bytes -= int(d.nbytes)
    return resident_bytes


def _resident_bytes(
    descs: Sequence[WeightDesc], host: set[str], cpu: set[str]
) -> int:
    return sum(
        int(d.nbytes) for d in descs if d.name not in host and d.name not in cpu
    )


def _refill_mlp(
    descs: Sequence[WeightDesc],
    host: set[str],
    cpu: set[str],
    cap: int,
    *,
    n_pairs: int = _REFILL_PAIRS,
    n_downs: int = _REFILL_DOWNS,
) -> None:
    """Move whole HOST pairs, then downs, back to DEVICE while they fit in cap."""
    pair_of = _pair_of(descs)
    host_pair_layers = sorted(
        L
        for L, sides in pair_of.items()
        if L is not None and sides and all(d.name in host for d in sides)
    )
    restored = 0
    for layer in host_pair_layers:
        if restored >= n_pairs:
            break
        sides = pair_of[layer]
        cost = sum(int(d.nbytes) for d in sides)
        if _resident_bytes(descs, host, cpu) + cost > cap:
            continue
        for d in sides:
            host.discard(d.name)
        restored += 1
    downs = sorted(
        (d for d in descs if d.kind == _EVICT_DOWN and d.name in host),
        key=lambda d: d.layer if d.layer is not None else -1,
    )
    restored_d = 0
    for d in downs:
        if restored_d >= n_downs:
            break
        if _resident_bytes(descs, host, cpu) + int(d.nbytes) > cap:
            continue
        host.discard(d.name)
        restored_d += 1


def _embed_to_cpu(descs: Sequence[WeightDesc], host: set[str], cpu: set[str]) -> None:
    for d in descs:
        if d.kind != "embed":
            continue
        host.discard(d.name)
        cpu.add(d.name)


def _streamed_tape(descs: Sequence[WeightDesc], host: set[str]) -> tuple[str, ...]:
    """L0..L{n-1} ``q,k,v,o,gate,up,down`` (HOST only), then lm_head if HOST."""
    by_layer: dict[int, dict[str, str]] = {}
    embed_host: list[str] = []
    lm_host: list[str] = []
    leftover: list[str] = []
    for d in descs:
        if d.name not in host:
            continue
        if d.kind == "embed":
            embed_host.append(d.name)
        elif d.kind == "lm_head":
            lm_host.append(d.name)
        elif d.layer is not None and d.kind in _CONSUME_KINDS:
            by_layer.setdefault(d.layer, {})[d.kind] = d.name
        else:
            leftover.append(d.name)

    tape: list[str] = list(embed_host)
    for layer in sorted(by_layer):
        slots = by_layer[layer]
        for kind in _CONSUME_KINDS:
            name = slots.get(kind)
            if name is not None:
                tape.append(name)
    tape.extend(lm_host)
    seen = set(tape)
    tape.extend(n for n in leftover if n not in seen)
    return tuple(tape)


def plan_residency(
    descs: Sequence[WeightDesc],
    max_resident_bytes: int,
    *,
    policy: str = DEFAULT_POLICY,
    pin_embed: bool = True,
    refill_embed: bool = True,
) -> ResidencyPlan:
    """Pin qkvo/lm_head (and embed unless host-embed); evict MLP under ``policy``.

    ``D`` / ``downs_only``: tail-first ``down``, then tail ``gate+up`` pairs
    only if still over cap. ``interleaved_down``: head-first downs, then tail
    pairs. ``pairs_first``: tail pairs, then tail-first downs. ``all_mlp``:
    every down and every pair (more HOST bytes; comparison only).
    ``pairs_stride``: D downs, then pairs on layers 3,7,11,... until cap.
    ``D_host_embed`` or ``pin_embed=False``: embed is CPU, not DEVICE and not
    CopyRing tape; refill restores 2 whole gate+up pairs and 1 down.
    Tied embeddings (no ``lm_head`` desc) refuse host-embed.
    """
    if not descs:
        return ResidencyPlan(frozenset(), (), 0, 0, 0)

    names = [d.name for d in descs]
    if len(names) != len(set(names)):
        raise ValueError("plan_residency: duplicate WeightDesc.name")
    unknown = sorted({d.kind for d in descs} - _ALL_KINDS)
    if unknown:
        raise ValueError(f"plan_residency: unknown kind(s) {unknown}")
    if policy not in POLICIES:
        raise ValueError(
            f"plan_residency: unknown policy {policy!r}; expected one of "
            f"{sorted(POLICIES)}"
        )

    host_embed = (policy == HOST_EMBED_POLICY) or (not pin_embed)
    evict_policy = "D" if policy == HOST_EMBED_POLICY else policy
    if host_embed and embeddings_tied(descs):
        raise ValueError(
            "host-embed: tied embeddings share one packed table; refusing to "
            "put embed on CPU while lm_head stays DEVICE (3B)."
        )

    cap = int(max_resident_bytes)
    pin = pin_nbytes(descs, pin_embed=not host_embed)
    pin_set = pin_kinds(pin_embed=not host_embed)
    if cap < pin:
        raise CapTooSmallError(
            f"max_resident_bytes={cap} is below the pin set ({pin} bytes: "
            + ("lm_head + qkvo" if host_embed else "embed + lm_head + qkvo")
            + "). Attention stays DEVICE; raise the cap, do not break that invariant."
        )

    host: set[str] = set()
    resident_bytes = sum(int(d.nbytes) for d in descs)
    evict_all = evict_policy == "all_mlp"
    downs_tail = evict_policy != "interleaved_down"

    if evict_policy == "pairs_first":
        resident_bytes = _evict_pairs(
            descs, host, resident_bytes, cap, tail_first=True
        )
        resident_bytes = _evict_downs(
            descs, host, resident_bytes, cap, tail_first=True
        )
    elif evict_policy == "pairs_stride":
        # Downs like D, then every-4th gate+up until cap (leftover tail-first).
        resident_bytes = _evict_downs(
            descs,
            host,
            resident_bytes,
            cap,
            tail_first=True,
            stop_at_cap=True,
        )
        resident_bytes = _evict_pairs_stride(
            descs, host, resident_bytes, cap, stop_at_cap=True
        )
    else:
        # D, downs_only, interleaved_down, all_mlp, D_host_embed: downs before pairs.
        # downs_only == D: never touch pairs unless every down still leaves
        # resident > cap (on 32B overflow that is required, so WHO matches D).
        resident_bytes = _evict_downs(
            descs,
            host,
            resident_bytes,
            cap,
            tail_first=downs_tail,
            stop_at_cap=not evict_all,
        )
        resident_bytes = _evict_pairs(
            descs,
            host,
            resident_bytes,
            cap,
            tail_first=True,
            stop_at_cap=not evict_all,
        )

    cpu: set[str] = set()
    if host_embed:
        _embed_to_cpu(descs, host, cpu)
        if refill_embed:
            _refill_mlp(descs, host, cpu, cap)

    if any(d.kind in pin_set and d.name in host for d in descs):
        raise CapTooSmallError(
            f"max_resident_bytes={cap} would stream a pinned matrix "
            "(embed / lm_head / qkvo); raise the cap."
        )
    resident_bytes = _resident_bytes(descs, host, cpu)
    if resident_bytes > cap:
        raise CapTooSmallError(
            f"max_resident_bytes={cap} still short after evicting every down "
            f"and gate+up pair (resident {resident_bytes} bytes, pin {pin})."
        )

    streamed = _streamed_tape(descs, host)
    resident = frozenset(
        d.name for d in descs if d.name not in host and d.name not in cpu
    )
    streamed_bytes = sum(int(d.nbytes) for d in descs if d.name in host)
    resident_bytes = sum(int(d.nbytes) for d in descs if d.name in resident)
    slot_nbytes = max((int(d.nbytes) for d in descs if d.name in host), default=0)
    return ResidencyPlan(
        resident=resident,
        streamed=streamed,
        slot_nbytes=slot_nbytes,
        resident_bytes=resident_bytes,
        streamed_bytes=streamed_bytes,
        cpu=frozenset(cpu),
    )


def summarize_residency(descs: Sequence[WeightDesc], plan: ResidencyPlan) -> dict:
    """Kind counts and bytes on DEVICE vs HOST tape vs CPU. Tape is ``plan.streamed``."""
    cpu = plan.cpu
    kinds: dict[str, dict[str, dict[str, int]]] = {}
    for d in descs:
        if d.name in plan.resident:
            home = "device"
        elif d.name in cpu:
            home = "cpu"
        else:
            home = "host"
        slot = kinds.setdefault(
            d.kind,
            {
                "device": {"n": 0, "bytes": 0},
                "host": {"n": 0, "bytes": 0},
                "cpu": {"n": 0, "bytes": 0},
            },
        )
        slot[home]["n"] += 1
        slot[home]["bytes"] += int(d.nbytes)
    host_layers: dict[str, list[int]] = {"down": [], "gate": [], "up": []}
    streamed_set = set(plan.streamed)
    for d in descs:
        if d.name not in streamed_set or d.kind not in host_layers or d.layer is None:
            continue
        host_layers[d.kind].append(int(d.layer))
    for kind in host_layers:
        host_layers[kind] = sorted(host_layers[kind])
    return {
        "n_resident": len(plan.resident),
        "n_streamed": len(plan.streamed),
        "n_cpu": len(cpu),
        "resident_bytes": int(plan.resident_bytes),
        "streamed_bytes": int(plan.streamed_bytes),
        "slot_nbytes": int(plan.slot_nbytes),
        "kinds": kinds,
        "host_layers": host_layers,
        "streamed": list(plan.streamed),
        "cpu": sorted(cpu),
    }
