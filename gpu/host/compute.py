"""Where repeating layers compute: GPU prefix vs CPU suffix.

``ResidencyPlan.cpu`` is host-embed (packed embed on CPU, not a transformer
suffix). This module is the layer split. No ``.chr`` and no CUDA for tests;
shapes come from ``descs_from_qwen`` / ``descs_from_header``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .residency import (
    DEFAULT_POLICY,
    KV_BYTES_PER_TOKEN_32B,
    MIB,
    RUNTIME_OVERHEAD_MIB,
    WeightDesc,
    overflow_resident_cap,
    plan_residency,
)

__all__ = [
    "COMPUTE_MODES",
    "DEFAULT_CPU_SUFFIX_GPU_LAYERS",
    "DEFAULT_HYBRID_GPU_LAYERS",
    "SLACK_MIB",
    "VRAM_MIB_3080",
    "ComputePlan",
    "ComputePlanError",
    "format_compute_stderr",
    "kv_mib_per_layer",
    "net_fits_resident",
    "plan_compute",
    "prefix_budget_mib",
    "prefix_weight_descs",
]

COMPUTE_MODES = frozenset({"gpu", "cpu-suffix", "hybrid"})
CPU_CODECS = frozenset({"nf4", "i4c"})
DEFAULT_HYBRID_GPU_LAYERS = 36
DEFAULT_CPU_SUFFIX_GPU_LAYERS = 32
# Auto-fit / unsafe-N pad. On 12 GB / 2048 / split this keeps N=38 off the
# first live try (docs/plan-cpu-hybrid.md §2.1–2.2).
SLACK_MIB = 256
VRAM_MIB_3080 = 12288


class ComputePlanError(ValueError):
    """User-facing placement error (CLI flags, 3B refuse, unsafe N)."""


@dataclass(frozen=True)
class ComputePlan:
    """Layer split for one generate session. Suffix is contiguous and tail-only."""

    compute: str
    n_gpu: int
    n_cpu: int
    n_layers: int
    lm_head: str = "device"
    embed: str = "device"
    kv: str = "gpu-all"
    ring: str = "none"
    max_seq: int = 2048
    vram_mib: int = VRAM_MIB_3080
    resident_mib: float = 0.0
    cpu_weight_mib: float = 0.0
    kv_gpu_mib: float = 0.0
    kv_cpu_mib: float = 0.0
    used_mib: float = 0.0
    used_plus_overhead_mib: float = 0.0
    ring_matrices: int = 0
    ring_streamed_mib: float = 0.0
    layer_mib: float = 0.0
    cpu_codec: str = "nf4"

    @property
    def gpu_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.n_gpu))

    @property
    def cpu_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.n_gpu, self.n_layers))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpu_layer_ids"] = list(self.gpu_layer_ids)
        payload["cpu_layer_ids"] = list(self.cpu_layer_ids)
        return payload


def kv_mib_per_layer(cfg: Mapping[str, object], max_seq: int) -> float:
    """K+V bf16 for one repeating layer at ``max_seq`` (8 MiB on 32B @2048)."""
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads") or n_q)
    hidden = int(cfg["hidden_size"])
    head_dim = int(cfg["head_dim"]) if cfg.get("head_dim") else hidden // n_q
    nbytes = 2 * int(max_seq) * n_kv * head_dim * 2
    return nbytes / MIB


def _embed_lm_nbytes(descs: Sequence[WeightDesc]) -> int:
    return sum(int(d.nbytes) for d in descs if d.kind in ("embed", "lm_head"))


def _layer_nbytes(descs: Sequence[WeightDesc], layer: int) -> int:
    return sum(int(d.nbytes) for d in descs if d.layer == layer)


def _typical_layer_nbytes(descs: Sequence[WeightDesc]) -> int:
    layers = sorted({int(d.layer) for d in descs if d.layer is not None})
    if not layers:
        return 0
    return _layer_nbytes(descs, layers[0])


def net_fits_resident(
    descs: Sequence[WeightDesc],
    cfg: Mapping[str, object],
    *,
    vram_mib: int = VRAM_MIB_3080,
    max_seq: int = 2048,
) -> bool:
    """True when packed NF4 + KV + overhead sits on the card (3B/14B/20B)."""
    packed = sum(int(d.nbytes) for d in descs) / MIB
    n_layers = int(cfg["num_hidden_layers"])
    kv = kv_mib_per_layer(cfg, max_seq) * n_layers
    return packed + RUNTIME_OVERHEAD_MIB + kv <= int(vram_mib)


def prefix_weight_descs(descs: Sequence[WeightDesc], n_gpu: int) -> tuple[WeightDesc, ...]:
    """Embed, lm_head, and repeating layers ``[0, n_gpu)``. Suffix stays off the tape."""
    n_gpu = int(n_gpu)
    out: list[WeightDesc] = []
    for desc in descs:
        if desc.kind in ("embed", "lm_head"):
            out.append(desc)
        elif desc.layer is not None and int(desc.layer) < n_gpu:
            out.append(desc)
    return tuple(out)


def prefix_budget_mib(
    descs: Sequence[WeightDesc],
    cfg: Mapping[str, object],
    n_gpu: int,
    *,
    max_seq: int = 2048,
    kv: str = "split",
) -> dict[str, float]:
    """Whole-layer GPU prefix, no CopyRing slots (docs/plan-cpu-hybrid.md §2.1)."""
    n_layers = int(cfg["num_hidden_layers"])
    n_gpu = int(n_gpu)
    n_cpu = n_layers - n_gpu
    layer_n = _typical_layer_nbytes(descs)
    embed_lm = _embed_lm_nbytes(descs)
    weights = embed_lm + n_gpu * layer_n
    per = kv_mib_per_layer(cfg, max_seq)
    kv_all = per * n_layers
    if kv == "split":
        kv_gpu = per * n_gpu
        kv_cpu = per * n_cpu
    else:
        kv_gpu = kv_all
        kv_cpu = 0.0
    used = weights / MIB + kv_gpu
    return {
        "layer_mib": layer_n / MIB,
        "resident_mib": weights / MIB,
        "cpu_weight_mib": (n_cpu * layer_n) / MIB,
        "kv_gpu_mib": kv_gpu,
        "kv_cpu_mib": kv_cpu,
        "used_mib": used,
        "used_plus_overhead_mib": used + RUNTIME_OVERHEAD_MIB,
    }


def _resolve_n_gpu(
    n_layers: int,
    compute: str,
    *,
    gpu_layers: int | None,
    cpu_layers: int | None,
    gpu_frac: float | None,
    fits: bool,
) -> int:
    has_gpu = gpu_layers is not None
    has_cpu = cpu_layers is not None
    has_frac = gpu_frac is not None
    n_spec = int(has_gpu) + int(has_cpu) + int(has_frac)

    if compute == "gpu":
        if n_spec == 0:
            return n_layers
        if has_gpu and not has_cpu and not has_frac and int(gpu_layers) == n_layers:
            return n_layers
        raise ComputePlanError(
            "--gpu-layers / --cpu-layers / --gpu-frac require --compute "
            "cpu-suffix|hybrid (with --compute gpu, --gpu-layers equal to "
            "num_hidden_layers is a no-op)."
        )

    if n_spec == 0:
        if fits:
            raise ComputePlanError(
                f"--compute {compute} on a net that already fits in VRAM needs "
                "an explicit --gpu-layers (canary). Do not silently split 3B/14B/20B."
            )
        default = (
            DEFAULT_CPU_SUFFIX_GPU_LAYERS
            if compute == "cpu-suffix"
            else DEFAULT_HYBRID_GPU_LAYERS
        )
        return min(int(default), n_layers)

    derived: list[int] = []
    if has_gpu:
        derived.append(int(gpu_layers))
    if has_cpu:
        if int(cpu_layers) < 0:
            raise ComputePlanError(f"--cpu-layers must be >= 0, got {cpu_layers}")
        derived.append(n_layers - int(cpu_layers))
    if has_frac:
        frac = float(gpu_frac)
        if not 0.0 <= frac <= 1.0:
            raise ComputePlanError(f"--gpu-frac must be in 0..1, got {gpu_frac}")
        derived.append(int(math.floor(frac * n_layers)))

    if has_gpu and has_cpu and int(gpu_layers) + int(cpu_layers) != n_layers:
        raise ComputePlanError(
            f"--gpu-layers {gpu_layers} + --cpu-layers {cpu_layers} "
            f"must sum to num_hidden_layers={n_layers}"
        )
    if len(set(derived)) > 1:
        raise ComputePlanError(
            f"--gpu-layers/--cpu-layers/--gpu-frac disagree: {derived}"
        )
    n_gpu = derived[0]
    if not 0 <= n_gpu <= n_layers:
        raise ComputePlanError(
            f"gpu_layers={n_gpu} is outside 0..{n_layers} (num_hidden_layers)"
        )
    return n_gpu


def plan_compute(
    descs: Sequence[WeightDesc],
    cfg: Mapping[str, object],
    *,
    compute: str = "gpu",
    gpu_layers: int | None = None,
    cpu_layers: int | None = None,
    gpu_frac: float | None = None,
    vram_mib: int = VRAM_MIB_3080,
    max_seq: int = 2048,
    residency_policy: str = DEFAULT_POLICY,
    cpu_codec: str = "nf4",
) -> ComputePlan:
    """Build a :class:`ComputePlan`. CPU-only; no ``.chr`` required."""
    compute = str(compute)
    if compute not in COMPUTE_MODES:
        raise ComputePlanError(
            f"--compute {compute!r}; expected one of gpu, cpu-suffix, hybrid"
        )
    cpu_codec = str(cpu_codec or "nf4")
    if cpu_codec not in CPU_CODECS:
        raise ComputePlanError(
            f"--cpu-codec {cpu_codec!r}; expected nf4 or i4c"
        )
    n_layers = int(cfg["num_hidden_layers"])
    if n_layers < 1:
        raise ComputePlanError(f"num_hidden_layers={n_layers}")
    vram = int(vram_mib)
    seq = int(max_seq)
    fits = net_fits_resident(descs, cfg, vram_mib=vram, max_seq=seq)
    n_gpu = _resolve_n_gpu(
        n_layers,
        compute,
        gpu_layers=gpu_layers,
        cpu_layers=cpu_layers,
        gpu_frac=gpu_frac,
        fits=fits,
    )
    n_cpu = n_layers - n_gpu
    if cpu_codec == "i4c" and n_cpu == 0:
        raise ComputePlanError(
            "--cpu-codec i4c needs --compute cpu-suffix or hybrid"
        )
    layer_n = _typical_layer_nbytes(descs)
    layer_mib = layer_n / MIB

    if compute == "gpu" or n_cpu == 0:
        kv_policy = "gpu-all"
        budget = prefix_budget_mib(
            descs, cfg, n_layers, max_seq=seq, kv=kv_policy
        )
        ring = "none"
        ring_n = 0
        ring_mib = 0.0
        if not fits:
            cap = overflow_resident_cap(vram, seq, descs)
            rplan = plan_residency(descs, cap, policy=residency_policy)
            ring = "D" if residency_policy in ("D", DEFAULT_POLICY) else str(residency_policy)
            ring_n = len(rplan.streamed)
            ring_mib = rplan.streamed_bytes / MIB
            budget = {
                **budget,
                "resident_mib": rplan.resident_bytes / MIB,
                "cpu_weight_mib": 0.0,
                "used_mib": rplan.resident_bytes / MIB + budget["kv_gpu_mib"],
                "used_plus_overhead_mib": rplan.resident_bytes / MIB
                + budget["kv_gpu_mib"]
                + RUNTIME_OVERHEAD_MIB,
            }
        return ComputePlan(
            compute="gpu",
            n_gpu=n_layers,
            n_cpu=0,
            n_layers=n_layers,
            lm_head="device",
            embed="device",
            kv=kv_policy,
            ring=ring,
            max_seq=seq,
            vram_mib=vram,
            resident_mib=budget["resident_mib"],
            cpu_weight_mib=0.0,
            kv_gpu_mib=budget["kv_gpu_mib"],
            kv_cpu_mib=0.0,
            used_mib=budget["used_mib"],
            used_plus_overhead_mib=budget["used_plus_overhead_mib"],
            ring_matrices=ring_n,
            ring_streamed_mib=ring_mib,
            layer_mib=layer_mib,
            cpu_codec="nf4",
        )

    kv_policy = "split"
    budget = prefix_budget_mib(descs, cfg, n_gpu, max_seq=seq, kv=kv_policy)
    ring = "none"
    ring_n = 0
    ring_mib = 0.0
    resident_mib = budget["resident_mib"]
    used_mib = budget["used_mib"]
    used_plus = budget["used_plus_overhead_mib"]
    # Whole-layer prefix fits: no tape. If it does not, GPU still *computes*
    # those layers: policy D streams prefix MLP from pinned RAM (CopyRing),
    # suffix stays pageable CPU. Join the ring before the 10 KiB bounce.
    if used_plus + SLACK_MIB > vram:
        prefix = prefix_weight_descs(descs, n_gpu)
        kv_tok = max(
            1, KV_BYTES_PER_TOKEN_32B * n_gpu // max(1, n_layers)
        )
        cap = overflow_resident_cap(
            vram, seq, prefix, kv_bytes_per_token=kv_tok
        )
        rplan = plan_residency(prefix, cap, policy=residency_policy)
        ring = "D" if residency_policy in ("D", DEFAULT_POLICY) else str(residency_policy)
        ring_n = len(rplan.streamed)
        ring_mib = rplan.streamed_bytes / MIB
        resident_mib = rplan.resident_bytes / MIB
        used_mib = resident_mib + budget["kv_gpu_mib"]
        used_plus = used_mib + RUNTIME_OVERHEAD_MIB
    return ComputePlan(
        compute=compute,
        n_gpu=n_gpu,
        n_cpu=n_cpu,
        n_layers=n_layers,
        lm_head="device",
        embed="device",
        kv=kv_policy,
        ring=ring,
        max_seq=seq,
        vram_mib=vram,
        resident_mib=resident_mib,
        cpu_weight_mib=budget["cpu_weight_mib"],
        kv_gpu_mib=budget["kv_gpu_mib"],
        kv_cpu_mib=budget["kv_cpu_mib"],
        used_mib=used_mib,
        used_plus_overhead_mib=used_plus,
        ring_matrices=ring_n,
        ring_streamed_mib=ring_mib,
        layer_mib=layer_mib,
        cpu_codec=cpu_codec,
    )


def format_compute_stderr(plan: ComputePlan) -> str:
    """Load-time dump. Hybrid example is in docs/plan-cpu-hybrid.md §5.1."""
    ring = "off" if plan.ring == "none" else "on"
    codec = f" cpu_codec={plan.cpu_codec}" if plan.n_cpu else ""
    lines = [
        f"compute={plan.compute} gpu_layers={plan.n_gpu}/{plan.n_layers} "
        f"cpu_layers={plan.n_cpu}{codec} lm_head={plan.lm_head}",
        f"kv gpu={plan.kv_gpu_mib:.0f} MiB cpu={plan.kv_cpu_mib:.0f} MiB  "
        f"({plan.kv}, max_seq={plan.max_seq})",
    ]
    if plan.n_cpu:
        extra = ""
        if plan.ring != "none":
            extra = (
                f" streamed={plan.ring_matrices} ({plan.ring_streamed_mib:.0f} MiB)"
            )
        lines.append(
            f"resident {plan.resident_mib:.0f} MiB  "
            f"cpu_weights {plan.cpu_weight_mib:.0f} MiB pageable  "
            f"ring={ring}{extra}"
        )
    elif plan.ring != "none":
        lines.append(
            f"resident {plan.resident_mib:.0f} MiB  "
            f"ring={ring} streamed={plan.ring_matrices} "
            f"({plan.ring_streamed_mib:.0f} MiB)"
        )
    else:
        lines.append(f"resident {plan.resident_mib:.0f} MiB  ring={ring}")
    return "\n".join(lines)
