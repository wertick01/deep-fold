"""Load a CHR0 NF4 checkpoint into Decode V2. TokenLoop stays the product path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gpu.host import load_model
from gpu.loop.generate import _rope_tables

from .linear import DeviceWeights
from .plan import ArchSpec, spec_from_loaded
from .state import DecodeState

__all__ = ["LoadedDecode", "load_chr"]


@dataclass
class LoadedDecode:
    model: object
    report: object
    spec: ArchSpec
    weights: DeviceWeights
    state: DecodeState


def load_chr(
    model_dir: str,
    chr_path: str,
    *,
    max_seq: int,
    trust_remote_code: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> LoadedDecode:
    """``load_model`` + alias packed linears. Embed stays NF4 rows."""
    model, report = load_model(
        model_dir,
        chr_path,
        trust_remote_code=trust_remote_code,
        strict=True,
    )
    plan = model.deepfold_plan
    spec = spec_from_loaded(model, plan, max_seq=int(max_seq))
    weights = DeviceWeights.from_loaded(model, spec, dtype=dtype)
    device = weights.embed.device
    cos, sin = _rope_tables(model, plan, spec.max_seq, device, dtype)
    state = DecodeState.allocate(spec, weights.embed, dtype=dtype, cos=cos, sin=sin)
    return LoadedDecode(model=model, report=report, spec=spec, weights=weights, state=state)
