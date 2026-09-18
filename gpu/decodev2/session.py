"""CLI-shaped greedy session on Decode V2.

Default CLI is ``--executor auto``: resident NF4 (3B/14B/20B) uses this
loop; overflow / VQ / CopyRing stay on TokenLoop. Force with
``--executor tokenloop`` or ``--executor decodev2``. Streaming uses host
``.item()`` like TokenLoop.
"""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from gpu.graphs import IMPLEMENTED_FAMILIES
from gpu.loop.generate import Generation

from .budget import estimate_v2_from_config
from .embed import PackedEmbed
from .graph import capture_greedy
from .linear import DeviceWeights, set_linear_backend
from .plan import spec_from_loaded
from .prefill import prefill_chunk_width
from .runner import consume_prompt
from .state import DecodeState
from .step import greedy_decode, teacher_force_token

__all__ = ["BoundV2", "DecodeV2Loop", "decodev2_refused", "pick_executor"]


@dataclass
class BoundV2:
    """Packed aliases + RoPE, before KV/arena. Drop the HF tree, then finish."""

    spec: object
    weights: DeviceWeights
    cos: torch.Tensor
    sin: torch.Tensor
    dtype: torch.dtype


class _KvShim:
    """TokenLoop ``loop.kv.mib`` / ``seq_len`` for the hard-eval session."""

    __slots__ = ("seq_len", "_k", "_v")

    def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.seq_len = 0
        self._k = k
        self._v = v

    @property
    def mib(self) -> float:
        return (int(self._k.nbytes) + int(self._v.nbytes)) / (1024 * 1024)


def decodev2_refused(
    report,
    plan=None,
    *,
    max_seq: int | None = None,
    vram_mib: int | None = None,
    config=None,
    packed_embed: bool = True,
) -> str | None:
    """Why Decode V2 cannot drive this load, or None."""
    if bool(getattr(report, "overflow", False)):
        return "decodev2: overflow / CopyRing is TokenLoop-only"
    codec = str(getattr(report, "codec", "nf4") or "nf4")
    if codec == "vq":
        return "decodev2: VQ is not supported"
    family = getattr(plan, "family", None) if plan is not None else None
    if family is not None and family not in IMPLEMENTED_FAMILIES:
        return f"decodev2: family {family} is TokenLoop-only"
    if vram_mib is None or config is None or max_seq is None:
        return None
    packed = float(getattr(report, "device_mib", 0.0) or 0.0)
    try:
        est = estimate_v2_from_config(
            packed, config, max_seq=int(max_seq), packed_embed=bool(packed_embed)
        )
    except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError):
        return None
    card = int(vram_mib)
    if est.total_mib <= card:
        return None
    extra = ""
    if est.dense_embed_extra_mib:
        extra = f" + dense embed {est.dense_embed_extra_mib:.0f}"
    return (
        f"decodev2: estimated {est.total_mib:.0f} MiB "
        f"(packed {est.packed_mib:.0f} + kv {est.kv_mib:.0f} + graph {est.graph_mib:.0f} "
        f"+ WDDM {est.wddm_reserve_mib:.0f}{extra}) exceeds {card} MiB VRAM"
    )


def pick_executor(
    requested: str,
    report,
    plan=None,
    *,
    max_seq: int | None = None,
    vram_mib: int | None = None,
    config=None,
    packed_embed: bool = True,
) -> tuple[str, str | None]:
    """``(engine, reason)``. ``auto`` falls back to TokenLoop when V2 cannot.

    Forced ``decodev2`` still returns ``("decodev2", reason)`` so the CLI can
    raise; it does not silently downgrade.
    """
    want = str(requested or "auto").strip().lower() or "auto"
    why = decodev2_refused(
        report,
        plan,
        max_seq=max_seq,
        vram_mib=vram_mib,
        config=config,
        packed_embed=packed_embed,
    )
    if want == "tokenloop":
        return "tokenloop", None
    if want == "decodev2":
        return "decodev2", why
    if why is None:
        return "decodev2", None
    return "tokenloop", why


class DecodeV2Loop:
    """TokenLoop-compatible generate/reset for ``deepfold run`` / ``chat``."""

    def __init__(self, state: DecodeState, weights: DeviceWeights) -> None:
        self.state = state
        self.weights = weights
        spec = state.spec
        self.max_seq = int(spec.max_seq)
        self.prefill_chunk = prefill_chunk_width(state)
        self.plan = SimpleNamespace(
            family=spec.family,
            qkv_pack="internlm" if spec.family == "internlm_gqa" else None,
        )
        self.graph_mode = "off"
        self.graph_error: str | None = None
        self._graph = None
        self.kv = _KvShim(state.kv.k, state.kv.v)

    @property
    def weight_bytes(self) -> int:
        """Device bytes of packed NF4 + embed rows, deduplicated by pointer."""
        seen: dict[int, int] = {}

        def add(t) -> None:
            if t is None or int(t.numel()) == 0:
                return
            seen[int(t.data_ptr())] = int(t.numel()) * int(t.element_size())

        emb = self.weights.embed
        if isinstance(emb, PackedEmbed):
            add(emb.packed)
            add(emb.scale)
            add(emb.lut)
        else:
            add(emb)
        add(self.weights.final_norm)
        add(self.weights.lm_head.packed)
        add(self.weights.lm_head.scale)
        add(self.weights.lm_head.bias)
        for ly in self.weights.layers:
            add(ly.norm1)
            add(ly.norm2)
            for lin in (ly.q, ly.k, ly.v, ly.wqkv, ly.o, ly.gate, ly.up, ly.down):
                if lin is None:
                    continue
                add(lin.packed)
                add(lin.scale)
                add(lin.bias)
        return sum(seen.values())

    @property
    def device(self) -> torch.device:
        return self.state.device

    def _sync_kv_len(self) -> None:
        self.kv.seq_len = int(self.state.valid_len.item())

    def _id_list(self, prompt_ids: torch.Tensor | Sequence[int]) -> list[int]:
        if hasattr(prompt_ids, "reshape"):
            return [int(x) for x in prompt_ids.reshape(-1).tolist()]
        return [int(x) for x in prompt_ids]

    @classmethod
    def bind(
        cls, model, *, max_seq: int, dtype: torch.dtype = torch.bfloat16
    ) -> BoundV2:
        """Alias packed weights. Does not allocate the KV cache."""
        from gpu.loop.generate import _rope_tables

        plan = getattr(model, "deepfold_plan", None)
        if plan is None:
            raise RuntimeError("decodev2: model has no deepfold_plan")
        set_linear_backend("gemv")
        try:
            spec = spec_from_loaded(model, plan, max_seq=int(max_seq))
            weights = DeviceWeights.from_loaded(model, spec, dtype=dtype)
            device = weights.embed.device
            cos, sin = _rope_tables(model, plan, spec.max_seq, device, dtype)
        except RuntimeError:
            raise
        except (TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError(f"decodev2: {exc}") from exc
        return BoundV2(spec=spec, weights=weights, cos=cos, sin=sin, dtype=dtype)

    @classmethod
    def finish(cls, bound: BoundV2, *, reclaim: bool = False) -> "DecodeV2Loop":
        """Allocate KV/arena. ``reclaim`` drops CPython refs after the HF tree.

        Does not call ``empty_cache``. At the 12 GB cap that returned pages to
        WDDM and the first GEMV/graph step paged the working set back in.
        """
        if reclaim:
            gc.collect()
        state = DecodeState.allocate(
            bound.spec, bound.weights.embed, dtype=bound.dtype, cos=bound.cos, sin=bound.sin
        )
        return cls(state, bound.weights)

    @classmethod
    def from_model(
        cls, model, *, max_seq: int, dtype: torch.dtype = torch.bfloat16
    ) -> "DecodeV2Loop":
        """Bind + allocate in one call. Prefer bind / drop HF / finish on 20B."""
        return cls.finish(cls.bind(model, max_seq=max_seq, dtype=dtype))

    def reset(self) -> None:
        self.state.reset()
        self.kv.seq_len = 0

    @torch.no_grad()
    def prefill_from(self, ids: torch.Tensor | Sequence[int], start_pos: int) -> torch.Tensor:
        """Walk ``ids`` into KV slots ``start_pos..``. Chat session path."""
        seq = self._id_list(ids)
        n = len(seq)
        pos = int(start_pos)
        if n == 0:
            raise ValueError("empty prompt")
        if pos < 0 or pos + n > self.max_seq:
            raise ValueError(f"position {pos}+{n} past max_seq={self.max_seq}")
        consume_prompt(self.state, self.weights, seq, start=pos)
        self._sync_kv_len()
        return self.state.arena.logits[0]

    @torch.no_grad()
    def forward(
        self,
        ids: torch.Tensor | Sequence[int],
        start_pos: int,
        *,
        logits: bool = True,
        all_positions: bool = False,
    ) -> torch.Tensor:
        """Rewrite slots ``start_pos..`` (repeat-last-token for session KV)."""
        del logits, all_positions
        return self.prefill_from(ids, start_pos)

    @torch.no_grad()
    def seal_last(self, token_id: int) -> None:
        """Write the last sampled token so the next suffix can append."""
        if int(self.state.valid_len.item()) >= self.max_seq:
            return
        teacher_force_token(self.state, self.weights, int(token_id))
        self._sync_kv_len()

    @torch.no_grad()
    def decode_from_logits(
        self,
        logits: torch.Tensor,
        max_new_tokens: int,
        *,
        prompt_len: int,
        stop: Sequence[int] = (),
        on_token: Callable[[int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        prefill_ms: float = 0.0,
    ) -> Generation:
        """Greedy decode without ``reset``/prefill. Matches ``generate()`` tokens."""
        del logits
        stop_set = frozenset(int(s) for s in stop)
        out = Generation(
            prompt_len=int(prompt_len),
            prefill_chunk=self.prefill_chunk,
            graph=self.graph_mode,
        )
        token = int(self.state.next_token.item())
        seq = int(self.state.valid_len.item())
        self.kv.seq_len = seq
        step = self._graph if self._graph is not None else greedy_decode
        self.state.token.copy_(self.state.next_token)
        out.prefill_ms = float(prefill_ms)
        start_ev = end_ev = None
        if self.state.device.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
        self._sync()
        t1 = time.perf_counter()
        max_new = max(0, int(max_new_tokens))
        for i in range(max_new):
            if should_stop is not None and should_stop():
                out.interrupted = True
                break
            out.tokens.append(token)
            if on_token is not None:
                on_token(token)
            if token in stop_set:
                out.stop_token = token
                break
            if i + 1 == max_new or seq >= self.max_seq:
                break
            if start_ev is not None:
                start_ev.record()
                step(self.state, self.weights)
                end_ev.record()
                end_ev.synchronize()
                if os.environ.get("CHR_V2_STEP_LOG") and out.decode_steps < 8:
                    print(
                        f"  decode_step[{out.decode_steps + 1}] "
                        f"gpu={start_ev.elapsed_time(end_ev):.1f}ms",
                        flush=True,
                    )
            else:
                step(self.state, self.weights)
            out.decode_steps += 1
            token = int(self.state.token.item())
            seq += 1
            self.kv.seq_len = seq
        self._sync()
        out.decode_ms = (time.perf_counter() - t1) * 1000.0
        self._sync_kv_len()
        return out

    def _sync(self) -> None:
        if self.state.device.type == "cuda":
            torch.cuda.synchronize()

    def warmup(self, *, prompt: int = 8, tokens: int = 8) -> float:
        n = max(1, min(int(prompt), self.max_seq - 2))
        self.reset()
        self._sync()
        t0 = time.perf_counter()
        consume_prompt(self.state, self.weights, [1] * n)
        self.state.token.copy_(self.state.next_token)
        for _ in range(max(0, min(int(tokens), 8))):
            if int(self.state.position.item()) >= self.max_seq:
                break
            greedy_decode(self.state, self.weights)
        self._sync()
        self.reset()
        return (time.perf_counter() - t0) * 1000.0

    def capture_graphs(self) -> str:
        if self.state.device.type != "cuda":
            self.graph_mode = "off"
            self.graph_error = "cpu"
            return self.graph_mode
        try:
            self._graph = capture_greedy(self.state, self.weights, warmup=2)
            self.graph_mode = "decodev2"
            self.graph_error = None
        except Exception as exc:  # noqa: BLE001
            self._graph = None
            self.graph_mode = "off"
            self.graph_error = f"{type(exc).__name__}: {exc}"
        self.reset()
        return self.graph_mode

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
        oracle_ids=None,
        drafter=None,
    ) -> Generation:
        if str(draft) != "none" and int(speculate) > 1:
            raise ValueError("decodev2 does not speculate; omit --executor decodev2")
        self.reset()
        ids = prompt_ids.reshape(-1)
        n = int(ids.numel())
        if n == 0:
            raise ValueError("empty prompt")
        if n > self.max_seq:
            raise ValueError(f"prompt {n} exceeds max_seq={self.max_seq}")
        self._sync()
        t0 = time.perf_counter()
        logits = self.prefill_from(ids, 0)
        self._sync()
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        return self.decode_from_logits(
            logits,
            max_new_tokens,
            prompt_len=n,
            stop=stop,
            on_token=on_token,
            should_stop=should_stop,
            prefill_ms=prefill_ms,
        )
