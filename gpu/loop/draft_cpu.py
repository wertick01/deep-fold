"""CPU HuggingFace greedy draft for speculative verify. No GPU, no sampling.

Lab-only: ``TokenLoop.generate(..., draft="cpu", drafter=CpuHfDraft(path))``.
The 32B target stays on the GPU; this object lives in RAM. Same-GPU draft is
still forbidden. Product default remains ``draft="none"``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .speculate import _id_list, _pad_k

__all__ = ["CpuHfDraft"]


def _clone_past(past):
    """Independent copy of HF ``past_key_values`` / ``Cache``. Draft must not mutate commit."""
    if past is None:
        return None
    import copy

    return copy.deepcopy(past)


class CpuHfDraft:
    """Greedy ``k`` tokens from a CPU HF causal LM. Incremental KV on commit.

    ``__call__(known_ids, k)`` returns ``LongTensor[k]``. Draft tokens are
    **not** written into the cache; only the ``known`` prefix is. ``reset()``
    at the start of each ``generate``.
    """

    def __init__(self, model_dir: str | Path, *, dtype: torch.dtype | None = None) -> None:
        from transformers import AutoModelForCausalLM

        path = str(model_dir)
        if dtype is None:
            dtype = torch.float32
        self.model_dir = path
        kwargs = dict(
            local_files_only=True,
            trust_remote_code=False,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype, **kwargs
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, **kwargs
            )
        self.model.eval()
        self.model.to("cpu")
        self._ids: list[int] = []
        self._past = None
        self._last_logits: torch.Tensor | None = None
        self.propose_calls = 0
        self.propose_ms = 0.0

    def reset(self) -> None:
        self._ids = []
        self._past = None
        self._last_logits = None

    def __call__(self, known, k: int) -> torch.Tensor:
        import time

        k = int(k)
        if k < 1:
            raise ValueError(f"CpuHfDraft k={k}; expected k>=1")
        t0 = time.perf_counter()
        known = _id_list(known)
        if not known:
            self.reset()
            return torch.tensor(_pad_k([0], k), dtype=torch.long)
        self._commit(known)
        drafted = self._greedy_k(k)
        self.propose_calls += 1
        self.propose_ms += (time.perf_counter() - t0) * 1000.0
        return torch.tensor(drafted, dtype=torch.long)

    def _commit(self, known: list[int]) -> None:
        if known == self._ids:
            return
        if self._ids and known[: len(self._ids)] == self._ids:
            extra = known[len(self._ids) :]
            if extra:
                self._forward(extra)
            return
        self.reset()
        self._forward(known)

    def _forward(self, tokens: list[int]) -> None:
        if not tokens:
            return
        x = torch.tensor([tokens], dtype=torch.long)
        with torch.inference_mode():
            out = self.model(x, past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        self._last_logits = out.logits[0, -1].detach().to("cpu")
        self._ids.extend(int(t) for t in tokens)

    def _greedy_k(self, k: int) -> list[int]:
        if self._last_logits is None:
            return _pad_k([0], k)
        logits = self._last_logits
        past = _clone_past(self._past)
        drafted: list[int] = []
        with torch.inference_mode():
            for _ in range(k):
                tok = int(logits.reshape(-1).argmax())
                drafted.append(tok)
                x = torch.tensor([[tok]], dtype=torch.long)
                out = self.model(x, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
        return drafted
