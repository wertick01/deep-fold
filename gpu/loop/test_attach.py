"""``TokenLoop`` binds from a :class:`DriverPlan`, on CPU, with no kernel.

    python -m pytest gpu/loop/test_attach.py -q
    python gpu/loop/test_attach.py

No CUDA, no ``.chr``, no 3B. The seats are real
:class:`~gpu.host.linear.CompressedLinear` modules holding four bytes each, which
is enough for :meth:`gpu.loop.graph.Gemm.of` to read them and therefore enough to
prove *which module ended up in which slot*. The GEMMs are never launched.

What this file is for:

1. the loop's four-groups-per-layer structure now comes from
   ``plan.layers[i].gemms``, and the same code binds Qwen2 names and InternLM2
   names -- there is no ``hasattr(layer0.attention, "wqkv")`` left to fork on;
2. the packer is a table entry, not a boolean: ``internlm_gqa`` and ``concat``
   disagree on the same ``y``, which is exactly why choosing one from
   "the matrix is fused" would decode fluent garbage;
3. a family the loop does not implement never gets a ``TokenLoop`` at all;
4. the walker still recognises a tree whose Linears are already compressed
   seats, so ``TokenLoop(model)`` works after ``load_chr_nf4``.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.attach import DriverPlan, attach_module  # noqa: E402
from gpu.host.linear import CompressedLinear, k_pad  # noqa: E402
from gpu.host.model import replace_linears  # noqa: E402
from gpu.host.vq_linear import CompressedVqLinear, k_pad_vq  # noqa: E402
from gpu.host.test_attach import (  # noqa: E402
    HEAD_DIM,
    internlm_model,
    qwen2_model,
)
from gpu.loop import generate as generate_mod  # noqa: E402
from gpu.loop.generate import (  # noqa: E402
    PACKERS,
    TokenLoop,
    split_concat_qkv,
    split_internlm_wqkv,
    split_neox_qkv,
)
from gpu.tests.skips import Skip  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------------------- #
# a model that can be bound without a kernel
# --------------------------------------------------------------------------- #


class _Rotary(nn.Module):
    """``(cos, sin)`` for the positions asked for. Shapes only; values unused here."""

    def forward(self, ref: torch.Tensor, pos: torch.Tensor):
        n = int(pos.shape[-1])
        table = torch.zeros((1, n, HEAD_DIM), dtype=torch.float32)
        return table, table.clone()


def _fake_matrix(mod: CompressedLinear, name: str):
    """The smallest object :meth:`CompressedLinear.attach` accepts."""
    return SimpleNamespace(
        name=name,
        M=mod.M,
        K=mod.K,
        K_pad=k_pad(mod.K),
        packed=torch.zeros(4, dtype=torch.uint8),
        scale=torch.ones(2, dtype=torch.float16),
    )


def _bound(model):
    """Replace the plan's Linears with loaded seats. Returns ``(model, plan)``."""
    plan = attach_module(model)
    replace_linears(model, slots=plan.gemm_names)
    for name in plan.gemm_names:
        seat = model.get_submodule(name)
        seat.attach(_fake_matrix(seat, name))
    model.model.rotary_emb = _Rotary()
    model.deepfold_plan = plan
    return model, plan


def _loop(model, **over) -> TokenLoop:
    """A CPU ``TokenLoop``. ``linear_max_n`` is stubbed so no kernel is probed."""
    original = generate_mod.linear_max_n
    generate_mod.linear_max_n = lambda codec="nf4", probe=16: 1
    try:
        return TokenLoop(model, max_seq=8, overlap=False, **over)
    finally:
        generate_mod.linear_max_n = original


# --------------------------------------------------------------------------- #
# 1. binding comes from the plan
# --------------------------------------------------------------------------- #


def test_qwen2_names_bind_through_the_plan() -> None:
    model, plan = _bound(qwen2_model())
    loop = _loop(model)

    assert loop.plan is plan
    assert loop._pack is None, "split attention has no packer"
    assert len(loop._groups) == 4 * plan.n_layers + 1
    assert loop.embed is model.model.embed_tokens
    assert loop._lm is model.lm_head
    assert loop._norm1[1] is model.model.layers[1].input_layernorm.weight
    assert loop._norm2[1] is model.model.layers[1].post_attention_layernorm.weight

    qkv = loop._groups[0]
    assert [g.name for g in qkv.gemms] == ["L0.q", "L0.k", "L0.v"]
    assert [g.name for g in loop._groups[2].gemms] == ["L0.gate", "L0.up"]
    assert loop._groups[-1].gemms[0].name == "lm_head"


def test_vq_seats_bind_as_codec_vq() -> None:
    """H3: TokenLoop reads CompressedVqLinear the same way as NF4."""
    model = qwen2_model()
    plan = attach_module(model)
    replace_linears(model, slots=plan.gemm_names, seat=CompressedVqLinear)
    for name in plan.gemm_names:
        seat = model.get_submodule(name)
        g = seat.K_pad // 8
        seat.attach(
            SimpleNamespace(
                name=name,
                M=seat.M,
                K=seat.K,
                K_pad=k_pad_vq(seat.K),
                index=torch.zeros(seat.M, g, 2, dtype=torch.uint8),
                book=torch.zeros(2, 256, 8, dtype=torch.float16),
            )
        )
    model.model.rotary_emb = _Rotary()
    model.deepfold_plan = plan
    loop = _loop(model)
    assert loop._groups[0].gemms[0].codec == "vq"
    assert loop._groups[0].gemms[0].index is not None
    assert loop.prefill_chunk == 1, "VQ kernel is decode-only"


def test_internlm2_names_bind_through_the_same_code() -> None:
    model, plan = _bound(internlm_model())
    loop = _loop(model)

    assert loop.plan.family == "internlm_gqa"
    assert loop._pack is split_internlm_wqkv, "the packer is a table entry"
    assert len(loop._groups) == 4 * plan.n_layers + 1
    assert loop.embed is model.model.tok_embeddings
    assert loop._lm is model.output
    assert loop._norm1[0] is model.model.layers[0].attention_norm.weight
    assert loop._norm2[0] is model.model.layers[0].ffn_norm.weight

    assert [g.name for g in loop._groups[0].gemms] == ["L0.qkv"], "one fused GEMM"
    assert [g.name for g in loop._groups[2].gemms] == ["L0.gate", "L0.up"]


def test_the_loop_reports_its_family() -> None:
    model, _ = _bound(internlm_model())
    text = repr(_loop(model))
    assert "internlm_gqa" in text
    assert "pack=internlm_gqa" in text


def test_a_family_the_loop_does_not_implement_never_builds() -> None:
    model, plan = _bound(qwen2_model())
    pretend = DriverPlan(
        **{**plan.__dict__, "family": "phi3_concat", "mlp": "swiglu_fused_gate_up"}
    )
    try:
        _loop(model, plan=pretend)
    except RuntimeError as exc:
        assert "phi3_concat" in str(exc)
        assert "TokenLoop implements llama_swiglu and internlm_gqa only" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("TokenLoop accepted an unimplemented family")


def test_attach_still_reads_a_tree_of_compressed_seats() -> None:
    """After ``load_chr_nf4`` the Linears are gone; the walker must still see them."""
    model, plan = _bound(qwen2_model())
    again = attach_module(model)
    assert again.gemm_names == plan.gemm_names
    assert again.family == "llama_swiglu"
    assert not any(isinstance(m, nn.Linear) for _, m in model.named_modules())


# --------------------------------------------------------------------------- #
# 2. the packers disagree, on purpose
# --------------------------------------------------------------------------- #


def test_internlm_and_concat_disagree_on_the_same_y() -> None:
    """The footgun this whole refactor exists to remove."""
    n_q, n_kv, hd, n = 8, 2, 4, 3
    torch.manual_seed(0)
    y = torch.randn(n, (n_q + 2 * n_kv) * hd)

    q1, k1, v1 = split_internlm_wqkv(y, n_q, n_kv, hd)
    q2, k2, v2 = split_concat_qkv(y, n_q, n_kv, hd)

    assert q1.shape == q2.shape == (n, n_q, hd)
    assert k1.shape == k2.shape == (n, n_kv, hd)
    assert not k1.is_contiguous(), "internlm K is a view into packed"
    assert not torch.equal(q1, q2), "fused does not imply InternLM"
    assert not torch.equal(k1, k2)
    assert not torch.equal(v1, v2)


def test_concat_is_the_plain_q_then_k_then_v_slice() -> None:
    n_q, n_kv, hd, n = 8, 2, 4, 2
    y = torch.arange(n * (n_q + 2 * n_kv) * hd, dtype=torch.float32).view(n, -1)
    q, k, v = split_concat_qkv(y, n_q, n_kv, hd)
    assert torch.equal(q, y[:, : n_q * hd].reshape(n, n_q, hd))
    assert torch.equal(k, y[:, n_q * hd : (n_q + n_kv) * hd].reshape(n, n_kv, hd))
    assert torch.equal(v, y[:, (n_q + n_kv) * hd :].reshape(n, n_kv, hd))
    assert not k.is_contiguous() and not v.is_contiguous(), "packer must not copy K/V"


def test_neox_is_per_head_interleaved_and_mha_only() -> None:
    n_q, hd, n = 4, 4, 2
    y = torch.arange(n * n_q * 3 * hd, dtype=torch.float32).view(n, -1)
    q, k, v = split_neox_qkv(y, n_q, n_q, hd)
    assert torch.equal(q[0, 0], y[0, 0:hd])
    assert torch.equal(k[0, 0], y[0, hd : 2 * hd])
    assert torch.equal(v[0, 0], y[0, 2 * hd : 3 * hd])
    try:
        split_neox_qkv(y, n_q, 2, hd)
    except ValueError as exc:
        assert "MHA only" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("neox_interleaved accepted GQA shapes")


def test_the_packer_table_is_keyed_by_the_matrix_name_not_the_family() -> None:
    from gpu.graphs import QKV_PACKERS

    assert set(QKV_PACKERS) == {"wqkv", "qkv_proj", "query_key_value"}
    assert set(PACKERS) == set(QKV_PACKERS.values())
    assert PACKERS["internlm_gqa"] is split_internlm_wqkv
    assert PACKERS["concat"] is split_concat_qkv


# --------------------------------------------------------------------------- #
# 3. stop tokens: the product function has no default
# --------------------------------------------------------------------------- #


class _FakeTokenizer:
    unk_token_id = 0

    def __init__(self, vocab: dict[str, int], eos_token_id=None) -> None:
        self._vocab = dict(vocab)
        self._ids = {i: t for t, i in vocab.items()}
        self.eos_token_id = eos_token_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id: int):
        return self._ids.get(token_id)


def test_stop_token_ids_requires_a_tokenizer() -> None:
    from gpu.loop.stop import stop_token_ids

    try:
        stop_token_ids(None)
    except ValueError as exc:
        assert "a tokenizer is required" in str(exc)
        assert "151643" in str(exc), "the message names the ids it refuses to invent"
    else:  # pragma: no cover
        raise AssertionError("stop_token_ids(None) returned a default")


def test_stop_token_ids_is_per_tokenizer() -> None:
    from gpu.loop.stop import stop_token_ids

    qwen = _FakeTokenizer({"<|endoftext|>": 151643, "<|im_end|>": 151645}, 151645)
    internlm = _FakeTokenizer({"</s>": 2, "<|im_end|>": 92542, "<|im_start|>": 92543}, 2)
    llama = _FakeTokenizer({"<|eot_id|>": 128009, "<|end_of_text|>": 128001}, 128009)
    gemma = _FakeTokenizer({"<end_of_turn>": 107, "<eos>": 1}, 1)

    assert stop_token_ids(qwen) == (151643, 151645), "the measured labs must not move"
    assert stop_token_ids(internlm) == (2, 92542)
    assert stop_token_ids(llama) == (128001, 128009)
    assert 151645 not in stop_token_ids(llama)
    assert stop_token_ids(gemma) == (1, 107)


def test_a_tokenizer_with_no_end_of_turn_id_is_an_error() -> None:
    from gpu.loop.stop import stop_token_ids

    try:
        stop_token_ids(_FakeTokenizer({"hello": 5}, None))
    except ValueError as exc:
        assert "no EOS / end-of-turn id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an empty stop set was accepted")


def test_pad_is_not_a_stop() -> None:
    from gpu.loop.stop import stop_token_ids

    tok = _FakeTokenizer({"</s>": 2, "<pad>": 3}, 2)
    tok.pad_token_id = 3
    assert stop_token_ids(tok) == (2,), "pad is not an end of turn"


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/loop plan-binding acceptance, {len(TESTS)} tests, CPU only\n")
    for fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
        except Skip as exc:
            SKIPPED.append((fn.__name__, str(exc)))
            print(f"  SKIP  {fn.__name__}  -- {exc}")
        except AssertionError as exc:
            check(fn.__name__, False, str(exc).splitlines()[0] if str(exc) else "assert")
        except Exception as exc:  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
        else:
            check(fn.__name__, True)

    failed = [name for name, ok, _ in CHECKS if not ok]
    tail = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed{tail}")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())
