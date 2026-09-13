"""Acceptance for :mod:`gpu.host.attach`: dummy module trees, CPU, no weights.

    python -m pytest gpu/host/test_attach.py -q
    python gpu/host/test_attach.py

Nothing here calls ``from_pretrained``, downloads a repo, or touches a device.
Every fixture is a hand-built ``nn.Module`` with the *real* attribute names of
the family it stands for, because those names are the whole input to the walker.

The list is ``docs/tz/wave10-product.md`` §2.6 (which narrows wave8-arch §5.2):

* Qwen2-shaped split SwiGLU      -> ``llama_swiglu``, attn=split, pack=None
* InternLM2-shaped ``wqkv``      -> ``internlm_gqa``, pack=``internlm_gqa``
* Llama + ``q_norm``/``k_norm``  -> refuse, named qk-norm
* Mistral + ``sliding_window``   -> refuse SWA
* Mistral window ``null``        -> ``llama_swiglu``
* Qwen2.5's inert window         -> ``llama_swiglu`` (the measured 3B config!)
* Gemma ``gelu_pytorch_tanh``    -> refuse ``gemma_gelu``
* Phi-3 ``qkv_proj``/``gate_up`` -> refuse ``phi3_concat``
* Qwen2-MoE experts + router     -> refuse MoE
* a vision prefix                -> refuse vision
* mixed split/fused layers       -> refuse non-uniform
* an extra adapter ``Linear``    -> refuse leftover Linear
* an unknown fused name          -> refuse "no packer"
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

from gpu.host.attach import AttachError, attach, attach_module, plan_violations  # noqa: E402
from gpu.host.model import replace_linears  # noqa: E402
from gpu.tests.skips import Skip  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


# --------------------------------------------------------------------------- #
# fixtures: HuggingFace attribute names, no HuggingFace
# --------------------------------------------------------------------------- #

HIDDEN, HEADS, KV_HEADS, INTER, VOCAB = 32, 8, 2, 64, 40
HEAD_DIM = HIDDEN // HEADS
Q_DIM, KV_DIM = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM


def _config(**over) -> SimpleNamespace:
    base = dict(
        model_type="qwen2",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Norm(nn.Module):
    """Stands in for any RMSNorm: one weight, and not an ``nn.Linear``."""

    def __init__(self, size: int = HIDDEN) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))


def _split_attn(*, qk_norm: bool = False, out_name: str = "o_proj") -> nn.Module:
    attn = nn.Module()
    attn.q_proj = nn.Linear(HIDDEN, Q_DIM, bias=False)
    attn.k_proj = nn.Linear(HIDDEN, KV_DIM, bias=False)
    attn.v_proj = nn.Linear(HIDDEN, KV_DIM, bias=False)
    setattr(attn, out_name, nn.Linear(Q_DIM, HIDDEN, bias=False))
    attn.head_dim = HEAD_DIM
    if qk_norm:
        attn.q_norm = _Norm(HEAD_DIM)
        attn.k_norm = _Norm(HEAD_DIM)
    return attn


def _fused_attn(name: str = "wqkv", out_name: str = "wo") -> nn.Module:
    attn = nn.Module()
    setattr(attn, name, nn.Linear(HIDDEN, Q_DIM + 2 * KV_DIM, bias=False))
    setattr(attn, out_name, nn.Linear(Q_DIM, HIDDEN, bias=False))
    attn.head_dim = HEAD_DIM
    return attn


def _swiglu(gate: str = "gate_proj", up: str = "up_proj", down: str = "down_proj") -> nn.Module:
    mlp = nn.Module()
    setattr(mlp, gate, nn.Linear(HIDDEN, INTER, bias=False))
    setattr(mlp, up, nn.Linear(HIDDEN, INTER, bias=False))
    setattr(mlp, down, nn.Linear(INTER, HIDDEN, bias=False))
    return mlp


def _llama_layer(*, qk_norm: bool = False) -> nn.Module:
    layer = nn.Module()
    layer.self_attn = _split_attn(qk_norm=qk_norm)
    layer.mlp = _swiglu()
    layer.input_layernorm = _Norm()
    layer.post_attention_layernorm = _Norm()
    return layer


def _internlm_layer() -> nn.Module:
    layer = nn.Module()
    layer.attention = _fused_attn()
    layer.feed_forward = _swiglu(gate="w1", up="w3", down="w2")
    layer.attention_norm = _Norm()
    layer.ffn_norm = _Norm()
    return layer


def _phi3_layer() -> nn.Module:
    layer = nn.Module()
    layer.self_attn = _fused_attn(name="qkv_proj", out_name="o_proj")
    mlp = nn.Module()
    mlp.gate_up_proj = nn.Linear(HIDDEN, 2 * INTER, bias=False)
    mlp.down_proj = nn.Linear(INTER, HIDDEN, bias=False)
    layer.mlp = mlp
    layer.input_layernorm = _Norm()
    layer.post_attention_layernorm = _Norm()
    return layer


def _moe_layer() -> nn.Module:
    layer = nn.Module()
    layer.self_attn = _split_attn()
    mlp = nn.Module()
    mlp.gate = nn.Linear(HIDDEN, 4, bias=False)  # the router
    mlp.experts = nn.ModuleList([_swiglu() for _ in range(2)])
    layer.mlp = mlp
    layer.input_layernorm = _Norm()
    layer.post_attention_layernorm = _Norm()
    return layer


class _CausalLM(nn.Module):
    """``model.layers`` / ``model.norm`` / ``model.embed_tokens`` + a head."""

    def __init__(
        self,
        layers,
        *,
        config=None,
        embed: str = "embed_tokens",
        head: str = "lm_head",
        extra=None,
        vision=False,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else _config()
        base = nn.Module()
        base.layers = nn.ModuleList(layers)
        setattr(base, embed, nn.Embedding(VOCAB, HIDDEN))
        base.norm = _Norm()
        if vision:
            tower = nn.Module()
            tower.merger = nn.Linear(HIDDEN, HIDDEN, bias=False)
            base.visual = tower
        self.model = base
        setattr(self, head, nn.Linear(HIDDEN, VOCAB, bias=False))
        if extra is not None:
            name, module = extra
            setattr(self.model.layers[0], name, module)


def qwen2_model() -> _CausalLM:
    return _CausalLM([_llama_layer(), _llama_layer()])


def internlm_model() -> _CausalLM:
    return _CausalLM(
        [_internlm_layer(), _internlm_layer()],
        config=_config(model_type="internlm2"),
        embed="tok_embeddings",
        head="output",
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _refusal(model, *, config=None) -> str:
    try:
        attach_module(model, config=config)
    except AttachError as exc:
        return str(exc)
    raise AssertionError("attach accepted a graph it must refuse")


# --------------------------------------------------------------------------- #
# 1. the two implemented families
# --------------------------------------------------------------------------- #


def test_qwen2_split_swiglu_is_llama_swiglu() -> None:
    plan = attach_module(qwen2_model())
    assert plan.family == "llama_swiglu", plan.family
    assert plan.attn == "split" and plan.qkv_pack is None
    assert plan.mlp == "swiglu_split" and plan.act == "silu" and plan.norm == "rms_llama"
    assert plan.n_layers == 2 and not plan.qk_norm
    assert plan.embed == "model.embed_tokens" and plan.lm_head == "lm_head"
    assert plan.final_norm == "model.norm"
    assert plan.layers[1].gemms["q"] == "model.layers.1.self_attn.q_proj"
    assert plan.layers[0].norm1 == "model.layers.0.input_layernorm"
    assert plan.layers[0].norm2 == "model.layers.0.post_attention_layernorm"
    assert plan.layers[0].attn == "model.layers.0.self_attn"
    assert not plan.trust_remote


def test_internlm2_fused_wqkv_selects_the_internlm_packer() -> None:
    plan = attach_module(internlm_model())
    assert plan.family == "internlm_gqa", plan.family
    assert plan.attn == "fused" and plan.qkv_pack == "internlm_gqa"
    assert plan.mlp == "swiglu_split", "w1/w3/w2 are gate/up/down, not a fused MLP"
    assert plan.embed == "model.tok_embeddings" and plan.lm_head == "output"
    assert plan.layers[0].gemms["qkv"] == "model.layers.0.attention.wqkv"
    assert plan.layers[0].gemms["gate"] == "model.layers.0.feed_forward.w1"
    assert plan.trust_remote, "model_type=internlm2 without auto_map still needs it"


def test_the_plan_names_every_gemm_and_nothing_else() -> None:
    """``replace_linears(slots=...)`` must cover the tree exactly."""
    model = qwen2_model()
    plan = attach_module(model)
    replaced = replace_linears(model, slots=plan.gemm_names)
    assert sorted(replaced) == sorted(plan.gemm_names)
    leftover = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    assert not leftover, leftover
    assert len(plan.gemm_names) == 2 * 7 + 1, len(plan.gemm_names)


def test_a_qwen2_config_with_an_inert_sliding_window_attaches() -> None:
    """The measured 3B ships sliding_window=32768 **and** use_sliding_window=false."""
    cfg = _config(sliding_window=32768, use_sliding_window=False)
    plan = attach_module(qwen2_model(), config=cfg)
    assert plan.family == "llama_swiglu"
    assert plan.sliding_window is None, "the switch is off, so it is full causal"


def test_a_mistral_shaped_config_without_a_window_attaches() -> None:
    cfg = _config(model_type="mistral", sliding_window=None)
    plan = attach_module(qwen2_model(), config=cfg)
    assert plan.family == "llama_swiglu"


def test_an_unknown_model_type_that_is_split_swiglu_attaches() -> None:
    """wave8-arch §2.2: model_type is not the authority, the graph is."""
    plan = attach_module(qwen2_model(), config=_config(model_type="olmo"))
    assert plan.family == "llama_swiglu"


# --------------------------------------------------------------------------- #
# 2. named refusals
# --------------------------------------------------------------------------- #


def test_mistral_with_a_live_sliding_window_refuses() -> None:
    cfg = _config(model_type="mistral", sliding_window=4096)
    text = _refusal(qwen2_model(), config=cfg)
    assert "sliding_window=4096" in text
    assert "SWA" in text and "not full causal" in text


def test_llama_with_qk_norm_refuses_rather_than_skipping_the_norms() -> None:
    model = _CausalLM(
        [_llama_layer(qk_norm=True), _llama_layer(qk_norm=True)],
        config=_config(model_type="llama"),
    )
    text = _refusal(model)
    assert "qk-norm" in text
    assert "q_norm" in text
    assert "silent quality bug" in text


def test_gemma_refuses_on_the_activation_before_any_walk() -> None:
    cfg = _config(model_type="gemma", hidden_act="gelu_pytorch_tanh")
    text = _refusal(qwen2_model(), config=cfg)
    assert "gemma_gelu" in text
    assert "TokenLoop implements llama_swiglu and internlm_gqa only" in text


def test_phi3_is_named_not_guessed() -> None:
    model = _CausalLM([_phi3_layer(), _phi3_layer()], config=_config(model_type="phi3"))
    text = _refusal(model)
    assert "phi3_concat" in text, text


def test_a_fused_qkv_with_no_packer_is_refused_by_name() -> None:
    """Never "it is fused, therefore InternLM": that decodes fluent garbage."""
    model = _CausalLM(
        [_internlm_layer(), _internlm_layer()], config=_config(model_type="mpt")
    )
    # rename wqkv -> Wqkv on both layers: fused, plausible, and unknown
    for layer in model.model.layers:
        layer.attention.Wqkv = layer.attention.wqkv
        del layer.attention.wqkv
    text = _refusal(model)
    assert "has no packer" in text
    assert "internlm_gqa (wqkv)" in text
    assert "rather than applying InternLM's split" in text


def test_moe_router_and_experts_are_refused() -> None:
    model = _CausalLM([_moe_layer(), _moe_layer()], config=_config(model_type="qwen2_moe"))
    text = _refusal(model)
    assert "MoE experts are not in this wave" in text


def test_moe_is_refused_from_the_config_before_the_walk() -> None:
    text = _refusal(qwen2_model(), config=_config(num_experts=60))
    assert "MoE experts are not in this wave" in text
    assert "num_experts=60" in text


def test_a_vision_tower_refuses_the_whole_model() -> None:
    model = _CausalLM([_llama_layer(), _llama_layer()], vision=True)
    text = _refusal(model)
    assert "vision" in text.lower()
    assert "visual" in text
    assert "skipping the tower" in text


def test_mixed_split_and_fused_layers_refuse() -> None:
    model = _CausalLM([_llama_layer(), _internlm_layer()])
    text = _refusal(model)
    assert "non-uniform" in text
    assert "do not share one slot set" in text


def test_an_extra_adapter_linear_is_not_silently_skipped() -> None:
    adapter = nn.Module()
    adapter.down_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
    model = _CausalLM([_llama_layer(), _llama_layer()])
    model.model.layers[0].adapter = adapter
    text = _refusal(model)
    # `adapter.down_proj` collides with the MLP's `down` slot before it can be
    # called a leftover; either way it is refused with the name printed.
    assert "model.layers.0.adapter.down_proj" in text
    assert "not a silent skip" in text or "two Linears claim slot" in text


def test_an_unclassified_linear_is_listed_by_name() -> None:
    model = _CausalLM([_llama_layer(), _llama_layer()])
    model.model.layers[0].self_attn.c_attn_extra = nn.Linear(HIDDEN, HIDDEN, bias=False)
    text = _refusal(model)
    assert "unclassified Linear(s)" in text
    assert "c_attn_extra" in text
    assert "not a silent skip" in text


def test_gpt2_is_refused_by_family_from_the_config() -> None:
    text = _refusal(qwen2_model(), config=_config(model_type="gpt2"))
    assert "gpt2_conv1d" in text
    assert "TokenLoop implements llama_swiglu and internlm_gqa only" in text


def test_a_transformer_h_backbone_is_refused_and_says_why() -> None:
    """``h.<n>`` + ``Conv1D`` is not a graph ``replace_linears`` can even see."""

    class Conv1D(nn.Module):  # the real one is transformers.pytorch_utils.Conv1D
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(HIDDEN, 3 * HIDDEN))

    block = nn.Module()
    block.attn = nn.Module()
    block.attn.c_attn = Conv1D()

    class GPT2Fork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # An unknown model_type, so the config gate defers and the walker
            # has to reach the backbone check on its own.
            self.config = _config(model_type="some_gpt2_fork")
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([block])

    text = _refusal(GPT2Fork())
    assert "gpt2_conv1d" in text
    assert "Conv1D" in text
    assert "transformer.h" in text


def test_no_decoder_layer_list_is_a_named_refusal() -> None:
    class Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _config()

    text = _refusal(Bare())
    assert "no decoder layer list" in text


def test_partial_rope_refuses() -> None:
    text = _refusal(qwen2_model(), config=_config(rotary_pct=0.25))
    assert "partial RoPE" in text


def test_a_gguf_path_is_refused_without_being_opened() -> None:
    import builtins

    real_open = builtins.open
    opened: list[str] = []

    def watched(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    builtins.open = watched  # type: ignore[assignment]
    try:
        for suspect in (
            r"C:\models\qwen2.5-3b-q4_k_m.gguf",
            r"C:\Users\p\.ollama\models\blobs\sha256-dead",
        ):
            try:
                attach(suspect)
            except AttachError as exc:
                assert "GGUF" in str(exc) or "Ollama" in str(exc), suspect
            else:  # pragma: no cover
                raise AssertionError(f"attach accepted {suspect}")
    finally:
        builtins.open = real_open  # type: ignore[assignment]
    leaked = [p for p in opened if ".gguf" in p.lower() or "blobs" in p.lower()]
    assert not leaked, leaked


# --------------------------------------------------------------------------- #
# 3. the load report is not allowed to be quiet
# --------------------------------------------------------------------------- #


def test_plan_violations_refuse_a_partial_load() -> None:
    plan = attach_module(qwen2_model())
    slot = plan.layers[0].gemms["q"]

    clean = SimpleNamespace(missing=[], skipped=[], leftover_meta=[])
    assert plan_violations(plan, clean) == []

    missing = SimpleNamespace(missing=[slot], skipped=[], leftover_meta=[])
    assert any(slot in line for line in plan_violations(plan, missing))

    skipped = SimpleNamespace(
        missing=[], skipped=[f"{slot} (kind=other, codec=bf16)"], leftover_meta=[]
    )
    violations = plan_violations(plan, skipped)
    assert violations and "kind=other" in violations[0]

    meta = SimpleNamespace(missing=[], skipped=[], leftover_meta=["model.norm.weight"])
    assert plan_violations(plan, meta) == ["still on meta: model.norm.weight"]

    # A skipped tensor that is not a plan slot (a norm the header lists twice,
    # say) is not a refusal: only the slots the loop reads matter.
    other = SimpleNamespace(
        missing=["model.layers.0.mlp.router"], skipped=["x.y (no module)"], leftover_meta=[]
    )
    assert plan_violations(plan, other) == []


# --------------------------------------------------------------------------- #
# 4. optional: a real tiny meta skeleton, if transformers has the class
# --------------------------------------------------------------------------- #


def test_meta_skeleton_qwen2_attaches() -> None:
    """No weights, no download: ``from_config`` under ``torch.device("meta")``."""
    try:
        from transformers import AutoModelForCausalLM, Qwen2Config
    except ImportError as exc:  # pragma: no cover
        raise Skip(f"transformers has no Qwen2Config: {exc}") from exc

    cfg = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=64,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    plan = attach_module(model)
    assert plan.family == "llama_swiglu", plan.family
    names = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    assert names == set(plan.gemm_names), names ^ set(plan.gemm_names)


def test_meta_skeleton_mistral_with_a_window_refuses() -> None:
    try:
        from transformers import AutoModelForCausalLM, MistralConfig
    except ImportError as exc:  # pragma: no cover
        raise Skip(f"transformers has no MistralConfig: {exc}") from exc

    cfg = MistralConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=64,
        sliding_window=4096,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    text = _refusal(model)
    assert "sliding_window=4096" in text


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/host attach acceptance, {len(TESTS)} tests, CPU only\n")
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
