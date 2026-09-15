"""CPU: policy D residency plan + HostImage packed‖scale layout. No GPU, no .chr.

    python gpu/host/test_residency.py
    python -m pytest gpu/host/test_residency.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.host_image import (  # noqa: E402
    HostImage,
    maybe_pin,
    pack_arena,
    unpack_arena,
)
from gpu.host.residency import (  # noqa: E402
    DEFAULT_POLICY,
    HOST_EMBED_POLICY,
    KV_BYTES_PER_TOKEN_32B,
    MIB,
    PIN_KINDS,
    POLICIES,
    RUNTIME_OVERHEAD_MIB,
    CapTooSmallError,
    WeightDesc,
    descs_from_qwen,
    embeddings_tied,
    mlp_slot_nbytes,
    nf4_nbytes,
    overflow_resident_cap,
    pin_nbytes,
    plan_residency,
    resident_cap_bytes,
    stride_pair_layer_ids,
    summarize_residency,
)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# gpu/cli/test_codec.py QWEN_32B / QWEN_3B (shapes only; no model dir).
QWEN_32B = dict(
    hidden_size=5120,
    intermediate_size=27648,
    num_hidden_layers=64,
    num_attention_heads=40,
    num_key_value_heads=8,
    vocab_size=152064,
    tie_word_embeddings=False,
    model_type="qwen2",
)
QWEN_3B = dict(
    hidden_size=2048,
    intermediate_size=11008,
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=2,
    vocab_size=151936,
    tie_word_embeddings=True,
    model_type="qwen2",
)

# gate/up [27648, 5120] and down [5120, 27648] — same NF4 bytes.
GATE_NBYTES = nf4_nbytes(27648, 5120)
DOWN_NBYTES = nf4_nbytes(5120, 27648)


def _by_kind(descs, kind: str):
    return [d for d in descs if d.kind == kind]


def _name(layer: int, slot: str) -> str:
    if slot in ("q", "k", "v", "o"):
        return f"model.layers.{layer}.self_attn.{slot}_proj"
    return f"model.layers.{layer}.mlp.{slot}_proj"


def test_nf4_nbytes_matches_chr_matrix_formula() -> None:
    m, k = 27648, 5120
    k_pad = 64 * ((k + 63) // 64)
    packed = m * k_pad // 2
    scale = m * (k_pad // 64) * 2
    n = nf4_nbytes(m, k)
    check("nf4_nbytes packed+scale", n == packed + scale, f"{n}")
    check("gate nbytes == down nbytes", GATE_NBYTES == DOWN_NBYTES, f"{GATE_NBYTES}")
    check(
        "gate/down 71.71875 MiB",
        GATE_NBYTES / MIB == 71.71875,
        f"{GATE_NBYTES / MIB}",
    )
    k65 = nf4_nbytes(2, 65)
    k_pad65 = 128
    want = 2 * (k_pad65 // 2) + 2 * (k_pad65 // 64) * 2
    check("K=65 pads to 128", k65 == want, f"{k65}")


def test_descs_qwen_32b_shapes() -> None:
    descs = descs_from_qwen(QWEN_32B)
    n_layers = 64
    check("32B desc count", len(descs) == n_layers * 7 + 2, f"{len(descs)}")
    check("32B has lm_head", any(d.kind == "lm_head" for d in descs), "")
    check("32B one embed", sum(1 for d in descs if d.kind == "embed") == 1, "")
    q0 = next(d for d in descs if d.name == _name(0, "q"))
    k0 = next(d for d in descs if d.name == _name(0, "k"))
    g0 = next(d for d in descs if d.name == _name(0, "gate"))
    d0 = next(d for d in descs if d.name == _name(0, "down"))
    check("q [5120,5120]", (q0.M, q0.K) == (5120, 5120), f"{q0.M},{q0.K}")
    check("k GQA [1024,5120]", (k0.M, k0.K) == (1024, 5120), f"{k0.M},{k0.K}")
    check("gate [27648,5120]", (g0.M, g0.K) == (27648, 5120), f"{g0.M},{g0.K}")
    check("down [5120,27648]", (d0.M, d0.K) == (5120, 27648), f"{d0.M},{d0.K}")
    check("gate.nbytes == nf4", g0.nbytes == GATE_NBYTES, f"{g0.nbytes}")


def test_descs_qwen_3b_tied_no_second_lm_head() -> None:
    descs = descs_from_qwen(QWEN_3B)
    check("3B no lm_head desc", not any(d.kind == "lm_head" for d in descs), "")
    check("3B one embed", sum(1 for d in descs if d.kind == "embed") == 1, "")
    check("3B 36 layers * 7 + embed", len(descs) == 36 * 7 + 1, f"{len(descs)}")


def test_32b_policy_d_downs_then_tail_pairs() -> None:
    descs = descs_from_qwen(QWEN_32B)
    pin = pin_nbytes(descs)
    pair = 2 * GATE_NBYTES
    # 48 gate+up pairs stay DEVICE; slack < one down so every down is HOST
    # and layers 48..63 gate+up are HOST. Not "first 34 blocks DEVICE".
    cap = pin + 48 * pair + (DOWN_NBYTES - 1)
    plan = plan_residency(descs, cap)

    downs = _by_kind(descs, "down")
    gates = _by_kind(descs, "gate")
    ups = _by_kind(descs, "up")
    pin_descs = [d for d in descs if d.kind in PIN_KINDS]

    check(
        "all 64 down HOST",
        all(d.name in plan.streamed for d in downs) and len(downs) == 64,
        f"host downs={sum(1 for d in downs if d.name in plan.streamed)}",
    )
    check(
        "all qkvo+embed+lm_head DEVICE",
        all(d.name in plan.resident for d in pin_descs),
        "",
    )
    head_gate = [d for d in gates if d.layer is not None and d.layer <= 47]
    tail_gate = [d for d in gates if d.layer is not None and d.layer >= 48]
    head_up = [d for d in ups if d.layer is not None and d.layer <= 47]
    tail_up = [d for d in ups if d.layer is not None and d.layer >= 48]
    check(
        "gate+up L0..L47 DEVICE",
        all(d.name in plan.resident for d in head_gate + head_up)
        and len(head_gate) == 48
        and len(head_up) == 48,
        "",
    )
    check(
        "tail gate+up HOST",
        all(d.name in plan.streamed for d in tail_gate + tail_up)
        and len(tail_gate) == 16
        and len(tail_up) == 16,
        "",
    )
    check(
        "slot_nbytes == down == gate, not embed",
        plan.slot_nbytes == DOWN_NBYTES == GATE_NBYTES,
        f"{plan.slot_nbytes}",
    )
    embed_n = next(d.nbytes for d in descs if d.kind == "embed")
    check("slot is not embed", plan.slot_nbytes != embed_n, f"embed={embed_n}")
    check(
        "L0 down HOST (not policy A)",
        _name(0, "down") in plan.streamed and _name(0, "q") in plan.resident,
        "",
    )
    check(
        "L0 gate DEVICE while down HOST",
        _name(0, "gate") in plan.resident and _name(0, "up") in plan.resident,
        "",
    )
    total = sum(d.nbytes for d in descs)
    check(
        "resident+streamed == total",
        plan.resident_bytes + plan.streamed_bytes == total,
        f"{plan.resident_bytes}+{plan.streamed_bytes} vs {total}",
    )
    check(
        "names partition",
        plan.resident.isdisjoint(plan.streamed)
        and plan.resident | set(plan.streamed) == {d.name for d in descs},
        "",
    )


def test_streamed_order_is_tokenloop_tape() -> None:
    descs = descs_from_qwen(QWEN_32B)
    pin = pin_nbytes(descs)
    cap = pin + 48 * (2 * GATE_NBYTES) + (DOWN_NBYTES - 1)
    plan = plan_residency(descs, cap)
    # L0..L47: only down HOST. L48: gate, up, down.
    check(
        "tape starts L0 down",
        plan.streamed[0] == _name(0, "down"),
        plan.streamed[0],
    )
    check(
        "tape L1 down after L0",
        plan.streamed[1] == _name(1, "down"),
        plan.streamed[1],
    )
    i48 = plan.streamed.index(_name(48, "gate"))
    check(
        "L48 tape gate,up,down",
        plan.streamed[i48 : i48 + 3]
        == (_name(48, "gate"), _name(48, "up"), _name(48, "down")),
        str(plan.streamed[i48 : i48 + 3]),
    )
    check("lm_head not on tape", "lm_head" not in plan.streamed, "")


def test_3b_tied_small_cap() -> None:
    descs = descs_from_qwen(QWEN_3B)
    pin = pin_nbytes(descs)
    gate_n = next(d.nbytes for d in descs if d.kind == "gate")
    cap = pin + 4 * (2 * gate_n)
    plan = plan_residency(descs, cap)
    downs = _by_kind(descs, "down")
    pin_descs = [d for d in descs if d.kind in PIN_KINDS]
    check("3B all down HOST", all(d.name in plan.streamed for d in downs), "")
    check("3B qkvo+embed DEVICE", all(d.name in plan.resident for d in pin_descs), "")
    check("3B no lm_head in plan names", "lm_head" not in plan.resident | set(plan.streamed), "")
    check(
        "3B bytes partition",
        plan.resident_bytes + plan.streamed_bytes == sum(d.nbytes for d in descs),
        "",
    )


def test_gate_up_pair_not_mixed() -> None:
    hidden, inter, layers = 64, 128, 2
    descs = []
    descs.append(WeightDesc.nf4("model.embed_tokens", "embed", None, 32, hidden))
    for i in range(layers):
        descs.append(WeightDesc.nf4(_name(i, "q"), "q", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "k"), "k", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "v"), "v", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "o"), "o", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "gate"), "gate", i, inter, hidden))
        descs.append(WeightDesc.nf4(_name(i, "up"), "up", i, inter, hidden))
        descs.append(WeightDesc.nf4(_name(i, "down"), "down", i, hidden, inter))
    pin = pin_nbytes(descs)
    gate_n = nf4_nbytes(inter, hidden)
    # After both downs leave, room for 1.5 pairs. Evict the tail pair whole
    # (need ~one matrix) — never leave up DEVICE and gate HOST.
    cap = pin + 3 * gate_n
    plan = plan_residency(descs, cap)
    for i in range(layers):
        g_host = _name(i, "gate") in plan.streamed
        u_host = _name(i, "up") in plan.streamed
        check(
            f"L{i} gate/up same side",
            g_host == u_host,
            f"gate_host={g_host} up_host={u_host}",
        )
    check(
        "L0 pair DEVICE, L1 pair HOST",
        _name(0, "gate") in plan.resident and _name(1, "gate") in plan.streamed,
        "",
    )
    check("qkvo still DEVICE", _name(0, "q") in plan.resident, "")


def test_qkv_kind_is_pinned_like_qkvo() -> None:
    qkv = WeightDesc.nf4("model.layers.0.attention.wqkv", "qkv", 0, 48, 32)
    o = WeightDesc.nf4("model.layers.0.attention.wo", "o", 0, 32, 32)
    down = WeightDesc.nf4("model.layers.0.mlp.down_proj", "down", 0, 32, 64)
    gate = WeightDesc.nf4("model.layers.0.mlp.gate_proj", "gate", 0, 64, 32)
    up = WeightDesc.nf4("model.layers.0.mlp.up_proj", "up", 0, 64, 32)
    embed = WeightDesc.nf4("model.embed_tokens", "embed", None, 16, 32)
    descs = (embed, qkv, o, gate, up, down)
    pin = pin_nbytes(descs)
    cap = pin  # evict all MLP
    plan = plan_residency(descs, cap)
    check("qkv DEVICE", qkv.name in plan.resident, "")
    check("down HOST at pin cap", down.name in plan.streamed, "")


def test_cap_below_pin_raises() -> None:
    descs = descs_from_qwen(QWEN_3B)
    pin = pin_nbytes(descs)
    try:
        plan_residency(descs, pin - 1)
    except CapTooSmallError as exc:
        msg = str(exc)
        check(
            "cap < pin raises",
            "pin" in msg.lower() or "qkvo" in msg.lower() or "DEVICE" in msg,
            msg[:120],
        )
        return
    check("cap < pin raises", False, "no error")


def test_all_resident_slot_is_zero() -> None:
    descs = descs_from_qwen(QWEN_3B)
    total = sum(d.nbytes for d in descs)
    plan = plan_residency(descs, total)
    check("full cap streamed empty", plan.streamed == (), str(plan.streamed[:3]))
    check("full cap slot 0", plan.slot_nbytes == 0, f"{plan.slot_nbytes}")
    check("full cap all resident", plan.resident_bytes == total, f"{plan.resident_bytes}")


def test_resident_cap_bytes_no_978() -> None:
    slot = GATE_NBYTES
    cap = resident_cap_bytes(12288, 2048, slot)
    want = (12288 - RUNTIME_OVERHEAD_MIB) * MIB - 2048 * KV_BYTES_PER_TOKEN_32B - 2 * slot
    check("cap = (vram-1800)MiB - kv - 2*slot", cap == want, f"{cap}")
    check("did not subtract 978 MiB", cap != want - 978 * MIB, "")
    check("KV 2048 tok is 512 MiB", 2048 * KV_BYTES_PER_TOKEN_32B == 512 * MIB, "")


def test_overflow_cap_32b_between_pin_and_packed() -> None:
    descs = descs_from_qwen(QWEN_32B)
    total = sum(d.nbytes for d in descs)
    pin = pin_nbytes(descs)
    cap = overflow_resident_cap(12288, 2048, descs)
    check("32B auto cap < full packed", cap < total, f"{cap} vs packed {total}")
    check("32B auto cap > pin", cap > pin, f"{cap} vs pin {pin}")
    check("mlp slot is gate/down", mlp_slot_nbytes(descs) == GATE_NBYTES, f"{mlp_slot_nbytes(descs)}")
    check(
        "overflow_resident_cap uses that slot",
        cap == resident_cap_bytes(12288, 2048, GATE_NBYTES),
        f"{cap}",
    )
    # Toy header-shaped set: embed + qkvo + one MLP triple.
    toy = (
        WeightDesc.nf4("model.embed_tokens", "embed", None, 32, 64),
        WeightDesc.nf4("model.layers.0.self_attn.q_proj", "q", 0, 64, 64),
        WeightDesc.nf4("model.layers.0.self_attn.k_proj", "k", 0, 64, 64),
        WeightDesc.nf4("model.layers.0.self_attn.v_proj", "v", 0, 64, 64),
        WeightDesc.nf4("model.layers.0.self_attn.o_proj", "o", 0, 64, 64),
        WeightDesc.nf4("model.layers.0.mlp.gate_proj", "gate", 0, 128, 64),
        WeightDesc.nf4("model.layers.0.mlp.up_proj", "up", 0, 128, 64),
        WeightDesc.nf4("model.layers.0.mlp.down_proj", "down", 0, 64, 128),
    )
    toy_cap = overflow_resident_cap(12288, 2048, toy)
    toy_pin = pin_nbytes(toy)
    check("toy cap > pin", toy_cap > toy_pin, f"{toy_cap} vs {toy_pin}")
    check(
        "toy cap uses mlp slot",
        toy_cap == resident_cap_bytes(12288, 2048, mlp_slot_nbytes(toy)),
        f"{toy_cap}",
    )


def test_summarize_32b_overflow_cap() -> None:
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    plan = plan_residency(descs, cap)
    summary = summarize_residency(descs, plan)
    check(
        "summarize n_streamed is tape",
        summary["n_streamed"] == len(plan.streamed) == len(summary["streamed"]),
        f"{summary['n_streamed']}",
    )
    check(
        "all 64 down HOST in summary",
        summary["kinds"]["down"]["host"]["n"] == 64
        and summary["kinds"]["down"]["device"]["n"] == 0,
        str(summary["kinds"]["down"]),
    )
    check(
        "q stays DEVICE",
        summary["kinds"]["q"]["host"]["n"] == 0 and summary["kinds"]["q"]["device"]["n"] == 64,
        str(summary["kinds"]["q"]),
    )
    check("embed DEVICE", summary["kinds"]["embed"]["host"]["n"] == 0, str(summary["kinds"]["embed"]))
    check(
        "bytes match plan",
        summary["resident_bytes"] == plan.resident_bytes
        and summary["streamed_bytes"] == plan.streamed_bytes,
        "",
    )


def test_unknown_policy_raises() -> None:
    descs = descs_from_qwen(QWEN_3B)
    total = sum(d.nbytes for d in descs)
    try:
        plan_residency(descs, total, policy="lru")
    except ValueError as exc:
        check("unknown policy raises", "lru" in str(exc), str(exc)[:120])
        return
    check("unknown policy raises", False, "no error")


def test_default_policy_is_d() -> None:
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    implicit = plan_residency(descs, cap)
    explicit = plan_residency(descs, cap, policy="D")
    check("DEFAULT_POLICY is D", DEFAULT_POLICY == "D", DEFAULT_POLICY)
    check(
        "omitted policy == policy=D",
        implicit.streamed == explicit.streamed
        and implicit.resident == explicit.resident
        and implicit.streamed_bytes == explicit.streamed_bytes,
        "",
    )


def _assert_pin_and_pairs(descs, plan, label: str, *, pin_embed: bool = True) -> None:
    pin_descs = [
        d for d in descs if d.kind in PIN_KINDS and (pin_embed or d.kind != "embed")
    ]
    check(
        f"{label} pin set DEVICE",
        all(d.name in plan.resident for d in pin_descs),
        "",
    )
    split: list[int] = []
    by_layer: dict[int, dict[str, bool]] = {}
    for d in descs:
        if d.kind not in ("gate", "up") or d.layer is None:
            continue
        by_layer.setdefault(d.layer, {})[d.kind] = d.name in plan.streamed
    for layer, sides in by_layer.items():
        if "gate" in sides and "up" in sides and sides["gate"] != sides["up"]:
            split.append(layer)
    check(f"{label} gate/up not split", not split, f"split={split[:8]}")


def test_extra_policies_pin_and_pairs_overflow_cap() -> None:
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    check("cap is live 32B helper", cap == 10310189056, f"{cap}")
    d_plan = plan_residency(descs, cap, policy="D")
    for name in sorted(POLICIES):
        host_embed = name == HOST_EMBED_POLICY
        if host_embed and embeddings_tied(descs):
            try:
                plan_residency(descs, cap, policy=name)
            except ValueError as exc:
                check(f"{name} refuses tied", "tied" in str(exc).lower(), str(exc)[:120])
                continue
            check(f"{name} refuses tied", False, "no error")
            continue
        plan = plan_residency(descs, cap, policy=name)
        _assert_pin_and_pairs(descs, plan, name, pin_embed=not host_embed)
        check(
            f"{name} no pin kind on tape",
            not any(
                next(d.kind for d in descs if d.name == n)
                in (PIN_KINDS - ({"embed"} if host_embed else set()))
                for n in plan.streamed
            ),
            "",
        )
        if host_embed:
            embed_name = next(d.name for d in descs if d.kind == "embed")
            check(
                f"{name} embed CPU not tape",
                embed_name in plan.cpu
                and embed_name not in plan.resident
                and embed_name not in plan.streamed,
                "",
            )
    only = plan_residency(descs, cap, policy="downs_only")
    inter = plan_residency(descs, cap, policy="interleaved_down")
    check(
        "downs_only WHO == D (cap still needs tail pairs)",
        set(d_plan.streamed) == set(only.streamed)
        and d_plan.streamed_bytes == only.streamed_bytes,
        "",
    )
    check(
        "interleaved_down WHO == D (every down HOST)",
        set(d_plan.streamed) == set(inter.streamed)
        and d_plan.streamed == inter.streamed,
        "",
    )
    summary_d = summarize_residency(descs, d_plan)
    check(
        "D overflow: 64 down HOST",
        summary_d["kinds"]["down"]["host"]["n"] == 64,
        str(summary_d["kinds"]["down"]),
    )
    check(
        "D overflow: tail gate 48..63",
        summary_d["host_layers"]["gate"] == list(range(48, 64)),
        str(summary_d["host_layers"]["gate"][:5]),
    )


def test_pairs_first_and_all_mlp_overflow_cap() -> None:
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    d_plan = plan_residency(descs, cap, policy="D")
    pf = plan_residency(descs, cap, policy="pairs_first")
    am = plan_residency(descs, cap, policy="all_mlp")
    pf_s = summarize_residency(descs, pf)
    am_s = summarize_residency(descs, am)
    check(
        "pairs_first same streamed bytes as D",
        pf.streamed_bytes == d_plan.streamed_bytes,
        f"{pf.streamed_bytes} vs {d_plan.streamed_bytes}",
    )
    check(
        "pairs_first same n_streamed as D",
        len(pf.streamed) == len(d_plan.streamed) == 96,
        f"{len(pf.streamed)} vs {len(d_plan.streamed)}",
    )
    check(
        "pairs_first zero HOST downs",
        pf_s["kinds"]["down"]["host"]["n"] == 0
        and pf_s["kinds"]["down"]["device"]["n"] == 64,
        str(pf_s["kinds"]["down"]),
    )
    check(
        "pairs_first 48 HOST gate+up (L16..63)",
        pf_s["kinds"]["gate"]["host"]["n"] == 48
        and pf_s["host_layers"]["gate"] == list(range(16, 64))
        and pf_s["host_layers"]["up"] == list(range(16, 64)),
        str(pf_s["host_layers"]["gate"][:3]),
    )
    check(
        "all_mlp 192 HOST MLP",
        len(am.streamed) == 192
        and am_s["kinds"]["down"]["host"]["n"] == 64
        and am_s["kinds"]["gate"]["host"]["n"] == 64
        and am_s["kinds"]["up"]["host"]["n"] == 64,
        f"n={len(am.streamed)}",
    )
    check(
        "all_mlp more HOST bytes than D",
        am.streamed_bytes > d_plan.streamed_bytes,
        f"{am.streamed_bytes} vs {d_plan.streamed_bytes}",
    )
    check(
        "all_mlp pin still DEVICE",
        all(d.name in am.resident for d in descs if d.kind in PIN_KINDS),
        "",
    )


def test_interleaved_down_head_first_when_partial_downs() -> None:
    descs = descs_from_qwen(QWEN_32B)
    pin = pin_nbytes(descs)
    pair = 2 * GATE_NBYTES
    # All pairs DEVICE; 32 downs HOST. Eviction order picks which 32.
    cap = pin + 64 * pair + 32 * DOWN_NBYTES
    d_plan = plan_residency(descs, cap, policy="D")
    i_plan = plan_residency(descs, cap, policy="interleaved_down")
    d_s = summarize_residency(descs, d_plan)
    i_s = summarize_residency(descs, i_plan)
    check(
        "partial: both 32 HOST downs, no pairs",
        d_s["kinds"]["down"]["host"]["n"] == 32
        and i_s["kinds"]["down"]["host"]["n"] == 32
        and d_s["kinds"]["gate"]["host"]["n"] == 0
        and i_s["kinds"]["gate"]["host"]["n"] == 0,
        "",
    )
    check(
        "D tail-first downs 32..63",
        d_s["host_layers"]["down"] == list(range(32, 64)),
        str(d_s["host_layers"]["down"][:3]),
    )
    check(
        "interleaved_down head-first downs 0..31",
        i_s["host_layers"]["down"] == list(range(0, 32)),
        str(i_s["host_layers"]["down"][:3]),
    )
    check(
        "D tape starts L32 down (consume order, not L63)",
        d_plan.streamed[0] == _name(32, "down"),
        d_plan.streamed[0],
    )
    check(
        "interleaved tape starts L0 down",
        i_plan.streamed[0] == _name(0, "down"),
        i_plan.streamed[0],
    )
    check(
        "tapes are increasing layer index",
        d_plan.streamed == tuple(_name(i, "down") for i in range(32, 64))
        and i_plan.streamed == tuple(_name(i, "down") for i in range(0, 32)),
        "",
    )


def test_policies_pairs_not_split_on_toy() -> None:
    hidden, inter, layers = 64, 128, 2
    descs = [
        WeightDesc.nf4("model.embed_tokens", "embed", None, 32, hidden),
    ]
    for i in range(layers):
        descs.append(WeightDesc.nf4(_name(i, "q"), "q", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "k"), "k", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "v"), "v", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "o"), "o", i, hidden, hidden))
        descs.append(WeightDesc.nf4(_name(i, "gate"), "gate", i, inter, hidden))
        descs.append(WeightDesc.nf4(_name(i, "up"), "up", i, inter, hidden))
        descs.append(WeightDesc.nf4(_name(i, "down"), "down", i, hidden, inter))
    pin = pin_nbytes(descs)
    gate_n = nf4_nbytes(inter, hidden)
    cap = pin + 3 * gate_n
    for name in sorted(POLICIES):
        if name == HOST_EMBED_POLICY:
            try:
                plan_residency(descs, cap, policy=name)
            except ValueError as exc:
                check(
                    "toy D_host_embed refuses tied",
                    "tied" in str(exc).lower(),
                    str(exc)[:120],
                )
                continue
            check("toy D_host_embed refuses tied", False, "no error")
            continue
        plan = plan_residency(descs, cap, policy=name)
        _assert_pin_and_pairs(descs, plan, f"toy {name}")
        check(f"toy {name} q DEVICE", _name(0, "q") in plan.resident, "")


def test_policies_cap_below_pin_raises() -> None:
    descs = descs_from_qwen(QWEN_3B)
    pin = pin_nbytes(descs)
    for name in sorted(POLICIES):
        try:
            plan_residency(descs, pin - 1, policy=name)
        except CapTooSmallError:
            check(f"{name} cap < pin raises", True, "")
            continue
        except ValueError as exc:
            if name == HOST_EMBED_POLICY and "tied" in str(exc).lower():
                check(f"{name} tied refuses before cap", True, str(exc)[:80])
                continue
            check(f"{name} cap < pin raises", False, str(exc)[:120])
            continue
        check(f"{name} cap < pin raises", False, "no error")


def test_h2_place_table_rows() -> None:
    from gpu.lab.h2_place import compare_policies, overlap_story

    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    rows = compare_policies(descs, cap)
    names = [r["policy"] for r in rows]
    check(
        "h2_place PRINT_ORDER",
        names == ["D", "downs_only", "interleaved_down", "pairs_first", "all_mlp"],
        str(names),
    )
    by_name = {r["policy"]: r for r in rows}
    check("h2_place D 96 streamed", by_name["D"]["n_streamed"] == 96, "")
    check(
        "h2_place D ~6885 MiB",
        abs(by_name["D"]["streamed_mib"] - 6885.0) < 0.1,
        f"{by_name['D']['streamed_mib']}",
    )
    check(
        "h2_place D H2D uses 256/10.31",
        abs(by_name["D"]["h2d_ms"] - by_name["D"]["streamed_mib"] * 10.31 / 256)
        < 1e-6,
        f"{by_name['D']['h2d_ms']}",
    )
    d_plan = plan_residency(descs, cap, policy="D")
    story = overlap_story(descs, d_plan)
    check("D overlap mentions copy during qkv", "qkv" in story.lower(), story)
    pf_story = overlap_story(descs, plan_residency(descs, cap, policy="pairs_first"))
    check(
        "pairs_first overlap mentions wait or prefix",
        "wait" in pf_story.lower() or "all-resident" in pf_story.lower(),
        pf_story,
    )


def test_pairs_stride_32b_overflow_who() -> None:
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    d_plan = plan_residency(descs, cap, policy="D")
    s_plan = plan_residency(descs, cap, policy="pairs_stride")
    want = list(range(3, 64, 4))
    check("stride ids 3,7,...,63", stride_pair_layer_ids(descs) == want, str(want[:4]))
    check("stride is 16 layers", len(want) == 16, f"{len(want)}")
    s_s = summarize_residency(descs, s_plan)
    d_s = summarize_residency(descs, d_plan)
    pin_descs = [d for d in descs if d.kind in PIN_KINDS]
    check(
        "pairs_stride pin qkvo+embed+lm_head DEVICE",
        all(d.name in s_plan.resident for d in pin_descs),
        "",
    )
    check(
        "pairs_stride all 64 down HOST",
        s_s["kinds"]["down"]["host"]["n"] == 64
        and s_s["host_layers"]["down"] == list(range(64)),
        str(s_s["host_layers"]["down"][:3]),
    )
    check(
        "pairs_stride gate WHO is every 4th",
        s_s["host_layers"]["gate"] == want and s_s["host_layers"]["up"] == want,
        str(s_s["host_layers"]["gate"]),
    )
    check(
        "pairs_stride n_host pairs == D",
        s_s["kinds"]["gate"]["host"]["n"] == d_s["kinds"]["gate"]["host"]["n"] == 16,
        f"{s_s['kinds']['gate']['host']['n']} vs {d_s['kinds']['gate']['host']['n']}",
    )
    check(
        "pairs_stride streamed_bytes == D",
        s_plan.streamed_bytes == d_plan.streamed_bytes,
        f"{s_plan.streamed_bytes} vs {d_plan.streamed_bytes}",
    )
    check(
        "pairs_stride n_streamed == D (96)",
        len(s_plan.streamed) == len(d_plan.streamed) == 96,
        f"{len(s_plan.streamed)} vs {len(d_plan.streamed)}",
    )
    check(
        "pairs_stride WHO != D tail 48..63",
        s_s["host_layers"]["gate"] != d_s["host_layers"]["gate"],
        "",
    )
    check("pairs_stride cpu empty", s_plan.cpu == frozenset(), str(s_plan.cpu))
    i3 = s_plan.streamed.index(_name(3, "gate"))
    check(
        "L3 tape gate,up,down",
        s_plan.streamed[i3 : i3 + 3]
        == (_name(3, "gate"), _name(3, "up"), _name(3, "down")),
        str(s_plan.streamed[i3 : i3 + 3]),
    )
    check("L0 gate DEVICE under stride", _name(0, "gate") in s_plan.resident, "")
    check("L4 gate DEVICE (not stride)", _name(4, "gate") in s_plan.resident, "")


def test_host_embed_32b_refill_and_tied_3b() -> None:
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    d_plan = plan_residency(descs, cap, policy="D")
    he = plan_residency(descs, cap, policy=HOST_EMBED_POLICY)
    he_pin = plan_residency(descs, cap, policy="D", pin_embed=False)
    embed_name = next(d.name for d in descs if d.kind == "embed")
    embed_n = next(d.nbytes for d in descs if d.kind == "embed")
    he_s = summarize_residency(descs, he)
    d_s = summarize_residency(descs, d_plan)
    pin_no_embed = [d for d in descs if d.kind in PIN_KINDS and d.kind != "embed"]
    check(
        "D_host_embed == pin_embed=False",
        he.streamed == he_pin.streamed
        and he.resident == he_pin.resident
        and he.cpu == he_pin.cpu,
        "",
    )
    check(
        "host-embed embed in cpu",
        embed_name in he.cpu and len(he.cpu) == 1,
        str(he.cpu),
    )
    check(
        "host-embed embed not DEVICE",
        embed_name not in he.resident,
        "",
    )
    check(
        "host-embed embed not on tape",
        embed_name not in he.streamed,
        "",
    )
    check(
        "host-embed qkvo+lm_head DEVICE",
        all(d.name in he.resident for d in pin_no_embed),
        "",
    )
    check(
        "host-embed kinds embed cpu n=1",
        he_s["kinds"]["embed"]["cpu"]["n"] == 1
        and he_s["kinds"]["embed"]["host"]["n"] == 0
        and he_s["kinds"]["embed"]["device"]["n"] == 0,
        str(he_s["kinds"]["embed"]),
    )
    # D: 64 downs + 16 pairs. Refill 2 pairs + 1 down → 63 downs + 14 pairs.
    check(
        "host-embed 63 HOST downs (1 refilled)",
        he_s["kinds"]["down"]["host"]["n"] == 63
        and 0 not in he_s["host_layers"]["down"]
        and _name(0, "down") in he.resident,
        str(he_s["host_layers"]["down"][:3]),
    )
    check(
        "host-embed 14 HOST pairs (2 refilled)",
        he_s["kinds"]["gate"]["host"]["n"] == 14
        and he_s["host_layers"]["gate"] == list(range(50, 64)),
        str(he_s["host_layers"]["gate"][:3]),
    )
    check(
        "host-embed n_streamed is D minus 5",
        len(he.streamed) == len(d_plan.streamed) - 5,
        f"{len(he.streamed)} vs {len(d_plan.streamed)}",
    )
    want_bytes = d_plan.streamed_bytes - (2 * 2 * GATE_NBYTES) - DOWN_NBYTES
    check(
        "host-embed streamed_bytes D minus 2 pairs + 1 down",
        he.streamed_bytes == want_bytes,
        f"{he.streamed_bytes} vs {want_bytes}",
    )
    check(
        "host-embed names partition with cpu",
        he.resident.isdisjoint(he.streamed)
        and he.resident.isdisjoint(he.cpu)
        and set(he.streamed).isdisjoint(he.cpu)
        and he.resident | set(he.streamed) | he.cpu == {d.name for d in descs},
        "",
    )
    check(
        "refill bytes under embed nbytes",
        (2 * 2 * GATE_NBYTES + DOWN_NBYTES) < embed_n,
        f"5*mlp={2 * 2 * GATE_NBYTES + DOWN_NBYTES} embed={embed_n}",
    )
    check("D host layers still tail 48..63", d_s["host_layers"]["gate"] == list(range(48, 64)), "")
    no_refill = plan_residency(descs, cap, policy="D", pin_embed=False, refill_embed=False)
    nr_s = summarize_residency(descs, no_refill)
    check(
        "no-refill MLP WHO == D",
        nr_s["host_layers"]["gate"] == d_s["host_layers"]["gate"]
        and nr_s["host_layers"]["down"] == d_s["host_layers"]["down"]
        and embed_name in no_refill.cpu,
        str(nr_s["host_layers"]["gate"][:3]),
    )
    descs3 = descs_from_qwen(QWEN_3B)
    check("3B embeddings_tied", embeddings_tied(descs3), "")
    check("32B not tied", not embeddings_tied(descs), "")
    try:
        plan_residency(descs3, overflow_resident_cap(12288, 512, descs3), policy=HOST_EMBED_POLICY)
    except ValueError as exc:
        check("3B host-embed refuses tied", "tied" in str(exc).lower(), str(exc)[:160])
    else:
        check("3B host-embed refuses tied", False, "no error")


def test_host_image_roundtrip_views() -> None:
    m, k = 5, 70  # K_pad = 128
    kpad = 128
    packed = torch.arange(m * (kpad // 2), dtype=torch.uint8).view(m, kpad // 2)
    scale = torch.arange(m * (kpad // 64), dtype=torch.int32).to(torch.float16).view(m, kpad // 64)
    arena = pack_arena(packed, scale)
    p2, s2 = unpack_arena(arena, m, k)
    check("packed roundtrip", torch.equal(p2.cpu(), packed), "")
    check("scale roundtrip", torch.equal(s2.cpu(), scale), "")
    check("scale.dtype fp16", s2.dtype is torch.float16, str(s2.dtype))
    packed_n = packed.numel()
    check("same storage packed", p2.data_ptr() == arena.data_ptr(), "")
    check(
        "scale offset = packed bytes",
        s2.data_ptr() == arena.data_ptr() + packed_n,
        f"delta={s2.data_ptr() - arena.data_ptr()}",
    )
    check(
        "packed/scale do not overlap",
        s2.data_ptr() >= p2.data_ptr() + packed_n,
        "",
    )
    check(
        "arena = packed + scale bytes",
        int(arena.numel()) == packed_n + scale.numel() * 2,
        f"{int(arena.numel())}",
    )
    img = HostImage.from_blobs(packed, scale, k)
    check("HostImage packed view", torch.equal(img.packed, packed), "")
    check("HostImage scale fp16", img.scale.dtype is torch.float16, str(img.scale.dtype))
    pinned = maybe_pin(arena)
    check(
        "maybe_pin same bytes",
        pinned.numel() == arena.numel() and pinned.dtype == arena.dtype,
        str(pinned.dtype),
    )


TESTS = [
    test_nf4_nbytes_matches_chr_matrix_formula,
    test_descs_qwen_32b_shapes,
    test_descs_qwen_3b_tied_no_second_lm_head,
    test_32b_policy_d_downs_then_tail_pairs,
    test_streamed_order_is_tokenloop_tape,
    test_3b_tied_small_cap,
    test_gate_up_pair_not_mixed,
    test_qkv_kind_is_pinned_like_qkvo,
    test_cap_below_pin_raises,
    test_all_resident_slot_is_zero,
    test_resident_cap_bytes_no_978,
    test_overflow_cap_32b_between_pin_and_packed,
    test_summarize_32b_overflow_cap,
    test_unknown_policy_raises,
    test_default_policy_is_d,
    test_extra_policies_pin_and_pairs_overflow_cap,
    test_pairs_first_and_all_mlp_overflow_cap,
    test_interleaved_down_head_first_when_partial_downs,
    test_policies_pairs_not_split_on_toy,
    test_policies_cap_below_pin_raises,
    test_h2_place_table_rows,
    test_pairs_stride_32b_overflow_who,
    test_host_embed_32b_refill_and_tied_3b,
    test_host_image_roundtrip_views,
]


def main() -> int:
    print("gpu/host residency + HostImage, CPU only\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
