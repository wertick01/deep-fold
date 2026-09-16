"""H2-4 CopyRing + H2-5 mixed CUDA graphs (CPU) + H2-9 lm_head prefetch + optional 3B (GPU).

    python gpu/loop/test_ring.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gpu.loop.graph as graph_mod  # noqa: E402
from gpu.cli.paths import models_root  # noqa: E402
from gpu.host.host_image import HostImage, maybe_pin  # noqa: E402
from gpu.host.linear import CompressedLinear  # noqa: E402
from gpu.host.slots import SlotPair  # noqa: E402
from gpu.loop.generate import PREFILL_HOLD_SUPERCHUNK, TokenLoop  # noqa: E402
from gpu.loop.graph import Gemm, GemmGroup, GraphedGemmGroup, group_is_resident  # noqa: E402
from gpu.loop.ring import CopyRing, default_join_copy  # noqa: E402
from gpu.loop.test_attach import _bound, qwen2_model  # noqa: E402
from gpu.loop import generate as generate_mod  # noqa: E402
from gpu.tests.skips import Skip  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []

CHR_3B = models_root() / "qwen25-3b.nf4.chr"
MODEL_3B = models_root() / "Qwen2.5-3B-Instruct"
DOWN_NAME = "model.layers.0.mlp.down_proj"
KERNEL_ABS_LIMIT = 0.05


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _host_image(m: int, k: int, fill: int = 1) -> HostImage:
    kpad = 64 * ((k + 63) // 64)
    packed = torch.full((m, kpad // 2), fill % 256, dtype=torch.uint8)
    scale = torch.full((m, kpad // 64), float(fill), dtype=torch.float16)
    return HostImage.from_blobs(packed, scale, k)


def _dev_gemm(name: str, m: int, k: int) -> Gemm:
    kpad = 64 * ((k + 63) // 64)
    packed = torch.zeros(m, kpad // 2, dtype=torch.uint8)
    scale = torch.zeros(m, kpad // 64, dtype=torch.float16)
    return Gemm(
        name=name, M=m, K=k, K_pad=kpad, bias=None, packed=packed, scale=scale, home="device"
    )


def _host_gemm(name: str, m: int, k: int, fill: int = 1) -> tuple[Gemm, HostImage]:
    img = _host_image(m, k, fill=fill)
    packed = torch.empty(0, dtype=torch.uint8)
    scale = torch.empty(0, dtype=torch.float16)
    g = Gemm(
        name=name,
        M=img.M,
        K=img.K,
        K_pad=img.K_pad,
        bias=None,
        packed=packed,
        scale=scale,
        host_image=img,
        home="host",
    )
    return g, img


def _cpu_loop(model, **over) -> TokenLoop:
    original = generate_mod.linear_max_n
    generate_mod.linear_max_n = lambda codec="nf4", probe=16: 1
    try:
        kw = dict(max_seq=8, overlap=False)
        kw.update(over)
        return TokenLoop(model, **kw)
    finally:
        generate_mod.linear_max_n = original


def _fake_nf4(launched: list):
    orig = graph_mod.nf4_gemm

    def fake(packed, scale, x, M, K, K_pad):  # noqa: N803
        launched.append(int(packed.data_ptr()))
        n = 1 if x.dim() == 1 else int(x.shape[-1])
        return torch.zeros(M, n, dtype=torch.bfloat16, device=x.device)

    graph_mod.nf4_gemm = fake
    return orig


# --------------------------------------------------------------------------- #
# CPU protocol
# --------------------------------------------------------------------------- #


def test_gemm_of_host_does_not_raise_on_empty_packed() -> None:
    m, k = 8, 64
    lin = CompressedLinear(k, m)
    img = _host_image(m, k)
    lin.attach_host(img)
    g = Gemm.of(lin, "L0.down")
    check("Gemm.of home=host", g.home == "host", g.home)
    check("Gemm.of stores host_image", g.host_image is lin.host_image, "")
    check("Gemm.of nbytes from image", g.nbytes == lin.host_image.nbytes, f"{g.nbytes}")
    check("empty packed not a raise", int(lin.packed.numel()) == 0, "")


def test_attach_host_pins_pageable_arena() -> None:
    """``Tensor.is_pinned`` is a method; ``not arena.is_pinned`` never pinned."""
    m, k = 8, 64
    img = _host_image(m, k)
    check("from_blobs pageable", not img.arena.is_pinned(), "")
    check(
        "method object is not a pin flag",
        bool(img.arena.is_pinned) and not img.arena.is_pinned(),
        "",
    )
    lin = CompressedLinear(k, m)
    lin.attach_host(img)
    arena = lin.host_image.arena
    if torch.cuda.is_available():
        check("attach_host pins", bool(arena.is_pinned()), "")
        check("HostImage replaced", lin.host_image is not img, "")
    else:
        check("no CUDA: still pageable", not arena.is_pinned(), "")


def test_bind_cpu_join_before_prefetch() -> None:
    """WDDM: synchronize copy_i before queueing copy_{i+1} (0.01 tok/s otherwise)."""
    import inspect

    src = inspect.getsource(CopyRing.bind_for_gemm)
    i_pref = src.find("self.prefetch()")
    i_join = src.find("synchronize()")
    check("bind source has prefetch()", i_pref >= 0, "" if i_pref >= 0 else "prefetch missing")
    check("bind source has synchronize()", i_join >= 0, "" if i_join >= 0 else "join missing")
    check("synchronize() before prefetch()", 0 <= i_join < i_pref, f"join={i_join} pref={i_pref}")


def test_join_copy_env_override() -> None:
    previous = os.environ.get("DEEPFOLD_COPY_JOIN")
    try:
        os.environ["DEEPFOLD_COPY_JOIN"] = "0"
        check("DEEPFOLD_COPY_JOIN=0", default_join_copy() is False, "")
        os.environ["DEEPFOLD_COPY_JOIN"] = "1"
        check("DEEPFOLD_COPY_JOIN=1", default_join_copy() is True, "")
    finally:
        if previous is None:
            os.environ.pop("DEEPFOLD_COPY_JOIN", None)
        else:
            os.environ["DEEPFOLD_COPY_JOIN"] = previous


def test_bind_prefetches_next_before_record_gemm() -> None:
    """copy(i+1) is issued in bind(i), before the caller GEMM / record_gemm."""
    g0, img0 = _host_gemm("a", 4, 64, fill=1)
    g1, _ = _host_gemm("b", 4, 64, fill=2)
    ring = CopyRing(SlotPair(img0.nbytes, "cpu"))
    ring.arm([g0, g1])
    ring.prefetch()
    ring.bind_for_gemm(g0)
    kinds = [op[0] for op in ring.ops]
    check("two H2D before GEMM", kinds.count("h2d") == 2, str(kinds))
    check("record_gemm not yet", "record_gemm" not in kinds, str(kinds))
    wait_i = kinds.index("wait_copy")
    check("next h2d after wait_copy", "h2d" in kinds[wait_i + 1 :], str(kinds))
    ring.record_gemm(g0)


def test_timing_false_skips_elapsed_time_fields() -> None:
    """Product ring: no elapsed_time counters; bind still prefetches next."""
    g0, img = _host_gemm("a", 4, 64, fill=1)
    g1, _ = _host_gemm("b", 4, 64, fill=2)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    check("timing default False", ring.timing is False, str(ring.timing))
    check("product join_copy follows profile", ring._join_copy is default_join_copy(), str(ring._join_copy))
    ring.arm([g0, g1])
    ring.prefetch()
    ring.bind_for_gemm(g0)
    kinds = [op[0] for op in ring.ops]
    check("prefetch next before record_gemm", kinds.count("h2d") == 2, str(kinds))
    check("record_gemm not yet", "record_gemm" not in kinds, str(kinds))
    snap = ring.snapshot()
    check("snapshot timing False", snap["timing"] is False, str(snap["timing"]))
    check("forward_copy_ms unused", snap["forward_copy_ms"] is None, str(snap["forward_copy_ms"]))
    check("total_copy_ms unused", snap["total_copy_ms"] is None, str(snap["total_copy_ms"]))
    check("no h2d start events", ring._e_h2d_start == (None, None), str(ring._e_h2d_start))
    ring.record_gemm(g0)


def test_three_overflow_slots_0_1_0() -> None:
    g0, img0 = _host_gemm("a", 4, 64, fill=1)
    g1, img1 = _host_gemm("b", 4, 64, fill=2)
    g2, img2 = _host_gemm("c", 4, 64, fill=3)
    ring = CopyRing(SlotPair(img0.nbytes, "cpu"))
    s0 = ring.issue(g0)
    check("first slot 0", s0 == 0 and ring.ahead == 1, f"s={s0} ahead={ring.ahead}")
    raised = False
    try:
        ring.issue(g1)
    except RuntimeError as exc:
        raised = "ahead" in str(exc) or "depth is 1" in str(exc)
    check("ahead<=1 refuses second issue", raised, "")
    packed, _ = ring.bind_for_gemm(g0)
    check("bind packed is slot0", packed.data_ptr() == ring.slots.arena[0].data_ptr(), "")
    check("bind packed is not host", packed.data_ptr() != img0.arena.data_ptr(), "")
    ring.record_gemm(g0)
    check("ahead 0 after bind", ring.ahead == 0, f"{ring.ahead}")
    s1 = ring.issue(g1)
    packed1, _ = ring.bind_for_gemm(g1)
    ring.record_gemm(g1)
    s2 = ring.issue(g2)
    packed2, _ = ring.bind_for_gemm(g2)
    ring.record_gemm(g2)
    check("3 overflow slots 0,1,0", ring.slot_order == [0, 1, 0], str(ring.slot_order))
    check("issue slots returned", (s0, s1, s2) == (0, 1, 0), f"{s0,s1,s2}")
    check("last packed is slot0", packed2.data_ptr() == ring.slots.arena[0].data_ptr(), "")
    check(
        "roundtrip slot1 fill",
        int(packed1.reshape(-1)[0].item()) == 2,
        f"{int(packed1.reshape(-1)[0].item())}",
    )
    kinds = [op[0] for op in ring.ops]
    check(
        "wait/record order starts wait_gemm,h2d,record_copy",
        kinds[:3] == ["wait_gemm", "h2d", "record_copy"],
        str(kinds[:6]),
    )
    check("has wait_copy then record_gemm", "wait_copy" in kinds and "record_gemm" in kinds, "")
    check("two-slot max_ahead 1", ring.max_ahead == 1 and ring.n_copy_streams == 1, "")


def test_three_arenas_ahead_2() -> None:
    """Product overflow: 3 slots, ahead=2, two copy-stream handles, one H2D engine."""
    gemms = [_host_gemm(n, 4, 64, fill=i + 1)[0] for i, n in enumerate("abcd")]
    ring = CopyRing(SlotPair(gemms[0].host_image.nbytes, "cpu", count=3))
    check("n_slots 3", ring.n_slots == 3, f"{ring.n_slots}")
    check("max_ahead 2", ring.max_ahead == 2, f"{ring.max_ahead}")
    check("n_copy_streams 2", ring.n_copy_streams == 2, f"{ring.n_copy_streams}")
    snap = ring.snapshot()
    check("snapshot n_slots", snap["n_slots"] == 3 and snap["max_ahead"] == 2, str(snap))
    ring.arm(gemms)
    ring.prefetch()
    check(
        "prefetch fills ahead 2",
        ring.ahead == 2 and ring.slot_order == [0, 1],
        f"ahead={ring.ahead} order={ring.slot_order}",
    )
    raised = False
    try:
        ring.issue(gemms[2])
    except RuntimeError as exc:
        raised = "ahead" in str(exc) or "depth is 2" in str(exc)
    check("ahead=2 refuses third issue", raised, "")
    ring.bind_for_gemm(gemms[0])
    check(
        "bind prefetches slot 2",
        ring.ahead == 2 and ring.slot_order == [0, 1, 2],
        str(ring.slot_order),
    )
    ring.record_gemm(gemms[0])
    ring.bind_for_gemm(gemms[1])
    ring.record_gemm(gemms[1])
    ring.bind_for_gemm(gemms[2])
    ring.record_gemm(gemms[2])
    ring.bind_for_gemm(gemms[3])
    ring.record_gemm(gemms[3])
    check("slot cycle 0,1,2,0", ring.slot_order == [0, 1, 2, 0], str(ring.slot_order))
    check("ahead 0 after tape", ring.ahead == 0, f"{ring.ahead}")


def test_prefetch_next_ahead_2_carry_prefix() -> None:
    gemms = [_host_gemm(n, 4, 64, fill=i + 1)[0] for i, n in enumerate("abc")]
    ring = CopyRing(SlotPair(gemms[0].host_image.nbytes, "cpu", count=3))
    ring.arm(gemms)
    ring.prefetch()
    for g in gemms:
        ring.bind_for_gemm(g)
        ring.record_gemm(g)
    copies = ring.total_copies
    check("three copies after tape", copies == 3, f"{copies}")
    ring.prefetch_next()
    check("prefetch_next ahead 2", ring.ahead == 2, f"{ring.ahead}")
    check("prefetch_next two extra H2D", ring.total_copies == copies + 2, f"{ring.total_copies}")
    check("queue is tape[0], tape[1]", [g.name for g, _ in ring._queue] == ["a", "b"], "")
    copies = ring.total_copies
    phase = ring._phase
    ring.arm(gemms)
    check("arm carry keeps ahead 2", ring.ahead == 2, f"{ring.ahead}")
    check("arm carry tape_i 2", ring._tape_i == 2, f"{ring._tape_i}")
    check("arm carry keeps phase", ring._phase == phase, f"{ring._phase}")
    ring.prefetch()
    check("arm+prefetch no re-H2D of prefix", ring.total_copies == copies, f"{ring.total_copies}")
    ring.bind_for_gemm(gemms[0])
    check(
        "bind first issues only tape[2]",
        ring.total_copies == copies + 1 and ring.issue_count == 1,
        f"copies={ring.total_copies} issue={ring.issue_count}",
    )
    ring.record_gemm(gemms[0])
    ring.bind_for_gemm(gemms[1])
    ring.record_gemm(gemms[1])
    ring.bind_for_gemm(gemms[2])
    ring.record_gemm(gemms[2])
    check("second forward one extra H2D", ring.total_copies == copies + 1, f"{ring.total_copies}")


def test_tape_prefetch_three_slots() -> None:
    gemms = [_host_gemm(n, 4, 64, fill=i + 1)[0] for i, n in enumerate("abc")]
    ring = CopyRing(SlotPair(gemms[0].host_image.nbytes, "cpu"))
    ring.arm(gemms)
    ring.prefetch()
    check("prefetch issues slot 0", ring.slot_order == [0] and ring.ahead == 1, str(ring.slot_order))
    for g in gemms:
        ring.bind_for_gemm(g)
        ring.record_gemm(g)
    check("tape slots 0,1,0", ring.slot_order == [0, 1, 0], str(ring.slot_order))
    check("tape issue_count 3", ring.issue_count == 3, f"{ring.issue_count}")
    check("ahead 0 at end", ring.ahead == 0, f"{ring.ahead}")


def test_ring_lifetime_bytes_survive_arm() -> None:
    g, img = _host_gemm("a", 4, 64)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    ring.arm([g])
    ring.prefetch()
    ring.bind_for_gemm(g)
    ring.record_gemm(g)
    check("lifetime copies 1", ring.total_copies == 1, f"{ring.total_copies}")
    check("lifetime bytes == arena", ring.total_bytes == img.nbytes, f"{ring.total_bytes}")
    check("forward bytes match", ring.forward_bytes == img.nbytes, f"{ring.forward_bytes}")
    snap = ring.snapshot()
    check("snapshot total_bytes", snap["total_bytes"] == img.nbytes, str(snap))
    ring.arm([g])
    check("arm zeros issue_count", ring.issue_count == 0, f"{ring.issue_count}")
    check("arm zeros forward_bytes", ring.forward_bytes == 0, f"{ring.forward_bytes}")
    check("arm keeps total_copies", ring.total_copies == 1, f"{ring.total_copies}")
    check("arm keeps total_bytes", ring.total_bytes == img.nbytes, f"{ring.total_bytes}")
    check("arm increments total_forwards", ring.total_forwards == 2, f"{ring.total_forwards}")


def test_group_is_resident_skips_host() -> None:
    host, _ = _host_gemm("down", 8, 64)
    dev = _dev_gemm("q", 16, 64)
    check("DEVICE group resident", group_is_resident(GemmGroup("q", [dev])), "")
    check("HOST group not resident", not group_is_resident(GemmGroup("down", [host])), "")
    mixed = GemmGroup("qkv", [dev, host, _dev_gemm("v", 8, 64)])
    check("mixed HOST not resident", not group_is_resident(mixed), "")
    cpu = Gemm(
        name="cpu.down",
        M=8,
        K=64,
        K_pad=64,
        bias=None,
        packed=torch.zeros(8, 32, dtype=torch.uint8),
        scale=torch.zeros(8, 1, dtype=torch.float16),
        home="cpu",
    )
    check("CPU group not resident", not group_is_resident(GemmGroup("cpu.down", [cpu])), "")
    fake_streams = (object(), object())
    cpu_grp = GemmGroup("cpu.down", [cpu], fake_streams)
    check("CPU group streams=()", cpu_grp.streams == (), str(cpu_grp.streams))


def test_tokenloop_cpu_suffix_no_ring_and_hold_refused() -> None:
    """Hybrid v1: no CopyRing even if slots exist; hold is gpu-only."""
    from gpu.host.compute import ComputePlan
    from gpu.host.linear import CompressedLinear
    from gpu.tests.nf4_oracle import k_pad, toy_nf4

    model, plan = _bound(qwen2_model())
    for slot in plan.layers[1].gemms.values():
        seat = model.get_submodule(slot)
        if not isinstance(seat, CompressedLinear):
            continue
        packed_np, scale_np = toy_nf4(seat.M, seat.K, seed=1)
        seat.attach_cpu(
            SimpleNamespace(
                name=slot,
                M=seat.M,
                K=seat.K,
                K_pad=k_pad(seat.K),
                packed=torch.from_numpy(packed_np),
                scale=torch.from_numpy(scale_np),
            )
        )
    model.deepfold_compute = ComputePlan(
        compute="hybrid",
        n_gpu=1,
        n_cpu=1,
        n_layers=2,
        kv="split",
        ring="none",
    )
    down0 = model.get_submodule(plan.layers[0].gemms["down"])
    img = _host_image(down0.M, down0.K)
    model.deepfold_slots = SlotPair(img.nbytes, "cpu")
    loop = _cpu_loop(model)
    check("suffix ignores slots / no CopyRing", loop._ring is None, str(loop._ring))
    check("n_gpu 1", loop.n_gpu == 1, f"{loop.n_gpu}")
    check("n_cpu 1", loop.n_cpu == 1, f"{loop.n_cpu}")
    check("split GPU KV 1 layer", loop.kv is not None and loop.kv.n_layers == 1, "")
    check("CPU KV 1 layer", loop.kv_cpu is not None and loop.kv_cpu.n_layers == 1, "")
    l1_qkv = loop._groups[4]
    check("L1 qkv home cpu", all(g.home == "cpu" for g in l1_qkv.gemms), str([g.home for g in l1_qkv.gemms]))
    check("L1 not CUDA-graph resident", not group_is_resident(l1_qkv), "")
    check("lm_head still device", loop._groups[-1].gemms[0].home == "device", "")
    mode = loop.capture_graphs()
    check("hybrid capture not overflow", loop.graph_error != "overflow", str(loop.graph_error))
    check("cpu seats capture off or prefix only", mode in ("off", "linears"), mode)
    hold_ok = False
    try:
        _cpu_loop(model, prefill_mode="hold")
    except ValueError as exc:
        hold_ok = "hold" in str(exc)
    check("hold refused with n_cpu", hold_ok, "")


def test_capture_cpu_does_not_run_host_groups() -> None:
    """HOST groups are not warmed inside capture() (no CopyRing, no host packed)."""
    host, img = _host_gemm("down", 8, 64, fill=7)
    dev = _dev_gemm("q", 16, 64)
    groups = [GemmGroup("q", [dev]), GemmGroup("down", [host])]
    launched: list[int] = []
    orig = _fake_nf4(launched)
    try:
        runners, mode, err = graph_mod.capture(groups, warmup=1)
    finally:
        graph_mod.nf4_gemm = orig
    check("cpu capture mode off", mode == "off", mode)
    check("cpu capture not overflow", err != "overflow", str(err))
    check("HOST runner is the eager group", runners[1] is groups[1], "")
    check("HOST packed never launched", img.arena.data_ptr() not in launched, str(launched))
    check("no nf4 during cpu capture", launched == [], f"{launched}")


def test_any_host_group_streams_empty() -> None:
    k = 64
    host, _ = _host_gemm("down", 8, k)
    fake_streams = (object(), object())
    grp = GemmGroup("down", [host], fake_streams)
    check("single HOST streams=()", grp.streams == (), str(grp.streams))
    mixed = GemmGroup(
        "qkv",
        [_dev_gemm("q", 16, k), host, _dev_gemm("v", 8, k)],
        fake_streams,
    )
    check("mixed HOST streams=()", mixed.streams == (), f"n={len(mixed.streams)}")
    qkv = GemmGroup(
        "qkv",
        [_dev_gemm("q", 16, k), _dev_gemm("k", 8, k), _dev_gemm("v", 8, k)],
        fake_streams,
    )
    check("all DEVICE QKV 2 side streams", len(qkv.streams) == 2, f"{len(qkv.streams)}")


def test_tokenloop_device_qkv_two_streams() -> None:
    model, _ = _bound(qwen2_model())
    loop = _cpu_loop(model, overlap=True)
    qkv = loop._groups[0]
    want = 2 if loop.overlap else 0
    check(
        "TokenLoop all-DEVICE QKV streams",
        len(qkv.streams) == want,
        f"overlap={loop.overlap} streams={len(qkv.streams)}",
    )
    check("no ring when no slots", loop._ring is None, "")


def test_tokenloop_host_down_weight_bytes_and_graphs() -> None:
    model, plan = _bound(qwen2_model())
    down = model.get_submodule(plan.layers[0].gemms["down"])
    img = _host_image(down.M, down.K)
    down.attach_host(img)
    slots = SlotPair(img.nbytes, "cpu")
    model.deepfold_slots = slots
    loop = _cpu_loop(model)
    hostish = [g for grp in loop._groups for g in grp.gemms if g.home == "host"]
    check("TokenLoop saw HOST down", len(hostish) >= 1, f"n={len(hostish)}")
    check("down group streams=()", loop._groups[3].streams == (), "")
    check("CopyRing from deepfold_slots", loop._ring is not None and loop._ring.slots is slots, "")
    nbytes = loop.weight_bytes
    check("weight_bytes skips HOST (no crash)", nbytes >= 0, f"{nbytes}")
    mode = loop.capture_graphs()
    # CPU seats: capture returns off ("no CUDA device"), never the old "overflow" refuse.
    check("capture_graphs not overflow-off", loop.graph_error != "overflow", str(loop.graph_error))
    check("cpu seats stay eager", mode == "off", mode)

    seen: list = []

    def fake_capture(groups, *, warmup=3):
        seen.append(tuple(groups))
        n_dev = sum(1 for g in groups if group_is_resident(g))
        return list(groups), ("linears" if n_dev else "off"), None

    orig = generate_mod.capture
    generate_mod.capture = fake_capture
    try:
        loop.drop_graphs()
        mixed_mode = loop.capture_graphs()
    finally:
        generate_mod.capture = orig
    check("mixed mock graph_mode linears", mixed_mode == "linears", mixed_mode)
    check("mixed mock graph_error None", loop.graph_error is None, str(loop.graph_error))
    check("capture received every group", bool(seen) and len(seen[0]) == len(loop._groups), "")
    check(
        "capture args include HOST",
        bool(seen) and any(not group_is_resident(g) for g in seen[0]),
        "",
    )


def test_mixed_copy_only_host_member() -> None:
    k = 64
    q = _dev_gemm("q", 16, k)
    host, img = _host_gemm("k", 8, k, fill=9)
    v = _dev_gemm("v", 8, k)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    grp = GemmGroup("qkv", [q, host, v], (object(), object()))
    check("mixed streams=()", grp.streams == (), "")
    launched: list[int] = []
    orig = _fake_nf4(launched)
    try:
        x = torch.zeros(2, k, dtype=torch.bfloat16)
        grp.run(x, ring=ring)
    finally:
        graph_mod.nf4_gemm = orig
    check("mixed issue_count 1", ring.issue_count == 1, f"{ring.issue_count}")
    check("mixed one H2D", sum(1 for op in ring.ops if op[0] == "h2d") == 1, str(ring.ops))
    check("launch count 3 (q,k,v)", len(launched) == 3, f"{len(launched)}")
    check(
        "HOST launch ptr is slot",
        launched[1] == ring.slots.arena[0].data_ptr(),
        hex(launched[1]) if launched else "none",
    )
    check(
        "HOST launch ptr != host arena",
        launched[1] != img.arena.data_ptr(),
        "",
    )
    check("DEVICE q launch is packed", launched[0] == q.packed.data_ptr(), "")


def test_n32_one_forward_one_copy_per_matrix() -> None:
    host, img = _host_gemm("down", 8, 64, fill=4)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    grp = GemmGroup("down", [host])
    grp.ring = ring
    launched: list[int] = []
    orig = _fake_nf4(launched)
    try:
        ring.arm([host])
        ring.prefetch()
        x = torch.zeros(32, 64, dtype=torch.bfloat16)
        grp.run(x)
    finally:
        graph_mod.nf4_gemm = orig
    check("N=32 issue_count 1", ring.issue_count == 1, f"{ring.issue_count}")
    check("N=32 one launch", len(launched) == 1, f"{len(launched)}")
    check(
        "N=32 launch ptr is slot",
        launched[0] == ring.slots.arena[0].data_ptr(),
        hex(launched[0]) if launched else "none",
    )
    check("N=32 launch ptr != host", launched[0] != img.arena.data_ptr(), "")


def test_prefetch_next_arm_does_not_double_h2d() -> None:
    """H2-9: lm_head prefetch of tape[0]; next arm+bind must not copy again."""
    gemms = [_host_gemm(n, 4, 64, fill=i + 1)[0] for i, n in enumerate("abc")]
    ring = CopyRing(SlotPair(gemms[0].host_image.nbytes, "cpu"))
    ring.arm(gemms)
    ring.prefetch()
    for g in gemms:
        ring.bind_for_gemm(g)
        ring.record_gemm(g)
    check("tape consumed ahead 0", ring.ahead == 0, f"{ring.ahead}")
    copies_after_tape = ring.total_copies
    check("three copies after tape", copies_after_tape == 3, f"{copies_after_tape}")
    ring.prefetch_next()
    check("prefetch_next ahead 1", ring.ahead == 1, f"{ring.ahead}")
    check("prefetch_next one extra H2D", ring.total_copies == 4, f"{ring.total_copies}")
    check("prefetch_next first of tape", ring._queue[0][0] is gemms[0], "")
    raised = False
    try:
        ring.issue(gemms[0])
    except RuntimeError as exc:
        raised = "ahead" in str(exc) or "depth is 1" in str(exc)
    check("prefetch_next keeps depth 1", raised, "")
    copies = ring.total_copies
    slot = ring._queue[0][1]
    phase = ring._phase
    ring.arm(gemms)
    check("arm carry keeps ahead 1", ring.ahead == 1, f"{ring.ahead}")
    check("arm carry zeros issue_count", ring.issue_count == 0, f"{ring.issue_count}")
    check("arm carry keeps total_copies", ring.total_copies == copies, f"{ring.total_copies}")
    check("arm carry keeps phase", ring._phase == phase, f"{ring._phase}")
    ring.prefetch()
    check("arm+prefetch no re-H2D of first", ring.total_copies == copies and ring.issue_count == 0, "")
    packed, _ = ring.bind_for_gemm(gemms[0])
    check(
        "bind first issues only the next tape GEMM",
        ring.total_copies == copies + 1 and ring.issue_count == 1,
        f"copies={ring.total_copies} issue={ring.issue_count}",
    )
    check(
        "carried packed is the prefetched slot",
        packed.data_ptr() == ring.slots.arena[slot].data_ptr(),
        "",
    )
    check("carried packed is not host", packed.data_ptr() != gemms[0].host_image.arena.data_ptr(), "")
    ring.record_gemm(gemms[0])
    ring.bind_for_gemm(gemms[1])
    ring.record_gemm(gemms[1])
    ring.bind_for_gemm(gemms[2])
    ring.record_gemm(gemms[2])
    check("second forward three HOST GEMMs", ring.total_copies == copies + 2, f"{ring.total_copies}")
    check("ahead 0 after second tape", ring.ahead == 0, f"{ring.ahead}")


def test_prefetch_next_noop_until_tape_done() -> None:
    g0, img = _host_gemm("a", 4, 64, fill=1)
    g1, _ = _host_gemm("b", 4, 64, fill=2)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    ring.arm([g0, g1])
    ring.prefetch_next()
    check("no prefetch_next before tape done", ring.total_copies == 0 and ring.ahead == 0, "")
    ring.prefetch()
    ring.prefetch_next()
    check("no prefetch_next while ahead", ring.total_copies == 1 and ring.ahead == 1, "")
    ring.bind_for_gemm(g0)
    ring.record_gemm(g0)
    ring.prefetch_next()
    check("no prefetch_next with unissued tail", ring.total_copies == 2, f"{ring.total_copies}")
    ring.bind_for_gemm(g1)
    ring.record_gemm(g1)
    ring.prefetch_next()
    check("prefetch_next after tape done", ring.total_copies == 3 and ring.ahead == 1, "")
    ring.arm([g0, g1])
    check("carry when first gemm matches", ring.ahead == 1, f"{ring.ahead}")
    ring.arm([g1, g0])
    check("mismatched first gemm drops carry", ring.ahead == 0, f"{ring.ahead}")
    check("mismatch does not extra-copy", ring.total_copies == 3, f"{ring.total_copies}")


def _bf16_loop_norms(loop: TokenLoop) -> None:
    loop.embed.weight.data = loop.embed.weight.data.to(torch.bfloat16)
    for w in (*loop._norm1, *loop._norm2, loop.final_norm):
        w.data = w.data.to(torch.bfloat16)


def test_tokenloop_dummy_forward_lm_head_prefetch() -> None:
    """Dummy N=1 forward: lm_head prefetch, next arm+bind does not H2D twice."""
    model, plan = _bound(qwen2_model())
    down = model.get_submodule(plan.layers[0].gemms["down"])
    img = _host_image(down.M, down.K)
    down.attach_host(img)
    slots = SlotPair(img.nbytes, "cpu")
    model.deepfold_slots = slots
    loop = _cpu_loop(model)
    _bf16_loop_norms(loop)
    check("dummy loop has ring", loop._ring is not None, "")
    n_host = len(loop._host_tape)
    check("dummy one HOST down on tape", n_host == 1, f"n={n_host}")
    first = loop._host_tape[0]
    slot_ptrs = {loop._ring.slots.arena[0].data_ptr(), loop._ring.slots.arena[1].data_ptr()}
    host_ptr = img.arena.data_ptr()

    launched: list[int] = []
    orig = _fake_nf4(launched)
    try:
        ids = torch.tensor([1], dtype=torch.long)
        loop.forward(ids, 0, logits=False)
        ring = loop._ring
        check(
            "logits=False skips lm_head prefetch",
            ring.ahead == 0 and ring.total_copies == n_host,
            f"ahead={ring.ahead} copies={ring.total_copies}",
        )
        loop._prefetch_next_token(2)
        check("N!=1 prefetch_next is no-op", ring.total_copies == n_host, f"{ring.total_copies}")

        loop.reset()
        launched.clear()
        fill_at = ring.total_copies
        loop.forward(ids, 0)
        copies = ring.total_copies
        check(
            "lm_head prefetch after dummy forward",
            ring.ahead == 1 and copies == fill_at + n_host + 1,
            f"ahead={ring.ahead} copies={copies} n_host={n_host}",
        )
        check("queue holds first HOST gemm", ring._queue[0][0].name == first.name, "")
        check("no host arena in nf4 launches", host_ptr not in launched, str(launched[:8]))

        loop.reset()
        launched.clear()
        before = ring.total_copies
        loop.forward(torch.tensor([2], dtype=torch.long), 0)
        check(
            "second dummy forward bytes/token = n_host copies",
            ring.total_copies == before + n_host,
            f"delta={ring.total_copies - before} n_host={n_host}",
        )
        check("second forward still ahead 1", ring.ahead == 1, f"{ring.ahead}")
        check("second forward host packed unused", host_ptr not in launched, "")

        copies = ring.total_copies
        ring.arm(loop._host_tape)
        ring.prefetch()
        packed, _ = ring.bind_for_gemm(first)
        check(
            "next arm+bind no double H2D",
            ring.total_copies == copies and ring.issue_count == 0,
            f"copies={ring.total_copies} issue={ring.issue_count}",
        )
        check("bind packed is slot", packed.data_ptr() in slot_ptrs, "")
        check("bind packed not host", packed.data_ptr() != host_ptr, "")
        ring.record_gemm(first)
    finally:
        graph_mod.nf4_gemm = orig


def test_bind_hold_one_issue_three_gemms() -> None:
    """Prefill hold: one H2D, three GEMMs on the slot, one record; copies==1."""
    import inspect

    src = inspect.getsource(CopyRing.bind_hold)
    rel = inspect.getsource(CopyRing.release_hold)
    check("bind_hold has synchronize()", "synchronize()" in src, "")
    check("bind_hold does not prefetch()", "self.prefetch()" not in src, "")
    check("release_hold prefetches", "self.prefetch()" in rel, "")
    check("PREFILL_HOLD_SUPERCHUNK is 256", PREFILL_HOLD_SUPERCHUNK == 256, str(PREFILL_HOLD_SUPERCHUNK))

    g, img = _host_gemm("down", 8, 64, fill=1)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    ring.arm([g])
    ring.prefetch()
    check("hold setup one issue", ring.total_copies == 1 and ring.ahead == 1, f"c={ring.total_copies}")
    packed, _ = ring.bind_hold(g)
    check("hold no extra copy on bind", ring.total_copies == 1, f"{ring.total_copies}")
    check("hold ahead 0 (no prefetch)", ring.ahead == 0, f"{ring.ahead}")
    check("hold packed is slot", packed.data_ptr() == ring.slots.arena[0].data_ptr(), "")
    check("hold packed is not host", packed.data_ptr() != img.arena.data_ptr(), "")
    check("hold keeps active slot", ring._active_slot == 0, str(ring._active_slot))
    for _ in range(3):
        ring.gemm_hold()
    kinds = [op[0] for op in ring.ops]
    check("hold three gemms no record yet", kinds.count("record_gemm") == 0, str(kinds))
    check("hold copies still 1 during GEMMs", ring.total_copies == 1, f"{ring.total_copies}")
    ring.release_hold(g)
    kinds = [op[0] for op in ring.ops]
    check("hold one record", kinds.count("record_gemm") == 1, str(kinds))
    check("hold copies==1 after release", ring.total_copies == 1, f"{ring.total_copies}")
    check("hold inactive after release", ring._active_slot is None, str(ring._active_slot))
    check("hold tape done ahead 0", ring.ahead == 0, f"{ring.ahead}")


def test_bind_during_hold_raises() -> None:
    g0, img = _host_gemm("a", 4, 64, fill=1)
    g1, _ = _host_gemm("b", 4, 64, fill=2)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    ring.arm([g0, g1])
    ring.prefetch()
    ring.bind_hold(g0)
    raised_hold = raised_bind = False
    try:
        ring.bind_hold(g1)
    except RuntimeError as exc:
        raised_hold = "hold already active" in str(exc) or "not record" in str(exc)
    try:
        ring.bind_for_gemm(g1)
    except RuntimeError as exc:
        raised_bind = "not record_gemm" in str(exc) or "hold already" in str(exc)
    check("second bind_hold during hold raises", raised_hold, "")
    check("bind_for_gemm during hold raises", raised_bind, "")
    check("hold copies still 1", ring.total_copies == 1, f"{ring.total_copies}")
    ring.release_hold(g0)
    check("release_hold prefetches next", ring.total_copies == 2 and ring.ahead == 1, f"c={ring.total_copies}")
    ring.bind_hold(g1)
    ring.gemm_hold()
    ring.release_hold(g1)
    check("second hold no extra issue", ring.total_copies == 2, f"{ring.total_copies}")


def test_decode_bind_for_gemm_still_works() -> None:
    """N=1 decode path stays bind_for_gemm + record_gemm (prefetch next)."""
    g0, img = _host_gemm("a", 4, 64, fill=1)
    g1, _ = _host_gemm("b", 4, 64, fill=2)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    ring.arm([g0, g1])
    ring.prefetch()
    ring.bind_for_gemm(g0)
    kinds = [op[0] for op in ring.ops]
    check("decode bind still prefetches next", kinds.count("h2d") == 2, str(kinds))
    check("decode record not yet", "record_gemm" not in kinds, str(kinds))
    ring.record_gemm(g0)
    ring.bind_for_gemm(g1)
    ring.record_gemm(g1)
    check("decode two copies", ring.total_copies == 2, f"{ring.total_copies}")
    check("decode ahead 0 after tape", ring.ahead == 0, f"{ring.ahead}")


def test_tokenloop_prefill_hold_vs_chunk_copies() -> None:
    """T=64, chunk=32, one HOST down: chunk → 2 copies; hold → 1."""
    model, plan = _bound(qwen2_model())
    down = model.get_submodule(plan.layers[0].gemms["down"])
    img = _host_image(down.M, down.K)
    down.attach_host(img)
    slots = SlotPair(img.nbytes, "cpu")
    model.deepfold_slots = slots

    def _make(mode: str) -> TokenLoop:
        original = generate_mod.linear_max_n
        generate_mod.linear_max_n = lambda codec="nf4", probe=16: 32
        try:
            loop = TokenLoop(model, max_seq=64, overlap=False, prefill_mode=mode)
        finally:
            generate_mod.linear_max_n = original
        _bf16_loop_norms(loop)
        loop.embed.weight.data = loop.embed.weight.data.to(torch.bfloat16)
        loop.prefill_chunk = 32
        return loop

    launched: list[int] = []
    orig = _fake_nf4(launched)
    orig_attend = TokenLoop._attend

    def _zeros_attend(self, q, k, v, n, mask):  # noqa: ARG001
        return torch.zeros(n, self.q_dim, dtype=q.dtype, device=q.device)

    TokenLoop._attend = _zeros_attend
    try:
        ids = torch.arange(64, dtype=torch.long) % 40
        loop_c = _make("chunk")
        check("default-like chunk mode", loop_c.prefill_mode == "chunk", loop_c.prefill_mode)
        check("chunk width 32", loop_c.prefill_chunk == 32, str(loop_c.prefill_chunk))
        loop_c.prefill(ids)
        copies_c = loop_c._ring.total_copies
        loop_h = _make("hold")
        check("hold mode flag", loop_h.prefill_mode == "hold", loop_h.prefill_mode)
        loop_h.prefill(ids)
        copies_h = loop_h._ring.total_copies
        check("chunk prefill 2 copies (two N=32 forwards)", copies_c == 2, f"{copies_c}")
        check("hold prefill 1 copy (one superchunk)", copies_h == 1, f"{copies_h}")
        check("hold one forward", loop_h._ring.total_forwards == 1, f"{loop_h._ring.total_forwards}")
        check("chunk two forwards", loop_c._ring.total_forwards == 2, f"{loop_c._ring.total_forwards}")
    finally:
        graph_mod.nf4_gemm = orig
        TokenLoop._attend = orig_attend


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #


def _smi_used_mib() -> int | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _gpu_skip(label: str) -> bool:
    if not torch.cuda.is_available():
        print(f"  SKIP  {label}  -- no CUDA")
        return True
    used = _smi_used_mib()
    if used is not None and used > 4000:
        print(f"  SKIP  {label}  -- nvidia-smi used {used} MiB > 4000")
        return True
    return False


def test_gpu_overflow_down_vs_resident() -> None:
    """One overflow down_proj through CopyRing vs nf4_gemm of a resident copy."""
    if not CHR_3B.is_file():
        print("  SKIP  gpu down vs resident  -- no qwen25-3b.nf4.chr")
        return
    if _gpu_skip("gpu down vs resident"):
        return

    from gpu.chr0 import materialize_nf4
    from gpu.nf4 import nf4_gemm

    mat = materialize_nf4(str(CHR_3B), DOWN_NAME, "cpu")
    image = HostImage.from_blobs(mat.packed, mat.scale, mat.K)
    pinned = maybe_pin(image.arena)
    if pinned is not image.arena:
        image = HostImage(pinned, image.M, image.K, image.K_pad)

    g = Gemm(
        name="down",
        M=image.M,
        K=image.K,
        K_pad=image.K_pad,
        bias=None,
        packed=torch.empty(0, dtype=torch.uint8, device="cuda"),
        scale=torch.empty(0, dtype=torch.float16, device="cuda"),
        host_image=image,
        home="host",
    )
    slots = SlotPair(image.nbytes, "cuda")
    ring = CopyRing(slots)
    torch.manual_seed(0)
    x = torch.randn(image.K, 1, dtype=torch.bfloat16, device="cuda")
    ring.issue(g)
    packed_s, scale_s = ring.bind_for_gemm(g)
    check(
        "slot packed cuda ptr != host",
        packed_s.is_cuda and packed_s.data_ptr() != image.arena.data_ptr(),
        f"dev={packed_s.device}",
    )
    y_slot = nf4_gemm(packed_s, scale_s, x, g.M, g.K, g.K_pad)
    ring.record_gemm(g)
    packed_d = image.packed.to("cuda")
    scale_d = image.scale.to("cuda")
    y_res = nf4_gemm(packed_d, scale_d, x, g.M, g.K, g.K_pad)
    torch.cuda.synchronize()
    maxabs = (y_slot.float() - y_res.float()).abs().max().item()
    check(
        "overflow vs resident maxabs <= 0.05",
        maxabs <= KERNEL_ABS_LIMIT,
        f"maxabs={maxabs:.5g}",
    )
    del packed_d, scale_d, y_slot, y_res, slots, ring, x
    torch.cuda.empty_cache()


def test_gpu_tokenloop_overflow_3b() -> None:
    if not CHR_3B.is_file() or not MODEL_3B.is_dir():
        print("  SKIP  gpu TokenLoop overflow 3B  -- no 3B chr/config")
        return
    if _gpu_skip("gpu TokenLoop overflow 3B"):
        return

    from gpu.chr0.header import load_header
    from gpu.host.model import load_model
    from gpu.host.residency import descs_from_header, pin_nbytes

    hdr = load_header(str(CHR_3B))
    descs = descs_from_header(hdr)
    pin = pin_nbytes(descs)
    gate_n = next(d.nbytes for d in descs if d.kind == "gate")
    cap = pin + 4 * (2 * gate_n)

    model, report = load_model(str(MODEL_3B), str(CHR_3B), max_resident_bytes=cap)
    try:
        check("load overflow", report.overflow is True, str(report))
        check(
            "deepfold_slots set",
            getattr(model, "deepfold_slots", None) is report.slots,
            "",
        )
        loop = TokenLoop(model, max_seq=64, overlap=True)
        check("TokenLoop has ring", loop._ring is not None, "")
        check("compute stream (not default)", loop._compute is not None, "")
        loop.warmup(prompt=4, tokens=2)
        mode = loop.capture_graphs()
        check("mixed graph_mode linears", mode == "linears", f"{mode} err={loop.graph_error}")
        check("graph_error not overflow", loop.graph_error != "overflow", str(loop.graph_error))
        check("graph_error None", loop.graph_error is None, str(loop.graph_error))
        graphed = eager_host = mismatch = 0
        for li, runners in enumerate(loop._layer):
            orig = loop._groups[4 * li : 4 * li + 4]
            for runner, eager in zip(runners, orig):
                host = any(m.home == "host" for m in eager.gemms)
                if host:
                    eager_host += 1
                    if type(runner) is not GemmGroup:
                        mismatch += 1
                else:
                    graphed += 1
                    if not isinstance(runner, GraphedGemmGroup):
                        mismatch += 1
        check(
            "DEVICE groups graphed, HOST eager",
            mismatch == 0 and graphed > 0 and eager_host > 0,
            f"graphed={graphed} host_eager={eager_host} mismatch={mismatch}",
        )
        check(
            "lm_head DEVICE graphed",
            isinstance(loop._head, GraphedGemmGroup),
            type(loop._head).__name__,
        )
        qkv0, _, _, down0 = loop._layer[0]
        check("L0 qkv graphed", isinstance(qkv0, GraphedGemmGroup), type(qkv0).__name__)
        check("L0 down eager", type(down0) is GemmGroup, type(down0).__name__)

        loop.reset()
        ids = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.long)
        logits = loop.prefill(ids)
        check("prefill logits finite", bool(torch.isfinite(logits).all()), "")
        smi: list[int] = []
        sample = _smi_used_mib()
        if sample is not None:
            smi.append(sample)
        for i in range(4):
            logits = loop.step(int(logits.argmax()))
            check(f"step {i} logits finite", bool(torch.isfinite(logits).all()), "")
            sample = _smi_used_mib()
            if sample is not None:
                smi.append(sample)
        if len(smi) >= 2:
            drift = max(smi) - min(smi)
            check(
                "smi not growing per step",
                drift < 256,
                f"smi={smi} drift={drift} MiB",
            )
        else:
            print("  SKIP  smi drift  -- nvidia-smi unavailable")

        print("  (optional Paris: generating 16 greedy tokens on overflow loop)")
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(MODEL_3B), local_files_only=True)
            prompt = tok("The capital of France is", return_tensors="pt").input_ids[0].to(
                "cuda"
            )
            loop.reset()
            gen = loop.generate(prompt, max_new_tokens=16)
            text = tok.decode(gen.tokens, skip_special_tokens=True)
            hit = "Paris" in text or "paris" in text.lower()
            if hit:
                check("optional Paris in overflow greedy", True, text[:80])
            else:
                print(f"  NOTE  overflow greedy 16 tok, no Paris yet: {text!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP  optional Paris  -- {type(exc).__name__}: {exc}")
    finally:
        del model, report
        torch.cuda.empty_cache()


def test_gpu_tokenloop_resident_3b_graphs() -> None:
    """Fully resident 3B: graph_mode linears (H2-4/plan A, no regression)."""
    if not CHR_3B.is_file() or not MODEL_3B.is_dir():
        print("  SKIP  gpu TokenLoop resident 3B graphs  -- no 3B chr/config")
        return
    if _gpu_skip("gpu TokenLoop resident 3B graphs"):
        return

    from gpu.host.model import load_model

    model, report = load_model(str(MODEL_3B), str(CHR_3B))
    try:
        check("resident load no overflow", report.overflow is False, str(report))
        loop = TokenLoop(model, max_seq=64, overlap=True)
        loop.warmup(prompt=4, tokens=2)
        mode = loop.capture_graphs()
        check("resident graph_mode linears", mode == "linears", f"{mode} err={loop.graph_error}")
        check(
            "resident L0 qkv graphed",
            isinstance(loop._layer[0][0], GraphedGemmGroup),
            type(loop._layer[0][0]).__name__,
        )
        check(
            "resident L0 down graphed",
            isinstance(loop._layer[0][3], GraphedGemmGroup),
            type(loop._layer[0][3]).__name__,
        )
        loop.reset()
        ids = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.long)
        logits = loop.prefill(ids)
        check("resident prefill finite", bool(torch.isfinite(logits).all()), "")
        logits = loop.step(int(logits.argmax()))
        check("resident step finite", bool(torch.isfinite(logits).all()), "")
    finally:
        del model, report
        torch.cuda.empty_cache()


def _pin_host_gemm(name: str, m: int, k: int, fill: int = 1) -> tuple[Gemm, HostImage]:
    g, img = _host_gemm(name, m, k, fill=fill)
    pinned = maybe_pin(img.arena)
    if pinned is not img.arena:
        img = HostImage(pinned, img.M, img.K, img.K_pad)
        g = Gemm(
            name=name,
            M=img.M,
            K=img.K,
            K_pad=img.K_pad,
            bias=None,
            packed=g.packed,
            scale=g.scale,
            host_image=img,
            home="host",
        )
    return g, img


def test_gpu_bind_join_prefetches_and_skips_elapsed() -> None:
    """CUDA product bind: timing e_copy, profile join, prefetch next, no elapsed_time."""
    if not torch.cuda.is_available():
        print("  SKIP  gpu bind join prefetch  -- no CUDA")
        return
    g0, img0 = _pin_host_gemm("a", 64, 256, fill=1)
    g1, _ = _pin_host_gemm("b", 64, 256, fill=2)
    slots = SlotPair(img0.nbytes, "cuda")
    ring = CopyRing(slots, timing=False)
    check("gpu product timing False", ring.timing is False, "")
    check("gpu product join_copy follows profile", ring._join_copy is default_join_copy(), str(ring._join_copy))
    check("gpu no h2d start events", ring._e_h2d_start == (None, None), "")
    ring.arm([g0, g1])
    ring.prefetch()
    ring.bind_for_gemm(g0)
    kinds = [op[0] for op in ring.ops]
    check("gpu two H2D before GEMM", kinds.count("h2d") == 2, str(kinds))
    check("gpu record_gemm not yet", "record_gemm" not in kinds, str(kinds))
    wait_i = kinds.index("wait_copy")
    check("gpu next h2d after wait_copy", "h2d" in kinds[wait_i + 1 :], str(kinds))
    snap = ring.snapshot()
    check("gpu snapshot copy_ms None", snap["forward_copy_ms"] is None, str(snap))
    check("gpu snapshot total_copy_ms None", snap["total_copy_ms"] is None, str(snap))
    check("timing=False has no start events", ring._e_h2d_start == (None, None), "")
    ring.record_gemm(g0)
    ring.bind_for_gemm(g1)
    ring.record_gemm(g1)
    torch.cuda.synchronize()
    timed_ok = True
    try:
        _ = float(ring._e_copy[0].elapsed_time(ring._e_copy[1]))
    except Exception as exc:  # noqa: BLE001
        timed_ok = False
        check("product e_copy timing-enabled", False, f"{type(exc).__name__}: {exc}")
    if timed_ok:
        check("product e_copy timing-enabled", True, "")
    del slots, ring
    torch.cuda.empty_cache()


def test_gpu_two_copy_streams_three_slots() -> None:
    """Two copy-stream handles on one H2D engine; three slot events."""
    if not torch.cuda.is_available():
        print("  SKIP  gpu two copy streams  -- no CUDA")
        return
    g0, img0 = _pin_host_gemm("a", 64, 256, fill=1)
    g1, _ = _pin_host_gemm("b", 64, 256, fill=2)
    g2, _ = _pin_host_gemm("c", 64, 256, fill=3)
    slots = SlotPair(img0.nbytes, "cuda", count=3)
    ring = CopyRing(slots, timing=False)
    check("gpu 3-slot max_ahead 2", ring.max_ahead == 2, f"{ring.max_ahead}")
    check("gpu 3-slot two copy streams", ring.n_copy_streams == 2, f"{ring.n_copy_streams}")
    check("gpu two stream objects", len(ring._copy_streams) == 2, f"{len(ring._copy_streams)}")
    check(
        "gpu streams distinct",
        ring._copy_streams[0] is not ring._copy_streams[1],
        "",
    )
    check("copy_stream is first", ring.copy_stream is ring._copy_streams[0], "")
    check(
        "copy not default",
        ring.copy_stream is not torch.cuda.default_stream(),
        "",
    )
    check("three copy events", len(ring._e_copy) == 3, f"{len(ring._e_copy)}")
    ring.arm([g0, g1, g2])
    ring.prefetch()
    check("gpu prefetch ahead 2", ring.ahead == 2, f"{ring.ahead}")
    ring.bind_for_gemm(g0)
    check("gpu bind keeps ahead 2", ring.ahead == 2 and ring.slot_order == [0, 1, 2], str(ring.slot_order))
    ring.record_gemm(g0)
    ring.bind_for_gemm(g1)
    ring.record_gemm(g1)
    ring.bind_for_gemm(g2)
    ring.record_gemm(g2)
    torch.cuda.synchronize()
    check("gpu 3-slot tape 0,1,2", ring.slot_order == [0, 1, 2], str(ring.slot_order))
    del slots, ring
    torch.cuda.empty_cache()


def test_gpu_copy_join_microbench() -> None:
    """Pinned ping-pong: product (timed e_copy + join-after-prefetch) vs join vs timing=True."""
    if not torch.cuda.is_available():
        print("  SKIP  gpu copy join microbench  -- no CUDA")
        return
    # ~16.5 MiB packed‖scale; 32 ping-pong copies + tiny overlapping compute.
    g0, img = _pin_host_gemm("a", 7710, 4096, fill=1)
    g1, _ = _pin_host_gemm("b", 7710, 4096, fill=2)
    check("microbench src pinned", bool(img.arena.is_pinned()), "")
    nbytes = img.nbytes
    check("microbench size 8-72 MiB", 8 * 1024 * 1024 <= nbytes <= 72 * 1024 * 1024, f"{nbytes}")
    try:
        slots = SlotPair(nbytes, "cuda")
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP  gpu copy join microbench  -- SlotPair: {type(exc).__name__}: {exc}")
        return
    tape = [g0, g1] * 16  # 32 copies
    torch.cuda.synchronize()
    dummy = torch.empty(256, 256, device="cuda", dtype=torch.float32)
    compute = torch.cuda.Stream()

    def wall(ring: CopyRing) -> float:
        def once() -> None:
            with torch.cuda.stream(compute):
                ring.arm(tape)
                ring.prefetch()
                for g in tape:
                    ring.bind_for_gemm(g)
                    dummy.mul_(1.0001)
                    ring.record_gemm(g)
            torch.cuda.synchronize()

        once()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        once()
        return time.perf_counter() - t0

    try:
        product = CopyRing(slots, timing=False)
        t_prod = wall(product)
        snap = product.snapshot()
        check("microbench product timing False", snap["timing"] is False, "")
        check("microbench product no elapsed", snap["forward_copy_ms"] is None, str(snap))
        check("microbench prefetch still depth 1", product.ahead <= 1, f"{product.ahead}")

        joined = CopyRing(slots, timing=False)
        joined._join_copy = True
        t_join = wall(joined)

        timed = CopyRing(slots, timing=True)
        t_timed = wall(timed)
        check(
            "microbench timed uses elapsed_time",
            timed.snapshot()["forward_copy_ms"] is not None,
            "",
        )
        gib = (nbytes * len(tape)) / (1024**3)
        print(
            f"  NOTE  copy-join microbench {nbytes / 1024**2:.1f} MiB x {len(tape)}: "
            f"product={t_prod * 1000:.1f} ms ({gib / t_prod:.1f} GiB/s) "
            f"join={t_join * 1000:.1f} ms ({gib / t_join:.1f} GiB/s) "
            f"timing={t_timed * 1000:.1f} ms ({gib / t_timed:.1f} GiB/s)"
        )
        check(
            "product wall ~ timed bind",
            t_prod <= t_timed * 1.35 or t_prod < 0.050,
            f"product={t_prod:.4f}s timed={t_timed:.4f}s",
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print(f"  SKIP  gpu copy join microbench  -- {exc}")
            return
        raise
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print(f"  SKIP  gpu copy join microbench  -- {exc}")
            return
        raise
    finally:
        del slots
        torch.cuda.empty_cache()


TESTS = [
    test_gemm_of_host_does_not_raise_on_empty_packed,
    test_attach_host_pins_pageable_arena,
    test_bind_cpu_join_before_prefetch,
    test_join_copy_env_override,
    test_bind_prefetches_next_before_record_gemm,
    test_timing_false_skips_elapsed_time_fields,
    test_three_overflow_slots_0_1_0,
    test_three_arenas_ahead_2,
    test_prefetch_next_ahead_2_carry_prefix,
    test_tape_prefetch_three_slots,
    test_ring_lifetime_bytes_survive_arm,
    test_group_is_resident_skips_host,
    test_tokenloop_cpu_suffix_no_ring_and_hold_refused,
    test_capture_cpu_does_not_run_host_groups,
    test_any_host_group_streams_empty,
    test_tokenloop_device_qkv_two_streams,
    test_tokenloop_host_down_weight_bytes_and_graphs,
    test_mixed_copy_only_host_member,
    test_n32_one_forward_one_copy_per_matrix,
    test_prefetch_next_arm_does_not_double_h2d,
    test_prefetch_next_noop_until_tape_done,
    test_tokenloop_dummy_forward_lm_head_prefetch,
    test_bind_hold_one_issue_three_gemms,
    test_bind_during_hold_raises,
    test_decode_bind_for_gemm_still_works,
    test_tokenloop_prefill_hold_vs_chunk_copies,
    test_gpu_overflow_down_vs_resident,
    test_gpu_bind_join_prefetches_and_skips_elapsed,
    test_gpu_two_copy_streams_three_slots,
    test_gpu_copy_join_microbench,
    test_gpu_tokenloop_overflow_3b,
    test_gpu_tokenloop_resident_3b_graphs,
]


def main_runner() -> int:
    print(f"gpu/loop CopyRing H2-4 + mixed graphs H2-5 + H2-9 prefetch, {len(TESTS)} tests\n")
    for fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
            text = out.getvalue()
            if text:
                sys.stdout.write(text)
        except Skip as exc:
            SKIPPED.append((fn.__name__, str(exc)))
            print(f"  SKIP  {fn.__name__}  -- {exc}")
        except AssertionError as exc:
            dumped = out.getvalue()
            if dumped:
                sys.stdout.write(dumped)
            check(fn.__name__, False, str(exc).splitlines()[0] if str(exc) else "assert")
        except Exception as exc:  # noqa: BLE001
            dumped = out.getvalue()
            if dumped:
                sys.stdout.write(dumped)
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
        else:
            if not out.getvalue():
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
