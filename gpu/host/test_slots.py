"""CPU: SlotPair arenas + CompressedLinear.attach_host + overflow load. GPU canary optional.

    python gpu/host/test_slots.py
    python -m pytest gpu/host/test_slots.py -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.chr0._fixtures import build_chr, nf4_blob_sizes  # noqa: E402
from gpu.chr0.header import align64, load_header  # noqa: E402
from gpu.host import LoadReport, SlotPair  # noqa: E402
from gpu.host.host_image import HostImage  # noqa: E402
from gpu.host.linear import CompressedLinear  # noqa: E402
from gpu.host.model import (  # noqa: E402
    build_skeleton,
    linear_modules,
    load_chr_nf4,
    replace_linears,
)
from gpu.host.residency import (  # noqa: E402
    MIB,
    descs_from_header,
    nf4_nbytes,
    pin_nbytes,
)

CHECKS: list[tuple[str, bool, str]] = []

Q_NAME = "model.layers.0.self_attn.q_proj"
DOWN_NAME = "model.layers.0.mlp.down_proj"
CHR_3B = Path(r"C:\dev\models\qwen25-3b.nf4.chr")
MODEL_3B = Path(r"C:\dev\models\Qwen2.5-3B-Instruct")
DOWN_3B_NBYTES = nf4_nbytes(2048, 11008)


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _nf4_entry(m: int, k: int, start: int, kind: str, layer: int) -> tuple[dict, int]:
    data_n, scale_n = nf4_blob_sizes(m, k)
    entry = {
        "kind": kind,
        "codec": "nf4",
        "shape": [m, k],
        "group_size": 64,
        "layer": layer,
        "data": [start, start + data_n],
        "scale": [start + data_n, start + data_n + scale_n],
    }
    return entry, start + data_n + scale_n


def _write_toy_chr(path: Path) -> None:
    q, end = _nf4_entry(64, 128, 4096, "q", 0)
    down, _ = _nf4_entry(2, 65, align64(end), "down", 0)
    build_chr(
        path,
        {Q_NAME: q, DOWN_NAME: down},
        hidden_size=128,
        intermediate_size=65,
        num_layers=1,
        vocab_size=0,
    )


def _toy_model() -> nn.Module:
    root = nn.Module()
    attn = nn.Module()
    attn.q_proj = CompressedLinear(128, 64)
    mlp = nn.Module()
    mlp.down_proj = CompressedLinear(65, 2)
    layer = nn.Module()
    layer.self_attn = attn
    layer.mlp = mlp
    inner = nn.Module()
    inner.layers = nn.ModuleList([layer])
    root.model = inner
    return root


def _host_image(m: int, k: int) -> HostImage:
    kpad = 64 * ((k + 63) // 64)
    packed = torch.arange(m * (kpad // 2), dtype=torch.uint8).view(m, kpad // 2)
    scale = (
        torch.arange(m * (kpad // 64), dtype=torch.int32)
        .to(torch.float16)
        .view(m, kpad // 64)
    )
    return HostImage.from_blobs(packed, scale, k)


def test_slotpair_two_ptrs_and_view_roundtrip() -> None:
    img = _host_image(5, 70)
    slots = SlotPair(img.nbytes + 128, "cpu")
    check(
        "two different data_ptr",
        slots.arena[0].data_ptr() != slots.arena[1].data_ptr(),
        "",
    )
    check("arena0 16-align", slots.arena[0].data_ptr() % 16 == 0, hex(slots.arena[0].data_ptr()))
    check("arena1 16-align", slots.arena[1].data_ptr() % 16 == 0, hex(slots.arena[1].data_ptr()))
    p0 = slots.arena[0].data_ptr()
    slots.arena[0][: img.nbytes].copy_(img.arena)
    packed, scale = slots.view(0, img.M, img.K, img.K_pad)
    check("packed roundtrip", torch.equal(packed.cpu(), img.packed), "")
    check("scale roundtrip", torch.equal(scale.cpu(), img.scale), "")
    check("scale dtype fp16", scale.dtype is torch.float16, str(scale.dtype))
    check("view packed is arena prefix", packed.data_ptr() == slots.arena[0].data_ptr(), "")
    check("arena ptr stable after view", slots.arena[0].data_ptr() == p0, "")
    packed_n = img.M * (img.K_pad // 2)
    check(
        "slot larger than matrix (tail unread)",
        int(slots.arena[0].numel()) > packed_n,
        f"{int(slots.arena[0].numel())} vs packed {packed_n}",
    )


def test_attach_host_empty_packed_forward_raises() -> None:
    m, k = 8, 32
    lin = CompressedLinear(k, m)
    img = _host_image(m, k)
    lin.attach_host(img)
    check("host is_loaded", lin.is_loaded is True, "")
    check("host nbytes 0", lin.nbytes == 0, f"{lin.nbytes}")
    check("packed numel 0", int(lin.packed.numel()) == 0, "")
    check("scale numel 0", int(lin.scale.numel()) == 0, "")
    check("home host", lin.home == "host", lin.home)
    check("weight 0-numel", int(lin.weight.numel()) == 0, str(tuple(lin.weight.shape)))
    mk = False
    for buf in lin.buffers():
        if buf is not None and tuple(buf.shape) == (m, k):
            mk = True
    check("no [M,K] buffer", not mk, "")
    raised = False
    msg = ""
    try:
        lin(torch.zeros(1, k, dtype=torch.bfloat16))
    except RuntimeError as exc:
        msg = str(exc)
        raised = "host-resident" in msg or "CopyRing" in msg
        check("no silent H2D wording", "to('cuda')" not in msg.lower(), msg[:120])
    check("forward raises", raised, msg[:120])


def test_attach_device_is_loaded_nbytes() -> None:
    m, k = 8, 32
    lin = CompressedLinear(k, m)
    kpad = lin.K_pad
    packed = torch.zeros(m, kpad // 2, dtype=torch.uint8)
    scale = torch.zeros(m, kpad // 64, dtype=torch.float16)
    mat = SimpleNamespace(name="toy.q", M=m, K=k, K_pad=kpad, packed=packed, scale=scale)
    lin.attach(mat)
    want = int(packed.numel()) + int(scale.numel()) * 2
    check("device is_loaded", lin.is_loaded is True, "")
    check("device home", lin.home == "device", lin.home)
    check("device nbytes", lin.nbytes == want, f"{lin.nbytes} vs {want}")
    check("packed live", int(lin.packed.numel()) > 0, "")
    check("host_image cleared", lin.host_image is None, "")
    check("weight still 0", int(lin.weight.numel()) == 0, "")


def test_loadreport_default_slots_none() -> None:
    report = LoadReport()
    check("default overflow False", report.overflow is False, "")
    check("default slots is None", report.slots is None, "")
    check("default streamed 0", report.streamed == 0, "")
    check("default slot_nbytes 0", report.slot_nbytes == 0, "")
    check("str hides overflow when False", "overflow" not in str(report), str(report))
    shown = LoadReport(overflow=True, streamed=3, streamed_bytes=2 * MIB, slot_nbytes=MIB)
    check("str shows overflow when True", "overflow" in str(shown), str(shown))


def test_load_chr_nf4_toy_q_device_down_host() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "toy.nf4.chr"
        _write_toy_chr(path)
        hdr = load_header(str(path))
        descs = descs_from_header(hdr)
        check("descs count 2", len(descs) == 2, f"{len(descs)}")
        check(
            "CHR0 names",
            {d.name for d in descs} == {Q_NAME, DOWN_NAME},
            str(sorted(d.name for d in descs)),
        )
        check("kinds from TensorInfo", {d.kind for d in descs} == {"q", "down"}, "")
        q_n = next(d.nbytes for d in descs if d.kind == "q")
        down_n = next(d.nbytes for d in descs if d.kind == "down")
        q_info = hdr.tensors[Q_NAME]
        blob_q = q_info.blobs["data"].nbytes + q_info.blobs["scale"].nbytes
        check("q nbytes matches blobs", q_n == blob_q == nf4_nbytes(64, 128), f"{q_n}")

        model = _toy_model()
        report = load_chr_nf4(model, str(path), device="cpu")
        check("default overflow False", report.overflow is False, str(report))
        check("default slots is None", report.slots is None, "")
        q = model.model.layers[0].self_attn.q_proj
        down = model.model.layers[0].mlp.down_proj
        check("default q DEVICE", q.home == "device" and int(q.packed.numel()) > 0, q.home)
        check("default down DEVICE", down.home == "device" and int(down.packed.numel()) > 0, down.home)
        check("default linear_bytes both", report.linear_bytes == q_n + down_n, f"{report.linear_bytes}")

        model2 = _toy_model()
        report2 = load_chr_nf4(model2, str(path), device="cpu", max_resident_bytes=q_n)
        q2 = model2.model.layers[0].self_attn.q_proj
        d2 = model2.model.layers[0].mlp.down_proj
        check("overflow True", report2.overflow is True, str(report2))
        check("q DEVICE under cap", q2.home == "device" and int(q2.packed.numel()) > 0, q2.home)
        check(
            "down HOST",
            d2.home == "host" and int(d2.packed.numel()) == 0 and d2.host_image is not None,
            f"home={d2.home} packed={int(d2.packed.numel())}",
        )
        check("HOST nbytes 0", d2.nbytes == 0, f"{d2.nbytes}")
        check("linear_bytes is DEVICE q only", report2.linear_bytes == q_n, f"{report2.linear_bytes}")
        check("streamed 1", report2.streamed == 1, f"{report2.streamed}")
        check("streamed_bytes down", report2.streamed_bytes == down_n, f"{report2.streamed_bytes}")
        check("slot_nbytes down", report2.slot_nbytes == down_n, f"{report2.slot_nbytes}")
        check("slots allocated first", report2.slots is not None, "")
        if report2.slots is not None:
            check("slot pair nbytes", report2.slots.nbytes == down_n, f"{report2.slots.nbytes}")
            check(
                "two slot ptrs",
                report2.slots.arena[0].data_ptr() != report2.slots.arena[1].data_ptr(),
                "",
            )


def test_gpu_3b_overflow_canary() -> None:
    """Skip without the 3B .chr or CUDA. Does not load a BF16 embed table."""
    if not CHR_3B.is_file():
        print("  SKIP  gpu 3B overflow canary  -- no C:\\dev\\models\\qwen25-3b.nf4.chr")
        return
    if not torch.cuda.is_available():
        print("  SKIP  gpu 3B overflow canary  -- torch.cuda.is_available() is False")
        return
    if not MODEL_3B.is_dir():
        print("  SKIP  gpu 3B overflow canary  -- no Qwen2.5-3B-Instruct config dir")
        return

    hdr = load_header(str(CHR_3B))
    descs = descs_from_header(hdr)
    pin = pin_nbytes(descs)
    gate_n = next(d.nbytes for d in descs if d.kind == "gate")
    cap = pin + 4 * (2 * gate_n)
    full_nf4 = sum(d.nbytes for d in descs)
    linear_nf4 = sum(d.nbytes for d in descs if d.kind != "embed")

    model = build_skeleton(str(MODEL_3B))
    replace_linears(model)
    report = load_chr_nf4(
        model,
        str(CHR_3B),
        device="cuda",
        embed="rows",
        max_resident_bytes=cap,
    )
    try:
        mods = linear_modules(model)
        downs = [(n, m) for n, m in mods.items() if n.endswith("down_proj")]
        q0 = mods[Q_NAME]
        d0 = mods[DOWN_NAME]
        check("3B overflow", report.overflow is True, str(report))
        check(
            "some down HOST",
            int(d0.packed.numel()) == 0 and d0.host_image is not None,
            f"packed={int(d0.packed.numel())} home={d0.home}",
        )
        check(
            "all down HOST",
            all(int(m.packed.numel()) == 0 and m.host_image is not None for _, m in downs),
            f"n={len(downs)}",
        )
        check(
            "q_proj packed cuda",
            q0.packed.is_cuda and int(q0.packed.numel()) > 0,
            f"dev={q0.packed.device} n={int(q0.packed.numel())}",
        )
        check(
            "linear_bytes < full NF4",
            report.linear_bytes < full_nf4,
            f"{report.linear_bytes} vs {full_nf4}",
        )
        check(
            "linear_bytes < all linear NF4",
            report.linear_bytes < linear_nf4,
            f"{report.linear_bytes} vs {linear_nf4}",
        )
        check("slots present", report.slots is not None, "")
        if report.slots is not None:
            check(
                "SlotPair on cuda",
                report.slots.device.type == "cuda" and report.slots.arena[0].is_cuda,
                str(report.slots.device),
            )
            check(
                "slot_nbytes == 3B down",
                report.slot_nbytes == DOWN_3B_NBYTES == report.slots.nbytes,
                f"{report.slot_nbytes} vs {DOWN_3B_NBYTES}",
            )
            check(
                "slot ~11.42 MiB",
                abs(report.slot_nbytes / MIB - 11.421875) < 1e-6,
                f"{report.slot_nbytes / MIB}",
            )
        check("embed not bf16 table", report.embed_mode != "nf4-dequant-table", report.embed_mode)
    finally:
        del model, report
        torch.cuda.empty_cache()


TESTS = [
    test_slotpair_two_ptrs_and_view_roundtrip,
    test_attach_host_empty_packed_forward_raises,
    test_attach_device_is_loaded_nbytes,
    test_loadreport_default_slots_none,
    test_load_chr_nf4_toy_q_device_down_host,
    test_gpu_3b_overflow_canary,
]


def main() -> int:
    print("gpu/host SlotPair + attach_host + overflow load\n")
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
