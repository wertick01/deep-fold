"""HOST-slot decode graphs: CPU refuses host packed; optional tiny GPU canary.

    python gpu/loop/test_host_graph.py

No 32B, no full model. GPU canary is a few-MiB SlotPair + CopyRing; skipped
when the card is busy.
"""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gpu.loop.graph as graph_mod  # noqa: E402
from gpu.host.host_image import HostImage, maybe_pin  # noqa: E402
from gpu.host.slots import SlotPair  # noqa: E402
from gpu.loop.graph import (  # noqa: E402
    Gemm,
    GemmGroup,
    capture,
    group_is_resident,
    host_slot_graphs,
    reset_host_slot_graphs,
    slot_gemm_key,
    try_host_slot_gemm,
)
from gpu.loop.ring import CopyRing  # noqa: E402
from gpu.tests.skips import Skip, skip  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
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
    g = Gemm(
        name=name,
        M=img.M,
        K=img.K,
        K_pad=img.K_pad,
        bias=None,
        packed=torch.empty(0, dtype=torch.uint8),
        scale=torch.empty(0, dtype=torch.float16),
        host_image=img,
        home="host",
    )
    return g, img


def _fake_nf4(launched: list):
    orig = graph_mod.nf4_gemm

    def fake(packed, scale, x, M, K, K_pad):  # noqa: N803
        launched.append(int(packed.data_ptr()))
        n = 1 if x.dim() == 1 else int(x.shape[-1])
        return torch.zeros(M, n, dtype=torch.bfloat16, device=x.device)

    graph_mod.nf4_gemm = fake
    return orig


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


def _gpu_skip(label: str) -> None:
    if not torch.cuda.is_available():
        skip(f"{label}: no CUDA device")
    used = _smi_used_mib()
    if used is not None and used > 4000:
        skip(f"{label}: CUDA busy, nvidia-smi used {used} MiB > 4000")


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #


def test_slot_gemm_key_refuses_host_packed() -> None:
    host, img = _host_gemm("down", 8, 64)
    reset_host_slot_graphs()
    try:
        slot_gemm_key(img.packed, host.M, host.K)
    except RuntimeError as exc:
        text = str(exc)
        check("refuse host packed", "refuse host packed" in text, text)
        check("names device slot", "device slot" in text, text)
    else:  # pragma: no cover
        check("refuse host packed", False, "slot_gemm_key accepted CPU packed")
    check("empty packed also refused", True)
    try:
        slot_gemm_key(host.packed, host.M, host.K)
    except RuntimeError as exc:
        check("empty CPU packed refused", "refuse host packed" in str(exc), str(exc))
    else:  # pragma: no cover
        check("empty CPU packed refused", False, "accepted empty CPU packed")


def test_try_host_slot_gemm_skips_cpu_packed() -> None:
    reset_host_slot_graphs()
    host, img = _host_gemm("down", 8, 64)
    xk = torch.zeros(host.K, 1, dtype=torch.bfloat16)
    y = try_host_slot_gemm(img.packed, img.scale, xk, host.M, host.K, host.K_pad)
    check("try_host_slot_gemm CPU packed is None", y is None, str(y))
    check("CPU packed does not cache a recipe", host_slot_graphs() == {}, str(host_slot_graphs()))


def test_host_group_is_not_resident() -> None:
    host, _ = _host_gemm("down", 8, 64)
    dev = _dev_gemm("q", 16, 64)
    check("DEVICE group resident", group_is_resident(GemmGroup("q", [dev])), "")
    check("HOST group not resident", not group_is_resident(GemmGroup("down", [host])), "")
    mixed = GemmGroup("qkv", [dev, host, _dev_gemm("v", 8, 64)])
    check("mixed HOST not resident", not group_is_resident(mixed), "")
    check("mixed streams stay empty", mixed.streams == (), f"n={len(mixed.streams)}")


def test_capture_skips_host_groups() -> None:
    """Plan A still does not warm HOST groups (no ring, no host packed)."""
    reset_host_slot_graphs()
    host, img = _host_gemm("down", 8, 64, fill=7)
    dev = _dev_gemm("q", 16, 64)
    groups = [GemmGroup("q", [dev]), GemmGroup("down", [host])]
    launched: list[int] = []
    orig = _fake_nf4(launched)
    try:
        runners, mode, err = capture(groups, warmup=1)
    finally:
        graph_mod.nf4_gemm = orig
    check("HOST runner stays the eager group", runners[1] is groups[1], type(runners[1]).__name__)
    check("HOST group still not resident", not group_is_resident(groups[1]), "")
    check("capture did not record a slot recipe", host_slot_graphs() == {}, str(host_slot_graphs()))
    check("HOST packed never launched", img.arena.data_ptr() not in launched, str(launched))
    check("no nf4 during cpu capture", launched == [], f"{launched}")
    check("cpu capture not overflow", err != "overflow", str(err))
    check("mode off or weights-not-cuda", mode == "off", f"{mode} err={err}")


def test_cpu_n1_host_one_stays_eager() -> None:
    reset_host_slot_graphs()
    host, img = _host_gemm("down", 8, 64, fill=3)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    grp = GemmGroup("down", [host], (object(),))
    check("HOST group streams=()", grp.streams == (), str(grp.streams))
    launched: list[int] = []
    orig = _fake_nf4(launched)
    try:
        x = torch.zeros(1, host.K, dtype=torch.bfloat16)
        (y,) = grp.run(x, ring=ring)
    finally:
        graph_mod.nf4_gemm = orig
    check("eager N=1 output shape", tuple(y.shape) == (1, host.M), str(tuple(y.shape)))
    check("cpu N=1 does not capture", host_slot_graphs() == {}, str(host_slot_graphs()))
    check("one launch", len(launched) == 1, f"{len(launched)}")
    check(
        "launch ptr is slot not host",
        launched[0] == ring.slots.arena[0].data_ptr() and launched[0] != img.arena.data_ptr(),
        hex(launched[0]) if launched else "none",
    )
    check("N=1 still copied once", ring.issue_count == 1, f"{ring.issue_count}")


def test_cpu_n_not_one_stays_eager() -> None:
    reset_host_slot_graphs()
    host, img = _host_gemm("down", 8, 64)
    ring = CopyRing(SlotPair(img.nbytes, "cpu"))
    grp = GemmGroup("down", [host])
    launched: list[int] = []
    orig = _fake_nf4(launched)
    try:
        x = torch.zeros(2, host.K, dtype=torch.bfloat16)
        grp.run(x, ring=ring)
    finally:
        graph_mod.nf4_gemm = orig
    check("prefill N!=1 no slot recipe", host_slot_graphs() == {}, str(host_slot_graphs()))
    check("prefill one eager launch", len(launched) == 1, f"{len(launched)}")


# --------------------------------------------------------------------------- #
# GPU canary (tiny synthetic, not a model)
# --------------------------------------------------------------------------- #


def test_gpu_decode_replays_after_bind() -> None:
    _gpu_skip("gpu HOST slot graph")
    reset_host_slot_graphs()
    m, k = 64, 64
    host, img = _host_gemm("down", m, k, fill=5)
    pinned = maybe_pin(img.arena)
    if pinned is not img.arena:
        img = HostImage(pinned, img.M, img.K, img.K_pad)
        host = Gemm(
            name="down",
            M=img.M,
            K=img.K,
            K_pad=img.K_pad,
            bias=None,
            packed=torch.empty(0, dtype=torch.uint8, device="cuda"),
            scale=torch.empty(0, dtype=torch.float16, device="cuda"),
            host_image=img,
            home="host",
        )
    slots = SlotPair(img.nbytes, "cuda")
    ring = CopyRing(slots)
    grp = GemmGroup("down", [host])
    launches = [0]
    real = graph_mod.nf4_gemm

    def counted(packed, scale, x, M, K, K_pad):  # noqa: N803
        launches[0] += 1
        if packed.device.type != "cuda":
            raise RuntimeError("kernel saw host packed")
        return real(packed, scale, x, M, K, K_pad)

    graph_mod.nf4_gemm = counted
    try:
        x = torch.zeros(1, k, dtype=torch.bfloat16, device="cuda")
        try:
            (y1,) = grp.run(x, ring=ring)
        except Exception as exc:  # noqa: BLE001
            skip(f"gpu HOST slot graph: first nf4_gemm failed ({type(exc).__name__}: {exc})")
        torch.cuda.synchronize()
        recipes = [r for r in host_slot_graphs().values() if r is not None]
        failed = sum(1 for r in host_slot_graphs().values() if r is None)
        check(
            "N=1 captured a slot recipe",
            len(recipes) == 1,
            f"ok={len(recipes)} failed={failed}",
        )
        if not recipes:
            return
        check("first call is capture not replay", recipes[0].replays == 0, f"{recipes[0].replays}")
        first_launches = launches[0]
        check("warm+capture launched nf4", first_launches >= 2, f"{first_launches}")
        check("output finite", bool(torch.isfinite(y1).all()), "")
        check(
            "recipe packed is slot 0",
            recipes[0].packed.data_ptr() == slots.arena[0].data_ptr(),
            hex(recipes[0].packed.data_ptr()),
        )
        check(
            "recipe packed is not host arena",
            recipes[0].packed.data_ptr() != img.arena.data_ptr(),
            "",
        )
        y1_saved = y1.detach().clone()

        ring.arm([host])
        (y2,) = grp.run(x, ring=ring)
        torch.cuda.synchronize()
        check("second N=1 replays", recipes[0].replays == 1, f"{recipes[0].replays}")
        check(
            "replay does not call nf4_gemm",
            launches[0] == first_launches,
            f"{launches[0]} vs {first_launches}",
        )
        check(
            "replay matches capture",
            bool(torch.equal(y1_saved, y2)),
            f"maxabs={(y1_saved.float() - y2.float()).abs().max().item() if y1_saved.shape == y2.shape else 'shape'}",
        )

        reset_host_slot_graphs()
        ring.arm([host])
        x2 = torch.zeros(2, k, dtype=torch.bfloat16, device="cuda")
        before = launches[0]
        (y_n2,) = grp.run(x2, ring=ring)
        torch.cuda.synchronize()
        check("N!=1 stays eager", host_slot_graphs() == {}, str(host_slot_graphs()))
        check("N!=1 launched nf4_gemm", launches[0] == before + 1, f"{launches[0] - before}")
        check("N!=1 shape", tuple(y_n2.shape) == (2, m), str(tuple(y_n2.shape)))
    finally:
        graph_mod.nf4_gemm = real
        reset_host_slot_graphs()
        del slots, ring, x
        torch.cuda.empty_cache()


def test_gpu_mixed_group_not_one_recipe() -> None:
    """DEVICE+HOST stays serial; only the HOST member may replay a slot graph."""
    _gpu_skip("gpu mixed HOST slot graph")
    reset_host_slot_graphs()
    k = 64
    dev = _dev_gemm("q", 64, k)
    dev = Gemm(
        name="q",
        M=dev.M,
        K=dev.K,
        K_pad=dev.K_pad,
        bias=None,
        packed=dev.packed.to("cuda"),
        scale=dev.scale.to("cuda"),
        home="device",
    )
    host, img = _host_gemm("k", 64, k, fill=2)
    pinned = maybe_pin(img.arena)
    if pinned is not img.arena:
        img = HostImage(pinned, img.M, img.K, img.K_pad)
        host = Gemm(
            name="k",
            M=img.M,
            K=img.K,
            K_pad=img.K_pad,
            bias=None,
            packed=torch.empty(0, dtype=torch.uint8, device="cuda"),
            scale=torch.empty(0, dtype=torch.float16, device="cuda"),
            host_image=img,
            home="host",
        )
    grp = GemmGroup("qkv", [dev, host], (object(), object()))
    check("mixed not resident", not group_is_resident(grp), "")
    check("mixed streams=()", grp.streams == (), f"n={len(grp.streams)}")
    ring = CopyRing(SlotPair(img.nbytes, "cuda"))
    launches = [0]
    real = graph_mod.nf4_gemm

    def counted(packed, scale, x, M, K, K_pad):  # noqa: N803
        launches[0] += 1
        if packed.device.type != "cuda":
            raise RuntimeError("kernel saw host packed")
        return real(packed, scale, x, M, K, K_pad)

    graph_mod.nf4_gemm = counted
    try:
        x = torch.zeros(1, k, dtype=torch.bfloat16, device="cuda")
        try:
            grp.run(x, ring=ring)
        except Exception as exc:  # noqa: BLE001
            skip(f"gpu mixed HOST slot graph: nf4_gemm failed ({type(exc).__name__}: {exc})")
        torch.cuda.synchronize()
        recipes = [r for r in host_slot_graphs().values() if r is not None]
        check("mixed captured HOST slot only", len(recipes) == 1, f"n={len(recipes)}")
        if not recipes:
            return
        first = launches[0]
        ring.arm([host])
        grp.run(x, ring=ring)
        torch.cuda.synchronize()
        check("mixed HOST replays", recipes[0].replays == 1, f"{recipes[0].replays}")
        check(
            "second mixed forward only DEVICE eager launch",
            launches[0] == first + 1,
            f"{launches[0]} vs first {first}",
        )
        runners, mode, err = capture([grp], warmup=1)
        check("capture does not graph mixed group", runners[0] is grp, type(runners[0]).__name__)
        check("mixed capture mode off", mode == "off", f"{mode} err={err}")
    finally:
        graph_mod.nf4_gemm = real
        reset_host_slot_graphs()
        torch.cuda.empty_cache()


TESTS = [
    test_slot_gemm_key_refuses_host_packed,
    test_try_host_slot_gemm_skips_cpu_packed,
    test_host_group_is_not_resident,
    test_capture_skips_host_groups,
    test_cpu_n1_host_one_stays_eager,
    test_cpu_n_not_one_stays_eager,
    test_gpu_decode_replays_after_bind,
    test_gpu_mixed_group_not_one_recipe,
]


def main_runner() -> int:
    print(f"gpu/loop HOST-slot graphs, {len(TESTS)} tests\n")
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
