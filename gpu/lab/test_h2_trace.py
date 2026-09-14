"""CPU: H2 telemetry helpers. No GPU, no 32B load.

    python gpu/lab/test_h2_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.residency import MIB  # noqa: E402
from gpu.lab.h2_metrics import (  # noqa: E402
    H2D_CALIB_MS,
    expected_h2d_bytes,
    h2d_ms,
    host_pin_snapshot,
    pcie_gb_s,
    render_data_path,
)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def test_h2d_calibration() -> None:
    got = h2d_ms(256 * MIB)
    check("256 MiB is 10.31 ms", abs(got - H2D_CALIB_MS) < 1e-9, f"{got}")
    gb = pcie_gb_s()
    check("pinned H2D ~24.3 GB/s", 24.0 < gb < 24.6, f"{gb:.3f}")
    check("formula 71.72 MiB", abs(h2d_ms(71.72 * MIB) - 71.72 * 10.31 / 256) < 1e-9, "")
    check(
        "expected_h2d_bytes tape x forwards",
        expected_h2d_bytes(streamed_bytes=100, forwards=4) == 400,
        "",
    )


def test_host_pin_snapshot_pageable_is_not_pinned() -> None:
    """``bool(tensor.is_pinned)`` is always True; snapshot must call the method."""
    import torch
    import torch.nn as nn

    from gpu.host.host_image import HostImage

    class Seat(nn.Module):
        def __init__(self):
            super().__init__()
            packed = torch.zeros(8, 32, dtype=torch.uint8)
            scale = torch.zeros(8, 1, dtype=torch.float16)
            self.host_image = HostImage.from_blobs(packed, scale, 64)

    snap = host_pin_snapshot(Seat())
    check("pageable all_pinned False", snap["all_pinned"] is False, str(snap))
    check("pinned_bytes 0", snap["pinned_bytes"] == 0, str(snap))


def test_render_data_path_mentions_ring() -> None:
    md = render_data_path(
        {
            "cap": {
                "overflow": True,
                "codec": "nf4",
                "vram_mib": 12288,
                "cap_mib": 9800,
                "reason": "NF4 overflow (H2), not VQ.",
            },
            "residency": {
                "n_streamed": 96,
                "n_resident": 200,
                "streamed_bytes": int(6.7 * 1024 * MIB),
                "resident_bytes": int(10 * 1024 * MIB),
                "slot_nbytes": int(71.72 * MIB),
                "kinds": {
                    "down": {
                        "device": {"n": 0, "bytes": 0},
                        "host": {"n": 64, "bytes": 64},
                    },
                    "gate": {
                        "device": {"n": 48, "bytes": 1},
                        "host": {"n": 16, "bytes": 1},
                    },
                    "up": {
                        "device": {"n": 48, "bytes": 1},
                        "host": {"n": 16, "bytes": 1},
                    },
                    "q": {
                        "device": {"n": 64, "bytes": 1},
                        "host": {"n": 0, "bytes": 0},
                    },
                    "embed": {
                        "device": {"n": 1, "bytes": 1},
                        "host": {"n": 0, "bytes": 0},
                    },
                    "lm_head": {
                        "device": {"n": 1, "bytes": 1},
                        "host": {"n": 0, "bytes": 0},
                    },
                },
                "host_layers": {"down": list(range(64)), "gate": list(range(48, 64)), "up": []},
            },
            "pin": {
                "host_images": 96,
                "host_bytes": 100,
                "pinned_bytes": 100,
                "all_pinned": True,
            },
            "loop": {
                "prefill_chunk": 32,
                "n_groups": 257,
                "n_graphed_groups": 200,
                "n_eager_groups": 57,
                "n_device_gemms": 300,
                "n_host_gemms": 96,
                "graph_mode": "linears",
                "graph_error": None,
                "kv_mib": 512,
                "max_seq": 2048,
            },
            "device_mib": 9800,
            "smi_after_load": "11000 MiB",
            "messages": [
                {
                    "prompt_len": 40,
                    "prefill_ms": 800,
                    "prefill_chunks": 2,
                    "decode_tok_s": 2.5,
                    "decode_steps": 20,
                    "decode_ms": 8000,
                    "h2d_bytes": 200,
                    "h2d_copies": 200,
                    "h2d_forwards": 22,
                    "h2d_copy_ms": 1000,
                    "quality": True,
                    "needles": ("paris",),
                    "smi_used_mib": 11000,
                }
            ],
        }
    )
    check("mentions copy_stream", "copy_stream" in md, "")
    check("mentions chr_nf4_gemm", "chr_nf4_gemm" in md, "")
    check("mentions policy D downs", "HOST 64" in md, md[md.find("down") : md.find("down") + 80] if "down" in md else "")
    check("serial floor from calibration", "ms/tok" in md, "")


def main() -> int:
    print("gpu/lab H2 telemetry helpers, CPU only\n")
    test_h2d_calibration()
    test_host_pin_snapshot_pageable_is_not_pinned()
    test_render_data_path_mentions_ring()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
