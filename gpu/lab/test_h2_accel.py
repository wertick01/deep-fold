"""CPU: h2_accel parser, schema, plan-only. No GPU, no .chr required.

    python gpu/lab/test_h2_accel.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.residency import (  # noqa: E402
    DEFAULT_POLICY,
    MIB,
    descs_from_qwen,
    overflow_resident_cap,
)
from gpu.lab.h2_accel import (  # noqa: E402
    SCHEMA,
    VARIANT_FIELDS,
    VARIANT_IDS,
    _apply_plan_fields,
    _parser,
    main,
    parse_k,
    parse_variants,
    plan_variant,
    variant_config,
    write_summary,
)
from gpu.lab.h2_place import QWEN_32B  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_parse_variants() -> None:
    check("default baseline list", parse_variants("baseline") == ["baseline"], "")
    check(
        "comma list",
        parse_variants("baseline,verify-k,pairs-stride")
        == ["baseline", "verify-k", "pairs-stride"],
        "",
    )
    check("all expands", parse_variants("all") == list(VARIANT_IDS), "")
    check("dedupe keeps order", parse_variants("profile,profile") == ["profile"], "")
    try:
        parse_variants("baseline,nope")
    except argparse.ArgumentTypeError as exc:
        check("unknown variant raises", "nope" in str(exc), str(exc)[:120])
    else:
        check("unknown variant raises", False, "no error")
    check(
        "matrix ids",
        VARIANT_IDS
        == (
            "baseline",
            "profile",
            "verify-k",
            "spec-lookup",
            "prefill-hold",
            "pairs-stride",
            "host-embed",
        ),
        str(VARIANT_IDS),
    )


def test_parse_k_and_variant_config() -> None:
    check("k default", parse_k("") == [2, 4, 8], "")
    check("k csv", parse_k("2,4,8") == [2, 4, 8], "")
    check("k single", parse_k("4") == [4], "")
    cfg = variant_config("baseline")
    check("baseline policy D", cfg["residency_policy"] == DEFAULT_POLICY, cfg["residency_policy"])
    check("baseline pin_embed", cfg["pin_embed"] is True, "")
    check("baseline prefill chunk", cfg["prefill_mode"] == "chunk", "")
    check(
        "pairs-stride policy",
        variant_config("pairs-stride")["residency_policy"] == "pairs_stride",
        "",
    )
    he = variant_config("host-embed")
    check(
        "host-embed policy D_host_embed",
        he["residency_policy"] == "D_host_embed" and he["pin_embed"] is False,
        str(he),
    )
    check("prefill-hold mode", variant_config("prefill-hold")["prefill_mode"] == "hold", "")
    check("profile timing", variant_config("profile")["ring_timing"] is True, "")
    spec = variant_config("spec-lookup")
    check("spec-lookup k=4", spec["speculate"] == 4 and spec["draft"] == "lookup", str(spec))


def test_parser_plan_only_flags() -> None:
    p = _parser()
    args = p.parse_args(
        ["--plan-only", "--variant", "baseline,verify-k", "--k", "4", "--max-seq", "512"]
    )
    check("plan_only flag", args.plan_only is True, "")
    check("variant string kept", args.variant == "baseline,verify-k", args.variant)
    check("k string kept", args.k == "4", args.k)
    check("max_seq 512", args.max_seq == 512, str(args.max_seq))
    forced = p.parse_args(["--force-overflow", "--variant", "baseline", "--max-seq", "512"])
    check("force_overflow", forced.force_overflow is True, "")
    check("plan_only default off", forced.plan_only is False, "")


def test_plan_variant_schema_32b_shapes() -> None:
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(12288, 2048, descs)
    base = plan_variant("baseline", descs, cap)
    stride = plan_variant("pairs-stride", descs, cap)
    host = plan_variant("host-embed", descs, cap)
    for row in (base, stride, host):
        missing = [f for f in VARIANT_FIELDS if f not in row]
        check(f"{row['id']} schema fields", not missing, str(missing))
        check(f"{row['id']} status ok or blocked", row["status"] in ("ok", "blocked"), row["status"])
    check("baseline policy D", base["residency_policy"] == "D", base["residency_policy"])
    check("baseline n_host 96", base["n_host"] == 96, str(base["n_host"]))
    check(
        "pairs-stride n_host == D",
        stride["n_host"] == base["n_host"] == 96,
        f"{stride['n_host']} vs {base['n_host']}",
    )
    check(
        "pairs-stride h2d_bytes == D",
        stride["h2d_bytes"] == base["h2d_bytes"],
        f"{stride['h2d_bytes']} vs {base['h2d_bytes']}",
    )
    check(
        "pairs-stride WHO not tail",
        stride["host_layers"]["gate"] != base["host_layers"]["gate"]
        and stride["host_layers"]["gate"] == list(range(3, 64, 4)),
        str(stride["host_layers"]["gate"][:4]),
    )
    check(
        "host-embed n_host is D minus 5",
        host["n_host"] == base["n_host"] - 5,
        f"{host['n_host']} vs {base['n_host']}",
    )
    check("host-embed cpu has embed", "model.embed_tokens" in (host.get("cpu") or []), str(host.get("cpu")))
    check("host-embed not on tape", "model.embed_tokens" not in (host.get("host_layers") or {}), "")
    verify = plan_variant("verify-k", descs, cap, k_values=[2, 4, 8])
    check("verify-k has k", verify.get("k") == [2, 4, 8], str(verify.get("k")))
    check(
        "verify-k status ok or blocked",
        verify["status"] in ("ok", "blocked"),
        f"{verify['status']} {verify.get('notes')}",
    )
    hold = plan_variant("prefill-hold", descs, cap)
    check(
        "prefill-hold status ok or blocked",
        hold["status"] in ("ok", "blocked"),
        f"{hold['status']} {hold.get('notes')}",
    )
    profile = plan_variant("profile", descs, cap)
    check("profile copy_ms null at plan", profile["copy_ms"] is None, str(profile["copy_ms"]))
    check("profile still has floor", profile["copy_floor_ms"] is not None, str(profile["copy_floor_ms"]))


def test_plan_only_main_without_chr() -> None:
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "out"
        missing_chr = Path(td) / "nope.nf4.chr"
        missing_model = Path(td) / "no-model"
        code = main(
            [
                "--plan-only",
                "--variant",
                "baseline,pairs-stride,host-embed,verify-k",
                "--chr",
                str(missing_chr),
                "--model",
                str(missing_model),
                "--out",
                str(dest),
                "--max-seq",
                "2048",
            ]
        )
        plate_path = dest / "plate.json"
        summary_path = dest / "SUMMARY.txt"
        check("plan-only exit 0 without chr", code == 0, f"code={code}")
        check("wrote plate.json", plate_path.is_file(), str(plate_path))
        check("wrote SUMMARY.txt", summary_path.is_file(), str(summary_path))
        plate = json.loads(plate_path.read_text(encoding="utf-8"))
        check("schema v1", plate.get("schema") == SCHEMA == "deepfold.h2_accel.v1", str(plate.get("schema")))
        check("plan_only true", plate.get("plan_only") is True, "")
        check("descs qwen32b-shapes", plate.get("descs_source") == "qwen32b-shapes", str(plate.get("descs_source")))
        ids = [r["id"] for r in plate["variants"]]
        check(
            "variant order",
            ids == ["baseline", "pairs-stride", "host-embed", "verify-k"],
            str(ids),
        )
        by_id = {r["id"]: r for r in plate["variants"]}
        for vid, row in by_id.items():
            missing = [f for f in VARIANT_FIELDS if f not in row]
            check(f"plate {vid} fields", not missing, str(missing))
        check(
            "baseline ok",
            by_id["baseline"]["status"] == "ok" and by_id["baseline"]["n_host"] == 96,
            str(by_id["baseline"].get("status")),
        )
        check(
            "pairs-stride bytes match baseline",
            by_id["pairs-stride"]["h2d_bytes"] == by_id["baseline"]["h2d_bytes"],
            "",
        )
        check(
            "host-embed fewer HOST matrices",
            by_id["host-embed"]["n_host"] == by_id["baseline"]["n_host"] - 5,
            f"{by_id['host-embed']['n_host']}",
        )
        text = summary_path.read_text(encoding="utf-8")
        check("SUMMARY names variants", "baseline" in text and "host-embed" in text, text.splitlines()[0])
        check("SUMMARY schema line", SCHEMA in text, text.splitlines()[0] if text else "")


def test_write_summary_mentions_blocked() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "SUMMARY.txt"
        plate = {
            "schema": SCHEMA,
            "plan_only": True,
            "cap": {"cap_mib": 1.0},
            "descs_source": "toy",
            "chr": "",
            "variants": [
                {
                    "id": "verify-k",
                    "status": "blocked",
                    "residency_policy": "D",
                    "n_host": 96,
                    "resident_mib": 10,
                    "h2d_bytes": 100,
                    "copy_floor_ms": 1.0,
                    "decode_tok_s": None,
                    "prefill_ms": None,
                    "smoke": None,
                    "notes": "gpu.loop.speculate missing (Accel-2)",
                }
            ],
        }
        write_summary(path, plate)
        text = path.read_text(encoding="utf-8")
        check("summary blocked", "blocked" in text and "verify-k" in text, text[:200])


def test_apply_plan_keeps_live_h2d() -> None:
    plan = {
        "n_host": 96,
        "resident_mib": 9714.0,
        "streamed_bytes": 7219445760,
        "h2d_bytes": 7219445760,
        "h2d_forwards": 1,
        "copy_floor_ms": 277.0,
        "k": [2, 4, 8],
        "status": "ok",
    }
    live = {
        "h2d_bytes": 20000000,
        "h2d_forwards": 12,
        "h2d_copies": 120,
        "status": "ok",
        "notes": "",
    }
    _apply_plan_fields(live, plan, live=True)
    check("live bytes kept", live["h2d_bytes"] == 20000000, str(live["h2d_bytes"]))
    check("live forwards kept", live["h2d_forwards"] == 12, str(live["h2d_forwards"]))
    check("live copies kept", live["h2d_copies"] == 120, str(live["h2d_copies"]))
    check("plan n_host overlay", live["n_host"] == 96, str(live.get("n_host")))
    check("per-fwd from live totals", abs(live["h2d_mib_per_fwd"] - (20000000 / 12 / MIB)) < 1e-6, str(live["h2d_mib_per_fwd"]))
    plan_only = {"status": "ok"}
    _apply_plan_fields(plan_only, plan, live=False)
    check("plan-only takes tape bytes", plan_only["h2d_bytes"] == 7219445760, str(plan_only.get("h2d_bytes")))


def test_load_model_residency_default() -> None:
    import inspect

    from gpu.host.model import load_chr_nf4, load_model

    sig = inspect.signature(load_model)
    check(
        "load_model residency_policy default D",
        sig.parameters["residency_policy"].default == DEFAULT_POLICY,
        str(sig.parameters["residency_policy"].default),
    )
    check(
        "load_model pin_embed default True",
        sig.parameters["pin_embed"].default is True,
        str(sig.parameters["pin_embed"].default),
    )
    nf4 = inspect.signature(load_chr_nf4)
    check(
        "load_chr_nf4 residency_policy default D",
        nf4.parameters["residency_policy"].default == DEFAULT_POLICY,
        str(nf4.parameters["residency_policy"].default),
    )


TESTS = [
    test_parse_variants,
    test_parse_k_and_variant_config,
    test_parser_plan_only_flags,
    test_plan_variant_schema_32b_shapes,
    test_plan_only_main_without_chr,
    test_write_summary_mentions_blocked,
    test_apply_plan_keeps_live_h2d,
    test_load_model_residency_default,
]


def main_tests() -> int:
    print("gpu/lab h2_accel parser + plan-only, CPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_tests())
