"""CPU tests for ``gpu.cli.plate``. No Hub, no generate, no 3B required.

    python gpu/cli/test_plate.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.cli import hub as hub_mod  # noqa: E402
from gpu.cli import plate as plate_mod  # noqa: E402
from gpu.cli.ollama_map import ResolveError  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    extra = f"  -- {detail}" if detail and not ok else ""
    print(f"  {mark}  {name}{extra}")


def test_short_aliases_map_to_allowlist() -> None:
    assert plate_mod.resolve_spec("3b").hf_id == "Qwen/Qwen2.5-3B-Instruct"
    assert plate_mod.resolve_spec("14b").hf_id == "Qwen/Qwen2.5-14B-Instruct"
    assert plate_mod.resolve_spec("20b").hf_id == "internlm/internlm2_5-20b-chat"
    assert plate_mod.resolve_spec("32b").hf_id == "Qwen/Qwen2.5-32B-Instruct"


def test_hf_id_and_ollama_tag() -> None:
    spec = plate_mod.resolve_spec("Qwen/Qwen2.5-3B-Instruct")
    assert spec.hf_id == "Qwen/Qwen2.5-3B-Instruct"
    tagged = plate_mod.resolve_spec("qwen2.5:3b")
    assert tagged.hf_id == "Qwen/Qwen2.5-3B-Instruct"


def test_local_dir_with_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "Qwen2.5-3B-Instruct"
        dest.mkdir()
        (dest / "config.json").write_text("{}", encoding="utf-8")
        spec = plate_mod.resolve_spec(str(dest))
        assert spec.path is not None
        assert spec.path.resolve() == dest.resolve()
        assert spec.hf_id == "Qwen/Qwen2.5-3B-Instruct"


def test_gguf_is_refused_without_opening() -> None:
    opened: list[str] = []
    original = Path.open

    def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(self))
        return original(self, *args, **kwargs)

    Path.open = spy  # type: ignore[method-assign]
    try:
        try:
            plate_mod.resolve_spec(r"C:\blobs\sha256-dead.gguf")
        except ResolveError as exc:
            assert exc.kind == "gguf"
        else:  # pragma: no cover
            raise AssertionError("GGUF must be refused")
    finally:
        Path.open = original  # type: ignore[method-assign]
    assert opened == []


def test_unknown_hub_id_is_refused() -> None:
    try:
        plate_mod.resolve_spec("meta-llama/Llama-3.1-8B-Instruct")
    except ResolveError as exc:
        assert exc.kind == "unknown"
    else:  # pragma: no cover
        raise AssertionError("arbitrary Hub must be refused")


def test_dry_run_writes_plan_and_does_not_pull() -> None:
    called: list[object] = []
    original = hub_mod.snapshot_download
    hub_mod.snapshot_download = lambda **k: called.append(k) or ""  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(err):
                code = plate_mod.main(["3b", "--dry-run", "--out", tmp])
            assert code == 0, err.getvalue()
            payload = json.loads((Path(tmp) / "plate.json").read_text(encoding="utf-8"))
            assert payload["schema"] == plate_mod.SCHEMA
            assert payload["hf_id"] == "Qwen/Qwen2.5-3B-Instruct"
            assert payload["flags"]["dry_run"] is True
            assert any(s.get("name") == "dry-run" for s in payload["steps"])
            assert (Path(tmp) / "SUMMARY.txt").is_file()
    finally:
        hub_mod.snapshot_download = original  # type: ignore[assignment]
    assert called == []


def test_parser_requires_model() -> None:
    try:
        plate_mod._parser().parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("model is required")


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/cli plate, {len(TESTS)} tests, no GPU required\n")
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            check(fn.__name__, False, str(exc).splitlines()[0] if str(exc) else "assert")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
            failed += 1
        else:
            check(fn.__name__, True)
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} checks passed")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())
