"""GPU-free tests for Llama smoke discovery. No Hub, no 8B download, no GPU.

    python -m gpu.lab.test_llama_smoke
    python gpu/lab/test_llama_smoke.py
"""

from __future__ import annotations

import builtins
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab import llama_smoke  # noqa: E402
from gpu.lab.llama_smoke import (  # noqa: E402
    ENV_LLAMA,
    SKIP,
    discover,
    skip_reason,
    write_source,
)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _llama_dir(root: Path, name: str = "Llama-3.1-8B-Instruct", **fields: object) -> Path:
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    payload = {"model_type": "llama", "hidden_act": "silu", "hidden_size": 8}
    payload.update(fields)
    (dest / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return dest


def gate_no_tree_is_a_skip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        trees = discover(models_root=Path(tmp) / "models")
        reason = skip_reason(trees)
        check("an empty models root is a skip, not a tok/s", trees == [] and reason.startswith(SKIP), reason[:80])
        check("the skip names the gated 8B rather than downloading it", "did not download" in reason.lower(), reason[:80])
        check(
            "the skip does not add llama3.1:8b to from-ollama",
            "llama3.1:8b" in reason and "from-ollama" in reason,
            "tag stays unknown",
        )


def gate_discovers_local_llama() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "models"
        llama = _llama_dir(root)
        _llama_dir(root, "Qwen2.5-3B-Instruct", model_type="qwen2")
        trees = discover(models_root=root)
        check("model_type=llama is picked up and qwen2 is not", [t.path for t in trees] == [llama], str(trees))
        check("a Llama 3.x config without SWA is attachable from JSON", trees[0].attachable, trees[0].refusal)


def gate_swa_refuses() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "models"
        _llama_dir(root, sliding_window=4096)
        trees = discover(models_root=root)
        check("a live sliding_window is a skip, not a generate", trees and not trees[0].attachable, trees[0].refusal[:80])
        check("SWA copy is the graphs refusal", "sliding window" in trees[0].refusal.lower(), trees[0].refusal[:80])


def gate_env_and_gguf() -> None:
    previous = os.environ.get(ENV_LLAMA)
    with tempfile.TemporaryDirectory() as tmp:
        llama = _llama_dir(Path(tmp), "from-env")
        os.environ[ENV_LLAMA] = str(llama)
        try:
            trees = discover(models_root=Path(tmp) / "missing")
            check("$DEEPFOLD_LLAMA wins when it is a Llama tree", trees and trees[0].path == llama, str(trees))
        finally:
            if previous is None:
                os.environ.pop(ENV_LLAMA, None)
            else:
                os.environ[ENV_LLAMA] = previous
        gguf = discover(explicit=r"C:\Users\x\.ollama\models\blobs\sha256-dead")
        check("a GGUF / ollama blob argument is ignored, not opened as Llama", gguf == [], "no GGUF")


def gate_open_never_sees_ollama() -> None:
    opened: list[str] = []
    real_open = builtins.open

    def watched(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    builtins.open = watched  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            discover(models_root=Path(tmp))
            llama_smoke.main(["--out", str(Path(tmp) / "out"), "--model", r"C:\Users\x\.ollama\models\blobs\sha256-dead"])
    finally:
        builtins.open = real_open  # type: ignore[assignment]
    bad = [p for p in opened if ".ollama" in p.lower() or "sha256-dead" in p.lower()]
    check("discover/main never open() ~/.ollama or blobs", not bad, str(bad[:3]))


def gate_cli_writes_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "llama-smoke"
        err, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(stdout):
            code = llama_smoke.main(["--out", str(out), "--model", str(Path(tmp) / "missing")])
        check("missing --model is exit 1", code == 1, str(code))
        source = out / "SOURCE.txt"
        text = source.read_text(encoding="utf-8") if source.is_file() else ""
        check("SOURCE.txt is the skip, not a number", source.is_file() and "SKIP:" in text, text[:80])
        check("SOURCE forbids inventing a tok/s", "Do not invent a Llama tok/s" in text, "no fake tok/s")
        check("SOURCE does not add the Ollama tag", "does not grow a llama3.1:8b row" in text, text[:80])


def gate_generate_without_tree_does_not_download() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = llama_smoke.main(
                ["--out", str(Path(tmp) / "out"), "--generate", "--model", str(Path(tmp) / "nope")]
            )
        check("--generate with no tree is still a skip", code == 1 and "did not download" in err.getvalue().lower(), err.getvalue()[:80])


def gate_write_source_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = write_source(Path(tmp), [])
        check("write_source returns SOURCE.txt", path.name == "SOURCE.txt" and path.is_file(), str(path))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("gpu/lab llama smoke (discover + skip, no Hub, no GPU)\n")
    for body in (
        gate_no_tree_is_a_skip,
        gate_discovers_local_llama,
        gate_swa_refuses,
        gate_env_and_gguf,
        gate_open_never_sees_ollama,
        gate_cli_writes_source,
        gate_generate_without_tree_does_not_download,
        gate_write_source_roundtrip,
    ):
        try:
            body()
        except Exception as error:  # noqa: BLE001
            check(body.__name__, False, f"{type(error).__name__}: {error}")
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
