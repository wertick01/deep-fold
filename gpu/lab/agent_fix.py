"""Live 14B agent acceptance: grep + str_replace + run_tests on a tiny fixture.

Not a TTY chat. Session KV is the same path as ``deepfold chat --agent``.
Do not quote tok/s. Metrics: tools used, pytest green, turn-2 suffix hit.

    python -m gpu.lab.agent_fix

Uses torch-gpu / CUDA. Writes a temp workspace unless ``--workspace``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from gpu.cli import agent as agent_mod
from gpu.cli import chat as chat_mod
from gpu.cli import run as run_mod
from gpu.cli.codec import load_config
from gpu.cli.doctor import probe
from gpu.cli.paths import ENV_MODEL
from gpu.lab.catalog import lab_by_slug

BOX = """def add(a, b):
    return a - b
"""

TEST = """from box import add


def test_add():
    assert add(2, 3) == 5
"""

PROMPT = (
    "test_add.py fails. Find the bug with grep, fix box.py with str_replace "
    "so add(a, b) returns a+b, then run_tests. Do not write_file the whole module."
)
FOLLOW = "Which file did you change, in one short sentence?"


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def write_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "box.py").write_text(BOX, encoding="utf-8")
    (root / "test_add.py").write_text(TEST, encoding="utf-8")
    (root / "AGENTS.md").write_text(
        "Fix failing pytest with grep, str_replace, run_tests.\n",
        encoding="utf-8",
    )


def pytest_ok(root: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_add.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0


def _args(ns: argparse.Namespace, workspace: Path) -> Namespace:
    lab = lab_by_slug("qwen25-14b")
    model = ns.model or os.environ.get(ENV_MODEL) or lab.model_dir
    packed = ns.chr or lab.chr_path
    args = Namespace(
        model=model,
        chr=packed,
        codec="auto",
        chr_bin=None,
        max_new_tokens=int(ns.max_new_tokens),
        max_seq=ns.max_seq,
        max_resident_mib=None,
        raw=False,
        warmup=not ns.no_warmup,
        no_compress=False,
        quiet=False,
        debug=False,
        residency="D",
        executor=ns.executor,
        agent=True,
        agent_web=False,
        agent_trust="workspace",
        workspace=str(workspace),
        max_tool_rounds=int(ns.max_tool_rounds),
    )
    overflow = False
    try:
        overflow = chat_mod.looks_overflow_model(load_config(model))
    except (OSError, TypeError, ValueError):
        overflow = False
    args.max_seq = chat_mod.pick_max_seq(
        ns.max_seq,
        agent=True,
        vram_mib=probe().vram_total_mib,
        overflow=overflow,
    )
    return args


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="HuggingFace directory")
    p.add_argument("--chr", help="packed .chr")
    p.add_argument("--workspace", help="reuse this dir (else a temp fixture)")
    p.add_argument("--max-seq", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-tool-rounds", type=int, default=24)
    p.add_argument("--executor", default="auto")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--keep", action="store_true", help="do not delete the temp workspace")
    ns = p.parse_args(argv)

    own = False
    if ns.workspace:
        workspace = Path(ns.workspace).expanduser().resolve()
        write_fixture(workspace)
    else:
        workspace = Path(tempfile.mkdtemp(prefix="deepfold-agent-fix-"))
        own = True
        write_fixture(workspace)

    if pytest_ok(workspace):
        _err("fixture pytest is already green; box.py was not the broken add()")
        return 2
    _err(f"workspace {workspace}  pytest red (expected)")

    args = _args(ns, workspace)
    _err(f"max_seq={args.max_seq} executor={args.executor} trust=workspace web=off")
    code, ctx = run_mod.prepare_run(args)
    if ctx is None:
        return code
    model, chr_file, checked, machine, _verdict = ctx
    try:
        tok, loop, stop, report = run_mod._open_loop(
            args,
            model,
            chr_file,
            checked.trust_remote_code,
            vram_mib=machine.vram_total_mib,
        )
    except RuntimeError as exc:
        _err(str(exc))
        return 1
    _err(
        f"loop {type(loop).__name__} session={chat_mod.session_capable(loop)} "
        f"codec={getattr(report, 'codec', 'nf4')}"
    )

    history: list[dict] = []
    agent_state = agent_mod.AgentSession(trust="workspace", web=False)
    prefix: list[int] | None = None

    def status(line: str) -> None:
        _err(line)

    outs1, prefix, reason1, tools1 = chat_mod.run_agent_message(
        tok,
        loop,
        stop,
        history,
        PROMPT,
        workspace=workspace,
        agent_state=agent_state,
        max_seq=int(args.max_seq),
        max_new=int(args.max_new_tokens),
        max_rounds=int(args.max_tool_rounds),
        prefix_ids=prefix,
        on_status=status,
    )
    green = pytest_ok(workspace)
    names1 = [name for name, _a, _c in tools1]
    _err(f"turn1 reason={reason1} tools={names1} pytest={'pass' if green else 'fail'}")

    outs2, prefix, reason2, tools2 = chat_mod.run_agent_message(
        tok,
        loop,
        stop,
        history,
        FOLLOW,
        workspace=workspace,
        agent_state=agent_state,
        max_seq=int(args.max_seq),
        max_new=int(args.max_new_tokens),
        max_rounds=int(args.max_tool_rounds),
        prefix_ids=prefix,
        on_status=status,
    )
    hits = [bool(getattr(o, "session_hit", None)) for o in outs2 if o is not None]
    prefs_n = [getattr(o, "prefill_n", None) for o in outs2 if o is not None]
    _err(f"turn2 reason={reason2} session_hit={hits} prefill_n={prefs_n}")

    payload = {
        "reason1": reason1,
        "reason2": reason2,
        "tools1": names1,
        "tools2": [name for name, _a, _c in tools2],
        "pytest": green,
        "box": (workspace / "box.py").read_text(encoding="utf-8"),
        "turn1_session_hit": [
            getattr(o, "session_hit", None) for o in outs1 if o is not None
        ],
        "turn1_prefill_n": [
            getattr(o, "prefill_n", None) for o in outs1 if o is not None
        ],
        "turn2_session_hit": [
            getattr(o, "session_hit", None) for o in outs2 if o is not None
        ],
        "turn2_prefill_n": [
            getattr(o, "prefill_n", None) for o in outs2 if o is not None
        ],
        "max_seq": int(args.max_seq),
        "executor": type(loop).__name__,
        "session_capable": chat_mod.session_capable(loop),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    ok = (
        green
        and "grep" in names1
        and "str_replace" in names1
        and "run_tests" in names1
        and any(h is True for h in hits)
    )
    if own and not ns.keep:
        shutil.rmtree(workspace, ignore_errors=True)
    elif own:
        _err(f"kept {workspace}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
