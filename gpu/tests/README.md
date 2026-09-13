# `gpu/tests` — floor-1 oracle and safety gate (wave 2, agent 4)

Protocol, thresholds and failure semantics: [docs/spec/gpu-safety.md](../../docs/spec/gpu-safety.md).

## Run

```text
conda activate torch-gpu
python gpu/tests/oracle_gate.py --chr C:\dev\models\qwen25-3b.nf4.chr
```

Exit code: `0` = PASS, `2` = FAIL, `3` = BLOCKED (a required check could not run,
e.g. `gpu/nf4` is not built yet). `3` is **not** green.

Toy checks (F1 `128x256`, F2 `130x65`) and every self-check run without the 3B
file; `--chr` only adds F3/F4/F5 and the VRAM/safety block.

Useful flags: `--name` (another tensor), `--tol`, `--json <path>`,
`--crosscheck-f32 <path>` (opt-in, see the protocol §7).

If K1 reports a stale kernel binary, rebuild it — the JIT falls back to the
previously built `.pyd` when MSVC or ninja are missing from `PATH`:

```text
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
conda activate torch-gpu
python gpu/nf4/setup.py build_ext --inplace
```

## Files

| File | Role |
|---|---|
| `oracle_gate.py` | the gate: runs F/S checks, prints the table and the numbers |
| `nf4_oracle.py` | CPU NF4 decode/encode, LUT bit-exact against `docs/spec/nf4.md` §1 |
| `chr0_min.py` | read-only CHR0 parsing for the byte cross-check + toy/malformed fixtures |
| `gpu_probe.py` | `nvidia-smi`, VRAM budget, dequant-W watcher, read-only file guard |
| `backend.py` | import shims for `gpu.chr0` (agent 1) and `gpu.nf4` (agent 2) |

## Forbidden inside this gate

- **`chr decode` of the whole model is forbidden here.** The F32 dump of
  Qwen2.5-3B is ~12 GiB; producing it is not part of any test, and no test may
  shell out to `chr decode`. The oracle decodes packed bytes in numpy instead
  (S8). `--crosscheck-f32` may read **one** tensor slice from an F32 dump that
  already exists on disk, capped at 256 MiB, and it never creates one.
- No original BF16 safetensors, no `from_pretrained`, no `generate`, no KL —
  that is floor 2 of `docs/token-loop.md` §7.3, a different wave.
- No writes to any `.chr`: the gate opens them `"rb"` and asserts `mtime`/size
  are unchanged (S3).
- No "make the numbers nicer" edits to `gpu/nf4/*.cu`. A kernel patch from this
  directory is only legitimate for an out-of-bounds/safety bug, and then it must
  arrive with a test.
