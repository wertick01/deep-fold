# Qwen2.5-32B: question list

Model: `C:\dev\models\Qwen2.5-32B-Instruct`  
CHR: `C:\dev\models\qwen25-32b.nf4.chr`  
`max_seq=2048`, greedy, isolated process, 12 GB, H2 overflow (NF4, not VQ).

## A — smoke (3) — PASS 2026-09-14

The same `gpu.lab.script.MESSAGES` / needles Paris / Berlin / 323.
`max_new_tokens=64`. PASS = 3/3. This is “it talks”, not 32B quality.

Command:

```
python -m gpu.lab.h2_trace --no-timing
```

Live dump: `C:\dev\models\runs\h2-qwen25-32b-20260914-234048`  
Copy in git: [`docs/runs/h2-qwen25-32b/`](runs/h2-qwen25-32b/)  
Plate: [`docs/img/h2-qwen25-32b.png`](img/h2-qwen25-32b.png)

| # | Prompt | Reply | tok/s | TTFT |
|---|---|---|---:|---:|
| 1 | What is the capital of France? | The capital of France is Paris. | 2.31 | 1004 ms |
| 2 | And the capital of Germany? | The capital of Germany is Berlin. | 2.32 | 993 ms |
| 3 | What is 17 times 19? | 323 | 2.30 | 1021 ms |

Mean decode **2.31 tok/s**, mean TTFT **1006 ms**. `gate.txt` = PASS.

H2 sanity from this run:

- `overflow=True`, `codec=nf4`, not VQ
- `report.device_mib` = **9716** (not ~16601)
- two slots, all `down_proj` HOST, gate+up HOST on L48–63
- smi decode **11926–11933**, does not grow across steps
- tok/s **2.31** > the 1–2 floor; ~3 ceiling not reached
- pin 6885/6885 MiB

Do not quote pageable ~0.8 tok/s: that is a `Tensor.is_pinned` as bool bug.

## llama.cpp Q4_K_M — same smoke, 2026-09-15

Same card, same `Qwen2.5-32B-Instruct`, same greedy Paris / Berlin / 323,
`ctx=2048`, `n_predict=64`. Different codec: bartowski **Q4_K_M GGUF**
(~18.5 GiB) through official **llama.cpp b10964** Windows CUDA 12.4
(`llama-server`, `-ngl 99`, `--parallel 1`). Not `llama-cpp-python`
(that wheel dies on Windows Long Paths here). GGUF is never converted to `.chr`.

Command: `python -m gpu.lab.llamacpp_h2 --bench`  
Live dump: `C:\dev\models\runs\llamacpp-h2-20260915-224614`  
Copy in git: [`docs/runs/llamacpp-h2/`](runs/llamacpp-h2/)

| | tok/s |
|---|---:|
| Smoke mean (three short replies) | **1.55** |
| Long decode, 64 tokens, `ignore_eos` | **1.52** |
| `llama-bench` tg64 × 3 | **1.47** |
| `llama-bench` pp512 | **69.8** |
| Mean TTFT (smoke) | **1010 ms** |
| nvidia-smi after load | **11520 MiB** |
| Smoke needles | **3/3** |

H2 NF4 overflow on the same prompts is **2.31 tok/s** (`docs/plan-h2-ring.md`).
llama.cpp is slower on decode here because 18.5 GiB of Q4_K_M does not fit
12 GiB even with `-ngl 99`; part of the net stays in RAM. Prefill is the
other way around (pp512 ~70 tok/s vs H2 TTFT ~1.0 s on ~40-token prompts).

Do **not** cite a Korean Ollama blog (~2.9–3.1 tok/s): that run was
`qwen2.5-coder:32b`, `num_ctx=32768`, temperature 0.2, Ollama — not this
plate. Ollama itself is not installed on this box yet.

The 3B competitor grid in `docs/competitor-venvs.md` is still empty.
This row is 32B overflow only.

## B — hard 12 — not run

Same as 14B NF4 (10/12). Compare with 14B NF4, not with BF16 32B (will not fit).
`gpu/lab/data/hard_items.json` → `python -m gpu.lab.hard`. `max_new_tokens=256`.

Do not start without Pavel’s explicit “run it”: 12 items × ~256 tokens at 2.3 tok/s —
that is already minutes per item, and it is a regression, not WikiText / GSM8K / MMLU.

| id | gold |
|---|---|
| gsm8k-lamps | 164 |
| gsm8k-money | 44 |
| gsm8k-tank | 90 |
| gsm8k-train | 240 |
| gsm8k-machines | 108 |
| gsm8k-stickers | 48 |
| trap-sheep | 9 |
| logic-yesno | no |
| code-sum | 15 |
| code-loop | 33 |
| trap-batball | 0.05 |
| prefill-warehouse | 7429 |

Optional history crate (4 turns) from the same JSON — KV, not codec quality.

```
python -m gpu.lab.hard --model C:\dev\models\Qwen2.5-32B-Instruct --chr C:\dev\models\qwen25-32b.nf4.chr
```
