# Matched 3080 comparison (2026-09-16, auto-fit 2026-09-17)

One RTX 3080 12 GB, Windows/WDDM, greedy decode, `ctx=2048`, `n_predict=64`.
Qwen2.5 Instruct. Quote **long** ignore-EOS 64-token plateaus for 32B decode,
not Ollama’s short-EOS smoke mean (8+8+4 tokens → 3.18 tok/s).

Plate: [`docs/img/compare-3080.png`](img/compare-3080.png)
JSON: [`docs/runs/compare-3080/`](runs/compare-3080/)
32B protocol: [`docs/eval-32b.md`](eval-32b.md)

Redraw: `python -m gpu.lab.compare_plate --redraw`

## Numbers

Left column is the engine. Rows under the same engine are launches or
counters, not different products.

<table>
<thead>
<tr>
<th>Engine</th>
<th>Launch / counter</th>
<th align="right">3B long</th>
<th align="right">32B long</th>
<th align="right">32B smoke</th>
<th align="right">smi 32B</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ollama 0.34.0</td>
<td>Q4_K_M <code>qwen2.5:*</code></td>
<td align="right"><strong>187.3</strong></td>
<td align="right"><strong>2.54</strong></td>
<td align="right">3.18</td>
<td align="right">11559</td>
</tr>
<tr>
<td rowspan="2">llama.cpp b10964 Q4_K_M</td>
<td>auto-fit (no <code>-ngl</code>)</td>
<td align="right" rowspan="2"><strong>187.0</strong></td>
<td align="right"><strong>2.54</strong></td>
<td align="right">2.62</td>
<td align="right">11636</td>
</tr>
<tr>
<td><code>-ngl 99</code></td>
<td align="right"><strong>1.52</strong></td>
<td align="right">1.55</td>
<td align="right">11520</td>
</tr>
<tr>
<td rowspan="2">deep-fold NF4 TokenLoop</td>
<td>63 <code>step()</code> after first token</td>
<td align="right"><strong>35.2</strong></td>
<td align="right"><strong>2.49</strong></td>
<td align="right" rowspan="2">2.31</td>
<td align="right" rowspan="2">11926</td>
</tr>
<tr>
<td>64 generated tokens / same wall</td>
<td align="right"><strong>35.8</strong></td>
<td align="right"><strong>2.53</strong></td>
</tr>
<tr>
<td>bitsandbytes</td>
<td>NF4 <code>Linear4bit</code></td>
<td align="right">22.2 <em>smoke</em></td>
<td align="right">—</td>
<td align="right">—</td>
<td align="right">—</td>
</tr>
<tr>
<td rowspan="4">Not run on this box</td>
<td>AWQ</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
</tr>
<tr>
<td>GPTQ+Marlin</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
</tr>
<tr>
<td>ExLlamaV2</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
</tr>
<tr>
<td>vLLM</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
<td align="right">SKIP</td>
</tr>
</tbody>
</table>

SKIP is a row, not a borrowed tok/s. Isolated venvs:
[`docs/competitor-venvs.md`](competitor-venvs.md).

llama.cpp 3B long **187.0** is one plate (the net fits; `-ngl` is a 32B
placement issue). TokenLoop `decode_tok_s` is 63 `step()` calls after the
first token / `decode_ms`. The second deep-fold row is `eval_tok_s` = 64
generated tokens / the same wall — same counting as Ollama `eval_count` /
`eval_duration`. Quote **2.53 vs 2.54** when matching that timer; keep
**2.49** as the step rate.

Commands:

```text
python -m gpu.lab.ollama_h2 --model qwen2.5:32b --size 32B
python -m gpu.lab.deepfold_long --size 32B
python -m gpu.lab.llamacpp_h2
python -m gpu.lab.llamacpp_h2 --ngl 99
python -m gpu.lab.ollama_hard --model qwen2.5:3b
python -m gpu.lab.ollama_hard --model qwen2.5:32b
python -m gpu.lab.hard --lab qwen25-32b --codec nf4
python -m gpu.lab.compare --seed-known
```

`llamacpp_h2` omits `--n-gpu-layers` unless you pass `--ngl ≥ 0`. `--ngl 99`
is the old fill-card launch (32B long **1.52**). Auto-fit writes a separate
compare.json id (`…-autofit`), so 1.52 stays. `ollama_h2` uses `stream=True`
and records `client_ttft_ms`. Historical Ollama TTFT cells stay null (cached
`prompt_eval`, `stream=False`).

3B protocol check 2026-09-17: stream smoke still `cached=24`, so
`mean_ttft_ms` stays null; `mean_client_ttft_ms` ≈ 31 ms. Decode that day
was ~226 tok/s long — do **not** replace the matched **187.3** plate.
Independent hard-12 on `qwen2.5:3b`: **7/12**
([`docs/runs/ollama-hard-3b/`](runs/ollama-hard-3b/)). On `qwen2.5:32b`:
**11/12**, miss `gsm8k-stickers` (`72` vs gold `48`)
([`docs/runs/ollama-hard-32b/`](runs/ollama-hard-32b/)). TokenLoop 32B NF4
overflow: **12/12**
([`docs/runs/hard-qwen25-32b/`](runs/hard-qwen25-32b/)). InternLM 20B is a
different Instruct: TokenLoop NF4 **8/12** (2026-09-14); Ollama Q4_K_M
**9/12** ([`docs/runs/ollama-hard-20b/`](runs/ollama-hard-20b/)).

<table>
<thead>
<tr>
<th>Engine</th>
<th>Size</th>
<th align="right">hard-12</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">deep-fold NF4 TokenLoop</td>
<td>3B resident</td>
<td align="right">8/12</td>
</tr>
<tr>
<td>14B resident</td>
<td align="right">10/12</td>
</tr>
<tr>
<td>20B InternLM resident</td>
<td align="right">8/12</td>
</tr>
<tr>
<td>32B overflow</td>
<td align="right">12/12</td>
</tr>
<tr>
<td rowspan="3">Ollama 0.34.0</td>
<td>3B Q4_K_M</td>
<td align="right">7/12</td>
</tr>
<tr>
<td>20B InternLM Q4_K_M</td>
<td align="right">9/12</td>
</tr>
<tr>
<td>32B Q4_K_M</td>
<td align="right">11/12</td>
</tr>
</tbody>
</table>

## What Ollama actually runs

Ollama 0.34.0 is a scheduler around vendored `llama-server.exe` (llama.cpp
**b10760**). It is not a separate GEMM. Live argv from
`%LOCALAPPDATA%\Ollama\server.log` for `qwen2.5:32b`:

```text
--load-mode none --flash-attn auto -c 2048 -np 1 -b 512 -ub 512
```

`-ngl` is omitted (`NumGPU = -1`). llama-server auto-fits. On this 3080:

| | GPU | CPU / CUDA_Host |
|---|---|---|
| Layers | **33 / 65** (32 blocks + `lm_head`) | remaining **32** blocks |
| Weights | **9559 MiB** | **9367 MiB** |
| KV @2048 | 256 MiB | 256 MiB |

`sched_reserve: graph splits = 2` at `bs=1`. Hidden state (~10 KiB) crosses
the split once per token. Weights stay put. `--load-mode none` is forced on
Windows+CUDA (`disableMmapDefaultReason` → `windows_cuda`) so the CPU suffix
is in RAM, not mmap page faults. Flash Attention enabled. `USE_GRAPHS=1`.
16 AVX2 threads on the 5950X.

3B: **37/37 layers on GPU** → 187 tok/s.

Standalone llama.cpp **`-ngl 99`** aborts auto-fit (`n_gpu_layers already set
by user to 99`) and is **1.52**. Omit `--n-gpu-layers` (runner default,
2026-09-17): long **2.54**, smoke mean 2.62, smi **11636 MiB**, TTFT ~1444 ms.
Dump: [`docs/runs/llamacpp-h2-autofit/`](runs/llamacpp-h2-autofit/). Same
Q4_K_M family as Ollama **2.54**; placement was the 1.52 miss, not a newer
algorithm (0.34.0 vendors **b10760**; our zip is **b10964**).

## Overflow: layers vs matrices

**Ollama / llama.cpp.** First *N* transformer blocks fully on GPU. The rest
fully on CPU. Overflow costs CPU Q4_K GEMM plus one activation bounce.

**deep-fold H2 (policy D).** Every layer still runs on the GPU. Embed,
`lm_head`, and all `q/k/v/o` stay resident. Every `down_proj` and tail
`gate`/`up` (L48–63) live in pinned host memory. `CopyRing` copies **96**
packed matrices, **6885 MiB**, into two device slots each decode step.
Serial copy floor **277 ms/tok**; measured long wall **~402 ms → 2.49 tok/s**.
WDDM joins `e_copy` before the next prefetch (depth 1). Reconstruct is still
tile-local in `chr_nf4_gemm`.

The 32B long numbers that share a counting convention are Ollama **2.54**,
llama.cpp auto-fit **2.54**, and H2 `eval_tok_s` **2.53**. H2 step rate is
**2.49**. That is two (now three) ceilings in the same place: CPU-suffix Q4_K
versus 6885 MiB over PCIe. It is **not** evidence that NF4 decode matches
Q4_K mmvq. `-ngl 99` **1.52** is a worse placement of the same GGUF.

## Why 3B is not a tie

Both 3B nets fit. Ollama/llama.cpp Q4_K fused CUDA (mmvq, graphs, FA) is
~187 tok/s. Our resident NF4 TokenLoop is 35.2 (63 steps) / 35.8 (64 tokens).
`LIVE_MAX_N=32`; decode N=1 does not feed the prefill tile. That is the
**whole generate path** (GEMM, attention, Python, launches), not an isolated
NF4 kernel bake-off, and it is not overflow. Nsight snippets under
`docs/runs/ncu/` are `q_proj`/`k_proj` GEMM occupancy at N=1 and N=16, not a
full TokenLoop forward.

## Decode and TTFT definitions

H2 long: **63** `step()` calls after the first token, `decode_tok_s =
decode_steps / decode_ms` → **2.49**. First token sits in `prefill_ms`
(925 ms). Same wall, generated-token counting: `eval_tok_s =
n_tokens / decode_ms` → **2.53**. Ollama long: `eval_count=64` /
`eval_duration` → **2.54**. Quote **2.53 vs 2.54** when matching Ollama’s
timer; keep **2.49** as the step rate. Not enough to rank 2%.

Do **not** quote Ollama smoke `prompt_eval_duration` as TTFT. The committed
32B/3B Ollama dumps used `stream=False` and `prompt_eval_cached_count=24`.
TokenLoop resets KV. Server `prompt_eval` on 32B smoke averaged ~901 ms;
that is not a matched first-token latency. The runner now uses `stream=True`
and records `client_ttft_ms` (first NDJSON chunk). `mean_ttft_ms` is that
value only when `prompt_eval_cached_count=0`.

Independent check on another PC is `scripts/plate.ps1` / `scripts/setup.ps1`.
It is not a substitute for this 3080 plate.

## Hybrid CPU option (tried; did not ship)

`exp/cpu-hybrid-overflow` tried an Ollama-style layer split (resident NF4
GPU prefix, CPU suffix, optional `i4c` sidecar). Same 64-token travelogue.
It did **not** beat **2.54**. Product generate is still `--compute gpu` /
CopyRing **2.49**. This branch does not contain that code.

<table>
<thead>
<tr>
<th>Engine</th>
<th>Launch</th>
<th align="right">32B long tok/s</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ollama 0.34.0</td>
<td>Q4_K_M library tag</td>
<td align="right"><strong>2.54</strong></td>
</tr>
<tr>
<td>llama.cpp b10964</td>
<td>Q4_K_M auto-fit</td>
<td align="right"><strong>2.54</strong></td>
</tr>
<tr>
<td rowspan="3">deep-fold NF4</td>
<td>CopyRing (product)</td>
<td align="right"><strong>2.49</strong></td>
</tr>
<tr>
<td>hybrid 36 GPU + 28 CPU i4c</td>
<td align="right">2.091</td>
</tr>
<tr>
<td>cpu-suffix 32, NF4 on CPU</td>
<td align="right">1.694</td>
</tr>
</tbody>
</table>

Write-up on the experiment branch:
[results.md](https://github.com/wertick01/deep-fold/blob/exp/cpu-hybrid-overflow/docs/runs/cpu-hybrid-overflow/results.md)
([GitLab](https://gitlab.com/wertick01/deep-fold/-/blob/exp/cpu-hybrid-overflow/docs/runs/cpu-hybrid-overflow/results.md)).
