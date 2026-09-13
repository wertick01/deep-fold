# Size vs efficiency: compressed vs uncompressed on a 12 GB 3080

Same GPU, same three English prompts, same harness (`gpu.lab`). The question is
not “is the NF4 kernel faster than dense GEMM.” It is: **as the model grows,
when does packing weights beat dense BF16 on the things that matter?**

Decode tok/s is a different stack on each side (HuggingFace `generate` + dense
GEMM vs `CompressedLinear` + `TokenLoop`). Report both. Do not rank them as a
kernel benchmark. Method: [`lab.md`](lab.md).

**20B is measured.** 14B and 20B speed cells below come only from the committed
CSVs. BF16 decode / TTFT stay blank on 20B: HuggingFace generate did not succeed
on this env (InternLM remote code vs transformers 5). Nothing is guessed from
a catalog budget.

## Sources

| Model | Committed `summary.csv` |
|---|---|
| Qwen2.5-3B-Instruct | [`runs/qwen25-3b/summary.csv`](runs/qwen25-3b/summary.csv) |
| Qwen2.5-14B-Instruct | [`runs/qwen25-14b/summary.csv`](runs/qwen25-14b/summary.csv) |
| internlm2.5-20B-chat | [`runs/internlm20b/summary.csv`](runs/internlm20b/summary.csv) |

Fields: `weight_mib`, `vram_after_load_smi_mib`, `vram_after_load_torch_mib`
(CUDA working set), `mean_ttft_ms`, `mean_decode_tok_s`, `quality_all_ok`.
Integers below are rounded from those columns the same way as the per-model
tables in [`lab.md`](lab.md), **except 3B NF4 tok/s and TTFT**.

**3B NF4 speed is not the committed CSV.** That file still has NF4 mean decode
17.0 tok/s and TTFT 212 ms (the plate copied into git). After the occupancy fix
(64-row tile + split-K), a live re-measure on 2026-09-13 was **31.6 tok/s**
and **167 ms** TTFT, same harness (`gpu.lab.worker`). The BF16 row (23.1
tok/s, 52 ms) is the committed CSV and was **not** re-run that day. Memory
columns are still the committed CSV.

## Comparison

| Model | Weight MiB (BF16 / NF4) | nvidia-smi after load (BF16 / NF4) | CUDA working set (BF16 / NF4) | Mean TTFT ms (BF16 / NF4) | Mean decode tok/s (BF16 / NF4) | Smoke |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-3B-Instruct | 5,886 / 1,563 | 7,477 / 3,142 | 5,886 / 1,618 | 52 / 167 | 23.1 / 31.6 | pass / pass |
| Qwen2.5-14B-Instruct | 28,172 / 7,483 | 11,955† / 8,913 | 28,270 / 7,539 | 1,028 / 759 | 0.92 / 6.56 | pass / pass |
| internlm2.5-20B-chat | 37,882 / 10,062 | 11,892‡ / 11,578‡ | 37,882 / 10,273 | — / 605 | — / 5.01 | not run / pass |

† **14B `nvidia-smi` is capped at 12288 MiB.** Do not quote 11,955 vs 8,913 as
the result. The win is the CUDA working set after load: **~28 GiB vs ~7.8 GiB**
(CSV: 28,270 MiB vs 7,539 MiB). 11,955 only means the dedicated-VRAM meter ran
out of scale; the rest of BF16 sits in Windows shared GPU memory (system RAM).
NF4 stayed on the card.

‡ **20B both `nvidia-smi` sit near 12288 MiB.** Do not quote 11,976 vs 11,828
(peaks) or 11,892 vs 11,578 as the product win. The result is the CUDA working
set after load: **~37.9 GiB vs ~10.3 GiB** (CSV: 37,882 MiB vs 10,273 MiB) and
NF4 decode **5.01 tok/s** with smoke pass. BF16 generate failed; those TTFT /
tok/s cells stay blank on purpose.

On 3B the card is not saturated: `nvidia-smi` and the CUDA working set are both
honest, and the working set is listed because the CSV has it.

## Who wins, by size

| Model | Dedicated 12 GB after load | Working set | Speed (TTFT + decode) | Compressed vs uncompressed |
|---|---|---|---|---|
| 3B | both fit | NF4 smaller, unused headroom | decode **NF4** 31.6 vs committed BF16 23.1 (not same-session); TTFT still **BF16** (52 vs 167 ms) | both fit; packing still pays when dense does not |
| 14B | BF16 spilled; NF4 on-card | **NF4** (~28 GiB vs ~7.8 GiB) | **NF4** | Compressed wins on everything that matters |
| 20B | both smi near 12288; BF16 spilled; NF4 on-card | **NF4** (~37.9 GiB vs ~10.3 GiB) | **NF4** 5.01 tok/s (no BF16 generate) | Compressed wins on working set; no dense speed baseline |

Weight packing itself is ~3.8× at all three sizes (5,886→1,563, 28,172→7,483,
and 37,882→10,062). That ratio does not decide the serving result. The crossover
is whether dense BF16 still fits in dedicated VRAM.

- **3B.** Both codecs live on the card. Mean decode **23.1 vs 31.6 tok/s**,
  mean TTFT **52 vs 167 ms**. The BF16 speed cells are the committed
  `qwen25-3b` CSV, not a same-day pair. NF4 uses less memory (7,477 vs 3,142
  MiB `nvidia-smi`; working set 5,886 vs 1,618 MiB). Occupancy was the old
  decode floor (16 CTAs on q/o, 2 on GQA k/v, 70 SMs); after split-K those
  grids are 128 and 64. Prefill (TTFT) is still slower on the packed path.
  Compression still pays when the uncompressed model does not fit.
- **14B.** BF16’s working set is 28,270 MiB. Packed NF4 is 7,539 MiB and stays
  on the card. Decode **0.92 vs 6.56 tok/s**, TTFT **1,028 vs 759 ms**. The
  slow BF16 path is weights crossing into system RAM every token, not a
  kernel-versus-kernel score.
- **20B.** BF16’s working set is 37,882 MiB (~26 GiB in Windows shared GPU
  memory). Packed NF4 is 10,273 MiB and stays on the card (peak `nvidia-smi`
  11,828 / 12,288, ~460 MiB headroom). Mean decode **5.01 tok/s**, mean TTFT
  **605 ms**, smoke pass (Paris / Berlin / 323). HuggingFace BF16 generate did
  not run: `internlm2_5-20b-chat` ships transformers-4.41 remote code, and
  `prepare_inputs_for_generation` raises `TypeError: can only concatenate
  tuple (not "int") to tuple` on transformers 5. No BF16 TTFT or tok/s.
  Recorded miss in `summary.notes`. Not patched on purpose — forcing the stock
  method drops `cache_position` and the model decodes fluent repetition, which
  would be a fake baseline.
