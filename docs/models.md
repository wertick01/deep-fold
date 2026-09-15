# Models for checking `chr`

We download **BF16 safetensors**, not GGUF and not GPTQ: the compressor needs
the original weights. The `models/` directory is not in git.

| Role on a 3080 12 GB | Repository | Download | BF16 in 12 GB | After our NF4 |
|---|---|---|---|---|
| With headroom | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | **6.2 GB** | yes (~6.2 GB weights, leftover ~5 GB) | yes, room to spare |
| Tight | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | **29.5 GB** | not on the card: CUDA working set ~28 GiB, spill into shared GPU memory; `nvidia-smi` hits 12288 MiB | yes, leftover ~3.5 GB (on the card, no spill) |
| Does not fit | [internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) | **~40 GB** | no (~40 GB) | NF4 ~11.3 GB weights: first a standard `.chr`; TokenLoop splits fused `wqkv` |
| Overflow | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | **~65 GB** | no | packed NF4 ~16.6 GiB does not fit either; H2: resident ~9.7 GiB + pinned host tail ~6.9 GiB. Smoke 2.31 tok/s. Not VQ |

32B lives in `C:\dev\models\Qwen2.5-32B-Instruct`, CHR `C:\dev\models\qwen25-32b.nf4.chr`.
Paris/Berlin/323 smoke was recorded 2026-09-14 (`docs/eval-32b.md`, plate
`docs/img/h2-qwen25-32b.png`). Do not run hard-12 on 32B without an explicit “run it”.
20B remains the resident-NF4 caliber (all weights on the card).

Optional later: `Qwen/Qwen2.5-7B-Instruct` (~15 GB) — already no in BF16, freely in NF4. Not required for the bins above.

Download order: 3B → 14B → 20B → 32B (32B is already on disk).
CLI: `deepfold pull Qwen/Qwen2.5-3B-Instruct --yes` (allowlist, not an arbitrary Hub id).
