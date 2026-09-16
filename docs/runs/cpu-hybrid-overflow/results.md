# CPU/GPU hybrid overflow — measured (2026-09-16)

Branch: [`exp/cpu-hybrid-overflow`](../../../). Same machine as
[`docs/compare-3080.md`](../../compare-3080.md): RTX 3080 12 GB WDDM, Ryzen 9
5950X, Qwen2.5-32B-Instruct, greedy, `ctx=2048`, ignore-EOS 64-token plateau.
Product generate stays `--compute gpu` (CopyRing **2.49**). This folder is the
git copy; live plates stay under `C:\dev\models\runs\`.

Do not merge these rows into `deepfold-nf4-32B-overflow` in `compare.json`.

## Did not beat Ollama

The hypothesis was: keep a resident NF4 GPU prefix, run a shorter CPU suffix
than Ollama’s 32/32, maybe recode the suffix to affine INT4 (`i4c`). Long
decode did not pass **2.54 tok/s**. Prefill on the i4c suffix was far worse
than decode. CUDA sat idle for the CPU walk (residual is a chain; CopyRing is
H2D slots, not a second compute device).

| Stack | 32B long tok/s | Notes |
|---|---:|---|
| Ollama 0.34.0 Q4_K_M | **2.54** | 32 repeating + `lm_head` GPU, 32 repeating CPU. Weights stay put. |
| deep-fold `--compute gpu` | **2.49** | All 64 layers on GPU. Policy D, 96 matrices / 6885 MiB/token. |
| `--compute hybrid --gpu-layers 36 --cpu-codec i4c --no-graphs` | **2.091** | 36 GPU NF4 + 28 CPU i4c sidecar. smi after load **11747 MiB**. Prefill **209 s**. `graph=off`. Plate `C:\dev\models\runs\deepfold-long-32B-hybrid-i4c-36-nographs`. |
| `--compute cpu-suffix` (N=32, NF4 CPU) | **1.694** | Same bounce, longer NF4 suffix. |
| hybrid-36 i4c **with** DEVICE graphs | — | Hung after `capture DEVICE 145/145`. Killed (`4294967295`). |
| hybrid-38 (earlier sweep) | — | Died at load (`4294967295`). Plan turns CopyRing on at N=38. |
| hybrid-48 + CopyRing + graphs | **0.592** | Mixed tape + CPU suffix. Do not retry with graphs. |

i4c AVX2 N=1 (float-x FMA, 16 threads) is ~9–10 ms/layer on 32B shapes in
`python -m gpu.lab.cpu_i4c_bench`, faster than CPU NF4, still not Q4_K’s
implied ~6 ms/layer. Live sidecar is packed from an NF4 decode (no 64 GB BF16
in RAM), not written into the `.chr`.

## Why the GPU looked idle

`--compute hybrid` is serial: layers 0..N-1 on CUDA, `synchronize`, ~10 KiB
bounce, layers N..63 on the 5950X, bounce back, `lm_head` on GPU. N=36 already
fills the card (~11750 MiB). N=37 is the last whole-layer prefix (`ring=none`).
N=50–56 as *resident* GPU layers do not fit; streaming them is hybrid-48.

CopyRing cannot overlap a contiguous CPU suffix with GPU GEMM: after the
prefix there is no HOST tape left except `lm_head`. Two codecs in one group
are refused (`device`/`host` vs `cpu`). WDDM still joins `e_copy` before
prefetch (depth > 1 was 0.01 tok/s).

Every decode token still has to *read* ~16 GiB of packed 32B weights. 12 GB
cannot hold them. Either PCIe (gpu mode, 277 ms copy floor) or DDR4 on the
suffix (hybrid / Ollama). That is the box ceiling around **~2.5 tok/s**, not a
missing kernel that 2× Ollama. 3B is a different gap (resident, 187 vs 35).

## Commands (this branch)

```text
python gpu/host/test_i4c.py
python gpu/host/test_compute.py
python gpu/host/test_cpu_linear.py

python -m gpu.lab.deepfold_long --size 32B --compute hybrid --cpu-codec i4c --no-graphs
```

`--compute gpu` is unchanged. `--cpu-codec i4c` needs `cpu-suffix` or `hybrid`.
