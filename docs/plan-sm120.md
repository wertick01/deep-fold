# GPU capability families: SM120 first

**Status 2026-09-18:** Track 0 done. Track 1/2 in tree without a card.
Remote SKU is **RTX 5070 Ti** (GB203, **70 SM** — same occupancy as the 3080
plate — 16 GB GDDR7, ~896 GB/s, L2 48 MB, `sm_120`). Not a Deepfold plate
until a friend run of `python -m gpu.lab.sm120_remote`.

`deepfold doctor` / `run` treat `(12,0)` / `(12,1)` as experimental generate.
Hopper and SM100 stay class-3 refuse. Occupancy planner reads
`multiProcessorCount` (fallback 70). Native `sm_120` cubin only if nvcc is
12.8+; the 3080 lab stays on CUDA 12.4 + PTX `compute_80`.

Measured plate stays RTX 3080 (`sm_86`). Those tok/s do not transfer. Author
has no SM120; first launch is a remote lab (friend) under the protocol below.

## Why generate used to refuse

Track 0 closed the named refuse for SM120. Remaining facts:

- Fatbinary `sm_80 / sm_86 / sm_89` + PTX `compute_80`; native `sm_120` when
  nvcc is 12.8+ — [`gpu/ampere_gencode.py`](../gpu/ampere_gencode.py)
- `GENERATE_CAPABILITIES = Ampere ∪ {(12,0),(12,1)}` —
  [`gpu/arch_family.py`](../gpu/arch_family.py)
- Turing / Hopper / **SM100 named refuse (D1)** — not a silent PTX try
- Occupancy: C planner probes `multiProcessorCount` (`one_wave=0`); Python
  default stays **70 SMs** because that is both the 3080 plate and the
  **RTX 5070 Ti**. A 5080 (84 SM) may split more.
- Setup: **3080 stays CUDA 12.4 / cu124**. Neighbor SM120 (5070 Ti) is
  **cu128 + CUDA 12.8+** (`nvidia-smi` name / `compute_cap`, or
  `DEEPFOLD_TORCH_INDEX=cu128`). 12.4 on SM120 is PTX JIT, not a refuse.

CHR0 and `chr_nf4_dev_t` stay frozen. Packed weights are arch-independent;
only the device path changes.

## The trap: Blackwell is two ISAs

RTX 50 is **not** a cut-down B200. Cubin `sm_100` does not load on SM120.
`tcgen05.*` is rejected by ptxas on `sm_120a`. CUTLASS / vLLM already split
`Sm100` vs `Sm120` trees.

| | Ampere SM80/86 | Hopper SM90 | Blackwell DC SM100 | **GeForce 50 SM120** |
|---|---|---|---|---|
| Compute | 8.0 / 8.6 / 8.9 | 9.0 | 10.0 | **12.0** (GB10 Spark: 12.1) |
| MMA | `mma.sync` HMMA.16816, registers | `wgmma.mma_async` | `tcgen05.mma` + TMEM | **`mma.sync` registers** (+ `kind::f8f6f4` / MXFP4) |
| TMEM / UMMA | no | no | yes | **no** |
| TMA | no (`cp.async`) | yes, clusters | multicast clusters | **yes, cluster 1×1×1 only** |
| SMEM / block | 99–164 KiB | 228 KiB | 228 KiB | **49 KiB default, 99 KiB opt-in** |
| Our kernel | ship / experimental | refuse | refuse | **experimental** |

Port to 50-series **starts from Ampere `mma.sync`**, not from B200. Our tiles
already fit 99 KiB. Occupancy now follows SM count (70 on 3080 and 5070 Ti).
What remains: CUDA 12.8+ for native cubin, and a named 5070 Ti plate.

RTX **5070 Ti** (the remote SKU): **70 SM**, 16 GB GDDR7, ~896 GB/s, L2 48 MB.
Same occupancy freeze as the 3080; 3B/14B NF4 fit; **32B NF4 still H2
overflow** (~16.6 GB packed). RTX 5090 ballpark vs 3080 plate: **170 SM**,
~32 GB GDDR7, ~1.79 TB/s — do not assume that card.

## Industry notes (do not copy kernels)

1. **NVIDIA Blackwell Compatibility Guide (CUDA 12.8).** Cubin is compatible
   only same major, same-or-higher minor. PTX `compute_80` **can** JIT on
   SM120. Native cubin + toolkit 12.8+ are for speed and stability, not for
   “will it launch”. Probe: `CUDA_FORCE_PTX_JIT=1`. Arch-conditional PTX
   (`sm_90a`, `sm_100a`) is **not** forward-compatible.
2. **CUTLASS / vLLM.** Separate `Sm100` vs `Sm120`. SM120: `mma.sync`,
   cluster `1×1×1`, tiles under 99 KiB. 228 KiB Hopper/SM100 templates
   **crash** on 5090. NVFP4 dense GEMM and MoE are different tickets.
3. **llama.cpp.** Native NVFP4 MMQ on `sm_120` is a **different codec**
   (block-scaled e2m1), not our NF4 LUT. Prefill wins; decode is often still
   bandwidth-bound.
4. **Marlin / vLLM W4A16.** Weight-only 4-bit stays dequant→BF16→MMA even on
   5090. Native FP4 tensor cores need W4A4 + NVFP4/MXFP4 scales. Our NF4 is
   Marlin-class, not llama.cpp NVFP4.
5. **Blackwell Decompression Engine** is LZ4/Snappy/Deflate, **not** NF4.
   Do not wire it into GEMM.
6. **PyTorch.** SM120 lab needs a **cu128+** wheel, not cu124.

Sources: [Blackwell Compatibility Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html),
[CUDA GPU CC](https://developer.nvidia.com/cuda/gpus),
[SM120 architecture notes](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm/blob/main/docs/sm120-architecture.md),
vLLM NVFP4 SM120, llama.cpp NVFP4 MMQ.

## Target runtime

```
CHR0 / chr_nf4_dev_t          ← frozen ABI
        ↓
DeviceCaps (probe)            ← capability, SM count, smem, L2, toolkit
        ↓
Family dispatcher             ← Ampere | Ada | SM120 | (Hopper/SM100 later)
        ↓
Launch planner                ← one_wave = sm_count, not 70
        ↓
Kernel image                  ← family cubin + fresh PTX
        ↓
Plate                         ← numbers only with a card name; else experimental
```

**Family A — compatibility (Ampere / Ada / first SM120).** Keep fused NF4 LUT
→ BF16 → `mma.sync.m16n8k16` + `cp.async`. Legal on SM120: same MMA model,
same 99 KiB. Change: native `sm_120` cubin, planner from
`multiProcessorCount`, doctor does not refuse.

**Family B — native SM120 (only after ncu).** If prefill is dequant/MMA-bound
(~5% DRAM on 3080), not bus-bound: TMA GMEM→SMEM; optionally an **additive**
CHR0 NVFP4/MXFP4 codec. Do not break NF4. Decode will most likely gain from
GDDR7 width, not from FP4 MMA.

**Family C — Hopper / SM100.** Out of this epic. `wgmma` / `tcgen05` is a
different skeleton.

Doctor:

| capability | generate |
|---|---|
| `(8,6)` | ship (3080 plate) |
| `(8,0)/(8,7)/(8,9)` | experimental (as today) |
| `(12,0)/(12,1)` | **experimental SM120**, not class-3 refuse |
| `(9,x)` Hopper, `(10,x)` SM100, `(7,5)` Turing | still refuse |

Do not turn `DEEPFOLD_ALLOW_UNMEASURED_ARCH` into a silent Hopper switch.

## Tracks

### Track 0 — DeviceCaps + doctor (no 5090) — done

- Probe: compute capability, `multiProcessorCount`, device name, VRAM.
- Planner: `one_wave` from SM count. 3080 and **5070 Ti stay 70 SM / 140 CTA**.
- Family table: Ampere `{80,86,87,89}`, SM120 `{120,121}`. No Hopper/SM100.
- Doctor: `(12,0)` / `(12,1)` experimental, exit 0. Hopper and SM100 stay 3.
- Tests: fake `Machine(..., device_name="RTX 5070 Ti")` and `(8,6)` regression.

### Track 1 — CUDA 12.8 fatbinary — code in tree

- `nvcc -gencode=arch=compute_120,code=sm_120` plus Ampere + PTX `compute_80`
  when nvcc ≥ 12.8. Toolkit < 12.8: Ampere as today; doctor warns, 3080 must
  not fail.
- Do not emit `sm_100` “for Blackwell”.
- Two install profiles: lab 3080 (cu124) vs neighbor SM120 (cu128, nvcc 12.8
  preferred even if 12.4 is on PATH). `nvidia-smi` / `DEEPFOLD_TORCH_INDEX=cu128`.
- Without a 12.8 toolkit: cannot `cuobjdump` a cubin on the author box.

### Track 2 — remote SM120 plate — protocol shipped

Friend box: driver **570+** (580+ better), CUDA Toolkit **12.8+**, PyTorch
**cu128+**, Python 3.11/3.12, Windows VS 2022 x64. Do not use current setup
(CUDA 12.4 / cu124) as the only path.

```
python -m gpu.lab.sm120_remote --out docs/runs/sm120-5070ti
python -m gpu.lab.sm120_remote --verify   # optional, skips without a GPU
```

Collect in one bundle:

1. `nvidia-smi` (name, VRAM, driver)
2. `torch.__version__`, `torch.version.cuda`, capability, `get_device_properties(0)`
3. `deepfold doctor` — experimental SM120, not exit 3
4. `python -m gpu.nf4.verify` + numerics
5. `deepfold run` Qwen2.5-3B-Instruct `--max-new-tokens 32`, then the 3B plate
6. If ncu exists: DRAM %, SM busy; one `CUDA_FORCE_PTX_JIT=1` vs native cubin
7. `SUMMARY.txt`, `plate.json`, doctor dump

First generate may JIT ~1 min — record it, do not “fix” it as tok/s.

Ship a repo entrypoint (`python -m gpu.lab.sm120_remote` or equivalent) that
runs the steps and packs artifacts. No card → skip, never a green pass.

### Track 3 — native SM120 (gated on ncu)

Do not start until Track 2 has ncu. Family A is already legal. Family B only
if the plate shows tensor cores / bus left on the table.

Candidates (pick after measurement, not all three):

1. TMA GMEM→SMEM, cluster `1×1×1`, then `ldmatrix` + the same NF4 LUT.
2. Native MXFP4/NVFP4 MMA — **new additive CHR0 codec**, oracle vs NF4.
3. Nothing. If decode is GDDR7-bound and occupancy is already 2 CTA/SM, stop.

## Frozen (do not dispute)

- `chr_nf4_dev_t` / CHR0 tile `{row:64, col_group:8}`
- NF4 group 64, LUT in [`docs/spec/nf4.md`](spec/nf4.md)
- 3080 plate remains the ship claim until a named SM120 plate exists
- No `sm_100` cubin in the GeForce fatbinary
- No Decompression Engine, ROCm, macOS in this epic
- No `#ifdef` ISA soup in one `.cu` before Family B

## Epic done when

- [ ] 3080 plate did not regress (same tok/s and VRAM on the same scenario)
- [x] `deepfold doctor` on fake SM120 is experimental, exit 0 on a healthy
      install; Hopper still 3 (needs a real 5070 Ti for the live half)
- [ ] First SM120 lab: 3B generate, numbers with a card name, no silent fallback
- [ ] ncu decode vs prefill (bus vs MMA) — go/no-go for Family B
- [x] `docs/install.md` / README GPU table: SM120 experimental, not
      “Blackwell refuse”
