# Spec stitches (wave 1, CPU)

First-wave disagreements, **locked** before code. If a module spec contradicts this page — this page wins.

GPU wave: [stitch-gpu.md](stitch-gpu.md).

| Topic | Decision |
|---|---|
| `inv_freq` / `rotary_emb` / `.sin` / `.cos` | **skip**, not in `.chr`. Classifier — [integrity-cli.md](integrity-cli.md) §3.1. |
| `--chunk` default | **262144** ([vq.md](vq.md) §8.3). Not 1048576. |
| `--group-size` | nf4 64 only, vq 8 only. |
| Names in `.chr` | no `.weight` suffix; bias keeps `.bias`. |
| Decode safetensors | always **F32**, one file. |
| Norms / bias | `codec=bf16`, bit-exact to the BF16 projection of orig. |
| Linear / embed / lm_head | one `--codec` per file (`nf4` or `vq`). |
| Blobs | row-major, not Ampere fragment-major. |
| File reads | `os.File.ReadAt`, no mmap, no CGO. |
| CHR0 JSON | compact, `DisallowUnknownFields`, integer offsets. |
| `hidden_size` | ≥1; compress derives it from config or tensors, does not leave 0. |
| VQ RNG | `math/rand/v2` PCG(seed, 0) **anew for every** matrix. |
| Bands | `full_f32 > 256MiB`; unit tests do not include them. |
| Go module | `chr` (not a github path). Packages as in integrity-cli §5. |

Packages:

```
chr/
  cmd/chr/          # flags + run()
  internal/f16/     # binary16 ↔ float32
  internal/safetensors/
  internal/chr0/
  internal/nf4/
  internal/vq/
  internal/verify/  # classify + metrics + compress/decode orchestration
```

The classifier lives in `internal/verify` (or `internal/tensor`), one for compress and verify.
