# CPU check of compression (no GPU)

The `chr` utility: safetensors → `.chr` → decompress / compare to original. Video memory is not needed. The model is not placed in the repository.

## Build and tests

```bash
go test ./...
go build -o chr ./cmd/chr
```

The tests themselves write tiny safetensors. LLM weights are not downloaded.

## After downloading a model

`$MODEL` is a directory with `model.safetensors` or `model.safetensors.index.json`.

On **Windows** a cloud agent does not see the disk. 3B (~6.2 GB) is downloaded on your machine:

```powershell
# PowerShell, not Downloads
python -m pip install -U huggingface_hub
hf download Qwen/Qwen2.5-3B-Instruct --local-dir C:\dev\models\Qwen2.5-3B-Instruct
```

Or the script from the repo: `scripts\download-qwen25-3b.ps1`.

The thresholds below are **not** the binary defaults: they are for live weights. Unit tests run stricter numbers.

```bash
# NF4, group 64, no calibration
./chr compress --in "$MODEL" --out llama8b.nf4.chr --codec nf4 --quiet
./chr verify  --orig "$MODEL" --chr llama8b.nf4.chr \
    --fail-rmse 0.12 --fail-maxabs 2.0 --json > llama8b.nf4.verify.json

# Codebook 2×8 (residual k-means, seed 0). On 8B this is already minutes–tens of minutes on CPU.
./chr compress --in "$MODEL" --out llama8b.vq2.chr --codec vq --seed 0 --iters 20 --chunk 262144
./chr verify  --orig "$MODEL" --chr llama8b.vq2.chr \
    --fail-rmse 0.50 --fail-maxabs 8.0 --json > llama8b.vq2.verify.json
```

`PASS` / exit 0 means: the container is intact, norms are bit-exact, lossy did not explode. This is **not** WikiText and not chat.

A full F32 dump (`chr decode`) for 8B ≈ 32 GB — not needed for acceptance.

Peak RAM: one tensor in float32. `lm_head` / `embed` 8B ≈ 2 GB F32 plus packed output. On 32B `lm_head` is even larger; if it does not fit — say so, we will add stripes (they are already described in the spec; this slice’s units do not run them).

## What is inside

| Command | Meaning |
|---|---|
| `compress --codec nf4` | QLoRA NF4, group 64 |
| `compress --codec vq` | two 256×8 codebooks, 2 bit/weight |
| `decode` | one safetensors, **F32** |
| `verify` | orig vs `.chr`, tensor by tensor |

`inv_freq` / rotary are not written into `.chr` (they are not GEMM weights). Norms and bias — raw BF16.

Specs: [docs/spec/](spec/). Seams: [spec/stitch.md](spec/stitch.md).
