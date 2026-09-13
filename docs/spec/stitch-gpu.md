# Стыки: CPU `.chr` → GPU (волна 2)

Если ядро, лоадер или [kernel-ampere.md](../kernel-ampere.md) спорят с этой страницей — **побеждает эта страница**.
Кодеки и контейнер не переписываем: [nf4.md](nf4.md), [vq.md](vq.md), [chr0.md](chr0.md), [stitch.md](stitch.md).

## Зафиксировано

| Тема | Решение |
|---|---|
| Кто хозяин байт | Уже записанный `.chr`. GPU **читает**, не перекодирует. |
| `group_size` NF4 | **64**, не 32 и не 128. |
| Нибблы | младшие 4 бита = `W[r, 2c]`, старшие = `W[r, 2c+1]`. Не packing bitsandbytes CUDA. |
| Шкала | FP16. Dequant: `float32(LUT[nib]) * float32(scale)`, затем pack BF16 в A-фрагмент. |
| LUT NF4 | 16 литералов из [nf4.md](nf4.md) §1, бит-в-бит. |
| Расклад блобов | **row-major**, как CPU. Ampere fragment-major — **не** эта волна. Permute в регистрах. |
| Книга 2¹⁶ | не читаем, даже если когда-нибудь попадёт в файл. |
| MMA | только `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Не `wmma`, не `m8n8k4`, не INT4 MMA. |
| Decode-тайл | `BM=128`, `BN=8` (pad), `BK=256` (кратно 64), `block=256`, stages=3. |
| Prefill-тайл (волна 3) | `BM=64`, `BN=16`, `BK=128`, `block=256`, stages=3. `N∈[2,16]`; `N>16` режет хост на куски ≤16. `N=1` остаётся decode-launch. |
| Prefill N=17..64 (план) | `BN=32` (N=17..32) и `BN=64` (N=33..64), тот же `BM=64`/`BK=128`. Описывает `gpu/nf4/plan.py`; `chr_nf4_gemm` по-прежнему -2. TokenLoop не поднимать. Следующий пол TTFT (167 vs 52 мс); ncu ~5% DRAM. |
| `x` в ядре | BF16, row-major **`[K, N]`**. `y` — BF16 **`[M, N]`**. HF `[..., K]` транспонирует хост. |
| Память | Python владеет тензорами. Ядро не делает `cudaMalloc` на токене. |
| Черновик `W` | в HBM **нет**. Деквант только в регистрах / smem-кольце packed. |
| Orig safetensors | на GPU **не открывать**. Оракул — packed + LUT, не `from_pretrained`. |
| Go-пакеты | не менять (`internal/*`, `cmd/chr`). Лоадер — Python/C++. |
| Первая модель | `Qwen2.5-3B-Instruct`, файл `C:\dev\models\qwen25-3b.nf4.chr`. |
| Карта | RTX 3080 12 ГБ, **sm_86**. Дисплей занимает VRAM — закладывать в smi. |

## Формулы размеров (лоадер отвергает mismatch)

`K_pad = 64 * ceil(K / 64)`, `n_groups = K_pad / 64`.

| Блоб | Байты |
|---|---|
| NF4 `data` | `M * K_pad / 2` |
| NF4 `scale` | `M * n_groups * 2` |
| VQ `index` | `M * (K_pad_vq / 8) * 2` при `K_pad_vq = 8 * ceil(K / 8)` |
| VQ `codebook` | `2 * 256 * 8 * 2` = 8192 |

## Эталон ошибки ядра (этаж 1)

На случайном `x` (RMSNorm-калибр, rms ≈ 1) и реальном packed `W`:

`max |Y_gpu − Y_cpu| ≤ 0.05` в float32, где `Y_cpu = dequant_nf4(packed) @ x` (матмул float32).

Это баг ядра, не квантования. Чат и KL — не эта волна.
