"""Generate notebooks/01_bf16_gpu_baseline.ipynb. Run once, then this file can stay as the source of cells."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


md(
    """
> Superseded by `03_codec_lab.ipynb` / Заменён на `03_codec_lab.ipynb`.

# Лабораторный прогон: несжатая 3B на RTX 3080

Это журнал эксперимента, не чат-бот и не «наша сжатая схема».

Две цели, которые мы обсуждали:

1. **База.** Положить на видеокарту *исходную* модель (как скачали с Hugging Face), задать вопрос, записать скорость, ответ и график видеопамяти от загрузки до выгрузки.
2. **Наш алгоритм.** Положить на карту *сжатый* `.chr` и считать токен так, как в схеме: тайл в регистрах, без разжатия слоя в HBM.

**Пункт 2 сейчас нельзя и не нужно пытаться.** В репозитории есть только CPU-утилита `chr` (сжать / сверить / разжать в файл). Нет CUDA-ядра, нет загрузчика `.chr` в HuggingFace, нет цикла токена. Файл `qwen25-3b.nf4.chr` — это контейнер для проверки целостности, не рантайм.

Поэтому этот ноутбук делает **только пункт 1**. Пункт 2 откладываем, пока не появятся ядро и лоадер. Ниже есть ячейка-стоп, чтобы случайно не засунуть сжатый файл на карту.
"""
)

md(
    """
## Что здесь мерим (простыми словами)

Карта: RTX 3080, 12 ГБ. Модель: Qwen2.5-3B-Instruct, исходные веса BF16 (~6,2 ГБ). Они как раз влезают; 14B в BF16 — уже нет.

| Метрика | Зачем |
|---|---|
| Имя GPU, драйвер, сколько памяти уже занято экраном | Понять, это та же 12 ГБ карта, не 10 ГБ |
| Время загрузки | Сколько ждать, пока веса переедут с диска в видеопамять |
| Память после загрузки (`nvidia-smi`) | Правда жизни: контекст CUDA + дисплей + веса. `torch` показывает меньше |
| Время до первого токена | «Подумал», это префилл промпта, не скорость болтовни |
| Токенов в секунду на хвосте | Скорость ответа после прогрева |
| Текст ответа | Чтобы глазами увидеть, не бред |
| Простая проверка качества | Для вопроса про столицу Франции — есть ли слово Paris / Париж. Это дым, не WikiText |
| График МБ видеопамяти по времени | От «ещё пусто» → загрузка → ответ → выгрузка |
| Память после `del` + `empty_cache` | Ушла ли модель или аллокатор держит дырку |

Это **не** качество сжатия. Сжатие мы уже проверили на CPU. Здесь — калибр обычного PyTorch BF16, с которым потом сравнивать наше ядро.
"""
)

md(
    """
## Пункт 2: почему не грузим `.chr` на карту

Коротко:

- Сжатый файл **1,6 ГБ** — это коды NF4. Чтобы из них получить ответ, нужно ядро «ниббл → Tensor Core». Его в коде нет, только текст в `docs/kernel-ampere.md`.
- `chr decode` пишет **12 ГБ float32**. Это *разжатый архив для проверки на диске*, не модель для чата. На карту 12 ГБ его класть нельзя: веса уже ≥ карты, плюс CUDA и KV. Получится OOM.
- HuggingFace не умеет читать `.chr`. Если вызвать `from_pretrained` на каталоге с исходником — это снова несжатая модель (пункт 1), не наш алгоритм.

Дальше по плану: CPU-тесты, при желании VQ, потом CUDA. Не «затолкать `.chr` в `model.cuda()`».
"""
)

code(
    r"""
# Каталог модели на этой машине. Веса не в git.
MODEL_DIR = r"C:\dev\models\Qwen2.5-3B-Instruct"
CHR_PATH = r"C:\dev\models\qwen25-3b.nf4.chr"
F32_DUMP = r"C:\dev\models\qwen25-3b.nf4.f32.safetensors"

PROMPT = (
    "Ответь одним коротким предложением на русском. "
    "Столица Франции?"
)
MAX_NEW_TOKENS = 64
WARMUP_TOKENS = 16

print("MODEL_DIR =", MODEL_DIR)
"""
)

code(
    r"""
# СТОП для пункта 2. Не запускает инференс. Только напоминание.
from pathlib import Path

print("Сжатый .chr существует:", Path(CHR_PATH).is_file(), CHR_PATH)
print("F32 dump существует:", Path(F32_DUMP).is_file(), F32_DUMP)
print()
print("Не делаем:")
print("  - model.cuda() на .chr  — HuggingFace это не модель")
print("  - from_pretrained(F32 dump) — ~12 ГБ, на 3080 12 ГБ это OOM")
print("  - выдавать decode(.chr) за «наш алгоритм на GPU»")
print()
print("Пункт 2 отложен, пока нет CUDA-ядра и лоадера CompressedLinear.")
"""
)

md(
    """
## Проверка железа и PyTorch

Нужен PyTorch **с CUDA**, не `+cpu`. Если следующая ячейка скажет `cuda=False`, пункт 1 не гонять: модель уедет в оперативку и будет ползти минутами, это не замер карты.
"""
)

code(
    r"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch
from IPython.display import SVG, display

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
else:
    print()
    print("Сейчас стоит CPU-only PyTorch. Карта в nvidia-smi есть,")
    print("но этот Python её не видит. Пункт 1 не запускать, пока не поставлен")
    print("CUDA-пакет, например:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print("После установки проверь: python -c \"import torch; print(torch.cuda.is_available())\"")
"""
)

code(
    r"""
def nvidia_smi_query(fields: str) -> list[str]:
    # Один снимок nvidia-smi. МБ как в драйвере, не как в torch.
    cmd = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,nounits,noheader",
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    return [x.strip() for x in out.strip().split(",")]


name, total, used, driver = nvidia_smi_query(
    "name,memory.total,memory.used,driver_version"
)
print(f"GPU: {name}")
print(f"Драйвер: {driver}")
print(f"Память всего: {total} МиБ")
print(f"Уже занято (экран / другие программы): {used} МиБ")
print(f"Свободно примерно: {int(total) - int(used)} МиБ")
print()
print("Для 3B BF16 нужно порядка 6–8 ГБ с запасом на KV и CUDA.")
print("Если свободно меньше ~8 ГБ — закрой лишнее или не гоняй этот ноутбук.")
if float(used) > 4000:
    raise SystemExit(
        f"nvidia-smi is {used} MiB. Kernel → Restart Kernel, then Run All. "
        "A previous model is still on the card; do not load a second one on top."
    )
"""
)

md(
    """
## Самописец видеопамяти

Фоновый поток раз в 0,2 с спрашивает `nvidia-smi`: сколько МБ занято и загрузка GPU. В важные моменты ставим флажки: начали грузить, загрузили, первый токен, конец ответа, выгрузили. Потом рисуем график с этими флажками.
"""
)

code(
    r'''
class VramTimeline:
    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self.t0 = None
        self.rows = []  # dicts
        self.events = []  # (t, label)
        self._stop = threading.Event()
        self._thread = None

    def _now(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, label: str) -> None:
        if self.t0 is None:
            return
        t = self._now()
        self.events.append((t, label))
        print(f"[vram] {t:7.2f}s  {label}")

    def _poll(self) -> dict:
        used, total, util, power = nvidia_smi_query(
            "memory.used,memory.total,utilization.gpu,power.draw"
        )
        row = {
            "t": self._now(),
            "used_mib": float(used),
            "total_mib": float(total),
            "util": float(util),
            "power_w": float(power) if power not in ("N/A", "[N/A]") else None,
        }
        if torch.cuda.is_available():
            row["torch_alloc_mib"] = torch.cuda.memory_allocated() / (1024 * 1024)
            row["torch_reserved_mib"] = torch.cuda.memory_reserved() / (1024 * 1024)
        return row

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.rows.append(self._poll())
            except Exception as e:
                self.rows.append({"t": self._now(), "error": str(e)})
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.rows.clear()
        self.events.clear()
        self._stop.clear()
        self.rows.append(self._poll())
        self.mark("start")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.rows.append(self._poll())
        self.mark("stop")

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ["t", "used_mib", "total_mib", "util", "power_w", "torch_alloc_mib", "torch_reserved_mib"]
        lines = [",".join(keys)]
        for r in self.rows:
            lines.append(",".join("" if r.get(k) is None else str(r.get(k, "")) for k in keys))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def vram_svg(tl: VramTimeline, width=920, height=280) -> str:
    rows = [r for r in tl.rows if "used_mib" in r]
    if len(rows) < 2:
        return "<svg></svg>"
    ts = [r["t"] for r in rows]
    ys = [r["used_mib"] for r in rows]
    total = rows[0]["total_mib"]
    tmin, tmax = ts[0], ts[-1]
    ymin, ymax = 0.0, max(total, max(ys)) * 1.05
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 36
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b

    def x(t):
        return pad_l + (t - tmin) / max(tmax - tmin, 1e-6) * w

    def y(v):
        return pad_t + (1 - (v - ymin) / max(ymax - ymin, 1e-6)) * h

    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(ts, ys))
    marks = []
    for t, label in tl.events:
        xi = x(t)
        marks.append(
            f'<line x1="{xi:.1f}" y1="{pad_t}" x2="{xi:.1f}" y2="{pad_t+h}" '
            f'stroke="#888" stroke-dasharray="3 3" />'
            f'<text x="{xi:.1f}" y="{pad_t + 10}" font-size="10" fill="#444" '
            f'transform="rotate(-90 {xi:.1f},{pad_t + 12})" >{label}</text>'
        )
    peak = max(ys)
    y_tot = y(total)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{pad_l}" y="14" font-size="12">'
        f"VRAM nvidia-smi, MiB (peak {peak:.0f} / {total:.0f})</text>"
        f'<line x1="{pad_l}" y1="{y_tot:.1f}" x2="{pad_l+w}" y2="{y_tot:.1f}" '
        f'stroke="#c44" stroke-dasharray="4 4"/>'
        f'<text x="{pad_l+2}" y="{y_tot+12:.1f}" font-size="10" fill="#c44">'
        f"GPU limit {total:.0f}</text>"
        f'<polyline fill="none" stroke="#1f4e79" stroke-width="2" points="{pts}"/>'
        + "".join(marks)
        + f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#333"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#333"/>'
        f'<text x="{pad_l}" y="{height-8}" font-size="11">time, s (0 = recorder start)</text>'
        "</svg>"
    )
    return svg
'''
)

md(
    """
## Пункт 1. Загрузка несжатой модели на карту

Дальше ячейки сами останавливаются, если CUDA нет. Если CUDA есть — грузим локальный каталог, без интернета.
"""
)

code(
    r"""
from transformers import AutoModelForCausalLM, AutoTokenizer

CUDA_OK = torch.cuda.is_available()
if not CUDA_OK:
    raise SystemExit(
        "Стоп: PyTorch без CUDA. Пункт 1 не гоняем. См. ячейку с pip cu124."
    )

assert Path(MODEL_DIR).is_dir(), MODEL_DIR
assert (Path(MODEL_DIR) / "config.json").is_file()

import gc

if "model" in globals():
    del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tl = VramTimeline(interval_s=0.2)
tl.start()
time.sleep(0.6)  # чуть пустой базы на графике

metrics = {
    "gpu": nvidia_smi_query("name")[0],
    "vram_total_mib": int(float(nvidia_smi_query("memory.total")[0])),
    "vram_before_load_mib": int(float(nvidia_smi_query("memory.used")[0])),
    "prompt": PROMPT,
}

tl.mark("load_start")
t_load0 = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,
    attn_implementation="sdpa",
)
model.eval()
torch.cuda.synchronize()

metrics["load_s"] = time.perf_counter() - t_load0
tl.mark("load_end")
metrics["vram_after_load_smi_mib"] = int(float(nvidia_smi_query("memory.used")[0]))
metrics["vram_after_load_torch_mib"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)

print(f"Загрузка: {metrics['load_s']:.1f} с")
print(f"nvidia-smi после загрузки: {metrics['vram_after_load_smi_mib']} МиБ")
print(f"torch allocated: {metrics['vram_after_load_torch_mib']} МиБ")
print("Разница smi−torch ≈ CUDA-контекст + кэш аллокатора + дисплей.")
"""
)

md(
    """
## Вопрос и ответ

Считаем отдельно:

- **до первого токена** — модель проглотила вопрос (префилл);
- **ток/с на хвосте** — уже болтовня.

Первые несколько токенов после загрузки не мешаем в «официальную» скорость: карта ещё разгоняется. После короткого прогрева — замер.
"""
)

code(
    r"""
def qwen_chat_prompt(text: str) -> str:
    # Same layout as Qwen2.5's default template, without Jinja.
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def chat_inputs(text: str):
    messages = [{"role": "user", "content": text}]
    try:
        packed = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as err:
        print("chat template fallback:", type(err).__name__, err)
        packed = qwen_chat_prompt(text)
    return tokenizer(packed, return_tensors="pt").to(model.device)


@torch.inference_mode()
def generate_timed(prompt: str, max_new: int, do_warmup: bool):
    inputs = chat_inputs(prompt)
    prompt_len = int(inputs["input_ids"].shape[1])

    if do_warmup:
        tl.mark("warmup_start")
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        tl.mark("warmup_end")

    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kw = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        streamer=streamer,
    )

    tl.mark("generate_start")
    t0 = time.perf_counter()
    first = None
    pieces = []

    def run():
        model.generate(**gen_kw)

    th = threading.Thread(target=run)
    th.start()
    for piece in streamer:
        if not piece:
            continue
        if first is None:
            first = time.perf_counter()
            tl.mark("first_token")
        pieces.append(piece)
    th.join()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    tl.mark("generate_end")

    text = "".join(pieces)
    # точный счёт новых токенов
    full = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_new = max(len(full), 1)
    ttft = (first - t0) if first else None
    decode_s = (t1 - first) if first else (t1 - t0)
    tok_s = (n_new - 1) / decode_s if first and decode_s > 0 and n_new > 1 else n_new / (t1 - t0)

    return {
        "prompt_tokens": prompt_len,
        "new_tokens": n_new,
        "ttft_s": ttft,
        "total_s": t1 - t0,
        "tok_s": tok_s,
        "text": text,
    }


out = generate_timed(PROMPT, MAX_NEW_TOKENS, do_warmup=True)
metrics.update(out)
metrics["vram_after_generate_smi_mib"] = int(float(nvidia_smi_query("memory.used")[0]))

print("Промпт:", PROMPT)
print("--- ответ ---")
print(out["text"].strip())
print("---")
print(f"Токенов в вопросе (после шаблона чата): {out['prompt_tokens']}")
print(f"Новых токенов: {out['new_tokens']}")
print(f"До первого токена: {out['ttft_s']:.3f} с" if out["ttft_s"] else "TTFT неизвестен")
print(f"Скорость после первого токена: {out['tok_s']:.1f} ток/с")
print(f"nvidia-smi после ответа: {metrics['vram_after_generate_smi_mib']} МиБ")
"""
)

md(
    """
## Качество ответа (очень простое)

Мы **не** считаем WikiText и не сравниваем логиты с эталоном. Для этого прогона:

- напечатали текст;
- для вопроса про столицу Франции проверяем, есть ли Paris / Париж.

Если слова нет — это либо модель съехала, либо шаблон чата кривой, либо слишком жёсткий `max_new_tokens`. Не чинить сжатие: здесь ещё несжатая модель.
"""
)

code(
    r"""
text_l = metrics["text"].lower()
needles = ["paris", "париж"]
metrics["quality_smoke_ok"] = any(n in text_l for n in needles)
metrics["quality_note"] = (
    "дым: в ответе есть Paris/Париж"
    if metrics["quality_smoke_ok"]
    else "в ответе нет Paris/Париж — смотри текст выше, это не метрика сжатия"
)
print(metrics["quality_note"])
"""
)

md(
    """
## Выгрузка с карты

Удаляем модель, чистим кэш CUDA, смотрим, упала ли кривая на графике. На 3080 с монитором «ноль» не будет: драйвер и рабочий стол оставляют сотни мегабайт.
"""
)

code(
    r"""
tl.mark("unload_start")
del model
del tokenizer
import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
time.sleep(0.8)
metrics["vram_after_unload_smi_mib"] = int(float(nvidia_smi_query("memory.used")[0]))
tl.mark("unload_end")
time.sleep(0.5)
tl.stop()

print(f"nvidia-smi после выгрузки: {metrics['vram_after_unload_smi_mib']} МиБ")
print(f"До загрузки было: {metrics['vram_before_load_mib']} МиБ")
"""
)

md(
    """
## График видеопамяти

Синяя линия — `nvidia-smi memory.used`. Пунктир сверху — 12288 МиБ. Вертикальные подписи — события (load / first_token / unload).
"""
)

code(
    r"""
out_dir = Path(r"C:\dev\models\runs") / time.strftime("%Y%m%d-%H%M%S")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "vram.csv"
svg_path = out_dir / "vram.svg"
json_path = out_dir / "metrics.json"

tl.save_csv(csv_path)
svg = vram_svg(tl)
svg_path.write_text(svg, encoding="utf-8")
json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

display(SVG(svg))
print("Сохранено в", out_dir)
print("  ", csv_path.name)
print("  ", svg_path.name)
print("  ", json_path.name)
"""
)

md(
    """
## Сводка одним взглядом
"""
)

code(
    r"""
rows = [
    ("Карта", metrics["gpu"]),
    ("VRAM всего, МиБ", metrics["vram_total_mib"]),
    ("До загрузки, МиБ", metrics["vram_before_load_mib"]),
    ("Загрузка, с", round(metrics["load_s"], 2)),
    ("После загрузки smi, МиБ", metrics["vram_after_load_smi_mib"]),
    ("После загрузки torch, МиБ", metrics["vram_after_load_torch_mib"]),
    ("До первого токена, с", None if metrics["ttft_s"] is None else round(metrics["ttft_s"], 3)),
    ("Новых токенов", metrics["new_tokens"]),
    ("Ток/с после первого", round(metrics["tok_s"], 1)),
    ("После ответа smi, МиБ", metrics["vram_after_generate_smi_mib"]),
    ("После выгрузки smi, МиБ", metrics["vram_after_unload_smi_mib"]),
    ("Дым Paris/Париж", metrics["quality_smoke_ok"]),
]
print(f"{'метрика':<32} {'значение'}")
print("-" * 52)
for k, v in rows:
    print(f"{k:<32} {v}")
print()
print("Ответ:")
print(metrics["text"].strip())
"""
)

md(
    """
## Что дальше (без GPU-сжатия)

Пункт 2 не открываем, пока нет этих вещей — их ещё писать:

1. CUDA-ядро fused dequant-MMA (`docs/kernel-ampere.md`)
2. Лоадер: скелет HuggingFace + `CompressedLinear`, без `from_pretrained` BF16
3. Свой цикл токена, не `transformers.generate` как единственный замер

Пока это не готово, полезнее на CPU: VQ 2×8 на той же 3B, ещё проверки `chr`, не «затолкать `.chr` на карту».
"""
)

nb.cells = cells
out = Path(__file__).with_name("01_bf16_gpu_baseline.ipynb")
nbf.write(nb, out)
print("wrote", out)
