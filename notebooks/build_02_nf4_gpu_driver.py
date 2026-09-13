"""Generate notebooks/02_nf4_gpu_driver.ipynb. Run once; this file is the source of cells."""

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

# Лабораторный прогон: наш драйвер NF4 на RTX 3080

Это **пункт 2** того же журнала, что `01_bf16_gpu_baseline.ipynb`.

Там на карту клали исходный BF16 из Hugging Face. Здесь — уже записанный `qwen25-3b.nf4.chr`: скелет на `meta`, `CompressedLinear`, fused dequant-MMA, свой цикл токена. Исходные `model-*.safetensors` **не открываем**.

Ядро: conda `torch-gpu`, CUDA, `sm_86`. Первая ячейка с `nf4_gemm` сама подхватывает `vcvars64.bat` (Jupyter не наследует VS prompt). Если исходники новее `.pyd`, пересборка займёт около минуты. Если MSVC нет — возьмёт уже собранный `.pyd`.
"""
)

md(
    """
## Что мерим

Те же величины, что в BF16-ноутбуке, плюс то, чего у несжатой модели не было.

| Метрика | Зачем |
|---|---|
| Имя GPU, драйвер, память экрана | Та же 12 ГБ карта |
| Время загрузки `.chr` | Не `from_pretrained` шардов |
| `nvidia-smi` после загрузки vs torch | Правда vs аллокатор |
| Байты packed+scale / размер `.chr` | Сжатые веса, не BF16-слой |
| Prefill, мс | До первого сгенерированного токена |
| Ток/с decode после прогрева | Хвост, без префилла в среднем |
| Латентность каждого decode-токена | Видно ли просадку / график |
| Загрузка GPU, мощность, температура, частота | Самописец каждые 0,15 с |
| Дым Paris / Париж | Тот же вопрос, что в пункте 1 |
| Нет тензора `[M,K]` BF16 | Слой не разжали в HBM |
| График VRAM + утилизация | От пусто → загрузка → ответ → выгрузка |
| Столбики vs BF16-прогон | Если лежит `runs/.../metrics.json` пункта 1 |
"""
)

code(
    r"""
# Пути на этой машине. Веса не в git.
import sys
from pathlib import Path

REPO = Path(r"C:\dev\deep-fold")
sys.path.insert(0, str(REPO))

MODEL_DIR = r"C:\dev\models\Qwen2.5-3B-Instruct"
CHR_PATH = r"C:\dev\models\qwen25-3b.nf4.chr"
BF16_METRICS = r"C:\dev\models\runs\20260912-203615\metrics.json"

PROMPT = (
    "Ответь одним коротким предложением на русском. "
    "Столица Франции?"
)
MAX_NEW_TOKENS = 64
WARMUP_TOKENS = 16
MAX_SEQ = 512
POLL_S = 0.15

_chr = Path(CHR_PATH)
print("REPO     ", REPO)
print("MODEL_DIR", MODEL_DIR)
print("CHR      ", CHR_PATH)
print("CHR size ", f"{_chr.stat().st_size / 1024**2:.1f} MiB" if _chr.is_file() else "MISSING")
"""
)

md(
    """
## Jinja2 для chat-шаблона Qwen

Как в пункте 1: `%pip` с кавычками, иначе `>=` съест шелл.
"""
)

code(
    r"""
# Quotes are required: unquoted jinja2>=3.1.0 is a shell redirect, not a version pin.
%pip install "jinja2>=3.1.0"
"""
)

md(
    """
## Железо и то, что ядро находится
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

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("bf16:", torch.cuda.is_bf16_supported())
else:
    raise SystemExit("Нужен PyTorch с CUDA (conda env torch-gpu). CPU-only сюда не годится.")

from gpu.win_toolchain import inject_msvc_env, which_cl
from gpu.nf4 import nf4_gemm  # may JIT on first import if sources are newer than the .pyd

print("cl.exe:", which_cl() or "(will inject vcvars)")
if not which_cl():
    print("inject vcvars:", inject_msvc_env(), "→", which_cl())

x = torch.zeros(64, 1, dtype=torch.bfloat16, device="cuda")
pk = torch.zeros(64, 32, dtype=torch.uint8, device="cuda")
sc = torch.ones(64, 1, dtype=torch.float16, device="cuda")
y = nf4_gemm(pk, sc, x, 64, 64, 64)
print("nf4_gemm ok", tuple(y.shape), y.dtype)
"""
)

code(
    r"""
def nvidia_smi_query(fields: str) -> list[str]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,nounits,noheader",
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    return [x.strip() for x in out.strip().split(",")]


name, total, used, driver, temp, clock, pwr = nvidia_smi_query(
    "name,memory.total,memory.used,driver_version,temperature.gpu,clocks.sm,power.draw"
)
print(f"GPU: {name}")
print(f"Драйвер: {driver}")
print(f"Память: {used} / {total} МиБ (экран и прочее уже внутри used)")
print(f"Свободно примерно: {int(float(total)) - int(float(used))} МиБ")
print(f"Сейчас: {temp} °C, SM {clock} МГц, {pwr} Вт")
print()
print("NF4 3B ~1.6 ГиБ весов. Если свободно меньше ~4 ГБ — закрой лишнее.")
if float(used) > 4000:
    raise SystemExit(
        f"nvidia-smi is {used} MiB. Kernel → Restart Kernel, then Run All. "
        "A previous model is still on the card; do not load NF4 on top of it."
    )
"""
)

md(
    """
## Самописец

Фоновый поток каждые 0,15 с снимает `nvidia-smi` и цифры аллокатора PyTorch. На загрузке, префилле, каждом этапе decode и выгрузке ставим флажки.
"""
)

code(
    r'''
class VramTimeline:
    FIELDS = (
        "memory.used,memory.total,utilization.gpu,utilization.memory,"
        "power.draw,temperature.gpu,clocks.sm,clocks.mem"
    )

    def __init__(self, interval_s: float = 0.15):
        self.interval_s = interval_s
        self.t0 = None
        self.rows = []
        self.events = []
        self._stop = threading.Event()
        self._thread = None

    def _now(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, label: str) -> None:
        if self.t0 is None:
            return
        t = self._now()
        self.events.append((t, label))
        print(f"[trace] {t:7.2f}s  {label}")

    def _f(self, s):
        if s in ("N/A", "[N/A]", "", None):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _poll(self) -> dict:
        used, total, ug, um, pwr, temp, csm, cmem = nvidia_smi_query(self.FIELDS)
        row = {
            "t": self._now(),
            "used_mib": self._f(used),
            "total_mib": self._f(total),
            "util_gpu": self._f(ug),
            "util_mem": self._f(um),
            "power_w": self._f(pwr),
            "temp_c": self._f(temp),
            "clock_sm_mhz": self._f(csm),
            "clock_mem_mhz": self._f(cmem),
        }
        if torch.cuda.is_available():
            row["torch_alloc_mib"] = torch.cuda.memory_allocated() / (1024 * 1024)
            row["torch_reserved_mib"] = torch.cuda.memory_reserved() / (1024 * 1024)
            row["torch_max_alloc_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
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
        keys = [
            "t", "used_mib", "total_mib", "util_gpu", "util_mem", "power_w",
            "temp_c", "clock_sm_mhz", "clock_mem_mhz",
            "torch_alloc_mib", "torch_reserved_mib", "torch_max_alloc_mib",
        ]
        lines = [",".join(keys)]
        for r in self.rows:
            lines.append(",".join("" if r.get(k) is None else str(r.get(k, "")) for k in keys))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _xy(ts, vs, pad_l, pad_t, w, h, tmin, tmax, ymin, ymax):
    def x(t):
        return pad_l + (t - tmin) / max(tmax - tmin, 1e-6) * w

    def y(v):
        return pad_t + (1 - (v - ymin) / max(ymax - ymin, 1e-6)) * h

    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(ts, vs) if v is not None)
    return x, y, pts


def _event_marks(tl, x, pad_t, h):
    marks = []
    for t, label in tl.events:
        xi = x(t)
        marks.append(
            f'<line x1="{xi:.1f}" y1="{pad_t}" x2="{xi:.1f}" y2="{pad_t+h}" '
            f'stroke="#888" stroke-dasharray="3 3" />'
            f'<text x="{xi:.1f}" y="{pad_t + 10}" font-size="10" fill="#444" '
            f'transform="rotate(-90 {xi:.1f},{pad_t + 12})">{label}</text>'
        )
    return "".join(marks)


def vram_svg(tl: VramTimeline, width=920, height=300) -> str:
    rows = [r for r in tl.rows if r.get("used_mib") is not None]
    if len(rows) < 2:
        return "<svg></svg>"
    ts = [r["t"] for r in rows]
    ys = [r["used_mib"] for r in rows]
    alloc = [r.get("torch_alloc_mib") for r in rows]
    total = rows[0]["total_mib"] or 12288
    tmin, tmax = ts[0], ts[-1]
    ymin, ymax = 0.0, max(total, max(ys)) * 1.05
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 40
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    x, y, pts = _xy(ts, ys, pad_l, pad_t, w, h, tmin, tmax, ymin, ymax)
    _, _, pts_a = _xy(ts, alloc, pad_l, pad_t, w, h, tmin, tmax, ymin, ymax)
    peak = max(ys)
    y_tot = y(total)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{pad_l}" y="14" font-size="12">'
        f"VRAM, MiB — nvidia-smi (peak {peak:.0f} / {total:.0f}) и torch allocated</text>"
        f'<line x1="{pad_l}" y1="{y_tot:.1f}" x2="{pad_l+w}" y2="{y_tot:.1f}" '
        f'stroke="#c44" stroke-dasharray="4 4"/>'
        f'<text x="{pad_l+2}" y="{y_tot+12:.1f}" font-size="10" fill="#c44">лимит {total:.0f}</text>'
        f'<polyline fill="none" stroke="#1f4e79" stroke-width="2" points="{pts}"/>'
        f'<polyline fill="none" stroke="#2a9d8f" stroke-width="1.5" stroke-dasharray="4 2" points="{pts_a}"/>'
        + _event_marks(tl, x, pad_t, h)
        + f'<text x="{pad_l}" y="{height-22}" font-size="11" fill="#1f4e79">nvidia-smi used</text>'
        f'<text x="{pad_l+160}" y="{height-22}" font-size="11" fill="#2a9d8f">torch allocated</text>'
        f'<text x="{pad_l}" y="{height-8}" font-size="11">время, с</text>'
        f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#333"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#333"/>'
        "</svg>"
    )


def util_svg(tl: VramTimeline, width=920, height=220) -> str:
    rows = [r for r in tl.rows if r.get("util_gpu") is not None]
    if len(rows) < 2:
        return "<svg></svg>"
    ts = [r["t"] for r in rows]
    ug = [r["util_gpu"] for r in rows]
    um = [r.get("util_mem") for r in rows]
    tmin, tmax = ts[0], ts[-1]
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 36
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    x, y, pts = _xy(ts, ug, pad_l, pad_t, w, h, tmin, tmax, 0, 100)
    _, _, pts_m = _xy(ts, um, pad_l, pad_t, w, h, tmin, tmax, 0, 100)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{pad_l}" y="14" font-size="12">Загрузка GPU и памяти контроллера, %</text>'
        f'<polyline fill="none" stroke="#e76f51" stroke-width="2" points="{pts}"/>'
        f'<polyline fill="none" stroke="#264653" stroke-width="1.5" points="{pts_m}"/>'
        + _event_marks(tl, x, pad_t, h)
        + f'<text x="{pad_l}" y="{height-8}" font-size="11" fill="#e76f51">SM util</text>'
        f'<text x="{pad_l+90}" y="{height-8}" font-size="11" fill="#264653">mem util</text>'
        f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#333"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#333"/>'
        "</svg>"
    )


def power_svg(tl: VramTimeline, width=920, height=200) -> str:
    rows = [r for r in tl.rows if r.get("power_w") is not None]
    if len(rows) < 2:
        return "<svg></svg>"
    ts = [r["t"] for r in rows]
    pw = [r["power_w"] for r in rows]
    temp = [r.get("temp_c") for r in rows]
    tmin, tmax = ts[0], ts[-1]
    pmax = max(x for x in pw if x is not None)
    tmax_c = max((x or 0) for x in temp) if temp else 0
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 36
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    x, y, pts = _xy(ts, pw, pad_l, pad_t, w, h, tmin, tmax, 0, max(pmax, 50) * 1.1)
    _, _, pts_t = _xy(ts, temp, pad_l, pad_t, w, h, tmin, tmax, 0, max(tmax_c, 40) * 1.1)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{pad_l}" y="14" font-size="12">'
        f"Мощность, Вт (пик {pmax:.0f}) и температура, °C (пик {tmax_c:.0f})</text>"
        f'<polyline fill="none" stroke="#e9c46a" stroke-width="2" points="{pts}"/>'
        f'<polyline fill="none" stroke="#c44" stroke-width="1.5" points="{pts_t}"/>'
        + _event_marks(tl, x, pad_t, h)
        + f'<text x="{pad_l}" y="{height-8}" font-size="11" fill="#e9c46a">Вт</text>'
        f'<text x="{pad_l+50}" y="{height-8}" font-size="11" fill="#c44">°C</text>'
        f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#333"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#333"/>'
        "</svg>"
    )


def tok_svg(times_ms, width=920, height=180) -> str:
    if not times_ms:
        return "<svg></svg>"
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 36
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    ymax = max(times_ms) * 1.15
    n = len(times_ms)
    bars = []
    bw = w / max(n, 1)
    for i, v in enumerate(times_ms):
        bh = (v / max(ymax, 1e-6)) * h
        x = pad_l + i * bw
        y = pad_t + h - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bw - 1, 0.5):.1f}" height="{bh:.1f}" fill="#1f4e79"/>'
        )
    mean = sum(times_ms) / n
    ymean = pad_t + h - (mean / max(ymax, 1e-6)) * h
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{pad_l}" y="14" font-size="12">'
        f"Decode, мс/токен (среднее {mean:.1f}, n={n})</text>"
        + "".join(bars)
        + f'<line x1="{pad_l}" y1="{ymean:.1f}" x2="{pad_l+w}" y2="{ymean:.1f}" '
        f'stroke="#e76f51" stroke-dasharray="4 3"/>'
        f'<line x1="{pad_l}" y1="{pad_t+h}" x2="{pad_l+w}" y2="{pad_t+h}" stroke="#333"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+h}" stroke="#333"/>'
        f'<text x="{pad_l}" y="{height-8}" font-size="11">номер токена decode</text>'
        "</svg>"
    )


def bars_svg(items, title, width=920, height=160) -> str:
    # items: list of (label, value, color)
    if not items:
        return "<svg></svg>"
    pad_l, pad_r, pad_t, pad_b = 200, 40, 24, 16
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    vmax = max(v for _, v, _ in items) * 1.15
    bh = h / len(items)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="16" y="16" font-size="12">{title}</text>'
    ]
    for i, (lab, val, col) in enumerate(items):
        y = pad_t + i * bh
        bw = (val / max(vmax, 1e-6)) * w
        parts.append(
            f'<text x="12" y="{y + bh * 0.65:.1f}" font-size="11">{lab}</text>'
            f'<rect x="{pad_l}" y="{y + 4:.1f}" width="{bw:.1f}" height="{bh - 10:.1f}" fill="{col}"/>'
            f'<text x="{pad_l + bw + 6:.1f}" y="{y + bh * 0.65:.1f}" font-size="11">{val:.0f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
'''
)

md(
    """
## Загрузка: meta-скелет + `.chr`

Порядок как в `docs/token-loop.md`: `config.json` → пустые Linear → байты NF4. Каталог шардов рядом не читаем.
"""
)

code(
    r"""
from transformers import AutoTokenizer

from gpu.host import load_model
from gpu.loop import TokenLoop

assert Path(CHR_PATH).is_file(), CHR_PATH
assert (Path(MODEL_DIR) / "config.json").is_file()

import gc

if "model" in globals():
    del model
if "loop" in globals():
    del loop
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tl = VramTimeline(interval_s=POLL_S)
tl.start()
time.sleep(0.6)

bf16 = None
metrics = {
    "gpu": nvidia_smi_query("name")[0],
    "driver": nvidia_smi_query("driver_version")[0],
    "vram_total_mib": int(float(nvidia_smi_query("memory.total")[0])),
    "vram_before_load_mib": int(float(nvidia_smi_query("memory.used")[0])),
    "prompt": PROMPT,
    "chr_path": CHR_PATH,
    "chr_mib": round(Path(CHR_PATH).stat().st_size / (1024 * 1024), 1),
    "codec": "nf4",
    "group_size": 64,
}

tl.mark("load_start")
t_load0 = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model, report = load_model(MODEL_DIR, CHR_PATH)
torch.cuda.synchronize()

metrics["load_s"] = time.perf_counter() - t_load0
tl.mark("load_end")
metrics["vram_after_load_smi_mib"] = int(float(nvidia_smi_query("memory.used")[0]))
metrics["vram_after_load_torch_mib"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
metrics["load_report"] = str(report)
metrics["nf4_linears"] = report.linears
metrics["embed_mode"] = report.embed_mode
metrics["device_weight_mib"] = round(report.device_mib, 1)

print(f"Загрузка: {metrics['load_s']:.1f} с")
print("report:", report)
print(f"nvidia-smi после загрузки: {metrics['vram_after_load_smi_mib']} МиБ")
print(f"torch allocated: {metrics['vram_after_load_torch_mib']} МиБ")
print("BF16 3B в пункте 1 после загрузки было ~7850 МиБ smi / ~5886 torch.")
"""
)

md(
    """
## Цикл токена

Прогрев (ядра, SDPA, частота карты), затем захват CUDA graph только линейных, затем замер. Prefill и decode считаем отдельно. `transformers.generate` не вызываем.
"""
)

code(
    r"""
def qwen_chat_prompt(text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def chat_ids(text: str):
    messages = [{"role": "user", "content": text}]
    try:
        packed = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as err:
        print("chat template fallback:", type(err).__name__, err)
        packed = qwen_chat_prompt(text)
    ids = tokenizer(packed, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return ids.cuda()


stop = []
if tokenizer.eos_token_id is not None:
    stop.append(int(tokenizer.eos_token_id))
if getattr(tokenizer, "pad_token_id", None) not in (None, tokenizer.eos_token_id):
    pass
# Qwen2.5-Instruct <|im_end|>
if 151645 not in stop:
    stop.append(151645)

tl.mark("loop_init")
loop = TokenLoop(model, max_seq=MAX_SEQ, norm="exact", overlap=True)
print(loop)
print("kv:", loop.kv)
print(f"weight_bytes {loop.weight_bytes / 1024**2:.1f} MiB  kv {loop.kv.nbytes / 1024**2:.1f} MiB")

tl.mark("warmup_start")
warm_ms = loop.warmup(prompt=8, tokens=WARMUP_TOKENS)
torch.cuda.synchronize()
tl.mark("warmup_end")
print(f"warmup {warm_ms:.0f} мс")

tl.mark("graph_capture_start")
gmode = loop.capture_graphs()
torch.cuda.synchronize()
tl.mark("graph_capture_end")
print("graph:", gmode, loop.graph_error or "")

ids = chat_ids(PROMPT)
prompt_len = int(ids.numel())
decode_ms_each = []
t_prev = None


def on_token(_tid: int) -> None:
    global t_prev
    now = time.perf_counter()
    if t_prev is not None:
        decode_ms_each.append((now - t_prev) * 1000.0)
    else:
        tl.mark("first_token")
    t_prev = now


tl.mark("generate_start")
t0 = time.perf_counter()
t_prev = None
gen = loop.generate(ids, MAX_NEW_TOKENS, stop=stop, on_token=on_token)
torch.cuda.synchronize()
t1 = time.perf_counter()
tl.mark("generate_end")

text = tokenizer.decode(gen.tokens, skip_special_tokens=True)
metrics.update(
    {
        "prompt_tokens": gen.prompt_len,
        "new_tokens": len(gen.tokens),
        "prefill_ms": round(gen.prefill_ms, 2),
        "ttft_s": round(gen.prefill_ms / 1000.0, 4),
        "decode_ms": round(gen.decode_ms, 2),
        "decode_steps": gen.decode_steps,
        "tok_s": round(gen.decode_tok_s, 2),
        "total_s": round(t1 - t0, 3),
        "text": text,
        "graph": gen.graph,
        "prefill_chunk": gen.prefill_chunk,
        "warmup_ms": round(warm_ms, 1),
        "kv_mib": round(loop.kv.nbytes / (1024 * 1024), 2),
        "weight_mib": round(loop.weight_bytes / (1024 * 1024), 1),
        "decode_ms_each": [round(x, 3) for x in decode_ms_each],
    }
)
metrics["vram_after_generate_smi_mib"] = int(float(nvidia_smi_query("memory.used")[0]))

print("Промпт:", PROMPT)
print("--- ответ ---")
print(text.strip())
print("---")
print(f"Токенов промпта: {gen.prompt_len}  новых: {len(gen.tokens)}")
print(f"Prefill (TTFT): {gen.prefill_ms:.1f} мс  chunk={gen.prefill_chunk}")
print(f"Decode: {gen.decode_tok_s:.1f} ток/с  ({gen.decode_ms:.0f} мс / {gen.decode_steps} шагов)")
print(f"graph={gen.graph}  smi после ответа: {metrics['vram_after_generate_smi_mib']} МиБ")
"""
)

md(
    """
## Дым качества и аудит памяти

Тот же вопрос, что в BF16-ноутбуке. Плюс проверка: на карте нет разжатой матрицы слоя.
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
    else "в ответе нет Paris/Париж — смотри текст, шаблон чата или max_new"
)
print(metrics["quality_note"])

# Аудит: нет cuda-тензора размера BF16-матрицы слоя (q/o 8 МиБ, gate ~43 МиБ).
import gc

LAYER_SHAPES = {(2048, 2048), (11008, 2048), (2048, 11008)}
suspect = []
for obj in gc.get_objects():
    try:
        if not torch.is_tensor(obj) or obj.device.type != "cuda":
            continue
        if obj.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue
        n = obj.numel()
        if n < 2048 * 2048:
            continue
        suspect.append(
            (tuple(obj.shape), str(obj.dtype), round(n * obj.element_size() / 1024**2, 2))
        )
    except Exception:
        continue

metrics["cuda_large_float_tensors"] = suspect[:20]
metrics["no_layer_bf16_w"] = not any(
    tuple(sh) in LAYER_SHAPES for sh, _dt, _mib in suspect
)
print("крупные float-тензоры на cuda (до 20):")
for row in suspect[:20]:
    print(" ", row)
print("слой [M,K] как BF16 Linear:", "нет — ок" if metrics["no_layer_bf16_w"] else "НАЙДЕН, это баг")
"""
)

md(
    """
## Выгрузка
"""
)

code(
    r"""
tl.mark("unload_start")
del loop
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

def _peak(key):
    vals = [r[key] for r in tl.rows if r.get(key) is not None]
    return round(max(vals), 1) if vals else None

metrics["peak_smi_mib"] = _peak("used_mib")
metrics["peak_torch_alloc_mib"] = _peak("torch_alloc_mib")
metrics["peak_torch_reserved_mib"] = _peak("torch_reserved_mib")
metrics["peak_util_gpu"] = _peak("util_gpu")
metrics["peak_util_mem"] = _peak("util_mem")
metrics["peak_power_w"] = _peak("power_w")
metrics["peak_temp_c"] = _peak("temp_c")
metrics["peak_clock_sm_mhz"] = _peak("clock_sm_mhz")

print(f"smi после выгрузки: {metrics['vram_after_unload_smi_mib']} МиБ")
print(f"до загрузки было:   {metrics['vram_before_load_mib']} МиБ")
print(
    f"пики: smi {metrics['peak_smi_mib']} МиБ, GPU {metrics['peak_util_gpu']}%, "
    f"{metrics['peak_power_w']} Вт, {metrics['peak_temp_c']} °C, SM {metrics['peak_clock_sm_mhz']} МГц"
)
"""
)

md(
    """
## Инфографика

Синяя линия — `nvidia-smi`. Зелёный пунктир — `torch.cuda.memory_allocated`. Красный потолок — 12288 МиБ. Второй график — загрузка SM. Столбики — мс на decode-токен. Внизу — сравнение с пунктом 1, если JSON на месте.
"""
)

code(
    r"""
out_dir = Path(r"C:\dev\models\runs") / ("nf4-" + time.strftime("%Y%m%d-%H%M%S"))
out_dir.mkdir(parents=True, exist_ok=True)

tl.save_csv(out_dir / "vram.csv")
svg_vram = vram_svg(tl)
svg_util = util_svg(tl)
svg_power = power_svg(tl)
svg_tok = tok_svg(metrics.get("decode_ms_each") or [])
(out_dir / "vram.svg").write_text(svg_vram, encoding="utf-8")
(out_dir / "util.svg").write_text(svg_util, encoding="utf-8")
(out_dir / "power.svg").write_text(svg_power, encoding="utf-8")
(out_dir / "decode_ms.svg").write_text(svg_tok, encoding="utf-8")

bf16 = None
if Path(BF16_METRICS).is_file():
    bf16 = json.loads(Path(BF16_METRICS).read_text(encoding="utf-8"))
    metrics["bf16_compare_source"] = BF16_METRICS

cmp_vram = bars_svg(
    [
        ("до загрузки, smi", metrics["vram_before_load_mib"], "#888"),
        ("NF4 после load, smi", metrics["vram_after_load_smi_mib"], "#1f4e79"),
        ("NF4 после generate, smi", metrics["vram_after_generate_smi_mib"], "#2a9d8f"),
        *(
            [("BF16 после load, smi", bf16["vram_after_load_smi_mib"], "#e76f51")]
            if bf16
            else []
        ),
        *(
            [("BF16 после generate, smi", bf16["vram_after_generate_smi_mib"], "#c44")]
            if bf16
            else []
        ),
    ],
    "Видеопамять, МиБ (меньше — лучше для схемы)",
)
cmp_tok = bars_svg(
    [
        ("NF4 decode, ток/с", metrics["tok_s"], "#1f4e79"),
        *(
            [("BF16 generate, ток/с", bf16["tok_s"], "#e76f51")]
            if bf16
            else []
        ),
    ],
    "Скорость хвоста (ток/с). BF16 из пункта 1 — transformers.generate, не тот же цикл.",
)

(out_dir / "compare_vram.svg").write_text(cmp_vram, encoding="utf-8")
(out_dir / "compare_tok.svg").write_text(cmp_tok, encoding="utf-8")

dump = {k: v for k, v in metrics.items() if k != "decode_ms_each"}
dump["decode_ms_each"] = metrics.get("decode_ms_each")
(out_dir / "metrics.json").write_text(
    json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8"
)

display(SVG(svg_vram))
display(SVG(svg_util))
display(SVG(svg_power))
display(SVG(svg_tok))
display(SVG(cmp_vram))
display(SVG(cmp_tok))
print("Сохранено в", out_dir)
"""
)

md(
    """
## Сводка
"""
)

code(
    r"""
rows = [
    ("Карта", metrics["gpu"]),
    ("Кодек", "NF4 group-64, наш драйвер"),
    (".chr, МиБ", metrics["chr_mib"]),
    ("Веса на device, МиБ", metrics["weight_mib"]),
    ("KV, МиБ", metrics["kv_mib"]),
    ("VRAM всего, МиБ", metrics["vram_total_mib"]),
    ("До загрузки, МиБ", metrics["vram_before_load_mib"]),
    ("Загрузка, с", round(metrics["load_s"], 2)),
    ("После загрузки smi, МиБ", metrics["vram_after_load_smi_mib"]),
    ("После загрузки torch, МиБ", metrics["vram_after_load_torch_mib"]),
    ("Prefill / TTFT, мс", metrics["prefill_ms"]),
    ("Промпт, токенов", metrics["prompt_tokens"]),
    ("Новых токенов", metrics["new_tokens"]),
    ("Decode, ток/с", metrics["tok_s"]),
    ("Graph", metrics["graph"]),
    ("После ответа smi, МиБ", metrics["vram_after_generate_smi_mib"]),
    ("После выгрузки smi, МиБ", metrics["vram_after_unload_smi_mib"]),
    ("Пик smi / torch alloc, МиБ", f"{metrics['peak_smi_mib']} / {metrics['peak_torch_alloc_mib']}"),
    ("Пик GPU / mem util, %", f"{metrics['peak_util_gpu']} / {metrics['peak_util_mem']}"),
    ("Пик мощность / °C / SM МГц", f"{metrics['peak_power_w']} / {metrics['peak_temp_c']} / {metrics['peak_clock_sm_mhz']}"),
    ("Слой не разжат в BF16", metrics["no_layer_bf16_w"]),
    ("Дым Paris/Париж", metrics["quality_smoke_ok"]),
]
print(f"{'метрика':<36} {'значение'}")
print("-" * 56)
for k, v in rows:
    print(f"{k:<36} {v}")
print()
print("Ответ:")
print(metrics["text"].strip())
if bf16:
    print()
    print("Пункт 1 (BF16, тот же вопрос):")
    print(f"  load smi {bf16['vram_after_load_smi_mib']}  tok/s {bf16['tok_s']:.1f}  ttft {bf16['ttft_s']:.3f} с")
    print(f"  текст: {bf16['text'].strip()}")
"""
)

md(
    """
## Как читать рядом с пунктом 1

- **Память.** NF4 должна сидеть заметно ниже BF16 (~7.8 ГиБ smi после load). Если цифры сошлись — лоадер снова материализовал слой, это баг.
- **Ток/с.** Циклы разные: там `transformers.generate` + dense GEMM, здесь fused NF4 + свой KV. Сравнивать как калибр, не как «кто честнее». Сейчас узкое место — мелкий `M` (16 блоков на 70 SM), не шина.
- **TTFT.** Здесь это префилл чанками по 16, не по одному токену промпта.
"""
)

nb.cells = cells
out = Path(__file__).with_name("02_nf4_gpu_driver.ipynb")
nbf.write(nb, out)
print("wrote", out)
