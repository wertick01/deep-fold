"""One comparison lab per downloaded model: BF16 then NF4, never both in VRAM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabModel:
    slug: str
    notebook: str
    title: str
    model_dir: str
    chr_path: str
    bf16_fits: bool
    nf4_fits: bool
    nf4_driver: str  # "qwen2" | "internlm2" | "unsupported"
    trust_remote_code: bool
    note: str
    compress_cmd: str

    @property
    def plate_title(self) -> str:
        return f"{self.title} on one RTX 3080 12 GB — NF4 driver against dense BF16"


SUPPORTED_NF4 = frozenset({"qwen2", "internlm2"})

LABS: tuple[LabModel, ...] = (
    LabModel(
        slug="qwen25-3b",
        notebook="03_codec_lab.ipynb",
        title="Qwen2.5-3B-Instruct",
        model_dir=r"C:\dev\models\Qwen2.5-3B-Instruct",
        chr_path=r"C:\dev\models\qwen25-3b.nf4.chr",
        bf16_fits=True,
        nf4_fits=True,
        nf4_driver="qwen2",
        trust_remote_code=False,
        note="Fits both ways on a 3080 12 GB. This is the live comparison.",
        compress_cmd="",
    ),
    LabModel(
        slug="qwen25-14b",
        notebook="04_qwen25_14b_lab.ipynb",
        title="Qwen2.5-14B-Instruct",
        model_dir=r"C:\dev\models\Qwen2.5-14B-Instruct",
        chr_path=r"C:\dev\models\qwen25-14b.nf4.chr",
        bf16_fits=False,
        nf4_fits=True,
        nf4_driver="qwen2",
        trust_remote_code=False,
        note=(
            "BF16 weights ~28 GiB: Phase A fills all 12 GB VRAM and spills into "
            "shared GPU memory (system RAM). nvidia-smi will sit near 12288; "
            "torch.cuda.memory_allocated is the real working set. That OOM-or-spill "
            "is the measurement. NF4 leftover is ~3.5 GiB per vram-3080.md."
        ),
        compress_cmd=(
            r"C:\dev\deep-fold\chr.exe compress --in C:\dev\models\Qwen2.5-14B-Instruct "
            r"--out C:\dev\models\qwen25-14b.nf4.chr --codec nf4 --quiet"
        ),
    ),
    LabModel(
        slug="internlm20b",
        notebook="05_internlm20b_lab.ipynb",
        title="internlm2_5-20b-chat",
        model_dir=r"C:\dev\models\internlm2_5-20b-chat",
        chr_path=r"C:\dev\models\internlm2_5-20b.nf4.chr",
        bf16_fits=False,
        nf4_fits=False,
        nf4_driver="internlm2",
        trust_remote_code=True,
        note=(
            "Negative control on 12 GB. BF16 ~37 GiB. NF4 weights ~11 GiB; "
            "try the standard NF4 .chr first. TokenLoop splits fused wqkv "
            "(gs = 2 + n_q/n_kv). HuggingFace remote code needs einops and "
            "sentencepiece==0.1.99."
        ),
        compress_cmd=(
            r"C:\dev\deep-fold\chr.exe compress --in C:\dev\models\internlm2_5-20b-chat "
            r"--out C:\dev\models\internlm2_5-20b.nf4.chr --codec nf4 --quiet"
        ),
    ),
)


def lab_by_slug(slug: str) -> LabModel:
    """Catalog row for notebooks 03/04/05/06. Unknown slug fails closed."""
    for lab in LABS:
        if lab.slug == slug:
            return lab
    known = ", ".join(lab.slug for lab in LABS)
    raise KeyError(f"unknown lab slug {slug!r}; known: {known}")
