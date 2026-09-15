"""Which VQ tensors kill greedy smoke: patch BF16 3B from the existing .vq2.chr.

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe gpu/lab/vq_which_layers.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpu.chr0 import load_header
from gpu.host.vq_blobs import materialize_vq, reconstruct_vq
from gpu.cli.paths import models_root
from gpu.lab.script import MESSAGES, chat_text, quality_ok, stop_token_ids

MODEL_DIR = os.environ.get("DEEPFOLD_MODEL", str(models_root() / "Qwen2.5-3B-Instruct"))
CHR = os.environ.get("DEEPFOLD_VQ_CHR", str(models_root() / "qwen25-3b.vq2.chr"))
MAX_NEW = 32


def _kind(name: str) -> str:
    last = name.rsplit(".", 1)[-1]
    for token, kind in (
        ("q_proj", "q"),
        ("k_proj", "k"),
        ("v_proj", "v"),
        ("o_proj", "o"),
        ("gate_proj", "gate"),
        ("up_proj", "up"),
        ("down_proj", "down"),
        ("embed_tokens", "embed"),
        ("lm_head", "lm_head"),
    ):
        if last == token or name.endswith(token):
            return kind
    return last


@torch.no_grad()
def _patch(model, chr_path: str, header, pred) -> int:
    n = 0
    named = dict(model.named_modules())
    for name in list(header.tensors):
        info = header.tensors[name]
        if info.codec != "vq":
            continue
        if not pred(name, _kind(name)):
            continue
        mod = named.get(name)
        if mod is None or not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        mat = materialize_vq(chr_path, name, "cpu", header=header)
        hat = reconstruct_vq(mat.index, mat.book, mat.M, mat.K, mat.K_pad)
        mod.weight.data.copy_(hat.to(dtype=mod.weight.dtype, device=mod.weight.device))
        n += 1
        del hat, mat
    return n


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, device: torch.device) -> str:
    packed = chat_text(tokenizer, prompt)
    ids = tokenizer(packed, return_tensors="pt").input_ids.to(device)
    stop = list(stop_token_ids(tokenizer))
    out = model.generate(
        ids,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        eos_token_id=stop,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def _smoke(model, tok, device) -> tuple[int, list[str]]:
    texts = []
    hits = 0
    for i, prompt in enumerate(MESSAGES, 1):
        text = _generate(model, tok, prompt, device)
        texts.append(text)
        hits += int(quality_ok(i, text))
    return hits, texts


def main() -> int:
    device = torch.device("cuda")
    hdr = load_header(CHR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    print(f"loading BF16 {MODEL_DIR}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, local_files_only=True
    )
    base.to(device).eval()

    variants = [
        ("attn qkvo", lambda n, k: k in {"q", "k", "v", "o"}),
        ("mlp gate/up/down", lambda n, k: k in {"gate", "up", "down"}),
        ("down only", lambda n, k: k == "down"),
        ("all except down", lambda n, k: k in {"q", "k", "v", "o", "gate", "up"}),
        ("all linears (no embed)", lambda n, k: k in {"q", "k", "v", "o", "gate", "up", "down"}),
    ]
    # Restore from a CPU clone of original weights between variants.
    orig = {n: m.weight.detach().cpu().clone() for n, m in base.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}

    def restore():
        for n, m in base.named_modules():
            if n in orig:
                m.weight.data.copy_(orig[n].to(device=m.weight.device, dtype=m.weight.dtype))

    hits0, texts0 = _smoke(base, tok, device)
    print(f"\nBF16 baseline {hits0}/3  {texts0[0][:80]!r}", flush=True)

    for title, pred in variants:
        restore()
        n = _patch(base, CHR, hdr, pred)
        hits, texts = _smoke(base, tok, device)
        preview = texts[0].replace("\n", " ")[:90]
        print(f"{title:24s} patched={n:3d}  smoke={hits}/3  {preview!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
