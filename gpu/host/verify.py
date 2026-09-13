"""Acceptance for ``gpu/host``: does our algorithm really drive ``nn.Linear``?

    python gpu/host/verify.py \\
        --chr C:\\dev\\models\\qwen25-3b.nf4.chr \\
        --model-id C:\\dev\\models\\Qwen2.5-3B-Instruct

Checks (docs/spec/gpu-safety.md floor 1, plus host invariants):

  H1  CompressedLinear on the real gate_proj == CPU NF4 decode @ x, maxabs <= 0.05
      (same floor-1 threshold and the same oracle module as gpu/tests/oracle_gate.py)
  H2  after replace_linears on a meta Qwen2ForCausalLM: no Linear-shaped
      weight.numel() == out*in anywhere, and no nn.Linear left
  H3  N<=16 is real prefill (y has N columns, not a silent decode of one)
  H4  CompressedLinear.weight allocates nothing
  H5  loading real matrices: nvidia-smi delta ~ .chr size, not BF16
  H6  static: no empty_cache / CUDA graph / from_pretrained-on-weights in gpu/host
  H7  one full-model decode step, then greedy tokens (bonus, not a blocker)

Exit code: 0 PASS, 2 FAIL, 3 BLOCKED (a required check could not run).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import os
import sys
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TESTS = os.path.join(REPO, "gpu", "tests")
for _p in (REPO, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import nf4_oracle as oracle  # noqa: E402  (agent 4's CPU oracle, imported as a library)

from gpu.chr0 import load_header, materialize_nf4  # noqa: E402
from gpu.host import (  # noqa: E402
    NF4_LEVELS,
    CompressedLinear,
    Nf4Embedding,
    build_skeleton,
    linear_modules,
    load_chr_nf4,
    replace_linears,
)

MIB = 1024 * 1024
GATE = "model.layers.0.mlp.gate_proj"
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    ident: str
    what: str
    status: str
    detail: str = ""
    required: bool = True


class ShardGuard:
    """Proof, not a promise: the original BF16 shards are never opened."""

    def __init__(self, model_dir: str | None) -> None:
        self.dir = os.path.abspath(model_dir) if model_dir else None
        self.hits: list[str] = []
        self._open = None

    def __enter__(self) -> "ShardGuard":
        self._open = builtins.open
        real = self._open
        hits = self.hits

        def patched(file, mode="r", *a, **kw):  # noqa: ANN001, ANN002
            try:
                p = str(file)
                base = os.path.basename(p).lower()
                if base.endswith(".safetensors") and base.startswith("model-"):
                    hits.append(p)
                elif base in ("model.safetensors", "pytorch_model.bin"):
                    hits.append(p)
            except Exception:
                pass
            return real(file, mode, *a, **kw)

        builtins.open = patched  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        if self._open is not None:
            builtins.open = self._open  # type: ignore[assignment]
            self._open = None


def smi_used_mib(index: int = 0) -> int | None:
    try:
        import gpu_probe

        return gpu_probe.smi_used_mib(index)
    except Exception:
        return None


class Verify:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.checks: list[Check] = []
        self.notes: dict[str, object] = {}
        self.header = None

    def add(self, ident: str, what: str, ok: bool | None, detail: str = "", required: bool = True) -> None:
        self.checks.append(
            Check(ident, what, SKIP if ok is None else (PASS if ok else FAIL), detail, required)
        )

    # --- H1 -------------------------------------------------------------
    def h1_gate_oracle(self) -> None:
        ident, what = "H1", "gate_proj through CompressedLinear vs CPU oracle"
        if not os.path.isfile(self.args.chr_path):
            self.add(ident, what, None, f"missing {self.args.chr_path}")
            return

        lut_ok = all(
            oracle.f32_bits(a) == oracle.f32_bits(b) for a, b in zip(NF4_LEVELS, oracle.NF4_LEVELS)
        )
        self.add("H0", "host NF4 LUT == oracle LUT bit for bit", lut_ok, f"16 levels identical={lut_ok}")

        self.header = load_header(self.args.chr_path)
        info = self.header.tensor(self.args.name)
        m, k = info.M, info.K
        w = materialize_nf4(self.args.chr_path, self.args.name, self.args.device, header=self.header)

        layer = CompressedLinear(in_features=k, out_features=m, bias=False)
        layer.attach(w)

        g = torch.Generator(device="cpu").manual_seed(self.args.seed)
        x_ref = torch.randn((1, 1, k), generator=g, dtype=torch.float32).to(torch.bfloat16)
        x = x_ref.to(self.args.device)
        rms = float(x_ref.to(torch.float32).pow(2).mean().sqrt())

        t0 = time.perf_counter()
        y = layer(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3

        w_hat = oracle.decode_nf4(w.packed.cpu().numpy(), w.scale.cpu().numpy(), m, k)
        x_host = x_ref.to(torch.float32).numpy().reshape(k, 1)
        y_cpu = oracle.matmul_f32(w_hat, x_host)
        y_gpu = y.to(torch.float32).cpu().numpy().reshape(m, 1)
        err = (y_gpu.astype(np.float64) - y_cpu.astype(np.float64)).ravel()
        maxabs, rmse = float(np.abs(err).max()), float(np.sqrt((err * err).mean()))
        self.notes["gate_maxabs"] = maxabs
        self.notes["gate_rmse"] = rmse

        shape_ok = tuple(y.shape) == (1, 1, m) and y.dtype is torch.bfloat16
        self.add(
            ident,
            what,
            maxabs <= self.args.tol and shape_ok,
            f"{self.args.name} [{m},{k}] x[1,1,{k}] rms(x)={rms:.4f} "
            f"y{tuple(y.shape)} {y.dtype} maxabs={maxabs:.6f} rmse={rmse:.6f} "
            f"tol={self.args.tol} forward={ms:.2f}ms",
        )

        # A second call on the same layer must not disturb the packed bytes.
        y2 = layer(x)
        torch.cuda.synchronize()
        self.add(
            "H1b",
            "second forward bit-identical, packed untouched",
            bool(torch.equal(y, y2)) and w.packed.data_ptr() == layer.packed.data_ptr(),
            f"y==y2 bitwise={bool(torch.equal(y, y2))} packed ptr stable=True",
        )
        self.h3_prefill_error(layer, k)
        self.h4_weight(layer, m, k)
        del layer, w, y, y2
        torch.cuda.empty_cache()  # outside any forward: see H6

    # --- H3 -------------------------------------------------------------
    def h3_prefill_error(self, layer: CompressedLinear, k: int) -> None:
        # Wave 2 refused N!=1. Wave 3 prefill must return N columns, not one.
        rows = []
        ok = True
        m = layer.out_features
        for shape, n_expect in (((1, 4, k), 4), ((2, 1, k), 2), ((8, k), 8)):
            x = torch.zeros(shape, dtype=torch.bfloat16, device=self.args.device)
            try:
                y = layer(x)
            except Exception as exc:
                rows.append(f"{shape}: {type(exc).__name__}")
                ok = False
                continue
            want = shape[:-1] + (m,)
            if tuple(y.shape) != want:
                rows.append(f"{shape}: got {tuple(y.shape)} want {want} (silent N=1?)")
                ok = False
            else:
                rows.append(f"{shape}: y{tuple(y.shape)} N={n_expect}")
        x1 = torch.zeros((1, 1, k), dtype=torch.bfloat16, device=self.args.device)
        try:
            y1 = layer(x1)
            if tuple(y1.shape) != (1, 1, m):
                rows.append(f"N=1 shape {tuple(y1.shape)}")
                ok = False
            else:
                rows.append("N=1 still fine")
        except Exception as exc:
            rows.append(f"N=1 broke: {type(exc).__name__}")
            ok = False
        x17 = torch.zeros((1, 17, k), dtype=torch.bfloat16, device=self.args.device)
        try:
            y17 = layer(x17)
            if tuple(y17.shape) != (1, 17, m):
                rows.append(f"N=17 got {tuple(y17.shape)}")
                ok = False
            else:
                rows.append("N=17 chunked to y(1, 17, M)")
        except Exception as exc:
            rows.append(f"N=17: {type(exc).__name__}")
            ok = False
        self.add(
            "H3",
            "N<=16 is real prefill (not a silent one-column decode)",
            ok,
            "; ".join(rows),
        )

    # --- H4 -------------------------------------------------------------
    def h4_weight(self, layer: CompressedLinear, m: int, k: int) -> None:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        peak_before = torch.cuda.max_memory_allocated()
        shapes = set()
        for _ in range(256):
            w = layer.weight
            shapes.add((tuple(w.shape), w.numel(), str(w.dtype), w.device.type))
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - before
        peak_delta = torch.cuda.max_memory_allocated() - peak_before
        ok = shapes == {((0,), 0, "torch.bfloat16", "cuda")} and delta == 0 and peak_delta == 0
        self.add(
            "H4",
            "CompressedLinear.weight allocates nothing",
            ok,
            f"256 reads -> {sorted(shapes)}; alloc delta={delta} B peak delta={peak_delta} B "
            f"(M*K BF16 would be {m * k * 2 / MIB:.1f} MiB)",
        )
        try:
            layer.weight = torch.zeros(1)
            self.add("H4b", "weight is not assignable", False, "assignment silently accepted")
        except AttributeError:
            self.add("H4b", "weight is not assignable", True, "AttributeError, as intended")

    # --- H2 -------------------------------------------------------------
    def h2_skeleton(self):
        ident, what = "H2", "meta skeleton after replace_linears has no [out,in] weight"
        if not self.args.model_id or not os.path.isdir(self.args.model_id):
            self.add(ident, what, None, f"missing model dir {self.args.model_id}")
            return None
        with ShardGuard(self.args.model_id) as guard:
            model = build_skeleton(self.args.model_id)
            n_linear_before = sum(1 for _, mo in model.named_modules() if isinstance(mo, nn.Linear))
            replaced = replace_linears(model)
        offenders = []
        for name, mod in model.named_modules():
            if not (hasattr(mod, "in_features") and hasattr(mod, "out_features")):
                continue
            w = getattr(mod, "weight", None)
            if w is not None and w.numel() == mod.in_features * mod.out_features:
                offenders.append((name, type(mod).__name__, w.numel()))
        left = [n for n, mo in model.named_modules() if isinstance(mo, nn.Linear)]
        cl = linear_modules(model)
        weight_numels = {int(m.weight.numel()) for m in cl.values()}
        on_device = [
            n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t is not None and not t.is_meta and t.numel() > 0
        ]
        ok = not offenders and not left and weight_numels <= {0} and not guard.hits
        self.add(
            ident,
            what,
            ok,
            f"nn.Linear before={n_linear_before} replaced={len(replaced)} left={left or 'none'}; "
            f"CompressedLinear={len(cl)} weight.numel in {sorted(weight_numels)}; "
            f"offenders={offenders or 'none'}; non-meta tensors={on_device or 'none'}; "
            f"shards opened={guard.hits or 'none'}",
        )
        self.add(
            "H2b",
            "lm_head replaced too (tied, no second [vocab,hidden])",
            "lm_head" in cl,
            f"lm_head is {type(cl.get('lm_head')).__name__ if 'lm_head' in cl else 'NOT a CompressedLinear'}",
        )
        return model

    # --- H5 -------------------------------------------------------------
    def h5_load(self, model) -> None:
        idents = [("H5", "smi delta ~ .chr size, not BF16"), ("H5b", "layer-0 wiring is complete")]
        if model is None or not os.path.isfile(self.args.chr_path):
            for i, w in idents:
                self.add(i, w, None, "no skeleton or no .chr")
            return

        torch.zeros(1, device=self.args.device).sum().item()  # make sure the context exists
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated()
        smi_before = smi_used_mib(self.args.gpu_index)

        layers = None if self.args.full else [0]
        with ShardGuard(self.args.model_id) as guard:
            report = load_chr_nf4(
                model,
                self.args.chr_path,
                device=self.args.device,
                embed=self.args.embed,
                layers=layers,
                header=self.header,
                verbose=self.args.verbose,
            )
        torch.cuda.synchronize()
        smi_after = smi_used_mib(self.args.gpu_index)
        alloc_delta = (torch.cuda.memory_allocated() - alloc_before) / MIB
        peak_delta = (torch.cuda.max_memory_allocated() - alloc_before) / MIB

        chr_mib = os.path.getsize(self.args.chr_path) / MIB
        bf16_mib = self.bf16_model_mib(model)
        self.notes.update(
            {
                "report": str(report),
                "smi_before_mib": smi_before,
                "smi_after_mib": smi_after,
                "smi_delta_mib": None if None in (smi_before, smi_after) else smi_after - smi_before,
                "alloc_delta_mib": round(alloc_delta, 2),
                "peak_delta_mib": round(peak_delta, 2),
                "chr_mib": round(chr_mib, 1),
                "bf16_model_mib": round(bf16_mib, 1),
            }
        )

        scope = "all 36 layers + embed" if self.args.full else "layer 0 only"
        budget = chr_mib + self.args.vram_extra_mib if self.args.full else 300.0
        if self.args.embed == "bf16":
            budget += 151936 * 2048 * 2 / MIB
        if smi_before is None or smi_after is None:
            self.add(idents[0][0], idents[0][1], None, "nvidia-smi unavailable")
        else:
            delta = smi_after - smi_before
            ok = delta <= budget and alloc_delta <= budget and not guard.hits
            self.add(
                idents[0][0],
                idents[0][1],
                ok,
                f"scope={scope} smi {smi_before}->{smi_after} delta={delta} MiB; "
                f"torch alloc delta={alloc_delta:.1f} MiB peak={peak_delta:.1f} MiB; "
                f"budget={budget:.1f} MiB; .chr={chr_mib:.1f} MiB; "
                f"BF16 model would be {bf16_mib:.0f} MiB; shards opened={guard.hits or 'none'}; "
                f"{report}",
            )

        wanted = [
            f"model.layers.0.self_attn.{p}_proj" for p in ("q", "k", "v", "o")
        ] + [f"model.layers.0.mlp.{p}_proj" for p in ("gate", "up", "down")]
        mods = linear_modules(model)
        unloaded = [n for n in wanted if n not in mods or not mods[n].is_loaded]
        biased = [
            n for n in wanted
            if n in mods and mods[n].bias is not None and mods[n].bias.numel() == mods[n].M
        ]
        norm_ok = not model.model.layers[0].input_layernorm.weight.is_meta
        self.add(
            idents[1][0],
            idents[1][1],
            not unloaded and norm_ok and len(biased) == 3,
            f"unloaded={unloaded or 'none'}; q/k/v bias loaded={len(biased)}/3; "
            f"layer-0 norm off meta={norm_ok}; embed={report.embed_mode}; "
            f"leftover meta={report.leftover_meta or 'none'}",
        )
        self.notes["load_report"] = report

    @staticmethod
    def bf16_model_mib(model) -> float:
        """What the same skeleton would cost as dense BF16 (the thing we refuse)."""
        total = 0
        for _, mod in model.named_modules():
            if isinstance(mod, CompressedLinear):
                total += mod.M * mod.K * 2
            elif isinstance(mod, Nf4Embedding):
                total += mod.num_embeddings * mod.embedding_dim * 2
            elif isinstance(mod, nn.Embedding):
                total += mod.num_embeddings * mod.embedding_dim * 2
        return total / MIB

    # --- H6 -------------------------------------------------------------
    def h6_static(self) -> None:
        """Called code only, via the AST: a docstring naming a ban is not a ban."""
        banned = {
            "empty_cache": "torch.cuda.empty_cache",
            "CUDAGraph": "CUDA graph",
            "make_graphed_callables": "CUDA graph",
            "graph_pool_handle": "CUDA graph",
            "compile": "torch.compile",
            "load_state_dict": "load_state_dict",
            "tie_weights": "tie_weights",
            "cuda": "model.cuda()",
        }
        # AutoConfig/AutoTokenizer read config.json and the tokenizer, never a shard.
        allowed_owners = {"AutoConfig", "AutoTokenizer"}
        hits = []
        files = [f for f in sorted(os.listdir(HERE)) if f.endswith(".py") and f != "verify.py"]
        for fn in files:
            tree = ast.parse(open(os.path.join(HERE, fn), encoding="utf-8").read(), filename=fn)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in ("CUDAGraph", "graphs"):
                    hits.append(f"{fn}:{node.lineno}:CUDA graph")
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                owner = (
                    func.value.id
                    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    else None
                )
                if attr == "from_pretrained" and owner not in allowed_owners:
                    hits.append(f"{fn}:{node.lineno}:from_pretrained on {owner}")
                if attr in banned:
                    hits.append(f"{fn}:{node.lineno}:{banned[attr]}")
                if any(kw.arg == "device_map" for kw in node.keywords):
                    hits.append(f"{fn}:{node.lineno}:device_map")
        self.add(
            "H6",
            "no empty_cache / CUDA graph / weight loading in gpu/host",
            not hits,
            f"AST-scanned {len(files)} files ({', '.join(files)}); violations={hits or 'none'}",
        )

    # --- H8 -------------------------------------------------------------
    def h8_embed(self, model) -> None:
        """The embedding is a lookup, not a GEMM -- but it still has to decode right."""
        ident, what = "H8", "Nf4Embedding rows == CPU oracle decode"
        emb = next(
            (m for _, m in model.named_modules() if isinstance(m, Nf4Embedding)), None
        ) if model is not None else None
        if emb is None or not emb.is_loaded:
            self.add(ident, what, None, "no loaded Nf4Embedding (embed != 'rows'?)")
            return
        ids = torch.tensor(
            [[0, 1, 872, 151935, emb.num_embeddings - 1]], device=self.args.device
        )
        got = emb(ids)
        flat = ids.reshape(-1)
        rows = emb.packed.index_select(0, flat).cpu().numpy()
        scales = emb.scale.index_select(0, flat).cpu().numpy()
        want = oracle.decode_nf4(rows, scales, flat.numel(), emb.embedding_dim)
        err = got.to(torch.float32).cpu().numpy().reshape(want.shape).astype(np.float64) - want
        maxabs = float(np.abs(err).max())
        # The only difference allowed is the final float32 -> bf16 cast.
        limit = float(np.abs(want).max()) * 2.0 ** -8
        shape_ok = tuple(got.shape) == (1, 5, emb.embedding_dim) and got.dtype is torch.bfloat16
        self.add(
            ident,
            what,
            maxabs <= limit and shape_ok and emb.weight.numel() == 0,
            f"ids{tuple(ids.shape)} -> {tuple(got.shape)} {got.dtype}; maxabs={maxabs:.3e} "
            f"<= bf16 ulp bound {limit:.3e}; weight.numel()={emb.weight.numel()}; "
            f"table held as {emb.nbytes / MIB:.1f} MiB packed "
            f"(BF16 table would be {emb.num_embeddings * emb.embedding_dim * 2 / MIB:.0f} MiB)",
        )

    # --- H7 -------------------------------------------------------------
    def h7_generate(self, model) -> None:
        ident, what = "H7", "full-model decode step + greedy tokens"
        if model is None or not self.args.full or self.args.generate <= 0:
            self.add(ident, what, None, "needs --full and --generate > 0", required=False)
            return
        try:
            from transformers import AutoTokenizer, DynamicCache

            tok = AutoTokenizer.from_pretrained(self.args.model_id)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": self.args.prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # transformers 5 hands back a BatchEncoding, 4.x a bare tensor.
            ids = chat["input_ids"] if hasattr(chat, "keys") else chat
            ids = ids.to(self.args.device)

            with ShardGuard(self.args.model_id) as guard, torch.inference_mode():
                cache = DynamicCache()
                logits = None
                t0 = time.perf_counter()
                # The kernel is decode-only: the prompt is walked one token at a
                # time instead of one prefill GEMM.
                for i in range(ids.shape[1]):
                    out = model(input_ids=ids[:, i : i + 1], past_key_values=cache, use_cache=True)
                    logits = out.logits
                prefill_s = time.perf_counter() - t0
                got = []
                t1 = time.perf_counter()
                for _ in range(self.args.generate):
                    nxt = int(logits[0, -1].argmax())
                    got.append(nxt)
                    out = model(
                        input_ids=torch.tensor([[nxt]], device=self.args.device),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    logits = out.logits
                decode_s = time.perf_counter() - t1
            text = tok.decode(got)
            finite = bool(torch.isfinite(logits).all())
            self.notes["generated"] = text
            self.add(
                ident,
                what,
                finite and len(got) == self.args.generate,
                f"prompt={self.args.prompt!r} ({ids.shape[1]} tok, {prefill_s:.1f}s one-at-a-time) "
                f"-> {text!r} ({self.args.generate} tok, {self.args.generate / decode_s:.2f} tok/s); "
                f"logits finite={finite}; shards opened={guard.hits or 'none'}",
                required=False,
            )
        except Exception as exc:
            self.add(ident, what, False, f"{type(exc).__name__}: {exc}", required=False)

    # --- driver ---------------------------------------------------------
    def run(self) -> int:
        if not torch.cuda.is_available():
            self.add("GPU", "cuda available", False, "torch.cuda.is_available() is False")
            return self.report()
        name = torch.cuda.get_device_name(0)
        cap = "".join(str(v) for v in torch.cuda.get_device_capability(0))
        self.add("H-", "environment", True, f"{name} sm_{cap} torch={torch.__version__}")
        self.h6_static()
        self.h1_gate_oracle()
        model = self.h2_skeleton()
        self.h5_load(model)
        self.h8_embed(model)
        self.h7_generate(model)
        return self.report()

    def report(self) -> int:
        print("")
        print("=" * 78)
        print("deep-fold wave 2 -- gpu/host acceptance (agent 3)")
        print("=" * 78)
        print(f"chr      : {self.args.chr_path}")
        print(f"model_id : {self.args.model_id}")
        print(f"embed    : {self.args.embed}   scope: {'all layers' if self.args.full else 'layer 0'}")
        print("")
        print(f"{'ID':<5} {'STATUS':<6} {'CHECK':<48} DETAIL")
        print("-" * 78)
        for c in self.checks:
            print(f"{c.ident:<5} {c.status:<6} {c.what:<48} {c.detail}")
        print("-" * 78)
        for key in (
            "gate_maxabs",
            "gate_rmse",
            "smi_before_mib",
            "smi_after_mib",
            "smi_delta_mib",
            "alloc_delta_mib",
            "peak_delta_mib",
            "chr_mib",
            "bf16_model_mib",
            "report",
            "generated",
        ):
            if key in self.notes:
                print(f"{key:<20}: {self.notes[key]}")
        counts = {s: sum(1 for c in self.checks if c.status == s) for s in (PASS, FAIL, SKIP)}
        verdict = (
            "FAIL"
            if any(c.status == FAIL and c.required for c in self.checks)
            else "BLOCKED"
            if any(c.status == SKIP and c.required for c in self.checks)
            else "PASS"
        )
        soft = [c.ident for c in self.checks if c.status == FAIL and not c.required]
        print("")
        print(
            f"{verdict}  (pass={counts[PASS]} fail={counts[FAIL]} skip={counts[SKIP]})"
            + (f"   non-blocking failures: {soft}" if soft else "")
        )
        return {"PASS": 0, "FAIL": 2, "BLOCKED": 3}[verdict]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gpu/host acceptance")
    p.add_argument("--chr", dest="chr_path", default=r"C:\dev\models\qwen25-3b.nf4.chr")
    p.add_argument("--model-id", dest="model_id", default=r"C:\dev\models\Qwen2.5-3B-Instruct")
    p.add_argument("--name", default=GATE, help="CHR0 tensor for the H1 oracle")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-index", dest="gpu_index", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.05, help="floor-1 maxabs threshold")
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--embed", default="rows", choices=("rows", "bf16", "skip"))
    p.add_argument("--full", action="store_true", help="load all 36 layers, not just layer 0")
    p.add_argument("--generate", type=int, default=0, help="greedy tokens after --full (bonus)")
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--vram-extra-mib", dest="vram_extra_mib", type=float, default=256.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return Verify(parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
