#!/usr/bin/env python3
"""E1 GPU gates: topc (moe), mlpskip/attnskip/vres/svdr (dense).

All on 77's paired ondist refs via score_accept.score_arm (unknown arm
names fall through to a plain forward; levers applied via context managers
or destructive transforms, per the 79/83 pattern). Appends to
data/beta_pool_ext.csv.

Usage: beta_pool_ext.py --stage {moe,dense,dense_svd}
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))
import score_accept as SA  # noqa: E402

OUT = PHASE / "data/beta_pool_ext.csv"
(PHASE / "data").mkdir(parents=True, exist_ok=True)


class DArgs:
    model, ctx, prompts, gen = "dense", 16384, 12, 96


class MArgs:
    model, ctx, prompts, gen = "moe", 16384, 12, 96


def append_row(row):
    exists = OUT.exists()
    with OUT.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[pool-ext] {row['arm']}: beta={row['beta_greedy']}", flush=True)


def done_arms():
    if not OUT.exists():
        return set()
    return {r["arm"] for r in csv.DictReader(OUT.open())}


class TopC:
    """Prune the router's top-k selection to the C largest gate weights
    (the fork's draft_topc semantics), renormalized."""

    def __init__(self, model, c: int):
        self.model, self.c, self.hooks = model, c, []

    def __enter__(self):
        def mk(router):
            def hook(mod, inp, out):
                hidden = inp[0].reshape(-1, mod.hidden_dim)
                logits = F.linear(hidden, mod.weight)
                probs = torch.softmax(logits, dtype=torch.float, dim=-1)
                val, idx = torch.topk(probs, mod.top_k, dim=-1)
                kv, kp = torch.topk(val, self.c, dim=-1)
                kidx = torch.gather(idx, -1, kp)
                if mod.norm_topk_prob:
                    kv = kv / kv.sum(-1, keepdim=True)
                # pad back to top_k width with zero-weight repeats (kernel-safe)
                pad = mod.top_k - self.c
                val = torch.cat([kv, torch.zeros_like(kv[..., :1]).expand(
                    *kv.shape[:-1], pad)], dim=-1)
                idx = torch.cat([kidx, kidx[..., :1].expand(
                    *kidx.shape[:-1], pad)], dim=-1)
                return probs, val.to(probs.dtype), idx
            return hook
        for mod in self.model.modules():
            gate = getattr(mod, "gate", None)
            if gate is not None and all(hasattr(gate, a) for a in
                                        ("top_k", "num_experts", "weight")):
                self.hooks.append(gate.register_forward_hook(mk(gate)))
        assert self.hooks
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


class SubLayerSkip:
    """Zero the MLP (or attention) contribution of chosen layers -- the
    residual passes through untouched."""

    def __init__(self, model, layers, part: str):
        self.mods = []
        for i in layers:
            layer = model.model.layers[i]
            self.mods.append(getattr(layer, "mlp" if part == "mlp"
                                     else "self_attn"))
        self.orig = []

    def __enter__(self):
        for m in self.mods:
            self.orig.append(m.forward)

            def zero_fwd(*a, _m=m, **kw):
                hidden = a[0] if a else kw.get("hidden_states")
                out = torch.zeros_like(hidden)
                # attention modules return a tuple (attn_out, weights)
                return (out, None) if _m.__class__.__name__.endswith(
                    "Attention") else out
            m.forward = zero_fwd
        return self

    def __exit__(self, *a):
        for m, f in zip(self.mods, self.orig):
            m.forward = f


class VocabRestrict:
    """Mask lm_head logits outside the top-N frequency set (frequencies
    from the refs corpus: prompts + continuations)."""

    def __init__(self, model, refdir: Path, n: int):
        counts = torch.zeros(model.config.vocab_size, dtype=torch.long)
        for pi in range(12):
            meta = torch.load(refdir / f"p{pi}_meta.pt", weights_only=False)
            ids = meta["prompt_ids"][0]
            counts += torch.bincount(ids, minlength=counts.numel())
            gen = torch.tensor([int(t) for t in meta["gen_tokens"]])
            counts += torch.bincount(gen, minlength=counts.numel())
        keep = counts.argsort(descending=True)[:n]
        self.mask = torch.full((counts.numel(),), float("-inf"),
                               device="cuda")
        self.mask[keep.cuda()] = 0
        self.model = model
        self.hook = None

    def __enter__(self):
        def hook(mod, inp, out):
            return out + self.mask.to(out.dtype)
        self.hook = self.model.lm_head.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.hook.remove()


def svd_compress_(model, byte_frac: float):
    """Destructive low-rank: W ~ (U S^0.5)(S^0.5 V^T), rank chosen so the
    two factors cost byte_frac of the original bytes."""
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "lm_head" in name:
            continue
        w = mod.weight.data
        m_, n_ = w.shape
        r = max(1, int(byte_frac * m_ * n_ / (m_ + n_)))
        U, S, V = torch.svd_lowrank(w.float(), q=min(r + 16, min(m_, n_)))
        w.copy_(((U[:, :r] * S[:r]) @ V[:, :r].T).to(w.dtype))
        n += 1
    print(f"[svdr] {n} linears -> rank frac {byte_frac}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["moe", "dense", "dense_svd"])
    a = ap.parse_args()
    done = done_arms()

    if a.stage == "moe":
        cfg = SA.MODELS["moe"]
        refdir = P77 / "data/refs/moe_c16384"
        model = SA.load_model(cfg)
        for c in (4, 2):
            arm = f"topc{c}"
            if arm in done:
                continue
            with TopC(model, c):
                append_row(SA.score_arm(model, cfg, arm, refdir, MArgs()))

    elif a.stage == "dense":
        cfg = SA.MODELS["dense"]
        refdir = P77 / "data/refs/dense_c16384"
        model = SA.load_model(cfg)
        # budget-matched to the layer-set frontier: 6 MLPs ~ 4 layers' bytes
        for arm, ctxman in (
            ("mlpskip6", SubLayerSkip(model, [3, 4, 5, 6, 7, 8], "mlp")),
            ("attnskip6", SubLayerSkip(model, [3, 4, 5, 6, 7, 8], "attn")),
            ("vres16k", VocabRestrict(model, refdir, 16384)),
            ("vres32k", VocabRestrict(model, refdir, 32768)),
        ):
            if arm in done:
                continue
            with ctxman:
                append_row(SA.score_arm(model, cfg, arm, refdir, DArgs()))

    elif a.stage == "dense_svd":
        cfg = SA.MODELS["dense"]
        refdir = P77 / "data/refs/dense_c16384"
        if "svdr50" not in done:
            model = SA.load_model(cfg)
            svd_compress_(model, 0.5)
            append_row(SA.score_arm(model, cfg, "svdr50", refdir, DArgs()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
