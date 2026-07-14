#!/usr/bin/env python3
"""E2: frequency-profiled expert selection vs the contiguous shard.

Stage U: count per-layer expert selections over the ref continuations
(hooks recompute router top-k; no masking). Stage F: mask routing to the
TOP-FREQUENCY expert set per layer at frac 0.5/0.25 and score beta on the
same refs (arms flr50/flr25; naive references lr50=0.826, lr25=0.693 at
moe/16k). Writes data/freq_lr.csv + data/expert_usage.pt.
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))
import score_accept as SA  # noqa: E402

CFG = SA.MODELS["moe"]
REFDIR = P77 / "data/refs/moe_c16384"
OUT = PHASE / "data/freq_lr.csv"
USAGE = PHASE / "data/expert_usage.pt"
(PHASE / "data").mkdir(parents=True, exist_ok=True)


class Args:
    model, ctx, prompts, gen = "moe", 16384, 12, 96


def routers_of(model):
    out = []
    for mod in model.modules():
        gate = getattr(mod, "gate", None)
        if gate is not None and all(hasattr(gate, a) for a in
                                    ("top_k", "num_experts", "weight")):
            out.append(gate)
    return out


class UsageCounter:
    def __init__(self, model):
        self.routers = routers_of(model)
        self.counts = [torch.zeros(r.num_experts, dtype=torch.long, device="cuda")
                       for r in self.routers]
        self.hooks = []

    def __enter__(self):
        def mk(li, router):
            def hook(mod, inp, out):
                hidden = inp[0].reshape(-1, mod.hidden_dim)
                logits = F.linear(hidden, mod.weight)
                idx = torch.topk(logits, mod.top_k, dim=-1).indices
                self.counts[li] += torch.bincount(
                    idx.reshape(-1), minlength=mod.num_experts)
                return None                      # keep original routing
            return hook
        for li, r in enumerate(self.routers):
            self.hooks.append(r.register_forward_hook(mk(li, r)))
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


class FreqRouterMask:
    """Mask routing to a per-layer KEEP SET (vs 77's contiguous [0:n))."""

    def __init__(self, model, keep_sets):
        self.model = model
        self.keep_sets = keep_sets               # list of LongTensor
        self.hooks = []

    def __enter__(self):
        def mk(li, router):
            drop_mask = torch.ones(router.num_experts, dtype=torch.bool,
                                   device="cuda")
            drop_mask[self.keep_sets[li]] = False

            def hook(mod, inp, out):
                hidden = inp[0].reshape(-1, mod.hidden_dim)
                logits = F.linear(hidden, mod.weight)
                logits[..., drop_mask] = float("-inf")
                probs = torch.softmax(logits, dtype=torch.float, dim=-1)
                val, idx = torch.topk(probs, mod.top_k, dim=-1)
                if mod.norm_topk_prob:
                    val = val / val.sum(-1, keepdim=True)
                return probs, val.to(probs.dtype), idx
            return hook
        for li, r in enumerate(routers_of(self.model)):
            self.hooks.append(r.register_forward_hook(mk(li, r)))
        assert self.hooks
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def collect_usage(model, args, n_prompts=4):
    """Teacher-forced steps over ref continuations with counting hooks
    (mirrors score_arm's cache mechanics, no scoring)."""
    with UsageCounter(model) as uc:
        n_layers = model.config.num_hidden_layers
        for pi in range(n_prompts):
            meta = torch.load(REFDIR / f"p{pi}_meta.pt", weights_only=False)
            kv_cpu = torch.load(REFDIR / f"p{pi}_cache.pt", weights_only=False)
            kv = {li: tuple(t.to("cuda") for t in kv_cpu[li]) for li in kv_cpu}
            T = meta["prompt_ids"].shape[1]
            for step, cur in enumerate(meta["gen_tokens"]):
                seqlen = T + step
                dcache = SA.view_cache(kv, n_layers, seqlen, window=None)
                model(input_ids=torch.tensor([[cur]], device="cuda"),
                      past_key_values=dcache, use_cache=True,
                      position_ids=torch.tensor([[seqlen]], device="cuda"),
                      cache_position=torch.tensor([dcache.get_seq_length()],
                                                  device="cuda"))
                del dcache
            del kv, kv_cpu
            print(f"[U] usage pass p{pi} done", flush=True)
        return [c.cpu() for c in uc.counts]


def append_row(row):
    exists = OUT.exists()
    with OUT.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[E2] {row['arm']}: beta={row['beta_greedy']}", flush=True)


def main() -> int:
    model = SA.load_model(CFG)
    if USAGE.exists():
        counts = torch.load(USAGE, weights_only=False)
        print("[U] loaded cached usage", flush=True)
    else:
        counts = collect_usage(model, Args())
        torch.save(counts, USAGE)
        cov = [float(c.sort(descending=True).values[:len(c) // 2].sum() / c.sum())
               for c in counts]
        print(f"[U] top-50% coverage per layer: min={min(cov):.3f} "
              f"median={sorted(cov)[len(cov)//2]:.3f} max={max(cov):.3f}",
              flush=True)
    done = set()
    if OUT.exists():
        done = {r["arm"] for r in csv.DictReader(OUT.open())}
    for frac, arm in ((0.5, "flr50"), (0.25, "flr25")):
        if arm in done:
            continue
        keep = [c.argsort(descending=True)[:max(1, int(len(c) * frac))].cuda()
                for c in counts]
        with FreqRouterMask(model, keep):
            append_row(SA.score_arm(model, CFG, arm, REFDIR, Args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
