"""Phase 59: assemble the context x {no-spec, EAGLE K1, K2} tok/s + speedup table.

2k reused from Phase 57 (chat, on-dist, EP16); 16k/32k from Phase 59 data.
"""
import json
import os

P57 = "/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy/data"
P59 = "/h/v-sukmincho/self-spec-moe/research/59_multicontext/data"
BATCHES = [8, 32, 64]


def load_rows(path):
    """Return {batch: (tok_s, accept_len)} from a w7_2node result JSON."""
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    res = d.get("results")
    if not res:
        bk = d.get("by_k", {})
        res = next(iter(bk.values()), []) if bk else []
    out = {}
    for r in res:
        if "batch" in r and "tok_s_mean" in r:
            out[r["batch"]] = (r["tok_s_mean"], r.get("accept_len"))
    return out


# context -> {mode -> path}
SRC = {
    "2k": {
        "nospec": f"{P57}/w72n_q30b_nospec_2n_chat_nospec_cg_nospec.json",
        "K1": f"{P57}/w72n_q30b_eagle_2n_chat_spec_cg_K1.json",
        "K2": f"{P57}/w72n_q30b_eagle_2n_chat_spec_cg_K2.json",
    },
    "16k": {
        "nospec": f"{P59}/w72n_q30b_nospec_2n_ctx16k_nospec_cg_nospec.json",
        "K1": f"{P59}/w72n_q30b_eagle_2n_ctx16k_spec_cg_K1.json",
        "K2": f"{P59}/w72n_q30b_eagle_2n_ctx16k_spec_cg_K2.json",
    },
    "32k": {
        "nospec": f"{P59}/w72n_q30b_nospec_2n_ctx32k_nospec_cg_nospec.json",
        "K1": f"{P59}/w72n_q30b_eagle_2n_ctx32k_spec_cg_K1.json",
        "K2": f"{P59}/w72n_q30b_eagle_2n_ctx32k_spec_cg_K2.json",
    },
}

data = {ctx: {m: load_rows(p) for m, p in modes.items()}
        for ctx, modes in SRC.items()}

print("\n=== tok/s (accept_len) ===")
hdr = f"{'ctx':>5} {'batch':>6} {'nospec':>10} {'K1':>16} {'K2':>16}"
print(hdr)
for ctx in ("2k", "16k", "32k"):
    for b in BATCHES:
        ns = data[ctx]["nospec"].get(b, (None, None))[0]
        k1 = data[ctx]["K1"].get(b, (None, None))
        k2 = data[ctx]["K2"].get(b, (None, None))
        f = lambda v: f"{v:.1f}" if v is not None else "--"
        fa = lambda t: (f"{t[0]:.1f} ({t[1]:.2f})" if t[0] is not None
                        and t[1] is not None else (f(t[0]) if t[0] else "--"))
        print(f"{ctx:>5} {b:>6} {f(ns):>10} {fa(k1):>16} {fa(k2):>16}")

print("\n=== EAGLE speedup vs no-spec (tok/s ratio) ===")
print(f"{'ctx':>5} {'batch':>6} {'K1x':>7} {'K2x':>7} {'best':>7}")
for ctx in ("2k", "16k", "32k"):
    for b in BATCHES:
        ns = data[ctx]["nospec"].get(b, (None,))[0]
        k1 = data[ctx]["K1"].get(b, (None,))[0]
        k2 = data[ctx]["K2"].get(b, (None,))[0]
        if not ns:
            print(f"{ctx:>5} {b:>6}   (no baseline)")
            continue
        s1 = k1 / ns if k1 else None
        s2 = k2 / ns if k2 else None
        best = max([x for x in (s1, s2) if x], default=None)
        g = lambda v: f"{v:.2f}x" if v else "--"
        print(f"{ctx:>5} {b:>6} {g(s1):>7} {g(s2):>7} {g(best):>7}")
