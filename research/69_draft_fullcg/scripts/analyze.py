#!/usr/bin/env python
"""Phase 69: print tok/s (accept) + F/D linear solve from w7_2node JSONs.

cycle_ms = accept_len * batch * 1000 / tok_s;  cycle = F + K*D.
Usage: analyze.py <tag_glob> [batch]   e.g. analyze.py baseline_clean 8
"""
import glob
import json
import sys

PHASE = "/h/v-sukmincho/self-spec-moe/research/69_draft_fullcg/data"


def load(tag):
    rows = {}
    for f in glob.glob(f"{PHASE}/*{tag}*spec_cg*.json"):
        d = json.load(open(f))
        for k, pts in d.get("by_k", {}).items():
            for pt in pts:
                b = pt.get("batch")
                ts = pt.get("tok_s_mean")
                acc = pt.get("accept_len")
                if b and ts and acc:
                    rows[(int(k), int(b))] = (float(ts), float(acc))
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "fullcg_clean"
    rows = load(tag)
    if not rows:
        # fall back: just dump any matching files raw
        for f in sorted(glob.glob(f"{PHASE}/*{tag}*.json")):
            print(f, "->", json.load(open(f)))
        return
    print(f"tag={tag}")
    for (k, b), (ts, acc) in sorted(rows.items()):
        cyc = acc * b * 1000 / ts
        print(f"  K{k} b{b}: {ts:8.1f} tok/s  accept={acc:.3f}  cycle={cyc:.1f} ms")
    # F/D solve per batch from K pairs
    batches = sorted({b for _, b in rows})
    for b in batches:
        ks = sorted(k for k, bb in rows if bb == b)
        if len(ks) >= 2:
            k1, k2 = ks[0], ks[-1]
            ts1, a1 = rows[(k1, b)]
            ts2, a2 = rows[(k2, b)]
            c1 = a1 * b * 1000 / ts1
            c2 = a2 * b * 1000 / ts2
            D = (c2 - c1) / (k2 - k1)
            F = c1 - k1 * D
            print(f"  b{b} F/D solve (K{k1},K{k2}): F={F:.1f} ms  D={D:.2f} ms/step")


if __name__ == "__main__":
    main()
