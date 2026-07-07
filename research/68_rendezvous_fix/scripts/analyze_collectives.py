"""Phase 68: per-cycle cross-node collective inventory from a rank-0 torch trace.

The decisive attribution for the DP-rendezvous hypothesis. Every GPU NCCL
collective kernel is attributed to the DRAFT-step path vs the VERIFY/target
path by CORRELATION: each kernel's `correlation` id is matched to its CPU-side
launch (cudaLaunchKernel / cudaGraphLaunch / cuLaunchKernelEx), and the launch's
timestamp is nested in the runner's named record_function spans (draft /
forward / preprocess / ...). Attributing by the GPU kernel's own timestamp is
WRONG here: async execution slides verify kernels into the neighbouring CPU
draft window. CPU gloo/nccl coordination all-reduces (DP rendezvous) are counted
from their user_annotation events, same nesting.

Answers: (1) collectives/cycle on draft vs verify path; (2) draft-side ms/cycle
(does it reconcile ~56 ms?); (3) is SKIP_DP_COORD engaged (draft coordinate
all-reduce count == 0?).

Usage: analyze_collectives.py TRACE_DIR_OR_FILE
"""

import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def pick(path):
    if os.path.isfile(path):
        return path
    paths = sorted(glob.glob(f"{path}/**/*.json*", recursive=True))
    if not paths:
        sys.exit(f"no trace files under {path}")
    return max(paths, key=os.path.getsize)


SPAN_NAMES = ("draft", "forward", "preprocess", "postprocess", "sample",
              "bookkeep")


def cclass(name):
    for k in ("AllGather", "ReduceScatter", "AllReduce", "Broadcast",
              "SendRecv", "AllToAll"):
        if k.lower() in name.lower():
            return k
    return "other"


def main(path):
    path = pick(path)
    print(f"# {path}")
    ev = load(path)["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]

    spans = defaultdict(list)
    for e in X:
        n = e.get("name", "")
        if n.startswith("gpu_model_runner: "):
            spans[n.split(": ", 1)[1]].append((e["ts"], e["ts"] + e["dur"]))
    # sort span windows for bisect-free lookup
    for k in spans:
        spans[k].sort()

    def span_of(ts):
        for name in SPAN_NAMES:
            for t0, t1 in spans.get(name, []):
                if t0 <= ts < t1:
                    return name
        return "other"

    # cycles = number of verify forwards = AllGather_in_forward / n_moe_layers,
    # but robustly use the count of `draft` spans that actually contain a
    # target propose. Use forward-attributed AllReduce count as cycle proxy;
    # fall back to distinct sample spans.
    n_sample = len(spans.get("sample", []))
    n_draft = len(spans.get("draft", []))

    # correlation -> CPU launch event
    rt_by_corr = {}
    for e in X:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                rt_by_corr[c] = e

    nccl = [e for e in X if e.get("cat") == "kernel"
            and "nccl" in e.get("name", "").lower()]

    # class x span -> [n, ms]
    cs = defaultdict(lambda: [0, 0.0])
    unmapped = Counter()
    for k in nccl:
        c = k.get("args", {}).get("correlation")
        r = rt_by_corr.get(c)
        cl = cclass(k["name"])
        if r is None:
            unmapped[cl] += 1
            s = "UNMAPPED"
        else:
            s = span_of(r["ts"])
        a = cs[(cl, s)]
        a[0] += 1
        a[1] += k["dur"] / 1e3

    # cycle count = number of coordinate AllReduce kernels (exactly one DP
    # coordinate per verify forward). This is the physical cross-node cycle
    # count; robust to async CPU-ahead span overlap (span wall-times overlap
    # across types, so span counts are NOT reliable cycle proxies).
    n_ar = sum(1 for e in X if e.get("cat") == "kernel"
               and cclass(e.get("name", "")) == "AllReduce")
    big_fwd = [(t0, t1) for t0, t1 in spans.get("forward", []) if t1 - t0 > 5e3]
    n_cyc = n_ar or len(big_fwd) or n_sample or n_draft or 1
    wall = (max(e["ts"] + e.get("dur", 0) for e in X)
            - min(e["ts"] for e in X)) / 1e3

    print(f"\n## cycles={n_cyc} (coordinate AllReduce kernels)  "
          f"wall={wall:.0f}ms  cycle={wall/n_cyc:.1f}ms")
    print("runner span totals (name count total_ms ms/cyc):")
    span_tot = {k: sum(t1 - t0 for t0, t1 in v) / 1e3 for k, v in spans.items()}
    for k in sorted(span_tot, key=lambda k: -span_tot[k]):
        print(f"  {k:<12} {len(spans[k]):>5} {span_tot[k]:>9.1f} "
              f"{span_tot[k]/n_cyc:>8.2f}")

    print("\n## GPU NCCL collectives by class x span "
          "(correlation-attributed launch nesting)")
    print(f"{'class':<14}{'span':<12}{'n':>6}{'n/cyc':>8}{'ms':>10}{'ms/cyc':>9}")
    draft_ms = draft_n = 0
    verify_ms = verify_n = 0
    for (cl, sp) in sorted(cs, key=lambda k: (k[0], -cs[k][1])):
        n, ms = cs[(cl, sp)]
        print(f"{cl:<14}{sp:<12}{n:>6}{n/n_cyc:>8.2f}{ms:>10.1f}{ms/n_cyc:>9.2f}")
        if sp == "draft":
            draft_ms += ms
            draft_n += n
        elif sp in ("forward", "preprocess", "postprocess", "sample"):
            verify_ms += ms
            verify_n += n
    if unmapped:
        print(f"  (unmapped kernels: {dict(unmapped)})")

    print(f"\n  => DRAFT-path NCCL: {draft_n} kernels "
          f"({draft_n/n_cyc:.2f}/cyc), {draft_ms/n_cyc:.2f} ms/cyc")
    print(f"  => VERIFY-path NCCL: {verify_n} kernels "
          f"({verify_n/n_cyc:.2f}/cyc), {verify_ms/n_cyc:.2f} ms/cyc")

    # CPU/user-annotation coordination (gloo / nccl all_reduce = DP rendezvous)
    print("\n## DP-coordination all-reduce (user_annotation gloo:/nccl:) by span")
    ann = [e for e in X if e.get("cat") in ("user_annotation",
           "gpu_user_annotation", "cpu_op")
           and ("all_reduce" in e.get("name", "").lower()
                or "allreduce" in e.get("name", "").lower())]
    aspan = defaultdict(lambda: [0, 0.0])
    for e in ann:
        aspan[(e["name"][:24], span_of(e["ts"]))][0] += 1
        aspan[(e["name"][:24], span_of(e["ts"]))][1] += e["dur"] / 1e3
    for (nm, sp) in sorted(aspan, key=lambda k: -aspan[k][1]):
        n, ms = aspan[(nm, sp)]
        print(f"  {nm:<26}{sp:<12} n={n:<5} {n/n_cyc:>5.2f}/cyc {ms:>8.1f} ms")

    # Aggregate MoE A2A (all verify-side since draft==0), per cycle.
    ag_ms = sum(cs[k][1] for k in cs if k[0] == "AllGather")
    rs_ms = sum(cs[k][1] for k in cs if k[0] == "ReduceScatter")
    ar_ms = sum(cs[k][1] for k in cs if k[0] == "AllReduce")
    print(f"\n## aggregate cross-node NCCL (draft==0 => all verify-side)")
    print(f"  MoE AllGather   {ag_ms:8.1f} ms  {ag_ms/n_cyc:6.2f} ms/cyc")
    print(f"  MoE ReduceScat  {rs_ms:8.1f} ms  {rs_ms/n_cyc:6.2f} ms/cyc")
    print(f"  coordinate AllR {ar_ms:8.1f} ms  {ar_ms/n_cyc:6.2f} ms/cyc")
    print(f"  TOTAL           {ag_ms+rs_ms+ar_ms:8.1f} ms  "
          f"{(ag_ms+rs_ms+ar_ms)/n_cyc:6.2f} ms/cyc")

    # Draft-span wall vs GPU-busy: is the draft wall inflated by idle waiting
    # (e.g. stalling on the verify forward's GPU drain / target hidden states)?
    draft_win = sorted(spans.get("draft", []))
    draft_win = [(t0, t1) for t0, t1 in draft_win if t1 - t0 > 1e3]
    gpu = sorted([(e["ts"], e["ts"] + e["dur"]) for e in X
                  if e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")],
                 key=lambda x: x[0])
    wall = busy = 0.0
    gi = 0
    for t0, t1 in draft_win:
        wall += t1 - t0
        # union of gpu-kernel coverage within [t0,t1]
        cov_end = t0
        while gi < len(gpu) and gpu[gi][0] < t1:
            s, e = gpu[gi]
            if e <= t0:
                gi += 1
                continue
            s = max(s, cov_end, t0)
            e = min(e, t1)
            if e > s:
                busy += e - s
                cov_end = max(cov_end, e)
            if gpu[gi][1] > t1:
                break
            gi += 1
    if wall > 0:
        print(f"\n## draft span wall vs GPU-busy (n={len(draft_win)} spans)")
        print(f"  wall {wall/1e3:.1f} ms ({wall/1e3/n_cyc:.2f}/cyc)  "
              f"gpu-busy {busy/1e3:.1f} ms ({busy/1e3/n_cyc:.2f}/cyc)  "
              f"idle {(wall-busy)/1e3:.1f} ms ({(wall-busy)/1e3/n_cyc:.2f}/cyc)")

    # AllReduce straggler distribution (the cross-node barrier / desync tell)
    ar = sorted([k for k in nccl if cclass(k["name"]) == "AllReduce"],
                key=lambda e: -e["dur"])
    if ar:
        durs = [e["dur"] / 1e3 for e in ar]
        print(f"\n## verify coordinate AllReduce straggler: n={len(durs)} "
              f"max={durs[0]:.2f} p50={durs[len(durs)//2]:.2f} "
              f"min={durs[-1]:.2f} ms  (sum/cyc={sum(durs)/n_cyc:.2f})")


if __name__ == "__main__":
    main(sys.argv[1])
