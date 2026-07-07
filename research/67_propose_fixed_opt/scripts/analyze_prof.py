"""Phase 67: decompose the propose-fixed block from SelfSpecProfiler dumps.

Reads data/prof_<TAG>/self_spec_profile_*.json (one per rank/PID), picks the
rank with the most `verify` samples (the real model-running rank), and prints
per-region mean_ms grouped into the cycle decomposition:

  verify                     : target forward (K+1 tokens)
  gap (cpu_*)                : rejection sample + parse + bookkeep + prepare +
                               next-input build (CPU glue between verify->propose)
  propose-fixed (step0_*)    : step-0 draft prep + attn-md + DP coord + forward
                               + sample + chain_setup (once per cycle)
  marginal (step_*/draft_fwd): per chain-step cost (scales with K)

Usage: analyze_prof.py <prof_dir> [<prof_dir> ...]
"""

import glob
import json
import os
import sys

# Region -> group. draft_forward_first is the step-0 forward (fixed);
# draft_forward is the per-chain-step forward (marginal).
FIXED_STEP0 = [
    "step0_set_inputs",
    "step0_build_attn_md",
    "step0_determine_batch",
    "draft_forward_first",
    "step0_sample",
    "chain_setup",
]
MARGINAL = [
    "step_pos_slot_update",
    "step_build_attn_md",
    "step_input_buffering",
    "draft_forward",
    "step_sample",
]
GAP_CPU = [
    "cpu_exec_prepare_inputs",
    "cpu_rejection_sample",
    "cpu_reject_parse",
    "cpu_bookkeep_loop",
    "cpu_next_input_build",
]


def pick_rank(prof_dir):
    best, best_n = None, -1
    for path in glob.glob(os.path.join(prof_dir, "self_spec_profile_*.json")):
        with open(path) as f:
            d = json.load(f)
        n = d.get("counts", {}).get("verify", 0)
        if n > best_n:
            best, best_n = d, n
    return best


def m(summ, label):
    r = summ.get(label)
    if not r or r.get("mean_ms") is None:
        return None
    return r["mean_ms"]


def report(prof_dir):
    d = pick_rank(prof_dir)
    if d is None:
        print(f"[{prof_dir}] no profiler dumps")
        return
    summ = d["summary"]
    cnt = d["counts"]
    print(f"\n=== {os.path.basename(prof_dir)} "
          f"(pid rank={d.get('dp_rank')}, verify n={cnt.get('verify')}, "
          f"warmup={d.get('warmup')}) ===")

    def line(label, indent=2):
        v = m(summ, label)
        n = cnt.get(label, 0)
        vs = f"{v:8.3f} ms" if v is not None else "     --  "
        print(f"{' ' * indent}{label:<26} {vs}  (n={n})")

    print("- coarse (always-on, minimal syncs):")
    for lab in ["verify", "draft_chain", "draft_forward_first", "draft_forward"]:
        line(lab)

    def group_sum(labels):
        s = 0.0
        any_ = False
        for lab in labels:
            v = m(summ, lab)
            if v is not None:
                s += v
                any_ = True
        return s if any_ else None

    print("- FIXED step0 sub-regions (fine):")
    for lab in FIXED_STEP0:
        line(lab)
    fs = group_sum(FIXED_STEP0)
    if fs is not None:
        print(f"    {'SUM step0 fixed':<26} {fs:8.3f} ms")

    print("- GAP cpu_* glue (verify->propose, fine):")
    for lab in GAP_CPU:
        line(lab)
    gs = group_sum(GAP_CPU)
    if gs is not None:
        print(f"    {'SUM gap cpu':<26} {gs:8.3f} ms")

    print("- MARGINAL per-step (fine, x(K-1)):")
    for lab in MARGINAL:
        line(lab)
    ms_ = group_sum(MARGINAL)
    if ms_ is not None:
        print(f"    {'SUM one marginal step':<26} {ms_:8.3f} ms")

    v = m(summ, "verify")
    dc = m(summ, "draft_chain")
    if v is not None and dc is not None:
        pf = (gs or 0.0) + (fs or 0.0) if fs is not None else None
        print("- rollup:")
        print(f"    verify                    {v:8.3f} ms")
        print(f"    draft_chain (whole propose) {dc:8.3f} ms")
        if fs is not None and gs is not None:
            print(f"    propose-fixed = step0+gap  {fs + gs:8.3f} ms "
                  f"(step0 {fs:.2f} + gap {gs:.2f})")


def main():
    dirs = sys.argv[1:]
    if not dirs:
        base = os.path.join(os.path.dirname(__file__), os.pardir, "data")
        dirs = sorted(glob.glob(os.path.join(base, "prof_*")))
    for d in dirs:
        report(d)


if __name__ == "__main__":
    main()
