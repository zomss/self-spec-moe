"""Compare W7-FP4 losslessness JSONs: exact-seq count + token agreement.

Reads w7fp4_lossless_{nospec,bf16,nvfp4}.json and reports, vs the nospec greedy
reference: exact-sequence count (of 16) and per-token agreement for bf16 and
nvfp4, plus the nvfp4-vs-bf16 precision delta (should be 16/16 since rejection
sampling makes the bf16 verify the sole source of truth). Prints acceptance.
"""
import json
import os

DATA = "/data/smcho/ssm-w7fp4/research/34_worldA_system/data"


def load(mode):
    p = os.path.join(DATA, f"w7fp4_lossless_{mode}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def compare(ref, other):
    exact = 0
    tot = match = 0
    for a, b in zip(ref["results"], other["results"]):
        ta, tb = a["token_ids"], b["token_ids"]
        if ta == tb:
            exact += 1
        n = min(len(ta), len(tb))
        tot += max(len(ta), len(tb))
        match += sum(1 for i in range(n) if ta[i] == tb[i])
    return exact, len(ref["results"]), (100.0 * match / tot if tot else 0.0)


def main():
    nospec = load("nospec")
    bf16 = load("bf16")
    nvfp4 = load("nvfp4")
    if not (nospec and bf16 and nvfp4):
        print("missing one of nospec/bf16/nvfp4 lossless JSONs")
        return
    print("vs nospec greedy reference (16 prompts):")
    for name, d in [("bf16 full replica", bf16), ("nvfp4 full replica", nvfp4)]:
        e, n, agr = compare(nospec, d)
        s = d.get("summary", {})
        print(f"  {name}: {e}/{n} exact, {agr:.1f}% token agreement, "
              f"accept_len={s.get('mean_accept_length')}, "
              f"accept_rate={s.get('acceptance_rate')}")
    e, n, agr = compare(bf16, nvfp4)
    print(f"  nvfp4 vs bf16 (precision delta): {e}/{n} exact, "
          f"{agr:.1f}% token agreement")


if __name__ == "__main__":
    main()
