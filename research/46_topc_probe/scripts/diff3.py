"""3-way losslessness diff: nospec vs spec-C8 vs spec-C4.

Separates the pre-existing spec-vs-nospec numerical divergence (batched K+1
verify forward vs single-token no-spec forward) from the top-C prune effect.
The prune is lossless iff spec-C4 == spec-C8 (both driven by the SAME full-top_k
verify; the draft only proposes candidates).

Usage: diff3.py <nospec.json> <specC8.json> <specC4.json>
"""
import json
import sys


def toks(path):
    return json.load(open(path)).get("token_ids")


def compare(a, b, na, nb):
    if a is None or b is None:
        return f"{na} vs {nb}: MISSING ({a is not None},{b is not None})"
    same = a == b
    n = sum(len(x) for x in a)
    ndiff_req = sum(1 for x, y in zip(a, b) if x != y)
    # first diff position
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            for j in range(min(len(x), len(y))):
                if x[j] != y[j]:
                    first = (i, j, x[j], y[j])
                    break
            if first:
                break
    msg = f"{na} vs {nb}: IDENTICAL={same}  toks={n}  reqs_differing={ndiff_req}/{len(a)}"
    if first:
        msg += f"  first: req {first[0]} pos {first[1]} ({na}={first[2]} {nb}={first[3]})"
    return msg


nospec, c8, c4 = toks(sys.argv[1]), toks(sys.argv[2]), toks(sys.argv[3])
print(compare(nospec, c8, "nospec", "specC8"))
print(compare(c8, c4, "specC8", "specC4"))
print(compare(nospec, c4, "nospec", "specC4"))
