"""Compare two topc job JSONs for byte-identical output token ids.
Usage: diff_tokens.py <nospec.json> <spec.json>
"""
import json
import sys

b = json.load(open(sys.argv[1]))
o = json.load(open(sys.argv[2]))
tb, to = b.get("token_ids"), o.get("token_ids")
if tb is None or to is None:
    print("MISSING token_ids", tb is not None, to is not None)
    sys.exit(1)
same = tb == to
ntb = sum(len(x) for x in tb)
nto = sum(len(x) for x in to)
print(f"nreqs nospec={len(tb)} spec={len(to)}  total_toks nospec={ntb} spec={nto}")
print(f"TOKEN IDS BYTE-IDENTICAL: {same}")
if not same:
    for i, (x, y) in enumerate(zip(tb, to)):
        if x != y:
            for j in range(min(len(x), len(y))):
                if x[j] != y[j]:
                    print(f"  req {i} pos {j}: nospec={x[j]} spec={y[j]}  "
                          f"(len nospec={len(x)} spec={len(y)})")
                    break
            else:
                print(f"  req {i}: len differs nospec={len(x)} spec={len(y)}")
            break
