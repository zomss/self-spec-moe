"""Compare two accept-job JSONs for byte-identical output token ids + accept_len.
Usage: diff_accept.py <base.json> <orch.json>
"""
import json
import sys

b = json.load(open(sys.argv[1]))
o = json.load(open(sys.argv[2]))
tb, to = b.get("token_ids"), o.get("token_ids")
alb = b["run_result"].get("accept_len")
alo = o["run_result"].get("accept_len")
print(f"accept_len  base={alb}  orch={alo}")
if tb is None or to is None:
    print("MISSING token_ids", tb is not None, to is not None)
    sys.exit(1)
same = tb == to
ntb = sum(len(x) for x in tb)
nto = sum(len(x) for x in to)
print(f"nreqs base={len(tb)} orch={len(to)}  total_toks base={ntb} orch={nto}")
print(f"TOKEN IDS IDENTICAL: {same}")
if not same:
    for i, (x, y) in enumerate(zip(tb, to)):
        if x != y:
            # first differing position
            for j in range(min(len(x), len(y))):
                if x[j] != y[j]:
                    print(f"  req {i} pos {j}: base={x[j]} orch={y[j]}  "
                          f"(len base={len(x)} orch={len(y)})")
                    break
            else:
                print(f"  req {i}: len differs base={len(x)} orch={len(y)}")
            break
