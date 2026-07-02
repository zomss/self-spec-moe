"""OV1(b2) DP1 repro analyzer: align run/consume events and determine the
TRUE pipeline depth empirically.

For each consume event (verify verdict: b, rej, committed, plus the drafts
being served next), test which earlier served-set the verdict fits:
depth d means the verdict at consume[i] judged served[i-d]. A verdict "fits"
a served set S if the first (K - rej) drafts of S were accepted -- we can't
see the accepted tokens directly, but committed deltas + rej counts give the
accepted count, and b must differ from S[accepted] on rejection (the
correction replaces the first rejected draft) or be a fresh bonus on rej==0.

Prints the interleaved timeline with positional bookkeeping so off-by-ones
are visible by inspection.
"""

import json
import sys

path = sys.argv[1]
events = [json.loads(x) for x in open(path)]

K = None
consumes = [e for e in events if e["ev"] == "consume"]
runs = [e for e in events if e["ev"] == "run"]
if consumes:
    K = len(consumes[0]["served"])

print(f"events: {len(runs)} runs, {len(consumes)} consumes, K={K}\n")
print("interleaved timeline (row 0):")
prev_committed = None
for e in events[:80]:
    if e["ev"] == "run":
        outs = e["outs"]
        print(
            f"  run {e['n']:>3} [{e['mode']:>6}] anchor={e['anchor']:>6}"
            f"@p1={e['p1']:<4} outs={outs}"
        )
    else:
        delta = (
            e["committed"] - prev_committed
            if prev_committed is not None
            else None
        )
        prev_committed = e["committed"]
        print(
            f"  consume {e['cyc']:>3}: b={e['b']:>6} rej={e['rej']} "
            f"committed={e['committed']:<4} (+{delta}) "
            f"served={e['served']}"
        )

# Depth determination: at consume[i] with rej=r, accepted = K - r drafts of
# the judged set. If the judged set is served[i-d], its first K-r tokens were
# accepted and (if r>0) b replaced served[i-d][K-r]. Score each d by how
# often the rejection boundary is CONSISTENT: on r>0, b != served[i-d][K-r]
# is expected (correction differs from the rejected draft) but the accepted
# prefix must have been plausible; the sharpest signal is r==0 cycles where
# the NEXT committed delta should be K+1, and value-level: for r>0 we check
# b == model-correction which we can't recompute -- so instead use the
# committed-delta consistency: delta_committed at consume[i] == (K - r) + 1.
print("\ncommitted-delta consistency (should be K - rej + 1 every cycle):")
ok = bad = 0
prev = None
for e in consumes:
    if prev is not None:
        delta = e["committed"] - prev["committed"]
        want = K - e["rej"] + 1
        flag = "ok" if delta == want else f"MISMATCH want {want}"
        if delta == want:
            ok += 1
        else:
            bad += 1
        if bad and bad <= 10 and delta != want:
            print(
                f"  cyc {e['cyc']}: delta={delta} rej={e['rej']} -> {flag}"
            )
    prev = e
print(f"  consistent: {ok}, mismatched: {bad}")

# Value-level depth test: on a FULL-ACCEPT verdict (rej==0), the accepted
# drafts equal the judged served-set verbatim; the judged set's tokens then
# appear in the committed stream. We can't see the stream, but we CAN check
# which served-set's first token the previous correction/bonus b' aligns
# with... simpler: for rej==0 at consume[i], test d by checking that
# served[i-d] is CONSISTENT with the previous cycles' bs: b at consume[i]
# is the bonus AFTER the accepted set. If the runs are pure continuations,
# outs chains: served[j] continued by b should equal the chain's outs.
print("\nrej==0 cycles and which served-set looks judged (by b-continuity):")
for i, e in enumerate(consumes[:40]):
    if e["rej"] == 0 and i >= 2:
        print(
            f"  cyc {e['cyc']}: b={e['b']} | served[i-1]={consumes[i-1]['served']}"
            f" served[i-2]={consumes[i-2]['served']}"
        )
