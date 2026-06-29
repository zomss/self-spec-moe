# Phase 32: Overlapped (continuous) spec -- does World A compose with conventional spec?

**Source:** Phase 30/31 (self-spec's local-routing draft is dominated by EAGLE in SEQUENTIAL
and CASCADE compositions, because a draft should be cheap and World A's is a full forward).
The user's question: apply the spec module *every step* (continuous / overlapped), so the
comm-free draft overlaps the comm-bound verify. Does that give World A a role?

**Objective:** model the overlapped speedup -- the comm-free draft compute hidden behind the
verify's all-to-all communication -- for World A (full-forward local-routing draft) vs EAGLE
(cheap head), as a function of the comm/compute ratio R (= f/(1-f)).

**Finding (the draft tradeoff FLIPS at R=2 / f~0.67):**
- R<2 (single-node PCIe): the draft COST dominates -> cheap EAGLE wins; World A's full
  draft adds compute that overlap can't hide (not enough comm to hide behind).
- R>2 (inter-node IB): the verify's comm leaves compute idle -> World A's comm-free draft
  HIDES for free -> only beta matters -> World A's full-model draft (higher fidelity than a
  small head) WINS, training-free.

So "spec every step" turns World A's expensive draft from a liability into an asset -- but
only past the crossover, i.e. INTER-NODE (the project's real-win regime).

**Status:** analytic model (`overlap_model.py`). Assumes perfect overlap (optimistic; real
DBO ~70-90%), and the data dependency (draft i+1 needs verify i) needs speculative-ahead
drafting (free when overlapped; wasted on rejection -> fine at high beta). The inter-node
measurement is hardware-gated (needs a multi-node box) -- the key experiment to confirm.

**Next artifact:** `results_overlap.md` -- the model, the crossover, the design implication.
