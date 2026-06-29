# Phase 30: Single-server-PCIe write-up (synthesis of Phases 24-31)

**Source:** Phases 24 (comm-bound speedup), 25 (acceptance levers), 26 (accept-length-per-GB
frontier), 27 (tree drafting), 28 (dynamic cache / batched memory), 29 (cross-model), 31
(EAGLE/cascade survival test).

**Objective:** a single coherent results document for the single-server-PCIe self-speculative
MoE decoding mechanism -- the measured speedup, the accept-length and memory mechanisms,
the cross-model validation, the EAGLE/cascade stress-test (Phase 31), and an honest
comparison to conventional speculative decoding.

**Artifact:** `single_node_pcie_writeup.md`. No new experiments; all numbers are cited from
the source phases' results docs and data/. (Folded in Phase 31 + the conventional-spec
comparison after they were authored.)
