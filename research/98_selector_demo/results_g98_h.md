# G98-H: MoE breadth — the path works, the measurement does not

The phase's largest stated limitation is one model, one hardware family. Its
central claim is architectural, so it deserves a sparse target, and this box
has one: **Qwen3-30B-A3B**, 48 layers, 128 experts, sharing Qwen3-8B's
tokenizer (vocab 151936) so the frozen prompt bundle transfers verbatim.

**Outcome: the infrastructure now exists and works; the measurements it
produces on this box do not support any lever conclusion.** Both halves are
reported because the first is reusable and the second is disqualifying.

## 1. Tensor parallelism works in this fork (two config fixes)

The record said "TP>1 fails to start in this fork" (`w98_cotenant_load.py`).
That was never a fork limitation — it was two configuration faults:

1. **`draft_tensor_parallel_size` must equal `tensor_parallel_size`.** Round 1
   pins it to 1 because the dense campaign is single-GPU, so raising TP alone
   fails at worker init with a plain `ValueError`.
2. **`custom_all_reduce` dies with an illegal memory access** on this
   hardware once the model is loaded. `disable_custom_all_reduce=True` falls
   back to NCCL and the boot completes.

With both applied, a 30B MoE target and its 15 GB W4A16 draft boot together
under TP=2 and decode normally — a combination that cannot fit on one 80 GB
device, so **TP is what makes the quant axis reachable on MoE at all**. That
is a reusable capability for any future breadth work.

## 2. The measurement is not trustworthy here

Eight cells, two rounds, same configuration each time:

| cell | R1 round 1 | round 2 | diff | R6 round 1 | round 2 | diff |
| --- | --- | --- | --- | --- | --- | --- |
| `tm/woff/skip0` | 95.8 | 95.7 | **0.1%** | 2351.1 | 2350.1 | **0.0%** |
| `q4/woff/skip0` | 126.4 | 124.0 | **1.8%** | 2820.4 | 2785.9 | **1.2%** |
| `tm/w256/skip0` | 111.6 | 83.2 | 25.5% | 2069.4 | 1989.2 | 3.9% |
| `tm/w1024/skip0` | 129.7 | 91.1 | 29.7% | 2427.1 | 2270.4 | 6.5% |
| `tm/woff/skip4` | 62.4 | 87.9 | 40.8% | 1740.1 | 2356.1 | 35.4% |
| `tm/woff/skip8` | 96.3 | 63.2 | 34.4% | 2250.1 | 1579.0 | 29.8% |
| `tm/w1024/skip8` | 94.5 | 70.1 | 25.9% | 2239.4 | 1736.3 | 22.5% |
| `q4/w256/skip4` | 113.2 | 82.8 | 26.8% | 2625.7 | 2243.7 | 14.6% |

Median round-to-round difference is **26.3% at R1** and **10.5% at R6**,
against **~1%** for armed cells in the dense TP=1 grid. A third measurement
of `q4/woff/skip0` during the smoke test read 67.2 against the 126.4 above —
1.9x apart. No lever ratio can be read off numbers that move like this, so
the quant, window and skip effects measured here are **not reported as
findings**.

## 3. The instability is localised, and that is the lead

The two stable cells are exactly the two **unlevered** ones — no window, no
skip — at 0.1-1.8%. Every cell carrying a window or a skip swings 25-41%.
The target, the draft, the parallelism and the box are identical across both
groups; the only difference is whether the draft chain engages the
window/skip machinery.

That points at an interaction between the scratchpad/gather path and tensor
parallelism, rather than at generic box noise, and it is worth chasing
because it is a property of the engine rather than of this VM. Two readings
are consistent with the data and are not yet separated:

* the levered paths do more host work per step and less GPU work, so they are
  more exposed to the same state changes that make OFF the noisiest cell in
  the dense grid; or
* the levered paths interact badly with TP specifically — plausible because
  the self-spec proposer contains no tensor-parallel logic at all.

The discriminating experiment is cheap and is **not** run here: the same
eight cells at TP=1 with shared weights, which fits on one device. If the
levered cells are stable at TP=1, the interaction is with TP; if they still
swing, it is exposure to the box.

## 4. What this does and does not establish

**Establishes:** TP works in this fork; MoE with a quantized draft boots and
decodes; the full three-lever model is reachable on a sparse architecture
once TP is available; and levered cells are ~25x noisier than unlevered ones
under TP=2 on this box.

**Does not establish:** anything about whether quantization, windowing or
layer skipping pays on MoE; whether the cost model's structure survives on a
sparse architecture; or C1's dense-versus-sparse batch-dependence, which was
the original motivation. All of that needs a measurement regime that has not
been achieved yet.

**Recommended next step:** run the same eight cells at TP=1 (shared weights,
four-parameter reduction) to separate the two readings in section 3. If TP is
implicated, the breadth experiment belongs on h103/h104 where a 30B target
and its draft can be given a quieter machine, or on a smaller MoE that fits
single-GPU with its draft.

## 5. A bug found and fixed here

`_slug` named MoE boots by window and skip only, so the two quant arms of the
same lever point collided on one filename and the quantized cell was skipped
silently. This is the same bug class that cost the dense grid a cell the same
night (`results_g98_g_fullgrid.md`). Fixed by keying the slug on all three
axes; the affected records were renamed by re-reading each file's own
`config` rather than trusting its name.
