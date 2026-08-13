# Why the model under-predicts quantized composed cells

Date: 2026-08-13
Status: root-caused. **The fit set cannot identify how quantization composes.**
This is a lattice design flaw, not a model failure, and it bounds what D1 can
establish for quantized configurations.

## The observation

G98-B v6, resolvable regimes, held-out residual `measured / predicted`:

| cell | R1 | R4 | R6 | R8 |
| --- | --- | --- | --- | --- |
| target-matching/w1024/skip4 | 1.021 | 1.013 | 1.020 | 1.021 |
| target-matching/w128/skip4 | 1.016 | 1.012 | 1.004 | 1.012 |
| target-matching/w256/skip8 | 1.041 | 1.035 | 1.029 | 1.034 |
| w4a16/woff/skip4 | 1.005 | 1.021 | 1.018 | 1.018 |
| **w4a16/w1024/skip8** | **1.112** | **1.139** | **1.087** | **1.121** |
| **w4a16/w128/skip8** | **1.114** | **1.220** | **1.088** | **1.157** |
| **w4a16/w256/skip4** | **1.040** | **1.130** | **1.047** | **1.061** |

bf16 composes to within 4%. Quantized `woff` composes to within 2%. Quantized
**windowed** misses by 4–23%, always in the same direction.

## What it is not

**Not a window fixed cost.** Adding a `f_win * [window on]` term to the model
moves the worst held-out residual from 0.1077 to 0.0977 at R1 and from 0.0848 to
0.0754 at R6, and makes it *worse* at R4 (0.1991 → 0.2037). The quant-windowed
ratios stay at 1.02–1.23. A window overhead exists — at 431 tokens the bf16
singles measure `woff` 30.36 ms against 31.25–31.63 ms windowed, so windowing
costs ~1 ms even while reading fewer KV bytes, which is why `kappa_kv` fits
**negative** at R1 — but it is not the explanation.

## What it is: the quant axis is not identified

The fit set has **8 single-lever profiles, of which exactly one is quantized**,
and `W_bytes` takes exactly **two** distinct values:

```text
W_bytes = 16.4e9 - 10.3e9 * q        (q = quantized indicator)
```

`W_bytes` is an affine function of the quant indicator, so it is perfectly
collinear with `[1, q]`. Two models that attribute the quant effect completely
differently are the *same model reparameterised*:

* **A** (registered): `keep * (W*kappa_w + KV*kappa_kv + c0)` — quant reduces
  weight bytes
* **D**: `keep * (KV*kappa_kv + c0 + dq*q)` — quant is a fixed offset

Fitted on the singles and used to predict the held-out cells, they agree to
every digit reported:

| regime | fit residual A / D | held-out worst A / D | quant-windowed A / D |
| --- | --- | --- | --- |
| R1 | 0.0254 / 0.0254 | 0.1077 / 0.1077 | 1.112,1.114,1.040 / identical |
| R4 | 0.0093 / 0.0093 | 0.1991 / 0.1991 | 1.139,1.220,1.130 / identical |
| R6 | 0.0194 / 0.0194 | 0.0848 / 0.0848 | 1.087,1.088,1.047 / identical |
| R8 | 0.0236 / 0.0236 | 0.1456 / 0.1456 | 1.121,1.157,1.061 / identical |

**`kappa_w` is not a bytes-per-second coefficient.** It is the measured quant
effect divided by 10.3e9. The fit contains no evidence that the effect scales
with weight bytes at all.

## Why that produces exactly this residual

Because `W` sits inside the `keep_frac` product, the registered model *assumes*
the quant benefit scales with the fraction of layers kept, and shrinks as KV
grows relative to weights. Neither assumption is tested by the fit set — quant
is measured once, at `woff / skip0`, the corner with the **largest** KV and
**no** skipping.

Every quantized held-out cell is an extrapolation of an untested assumption
along two axes at once, and the error grows with the distance:

| cell | distance from the quant single | residual |
| --- | --- | --- |
| w4a16/woff/skip4 | skip only | 1.005–1.021 |
| w4a16/w256/skip4 | window + skip4 | 1.040–1.130 |
| w4a16/w128/skip8 | window + skip8 | 1.088–1.220 |

The one quantized cell that shares the quant single's window setting is the one
that composes correctly.

## Contrast with the other two axes

| axis | levels in the fit set | identified? |
| --- | --- | --- |
| window | 5 (off, 128, 256, 512, 1024) | yes |
| skip | 3 (0, 4, 8) | yes |
| **quant** | **2, from a single point** | **no** |

Window and skip compose to within 4% precisely because they are sampled at
enough levels to constrain their functional form. Quant is not.

## What this bounds

**D1 cannot succeed for quantized composed cells on this lattice**, regardless
of the envelope, the runtime, or the kernel. The fit set carries no information
about how the quant effect composes, so the model's extrapolation is an
assumption rather than a prediction, and the 4–23% residual measures the error
in that assumption rather than the model's predictive quality.

This also reframes the whole quant×skip8 investigation. Of the original
1.19–1.52x anomaly: roughly two thirds was the host-bound runtime (removed by
Marlin, X14/v6), and the remainder is **not a physical composition effect at
all** — it is an unidentified coefficient extrapolated outside its support.

## What would fix it

Sample the quant axis at more than one point in the **fit set**, crossed with at
least one other axis — for example `w4a16/w256/skip0` (quant × window) and
`w4a16/woff/skip4` (quant × skip, currently held out). Two additional singles
would make `kappa_w` separable from a quant offset and would test the keep_frac
scaling directly.

That changes the frozen fit set and is therefore a **new preregistration**, not
an amendment.
