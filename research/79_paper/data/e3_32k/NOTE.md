# 32k cells (2026-07-14, fixed chain)

- b16/32k CLEAN: nospec 1079.6 +-22; spec K4 2994 +-517 (accept 4.331)
  = 2.77x; spec K6 2517 +-340 (accept 5.615) = 2.33x. Spec-arm variance
  is high (17%) -- iters=4; lower bound still 2.3x.
- b32/32k OVER-CAPACITY (both arms): KV demand ~58.7GB > pool 53.9GB
  (nospec) / ~49GB (spec, +W4 draft weights). Running counts oscillate
  2-31; the recorded tok/s (2236/2261/1910) are THRASH AGGREGATES, not
  decode numbers -- do not quote. Scoping: shared-KV self-spec cannot
  extend the residency envelope (window cuts reads, not pages); see
  sec4-F3.
- Tag note: first (timeout) attempt used max_model_len 36864 >
  max_position_embeddings and died at init; c16k-tagged JSONs from that
  attempt do not exist. ctx for c32k tags = 32000 (prompt ~31,870).
