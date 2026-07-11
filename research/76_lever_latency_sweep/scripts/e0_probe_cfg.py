#!/usr/bin/env python3
"""E0 config-plumbing probe: does an hf-override actually reach the engine config?

Two override arms cannot be asserted from engine logs:
  - sliding_window: the per-layer `Attention` debug line only fires when
    kv_cache_dtype_skip_layers is set (attention.py:271-293), so a log-grep is
    vacuous. But the value every Attention layer falls back to IS
    cache_config.sliding_window (attention.py:233-238), which this probe reads.
  - num_hidden_layers: the config dump does not include it.

Builds the engine config (no GPU workers spawned) and prints one greppable line:
  E0PROBE sliding_window=... num_hidden_layers=... quantization=... max_model_len=...

max_model_len is printed because vLLM historically CAPS max_model_len to the
sliding window for models it thinks can't mix SWA + long context -- if that
fires, every long-ctx window cell of the sweep is silently broken.
"""

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-overrides", default=None, help="JSON dict")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--kv-cache-dtype", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--enable-expert-parallel", action="store_true")
    a = ap.parse_args()

    from vllm.engine.arg_utils import EngineArgs

    kwargs = dict(
        model=a.model,
        trust_remote_code=a.trust_remote_code,
        load_format="dummy",  # config-only probe; never touch weights
        tensor_parallel_size=a.tensor_parallel_size,
        enable_expert_parallel=a.enable_expert_parallel,
    )
    if a.hf_overrides:
        kwargs["hf_overrides"] = json.loads(a.hf_overrides)
    if a.quantization:
        kwargs["quantization"] = a.quantization
    if a.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = a.kv_cache_dtype
    if a.max_model_len:
        kwargs["max_model_len"] = a.max_model_len

    cfg = EngineArgs(**kwargs).create_engine_config()
    mc, cc = cfg.model_config, cfg.cache_config
    print(
        "E0PROBE"
        f" sliding_window={cc.sliding_window}"
        f" num_hidden_layers={getattr(mc.hf_text_config, 'num_hidden_layers', None)}"
        f" quantization={mc.quantization}"
        f" kv_cache_dtype={cc.cache_dtype}"
        f" max_model_len={mc.max_model_len}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
