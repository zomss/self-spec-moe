#!/usr/bin/env python3
"""Phase 90 E1b: ACCEPTANCE-ALIGNED proxies for skip importance.

Two candidate screens, computed together per model:
  (A) margin-gradient Taylor: one forward+backward per ref on the
      margin objective m = z_top1 - z_top2. Score(layer i) =
      mean_pos |g_i . delta_i| where delta_i = the layer's residual
      update and g_i = dm/dh at the layer output (downstream healing
      is encoded in the gradient path).
  (B) micro-LOO: ablate one decoder layer at a time (identity
      passthrough), forward tiny refs, damage = 1 - argmax agreement
      with the FULL model. Objective-exact, subsampled.
Also emits per-layer QUANT Taylor scores: |grad_W . deltaW_rtn4|
summed per layer (analytic int4 g128 RTN error, no measurement).

env: MODEL, N_REFS, REF_TOK, DEV, OUT_PREFIX
"""
import glob
import gzip
import json
import os

import torch

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
N_REFS = int(os.environ.get("N_REFS", "6"))
REF_TOK = int(os.environ.get("REF_TOK", "2000"))
DEV = os.environ.get("DEV", "cuda:0")
OUT = os.environ.get("OUT_PREFIX", "q38b")


def refs_c4(tok):
    texts, buf = [], ""
    for path in sorted(glob.glob(
            "/data/smcho/huggingface/hub/datasets--allenai--c4/**/"
            "*.json.gz", recursive=True)):
        with gzip.open(path, "rt") as f:
            for line in f:
                buf += json.loads(line)["text"] + "\n\n"
                ids = tok.encode(buf)
                if len(ids) >= REF_TOK:
                    texts.append(tok.decode(ids[:REF_TOK]))
                    buf = ""
                    if len(texts) >= N_REFS:
                        return texts
    return texts


def rtn4_error(w, group=128):
    """int4 group-RTN quant error, analytic (symmetric absmax/group)."""
    out, inn = w.shape
    ng = inn // group
    wg = w[:, :ng * group].reshape(out, ng, group).float()
    scale = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = (wg / scale).round().clamp(-8, 7)
    return (wg - q * scale).reshape(out, ng * group).to(w.dtype)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = refs_c4(tok)
    dev_map = ("auto" if os.environ.get("TAYLOR_AUTO_MAP", "0") == "1"
               else DEV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=dev_map,
        attn_implementation="sdpa")
    model.eval()
    no_pgrad = os.environ.get("TAYLOR_NO_PARAM_GRAD", "0") == "1"
    if no_pgrad:
        # activation-grad-only backward (params would double memory);
        # quant-Taylor is skipped in this mode.
        for prm in model.parameters():
            prm.requires_grad_(False)
        model.model.embed_tokens.register_forward_hook(
            lambda m, a, o: o.requires_grad_(True))
    if os.environ.get("TAYLOR_CKPT", "0") == "1":
        model.gradient_checkpointing_enable()
    layers = model.model.layers
    L = len(layers)
    print(f"[e1b] {MODEL}: {L} layers, {len(texts)} refs x {REF_TOK} tok",
          flush=True)

    # ---- (A) margin-gradient Taylor ----
    deltas, grads = {}, {}

    def fwd_hook(i):
        def h(mod, args, kwargs, out):
            hin = args[0] if args else kwargs["hidden_states"]
            hout = out[0] if isinstance(out, tuple) else out
            deltas[i] = (hout - hin).detach()
        return h

    def bwd_hook(i):
        def h(mod, gin, gout):
            grads[i] = gout[0].detach()
        return h

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(fwd_hook(i),
                                                 with_kwargs=True))
        hooks.append(layer.register_full_backward_hook(bwd_hook(i)))

    taylor = torch.zeros(L, dtype=torch.float64)
    quant_t = torch.zeros(L, dtype=torch.float64)
    n_pos = 0
    for ti, text in enumerate(texts):
        ids = tok(text, return_tensors="pt").input_ids.to(
                model.get_input_embeddings().weight.device)
        model.zero_grad(set_to_none=True)
        out = model(ids, use_cache=False)
        z = out.logits[0].float()
        top2 = z.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).sum()
        margin.backward()
        for i in range(L):
            g, d = grads[i][0].float(), deltas[i][0].float()
            taylor[i] += float((g * d).sum(-1).abs().sum())
        n_pos += z.shape[0]
        # quant Taylor accumulates via param grads (signed-sum/ref)
        for i, layer in enumerate(layers) if not no_pgrad else []:
            s = 0.0
            for name, p in layer.named_parameters():
                if p.grad is None or p.dim() != 2:
                    continue
                dw = rtn4_error(p.data)
                s += float((p.grad.float()[:, :dw.shape[1]]
                            * dw.float()).sum().abs())
            quant_t[i] += s
        print(f"[e1b] taylor ref {ti+1}/{len(texts)}", flush=True)
    for h in hooks:
        h.remove()
    taylor /= n_pos

    base = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    with open(f"{base}/{OUT}_taylor_partial.csv", "w") as f:
        f.write("model,layer,taylor_margin,quant_taylor\n")
        for i in range(L):
            f.write(f"{MODEL},{i},{taylor[i]:.6e},{quant_t[i]:.6e}\n")
    print(f"[e1b] taylor saved (partial) -> {OUT}_taylor_partial.csv",
          flush=True)

    # ---- (B) micro-LOO ----
    with torch.no_grad():
        base_argmax = []
        for text in texts:
            ids = tok(text, return_tensors="pt").input_ids.to(
                model.get_input_embeddings().weight.device)
            base_argmax.append(model(ids, use_cache=False)
                               .logits[0].argmax(-1))
        damage = torch.zeros(L, dtype=torch.float64)
        import types

        def make_identity(layer):
            def fwd(self, hidden_states, *args, **kwargs):
                return hidden_states
            return types.MethodType(fwd, layer)

        for i, layer in enumerate(layers):
            orig_fwd = layer.forward
            layer.forward = make_identity(layer)
            agree = tot = 0
            for text, base in zip(texts, base_argmax):
                ids = tok(text, return_tensors="pt").input_ids.to(
                model.get_input_embeddings().weight.device)
                am = model(ids, use_cache=False).logits[0].argmax(-1)
                agree += int((am == base).sum())
                tot += am.numel()
            layer.forward = orig_fwd
            damage[i] = 1 - agree / tot
            if (i + 1) % 8 == 0:
                print(f"[e1b] micro-LOO {i+1}/{L}", flush=True)

    base = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    with open(f"{base}/{OUT}_acceptproxy.csv", "w") as f:
        f.write("model,layer,taylor_margin,microloo_damage,quant_taylor\n")
        for i in range(L):
            f.write(f"{MODEL},{i},{taylor[i]:.6e},{damage[i]:.6f},"
                    f"{quant_t[i]:.6e}\n")
    print(f"[e1b] saved -> {base}/{OUT}_acceptproxy.csv", flush=True)


if __name__ == "__main__":
    main()
