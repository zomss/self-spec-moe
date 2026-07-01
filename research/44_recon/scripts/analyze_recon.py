"""Phase 44: reconcile the measured comm-free self-spec cycle against the C1
cost model. Reads the three rc_*.json profiles (spec / nospec / skiptile) and
prints:

  1. Cycle decomposition (ms): draft x K, verify, sampling/rejection residual,
     CPU orchestration residual, total.
  2. Draft-cost verdict: full-replica draft forward vs T_compute(64) vs skip-A2A
     tile vs no-spec/verify forward.
  3. Term-by-term reconciliation (cost model 64.7ms -> measured cycle) with the
     dominant term.
  4. Speedup isolation: what the speedup would be if only the draft compute were
     as cheap as T_compute (and other single-error isolations).

All numbers measured; the cost-model coefficients are from
research/paper_section_C1_analysis.md.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# --- C1 cost-model coefficients (paper_section_C1_analysis.md) ---
def T_compute(T):   # comm-free draft (skip-A2A tile fit)
    return 15.3 + 0.0152 * T

def T_verify(T):    # full-EP verify fit
    return 22.7 + 0.0495 * T

BATCH = int(os.environ.get("RC_BATCH", "64"))
K = int(os.environ.get("RC_K", "2"))


def load(mode):
    p = os.path.join(DATA, f"rc_qwen30b_b{BATCH}_K{K}_{mode}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def region(prof, lbl):
    if not prof:
        return None
    s = (prof.get("rank0_profile") or {}).get("summary", {})
    d = s.get(lbl)
    if d and d.get("mean_ms") is not None:
        return d
    return None


def rmean(prof, lbl):
    d = region(prof, lbl)
    return d["mean_ms"] if d else None


def main():
    spec = load("spec")
    nospec = load("nospec")
    tile = load("skiptile")

    if spec is None:
        print("[analyze] spec profile missing; run driver first.")
        return

    rr = spec["run_result"]
    accept_len = rr.get("accept_len")
    sys_tok_s = rr.get("sys_tok_s")

    # --- Region means (rank-0, post-warmup steady) ---
    df_first = rmean(spec, "draft_forward_first")
    df_step = rmean(spec, "draft_forward")
    dchain = rmean(spec, "draft_chain")
    verify = rmean(spec, "verify")

    print("=" * 74)
    print(f"PHASE 44 RECON  Qwen3-30B-A3B DP=8/EP=8 forced-PCIe a2a=0 "
          f"K={K} batch={BATCH}")
    print("=" * 74)
    print(f"\naccept_len = {accept_len}   long-run sys tok/s = {sys_tok_s}")

    # cycle time from tok/s: the whole batch produces accept_len tokens/seq per
    # cycle. cycle_ms = batch*accept_len / sys_tok_s * 1000. Cross-check with the
    # K-retune 1021 tok/s -> ~181ms cited in the brief.
    cycle_from_tps = None
    if sys_tok_s and accept_len:
        cycle_from_tps = (BATCH * accept_len) / sys_tok_s * 1e3
    # Also compute from the target 1021 tok/s (K-retune measured operating point).
    cycle_1021 = (BATCH * (accept_len or 2.891)) / 1021.0 * 1e3

    print("\n--- 1. CYCLE DECOMPOSITION (ms, rank-0 steady means) ---")
    for lbl, v in (("draft_forward_first (step0)", df_first),
                   ("draft_forward (per K-step)", df_step),
                   ("draft_chain (whole propose)", dchain),
                   ("verify (target forward)", verify)):
        print(f"  {lbl:32s} {('%.3f'%v) if v is not None else 'NA':>10}")

    # Sum of GPU forwards inside the cycle: draft_chain already includes ALL K
    # draft forwards + draft-side CPU orchestration; verify is the target forward.
    # The full decode-step cycle also has sampling/rejection + outer CPU glue that
    # sits OUTSIDE both regions.
    gpu_plus_draftcpu = None
    if dchain is not None and verify is not None:
        gpu_plus_draftcpu = dchain + verify

    print(f"\n  draft_chain + verify (measured regions) = "
          f"{gpu_plus_draftcpu:.3f} ms" if gpu_plus_draftcpu else "")
    if cycle_from_tps:
        print(f"  cycle from THIS run's tok/s ({sys_tok_s:.1f}) = "
              f"{cycle_from_tps:.1f} ms")
    print(f"  cycle from K-retune 1021 tok/s (brief target) = "
          f"{cycle_1021:.1f} ms")

    # Residual = cycle - (draft_chain + verify) = sampling/rejection + outer CPU.
    # Authoritative cycle: the K-retune prefill-cancelled decode point (1021
    # tok/s) that defines the 0.55x result. THIS run's sys_tok_s includes prefill
    # so it is a lower bound (larger cycle); reported for cross-check only.
    cycle = cycle_1021
    residual = None
    if cycle and gpu_plus_draftcpu is not None:
        residual = cycle - gpu_plus_draftcpu
        print(f"\n  RESIDUAL (cycle - draft_chain - verify)"
              f" = {residual:.1f} ms  [sampling/rejection + outer CPU glue]")

    # Draft-chain internal split: K draft forwards vs draft-side CPU orchestration.
    if dchain is not None and df_step is not None and df_first is not None:
        # step0 uses draft_forward_first; the remaining K-1 use draft_forward.
        gpu_in_chain = df_first + (K - 1) * df_step
        draft_cpu = dchain - gpu_in_chain
        print(f"\n  Inside draft_chain: {K} draft GPU forwards "
              f"(1x{df_first:.2f} + {K-1}x{df_step:.2f}) = {gpu_in_chain:.2f} ms")
        print(f"                      draft-side CPU orchestration "
              f"= {draft_cpu:.2f} ms")

    # --- 2. DRAFT COST VERDICT ---
    print("\n--- 2. DRAFT COST VERDICT (batch %d, 1 tok/seq decode) ---" % BATCH)
    tile_df = rmean(tile, "draft_forward") if tile else None
    tile_df_first = rmean(tile, "draft_forward_first") if tile else None
    nospec_verify = rmean(nospec, "verify") if nospec else None
    Tc = T_compute(BATCH)
    print(f"  T_compute({BATCH}) cost-model value         = {Tc:.2f} ms")
    print(f"  full-replica draft forward (measured)      = "
          f"{('%.2f'%df_step) if df_step else 'NA'} ms")
    print(f"  skip-A2A tile draft forward (measured)     = "
          f"{('%.2f'%tile_df) if tile_df else 'NA'} ms")
    print(f"  no-spec / verify-compute forward (measured)= "
          f"{('%.2f'%nospec_verify) if nospec_verify else 'NA'} ms")
    if df_step and tile_df:
        print(f"\n  full-replica / skip-A2A tile  = {df_step/tile_df:.2f}x")
    if df_step:
        print(f"  full-replica / T_compute({BATCH})   = {df_step/Tc:.2f}x")
    if df_step and nospec_verify:
        print(f"  full-replica / no-spec verify = {df_step/nospec_verify:.2f}x")
    if tile_df:
        print(f"  skip-A2A tile / T_compute({BATCH})  = {tile_df/Tc:.2f}x")

    # --- 3. TERM-BY-TERM RECONCILIATION ---
    print("\n--- 3. TERM-BY-TERM RECONCILIATION (cost model -> measured cycle) ---")
    Tv = T_verify(BATCH * (K + 1))  # verify over B*(K+1) tokens
    cm_cycle = K * Tc + Tv
    print(f"  cost model: K*T_compute({BATCH}) + T_verify({BATCH*(K+1)})")
    print(f"            = {K}*{Tc:.2f} + {Tv:.2f} = {cm_cycle:.2f} ms")
    print(f"  measured cycle = {cycle:.1f} ms   (gap = {cycle-cm_cycle:.1f} ms)")

    if (df_step is not None and verify is not None and residual is not None
            and df_first is not None and dchain is not None):
        # Self-consistent decomposition against the AUTHORITATIVE cycle:
        #   cycle = draft_chain + verify + residual   (identity, by construction)
        # Cost model = K*T_compute (draft) + T_verify (verify). Attribute the gap
        # term-by-term so the parts sum EXACTLY to (cycle - cost_model):
        draft_gpu = df_first + (K - 1) * df_step          # both draft forwards
        draft_cpu = dchain - draft_gpu                    # draft-side CPU glue
        cm_draft = K * Tc                                 # cost-model draft term
        cm_verify = Tv                                    # cost-model verify term
        #  (a) draft GPU underestimate (real K forwards - K*T_compute)
        a = draft_gpu - cm_draft
        #  (b) verify underestimate (real verify region - T_verify)
        b = verify - cm_verify
        #  (c) draft-side CPU orchestration (unmodeled, inside draft_chain)
        c_draftcpu = draft_cpu
        #  (d) outer residual (sampling/rejection + engine/DP glue, unmodeled)
        d_outer = residual
        gap = cycle - cm_cycle
        print("\n  Gap attribution (self-consistent; sums to the gap):")
        print(f"   (a) draft-GPU underestimate  (df0 {df_first:.1f} + "
              f"{K-1}x df {df_step:.1f} = {draft_gpu:.1f}) - K*T_compute "
              f"{cm_draft:.1f} = {a:+.1f} ms")
        print(f"   (b) verify underestimate     (meas verify {verify:.1f}) - "
              f"T_verify {cm_verify:.1f} = {b:+.1f} ms")
        print(f"   (c) draft-side CPU (in draft_chain, UNMODELED) = "
              f"{c_draftcpu:+.1f} ms")
        print(f"   (d) outer CPU / sample-reject residual (UNMODELED) = "
              f"{d_outer:+.1f} ms")
        terms = [("a draft-GPU", a), ("b verify", b),
                 ("c draft-CPU", c_draftcpu), ("d outer-CPU", d_outer)]
        tot = sum(t[1] for t in terms)
        print(f"   sum(a..d) = {tot:.1f} ms  vs measured gap {gap:.1f} ms")
        dom = max(terms, key=lambda t: t[1])
        print(f"   DOMINANT TERM: {dom[0]} (+{dom[1]:.1f} ms, "
              f"{100*dom[1]/max(tot,1e-9):.0f}% of gap)")
        # Unmodeled CPU overhead (c+d) as a share:
        cpu_unmodeled = c_draftcpu + d_outer
        print(f"   Unmodeled CPU orchestration (c+d) = {cpu_unmodeled:.1f} ms "
              f"({100*cpu_unmodeled/max(tot,1e-9):.0f}% of gap)")

    # --- 4. SPEEDUP ISOLATION ---
    print("\n--- 4. SPEEDUP ISOLATION ---")
    # Baseline no-spec step. The K-retune 0.55x point used no-spec 1857.6 tok/s
    # (batch 64) -> 34.45 ms/token-step; use that so the reproduced speedup
    # matches the reported 0.55x. My fresh nospec run is reported for context.
    ns_step_kretune = BATCH / 1857.6 * 1e3
    ns_step_meas = (nospec["run_result"].get("nospec_step_ms")
                    if nospec else None)
    ns_step = ns_step_kretune
    print(f"  no-spec step (K-retune 1857.6 tok/s) = {ns_step_kretune:.2f} ms "
          f"[baseline for speedup]")
    print(f"  no-spec step (this run, measured)    = "
          f"{('%.2f'%ns_step_meas) if ns_step_meas else 'NA'} ms")

    def speedup(cycle_ms, al, ns):
        # tokens per cycle = al (accepted+1 verify). throughput = al/cycle.
        # baseline = 1/ns. speedup = (al/cycle)/(1/ns) = al*ns/cycle.
        return al * ns / cycle_ms

    if accept_len and ns_step and cycle:
        su_meas = speedup(cycle, accept_len, ns_step)
        print(f"\n  measured cycle={cycle:.1f}ms -> speedup {su_meas:.3f}x "
              f"(vs K-retune reported 0.55x)")
        # Isolate error (a): make ONLY the K draft forwards as cheap as
        # T_compute (shave real draft-GPU down to K*T_compute), keep verify +
        # all CPU as measured.
        if df_step is not None and df_first is not None:
            draft_gpu = df_first + (K - 1) * df_step
            shave_a = draft_gpu - K * Tc
            cyc_a = cycle - shave_a
            su_a = speedup(cyc_a, accept_len, ns_step)
            print(f"  (a) if draft-GPU == K*T_compute (shave {shave_a:.1f}ms): "
                  f"cycle {cyc_a:.1f}ms -> speedup {su_a:.3f}x")
        # Isolate error (b): make verify == T_verify.
        if verify is not None:
            shave_b = verify - Tv
            cyc_b = cycle - shave_b
            su_b = speedup(cyc_b, accept_len, ns_step)
            print(f"  (b) if verify == T_verify (shave {shave_b:.1f}ms): "
                  f"cycle {cyc_b:.1f}ms -> speedup {su_b:.3f}x")
        # Isolate error (c+d): remove all unmodeled CPU/residual overhead.
        if residual is not None and dchain is not None and df_first is not None:
            draft_cpu = dchain - (df_first + (K - 1) * df_step)
            shave_cd = residual + draft_cpu
            cyc_cd = cycle - shave_cd
            su_cd = speedup(cyc_cd, accept_len, ns_step)
            print(f"  (c+d) if CPU overhead == 0 (shave {shave_cd:.1f}ms): "
                  f"cycle {cyc_cd:.1f}ms -> speedup {su_cd:.3f}x")
        # Full cost-model cycle (all three errors removed), against the SAME
        # K-retune baseline -> apples-to-apples with the measured 0.55x.
        su_cm = speedup(cm_cycle, accept_len, ns_step)
        print(f"  (all) cost-model cycle {cm_cycle:.1f}ms -> speedup {su_cm:.3f}x "
              f"(all 3 errors removed, K-retune baseline)")
        # The paper's OWN ~1.1x: baseline = single-token verify T_verify(B), i.e.
        # E_tokens*T_verify(B) / (K*T_compute(B) + T_verify(B*(K+1))).
        Tv_base = T_verify(BATCH)
        su_paper = accept_len * Tv_base / cm_cycle
        print(f"  (paper) E_tok*T_verify({BATCH})/[K*T_compute+T_verify({BATCH*(K+1)})]"
              f" = {accept_len:.2f}*{Tv_base:.1f}/{cm_cycle:.1f} = {su_paper:.3f}x "
              f"(the model's ~1.1x)")


if __name__ == "__main__":
    sys.exit(main())
