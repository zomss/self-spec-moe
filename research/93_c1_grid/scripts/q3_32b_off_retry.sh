#!/bin/bash
# fires after final_block: retry any un-marked q3_32b Stage-B arm (the
# AR 'off' baseline failed on a transient co-tenant occupant at boot).
until grep -q LLAMA-REMEASURE-DONE /data/smcho/self-spec-moe/research/93_c1_grid/logs/final_block.log 2>/dev/null; do sleep 120; done
STAGEB_GPU=6,7 bash /data/smcho/self-spec-moe/research/93_c1_grid/scripts/run_stage_b.sh q3_32b
echo Q32B-OFF-RETRY-DONE
