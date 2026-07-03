#!/bin/bash
# Phase 54 pipeline (resume): ladder step3, then the Phase 53 spec rerun.
set -u
L=/h/v-sukmincho/self-spec-moe/research/54_node_local_draft/scripts/run_ladder.sh
D=/h/v-sukmincho/self-spec-moe/research/53_deepseek_scale/scripts/run_2node_dsv2.sh
bash "$L" step3
bash "$D" spec
echo PIPELINE2-DONE
