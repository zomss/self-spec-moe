#!/bin/bash
# Phase 54 pipeline: ladder steps 1-3, then the Phase 53 DeepSeek spec rerun.
set -u
L=/h/v-sukmincho/self-spec-moe/research/54_node_local_draft/scripts/run_ladder.sh
D=/h/v-sukmincho/self-spec-moe/research/53_deepseek_scale/scripts/run_2node_dsv2.sh
bash "$L" step1 && bash "$L" step2 && bash "$L" step3
bash "$D" spec
echo PIPELINE-DONE
