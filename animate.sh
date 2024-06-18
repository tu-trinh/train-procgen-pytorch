#!/bin/bash

directories=(
    "logs/procgen/coinrun_aisc/help_test_max_probs_og/RENDER_seed_8888_06-11-2024_14-06-20"
    "logs/procgen/coinrun_aisc/help_test_sample_probs_og/RENDER_seed_8888_06-11-2024_14-06-49"
    "logs/procgen/coinrun_aisc/help_test_max_logit_og/RENDER_seed_8888_06-11-2024_14-07-17"
    "logs/procgen/coinrun_aisc/help_test_sample_logit_og/RENDER_seed_8888_06-11-2024_14-07-45"
    "logs/procgen/coinrun_aisc/help_test_ent_og/RENDER_seed_8888_06-11-2024_14-08-18"
)

for i in "${!directories[@]}"; do
    python3 animate.py -d ${directories[i]} -e 0 100 200 300 400 500 600 700 800 900 -s 8888 8988 9088 9188 9288 9388 9488 9588 9688 9788
done
