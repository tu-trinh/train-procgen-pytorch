#!/bin/bash

seed=8888
# names=("help_test_max_probs_og" "help_test_sample_probs_og" "help_test_max_logit_og" "help_test_sample_logit_og" "help_test_ent_og")
names=("help_test_sample_probs_by_action_og" "help_test_sample_logit_by_action_og" "help_test_ent_by_action_og")
metrics=("sampled_p" "sampled_l" "ent")

for i in "${!names[@]}"; do
    python3 render.py --exp_name ${names[i]} --env_name coinrun_aisc --distribution_mode hard --param_name hard-plus --model_file logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/model_200015872.pth --select_mode sample --ood_metric ${metrics[i]} --risk 5 --quant_eval --seed ${seed} --save_run
done
