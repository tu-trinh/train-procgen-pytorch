#!/bin/bash

seed=8888
risk_values=(10 20 30 40 50 60 70 80 90)

# SIMPLE TRAIN ENVIRONMENT
# python3 render.py \
#     --exp_name eval_train_og \
#     --env_name coinrun \
#     --distribution_mode hard \
#     --param_name hard-plus \
#     --model_file logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/model_200015872.pth \
#     --select_mode sample \
#     --quant_eval \
#     --seed 8888 \
#     --save_run

# PROBABILIY-BASED METRICS, TRAIN ENVIRONMENT
# names=("receive_help_train_max_probs_og" "receive_help_train_sample_probs_og" "receive_help_train_max_logit_og" "receive_help_train_sample_logit_og" "receive_help_train_ent_og")
# metrics=("msp" "sampled_p" "ml" "sampled_l" "ent")
# for i in "${!names[@]}"; do
#     for risk in "${risk_values[@]}"; do
#         python3 render.py \
#             --exp_name ${names[i]} \
#             --env_name coinrun \
#             --distribution_mode hard \
#             --param_name hard-plus \
#             --model_file logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/model_200015872.pth \
#             --percentile_dir logs/procgen/coinrun/get_percentiles_og/RENDER_seed_8888_06-10-2024_13-45-01 \
#             --select_mode sample \
#             --ood_metric ${metrics[i]} \
#             --risk ${risk} \
#             --quant_eval \
#             --seed ${seed} \
#             --save_run
#     done
# done

# PER-ACTION METRICS, TRAIN ENVIRONMENT
# names=("help_train_sample_probs_by_action_og" "help_train_sample_logit_by_action_og" "help_train_ent_by_action_og")
# metrics=("sampled_p" "sampled_l" "ent")
# for i in "${!names[@]}"; do
#     python3 render.py --exp_name ${names[i]} --env_name coinrun --distribution_mode hard --param_name hard-plus --model_file logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/model_200015872.pth --percentile_dir logs/procgen/coinrun/get_percentiles_og/RENDER_seed_8888_06-10-2024_13-45-01 --select_mode sample --ood_metric ${metrics[i]} --risk 5 --by_action --quant_eval --seed ${seed}
# done

# PROBABILIY-BASED METRICS, TEST ENVIRONMENT
names=(
    "receive_help_test_max_probs_og"
    "receive_help_test_sample_probs_og"
    "receive_help_test_max_logit_og"
    "receive_help_test_sample_logit_og"
    "receive_help_test_ent_og"
)
metrics=(
    "msp"
    "sampled_p"
    "ml"
    "sampled_l"
    "ent"
)
for i in "${!names[@]}"; do
    for risk in "${risk_values[@]}"; do
        python3 render.py \
            --exp_name "${names[i]}_risk_${risk}" \
            --env_name coinrun_aisc \
            --distribution_mode hard \
            --param_name hard-plus \
            --model_file logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/model_200015872.pth \
            --percentile_dir logs/procgen/coinrun/get_percentiles_og/RENDER_seed_8888_06-10-2024_13-45-01 \
            --select_mode sample \
            --ood_metric ${metrics[i]} \
            --risk ${risk} \
            --expert_model_file logs/train/coinrun_aisc/expert/2024-06-10__08-20-01__seed_8888/model_200015872.pth \
            --expert_cost 5 \
            --switching_cost 5 \
            --quant_eval \
            --seed ${seed} \
            --save_run
    done
done

# PER-ACTION METRICS, TEST ENVIRONMENT
# names=("help_test_sample_probs_by_action_og" "help_test_sample_logit_by_action_og" "help_test_ent_by_action_og")
# metrics=("sampled_p" "sampled_l" "ent")
# for i in "${!names[@]}"; do
#     python3 render.py --exp_name ${names[i]} --env_name coinrun_aisc --distribution_mode hard --param_name hard-plus --model_file logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/model_200015872.pth --percentile_dir logs/procgen/coinrun/get_percentiles_og/RENDER_seed_8888_06-10-2024_13-45-01 --select_mode sample --ood_metric ${metrics[i]} --risk 5 --by_action --quant_eval --seed ${seed} --save_run
# done
