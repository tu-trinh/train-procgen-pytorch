#!/bin/bash

index=5
gpu_device=5
env_name="heist_aisc_many_keys"

seed=8888
risk_values=(55 60 65 70 75 80 85 90 95)

if [ "$env_name" == "maze" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze/model_200015872.pth"
elif [ "$env_name" == "maze_yellowstar_redgem" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_yellowstar_redgem/model_200015872.pth"
elif [ "$env_name" == "heist_aisc_many_keys" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_keys/model_200015872.pth"
elif [ "$env_name" == "coinrun_aisc" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun_aisc/model_200015872.pth"
fi

names=(
    "receive_help_test_max_probs"
    "receive_help_test_sample_probs"
    "receive_help_test_max_logit"
    "receive_help_test_sample_logit"
    "receive_help_test_ent"
    "receive_help_test_random"
)
metrics=(
    "msp"
    "sampled_p"
    "ml"
    "sampled_l"
    "ent"
    "random"
)
for risk in "${risk_values[@]}"; do
    python3 render.py \
        --exp_name "${names[index]}_risk_${risk}" \
        --env_name ${env_name} \
        --distribution_mode hard \
        --param_name hard-plus \
        --model_file ${model_file} \
        --percentile_dir ${percentile_dir} \
        --select_mode sample \
        --ood_metric ${metrics[index]} \
        --risk ${risk} \
        --expert_model_file ${expert_model_file} \
        --expert_cost 2 \
        --switching_cost 2 \
        --quant_eval \
        --seed ${seed} \
	--device gpu \
        --gpu_device ${gpu_device} \
        --save_run
done
