#!/bin/bash

env_name=$1
index=$2
risk_set=$3
gpu_device=$4

seed=8888
if [ "$risk_set" == "A" ]; then
    if [ "$index" == "6" ]; then
	risk_values=(50 60 70 80 90 100)
    else
	risk_values=(10 20 30 40 50 60 70 80 90)
    fi
elif [ "$risk_set" == "B" ]; then
    if [ "$index" == "6" ]; then
	risk_values=(110 120 130 140 150)
    else
	risk_values=(5 15 25 35 45 55 65 75 85 95)
    fi
fi

if [ "$env_name" == "maze" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze/model_200015872.pth"
    detector_percentile_dir=""
    detector_model_file=""
elif [ "$env_name" == "maze_yellowstar_redgem" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_yellowstar_redgem/model_200015872.pth"
    detector_percentile_dir=""
    detector_model_file=""
elif [ "$env_name" == "heist_aisc_many_keys" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_keys/model_200015872.pth"
    detector_percentile_dir=""
    detector_model_file=""
elif [ "$env_name" == "coinrun_aisc" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun_aisc/model_200015872.pth"
    detector_percentile_dir=""
    detector_model_file=""
elif [ "$env_name" == "coinrun" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun_aisc/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun_aisc"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun/model_200015872.pth"
elif [ "$env_name" == "heist_aisc_many_chests" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_keys/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_keys"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests/model_200015872.pth"
elif [ "$env_name" == "maze_redline_yellowgem" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_yellowstar_redgem/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_yellowstar_redgem"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem/model_200015872.pth"
elif [ "$env_name" == "maze_aisc" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc/model_200015872.pth"
fi

names=(
    "receive_help_test_max_probs"
    "receive_help_test_sample_probs"
    "receive_help_test_max_logit"
    "receive_help_test_sample_logit"
    "receive_help_test_ent"
    "receive_help_test_random"
    "recieve_help_test_svdd"
)
metrics=(
    "msp"
    "sampled_p"
    "ml"
    "sampled_l"
    "ent"
    "random"
    "detector"
)
for risk in "${risk_values[@]}"; do
    python3 render.py \
        --exp_name "${names[index]}_risk_${risk}" \
        --env_name ${env_name} \
        --distribution_mode hard \
        --param_name hard-plus \
        --model_file ${model_file} \
	--percentile_dir ${detector_percentile_dir if $index == 6 else percentile_dir} \
	--detector_model_file ${detector_model_file} if $index == 6 else EXCLUDE THIS LINE \
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

