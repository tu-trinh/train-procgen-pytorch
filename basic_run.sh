#!/bin/bash

# Used for running on training environment, collecting values for percentile-making, and collecting (latent) observations

env_name=$1
gpu_device=$2
save_style=$3  # 0 for raw, 1 for latent

if [ "$env_name" == "coinrun" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun/model_200015872.pth"
elif [ "$env_name" == "heist_aisc_many_chests" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests/model_200015872.pth"
elif [ "$env_name" == "maze_redline_yellowgem" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem/model_200015872.pth"
elif [ "$env_name" == "maze_aisc" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc/model_200015872.pth"
fi

if [ "$save_style" == "0" ]; then
    exp_name="save_observations"
else
    exp_name="save_latent_observations"
fi

python3 render.py \
    --exp_name ${exp_name} \
    --env_name ${env_name} \
    --distribution_mode hard \
    --param_name hard-plus \
    --model_file ${model_file} \
    --select_mode sample \
    --seed 8888 \
    --device gpu \
    --gpu_device ${gpu_device} \
    --save_as_npz \
    $(if [ "$save_style" == "0" ]; then echo "--save_observations"; else echo "--save_latent_observations"; fi)
