#!/bin/bash

exp_name="eval_weak_test"
env_name="heist_aisc_many_chests"
model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_keys/model_200015872.pth"
gpu_device=7

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
    --quant_eval
