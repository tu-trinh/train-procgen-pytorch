#!/bin/bash

exp_name=$1
env_name=$2
model_file=$3
gpu_device=2

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
