#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=ask_for_help
#SBATCH --output=experiments/slurm/%j.out
#SBATCH --time=72:00:00
#SBATCH --qos scavenger
#SBATCH --partition scavenger

eval "$(/nas/ucb/tutrinh/anaconda3/bin/conda shell.bash hook)"
conda activate ood

wandb login 4a6017fc91542ffdb82ee3d6213e9cf0c11fd892

export CUDA_LAUNCH_BLOCKING=1
cd /nas/ucb/tutrinh/train-procgen-pytorch
export PYTHONPATH="$PYTHONPATH:$PWD"

env_name=$1
index=$2
risk_set=$3
detector_type=$4
gpu_device=$5
seed=8888

if [ "$risk_set" == "A" ]; then
    if [ "$index" == "6" ] || [ "$index" == "7" ]; then
        risk_values=(1 5 10)
    else
        risk_values=(10 20 30 40 50 60 70 80 90)
    fi
elif [ "$risk_set" == "B" ]; then
    if [ "$index" == "6" ] || [ "$index" == "7" ]; then
        risk_values=(20 30 40)
    else
        risk_values=(5 15 25 35 45 55 65 75 85 95)
    fi
fi

if [ "$env_name" == "coinrun_aisc" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/coinrun_aisc/model_200015872.pth"
    if [ "$detector_type" == "raw" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/coinrun/perfect_raw/2024-07-31__18-26-08__seed_8888"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/coinrun/perfect_raw/2024-07-24__01-05-57__seed_8888/network.tar"
    elif [ "$detector_type" == "latent" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/coinrun/perfect_latent/2024-07-31__18-32-08__seed_8888"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/coinrun/perfect_latent/2024-07-24__19-35-37__seed_8888/network.tar"
    fi
elif [ "$env_name" == "maze" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_aisc"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze/model_200015872.pth"
    if [ "$detector_type" == "raw" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/maze_aisc/perfect_raw/2024-07-31__18-27-37__seed_8888"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/maze_aisc/perfect_raw/2024-07-24__01-09-01__seed_8888/network.tar"
    elif [ "$detector_type" == "latent" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/maze_aisc/perfect_latent/2024-07-31__18-32-36__seed_8888"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/maze_aisc/perfect_latent/2024-07-24__12-37-32__seed_8888/network.tar"
    fi
elif [ "$env_name" == "maze_yellowstar_redgem" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_redline_yellowgem"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/maze_yellowstar_redgem/model_200015872.pth"
    if [ "$detector_type" == "raw" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/maze_redline_yellowgem/perfect_raw/2024-07-31__18-28-38__seed_8888"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/maze_redline_yellowgem/perfect_raw/2024-07-24__01-09-31__seed_8888/network.tar"
    elif [ "$detector_type" == "latent" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/maze_redline_yellowgem/perfect_latent/2024-07-31__18-33-06__seed_8888/"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/maze_redline_yellowgem/perfect_latent/2024-07-24__12-38-06__seed_8888/network.tar"
    fi
elif [ "$env_name" == "heist_aisc_many_keys" ]; then
    model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests/model_200015872.pth"
    percentile_dir="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_chests"
    expert_model_file="/nas/ucb/tutrinh/train-procgen-pytorch/logs/using/heist_aisc_many_keys/model_200015872.pth"
    if [ "$detector_type" == "raw" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/heist_aisc_many_chests/perfect_raw/2024-07-31__18-31-36__seed_8888"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/heist_aisc_many_chests/perfect_raw/2024-07-24__01-09-31__seed_8888/network.tar"
    elif [ "$detector_type" == "latent" ]; then
        detector_percentile_dir="/nas/ucb/tutrinh/yield_request_control/logs/test_detector/heist_aisc_many_chests/perfect_latent/2024-07-31__18-33-06__seed_8888"
        detector_model_file="/nas/ucb/tutrinh/yield_request_control/logs/train_detector/heist_aisc_many_chests/perfect_latent/2024-07-24__12-38-40__seed_8888/network.tar"
    fi
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
    "recieve_help_test_svdd_raw"
    "receive_help_test_svdd_latent"
)
metrics=(
    "msp"
    "sampled_p"
    "ml"
    "sampled_l"
    "ent"
    "random"
    "detector"
    "detector"
)
for idx in "${!risk_values[@]}"; do
    risk=${risk_values[idx]}
    cmd="python3 render.py \
        --exp_name "${names[index]}_risk_${risk}" \
        --env_name ${env_name} \
        --distribution_mode hard \
        --param_name hard-plus \
        --model_file ${model_file} \
        --percentile_dir $( [ $index -eq 6 ] || [ $index -eq 7 ] && echo ${detector_percentile_dir} || echo ${percentile_dir} ) \
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
        --save_run"
    if [ $index -eq 6 ] || [ $index -eq 7 ]; then
        cmd="${cmd} --detector_model_file ${detector_model_file}"
    fi
    if [ "$detector_type" == "latent" ]; then
        cmd="${cmd} --use_latent"
    fi
    eval $cmd
done

