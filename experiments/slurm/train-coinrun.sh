#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=coinrun_easy
#SBATCH --output=experiments/slurm/%x.out
#SBATCH --time=72:00:00
#SBATCH --qos scavenger

eval "$(/nas/ucb/tutrinh/anaconda3/bin/conda shell.bash hook)"
conda activate ood

wandb login 4a6017fc91542ffdb82ee3d6213e9cf0c11fd892

export CUDA_LAUNCH_BLOCKING=1
cd /nas/ucb/tutrinh/train-procgen-pytorch
export PYTHONPATH="$PYTHONPATH:$PWD"

python3 train.py \
	--env_name coinrun \
	--exp_name easy \
	--num_levels 100000 \
	--distribution_mode easy \
	--param_name easy-200 \
	--num_timesteps 200000000 \
	--save_timesteps 390625 781250 1562500 3125000 6250000 10000000 12500000 20000000 25000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000 110000000 120000000 130000000 140000000 150000000 160000000 170000000 180000000 190000000 200000000 \
	--seed 8888 \
	--random_percent 0 \
	--use_wandb

