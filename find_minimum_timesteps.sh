#!/bin/bash

model_paths=(
	"model_393216.pth"
	"model_786432.pth"
	"model_1572864.pth"
	"model_3145728.pth"
	"model_6291456.pth"
	"model_10027008.pth"
	"model_12517376.pth"
	"model_20054016.pth"
	"model_25034752.pth"
	"model_30015488.pth"
	"model_40042496.pth"
	"model_50003968.pth"
	"model_60030976.pth"
	"model_70057984.pth"
	"model_80019456.pth"
	"model_90046464.pth"
	"model_100007936.pth"
	"model_110034944.pth"
	"model_120061952.pth"
	"model_130023424.pth"
	"model_140050432.pth"
	"model_150011904.pth"
	"model_160038912.pth"
	"model_170000384.pth"
	"model_180027392.pth"
	"model_190054400.pth"
	"model_200015872.pth"
)

for model_path in "${model_paths[@]}"; do
	command="python3 render.py --exp_name find_minimum_timesteps --env_name coinrun --distribution_mode hard --param_name hard-plus --model_file logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/${model_path} --select_mode sample --quant_eval"
	echo "Trying model ${model_path}"
	$command
done
