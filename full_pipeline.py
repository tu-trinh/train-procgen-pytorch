import subprocess
import re
import time


def run_render_script(exp_name, env_name, model_file, save_run = False, ood_metric = None, risk = None, percentile_dir = None, expert_model_file = None, expert_cost = 0, switching_cost = 0):
    command = [
        "python3", "render.py",
        "--exp_name", exp_name,
        "--env_name", env_name,
        "--distribution_mode", "hard",
        "--param_name", "hard-plus",
        "--model_file", model_file,
        "--select_mode", "sample",
        "--quant_eval",
        "--seed", "8888"
    ]
    if save_run:
        command.append("--save_run")
    if ood_metric is not None:
        assert risk is not None and percentile_dir is not None, "Check risk and percentile_dir"
        command.extend(["--ood_metric", ood_metric])
        command.extend(["--risk", risk])
        command.extend(["--percentile_dir", percentile_dir])
    if expert_model_file is not None:
        command.extend(["--expert_cost", expert_cost])
        command.extend(["--switching_cost", switching_cost])
    
    result = subprocess.run(command, capture_output = True, text = True)
    if result.returncode != 0:
        print(f"Error running render.py: {result.stderr}")
        return None
    
    logdir = None
    for line in result.stdout.splitlines():
        if "Logging dir:" in line:
            logdir = line.split("Logging dir:\n")[-1].strip()
            break
    return logdir

def main():
    exp_name = "debug"
    env_name = "coinrun"
    model_file = "logs/train/coinrun/og_actions/2024-05-25__19-35-47__seed_8888/model_200015872.pth"
    logdir = run_render_script(exp_name, env_name, model_file)
    print("LOGDIR")
    print(logdir)

if __name__ == "__main__":
    main()
