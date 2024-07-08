import os
import numpy as np
import ast
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument("--train_env_log_dir", type = str, help = "for example: 'logs/procgen/coinrun'")
parser.add_argument("--test_env_log_dir", type = str, help = "for example: 'logs/procgen/coinrun_aisc'")
parser.add_argument("--query_cost", type = int, default = 1)
parser.add_argument("--switching_cost", type = int, default = 0)
parser.add_argument("--prefix", type = str, default = "receive_help")
parser.add_argument("--suffix", type = str, default = "")
args = parser.parse_args()
if args.suffix != "" and args.suffix[-1] != "_":
    args.suffix = "_" + args.suffix


def get_mean_and_std(arr):
    assert isinstance(arr[0], int) or isinstance(arr[0], float), "Whoopsie"
    return np.mean(arr), np.std(arr)

def get_mean_and_std_nested(nested_arr):
    assert isinstance(nested_arr[0], list) or isinstance(nested_arr[0], np.array), "Oh no"
    means, stds = [], []
    for arr in nested_arr:
        mean, std = get_mean_and_std(arr)
        means.append(mean)
        stds.append(std)
    return np.array(means), np.array(stds)

def inf_list_eval(list_str):
    list_str = list_str.replace("inf", "'__inf__'")
    parsed_list = ast.literal_eval(list_str)
    parsed_list = [float("inf") if x == "__inf__" else x for x in parsed_list]
    return parsed_list


quant_eval_file_name = "AAA_quant_eval_model_200015872.txt"
colors = {"max prob": "blue", "sampled prob": "green", "max logit": "red", "sampled logit": "purple", "entropy": "orange", "random": "gold"}
helped_logs = {"max prob": {}, "sampled prob": {}, "max logit": {}, "sampled logit": {}, "entropy": {}, "random": {}}
for exp_dir in os.listdir(args.test_env_log_dir):
    # TODO: CHANGE HERE #
    if exp_dir.startswith("receive") and "unique_actions" not in exp_dir:
        perc = int(re.search(r"(\d+)", exp_dir).group(1))
        if "ent" in exp_dir:
            log_key = "entropy"
        elif "max" in exp_dir:
            log_key = "max prob" if "prob" in exp_dir else "max logit"
        elif "sample" in exp_dir:
            log_key = "sampled prob" if "prob" in exp_dir else "sampled logit"
        else:
            log_key = "random"
        render_logs = os.listdir(os.path.join(args.test_env_log_dir, exp_dir))
        helped_logs[log_key][perc] = os.path.join(args.test_env_log_dir, exp_dir, sorted(render_logs)[-1])  # always get last/most updated one


if "coinrun" in args.train_env_log_dir:
    # Weak agent in train environment
    with open(os.path.join("logs/procgen/coinrun/eval_train_og/RENDER_seed_8888_06-17-2024_18-10-21", quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                train_performance = eval(line[len("all rewards: "):].strip())
                break
    # Weak agent in test environment
    # any help_test_* one is fine for test performance
    with open(os.path.join("logs/procgen/coinrun_aisc/help_test_ent_og/RENDER_seed_8888_06-10-2024_20-50-28", quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                test_performance = eval(line[len("all rewards: "):].strip())
                break
    # Expert in test environment
    with open(os.path.join("logs/procgen/coinrun_aisc/expert/RENDER_seed_8888_06-21-2024_08-05-09", quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                expert_performance = eval(line[len("all rewards: "):].strip())
                break
else:
    # Weak agent in train environment
    temp = os.listdir(os.path.join(args.train_env_log_dir, "eval_weak_train"))
    with open(os.path.join(args.train_env_log_dir, "eval_weak_train", sorted(temp)[-1], quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                train_performance = eval(line[len("all rewards: "):].strip())
                break
    # Weak agent in test environment
    temp = os.listdir(os.path.join(args.test_env_log_dir, "eval_weak_test"))
    with open(os.path.join(args.train_env_log_dir, "eval_weak_train", sorted(temp)[-1], quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
            for line in evaluation:
                if "all rewards" in line.lower():
                    test_performance = eval(line[len("all rewards: "):].strip())
                    break
    # Expert in test environment
    temp = os.listdir(os.path.join(args.test_env_log_dir, "eval_expert"))
    with open(os.path.join(args.test_env_log_dir, "eval_expert", sorted(temp)[-1], quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                expert_performance = eval(line[len("all rewards: "):].strip())
                break
train_perf_mean, train_perf_std = get_mean_and_std(train_performance)
test_perf_mean, test_perf_std = get_mean_and_std(test_performance)
expert_perf_mean, expert_perf_std = get_mean_and_std(expert_performance)


percentiles = range(5, 96, 5)
print("Train:", args.train_env_log_dir.split("/")[-1])
print("Test:", args.test_env_log_dir.split("/")[-1])
for metric in helped_logs:
    rew_by_perc = []
    adj_rew_by_perc = []
    help_props_by_perc = []
    help_asks_by_timestep = {round(k, 1): [] for k in run_portions}
    for perc in percentiles:
        with open(os.path.join(helped_logs[metric][perc], quant_eval_file_name), "r") as f:
            evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                rewards = eval(line[len("all rewards: "):].strip())
                rew_by_perc.append(rewards)
            if "all queries" in line.lower():
                queries = eval(line[len("all queries: "):].strip())
            if "all switches" in line.lower():
                switches = eval(line[len("all switches: "):].strip())
            if "help times" in line.lower():
                help_times = eval(evaluation[-1])
                help_props = []
                for helps in help_times:
                    help_props.append(sum(helps) / len(helps))
                help_props_by_perc.append(help_props)
        for reward, query, switch in zip(rewards, queries, switches):
            adjusted_reward = reward
            for q, s in zip(query, switch):
                if query == 1:
                    adjusted_reward -= 10/256 * args.query_cost
                if switch == 1:
                    adjusted_reward -= 10/256 * args.switching_cost
            adj_rew_by_perc.append(adjusted_reward)

    # (x, y)_i = (average AFHP for percentile i, average performance for percentile i)
    afhp_means, afhp_stds = get_mean_and_std_nested(help_props_by_perc)
    rew_means, rew_stds = get_mean_and_std_nested(rew_by_perc)
    adj_rew_means, adj_rew_stds = get_mean_and_std_nested(adj_rew_by_perc)
    # Adding 0% ask for help
    afhp_means.insert(0, 0)
    rew_means.insert(0, test_perf_mean)
    adj_rew_means.insert(0, test_perf_mean)
    # Adding 100% ask for help
    afhp_means.append(1)
    rew_means.append(expert_perf_mean)
    adj_rew_means.append(expert_perf_mean - (10/256 * args.query_cost * 256))
    reward_area = np.trapz(rew_means, afhp_means)
    adjusted_reward_area = np.trapz(adj_rew_means, afhp_means)
    print("Reward AUC:", round(reward_area, 2))
    print("Adjusted reward AUC:", round(adjusted_reward_area, 2))
