import os
import numpy as np
import ast
import argparse
import re
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument("--train_env", type = str, required = True)
parser.add_argument("--test_env", type = str, required = True)
parser.add_argument("--include", type = str, nargs = "+", default = ["max prob", "sampled prob", "max logit", "sampled logit", "entropy", "random", "svdd"])
parser.add_argument("--exclude", type = str, nargs = "+", default = [])
parser.add_argument("--query_cost", type = int, default = 1)
parser.add_argument("--switching_cost", type = int, default = 0)
args = parser.parse_args()


def get_mean_and_std(arr):
    assert isinstance(arr[0], int) or isinstance(arr[0], float), "Whoopsie"
    return np.mean(arr), stats.sem(arr)

def get_mean_and_std_nested(nested_arr):
    assert isinstance(nested_arr[0], list) or isinstance(nested_arr[0], np.array), "Oh no"
    means, stds = [], []
    for arr in nested_arr:
        mean, std = get_mean_and_std(arr)
        means.append(mean)
        stds.append(std)
    return means, stds

def inf_list_eval(list_str):
    list_str = list_str.replace("inf", "'__inf__'")
    parsed_list = ast.literal_eval(list_str)
    parsed_list = [float("inf") if x == "__inf__" else x for x in parsed_list]
    return parsed_list

def flatten_list(lst):
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]


quant_eval_file_name = "AAA_quant_eval_model_200015872.txt"
colors = {"max prob": "blue", "sampled prob": "green", "max logit": "red", "sampled logit": "purple", "entropy": "orange", "random": "gold", "svdd": "fuchsia"}
helped_logs = {"max prob": {}, "sampled prob": {}, "max logit": {}, "sampled logit": {}, "entropy": {}, "random": {}, "svdd": {}}
for exp_dir in os.listdir(f"logs/procgen/{args.test_env}"):
    # TODO: CHANGE HERE #
    if exp_dir.startswith("receive") and "unique_actions" not in exp_dir:
        perc = int(re.search(r"(\d+)", exp_dir).group(1))
        if "ent" in exp_dir:
            log_key = "entropy"
        elif "max" in exp_dir:
            log_key = "max prob" if "prob" in exp_dir else "max logit"
        elif "sample" in exp_dir:
            log_key = "sampled prob" if "prob" in exp_dir else "sampled logit"
        elif "svdd" in exp_dir:
            log_key = "svdd"
        else:
            log_key = "random"
        render_logs = os.listdir(os.path.join(f"logs/procgen/{args.test_env}", exp_dir))
        helped_logs[log_key][perc] = os.path.join(f"logs/procgen/{args.test_env}", exp_dir, sorted(render_logs)[-1])  # always get last/most updated one


# Weak agent in train environment
temp = os.listdir(os.path.join(f"logs/procgen/{args.train_env}", "eval_weak_train"))
with open(os.path.join(f"logs/procgen/{args.train_env}", "eval_weak_train", sorted(temp)[-1], quant_eval_file_name), "r") as f:
    evaluation = f.readlines()
    for line in evaluation:
        if "all rewards" in line.lower():
            train_performance = eval(line[len("all rewards: "):].strip())
            break
# Weak agent in test environment
temp = os.listdir(os.path.join(f"logs/procgen/{args.test_env}", "eval_weak_test"))
with open(os.path.join(f"logs/procgen/{args.test_env}", "eval_weak_test", sorted(temp)[-1], quant_eval_file_name), "r") as f:
    evaluation = f.readlines()
    for line in evaluation:
        if "all rewards" in line.lower():
            test_performance = eval(line[len("all rewards: "):].strip())
            break
# Expert in test environment
temp = os.listdir(os.path.join(f"logs/procgen/{args.test_env}", "eval_expert"))
    evaluation = f.readlines()
    for line in evaluation:
        if "all rewards" in line.lower():
            expert_performance = eval(line[len("all rewards: "):].strip())
            break
# Skyline

print("Train:", args.train_env)
print("Test:", args.test_env)
if args.test_env == "heist_aisc_many_chests":
    norm_factor = 8
elif args.test_env == "heist_aisc_many_keys":
    norm_factor = 4
else:
    norm_factor = 10
train_perf_mean, train_perf_std = get_mean_and_std(train_performance)
print(f"Weak agent reward = {round(train_perf_mean / norm_factor, 2)}")
train_perf_mean /= norm_factor
test_perf_mean, test_perf_std = get_mean_and_std(test_performance)
print(f"Weak agent on TEST reward = {round(test_perf_mean / norm_factor, 2)}")
test_perf_mean /= norm_factor
expert_perf_mean, expert_perf_std = get_mean_and_std(expert_performance)
print(f"Expert reward = {round(expert_perf_mean / norm_factor, 2)}")
expert_perf_mean /= norm_factor


percentiles = range(5, 96, 5)
table_data = {"metric": [], "AUC": [], "mean reward": []}
include_keys = [k for k in helped_logs.keys() if k in args.include and k not in args.exclude]
for metric in include_keys:
    rew_by_perc = []
    adj_rew_by_perc = []
    help_props_by_perc = []
    for perc in percentiles:
        try:
            with open(os.path.join(helped_logs[metric][perc], quant_eval_file_name), "r") as f:
                evaluation = f.readlines()
            run_lengths = []
            for line in evaluation:
                if "all rewards" in line.lower():
                    rewards = eval(line[len("all rewards: "):].strip())
                    assert all([reward <= 10 for reward in rewards]), f"wtf {metric} {perc}"
                    rew_by_perc.append([reward / norm_factor for reward in rewards])
                    # print(f"Mean reward for {perc} = {np.mean(rewards)}")
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
                    run_lengths.append(len(helps))
            curr_idx = 0
            perc_adjusted_rewards = []
            for reward, run_length in zip(rewards, run_lengths):
                curr_run_length = 0
                adjusted_reward = reward
                while curr_run_length < run_length:
                    if queries[curr_idx] == 1:
                        adjusted_reward -= 10/256 * args.query_cost
                    if switches[curr_idx] == 1:
                        adjusted_reward -= 10/256 * args.switching_cost
                    curr_idx += 1
                    curr_run_length += 1
                perc_adjusted_rewards.append(adjusted_reward)
            adj_rew_by_perc.append(perc_adjusted_rewards)
            # print(f"Mean adj. reward for {perc} = {np.mean(perc_adjusted_rewards)}")
        except Exception as e:
            print(e)
            print(f"Missing data for {metric} at percentile {perc}")

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
    table_data["metric"].append(metric)
    table_data["mean reward"].append(round(np.mean(flatten_list(rew_by_perc)), 2))
    # table_data["std. err reward"].append(round(stats.sem(flatten_list(rew_by_perc), axis = None), 2))
    table_data["AUC"].append(round(reward_area, 2))
    # table_data["adj. reward AUC"].append(round(adjusted_reward_area, 2))

headings = [key.capitalize() for key in table_data.keys()]
values = list(zip(*table_data.values()))
column_widths = [max(len(str(item)) for item in [heading] + list(column)) for heading, column in zip(headings, table_data.values())]
row_format = "| " + " | ".join(f"{{:<{width}}}" for width in column_widths) + " |"
print(row_format.format(*headings))
print("-" * (sum(column_widths) + 3 * len(headings) + 1))
for value_set in values:
    print(row_format.format(*value_set))

