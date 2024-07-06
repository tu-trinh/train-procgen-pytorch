import matplotlib.pyplot as plt
import os
import numpy as np
import ast
import argparse
import re


PLOT_PERF_VS_PERC = 1  # performance vs percentile
PLOT_PROP_VS_PERC = 2  # proportion of asking for help vs percentile
PLOT_PERF_VS_PROP = 3  # performance vs proportion of asking for help
PLOT_PROP_VS_TIME = 4  # proportion of asking for help vs run timestep
PLOT_COST_BREAKEVEN = 5  # how many times ask for help to break even penalties


parser = argparse.ArgumentParser()
parser.add_argument("--plots", "-p", type = int, nargs = "+", default = [1, 2, 3, 4, 5])
parser.add_argument("--bucketed", "-b", action = "store_true")  # only applies to plot 3!
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
log_dir = "logs/procgen/coinrun_aisc"
for exp_dir in os.listdir(log_dir):
    # CHANGE HERE #
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
        render_logs = os.listdir(os.path.join(log_dir, exp_dir))
        helped_logs[log_key][perc] = os.path.join(log_dir, exp_dir, sorted(render_logs)[-1])  # always get last/most updated one


if PLOT_PERF_VS_PERC in args.plots:
    # Weak agent in train environment
    with open(os.path.join("logs/procgen/coinrun/eval_train_og/RENDER_seed_8888_06-17-2024_18-10-21", quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                train_performance = eval(line[len("all rewards: "):].strip())
                break
    train_perf_mean, train_perf_std = get_mean_and_std(train_performance)
    # Weak agent in test environment
    # any help_test_* one is fine for test performance
    with open(os.path.join("logs/procgen/coinrun_aisc/help_test_ent_og/RENDER_seed_8888_06-10-2024_20-50-28", quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                test_performance = eval(line[len("all rewards: "):].strip())
                break
    test_perf_mean, test_perf_std = get_mean_and_std(test_performance)
    # Expert in test environment
    with open(os.path.join("logs/procgen/coinrun_aisc/expert/RENDER_seed_8888_06-21-2024_08-05-09", quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                expert_performance = eval(line[len("all rewards: "):].strip())
                break
    expert_perf_mean, expert_perf_std = get_mean_and_std(expert_performance)


percentiles = range(10, 91, 10)
run_portions = [round(k, 1) for k in np.arange(0.1, 1.01, 0.1)]
if PLOT_PERF_VS_PERC in args.plots:
    fig1, axes1 = plt.subplots(2, 3, figsize = (15, 8))
if PLOT_PROP_VS_PERC in args.plots:
    fig2, axes2 = plt.subplots(2, 3, figsize = (15, 8))
if PLOT_PERF_VS_PROP in args.plots:
    fig3, axes3 = plt.subplots(2, 3, figsize = (15, 8))
if PLOT_PROP_VS_TIME in args.plots:
    fig4, axes4 = plt.subplots(2, 3, figsize = (15, 8))

i = 0
mega_mean_timestep_achieved = []
for metric in helped_logs:
    print("Doing metric", metric)
    rew_by_perc = []
    adj_rew_by_perc = []
    help_props_by_perc = []
    help_asks_by_timestep = {round(k, 1): [] for k in run_portions}
    for perc in percentiles:
        with open(os.path.join(helped_logs[metric][perc], quant_eval_file_name), "r") as f:
            evaluation = f.readlines()
        for line in evaluation:
            if PLOT_PERF_VS_PERC in args.plots or PLOT_PERF_VS_PROP in args.plots:
                if "all rewards" in line.lower():
                    rew_by_perc.append(eval(line[len("all rewards: "):].strip()))
                if "all adjusted rewards" in line.lower():
                    adj_rew_by_perc.append(eval(line[len("all adjusted rewards: "):].strip()))
            if PLOT_PERF_VS_PROP in args.plots or PLOT_PROP_VS_PERC in args.plots:
                if "help times" in line.lower():
                    help_times = eval(evaluation[-1])
                    help_props = []
                    for helps in help_times:
                        help_props.append(sum(helps) / len(helps))
                    help_props_by_perc.append(help_props)
            if PLOT_PROP_VS_TIME in args.plots:
                if "help times" in line.lower():
                    help_times = eval(evaluation[-1])
                    num_segments = 10
                    for helps in help_times:
                        run_length = len(helps)
                        segment_length = run_length // num_segments
                        for k in range(num_segments):
                            start = k * segment_length
                            end = start + segment_length if i < num_segments - 1 else run_length
                            segment = helps[start:end]
                            help_asks_by_timestep[round((k + 1) / 10, 1)].append(sum(segment) / len(segment))
            if "mean timestep achieved" in line.lower():
                mega_mean_timestep_achieved.append(int(re.search(r"(\d+)", line).group(1)))
    
    # 1: Plotting performance vs percentile
    if PLOT_PERF_VS_PERC in args.plots:
        ax = axes1[i // 3][i % 3]
        rew_means, rew_stds = get_mean_and_std_nested(rew_by_perc)
        adj_rew_means, adj_rew_stds = get_mean_and_std_nested(adj_rew_by_perc)
        ax.plot(percentiles, rew_means, color = colors[metric], label = f"Reward (mean SD: {round(np.mean(rew_stds), 2)})", linewidth = 2.5)
        # ax.fill_between(percentiles, rew_means - rew_stds, rew_means + rew_stds, color = colors[metric], alpha = 0.3)
        ax.plot(percentiles, adj_rew_means, color = colors[metric], linestyle = "dashed", label = f"Adj. Reward (mean SD: {round(np.mean(adj_rew_stds), 2)})", linewidth = 2.5)
        ax.fill_between(percentiles, adj_rew_means - adj_rew_stds, adj_rew_means + adj_rew_stds, color = colors[metric], alpha = 0.3)
        ax.plot(percentiles, [train_perf_mean] * len(percentiles), color = "black", label = f"Train (SD: {round(train_perf_std, 2)})", linewidth = 2.5)
        # ax.fill_between(percentiles, [train_perf_mean - train_perf_std] * len(percentiles), [train_perf_mean + train_perf_std] * len(percentiles), color = "black", alpha = 0.3)
        ax.plot(percentiles, [test_perf_mean] * len(percentiles), color = "gray", label = f"Test (SD: {round(test_perf_std, 2)})", linewidth = 2.5)
        # ax.fill_between(percentiles, [test_perf_mean - test_perf_std] * len(percentiles), [test_perf_mean + test_perf_std] * len(percentiles), color = "gray", alpha = 0.3)
        ax.plot(percentiles, [expert_perf_mean] * len(percentiles), color = "tab:brown", label = f"Expert (SD: {round(expert_perf_std, 2)})", linewidth = 2.5)
        # ax.fill_between(percentiles, [expert_perf_mean - expert_perf_std] * len(percentiles), [expert_perf_mean + expert_perf_std] * len(percentiles), color = "brown", alpha = 0.3)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Performance")
        # ax.set_yticks(range(-1, 11))
        ax.legend()
        # axes1[-1][-1].text(0.5, 0.5, "made you look", horizontalalignment = "center", verticalalignment = "center", transform = axes1[-1][-1].transAxes, fontsize = 6)
        fig1.suptitle("Performance by Percentile")
        fig1.tight_layout()
        fig1.subplots_adjust(hspace = 0.3, wspace = 0.2)
        fig1.savefig(f"{args.prefix}_performance_by_percentile{args.suffix}.png")
        print("Done one plot of", PLOT_PERF_VS_PERC)

    # 2: Plotting ask-for-help percentage vs percentile
    if PLOT_PROP_VS_PERC in args.plots:
        ax = axes2[i // 3][i % 3]
        help_prop_means, help_prop_stds = get_mean_and_std_nested(help_props_by_perc)
        ax.plot(percentiles, help_prop_means, color = colors[metric])
        ax.fill_between(percentiles, help_prop_means - help_prop_stds, help_prop_means + help_prop_stds, color = colors[metric], alpha = 0.3)
        # for x, y in zip(percentiles, help_prop_means):
            # ax.hlines(y, 0, x, colors = "gray", linestyles = "dashed", linewidth = 0.5)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Ask-For-Help Percentage")
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        # axes2[-1][-1].text(0.5, 0.5, "again?", horizontalalignment = "center", verticalalignment = "center", transform = axes2[-1][-1].transAxes, fontsize = 6)
        fig2.suptitle("Ask-For-Help Percentage vs. Percentile")
        fig2.tight_layout()
        fig2.subplots_adjust(hspace = 0.3, wspace = 0.2)
        fig2.savefig(f"{args.prefix}_help_percentage_by_percentile{args.suffix}.png")
        print("Done one plot of", PLOT_PROP_VS_PERC)

    # 3: Plotting performance vs ask-for-help percentage
    if PLOT_PERF_VS_PROP in args.plots:
        ax = axes3[i // 3][i % 3]
        if args.bucketed:
            # (x, y)_i = (average AFHP for percentile i, average performance for percentile i)
            afhp_means, afhp_stds = get_mean_and_std_nested(help_props_by_perc)
            rew_means, rew_stds = get_mean_and_std_nested(rew_by_perc)
            adj_rew_means, adj_rew_stds = get_mean_and_std_nested(adj_rew_by_perc)
            ax.plot(afhp_means, rew_means, color = colors[metric], label = f"Reward (mean SD: {np.round(np.mean(rew_stds), 2)})")
            # ax.fill_between(afhp_means, rew_means - rew_stds, rew_means + rew_stds, color = colors[metric], alpha = 0.3)
            ax.plot(afhp_means, adj_rew_means, color = colors[metric], linestyle = "dashed", label = f"Adj. Reward (mean SD: {np.round(np.mean(adj_rew_stds), 2)})")
            ax.fill_between(afhp_means, adj_rew_means - adj_rew_stds, adj_rew_means + adj_rew_stds, color = colors[metric], alpha = 0.3)
            # bucket_edges = np.linspace(0, 1, 21)
            # bucket_indices = np.digitize(flattened_help_props, bucket_edges)
            # bucket_rew_means = []
            # bucket_adj_rew_means = []
            # bucket_rew_stds = []
            # bucket_adj_rew_stds = []
            # for k in range(1, len(bucket_edges)):
                # bucket_mask = bucket_indices == k
                # if np.any(bucket_mask):
                    # bucket_rew_means.append(np.mean(flattened_rews[bucket_mask]))
                    # bucket_rew_stds.append(np.std(flattened_rews[bucket_mask]))
                    # bucket_adj_rew_means.append(np.mean(flattened_adj_rews[bucket_mask]))
                    # bucket_adj_rew_stds.append(np.std(flattened_adj_rews[bucket_mask]))
                # else:
                    # print("No datapoint for ask-for-help percentage", k)
                    # bucket_rew_means.append(np.nan)
                    # bucket_rew_stds.append(np.nan)
                    # bucket_adj_rew_means.append(np.nan)
                    # bucket_adj_rew_stds.append(np.nan)
            # bucket_rew_means = np.array(bucket_rew_means)
            # bucket_rew_stds = np.array(bucket_rew_stds)
            # bucket_adj_rew_means = np.array(bucket_adj_rew_means)
            # bucket_adj_rew_stds = np.array(bucket_adj_rew_stds)
            # bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
            # ax.plot(bucket_centers, bucket_rew_means, color = colors[metric], linestyle = "solid", label = f"Reward")
            # ax.fill_between(bucket_centers, bucket_rew_means - bucket_rew_stds, bucket_rew_means + bucket_rew_stds, color = colors[metric], alpha = 0.3)
            # ax.plot(bucket_centers, bucket_adj_rew_means, color = colors[metric], linestyle = "dashed", label = f"Adj. Reward")
            # ax.fill_between(bucket_centers, bucket_adj_rew_means - bucket_adj_rew_stds, bucket_adj_rew_means + bucket_adj_rew_stds, color = colors[metric], alpha = 0.3)
        else:
            flattened_help_props = np.array([help_prop for help_props in help_props_by_perc for help_prop in help_props])
            flattened_rews = np.array([rew for rews in rew_by_perc for rew in rews])
            flattened_adj_rews = np.array([adj_rew for adj_rews in adj_rew_by_perc for adj_rew in adj_rews])
            sorted_idx = np.argsort(flattened_help_props)
            flattened_help_props = flattened_help_props[sorted_idx]
            flattened_rews = flattened_rews[sorted_idx]
            flattened_adj_rews = flattened_adj_rews[sorted_idx]
            # ax.scatter(flattened_help_props, flattened_rews, color = colors[metric], label = f"Reward")
            every = 5
            ax.scatter(flattened_help_props[::every], flattened_adj_rews[::every], color = colors[metric], marker = "+", label = f"Adj. Reward")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Ask-For-Help Percentage")
        ax.set_ylabel("Performance")
        # ax.set_yticks(range(-10, 11))
        ax.legend()
        # axes3[-1][-1].text(0.5, 0.5, "( ͡° ͜ʖ ͡°)", horizontalalignment = "center", verticalalignment = "center", transform = axes3[-1][-1].transAxes, fontsize = 5)
        fig3.suptitle("Performance vs. Ask-For-Help Percentage")
        fig3.tight_layout()
        fig3.subplots_adjust(hspace = 0.3, wspace = 0.2)
        fig3.savefig(f"{args.prefix}_performance_by_help_percentage{'_bucketed' if args.bucketed else ''}{args.suffix}.png")
        print("Done one plot of", PLOT_PERF_VS_PROP)

    # 4: Plotting proportion of asking for help vs run timestep
    if PLOT_PROP_VS_TIME in args.plots:
        ax = axes4[i // 3][i % 3]
        asks_mean = np.array([np.mean(asks) for _, asks in help_asks_by_timestep.items()])
        asks_std = np.array([np.std(asks) for _, asks in help_asks_by_timestep.items()])
        ax.plot(run_portions, asks_mean, color = colors[metric])
        ax.fill_between(run_portions, asks_mean - asks_std, asks_mean + asks_std, color = colors[metric], alpha = 0.3)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Run Portion")
        ax.set_ylabel("Ask-For-Help Percentage")
        # axes4[-1][-1].text(0.5, 0.5, "hey there", horizontalalignment = "center", verticalalignment = "center", transform = axes4[-1][-1].transAxes, fontsize = 5)
        fig4.suptitle("Ask-For-Help Percentage by Segment of Run")
        fig4.tight_layout()
        fig4.subplots_adjust(hspace = 0.3, wspace = 0.2)
        fig4.savefig(f"{args.prefix}_help_percentage_by_timestep{args.suffix}.png")
        print("Done one plot of", PLOT_PROP_VS_TIME)

    i += 1

if PLOT_COST_BREAKEVEN in args.plots:
    mega_mean_timestep_achieved = round(np.mean(mega_mean_timestep_achieved))
    print(mega_mean_timestep_achieved)
    breakeven_func = np.vectorize(lambda cost: 256 / (mega_mean_timestep_achieved * cost))
    cost_range = np.arange(0.1, 5.01, 0.1)
    plt.plot(cost_range, breakeven_func(cost_range))
    plt.title("How often must agent ask for help to break even?")
    plt.xlabel("Ask-For-Help Cost")
    plt.ylabel("Ask-For-Help Percentage")
    plt.savefig("ask_for_help_breakeven.png")
    print("Done breakeven plot")
