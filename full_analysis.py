import matplotlib.pyplot as plt
import os
import numpy as np
import ast
import argparse
import re
from scipy import stats


PLOT_PERF_VS_PERC = 1  # performance vs percentile
PLOT_PROP_VS_PERC = 2  # proportion of asking for help vs percentile
PLOT_PERF_VS_PROP = 3  # performance vs proportion of asking for help
PLOT_PROP_VS_TIME = 4  # proportion of asking for help vs run timestep
PLOT_COST_BREAKEVEN = 5  # how many times ask for help to break even penalties

parser = argparse.ArgumentParser()
# General arguments
parser.add_argument("--train_env", type = str, required = True)
parser.add_argument("--test_env", type = str, required = True)
parser.add_argument("--query_cost", type = int, default = 1)
parser.add_argument("--switching_cost", type = int, default = 0)
# Grand metric arguments
parser.add_argument("--grand_metric", "-gm", action = "store_true", default = False)
parser.add_argument("--include", type = str, nargs = "+", default = ["max prob", "sampled prob", "max logit", "sampled logit", "entropy", "random", "svdd_raw", "svdd_latent"])
parser.add_argument("--exclude", type = str, nargs = "+", default = [])
# Plotting arguments
parser.add_argument("--plotting", "-pt", action = "store_true", default = False)
parser.add_argument("--plots", type = int, nargs = "+", default = [1, 2, 3])
parser.add_argument("--bucketed", "-b", action = "store_true")  # only applies to plot 3!
parser.add_argument("--grouped", "-g", action = "store_true")  # only applies to plot 3!
parser.add_argument("--prefix", type = str, default = "receive_help")
parser.add_argument("--suffix", type = str, default = "")

args = parser.parse_args()
assert (args.grand_metric and not args.plotting) or (args.plotting and not args.grand_metric), "Only one of grand_metric or plotting at a time"
if args.suffix != "" and args.suffix[-1] != "_":
    args.suffix = "_" + args.suffix
args.prefix += "_" + args.test_env
print("Train:", args.train_env)
print("Test:", args.test_env)


def get_mean_and_std(arr):
    assert isinstance(arr[0], int) or isinstance(arr[0], float), "Whoopsie"
    return np.mean(arr), stats.sem(arr)

def get_mean_and_std_nested(nested_arr, as_array):
    assert isinstance(nested_arr[0], list) or isinstance(nested_arr[0], np.array), "Oh no"
    means, stds = [], []
    for arr in nested_arr:
        mean, std = get_mean_and_std(arr)
        means.append(mean)
        stds.append(std)
    if as_array:
        return np.array(means), np.array(stds)
    return means, stds

def inf_list_eval(list_str):
    list_str = list_str.replace("inf", "'__inf__'")
    parsed_list = ast.literal_eval(list_str)
    parsed_list = [float("inf") if x == "__inf__" else x for x in parsed_list]
    return parsed_list

def flatten_list(lst):
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]


quant_eval_file_name = "AAA_quant_eval_model_200015872.txt"
colors = {"max prob": "blue", "sampled prob": "green", "max logit": "red", "sampled logit": "purple", "entropy": "orange", "random": "gold", "svdd_raw": "fuchsia", "svdd_latent": "lightseagreen"}
helped_logs = {"max prob": {}, "sampled prob": {}, "max logit": {}, "sampled logit": {}, "entropy": {}, "random": {}, "svdd_raw": {}, "svdd_latent": {}}
log_dir = f"logs/procgen/{args.test_env}"
for exp_dir in os.listdir(log_dir):
    # TODO: CHANGE HERE #
    if exp_dir.startswith("rec") and "unique_actions" not in exp_dir and ("svdd" not in exp_dir or ("svdd_raw" in exp_dir or "svdd_latent" in exp_dir)):
        perc = int(re.search(r"(\d+)", exp_dir).group(1))
        if "_ent" in exp_dir:
            log_key = "entropy"
        elif "max" in exp_dir:
            log_key = "max prob" if "prob" in exp_dir else "max logit"
        elif "sample" in exp_dir:
            log_key = "sampled prob" if "prob" in exp_dir else "sampled logit"
        elif "svdd" in exp_dir:
            log_key = "svdd_raw" if "raw" in exp_dir else "svdd_latent"
        else:
            log_key = "random"
        render_logs = os.listdir(os.path.join(f"logs/procgen/{args.test_env}", exp_dir))
        helped_logs[log_key][perc] = os.path.join(f"logs/procgen/{args.test_env}", exp_dir, sorted(render_logs)[-1])  # always get last/most updated one

if args.grand_metric or (args.plotting and (PLOT_PERF_VS_PERC in args.plots or PLOT_PERF_VS_PROP in args.plots)):
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
    with open(os.path.join(f"logs/procgen/{args.test_env}", "eval_expert", sorted(temp)[-1], quant_eval_file_name), "r") as f:
        evaluation = f.readlines()
        for line in evaluation:
            if "all rewards" in line.lower():
                expert_performance = eval(line[len("all rewards: "):].strip())
                break
    # Skyline
    # TODO: SKYLINE #

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
pseudo_percentiles = range(50, 151, 10)

if args.grand_metric:
    print("Calculating grand metric")
    table_data = {"metric": [], "AUC": [], "mean reward": []}
    include_metrics = [m for m in helped_logs.keys() if m in args.include and m not in args.exclude]
    for metric in include_metrics:
        print("Doing metric", metric)
        rew_by_perc = []
        adj_rew_by_perc = []
        help_props_by_perc = []
        iterable = percentiles if "svdd" not in metric else pseudo_percentiles
        for it in iterable:
            try:
                with open(os.path.join(helped_logs[metric][it], quant_eval_file_name), "r") as f:
                    evaluation = f.readlines()
                run_lengths = []
                for line in evaluation:
                    if "all rewards" in line.lower():
                        rewards = eval(line[len("all rewards: "):].strip())
                        assert all([reward <= 10 for reward in rewards]), f"wtf {metric} {it}"
                        rew_by_perc.append([reward / norm_factor for reward in rewards])
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
                # print(f"Mean adj. reward for {it} = {np.mean(perc_adjusted_rewards)}")
            except KeyError:
                print(f"Risk values found for {metric}: {helped_logs[metric].keys()}")
                exit()
            except FileNotFoundError:
                print(f"Missing data for {metric} at (pseudo) percentile {it}")

        # (x, y)_i = (average AFHP for percentile i, average performance for percentile i)
        afhp_means, afhp_stds = get_mean_and_std_nested(help_props_by_perc, False)
        rew_means, rew_stds = get_mean_and_std_nested(rew_by_perc, False)
        adj_rew_means, adj_rew_stds = get_mean_and_std_nested(adj_rew_by_perc, False)
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

elif args.plotting:
    print("Making plots")
    run_portions = [round(k, 1) for k in np.arange(0.1, 1.01, 0.1)]
    if PLOT_PERF_VS_PERC in args.plots:
        fig1, axes1 = plt.subplots(2, 4, figsize = (15, 8))
    if PLOT_PROP_VS_PERC in args.plots:
        fig2, axes2 = plt.subplots(2, 4, figsize = (15, 8))
    if PLOT_PERF_VS_PROP in args.plots:
        fig3, axes3 = plt.subplots(2, 4, figsize = (15, 8))
        if args.grouped:
            fig5, axes5 = plt.subplots(1, 1)
    if PLOT_PROP_VS_TIME in args.plots:
        fig4, axes4 = plt.subplots(2, 4, figsize = (15, 8))
    i = 0
    mega_mean_timestep_achieved = []
    rew_by_perc = {m: [] for m in helped_logs}
    adj_rew_by_perc = {m: [] for m in helped_logs}
    help_props_by_perc = {m: [] for m in helped_logs}
    help_asks_by_timestep = {m: {round(k, 1): [] for k in run_portions} for m in helped_logs}
    for metric in helped_logs:
        print("Doing metric", metric)
        iterable = percentiles if "svdd" not in metric else pseudo_percentiles
        for it in iterable:
            try:
                with open(os.path.join(helped_logs[metric][perc], quant_eval_file_name), "r") as f:
                    evaluation = f.readlines()
                run_lengths = []
                for line in evaluation:
                    if PLOT_PERF_VS_PERC in args.plots or PLOT_PERF_VS_PROP in args.plots:
                        if "all rewards" in line.lower():
                            rewards = eval(line[len("all rewards: "):].strip())
                            assert all([reward <= 10 for reward in rewards]), f"wtf {metric} {it}"
                            rew_by_perc[metric].append([reward / norm_factor for reward in rewards])
                        if "all queries" in line.lower():
                            queries = eval(line[len("all queries: "):].strip())
                        if "all switches" in line.lower():
                            switches = eval(line[len("all switches: "):].strip())
                        if "all adjusted rewards" in line.lower():
                            logged_adjusted_rewards = eval(line[len("all adjusted rewards: "):].strip())
                    if PLOT_PERF_VS_PROP in args.plots or PLOT_PROP_VS_PERC in args.plots:
                        if "help times" in line.lower():
                            help_times = eval(evaluation[-1])
                            help_props = []
                            for helps in help_times:
                                help_props.append(sum(helps) / len(helps))
                            help_props_by_perc[metric].append(help_props)
                            run_lengths.append(len(helps))
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
                                    help_asks_by_timestep[metric][round((k + 1) / 10, 1)].append(sum(segment) / len(segment))
                    if "mean timestep achieved" in line.lower():
                        mega_mean_timestep_achieved.append(int(re.search(r"(\d+)", line).group(1)))
                if PLOT_PERF_VS_PERC in args.plots or PLOT_PERF_VS_PROP in args.plots:
                    curr_idx = 0
                    perc_adjusted_rewards = []
                    try:
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
                        adj_rew_by_perc[metric].append(perc_adjusted_rewards)
                    except NameError:
                        adj_rew_by_perc[metric].append(logged_adjusted_rewards)
            except FileNotFoundError:
                print(f"Missing data for {metric} at (pseudo) percentile {it}")
        
        # 1: Plotting performance vs percentile
        if PLOT_PERF_VS_PERC in args.plots:
            ax = axes1[i // 4][i % 4]
            rew_means, rew_stds = get_mean_and_std_nested(rew_by_perc[metric], True)
            adj_rew_means, adj_rew_stds = get_mean_and_std_nested(adj_rew_by_perc[metric], True)
            ax.plot(iterable, rew_means, color = colors[metric], label = f"Reward (mean SD: {round(np.mean(rew_stds), 2)})", linewidth = 2.5)
            # ax.fill_between(percentiles, rew_means - rew_stds, rew_means + rew_stds, color = colors[metric], alpha = 0.3)
            ax.plot(iterable, adj_rew_means, color = colors[metric], linestyle = "dashed", label = f"Adj. Reward (mean SD: {round(np.mean(adj_rew_stds), 2)})", linewidth = 2.5)
            ax.fill_between(iterable, adj_rew_means - adj_rew_stds, adj_rew_means + adj_rew_stds, color = colors[metric], alpha = 0.3)
            ax.plot(iterable, [train_perf_mean] * len(iterable), color = "black", label = f"Train (SD: {round(train_perf_std, 2)})", linewidth = 2.5)
            # ax.fill_between(percentiles, [train_perf_mean - train_perf_std] * len(percentiles), [train_perf_mean + train_perf_std] * len(percentiles), color = "black", alpha = 0.3)
            ax.plot(iterable, [test_perf_mean] * len(iterable), color = "gray", label = f"Test (SD: {round(test_perf_std, 2)})", linewidth = 2.5)
            # ax.fill_between(percentiles, [test_perf_mean - test_perf_std] * len(percentiles), [test_perf_mean + test_perf_std] * len(percentiles), color = "gray", alpha = 0.3)
            ax.plot(iterable, [expert_perf_mean] * len(iterable), color = "tab:brown", label = f"Expert (SD: {round(expert_perf_std, 2)})", linewidth = 2.5)
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
            ax = axes2[i // 4][i % 4]
            help_prop_means, help_prop_stds = get_mean_and_std_nested(help_props_by_perc[metric], True)
            ax.plot(iterable, help_prop_means, color = colors[metric])
            ax.fill_between(iterable, help_prop_means - help_prop_stds, help_prop_means + help_prop_stds, color = colors[metric], alpha = 0.3)
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
            ax = axes3[i // 4][i % 4]
            if args.bucketed:
                # (x, y)_i = (average AFHP for percentile i, average performance for percentile i)
                afhp_means, afhp_stds = get_mean_and_std_nested(help_props_by_perc[metric], True)
                rew_means, rew_stds = get_mean_and_std_nested(rew_by_perc[metric], True)
                adj_rew_means, adj_rew_stds = get_mean_and_std_nested(adj_rew_by_perc[metric], True)
                ax.plot(afhp_means, rew_means, color = colors[metric], label = f"Reward (mean SD: {np.round(np.mean(rew_stds), 2)})")
                # ax.fill_between(afhp_means, rew_means - rew_stds, rew_means + rew_stds, color = colors[metric], alpha = 0.3)
                ax.plot(afhp_means, adj_rew_means, color = colors[metric], linestyle = "dashed", label = f"Adj. Reward (mean SD: {np.round(np.mean(adj_rew_stds), 2)})")
                ax.fill_between(afhp_means, adj_rew_means - adj_rew_stds, adj_rew_means + adj_rew_stds, color = colors[metric], alpha = 0.3)
            else:
                flattened_help_props = np.array([help_prop for help_props in help_props_by_perc[metric] for help_prop in help_props])
                flattened_rews = np.array([rew for rews in rew_by_perc[metric] for rew in rews])
                flattened_adj_rews = np.array([adj_rew for adj_rews in adj_rew_by_perc[metric] for adj_rew in adj_rews])
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
            ax = axes4[i // 4][i % 4]
            asks_mean = np.array([np.mean(asks) for _, asks in help_asks_by_timestep[metric].items()])
            asks_std = np.array([stats.sem(asks) for _, asks in help_asks_by_timestep[metric].items()])
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

    if PLOT_PERF_VS_PROP in args.plots and args.grouped:
        # This one is bucketed by default
        # (x, y)_i = (average AFHP for percentile i, average performance for percentile i)
        for metric in helped_logs:
            afhp_means, afhp_stds = get_mean_and_std_nested(help_props_by_perc[metric], True)
            rew_means, rew_stds = get_mean_and_std_nested(rew_by_perc[metric], True)
            adj_rew_means, adj_rew_stds = get_mean_and_std_nested(adj_rew_by_perc[metric], True)
            axes5.plot(afhp_means, rew_means, color = colors[metric], label = f"Reward (mean SD: {np.round(np.mean(rew_stds), 2)})")
            # ax.plot(afhp_means, adj_rew_means, color = colors[metric], linestyle = "dashed", label = f"Adj. Reward (mean SD: {np.round(np.mean(adj_rew_stds), 2)})")
        fig5.suptitle("Performance vs. AFHP, All Metrics")
        fig5.tight_layout()
        fig5.savefig(f"{args.prefix}_performance_by_afhp_all{args.suffix}.png")
    
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
