import matplotlib.pyplot as plt
import numpy as np


help_times = {"max prob": [], "sampled prob": [], "max logit": [], "sampled logit": [], "entropy": []}
logs = {
    "max prob": "logs/procgen/coinrun_aisc/help_test_max_probs_og/RENDER_seed_8888_06-10-2024_17-18-05/AAA_quant_eval_model_200015872.txt",
    "sampled prob": "logs/procgen/coinrun_aisc/help_test_sample_probs_og/RENDER_seed_8888_06-10-2024_17-46-10/AAA_quant_eval_model_200015872.txt",
    "max logit": "logs/procgen/coinrun_aisc/help_test_max_logit_og/RENDER_seed_8888_06-10-2024_18-13-44/AAA_quant_eval_model_200015872.txt",
    "sampled logit": "logs/procgen/coinrun_aisc/help_test_sample_logit_og/RENDER_seed_8888_06-10-2024_18-41-18/AAA_quant_eval_model_200015872.txt",
    "entropy": "logs/procgen/coinrun_aisc/help_test_ent_og/RENDER_seed_8888_06-10-2024_20-50-28/AAA_quant_eval_model_200015872.txt"
}
for metric in help_times:
    with open(logs[metric], "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if "Help times" in lines[i]:
            help_times[metric] = eval(lines[i + 1])
            break
colors = {"max prob": "blue", "sampled prob": "green", "max logit": "red", "sampled logit": "purple", "entropy": "orange"}

max_length = max(max(len(lst) for lst in val) for val in help_times.values() if len(val) > 0)

plt.figure(figsize = (10, 6))
for metric in help_times:
    if len(help_times[metric]) > 0:
        # for lst in dataset:
        #     plt.plot(lst, color = color, alpha = 0.3)
        interpolated = np.array([np.interp(np.arange(max_length), np.arange(len(lst)), lst) for lst in help_times[metric]])
        mean_interpolated = np.mean(interpolated, axis = 0)
        plt.plot(mean_interpolated, color = colors[metric], label = metric)
plt.xlabel("Timestep")
plt.ylabel("Asked for Help")
plt.title("Using Percentiles")
plt.legend()
plt.tight_layout()
plt.savefig("ask_for_help_times.png")

fig, axes = plt.subplots(1, len(help_times), figsize = (4 * len(help_times), 4))
for i, metric in enumerate(help_times):
    ax = axes[i]
    if len(help_times[metric]) > 0:
        sums = [sum(lst) for lst in help_times[metric]]
        ax.hist(sums, bins = range(min(sums), max(sums) + 2), color = colors[metric])
    else:
        ax.text(0.5, 0.5, "N/A", horizontalalignment = "center", verticalalignment = "center", transform = ax.transAxes)
    ax.set_title(metric)
    ax.set_xlabel("Times Asked For Help")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("ask_for_help_times_histogram.png")
