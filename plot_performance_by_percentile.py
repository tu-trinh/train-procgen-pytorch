import matplotlib.pyplot as plt
import os


quant_eval_file_name = "AAA_quant_eval_model_200015872.txt"
with open(os.path.join("logs/procgen/coinrun/eval_train_og/RENDER_seed_8888_06-17-2024_18-10-21", quant_eval_file_name), "r") as f:
    evaluation = f.readlines()
    for line in evaluation:
        if "mean reward" in line.lower():
            train_performance = float(line.split(":")[-1].strip())
            break
# any help_test_* one is fine for test performance
with open(os.path.join("logs/procgen/coinrun_aisc/help_test_ent_og/RENDER_seed_8888_06-10-2024_20-50-28", quant_eval_file_name), "r") as f:
    evaluation = f.readlines()
    for line in evaluation:
        if "mean reward" in line.lower():
            test_performance = float(line.split(":")[-1].strip())
            break

colors = {"max prob": "blue", "sampled prob": "green", "max logit": "red", "sampled logit": "purple", "entropy": "orange"}
helped_logs = {"max prob": {}, "sampled prob": {}, "max logit": {}, "sampled logit": {}, "entropy": {}}
log_dir = "logs/procgen/coinrun_aisc"
for exp_dir in os.listdir(log_dir):
    if exp_dir.startswith("receive"):
        perc = int(exp_dir.split("_")[-1])
        if "ent" in exp_dir:
            log_key = "entropy"
        elif "max" in exp_dir:
            log_key = "max prob" if "prob" in exp_dir else "max logit"
        else:
            log_key = "sampled prob" if "prob" in exp_dir else "sampled logit"
        render_logs = os.listdir(os.path.join(log_dir, exp_dir))
        helped_logs[log_key][perc] = os.path.join(log_dir, exp_dir, sorted(render_logs)[-1])

percentiles = range(10, 91, 10)
fig, axes = plt.subplots(2, 3, figsize = (15, 8))
i = 0
for metric in helped_logs:
    ax = axes[i // 3][i % 3]
    rew_by_perc = []
    adj_rew_by_perc = []
    for perc in percentiles:
        with open(os.path.join(helped_logs[metric][perc], quant_eval_file_name), "r") as f:
            evaluation = f.readlines()
        for line in evaluation:
            if "mean reward" in line.lower():
                rew_by_perc.append(float(line.split(":")[-1].strip()))
            if "mean adjusted reward" in line.lower():
                adj_rew_by_perc.append(float(line.split(":")[-1].strip()))
    ax.plot(percentiles, [train_performance] * len(percentiles), color = "black", label = "Train")
    ax.plot(percentiles, [test_performance] * len(percentiles), color = "gray", label = "Test")
    ax.plot(percentiles, rew_by_perc, color = colors[metric], label = "Reward")
    ax.plot(percentiles, adj_rew_by_perc, color = colors[metric], linestyle = "dashed", label = "Adjusted Reward")
    ax.set_title(metric.capitalize())
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Performance")
    ax.set_yticks(range(1, 10))
    ax.legend()
    i += 1
axes[-1][-1].text(0.5, 0.5, "made you look", horizontalalignment = "center", verticalalignment = "center", transform = axes[-1][-1].transAxes, fontsize = 5)
fig.suptitle("Performance by Percentile")
plt.tight_layout()
plt.savefig("receive_help_performance_by_percentile.png")
