import pickle
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--percentile_dir", "-d", type = str, required = True)
parser.add_argument("--second_percentile_dir", "-s", type = str, default = None)
parser.add_argument("--display", "-p", action = "store_true")
args = parser.parse_args()
percentile_dir = args.percentile_dir
second_percentile_dir = args.second_percentile_dir
metrics = {
    "all_max_probs": "max_probs",
    "all_sampled_probs": "sampled_probs",
    "all_max_logits": "max_logits",
    "all_sampled_logits": "sampled_logits",
    "all_entropies": "entropies",
    "entropies_by_action": "entropies_by_action",
    "logits_by_action": "logits_by_action",
    "probs_by_action": "probs_by_action"
}
percentiles = {}

metrics_as_list = list(metrics.keys())
if args.display:
    fig, axes = plt.subplots(2, 3, figsize = (15, 8))
    colors = ["blue", "green", "red", "purple", "orange"]
percentile_numbers = [1] + list(range(5, 96, 5)) + [99]
for i, curr_dir in enumerate([percentile_dir, second_percentile_dir]):
    if curr_dir is not None:
        for file in os.listdir(curr_dir):
            if file != "percentiles.pkl" and ".pkl" in file:
                prefix = file.split("_model")[0]
                with open(os.path.join(curr_dir, file), "rb") as f:
                    data = pickle.load(f)
                    percs = {}
                    if "all" in prefix:
                        for p in percentile_numbers:
                            percs[p] = np.percentile(data, p)
                        percentiles[metrics[prefix]] = percs
                        if args.display:
                            print("METRIC:", metrics[prefix].upper())
                            print("Mean:", np.mean(data))
                            print("Median:", np.median(data))
                            print(percs)
                            print("\n\n")
                    # else:
                    #     for action in data:
                    #         action_percs = {}
                    #         for p in [1] + list(range(5, 96, 5)) + [99]:
                    #             action_percs[p] = np.percentile(data[action], p)
                    #         percs[action] = action_percs
                        if args.display:
                            j = metrics_as_list.index(prefix)
                            ax = axes[j // 3][j % 3]
                            ax.plot(percentile_numbers, list(percs.values()), color = colors[j], linestyle = "solid" if i == 0 else "dashed", label = "train" if i == 0 else "test")
                            ax.legend()
                            ax.set_title(prefix.replace("all_", ""))
                            ax.set_xlabel("percentile")
                            ax.set_ylabel("value")
if args.display:
    axes[-1, -1].axis("off")
    plt.suptitle("Percentile Values In Train and Test")
    fig.subplots_adjust(hspace = 0.3, wspace = 0.2)
    plt.savefig("percentiles_in_train_and_test.png")
with open(os.path.join(percentile_dir, "percentiles.pkl"), "wb") as f:
    pickle.dump(percentiles, f)
