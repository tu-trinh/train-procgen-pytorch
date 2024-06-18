import pickle
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--percentile_dir", "-d", type = str, required = True)
parser.add_argument("--display", "-p", action = "store_true")
args = parser.parse_args()
percentile_dir = args.percentile_dir
metrics = {
    "all_entropies": "entropies",
    "all_max_logits": "max_logits",
    "all_max_probs": "max_probs",
    "all_sampled_logits": "sampled_logits",
    "all_sampled_probs": "sampled_probs",
    "entropies_by_action": "entropies_by_action",
    "logits_by_action": "logits_by_action",
    "probs_by_action": "probs_by_action"
}
percentiles = {}

for file in os.listdir(percentile_dir):
    if file != "percentiles.pkl":
        prefix = file.split("_model")[0]
        with open(os.path.join(percentile_dir, file), "rb") as f:
            data = pickle.load(f)
            percs = {}
            if "all" in prefix:
                for p in [1] + list(range(5, 96, 5)) + [99]:
                    percs[p] = np.percentile(data, p)
                if args.display:
                    print("METRIC:", metrics[prefix].upper())
                    print("Mean:", np.mean(data))
                    print("Median:", np.median(data))
                    print(percs)
                    print("\n\n")
            else:
                for action in data:
                    action_percs = {}
                    for p in [1] + list(range(5, 96, 5)) + [99]:
                        action_percs[p] = np.percentile(data[action], p)
                    percs[action] = action_percs
            percentiles[metrics[prefix]] = percs
with open(os.path.join(percentile_dir, "percentiles.pkl"), "wb") as f:
    pickle.dump(percentiles, f)
