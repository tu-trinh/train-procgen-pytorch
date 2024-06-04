import os
import re
import matplotlib.pyplot as plt

timesteps = []
mean_rewards = []
mean_timesteps = []
proportions = []

base_path = "logs/procgen/coinrun/find_minimum_timesteps/"

for root, dirs, files in os.walk(base_path):
    for file in files:
        match = re.search(r"model_(\d+)\.txt", file)
        if match:
            timestep = int(match.group(1))
            timesteps.append(timestep)
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
                mean_reward = float(re.search(r"Mean reward: ([\d.]+)", lines[0]).group(1))
                mean_timestep = int(re.search(r"Mean timestep achieved: (\d+)", lines[1]).group(1))
                proportion = float(re.search(r"Proportion of times achieved: ([\d.]+)", lines[2]).group(1))
                
                mean_rewards.append(mean_reward)
                mean_timesteps.append(mean_timestep)
                proportions.append(proportion)

sorted_indices = sorted(range(len(timesteps)), key = lambda i: timesteps[i])
timesteps = [timesteps[i] for i in sorted_indices]
mean_rewards = [mean_rewards[i] for i in sorted_indices]
mean_timesteps = [mean_timesteps[i] for i in sorted_indices]
proportions = [proportions[i] for i in sorted_indices]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(timesteps, mean_rewards, marker="o")
axs[0].set_title("Mean Reward vs. Training Timesteps")
axs[0].set_xlabel("Timestep")
axs[0].set_ylabel("Mean Reward")

axs[1].plot(timesteps, mean_timesteps, marker="o")
axs[1].set_title("Mean Achievement Timestep vs. Training Timesteps")
axs[1].set_xlabel("Timestep")
axs[1].set_ylabel("Mean Achievement Timestep")

axs[2].plot(timesteps, proportions, marker="o")
axs[2].set_title("Success Proportion vs. Training Timesteps")
axs[2].set_xlabel("Timestep")
axs[2].set_ylabel("Proportion of Successes")

plt.tight_layout()
plt.savefig("training_performance_by_timestep.png")
# plt.show()

# Seems like 180M timesteps is ok
print("All timesteps")
print(timesteps, "\n")
mean_reward_threshold = 6.7
print(f"Timesteps where mean reward was at least {mean_reward_threshold}")
print([timesteps[i] for i in [j for j in range(len(mean_rewards)) if mean_rewards[j] >= mean_reward_threshold]], "\n")
proportion_threshold = 0.88
print(f"Timesteps where proportion of achievement was at least {proportion_threshold * 100}%")
print([timesteps[i] for i in [j for j in range(len(proportions)) if proportions[j] >= proportion_threshold]], "\n")
