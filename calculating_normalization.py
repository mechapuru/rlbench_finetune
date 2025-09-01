import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# --- Start of modifications ---

# 1. Load dataset and calculate normalization stats
print("Loading dataset to calculate normalization stats...")
# The dataset repo_id from your training configuration.
dataset = LeRobotDataset("language_instructed")

states = []
actions = []
# LeRobotDataset iterates over frames. We collect all states and actions.
for i in tqdm(range(len(dataset)), desc="Iterating through dataset to get normalization stats"):
    data = dataset[i]
    states.append(data["observation.state"])
    actions.append(data["action"])

states_tensor = torch.stack(states)
actions_tensor = torch.stack(actions)

state_mean = states_tensor.mean(dim=0)
state_std = states_tensor.std(dim=0)
# Add a small epsilon to std to avoid division by zero
state_std = torch.max(state_std, torch.tensor(1e-6))


action_mean = actions_tensor.mean(dim=0)
action_std = actions_tensor.std(dim=0)
action_std = torch.max(action_std, torch.tensor(1e-6))


dataset_stats = {
    "observation.state": {"mean": state_mean, "std": state_std},
    "action": {"mean": action_mean, "std": action_std},
}
print("Normalization stats calculated.")

#save normalization

torch.save(dataset_stats, "normalization_stats.pt")
print("Normalization stats saved to normalization_stats.pt")
