#Infering from SmolVLA on Reach Target task

import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch
from rlbench.backend.scene import Scene
import matplotlib.pyplot as plt



# --- Start of modifications ---

# 1. Load normalization stats from file
print("Loading normalization stats from normalization_stats.pt...")
dataset_stats = torch.load("normalization_stats.pt")
print("Normalization stats loaded.")

# 2. Load local model
# Path to your trained model checkpoint.
model_path = "/home/paddy/rrc/RLBench/RLBench/outputs/train/my_smolvla/checkpoints/002000/pretrained_model"
policy = SmolVLAPolicy.from_pretrained(model_path, dataset_stats=dataset_stats)

# --- End of modifications ---

# Wrapper class to specify the correct arm action size
class ArmJointVelocity(JointVelocity):
    def action_shape(self, scene: Scene) -> tuple:
        # The arm has 7 joints
        return (7,)



print("The input features are",policy.config.input_features)
print("The output features are",policy.config.output_features)

device = "cuda"
policy.to(device)


# --- Start of modifications ---

# 3. Update observation config to match the trained model
obs_config = ObservationConfig()
# Set image size to 128x128 as expected by the trained model
obs_config.front_camera.image_size = (128, 128)
obs_config.left_shoulder_camera.image_size = (128, 128)
obs_config.right_shoulder_camera.image_size = (128, 128)
obs_config.overhead_camera.image_size = (128, 128)
obs_config.wrist_camera.image_size = (128, 128)

obs_config.set_all_high_dim(True)
obs_config.set_all_low_dim(True)

# --- End of modifications ---


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=ArmJointVelocity(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False,
    arm_max_velocity=1.0,
    arm_max_acceleration=4.0,
)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()

# The task returns multiple descriptions. We select descriptions[2], which is
# "reach the {color_name} sphere", and capitalize it to perfectly match
# the "Reach the {color_name} sphere" format from the conversion script.
instruction = descriptions[2].capitalize()

action_history = []
rewards = []
step = 0
done = False

while not done and step < 200:
    # --- Start of modifications ---

    # 4. Prepare observation for the policy
    # The trained model expects an 8D state: 7 joint positions + 1 gripper open state
    state = torch.from_numpy(np.concatenate([obs.joint_positions, [float(obs.gripper_open)]]))

    # Get all 5 images from the environment
    front_image = torch.from_numpy(obs.front_rgb)
    left_shoulder_image = torch.from_numpy(obs.left_shoulder_rgb)
    right_shoulder_image = torch.from_numpy(obs.right_shoulder_rgb)
    overhead_image = torch.from_numpy(obs.overhead_rgb)
    wrist_image = torch.from_numpy(obs.wrist_rgb)


    # Convert to float32 in [0, 1] range and permute channels to (C, H, W)
    state = state.to(torch.float32)
    front_image = front_image.to(torch.float32) / 255.0
    front_image = front_image.permute(2, 0, 1)
    left_shoulder_image = left_shoulder_image.to(torch.float32) / 255.0
    left_shoulder_image = left_shoulder_image.permute(2, 0, 1)
    right_shoulder_image = right_shoulder_image.to(torch.float32) / 255.0
    right_shoulder_image = right_shoulder_image.permute(2, 0, 1)
    overhead_image = overhead_image.to(torch.float32) / 255.0
    overhead_image = overhead_image.permute(2, 0, 1)
    wrist_image = wrist_image.to(torch.float32) / 255.0
    wrist_image = wrist_image.permute(2, 0, 1)


    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    front_image = front_image.to(device, non_blocking=True)
    left_shoulder_image = left_shoulder_image.to(device, non_blocking=True)
    right_shoulder_image = right_shoulder_image.to(device, non_blocking=True)
    overhead_image = overhead_image.to(device, non_blocking=True)
    wrist_image = wrist_image.to(device, non_blocking=True)


    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    front_image = front_image.unsqueeze(0)
    left_shoulder_image = left_shoulder_image.unsqueeze(0)
    right_shoulder_image = right_shoulder_image.unsqueeze(0)
    overhead_image = overhead_image.unsqueeze(0)
    wrist_image = wrist_image.unsqueeze(0)


    # Create the policy input dictionary with keys matching the model's config
    observation = {
        "observation.state": state,
        "observation.images.front_rgb": front_image,
        "observation.images.left_shoulder_rgb": left_shoulder_image,
        "observation.images.right_shoulder_rgb": right_shoulder_image,
        "observation.images.overhead_rgb": overhead_image,
        "observation.images.wrist_rgb": wrist_image,
        "task": instruction,
    }

    # --- End of modifications ---

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # --- Start of modifications ---

    # 5. Prepare the action for the environment
    # The policy outputs a 7D action, but the env expects 8D (7 for arm and 1 for gripper).
    # We pad the action with one zero for the gripper action.
    numpy_action = action.squeeze(0).to("cpu").numpy()
    action_history.append(numpy_action)
    numpy_action = np.pad(numpy_action, ((0, 0), (0, 1)), 'constant') if numpy_action.ndim == 2 else np.pad(numpy_action, (0, 1), 'constant')


    # --- End of modifications ---

    # Step through the environment and receive a new observation
    obs, reward, terminate = task.step(numpy_action)
    print(f"{step=} {reward=} {terminate=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)


    # The rollout is considered done when the success state is reached (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminate
    step += 1

print("Simulation finished. Shutting down environment.")
env.shutdown()

# Plotting the action values
action_history = np.array(action_history)
timesteps = np.arange(action_history.shape[0])

fig, axs = plt.subplots(7, 1, figsize=(10, 20), sharex=True)
fig.suptitle('Joint Velocity Actions over Time')

for i in range(7):
    axs[i].plot(timesteps, action_history[:, i])
    axs[i].set_ylabel(f'Joint {i+1} Velocity')

axs[-1].set_xlabel('Time Step')
plt.savefig('action_velocities.png')
print("Saved action velocity plots to action_velocities.png")