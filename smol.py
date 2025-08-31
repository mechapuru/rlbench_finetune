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


# Wrapper class to specify the correct arm action size
class ArmJointVelocity(JointVelocity):
    def action_shape(self, scene: Scene) -> tuple:
        # The arm has 7 joints
        return (7,)


policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
# policy.load_state_dict(torch.load("path/to/your/model.pt"))



print("The input features are",policy.config.input_features)
print("The output features are",policy.config.output_features)

device = "cuda"

obs_config = ObservationConfig()
obs_config.front_camera.image_size = (256, 256)
obs_config.left_shoulder_camera.image_size = (256, 256)
obs_config.right_shoulder_camera.image_size = (256, 256)
obs_config.set_all_high_dim(True)
obs_config.set_all_low_dim(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=ArmJointVelocity(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()

rewards = []
step = 0
done = False

while not done:
    # Prepare observation for the policy running in Pytorch
    # The policy expects a 6D state, but the env provides a 7D state for the arm.
    # We slice the state to 6D, ignoring the last joint.
    state = torch.from_numpy(obs.joint_positions[:6])
    front_image = torch.from_numpy(obs.front_rgb)
    left_shoulder_image = torch.from_numpy(obs.left_shoulder_rgb)
    right_shoulder_image = torch.from_numpy(obs.right_shoulder_rgb)

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [-1,1]
    state = state.to(torch.float32)
    front_image = front_image.to(torch.float32) / 127.5 - 1
    front_image = front_image.permute(2, 0, 1)
    left_shoulder_image = left_shoulder_image.to(torch.float32) / 127.5 - 1
    left_shoulder_image = left_shoulder_image.permute(2, 0, 1)
    right_shoulder_image = right_shoulder_image.to(torch.float32) / 127.5 - 1
    right_shoulder_image = right_shoulder_image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    front_image = front_image.to(device, non_blocking=True)
    left_shoulder_image = left_shoulder_image.to(device, non_blocking=True)
    right_shoulder_image = right_shoulder_image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    front_image = front_image.unsqueeze(0)
    left_shoulder_image = left_shoulder_image.unsqueeze(0)
    right_shoulder_image = right_shoulder_image.unsqueeze(0)

    print("The shapes of the input tensors are:")
    print("State:", state.shape)
    print("Front Image:", front_image.shape)
    print("Left Shoulder Image:", left_shoulder_image.shape)
    print("Right Shoulder Image:", right_shoulder_image.shape)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": front_image,
        "observation.image2": left_shoulder_image,
        "observation.image3": right_shoulder_image,
        "task": descriptions[0],
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    # The policy outputs a 6D action, but the env expects 7D for the arm and 1D for the gripper.
    # We pad the action with two zeros for the 7th arm joint and the gripper.
    numpy_action = action.squeeze(0).to("cpu").numpy()
    numpy_action = np.pad(numpy_action, (0, 2), 'constant')

    # Step through the environment and receive a new observation
    obs, reward, terminate = task.step(numpy_action)
    print(f"{step=} {reward=} {terminate=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)


    # The rollout is considered done when the success state is reached (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminate
    step += 1