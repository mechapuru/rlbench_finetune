import os
import pickle
import argparse
import numpy as np
from PIL import Image

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.utils import name_to_task_class
from rlbench.backend.const import *

def save_demo(demo, episode_path):
    """Saves a demonstration to the specified path."""

    # Camera names
    cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']
    camera_folders = {
        'left_shoulder': (LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_MASK_FOLDER),
        'right_shoulder': (RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER),
        'overhead': (OVERHEAD_RGB_FOLDER, OVERHEAD_DEPTH_FOLDER, OVERHEAD_MASK_FOLDER),
        'wrist': (WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER),
        'front': (FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER),
    }

    # Create directories for all cameras
    for cam_name, folders in camera_folders.items():
        rgb_path = os.path.join(episode_path, folders[0])
        depth_path = os.path.join(episode_path, folders[1])
        mask_path = os.path.join(episode_path, folders[2])
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

    # Save images
    for i, obs in enumerate(demo):
        for cam_name, folders in camera_folders.items():
            rgb_path = os.path.join(episode_path, folders[0])
            depth_path = os.path.join(episode_path, folders[1])
            mask_path = os.path.join(episode_path, folders[2])

            rgb_image = getattr(obs, f'{cam_name}_rgb')
            depth_image = getattr(obs, f'{cam_name}_depth')
            mask_image = getattr(obs, f'{cam_name}_mask')

            Image.fromarray(rgb_image).save(os.path.join(rgb_path, f"{i}.png"))
            depth_im = Image.fromarray((depth_image * 255).astype(np.uint8))
            depth_im.save(os.path.join(depth_path, f"{i}.png"))
            mask_im = Image.fromarray(mask_image.astype(np.uint8))
            mask_im.save(os.path.join(mask_path, f"{i}.png"))

    # Save low-dimensional data
    # We need to strip the images from the observation before pickling
    for obs in demo:
        for cam_name in cameras:
            setattr(obs, f'{cam_name}_rgb', None)
            setattr(obs, f'{cam_name}_depth', None)
            setattr(obs, f'{cam_name}_mask', None)

    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

def main(args):
    # Initialize RLBench environment
    obs_config = ObservationConfig()
    
    # Set image size
    image_size = [640, 480] # Width, Height
    obs_config.left_shoulder_camera.image_size = image_size
    obs_config.right_shoulder_camera.image_size = image_size
    obs_config.overhead_camera.image_size = image_size
    obs_config.wrist_camera.image_size = image_size
    obs_config.front_camera.image_size = image_size

    obs_config.set_all(True)

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True)
    env.launch()

    task_class = name_to_task_class(args.task_name)
    task = env.get_task(task_class)

    # Create dataset directory
    dataset_path = os.path.join(args.dataset_root, args.task_name)
    variation_path = os.path.join(dataset_path, f"variation{args.variation_number}")
    episodes_path = os.path.join(variation_path, "episodes")
    os.makedirs(episodes_path, exist_ok=True)

    print(f"Recording {args.num_episodes} episodes for task {args.task_name}...")

    for i in range(args.num_episodes):
        print(f"Recording episode {i}...")
        # Get a live demo
        demo = task.get_demos(1, live_demos=True)[0]

        # Save the demo
        episode_path = os.path.join(episodes_path, f"episode{i}")
        save_demo(demo, episode_path)

    print("Done.")
    env.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record expert trajectories in RLBench.')
    parser.add_argument('--dataset_root', type=str, default='./rlbench_data', help='Path to the root of the dataset.')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task to record.')
    parser.add_argument('--variation_number', type=int, default=0, help='Variation number of the task.')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to record.')
    args = parser.parse_args()
    main(args)