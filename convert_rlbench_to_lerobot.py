
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rlbench.const import colors

def convert_rlbench_to_lerobot(rlbench_data_dir: Path, lerobot_repo_id: str):
    """
    Converts an RLBench dataset to the LeRobotDataset format.

    Args:
        rlbench_data_dir: The path to the directory containing the RLBench data
                          (e.g., './rlbench_data/reach_target').
        lerobot_repo_id: The name for the new LeRobot dataset (e.g., 'reach_target_lerobot').
    """
    print(f"Converting RLBench data from: {rlbench_data_dir}")
    print(f"Saving LeRobot dataset to: {lerobot_repo_id}")

    # Define the features of your dataset.
    # This should match the data you are extracting from RLBench's observations.
    # We'll use the wrist camera, gripper state, and joint velocities.

    # ...existing code...
    features = {
        "observation.images.left_shoulder_rgb": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_shoulder_rgb": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.overhead_rgb": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.front_rgb": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "joint_positions_0", "joint_positions_1", "joint_positions_2",
                "joint_positions_3", "joint_positions_4", "joint_positions_5",
                "joint_positions_6", "gripper_open"
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "joint_velocities_0", "joint_velocities_1", "joint_velocities_2",
                "joint_velocities_3", "joint_velocities_4", "joint_velocities_5",
                "joint_velocities_6"
            ],
        },
    }
# ...existing code...




    # Create a new LeRobotDataset
    # The root directory will be `HF_LEROBOT_HOME/{lerobot_repo_id}` by default.
    # You can specify a different root with the `root` argument.
    dataset = LeRobotDataset.create(
        repo_id=lerobot_repo_id,
        features=features,
        fps=30, # RLBench default FPS
    )

    variation_dirs = sorted([d for d in rlbench_data_dir.iterdir() if d.is_dir() and d.name.startswith('variation')])

    if not variation_dirs:
        raise FileNotFoundError(f"No 'variation' directories found in {rlbench_data_dir}. Make sure you have generated the data.")

    episode_idx_counter = 0
    for variation_dir in tqdm(variation_dirs, desc="Processing variations"):
        variation_index = int(variation_dir.name.replace('variation', ''))
        color_name = colors[variation_index][0]
        task = f"Reach the {color_name} sphere"

        episode_dirs = sorted([d for d in (variation_dir / 'episodes').iterdir() if d.is_dir()])

        for episode_dir in tqdm(episode_dirs, desc=f"  Processing episodes in {variation_dir.name}", leave=False):
            try:
                with open(episode_dir / "low_dim_obs.pkl", "rb") as f:
                    low_dim_obs = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: low_dim_obs.pkl not found in {episode_dir}. Skipping.")
                continue

            # The task description for this episode. For reach_target, it's always the same.
            task = "Reach the target"

            for step_idx, obs in enumerate(low_dim_obs):
                # Load all 5 RGB images
                image_keys = [
                    "left_shoulder_rgb",
                    "right_shoulder_rgb",
                    "overhead_rgb",
                    "wrist_rgb",
                    "front_rgb"
                ]
                images = {}
                for key in image_keys:
                    img_path = episode_dir / key / f"{step_idx}.png"
                    if not img_path.exists():
                        print(f"Warning: Image not found at {img_path}. Skipping frame.")
                        break
                    images[key] = Image.open(img_path)
                else:
                    # Prepare the frame data only if all images are found
                    frame = {
                        "observation.images.left_shoulder_rgb": images["left_shoulder_rgb"],
                        "observation.images.right_shoulder_rgb": images["right_shoulder_rgb"],
                        "observation.images.overhead_rgb": images["overhead_rgb"],
                        "observation.images.wrist_rgb": images["wrist_rgb"],
                        "observation.images.front_rgb": images["front_rgb"],
                        "observation.state": np.concatenate([obs.joint_positions, [float(obs.gripper_open)]]).astype(np.float32),
                        "action": obs.joint_velocities.astype(np.float32),
                    }
                    dataset.add_frame(frame, task=task)

            dataset.save_episode()
            episode_idx_counter += 1

    print("\nConversion complete!")
    print(f"Total episodes converted: {episode_idx_counter}")
    print(f"Dataset saved at: {dataset.root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert RLBench dataset to LeRobot format.")
    parser.add_argument("--rlbench_data_dir", type=Path, required=True, help="Path to the RLBench data directory (e.g., ./rlbench_data/reach_target).")
    parser.add_argument("--lerobot_repo_id", type=str, required=True, help="The name for the new LeRobot dataset (e.g., reach_target_lerobot).")
    args = parser.parse_args()
    convert_rlbench_to_lerobot(args.rlbench_data_dir, args.lerobot_repo_id)
