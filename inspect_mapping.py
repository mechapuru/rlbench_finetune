
import pickle
import numpy as np

# Path to a sample pickle file from your dataset
pickle_path = '/scratch/rl_data/reach_target/variation0/episodes/episode0/low_dim_obs.pkl'

print(f"--- Inspecting data from: {pickle_path} ---")

try:
    with open(pickle_path, 'rb') as f:
        demo = pickle.load(f)

    if not demo:
        print("Pickle file is empty.")
        exit()

    # Get the first observation object from the demonstration list
    obs = demo[0]

    print("\n--- Attributes of a sample observation object ('obs') ---")
    # Print all attributes of the observation object for clarity
    for attr, value in vars(obs).items():
        # Truncate long arrays for readability
        if isinstance(value, np.ndarray) and value.size > 10:
            print(f"obs.{attr}: {value[:5]}... (shape: {value.shape}, dtype: {value.dtype})")
        else:
            print(f"obs.{attr}: {value}")


    print("\n--- Verifying the mapping to LeRobot format ---")

    # This is the exact logic from the conversion script
    observation_state = np.concatenate([obs.joint_positions, [float(obs.gripper_open)]])
    action = obs.misc.get('joint_position_action', np.zeros(8))

    print("\n1. `observation.state` is created from:")
    print(f"   - obs.joint_positions: {obs.joint_positions}")
    print(f"   - obs.gripper_open: {float(obs.gripper_open)}")
    print(f"   --> Resulting array for 'observation.state': {observation_state}")
    print(f"   --> Resulting shape: {observation_state.shape}")


    print("\n2. `action` is created from:")
    print(f"   - obs.misc.get('joint_position_action', np.zeros(8)): {action}")
    print(f"   --> Resulting array for 'action': {action}")
    print(f"   --> Resulting shape: {action.shape}")

    print("\n(Note: 'observation.images.wrist_rgb' is loaded separately from the corresponding image file.)")


except FileNotFoundError:
    print(f"Error: Could not find the file at {pickle_path}")
    print("Please ensure the data generation was successful and the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")
