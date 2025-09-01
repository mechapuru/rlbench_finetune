
import pickle
from pathlib import Path
import numpy as np
# Path to a sample pickle file from your dataset
pickle_path = Path('/home/puruojha/rlbench/data_set/reach_target/variation1/episodes/episode2/low_dim_obs.pkl')

print(f"--- Inspecting keys from: {pickle_path} ---")

try:
    with open(pickle_path, 'rb') as f:
        demo = pickle.load(f)

    if not demo:
        print("Pickle file is empty.")
        exit()

    # Get the first observation object from the demonstration list
    obs = demo[0]

    print("\nAvailable keys and a sample value from a single observation object:")
    # Print all attributes (keys) of the observation object along with their sizes or a sample value
    for key in vars(obs).keys():
        value = getattr(obs, key)
        if isinstance(value, np.ndarray):
            if value.size > 0:
                # For numpy arrays, print the first element or a small slice
                if value.ndim > 1:
                    print(f"- {key} (shape: {value.shape}): {value.flatten()[0] if value.size > 0 else 'Empty'}")
                else:
                    print(f"- {key} (shape: {value.shape}): {value[0] if value.size > 0 else 'Empty'}")
            else:
                print(f"- {key} (shape: {value.shape}): Empty array")
        else:
            print(f"- {key}: {value} (type: {type(value)})")

except FileNotFoundError:
    print(f"Error: Could not find the file at {pickle_path}")
    print("Please ensure the data generation was successful and the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")

