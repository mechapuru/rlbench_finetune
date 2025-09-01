
import pickle
from pathlib import Path

# Path to a sample pickle file from your dataset
pickle_path = Path('/scratch/rl_data/reach_target/variation0/episodes/episode0/low_dim_obs.pkl')

print(f"--- Inspecting keys from: {pickle_path} ---")

try:
    with open(pickle_path, 'rb') as f:
        demo = pickle.load(f)

    if not demo:
        print("Pickle file is empty.")
        exit()

    # Get the first observation object from the demonstration list
    obs = demo[0]

    print("\nAvailable keys in a single observation object:")
    # Print all attributes (keys) of the observation object
    for key in vars(obs).keys():
        print(f"- {key}")

except FileNotFoundError:
    print(f"Error: Could not find the file at {pickle_path}")
    print("Please ensure the data generation was successful and the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")

