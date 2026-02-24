import os
import sys

# Configure Qt for GUI BEFORE imports
os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
candidate_dirs = [
    os.path.join(coppelia_root, "platforms"),
    os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
    os.path.join(coppelia_root, "qt", "plugins", "platforms"),
]
for candidate in candidate_dirs:
    if candidate and os.path.isdir(candidate):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
        break

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vlm_pipeline.dynamic_state_extractor as ds

import rlbench_kitchen_env as env_module

def test_scene(scene_file):
    print("=" * 60)
    print(f"Testing Extract on scene: {scene_file}")
    
    # Overwrite the SCENE_FILE before importing/creating instance
    env_module.SCENE_FILE = scene_file
    
    from rlbench_kitchen_streams import RLBenchKitchenEnv
    env = RLBenchKitchenEnv(headless=False)
    
    try:
        # Settle physics
        for _ in range(20):
            env.pr.step()
            
        extractor = ds.DynamicStateExtractor(env)
        bundle = extractor.create_prompt_bundle("Dummy Goal")
        print("\n=== EXTRACTED STATE TEXT ===")
        print(bundle.state_text)
        print("============================\n")
    finally:
        print("\nNote: Skipping explicit PyRep shutdown to avoid Qt thread crashes.")
        
        # Keep window open if user wants to see it? Actually python will exit naturally.
        import time
        time.sleep(1) # Ensure release before next PyRep instance

if __name__ == "__main__":
    scenes = [
        "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/grill_task/grill_task.ttt",
        "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/kitchen_task/task_design_proposal_variation_1.ttt"
    ]
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, choices=[0, 1], default=0, help="0 for grill_task, 1 for kitchen_task")
    args = parser.parse_args()
    
    test_scene(scenes[args.scene])
