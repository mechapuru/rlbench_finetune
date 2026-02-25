import os
import sys
import argparse
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HEADLESS"] = "False"

parser = argparse.ArgumentParser(description="Standalone Extractor Test")
parser.add_argument("--scene", type=int, choices=[0, 1], default=1, 
                    help="0 for grill_task, 1 for kitchen_task (default: kitchen_task)")
args = parser.parse_args()

scenes = [
    "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/grill_task/grill_task.ttt",
    "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/kitchen_task/task_design_proposal_variation_1.ttt"
]

import rlbench_kitchen_env as env_module
env_module.SCENE_FILE = scenes[args.scene]

from rlbench_kitchen_streams import ENV
from vlm_pipeline.dynamic_state_extractor import DynamicStateExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    logging.info(f"Loading scene: {env_module.SCENE_FILE}")
    env = ENV
    
    logging.info("Settling physics...")
    for _ in range(20):
        env.pr.step()
        
    logging.info("Running Extractor...")
    extractor = DynamicStateExtractor(env)
    
    state = extractor.get_scene_state()
    
    print("\n" + "="*50)
    print("EXTRACTED SECENE STATE")
    print("="*50)
    
    # Manually recreate the string building logic from vlm_context_aggregator
    print(f"\n## Robot Status:")
    print(f"- gripper: {state.robot_gripper_state}")
    if state.robot_holding:
        print(f"  holding: {state.robot_holding}")
        
    print(f"\n## Dynamically Discovered Objects:")
    for obj in state.objects:
        status_str = f", STATUS=BLOCKED_BY_{obj.blocked_by}" if obj.blocked_by else ""
        state_str = f"state={obj.state}, " if obj.state else ""
        loc_str = f"location={obj.location}" if obj.location else f"region={obj.region}"
        print(f"- {obj.name}: {state_str}{loc_str}{status_str}")
        
    if state.regions:
        print(f"\n## Valid Target Regions:")
        for r in state.regions:
            print(f"- {r['name']}: {r['description']}")
            
    print("\n" + "="*50)
    print("DEBUG COMPLETE. Exiting.")
    print("="*50)
    
if __name__ == "__main__":
    main()
