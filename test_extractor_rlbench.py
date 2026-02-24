import argparse
import importlib
import sys
import os
import re

from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

# Set path for vlm_pipeline imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TAMP-PDDL'))
from vlm_pipeline.dynamic_state_extractor import DynamicStateExtractor

class DummyEnvWrapper:
    def __init__(self, rlbench_env):
        self.rlbench_env = rlbench_env
    
    def get_object(self, name):
        try:
            return Shape(name)
        except:
            return None

def get_task_class(task_name):
    module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', task_name).lower()
    mod = importlib.import_module(f"rlbench.tasks.{module_name}")
    return getattr(mod, task_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help="Task class name, e.g., LongHorizonGroceryTask")
    args = parser.parse_args()

    TaskClass = get_task_class(args.task)

    obs_config = ObservationConfig()
    obs_config.set_all(True) 
    
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete()
    )
    
    env = Environment(
        action_mode,
        str(obs_config.dataset_path) if hasattr(obs_config, 'dataset_path') else '',
        obs_config,
        headless=True
    )
    
    env.launch()
    
    try:
        task = env.get_task(TaskClass)
        
        print(f"\n--- Testing {args.task} ---")
        print("Resetting task...")
        descriptions, obs = task.reset()
        
        # Ensure physics settled
        for _ in range(10):
            env._scene.step()
        
        dummy_env = DummyEnvWrapper(env)
        extractor = DynamicStateExtractor(dummy_env)
        
        print("\n=== EXTRACTED STATE AT STEP 0 ===")
        bundle = extractor.create_prompt_bundle(descriptions[0])
        print(bundle.state_text)
        print("=================================\n")
        
    except Exception as e:
        print(f"Error: {e}")

    finally:
        env.shutdown()

if __name__ == "__main__":
    main()
