import numpy as np
import argparse
import importlib
from pyrep.objects.object import Object
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

class VisibilityTracker:
    def __init__(self):
        self.previous_handles = set()
        self.history = [] # storing (step, event_type, object_name)

    def get_visible_handles(self, obs):
        masks = [
            obs.left_shoulder_mask,
            obs.right_shoulder_mask,
            obs.overhead_mask,
            obs.wrist_mask,
            obs.front_mask
        ]
        all_handles = set()
        for mask in masks:
            if mask is not None:
                all_handles.update(np.unique(mask))
        if 0 in all_handles:
            all_handles.remove(0)
        return all_handles

    def update(self, obs, step_num):
        current_handles = self.get_visible_handles(obs)
        
        entered = current_handles - self.previous_handles
        exited = self.previous_handles - current_handles
        
        for h in entered:
            try:
                obj = Object.get_object(int(h))
                name = obj.get_name()
                print(f"[Step {step_num}] ENTERED: {name}")
                self.history.append((step_num, "ENTERED", name))
            except:
                pass

        for h in exited:
            try:
                obj = Object.get_object(int(h))
                name = obj.get_name()
                print(f"[Step {step_num}] EXITED:  {name}")
                self.history.append((step_num, "EXITED", name))
            except:
                pass
        
        self.previous_handles = current_handles

    def print_summary(self):
        print("\n\n=== Visibility Event Summary ===")
        for step, event, name in self.history:
            print(f"Step {step}: {name} {event}")

def get_task_class(task_name):
    try:
        # Convert CamelCase to snake_case for module name if needed, 
        # but typically RLBench works with explicit names.
        # Here we assume user passes 'reach_target' or 'put_groceries_in_cupboard'
        # OR 'ReachTarget' and we convert it.
        
        # Simple heuristic: RLBench task files are snake_case.
        # Task classes are CamelCase.
        
        import re
        # Convert CamelCase to snake_case
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', task_name).lower()
        
        class_name = task_name # User typically provides ClassName e.g. ReachTarget
        
        mod = importlib.import_module(f"rlbench.tasks.{module_name}")
        return getattr(mod, class_name)
    except Exception as e:
        print(f"Error loading task '{task_name}': {e}")
        print("Ensure you provide the exact class name (e.g. ReachTarget, PutGroceriesInCupboard)")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Track object visibility in RLBench tasks.")
    parser.add_argument('--task', type=str, required=True, help="Name of the task class (e.g., ReachTarget, PutGroceriesInCupboard)")
    parser.add_argument('--episodes', type=int, default=1, help="Number of episodes to run (default: 1)")
    args = parser.parse_args()

    task_class_name = args.task
    TaskClass = get_task_class(task_class_name)

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
        
        for ep in range(args.episodes):
            print(f"\n--- Episode {ep+1}/{args.episodes} ---")
            tracker = VisibilityTracker()
            
            print("Resetting task...")
            descriptions, obs = task.reset()
            tracker.update(obs, 0) # Initial state

            print("Getting live demo...")
            try:
                # Provide dummy env to match wrapper pattern in dynamic_state_extractor
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TAMP-PDDL'))
                from vlm_pipeline.background_extractor import BackgroundStateExtractor
                
                class DummyEnv:
                    def __init__(self, rlbench_env):
                        self.rlbench_env = rlbench_env
                    def get_object(self, name):
                        try:
                            from pyrep.objects.shape import Shape
                            return Shape(name)
                        except:
                            return None
                            
                bg_extractor = BackgroundStateExtractor(DummyEnv(env), task_class_name)
                bg_extractor.start()
                
                demo = task.get_demos(amount=1, live_demos=True)[0]
                print(f"Demo generated with {len(demo)} steps.")
                
                for i, demo_obs in enumerate(demo):
                    tracker.update(demo_obs, i + 1)
                    
                bg_extractor.stop()
                
                # Print final extracted state seen by the agent
                final_state = bg_extractor.get_latest_state()
                if final_state:
                    print("\n\n=== Final Semantic State Handed to Planner ===")
                    print(final_state)
                    
                tracker.print_summary()
            except Exception as e:
                print(f"\nFailed to get demo: {e}")

    finally:
        env.shutdown()

if __name__ == "__main__":
    main()
