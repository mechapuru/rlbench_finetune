import numpy as np
from pyrep.objects.object import Object
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutGroceriesInCupboard
import time

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

def main():
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
    tracker = VisibilityTracker()
    
    try:
        task = env.get_task(PutGroceriesInCupboard)
        print("Resetting task...")
        descriptions, obs = task.reset()
        
        # Track initial state
        tracker.update(obs, 0)

        # Get a demo to perform realistic actions
        print("Getting a live demo to simulate task execution...")
        # We need a demo to actually move objects around realistically
        # This might take a moment as it plans a path
        demo = task.get_demos(amount=1, live_demos=True)[0]
        
        print(f"Demo generated with {len(demo)} steps. Replaying for tracking...")
        
        # Iterate through demo observations
        # Note: In a real 'run', you'd be calling step(), but here we have the full trajectory
        for i, obs in enumerate(demo):
            tracker.update(obs, i + 1)
            
        tracker.print_summary()

    finally:
        env.shutdown()

if __name__ == "__main__":
    main()
