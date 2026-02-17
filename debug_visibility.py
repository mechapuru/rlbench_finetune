import numpy as np
from pyrep.objects.object import Object
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

class VisibilityTracker:
    def __init__(self):
        self.previous_handles = set()

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
        
        if entered or exited:
            print(f"\n[Step {step_num}] Visibility Change Detected:")
            
        if entered:
            print(f"  --> Objects ENTERED view:")
            for h in entered:
                try:
                    obj = Object.get_object(int(h))
                    print(f"      - {obj.get_name()} at {obj.get_position()}")
                except:
                    pass

        if exited:
            print(f"  <-- Objects EXITED view:")
            for h in exited:
                try:
                    obj = Object.get_object(int(h))
                    print(f"      - {obj.get_name()}")
                except:
                    pass
        
        self.previous_handles = current_handles

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
        task = env.get_task(ReachTarget)
        descriptions, obs = task.reset()
        print("Task Reset. Initializing Tracker...")
        
        tracker.update(obs, 0)
        
        # Simulation Loop
        for i in range(1, 20):
            # perform random action to move robot and potentially change visibility
            action = np.random.uniform(-0.5, 0.5, size=(8,)) 
            obs, reward, terminate = task.step(action)
            
            tracker.update(obs, i)
            
            if terminate:
                break

    finally:
        env.shutdown()

if __name__ == "__main__":
    main()
