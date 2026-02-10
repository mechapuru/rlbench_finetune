from rlbench_kitchen_env import RLBenchKitchenEnv
import numpy as np

class RLBenchKitchenEnvConstrained(RLBenchKitchenEnv):
    def compute_pick_trajectory(self, obj, pose):
        """
        Override to disable box collision when picking the mug inside it.
        This mimics 'ghost mode' for the box to allow the gripper to enter without
        getting stuck on the walls during planning.
        
        NOTE: Lid collision is now handled in the base class - only disabled if lid is OPEN.
        """
        # Check if we are picking the mug inside the box
        is_mug_inside_box = (obj.get_name() == 'mug4')
        
        box_obj = None
        box_collidable_state = True
        
        if is_mug_inside_box:
            # First check if lid is open - if not, don't even try
            lid_obj = self.get_object('box_lid')
            box_base = self.get_object('box_base')
            if lid_obj and box_base:
                lid_pos = lid_obj.get_position()
                box_pos = box_base.get_position()
                # Lid slides in X direction
                lid_offset = abs(lid_pos[0] - box_pos[0])
                LID_OPEN_THRESHOLD = 0.10
                print(f"DEBUG [Constrained]: Lid check - X_offset={lid_offset:.3f}")
                if lid_offset < LID_OPEN_THRESHOLD:
                    print(f"DEBUG: Cannot pick mug_inside_box - lid is CLOSED (X_offset={lid_offset:.3f})")
                    raise RuntimeError("Lid is closed, cannot pick object inside box")
            
            print("DEBUG: Ghost Mode Activated for Box Base during Pick Planning")
            box_obj = self.get_object('box_base')
            if box_obj:
                box_collidable_state = box_obj.is_collidable()
                box_obj.set_collidable(False)
            else:
                print("DEBUG: box_base not found for Ghost Mode")
                
        try:
            # Call the original method
            return super().compute_pick_trajectory(obj, pose)
        finally:
            # Restore box state
            if box_obj:
                box_obj.set_collidable(box_collidable_state)
                print("DEBUG: Ghost Mode Deactivated for Box Base")

    def compute_slide_lid_trajectory(self, obj, grasp_quat, initial_conf=None):
        """
        Override to enforce strict sliding distance constraint (0.25m).
        """
        if initial_conf is None:
            initial_conf = self.get_robot_conf()

        # 1. Get current gripper pose (World Frame)
        old_conf = self.get_robot_conf()
        self.set_robot_conf(initial_conf)
        
        try:
            curr_tip = self.robot.arm.get_tip()
        except AttributeError:
            curr_tip = self.robot.get_tip()

        start_pos = np.array(curr_tip.get_position())
        start_quat = np.array(curr_tip.get_quaternion()) # x,y,z,w
        self.set_robot_conf(old_conf) # Restore

        # CONSTRAINT: Slide exactly 0.25m along X
        slide_dist = 0.18
        slide_vec = np.array([slide_dist, 0, 0])
        target_pos = start_pos + slide_vec
        
        print(f"DEBUG: Constrained Slide Lid by {slide_dist:.3f}m")
        
        self.set_robot_conf(initial_conf)
        try:
            # steps=50 for smoother motion
            path = self.robot.get_linear_path(position=target_pos, quaternion=start_quat, steps=50, ignore_collisions=True)
            
            if path:
                traj_configs = path._path_points.reshape(-1, 7).tolist()
                
                # Return to Center Logic (same as base)
                # We want to return to a point slightly offset from start to avoid collision on release?
                # Actually, let's just return to start.
                # But base implementation did: x_shift = (lid_len / 2.0) - 0.03
                # Let's just reverse the path for return.
                
                traj_return = traj_configs[::-1]
                return traj_configs[0], traj_configs[-1], traj_configs, traj_return
            else:
                print("DEBUG: Failed to compute linear path for slide.")
                return initial_conf, initial_conf, [], []

        except Exception as e:
            print(f"DEBUG: Exception in slide computation: {e}")
            return initial_conf, initial_conf, [], []

    def compute_retreat_to_home(self, q_start):
        """
        Robust retreat to home. Tries direct, then various lift heights.
        """
        q_home = list(self.home_conf)
        
        # 1. Try simple interpolation first
        traj = self._interpolate_joint_path(q_start, q_home, steps=50, check_collisions=True)
        if traj:
            return q_home, traj
            
        # 2. Try Lifting strategies
        self.set_robot_conf(q_start)
        curr_pos = self.robot.get_position()
        curr_quat = self.robot.get_quaternion()
        
        for lift_height in [0.1, 0.2, 0.3]:
            # print(f"DEBUG: Trying retreat with lift {lift_height}m")
            lift_pos = [curr_pos[0], curr_pos[1], curr_pos[2] + lift_height]
            try:
                path_lift = self.robot.get_linear_path(position=lift_pos, quaternion=curr_quat, steps=30, ignore_collisions=False)
                if path_lift:
                    q_lift_end = path_lift._path_points[-1].tolist() # Use last point
                    # Interpolate from lift end to home
                    traj_home = self._interpolate_joint_path(q_lift_end, q_home, steps=50, check_collisions=True)
                    if traj_home:
                        traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                        # Ensure continuity
                        traj_lift[0] = list(q_start)
                        return q_home, traj_lift + traj_home
            except:
                continue
                
        print("DEBUG: All retreat strategies failed.")
        return None, None

    def set_box_collision(self, enabled):
        """Helper to toggle box collision state during execution."""
        box = self.get_object('box_base')
        if box:
            box.set_collidable(enabled)
            # Also toggle boundary if it exists
            if hasattr(self, 'box_boundary'):
                self.box_boundary.set_collidable(enabled)
            print(f"  [Env] Box Collision set to {enabled}")

