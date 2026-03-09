import os

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
        Constrained slide with stronger opening:
        try a larger distance first, then back off if planning fails.
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

        # Lid size-aware opening distance with explicit override:
        # export BOX_LID_SLIDE_DIST=0.27
        min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
        lid_len = max(max_x - min_x, max_y - min_y, 0.12)

        override = os.environ.get("BOX_LID_SLIDE_DIST")
        if override:
            try:
                primary_dist = float(override)
            except Exception:
                primary_dist = max(0.30, min(0.45, 1.5 * lid_len))
        else:
            # Default: open farther than before for larger box variants.
            primary_dist = max(0.30, min(0.45, 1.5 * lid_len))

        min_slide = float(os.environ.get("LID_SLIDE_MIN_DIST", "0.26"))
        candidates = []
        for d in [primary_dist, primary_dist - 0.04, primary_dist - 0.08, min_slide]:
            d = float(d)
            if d <= max(0.08, min_slide * 0.9):
                continue
            if all(abs(d - x) > 1e-6 for x in candidates):
                candidates.append(d)

        for slide_dist in candidates:
            slide_vec = np.array([slide_dist, 0, 0])
            target_pos = start_pos + slide_vec
            print(f"DEBUG: Constrained Slide Lid try distance={slide_dist:.3f}m (lid_len={lid_len:.3f})")

            self.set_robot_conf(initial_conf)
            try:
                path = self.robot.get_linear_path(
                    position=target_pos,
                    quaternion=start_quat,
                    steps=50,
                    ignore_collisions=True
                )
                if path:
                    traj_configs = path._path_points.reshape(-1, 7).tolist()
                    traj_return = traj_configs[::-1]
                    return traj_configs[0], traj_configs[-1], traj_configs, traj_return
            except Exception as e:
                print(f"DEBUG: Slide failed for {slide_dist:.3f}m: {e}")
                continue

        print("DEBUG: Failed to compute slide path for all candidate distances.")
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
            if getattr(self, 'box_boundary', None) is not None:
                self.box_boundary.set_collidable(enabled)
            print(f"  [Env] Box Collision set to {enabled}")
