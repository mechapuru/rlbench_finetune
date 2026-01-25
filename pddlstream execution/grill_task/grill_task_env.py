# grill_task_env.py
# Environment for the grill task scene (grill_task.ttt)

import numpy as np
import math
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import ConfigurationPathAlgorithms
from pyrep.backend import sim

SCENE_FILE = "/home/naren/iiith/Long_Horizon/TAMP-PDDL/grill_task.ttt"


def quaternion_from_euler(ai, aj, ak):
    """Convert Euler angles (XYZ convention) to quaternion [qx, qy, qz, qw]."""
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    q = [cj * sc - sj * cs, cj * ss + sj * cc, cj * cs - sj * sc, cj * cc + sj * ss]
    return q


class GrillTaskEnv:
    def __init__(self, headless=True):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()

        # ---- Robot ----
        self.robot = Panda()
        self.gripper = PandaGripper()
        # Store the initial joint configuration as "home" for retreating
        self.home_conf = self.robot.get_joint_positions()

        # ---- Cameras ----
        self.cams = {}
        cam_names = ['cam_over_shoulder_left', 'cam_over_shoulder_right', 
                     'cam_overhead', 'cam_wrist', 'cam_front']
        for name in cam_names:
            try:
                self.cams[name] = VisionSensor(name)
                self.cams[name].set_explicit_handling(1)
                self.cams[name].set_resolution([640, 480])
            except Exception:
                pass

        # Directory for saving/loading named joint configurations
        import os as _os
        self.state_dir = _os.path.join(_os.path.dirname(__file__), "data_states")
        _os.makedirs(self.state_dir, exist_ok=True)

        # ---- Objects based on scene hierarchy ----
        # From the scene hierarchy image:
        # - steak (with steak_visual)
        # - chicken (with chicken_visual)
        # - grill_root -> grill -> grill_visual, Shape1
        # - lid_joint -> lid -> lid_visual, handle_visual
        # - dish_rack with pillars and success positions
        # - plate with plate_visual
        
        try:
            # Meat objects (movable)
            self.steak = Shape('steak_visual')
            self.chicken = Shape('chicken_visual')
            
            # Grill components
            self.grill = Shape('grill_visual')
            self.grill_lid = Shape('lid_visual')
            
            # grill_boundary is the top surface where objects are placed
            # Make it the collision surface so objects don't fall through
            self.grill_surface = Shape('grill_boundary')
            self.grill_surface.set_collidable(True)
            self.grill_surface.set_respondable(True)
            self.grill_surface.set_dynamic(False)  # Static surface
            
            # Try to get the lid joint for hinged rotation
            try:
                self.lid_joint = Joint('lid_joint')
            except Exception:
                self.lid_joint = None
                print("Warning: 'lid_joint' not found, lid rotation may not work")
            
            # Try to get the handle for grasping
            try:
                self.handle = Shape('handle_visual')
            except Exception:
                self.handle = None
                print("Warning: 'handle_visual' not found")
            
            # Plate (for placing meat after cooking potentially)
            try:
                self.plate = Shape('plate_visual')
            except Exception:
                # Try 'plate' if 'plate_visual' not found
                try:
                    self.plate = Shape('plate')
                except Exception:
                    self.plate = None
                    print("Warning: Neither 'plate_visual' nor 'plate' found")
            
            # Plate boundary (placement surface on the plate)
            try:
                self.plate_boundary = Shape('plate_boundary')
                self.plate_boundary.set_collidable(True)
                self.plate_boundary.set_respondable(True)
                self.plate_boundary.set_dynamic(False)  # Static surface
            except Exception as e:
                self.plate_boundary = None
                print(f"Warning: 'plate_boundary' shape not found: {e}")
            
            # Dish rack
            try:
                self.dish_rack = Shape('dish_rack')
            except Exception:
                self.dish_rack = None
            
            # Success region for placing meat on grill (Shape boundary)
            try:
                self.grill_boundary = Shape('grill_boundary')
            except Exception:
                self.grill_boundary = None
                print("Warning: 'grill_boundary' shape not found")

        except Exception as e:
            print(f"Warning: Some objects not found in scene: {e}")

        # Object name mapping
        self.name_to_obj = {
            'steak': self.steak,
            'chicken': self.chicken,
            'meat1': self.steak,      # Alias
            'meat2': self.chicken,    # Alias
            'grill_lid': self.grill_lid,
            'lid': self.grill_lid,
        }
        
        # Add plate if exists
        if self.plate:
            self.name_to_obj['plate'] = self.plate

        # Define regions
        # grill-top uses the grill_boundary dummy for correct placement area
        # plate-top uses the plate_boundary for correct placement area
        self.regions = {
            'grill-top': self.grill_boundary if self.grill_boundary else self.grill,
            'table': self.grill,  # Fallback
        }
        
        if self.plate:
            self.regions['plate'] = self.plate
        if self.plate_boundary:
            self.regions['plate-top'] = self.plate_boundary
            self.regions['plate_boundary'] = self.plate_boundary  # Alias underscore
            self.regions['plate-boundary'] = self.plate_boundary  # Alias hyphen
        elif self.plate:
            # Fallback to plate_visual if no boundary
            self.regions['plate-top'] = self.plate
            self.regions['plate_boundary'] = self.plate
            self.regions['plate-boundary'] = self.plate
        if self.dish_rack:
            self.regions['dish_rack'] = self.dish_rack

    def get_object(self, name):
        """Get object by name."""
        if name in self.name_to_obj:
            return self.name_to_obj[name]
        try:
            # Dynamic lookup
            obj = Shape(name)
            self.name_to_obj[name] = obj
            return obj
        except Exception:
            return None

    def get_robot_conf(self):
        """Get current robot joint configuration."""
        return self.robot.get_joint_positions()

    def get_home_conf(self):
        """Return the stored home configuration captured at startup."""
        return list(self.home_conf)

    def set_robot_conf(self, q):
        """Set robot joint configuration."""
        self.robot.set_joint_positions(q)
        try:
            self.robot.set_joint_target_positions(q)
        except Exception:
            pass

    def save_conf(self, name, q=None):
        """Save a joint configuration under a given name."""
        import os as _os
        if q is None:
            q = self.get_robot_conf()
        path = _os.path.join(self.state_dir, f"{name}.npy")
        np.save(path, np.array(q, dtype=np.float32))

    def load_conf(self, name):
        """Load a previously saved joint configuration."""
        import os as _os
        path = _os.path.join(self.state_dir, f"{name}.npy")
        if not _os.path.exists(path):
            raise FileNotFoundError(f"Saved conf '{name}' not found at {path}")
        return np.load(path).tolist()

    def _get_world_bounding_box(self, obj):
        """Get the axis-aligned bounding box of an object in world coordinates."""
        min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
        corners = np.array([
            [min_x, min_y, min_z], [min_x, min_y, max_z],
            [min_x, max_y, min_z], [min_x, max_y, max_z],
            [max_x, min_y, min_z], [max_x, min_y, max_z],
            [max_x, max_y, min_z], [max_x, max_y, max_z]
        ])

        # Transform to world
        matrix = obj.get_matrix()
        m = np.array(matrix)
        if m.size == 16:
            m = m.reshape(4, 4)
        elif m.size == 12:
            m = m.reshape(3, 4)

        world_corners = []
        for c in corners:
            c_h = np.append(c, 1.0)
            if m.shape == (4, 4):
                wc = np.dot(m, c_h)[:3]
            else:
                wc = np.dot(m, c_h)
            world_corners.append(wc)
        world_corners = np.array(world_corners)

        w_min = np.min(world_corners, axis=0)
        w_max = np.max(world_corners, axis=0)
        return w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]

    def _get_linear_path(self, q_start, target_pos, target_quat, ignore_collisions=False, steps=50):
        """Plan a linear Cartesian path."""
        self.set_robot_conf(q_start)
        try:
            path = self.robot.get_linear_path(
                position=target_pos, 
                quaternion=target_quat, 
                steps=steps, 
                ignore_collisions=ignore_collisions
            )
            return path
        except Exception:
            return None

    def _interpolate_joint_path(self, q1, q2, steps=50, check_collisions=True):
        """Generate a simple joint-space interpolation, optionally collision-checked."""
        traj = []
        q1 = np.array(q1)
        q2 = np.array(q2)

        if steps < 50:
            steps = 50

        for i in range(steps + 1):
            t = i / steps
            q = (1 - t) * q1 + t * q2
            q_list = q.tolist()

            if check_collisions:
                self.set_robot_conf(q_list)
                if self.robot.check_collision():
                    return None

            traj.append(q_list)
        return traj

    def compute_hover_config(self, obj, pose, hover_offset=0.15, preferred_orientation=None):
        """
        Return a valid configuration q_hover strictly above the object.
        Also returns the orientation used so pick can use the same.
        
        Args:
            obj: The target object
            pose: Object pose [x,y,z,qx,qy,qz,qw]
            hover_offset: Height above object
            preferred_orientation: If provided, use this orientation (from previous pick computation)
        
        Returns:
            (q_hover, hover_orientation) - joint config and the quaternion used
        """
        original_conf = self.get_robot_conf()
        try:
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            top_z_local = max_z
            hover_z = pose[2] + top_z_local + hover_offset

            target_pos = [pose[0], pose[1], hover_z]

            # If preferred orientation is given, try that first
            if preferred_orientation is not None:
                path_configs = self.robot.solve_ik_via_sampling(
                    target_pos, quaternion=preferred_orientation, 
                    max_configs=5, max_time_ms=100, 
                    ignore_collisions=True
                )
                if path_configs is not None and len(path_configs) > 0:
                    q_hover = path_configs[0]
                    self.set_robot_conf(q_hover)
                    if not self.robot.check_collision():
                        return q_hover, preferred_orientation

            # Sample orientations pointing down
            grasp_quats = []
            for angle in np.linspace(0, 2 * np.pi, 16):
                q = quaternion_from_euler(np.pi, 0, angle)
                grasp_quats.append((angle, q))

            for angle, grasp_rot in grasp_quats:
                path_configs = self.robot.solve_ik_via_sampling(
                    target_pos, quaternion=grasp_rot, 
                    max_configs=5, max_time_ms=100, 
                    ignore_collisions=True
                )
                if path_configs is not None and len(path_configs) > 0:
                    q_hover = path_configs[0]
                    # Check collision
                    self.set_robot_conf(q_hover)
                    if not self.robot.check_collision():
                        return q_hover, grasp_rot

            raise RuntimeError("Could not find valid hover configuration")
        finally:
            self.set_robot_conf(original_conf)

    def compute_retreat_to_home(self, q_start):
        """Plan a retreat from the current config back to the stored home pose."""
        q_home = list(self.home_conf)

        # 1. Try simple interpolation first (fastest)
        traj = self._interpolate_joint_path(q_start, q_home, steps=50, check_collisions=True)
        if traj:
            return q_home, traj

        # 2. If blocked, try lifting first
        self.set_robot_conf(q_start)
        curr_pos = self.robot.get_position()
        curr_quat = self.robot.get_quaternion()

        # Lift by 20cm
        lift_pos = [curr_pos[0], curr_pos[1], curr_pos[2] + 0.2]
        path_lift = self.robot.get_linear_path(
            position=lift_pos, quaternion=curr_quat, 
            steps=30, ignore_collisions=False
        )

        if path_lift:
            q_lift_end = path_lift._path_points[-7:].tolist()
            traj_home = self._interpolate_joint_path(q_lift_end, q_home, steps=50, check_collisions=True)
            if traj_home:
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                traj_lift[0] = list(q_start)
                return q_home, traj_lift + traj_home

        return None, None

    def compute_motion_plan(self, q1, q2):
        """Plan a path from q1 to q2 with collision checking."""
        try:
            # Special case: retreat to home
            if np.allclose(q2, self.home_conf, atol=1e-3):
                _, traj = self.compute_retreat_to_home(q1)
                return traj

            # 1. Try simple interpolation first
            traj = self._interpolate_joint_path(q1, q2, steps=50, check_collisions=True)
            if traj:
                return traj

            # 2. Try lifting first, then move
            self.set_robot_conf(q1)
            p1 = self.robot.get_position()
            quat1 = self.robot.get_quaternion()

            p_lift = [p1[0], p1[1], p1[2] + 0.25]
            path_lift = self.robot.get_linear_path(
                position=p_lift, quaternion=quat1, 
                steps=30, ignore_collisions=False
            )

            if path_lift:
                q_lift = path_lift._path_points[-7:].tolist()
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                traj_lift[0] = list(q1)

                traj_rest = self._interpolate_joint_path(q_lift, q2, steps=100, check_collisions=True)
                if traj_rest:
                    return traj_lift + traj_rest

            # 3. Try via home
            if not np.allclose(q1, self.home_conf, atol=1e-3):
                _, traj_to_home = self.compute_retreat_to_home(q1)
                if traj_to_home:
                    traj_from_home = self._interpolate_joint_path(
                        self.home_conf, q2, steps=100, check_collisions=True
                    )
                    if traj_from_home:
                        return traj_to_home + traj_from_home

            return None
        except Exception:
            return None

    def sample_stable_pose(self, obj, region_name):
        """Return a stable 7D pose (x,y,z,qx,qy,qz,qw) for obj in region."""
        from pyrep.objects.dummy import Dummy
        
        region = self.regions.get(region_name)
        if not region:
            print(f"Region {region_name} not found, returning current pose")
            return obj.get_pose()

        current_pose = obj.get_pose()
        
        # Check if region is a Dummy (point in space) vs Shape (has bounding box)
        if isinstance(region, Dummy):
            # For Dummy objects, use the position directly with small random offset
            dummy_pos = region.get_position()
            # Add small random offset within a radius (e.g., 5cm)
            radius = 0.05
            offset_x = np.random.uniform(-radius, radius)
            offset_y = np.random.uniform(-radius, radius)
            
            sample_x = dummy_pos[0] + offset_x
            sample_y = dummy_pos[1] + offset_y
            sample_z = dummy_pos[2] + 0.01  # Slightly above the dummy point
            
            print(f"DEBUG: Using Dummy position for grill-top: {dummy_pos}")
            print(f"DEBUG: Sampled place position: [{sample_x:.3f}, {sample_y:.3f}, {sample_z:.3f}]")
        else:
            # For Shape objects, use bounding box
            w_min_x, w_max_x, w_min_y, w_max_y, w_min_z, w_max_z = self._get_world_bounding_box(region)

            # Sample x and y within world bounds (with padding)
            padding = 0.03
            if (w_max_x - w_min_x) < 2 * padding:
                padding = 0
            if (w_max_y - w_min_y) < 2 * padding:
                padding = 0

            sample_x = np.random.uniform(w_min_x + padding, w_max_x - padding)
            sample_y = np.random.uniform(w_min_y + padding, w_max_y - padding)

            # For grill-top, place on top of grill surface
            if region_name == 'grill-top':
                sample_z = w_max_z + 0.03  # Higher above grill surface to avoid clipping
            elif region_name in ['plate-top', 'plate_boundary', 'plate']:
                sample_z = w_max_z + 0.02  # Slightly above plate surface
            else:
                sample_z = w_max_z + 0.005

        new_pose = list(current_pose)
        new_pose[0] = sample_x
        new_pose[1] = sample_y
        new_pose[2] = sample_z

        return new_pose

    def compute_pick_trajectory(self, obj, pose, preferred_orientation=None, is_plate=False):
        """
        Return grasp, q_start, q_end, and trajectory for picking obj at pose.
        Uses top-down vertical grasp strategy.
        
        Args:
            obj: The target object
            pose: Object pose [x,y,z,qx,qy,qz,qw]
            preferred_orientation: If provided (from hover), try this orientation first
            is_plate: If True, go DEEPER for better grasp on plate
        
        Returns:
            (grasp, q_start, q_end, (approach_traj, retreat_traj))
        """
        original_conf = self.get_robot_conf()
        
        try:
            # 1. Analyze Object Geometry
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            obj_height = max_z - min_z
            top_z_local = max_z
            
            print(f"DEBUG pick: Object bounding box height = {obj_height:.4f}")

            # 2. Define Grasp Strategy (Top-Down)
            if is_plate:
                # For plate: slightly deeper to get a good grip on the rim
                grasp_depths = [0.025, 0.03, 0.035, 0.04, 0.045]
                print(f"DEBUG pick: PLATE mode - grasp depths: {grasp_depths}")
                valid_depths = grasp_depths  # Use all for plate
            else:
                grasp_depths = [0.02, 0.04, 0.06, 0.08]
                valid_depths = [d for d in grasp_depths if d < (obj_height - 0.01)]
                if not valid_depths:
                    valid_depths = [obj_height / 2.0]

            # 3. Define Grasp Orientations
            grasp_quats = []
            if is_plate:
                # For plate: try 90-degree rotated orientations first
                for angle in [np.pi/2, -np.pi/2, 0, np.pi]:
                    q = quaternion_from_euler(np.pi, 0, angle)
                    grasp_quats.append(q)
            elif preferred_orientation is not None:
                grasp_quats.append(preferred_orientation)
            
            angles = np.linspace(0, 2 * np.pi, 16)
            for angle in angles:
                q = quaternion_from_euler(np.pi, 0, angle)
                grasp_quats.append(q)

            # 4. Iterate and Solve
            for depth in valid_depths:
                target_z = pose[2] + top_z_local - depth
                target_pos = [pose[0], pose[1], target_z]

                hover_offset = 0.15
                hover_pos = [target_pos[0], target_pos[1], target_pos[2] + hover_offset]
                
                print(f"DEBUG pick: Trying depth={depth:.3f}, target_z={target_z:.3f}")

                for grasp_rot in grasp_quats:
                    try:
                        # A. Solve IK for Grasp Pose - increased sampling for plate
                        path_configs = self.robot.solve_ik_via_sampling(
                            target_pos, quaternion=grasp_rot,
                            max_configs=10, max_time_ms=200,
                            ignore_collisions=True
                        )
                        if path_configs is None or len(path_configs) == 0:
                            continue
                        q_grasp = path_configs[0]

                        # B. Solve IK for Hover Pose
                        path_configs_hover = self.robot.solve_ik_via_sampling(
                            hover_pos, quaternion=grasp_rot,
                            max_configs=10, max_time_ms=200,
                            ignore_collisions=True
                        )
                        if path_configs_hover is None or len(path_configs_hover) == 0:
                            continue
                        q_hover = path_configs_hover[0]

                        # Validate hover is collision-free (skip for plate since it's in dish rack)
                        self.set_robot_conf(q_hover)
                        if not is_plate and self.robot.check_collision():
                            continue

                        # C. Plan Hover -> Grasp (Linear Approach)
                        path2 = self._get_linear_path(q_hover, target_pos, grasp_rot, ignore_collisions=True)
                        if not path2:
                            continue

                        q_grasp_actual = path2._path_points[-7:].tolist()

                        # D. Plan Grasp -> Hover (Linear Retract)
                        path3 = self._get_linear_path(q_grasp_actual, hover_pos, grasp_rot, ignore_collisions=True)
                        if not path3:
                            path2.remove()
                            continue

                        q_hover_end = path3._path_points[-7:].tolist()

                        def get_configs(p):
                            return p._path_points.reshape(-1, 7).tolist()

                        t_approach = get_configs(path2)
                        t_retreat = get_configs(path3)

                        grasp = [0] * 7
                        print(f"DEBUG pick: SUCCESS at depth={depth:.3f}")
                        return grasp, q_hover, q_hover_end, (t_approach, t_retreat)

                    except Exception:
                        continue

            print(f"DEBUG: compute_pick_trajectory failed for {obj} at {pose}")
            raise RuntimeError("Could not find valid grasp configuration")
        finally:
            self.set_robot_conf(original_conf)

    def compute_place_trajectory(self, obj, pose, region_name=None, is_plate=False):
        """
        Return grasp, q_start, q_end, and trajectory for placing obj at pose.
        
        For regular objects: top-down vertical approach
        For plates: HORIZONTAL approach (gripper fingers vertical, approach from side)
        
        Args:
            obj: Object to place
            pose: Target pose [x,y,z,qx,qy,qz,qw]
            region_name: Target region name
            is_plate: If True, use HORIZONTAL approach for plate placement
        """
        original_conf = self.get_robot_conf()
        try:
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            top_z_local = max_z

            if is_plate:
                # HORIZONTAL APPROACH for plates
                # Plate should be placed FLAT on the surface
                # Gripper approaches from the side (negative X direction), fingers vertical
                
                place_z = pose[2] + 0.05  # Height of the target surface + small offset
                horizontal_offset = 0.20  # How far back to hover before approaching
                
                # Hover position: same height as place, but offset in X (approach from front)
                target_pos_hover = [pose[0] - horizontal_offset, pose[1], place_z]
                # Place position: at target location
                target_pos_place = [pose[0], pose[1], place_z]
                
                print(f"DEBUG plate place: HORIZONTAL approach")
                print(f"  Hover (offset): {target_pos_hover}")
                print(f"  Place target: {target_pos_place}")
                
                # Gripper orientation for horizontal approach:
                # Gripper pointing forward (+X), fingers opening vertically (up/down)
                # euler(0, π/2, 0) = gripper horizontal, fingers vertical
                grasp_quats = []
                # Try different yaw angles for horizontal approach
                for yaw in [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]:
                    # Gripper horizontal (pointing +X), fingers vertical
                    q = quaternion_from_euler(np.pi/2, yaw, np.pi/2)
                    grasp_quats.append(q)
                    # Also try slight tilts
                    q2 = quaternion_from_euler(np.pi/2 + 0.1, yaw, np.pi/2)
                    grasp_quats.append(q2)
                    q3 = quaternion_from_euler(np.pi/2 - 0.1, yaw, np.pi/2)
                    grasp_quats.append(q3)
            else:
                # VERTICAL APPROACH for regular objects (top-down)
                hover_z = pose[2] + top_z_local + 0.15
                place_z = pose[2] + 0.05

                target_pos_hover = [pose[0], pose[1], hover_z]
                target_pos_place = [pose[0], pose[1], place_z]
                
                print(f"DEBUG place: VERTICAL approach")
                print(f"  Hover (above): {target_pos_hover}")
                print(f"  Place target: {target_pos_place}")

                grasp_quats = []
                angles = np.linspace(0, 2 * np.pi, 32)
                for angle in angles:
                    q = quaternion_from_euler(np.pi, 0, angle)
                    grasp_quats.append(q)

            for i, grasp_rot in enumerate(grasp_quats):
                try:
                    # A. Solve IK for Hover Pose
                    path_configs_hover = self.robot.solve_ik_via_sampling(
                        target_pos_hover, quaternion=grasp_rot,
                        max_configs=10, max_time_ms=200,
                        ignore_collisions=True
                    )
                    if path_configs_hover is None or len(path_configs_hover) == 0:
                        continue
                    q_hover = path_configs_hover[0]

                    # B. Solve IK for Place Pose
                    path_configs_place = self.robot.solve_ik_via_sampling(
                        target_pos_place, quaternion=grasp_rot,
                        max_configs=10, max_time_ms=200,
                        ignore_collisions=True
                    )
                    if path_configs_place is None or len(path_configs_place) == 0:
                        continue
                    q_place = path_configs_place[0]

                    # C. Plan Hover -> Place (Linear path)
                    path_forward = self._get_linear_path(q_hover, target_pos_place, grasp_rot, ignore_collisions=True, steps=50)
                    if not path_forward:
                        continue

                    # D. Plan Place -> Hover (Linear Return - retreat)
                    path_back = self._get_linear_path(q_place, target_pos_hover, grasp_rot, ignore_collisions=True, steps=50)
                    if not path_back:
                        path_forward.remove()
                        continue

                    def get_configs(p):
                        return p._path_points.reshape(-1, 7).tolist()

                    t_forward = get_configs(path_forward)
                    t_back = get_configs(path_back)

                    grasp = [0] * 7
                    print(f"DEBUG: Found valid place config at orientation {i}")
                    return grasp, q_hover, q_hover, (t_forward, t_back)

                except Exception as e:
                    continue

            raise RuntimeError(f"Could not find valid place configuration for region {region_name}")
        finally:
            self.set_robot_conf(original_conf)

    def compute_close_grill_trajectory(self, lid):
        """
        Compute trajectory to close the grill lid (hinged rotation).
        This involves grasping the handle and rotating the lid down.
        """
        original_conf = self.get_robot_conf()
        try:
            # For now, return a placeholder - actual implementation needs
            # to understand the lid joint mechanism in the scene
            if self.lid_joint is not None:
                # Get current joint position and target
                current_angle = self.lid_joint.get_joint_position()
                # Assuming 0 is closed, and current is open
                # We need to plan a trajectory that follows the arc

            # Placeholder: return a simple approach-grasp-rotate sequence
            grasp = [0] * 7
            q_start = self.get_home_conf()
            q_end = self.get_home_conf()
            traj = ([], [])  # Placeholder empty trajectories

            raise RuntimeError("Close grill trajectory not fully implemented yet")
        finally:
            self.set_robot_conf(original_conf)
