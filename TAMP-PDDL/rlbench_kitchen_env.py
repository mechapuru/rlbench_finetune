# rlbench_kitchen_env.py
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import ConfigurationPathAlgorithms
from pyrep.backend import sim

SCENE_FILE = "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/kitchen_task/task_design_proposal_variation_1.ttt"

class RLBenchKitchenEnv:
    def __init__(self, headless=True):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()

        # ---- robot ----
        self.robot = Panda()
        # self.arm_joints = self.robot.get_joints() # Panda object doesn't expose get_joints directly like this
        self.gripper = PandaGripper()
        # Store the initial joint configuration as "home" for retreating
        self.home_conf = self.robot.get_joint_positions()
        
        # ---- cameras ----
        self.cams = {
            'left': VisionSensor('cam_over_shoulder_left'),
            'right': VisionSensor('cam_over_shoulder_right'),
            'overhead': VisionSensor('cam_overhead'),
            'wrist': VisionSensor('cam_wrist'),
            'front': VisionSensor('cam_front'),
        }
        
        # Configure cameras for recording
        for name, cam in self.cams.items():
            cam.set_explicit_handling(1) # We will manually trigger capture
            cam.set_resolution([640, 480]) # Set resolution
            # cam.set_render_mode(VisionSensor.RenderMode.OPENGL3) # Ensure OpenGL rendering
            
        # Set a starting configuration if needed
        # self.robot.set_joint_positions([...])

        # Directory for saving/loading named joint configurations
        import os as _os
        self.state_dir = _os.path.join(_os.path.dirname(__file__), "data_states")
        _os.makedirs(self.state_dir, exist_ok=True)

        # ---- objects (FIX NAMES TO MATCH .ttt SCENE) ----
        self.target_region_name = None # Global context for pick strategy
        
        self.name_to_obj = {}
        self.regions = {}
        # Using a try-except block to handle potential missing objects gracefully during dev
        try:
            self.mug_cupboard = Shape('mug3')
            self.mug_box      = Shape('mug2')
            self.mug_table    = Shape('mug1')
            self.mug_inside_box = Shape('mug4')

            self.bottle = Shape('mustard')
            self.can    = Shape('soup')
            self.tin    = Shape('spam')
            self.food_box = Shape('sugar')
            self.cereal   = Shape('crackers')

            self.table   = Shape('diningTable')
            self.box     = Shape('box_base')
            try:
                self.box.set_dynamic(False) # Ensure box base is static (Strong Hinge effect)
            except:
                pass
            try:
                self.box_lid = Shape('box_lid')
            except:
                print("Warning: 'box_lid' not found, using box_base as placeholder.")
                self.box_lid = self.box
            self.cupboard = Shape('cupboard')
            
            try:
                self.groceries_boundary = Shape('groceries_boundary')
            except:
                print("Warning: 'groceries_boundary' object not found, using table as boundary.")
                self.groceries_boundary = self.table

            try:
                self.placement_boundary = Shape('placement_boundary')
            except:
                print("Warning: 'placement_boundary' object not found, using table as boundary.")
                self.placement_boundary = self.table

            try:
                self.cupboard_boundary = Shape('cupboard_boundary')
            except:
                print("Warning: 'cupboard_boundary' object not found, using cupboard as boundary.")
                self.cupboard_boundary = self.cupboard

            try:
                self.box_boundary = Shape('box_boundary')
            except:
                print("Warning: 'box_boundary' object not found, using box as boundary.")
                self.box_boundary = self.box

            try:
                self.cupboard_boundary_top = Shape('cupboard_boundary_top')
            except:
                print("Warning: 'cupboard_boundary_top' object not found, using cupboard_boundary as fallback.")
                self.cupboard_boundary_top = self.cupboard_boundary

            # Map region names to objects
            self.regions = {
                'table': self.table,
                'box-top': self.box, # Assuming box has a top surface we can place on
                'box-inside': self.box, # This might need a specific dummy object for "inside"
                'shelf-lower': self.cupboard, # Assuming cupboard is the shelf
                'groceries_boundary': self.groceries_boundary,
                'placement_boundary': self.placement_boundary,
                'cupboard_boundary': self.cupboard_boundary,
                'cupboard_boundary_top': self.cupboard_boundary_top,
                'box_boundary': self.box_boundary
            }
            
            self.name_to_obj = {
                'mug_cupboard': self.mug_cupboard,
                'mug_box': self.mug_box,
                'mug_table': self.mug_table,
                'mug_inside_box': self.mug_inside_box,
                'bottle': self.bottle,
                'can': self.can,
                'tin': self.tin,
                'food_box': self.food_box,
                'cereal': self.cereal,
                # Aliases for user convenience
                'soup': self.can,
                'soup_box': self.can,
                'mustard': self.bottle,
                'mustard_box': self.bottle,
                'spam': self.tin,
                'spam_box': self.tin,
                'sugar': self.food_box,
                'crackers': self.cereal,
                'box_lid': self.box_lid,
                # Mug aliases
                'mug1': self.mug_table,
                'mug2': self.mug_box,
                'mug3': self.mug_cupboard,
                'mug4': self.mug_inside_box,
            }
            
            # Track placed positions to avoid overlapping placements
            self.placed_positions = []
            
            # DEBUG: Move mug_box to a reachable position
            # Assuming robot is at (0,0,0) or near it.
            # Move to x=0.5, y=0.0, z=table_height (approx 0.75) + box height
            # Let's check robot pos first
            # r_pos = self.robot.get_position()
            # print(f"DEBUG: Robot is at {r_pos}")
            
            # DEBUG: Find a reachable position for the mug
            # Instead of guessing, we search for a pose where IK succeeds
            # self.place_mug_in_reachable_pose()
            
        except Exception as e:
            print(f"Warning: Some objects not found in scene: {e}")

    def place_mug_in_reachable_pose(self):
        # print("DEBUG: Placing mug on top of the BOX (as requested)...")
        
        try:
            # Get box details
            box_pos = self.box.get_position()
            min_x, max_x, min_y, max_y, min_z, max_z = self.box.get_bounding_box()
            
            # Calculate top of the box
            # Bounding box is usually local. If box is at Z, top is Z + max_z
            box_top_z = box_pos[2] + max_z
            # print(f"DEBUG: Box found at {box_pos}, Top Z estimated at {box_top_z:.3f}")
            
            # Define a search grid ON THE BOX
            # We'll search a small area on top of the box
            # Assuming box is roughly centered at box_pos
            # We'll try a few spots relative to box center
            x_offsets = [0.0, 0.05, -0.05]
            y_offsets = [0.0, 0.05, -0.05]
            
            # Mug needs to be placed ON the box, so Z = box_top_z + mug_half_height?
            # Usually origin of mug is at bottom or center. 
            # If center, we need to add half height. If bottom, just box_top_z.
            # Let's assume origin is at bottom for now, or add a small safety margin.
            place_z = box_top_z + 0.005 
            
            grasp_quat = [1.0, 0.0, 0.0, 0.0] # Vertical grasp
            
            for x_off in x_offsets:
                for y_off in y_offsets:
                    test_pos = [box_pos[0] + x_off, box_pos[1] + y_off, place_z]
                    
                    # print(f"DEBUG: Checking reachability at {test_pos}...")
                    
                    # Check IK
                    path = self.robot.solve_ik_via_sampling(test_pos, quaternion=grasp_quat, max_configs=1, max_time_ms=50, ignore_collisions=True)
                    
                    if path is not None and len(path) > 0:
                        # print(f"DEBUG: SUCCESS - Found reachable pose on BOX at {test_pos}")
                        self.mug_box.set_position(test_pos)
                        return

            print("DEBUG: WARNING - Could not find reachable pose on box via IK check. Placing at center anyway.")
            self.mug_box.set_position([box_pos[0], box_pos[1], place_z])
            
        except Exception as e:
            print(f"DEBUG: Error placing mug on box: {e}")
            # Fallback
            r_pos = self.robot.get_position()
            self.mug_box.set_position([r_pos[0] + 0.5, r_pos[1], 0.9])

    def _get_path(self, q_start, target_pos, target_quat):
        # Helper to plan path
        self.set_robot_conf(q_start)
        try:
            # Plan to Cartesian target
            # linear=False means use OMPL (RRTConnect usually)
            # Reduced max_time_ms to fail faster if stuck
            # Increased trials to find better paths
            path = self.robot.get_path(position=target_pos, quaternion=target_quat,
                                       ignore_collisions=False,
                                       algorithm=ConfigurationPathAlgorithms.RRTConnect,
                                       max_configs=5, trials=20, max_time_ms=2000.0)
            return path
        except Exception as e:
            print(f"DEBUG: Planning failed: {e}")
            return None

    def get_object(self, name):
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
        return self.robot.get_joint_positions()

    def get_home_conf(self):
        """Return the stored home configuration captured at startup."""
        return list(self.home_conf)

    def save_conf(self, name, q=None):
        """Save a joint configuration under a given name."""
        import numpy as _np
        import os as _os
        if q is None:
            q = self.get_robot_conf()
        path = _os.path.join(self.state_dir, f"{name}.npy")
        _np.save(path, _np.array(q, dtype=_np.float32))

    def load_conf(self, name):
        """Load a previously saved joint configuration."""
        import numpy as _np
        import os as _os
        path = _os.path.join(self.state_dir, f"{name}.npy")
        if not _os.path.exists(path):
            raise FileNotFoundError(f"Saved conf '{name}' not found at {path}")
        return _np.load(path).tolist()

    def draw_trajectory(self, points, color=(1.0, 0.0, 0.0), size=4.0):
        """Draw a persistent 3D line strip trajectory in the scene.

        points: list of [x, y, z] world coordinates along the path.
        color:  RGB tuple in [0,1].
        size:   line width in pixels.
        """
        if not points:
            return

        options = sim.sim_drawing_lines | sim.sim_drawing_cyclic
        max_items = len(points)
        drawing_handle = sim.simAddDrawingObject(
            options,
            size,
            0.0,
            -1,
            max_items,
            [float(color[0]), float(color[1]), float(color[2])],
        )

        for p in points:
            coords = [float(p[0]), float(p[1]), float(p[2])]
            sim.simAddDrawingObjectItem(drawing_handle, coords)

    def set_robot_conf(self, q):
        self.robot.set_joint_positions(q)
        try:
            self.robot.set_joint_target_positions(q)
        except Exception:
            pass

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
            # Homogeneous coordinate for corner
            c_h = np.append(c, 1.0)
            if m.shape == (4, 4):
                wc = np.dot(m, c_h)[:3]
            else:
                # 3x4 matrix
                wc = np.dot(m, c_h)
            world_corners.append(wc)
        world_corners = np.array(world_corners)
        
        w_min = np.min(world_corners, axis=0)
        w_max = np.max(world_corners, axis=0)
        return w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]

    def sample_stable_pose(self, obj, region_name):
        """Return a stable 7D pose (x,y,z,qx,qy,qz,qw) for obj in region."""
        region = self.regions.get(region_name)
        if not region:
            print(f"Region {region_name} not found, returning current pose")
            return obj.get_pose()

        # Use robust world bounding box calculation
        w_min_x, w_max_x, w_min_y, w_max_y, w_min_z, w_max_z = self._get_world_bounding_box(region)
        
        # Sample x and y within world bounds (with padding)
        padding = 0.05
        
        # Ensure padding doesn't invert the range
        if (w_max_x - w_min_x) < 2*padding: padding = 0
        if (w_max_y - w_min_y) < 2*padding: padding = 0
        
        sample_x = np.random.uniform(w_min_x + padding, w_max_x - padding)
        sample_y = np.random.uniform(w_min_y + padding, w_max_y - padding)
        
        current_pose = obj.get_pose()
        
        # Adjust Z based on region
        if region_name == 'shelf-lower':
            sample_z = w_min_z + 0.01 
        elif region_name == 'box-inside':
            sample_z = w_min_z + 0.01
        elif region_name == 'placement_boundary':
             # For placement boundary, we want to be on the table surface.
             # Check table height
             table = self.regions.get('table')
             if table:
                 _, _, _, _, _, t_max_z = self._get_world_bounding_box(table)
                 sample_z = t_max_z + 0.005
             else:
                 sample_z = w_min_z + 0.005
        else:
            # Default (Table)
            sample_z = w_max_z + 0.005 
            
        # Use the sampled x,y and guessed z
        new_pose = list(current_pose)
        new_pose[0] = sample_x
        new_pose[1] = sample_y
        new_pose[2] = sample_z 
        
        return new_pose

    def _get_path(self, q_start, target_pos, target_quat):
        # Helper to plan path
        self.set_robot_conf(q_start)
        try:
            # Plan to Cartesian target
            # linear=False means use OMPL (RRTConnect usually)
            path = self.robot.get_path(position=target_pos, quaternion=target_quat,
                                       ignore_collisions=False,
                                       algorithm=ConfigurationPathAlgorithms.RRTConnect,
                                       max_configs=5, trials=5, max_time_ms=2000.0)
            return path
        except Exception as e:
            print(f"DEBUG: Planning failed: {e}")
            return None

    def _get_linear_path(self, q_start, target_pos, target_quat, ignore_collisions=False, steps=50):
        self.set_robot_conf(q_start)
        try:
            # steps=50 for finer resolution
            path = self.robot.get_linear_path(position=target_pos, quaternion=target_quat, steps=steps, ignore_collisions=ignore_collisions)
            return path
        except Exception as e:
            # print(f"Linear path failed: {e}")
            return None

    def _interpolate_joint_path(self, q1, q2, steps=50, check_collisions=True):
        """Generate a simple joint-space interpolation, optionally collision-checked."""
        traj = []
        q1 = np.array(q1)
        q2 = np.array(q2)
        
        # Use high resolution for safety
        if steps < 50: steps = 50

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

    def compute_retreat_to_home(self, q_start):
        """Plan a retreat from the current config back to the stored home pose."""
        q_home = list(self.home_conf)
        
        # 1. Try simple interpolation first (fastest)
        traj = self._interpolate_joint_path(q_start, q_home, steps=50, check_collisions=True) # Increased to 50
        if traj:
            return q_home, traj
            
        # 2. If blocked, use RRTConnect
        # print("DEBUG: Retreat interpolation blocked, trying RRT...")
        self.set_robot_conf(q_start)
        try:
            path = self.robot.get_path(position=None, quaternion=None,
                                     ignore_collisions=False,
                                     algorithm=ConfigurationPathAlgorithms.RRTConnect,
                                     max_configs=5, trials=5, max_time_ms=1000.0)
            # We need to set the target to home configuration, but get_path usually takes cartesian.
            # PyRep's get_path is for Cartesian. For joint space, we need get_linear_path (which is linear) 
            # or we need to use OMPL for joint space.
            # Actually, PyRep's get_path is Cartesian. 
            # Let's use a simple trick: Move to a high "safe" intermediate point if direct fails?
            # Or just rely on the fact that "Home" is usually safe.
            
            # If direct interpolation fails, it's likely we are deep in a bin.
            # Let's try to lift up first (Z+), then go home.
            
            # Get current cartesian pose
            curr_pos = self.robot.get_position()
            curr_quat = self.robot.get_quaternion()
            
            # Lift by 20cm
            lift_pos = [curr_pos[0], curr_pos[1], curr_pos[2] + 0.2]
            path_lift = self.robot.get_linear_path(position=lift_pos, quaternion=curr_quat, steps=30, ignore_collisions=False) # Increased to 30
            
            if path_lift:
                # From lift end, go home
                q_lift_end = path_lift._path_points[-7:].tolist()
                traj_home = self._interpolate_joint_path(q_lift_end, q_home, steps=50, check_collisions=True) # Increased to 50
                if traj_home:
                    # Combine
                    traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                    # FORCE CONTINUITY: Ensure the path starts exactly at q_start
                    traj_lift[0] = list(q_start)
                    return q_home, traj_lift + traj_home
            
            return None, None
        except Exception:
            return None, None

    def compute_motion_plan(self, q1, q2):
        """Plan a path from q1 to q2 with collision checking."""
        try:
            # Special case: retreat back to the startup configuration.
            if np.allclose(q2, self.home_conf, atol=1e-3):
                _, traj = self.compute_retreat_to_home(q1)
                return traj

            # Calculate FK for q1 (Start)
            self.set_robot_conf(q1)
            p1 = self.robot.get_position()
            quat1 = self.robot.get_quaternion()

            # Check holding status
            grasped_objects = self.gripper.get_grasped_objects()
            is_holding = (len(grasped_objects) > 0)

            # Check if we are "in the box" (or close to it)
            # If so, we MUST lift first to avoid rim collision during interpolation dip.
            in_box = False
            if hasattr(self, 'box'):
                box_pos = np.array(self.box.get_position())
                min_x, max_x, min_y, max_y, min_z, max_z = self.box.get_bounding_box()
                # World bounds with margin
                bx_min = box_pos[0] + min_x - 0.05
                bx_max = box_pos[0] + max_x + 0.05
                by_min = box_pos[1] + min_y - 0.05
                by_max = box_pos[1] + max_y + 0.05
                bz_max = box_pos[2] + max_z + 0.10 # Reduced safety height threshold (was 0.20)

                if (bx_min < p1[0] < bx_max) and (by_min < p1[1] < by_max) and (p1[2] < bz_max):
                    in_box = True
                    # print("DEBUG: Start position is inside/near box. Forcing Lift maneuver.")

            # 1. Try simple interpolation first (fast and clean)
            # ONLY if not in box AND not holding (to prevent dip)
            if not in_box and not is_holding:
                # Increased steps to 50 for smoothness
                traj = self._interpolate_joint_path(q1, q2, steps=50, check_collisions=True)
                if traj:
                    return traj
                
            # 2. SAFETY MANEUVER: If direct path fails (collision) OR we are in box OR holding, try to Lift/Retract first.
            # This fixes the "hitting cupboard" issue by forcing a crane-like move.
            
            # (p1 and quat1 are already calculated above)
            
            # Try to lift up significantly (0.25m) to clear obstacles (e.g. open lid)
            p_lift = [p1[0], p1[1], p1[2] + 0.25] # Increased from 0.15 for safety
            
            # Plan q1 -> q_lift
            path_lift = self.robot.get_linear_path(position=p_lift, quaternion=quat1, steps=30, ignore_collisions=False) # Increased to 30
            
            if path_lift:
                q_lift = path_lift._path_points[-7:].tolist()
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                # FORCE CONTINUITY: Ensure the path starts exactly at q1
                traj_lift[0] = list(q1)
                
                # Now try q_lift -> q2
                # We use RRTConnect here because q_lift -> q2 might be complex
                path_rest = self._get_path(q_lift, None, None) # This helper expects cartesian target...
                # We need joint path. Let's use interpolate first.
                traj_rest = self._interpolate_joint_path(q_lift, q2, steps=100, check_collisions=True) # Increased to 100
                
                if traj_rest:
                    return traj_lift + traj_rest
                
                # If interpolation fails, try via Home (High -> Home -> Target)
                if not np.allclose(q_lift, self.home_conf, atol=1e-3):
                    traj_home = self._interpolate_joint_path(q_lift, self.home_conf, steps=50, check_collisions=True) # Increased to 50
                    if traj_home:
                        traj_final = self._interpolate_joint_path(self.home_conf, q2, steps=100, check_collisions=True) # Increased to 100
                        if traj_final:
                            return traj_lift + traj_home + traj_final

            # 3. If that fails, try moving via Home configuration directly
            if not np.allclose(q1, self.home_conf, atol=1e-3):
                # Plan q1 -> Home
                _, traj_to_home = self.compute_retreat_to_home(q1)
                if traj_to_home:
                    # Plan Home -> q2
                    traj_from_home = self._interpolate_joint_path(self.home_conf, q2, steps=100, check_collisions=True) # Increased to 100
                    if traj_from_home:
                        return traj_to_home + traj_from_home

            # 4. If that fails, return None (Planner will retry or fail)
            # print(f"DEBUG: Motion plan failed for q1->q2 (Collision)")
            return None
        except Exception as e:
            # print(f"DEBUG: compute_motion_plan failed for q1={q1} q2={q2}: {e}")
            return None

    def set_target_region(self, name):
        self.target_region_name = name

    def compute_pick_trajectory(self, obj, pose):
        """Return grasp, q_start, q_end, and trajectory for picking obj at pose."""
        original_conf = self.get_robot_conf()
        
        # Handle obstructions for specific objects
        # ONLY disable lid collision if lid is actually OPEN (slid away)
        # Otherwise, let IK fail so COAST can learn the constraint
        lid_obj = None
        lid_collidable_state = True
        if obj.get_name() == 'mug4': # mug_inside_box is mug4
             lid_obj = self.get_object('box_lid')
             if lid_obj:
                 # Check if lid is actually open by checking its position
                 # Lid slides in X direction when opened (see compute_slide_lid_trajectory)
                 lid_pos = lid_obj.get_position()
                 box_obj = self.get_object('box_base')
                 if box_obj:
                     box_pos = box_obj.get_position()
                     # If lid has moved significantly in X, it's open
                     lid_offset = abs(lid_pos[0] - box_pos[0])
                     LID_OPEN_THRESHOLD = 0.10  # 10cm offset in X means open
                     print(f"DEBUG: Lid check - lid_pos={lid_pos}, box_pos={box_pos}, X_offset={lid_offset:.3f}")
                     if lid_offset > LID_OPEN_THRESHOLD:
                         # Lid is open, safe to disable collision for planning
                         lid_collidable_state = lid_obj.is_collidable()
                         lid_obj.set_collidable(False)
                         print(f"DEBUG: Lid is OPEN (X_offset={lid_offset:.3f}), disabling collision")
                     else:
                         # Lid is CLOSED - DO NOT disable collision
                         # Let IK fail naturally so COAST can learn
                         print(f"DEBUG: Lid is CLOSED (X_offset={lid_offset:.3f}), keeping collision ON - IK should fail!")
                         lid_obj = None  # Don't restore later since we didn't change it
                 
        try:
            # 1. Analyze Object Geometry
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            obj_height = max_z - min_z
            top_z_local = max_z
            
            # Check if object is inside cupboard_boundary
            in_cupboard = False
            # Explicitly check for mug_cupboard by name as well
            if obj.get_name() in ['mug_cupboard', 'mug3']:
                in_cupboard = True
                print(f"DEBUG: Object {obj.get_name()} identified as cupboard object by name.")
            elif hasattr(self, 'cupboard_boundary'):
                # Check if obj center is inside cupboard_boundary bbox
                # pose is the object's current pose
                bb_min_x, bb_max_x, bb_min_y, bb_max_y, bb_min_z, bb_max_z = self._get_world_bounding_box(self.cupboard_boundary)
                if (bb_min_x <= pose[0] <= bb_max_x) and (bb_min_y <= pose[1] <= bb_max_y) and (bb_min_z <= pose[2] <= bb_max_z):
                    in_cupboard = True
                    print(f"DEBUG: Object {obj.get_name()} detected inside cupboard boundary.")

            if in_cupboard:
                print(f"DEBUG: Using Horizontal Pick Strategy for {obj.get_name()}")
                # --- HORIZONTAL PICK STRATEGY ---
                # Target Pose: Object's current position
                target_pos = [pose[0], pose[1], pose[2]]
                
                # Hover Pose: In front of cupboard (shifted -X)
                # User requested 25cm clearance and strictly horizontal approach
                hover_dist = 0.25 
                hover_pos = [pose[0] - hover_dist, pose[1], pose[2]]
                
                # Grasp Orientation: Horizontal (Fingers Horizontal)
                # Base orientation: Ry=pi/2 (Z points +X)
                import math
                def quaternion_from_euler(ai, aj, ak):
                    ai /= 2.0
                    aj /= 2.0
                    ak /= 2.0
                    ci = math.cos(ai)
                    si = math.sin(ai)
                    cj = math.cos(aj)
                    sj = math.sin(aj)
                    ck = math.cos(ak)
                    sk = math.sin(ak)
                    cc = ci*ck
                    cs = ci*sk
                    sc = si*ck
                    ss = si*sk
                    q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
                    return q

                base_ry = np.pi/2
                # Strictly Horizontal Fingers (Roll = 0 or 180)
                grasp_quats = [
                    quaternion_from_euler(0, base_ry, 0),       # Fingers Horizontal
                    quaternion_from_euler(np.pi, base_ry, 0),   # Fingers Horizontal (flipped)
                ]
                
                # --- NEW: Extensive Sampling (Shotgun Approach) ---
                # Sample a grid around the object center to find ANY valid IK solution.
                # Added small positive Z bias to avoid scraping the shelf
                z_offsets = [0.02, 0.04, 0.0, -0.02, 0.05] 
                y_offsets = [0.0, 0.02, -0.02, 0.04, -0.04]
                
                for z_off in z_offsets:
                    for y_off in y_offsets:
                        target_pos_sample = [target_pos[0], target_pos[1] + y_off, target_pos[2] + z_off]
                        hover_pos_sample = [hover_pos[0], hover_pos[1] + y_off, hover_pos[2] + z_off]
                        
                        for grasp_rot in grasp_quats:
                            try:
                                # A. Solve IK for Hover Pose
                                path_configs_hover = self.robot.solve_ik_via_sampling(hover_pos_sample, quaternion=grasp_rot, max_configs=20, max_time_ms=500, ignore_collisions=True)
                                if path_configs_hover is None or len(path_configs_hover) == 0: 
                                    continue
                                
                                # Sort by distance to current config to find "natural" posture
                                curr_q = self.get_robot_conf()
                                path_configs_hover = sorted(path_configs_hover, key=lambda q: np.linalg.norm(np.array(q) - np.array(curr_q)))
                                q_hover = path_configs_hover[0]
                                
                                # B. Solve IK for Grasp Pose
                                path_configs_grasp = self.robot.solve_ik_via_sampling(target_pos_sample, quaternion=grasp_rot, max_configs=20, max_time_ms=500, ignore_collisions=True)
                                if path_configs_grasp is None or len(path_configs_grasp) == 0: 
                                    continue
                                
                                # Sort by distance to q_hover to ensure smooth transition
                                path_configs_grasp = sorted(path_configs_grasp, key=lambda q: np.linalg.norm(np.array(q) - np.array(q_hover)))
                                q_grasp = path_configs_grasp[0]
                                
                                # C. Plan Hover -> Grasp (Linear Approach)
                                # Force ignore_collisions=True to ensure we don't get blocked by minor grazes
                                path_approach = self._get_linear_path(q_hover, target_pos_sample, grasp_rot, steps=50, ignore_collisions=True)
                                if not path_approach: 
                                    continue
                                
                                # D. Plan Grasp -> Hover (Linear Retreat with Slant Lift)
                                # Lift 3cm during retreat to avoid friction
                                lifted_hover_pos = [hover_pos_sample[0], hover_pos_sample[1], hover_pos_sample[2] + 0.03]
                                path_retreat = self._get_linear_path(q_grasp, lifted_hover_pos, grasp_rot, steps=50, ignore_collisions=True)
                                if not path_retreat:
                                    path_approach.remove()
                                    continue
                                    
                                # Extract configs
                                def get_configs(p):
                                    return p._path_points.reshape(-1, 7).tolist()

                                t_approach = get_configs(path_approach)
                                t_retreat = get_configs(path_retreat)
                                
                                grasp = [0]*7
                                print(f"DEBUG: Found valid cupboard pick at offset Y={y_off}, Z={z_off}")
                                # Return split trajectories
                                return grasp, q_hover, q_hover, (t_approach, t_retreat)

                            except Exception:
                                continue
                
                print("DEBUG: Horizontal pick failed for all orientations and offsets.")
                # Don't raise yet, let it fall through? No, fall through means vertical grasp which is bad.
                raise RuntimeError("Could not find valid horizontal pick configuration for cupboard")

            # ALWAYS USE TOP GRASP (Vertical) per user request
            # 2. Define Grasp Strategy (Top-Down Depth Sampling)
            grasp_depths = [0.02, 0.04, 0.06, 0.08]
            valid_depths = [d for d in grasp_depths if d < (obj_height - 0.01)]
            if not valid_depths:
                valid_depths = [obj_height / 2.0]

            # 3. Define Grasp Orientations
            grasp_quats = []
            import math
            def quaternion_from_euler(ai, aj, ak):
                ai /= 2.0
                aj /= 2.0
                ak /= 2.0
                ci = math.cos(ai)
                si = math.sin(ai)
                cj = math.cos(aj)
                sj = math.sin(aj)
                ck = math.cos(ak)
                sk = math.sin(ak)
                cc = ci*ck
                cs = ci*sk
                sc = si*ck
                ss = si*sk
                q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
                return q

            # Check if object is inside box_boundary
            in_box_boundary = False
            if hasattr(self, 'box_boundary'):
                # Check if obj center is inside box_boundary bbox
                o_pos = obj.get_position()
                bb_min_x, bb_max_x, bb_min_y, bb_max_y, bb_min_z, bb_max_z = self._get_world_bounding_box(self.box_boundary)
                if (bb_min_x <= o_pos[0] <= bb_max_x) and (bb_min_y <= o_pos[1] <= bb_max_y) and (bb_min_z <= o_pos[2] <= bb_max_z):
                    in_box_boundary = True
                    print(f"DEBUG: Object {obj.get_name()} detected inside box_boundary. Restricting grasp orientation.")

            if in_box_boundary:
                 # User request: aligned with x axis, no orientation/angle in xy axis
                 # We'll use 0, pi/2, pi, 3pi/2 to cover both alignments (fingers along X or Y)
                 angles = [0, np.pi/2, np.pi, 3*np.pi/2]
                 print(f"DEBUG: Object {obj.get_name()} is inside box_boundary. Using axis-aligned grasps: {angles}")
            else:
                 # Increased density of grasp orientations
                 angles = np.linspace(0, 2*np.pi, 32)

            for angle in angles:
                q = quaternion_from_euler(np.pi, 0, angle)
                grasp_quats.append(q)
            
            # 4. Iterate and Solve
            for depth in valid_depths:
                # Target Z is Top - Depth
                target_z = pose[2] + top_z_local - depth
                target_pos = [pose[0], pose[1], target_z]
                
                # Adjust hover height for box boundary to ensure clearance
                hover_offset = 0.30 if in_box_boundary else 0.25
                hover_pos = [target_pos[0], target_pos[1], target_pos[2] + hover_offset]
                
                for i, grasp_rot in enumerate(grasp_quats):
                    try:
                        # A. Solve IK for Grasp Pose
                        # Increased max_configs and max_time_ms for better success rate
                        path_configs = self.robot.solve_ik_via_sampling(target_pos, quaternion=grasp_rot, max_configs=20, max_time_ms=300, ignore_collisions=True)
                        if path_configs is None or len(path_configs) == 0: 
                            continue
                        q_grasp = path_configs[0]
                        
                        # B. Solve IK for Hover Pose
                        path_configs_hover = self.robot.solve_ik_via_sampling(hover_pos, quaternion=grasp_rot, max_configs=20, max_time_ms=300, ignore_collisions=True)
                        if path_configs_hover is None or len(path_configs_hover) == 0: 
                            continue
                        q_hover = path_configs_hover[0]

                        # VALIDATE HOVER: Ensure q_hover is collision-free (relaxed - ignore minor collisions)
                        self.set_robot_conf(q_hover)
                        # Skip collision check for now - let linear path planning handle it
                        # if self.robot.check_collision():
                        #     continue
                        
                        # C. Plan Hover -> Grasp (Linear Approach)
                        # Use ignore_collisions=True initially to find feasible paths
                        path2 = self._get_linear_path(q_hover, target_pos, grasp_rot, ignore_collisions=True)
                        if not path2: 
                            continue
                        
                        q_grasp_actual = path2._path_points[-7:].tolist()

                        # D. Plan Grasp -> Hover (Linear Retract/Lift)
                        path3 = self._get_linear_path(q_grasp_actual, hover_pos, grasp_rot, ignore_collisions=True)
                        if not path3: 
                            path2.remove()
                            continue
                        
                        q_hover_end = path3._path_points[-7:].tolist()
                        
                        def get_configs(p):
                            return p._path_points.reshape(-1, 7).tolist()

                        t2 = get_configs(path2)
                        t3 = get_configs(path3)
                        
                        grasp = [0]*7
                        # Return split trajectories for precise execution
                        return grasp, q_hover, q_hover_end, (t2, t3)

                    except Exception:
                        continue

            total_tried = len(valid_depths) * len(grasp_quats)
            print(f"DEBUG: compute_pick_trajectory failed for {obj} at {pose}. Tried {total_tried} configs.")
            raise RuntimeError(f"Could not find valid grasp configuration after {total_tried} attempts")
        finally:
            self.set_robot_conf(original_conf)
            # Restore lid collidability
            if lid_obj:
                lid_obj.set_collidable(lid_collidable_state)

    def compute_place_trajectory(self, obj, pose, region_name=None):
        """Return grasp, q_start, q_end, and trajectory for placing obj at pose (LOWER & RELEASE)."""
        original_conf = self.get_robot_conf()
        try:
            # 1. Determine Strategy based on Region
            is_cupboard = (region_name == 'cupboard_boundary') or (region_name == 'cupboard_boundary_top')
            is_box_boundary = (region_name == 'box_boundary')
            
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            top_z_local = max_z
            
            if is_cupboard:
                # --- HORIZONTAL APPROACH (Front) ---
                # User requirement: "hover at some distance in front o the cupboard.. and then perform the place, where it just goes ahead in the x direction"
                
                # Target Pose (Final Place)
                # Use the sampled pose directly to avoid IK failures and deep placement
                target_pos_place = [pose[0], pose[1], pose[2]]
                
                # Hover Pose (Start/End)
                # "hover at some distance in front"
                # We keep the hover pose back relative to the original pose to ensure clearance
                # User requested strict straight line (horizontal) for cupboard, so removing Z offset
                # Increased hover_dist to 0.40 to give more room for horizontal alignment
                hover_dist = 0.40
                target_pos_hover = [pose[0] - hover_dist, pose[1], pose[2]]
                
                import math
                def quaternion_from_euler(ai, aj, ak):
                    ai /= 2.0
                    aj /= 2.0
                    ak /= 2.0
                    ci = math.cos(ai)
                    si = math.sin(ai)
                    cj = math.cos(aj)
                    sj = math.sin(aj)
                    ck = math.cos(ak)
                    sk = math.sin(ak)
                    cc = ci*ck
                    cs = ci*sk
                    sc = si*ck
                    ss = si*sk
                    q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
                    return q

                grasp_quats = []
                # Try multiple rolls to find one that works (e.g. fingers horizontal vs vertical)
                # Base orientation: Ry=pi/2 (Z points +X)
                base_ry = np.pi/2
                # User requires strict "FACING X DIRECTION" without weird rotations.
                # Restricting to roll=0 ensures the gripper is upright/aligned standardly.
                for roll in [0]:
                     q = quaternion_from_euler(roll, base_ry, 0)
                     grasp_quats.append(q)

            else:
                # --- VERTICAL APPROACH (Top-Down) ---
                # Existing logic
                hover_z = pose[2] + top_z_local + 0.12 # Reduced from 0.18
                place_z = pose[2] + 0.015 
                
                target_pos_hover = [pose[0], pose[1], hover_z]
                target_pos_place = [pose[0], pose[1], place_z]
                
                grasp_quats = []
                import math
                def quaternion_from_euler(ai, aj, ak):
                    ai /= 2.0
                    aj /= 2.0
                    ak /= 2.0
                    ci = math.cos(ai)
                    si = math.sin(ai)
                    cj = math.cos(aj)
                    sj = math.sin(aj)
                    ck = math.cos(ak)
                    sk = math.sin(ak)
                    cc = ci*ck
                    cs = ci*sk
                    sc = si*ck
                    ss = si*sk
                    q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
                    return q

                if is_box_boundary:
                    # User request: aligned with x axis
                    angles = [0, np.pi]
                else:
                    angles = np.linspace(0, 2*np.pi, 16)

                for angle in angles:
                    q = quaternion_from_euler(np.pi, 0, angle)
                    grasp_quats.append(q)
            
            # 3. Solve IK
            for grasp_rot in grasp_quats:
                try:
                    # A. Solve IK for Hover Pose
                    path_configs_hover = self.robot.solve_ik_via_sampling(target_pos_hover, quaternion=grasp_rot, max_configs=1, max_time_ms=50, ignore_collisions=True)
                    if path_configs_hover is None or len(path_configs_hover) == 0: 
                        continue
                    q_hover = path_configs_hover[0]
                    
                    # B. Solve IK for Place Pose
                    path_configs_place = self.robot.solve_ik_via_sampling(target_pos_place, quaternion=grasp_rot, max_configs=1, max_time_ms=50, ignore_collisions=True)
                    if path_configs_place is None or len(path_configs_place) == 0: 
                        continue
                    q_place = path_configs_place[0]
                    
                    # C. Plan Hover -> Place (Linear)
                    path_down = self._get_linear_path(q_hover, target_pos_place, grasp_rot, steps=50)
                    if not path_down: 
                        continue
                    
                    # D. Plan Place -> Hover (Linear Return)
                    path_up = self._get_linear_path(q_place, target_pos_hover, grasp_rot, steps=50)
                    if not path_up:
                        path_down.remove()
                        continue
                        
                    # Extract configs
                    def get_configs(p):
                        return p._path_points.reshape(-1, 7).tolist()

                    t_down = get_configs(path_down)
                    t_up = get_configs(path_up)
                    
                    grasp = [0]*7
                    # Return split trajectories
                    return grasp, q_hover, q_hover, (t_down, t_up)

                except Exception:
                    continue

            raise RuntimeError(f"Could not find valid place configuration for region {region_name}")
        finally:
            self.set_robot_conf(original_conf)

    def compute_hover_config(self, obj, pose, hover_offset=0.12):
        """Return a valid configuration q_hover strictly above the object."""
        original_conf = self.get_robot_conf()
        try:
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            # max_z is the local Z extent from origin
            top_z_local = max_z
            hover_z = pose[2] + top_z_local + hover_offset
            
            target_pos = [pose[0], pose[1], hover_z]
            
            # Sample orientations pointing down
            grasp_quats = []
            import math
            def quaternion_from_euler(ai, aj, ak):
                ai /= 2.0
                aj /= 2.0
                ak /= 2.0
                ci = math.cos(ai)
                si = math.sin(ai)
                cj = math.cos(aj)
                sj = math.sin(aj)
                ck = math.cos(ak)
                sk = math.sin(ak)
                cc = ci*ck
                cs = ci*sk
                sc = si*ck
                ss = si*sk
                q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
                return q

            for angle in np.linspace(0, 2*np.pi, 16):
                q = quaternion_from_euler(np.pi, 0, angle)
                grasp_quats.append(q)
            
            for grasp_rot in grasp_quats:
                # Solve IK for Hover Pose
                # ignore_collisions=True for IK solving, but we check it manually after
                path_configs = self.robot.solve_ik_via_sampling(target_pos, quaternion=grasp_rot, max_configs=1, max_time_ms=50, ignore_collisions=True)
                if path_configs is not None and len(path_configs) > 0:
                    q_hover = path_configs[0]
                    # Check collision
                    self.set_robot_conf(q_hover)
                    if not self.robot.check_collision():
                        return q_hover
            
            raise RuntimeError("Could not find valid hover configuration")
        finally:
            self.set_robot_conf(original_conf)

    def compute_lid_grasp_trajectory(self, lid):
        """Compute trajectory to grasp the box lid."""
        original_conf = self.get_robot_conf()
        try:
            # 1. Get Lid Geometry
            # get_bounding_box returns [min_x, max_x, min_y, max_y, min_z, max_z] in LOCAL frame
            min_x, max_x, min_y, max_y, min_z, max_z = lid.get_bounding_box()
            
            # Get transformation matrix to convert local to world
            m_raw = lid.get_matrix()
            m = np.array(m_raw)
            # Handle flat list vs structured
            if m.ndim == 1:
                if m.size == 12: m = m.reshape(3, 4)
                elif m.size == 16: m = m.reshape(4, 4)
            
            def to_world_point(lx, ly, lz):
                # m is 3x4 or 4x4
                wx = m[0,0]*lx + m[0,1]*ly + m[0,2]*lz + m[0,3]
                wy = m[1,0]*lx + m[1,1]*ly + m[1,2]*lz + m[1,3]
                wz = m[2,0]*lx + m[2,1]*ly + m[2,2]*lz + m[2,3]
                return [wx, wy, wz]

            def to_world_vec(vx, vy, vz):
                # Rotate only
                wx = m[0,0]*vx + m[0,1]*vy + m[0,2]*vz
                wy = m[1,0]*vx + m[1,1]*vy + m[1,2]*vz
                wz = m[2,0]*vx + m[2,1]*vy + m[2,2]*vz
                return [wx, wy, wz]

            # Calculate center in local frame
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            cz = (min_z + max_z) / 2.0
            
            # User Request: Force pick the "lengthier" side.
            len_x = max_x - min_x
            len_y = max_y - min_y
            
            candidates_info = []
            # If X is longer, the faces with normal Y are the long faces (area ~ X*Z)
            if len_x >= len_y:
                candidates_info.append({'name': '-Y Face', 'pt': [cx, min_y, cz], 'app': [0, 1, 0]})
                candidates_info.append({'name': '+Y Face', 'pt': [cx, max_y, cz], 'app': [0, -1, 0]})
            else:
                candidates_info.append({'name': '-X Face', 'pt': [min_x, cy, cz], 'app': [1, 0, 0]})
                candidates_info.append({'name': '+X Face', 'pt': [max_x, cy, cz], 'app': [-1, 0, 0]})
            
            candidates = []
            for c in candidates_info:
                w_pt = to_world_point(*c['pt'])
                w_app = to_world_vec(*c['app'])
                # Normalize approach vector
                norm = np.linalg.norm(w_app)
                w_app = [x/norm for x in w_app]
                
                dist = np.linalg.norm(w_pt) # Distance from robot (0,0,0)
                candidates.append({
                    'name': c['name'],
                    'dist': dist,
                    'grasp_pt': w_pt,
                    'approach': w_app
                })
            
            # Sort by distance to robot
            # DEBUG: Check robot position
            r_pos = self.robot.get_position()
            print(f"DEBUG: Robot Position: {r_pos}")
            
            # Recalculate distances relative to robot
            for c in candidates:
                c['dist'] = np.linalg.norm(np.array(c['grasp_pt']) - np.array(r_pos))

            candidates.sort(key=lambda c: c['dist'])
            
            # Try faces in order of distance
            for best_face in candidates:
                print(f"DEBUG: Testing Face: {best_face['name']} at dist {best_face['dist']:.3f}")
                # print(f"DEBUG: Approach Vector: {best_face['approach']}")
                
                # Refine Grasp Point
                grasp_overlap = 0.015 
                target_pos = list(best_face['grasp_pt'])
                target_pos[0] += best_face['approach'][0] * grasp_overlap
                target_pos[1] += best_face['approach'][1] * grasp_overlap
                target_pos[2] += best_face['approach'][2] * grasp_overlap
                
                # print(f"DEBUG: Target Grasp Point: {target_pos}")

                # Generate Horizontal Grasp Quaternions
                # Z_grip = Approach
                z_axis = np.array(best_face['approach'])
                
                world_z = np.array([0, 0, 1])
                
                # Check if approach is vertical (singularity)
                if abs(np.dot(z_axis, world_z)) > 0.95:
                    y_axis_cand = np.array([1, 0, 0])
                else:
                    y_axis_cand = np.cross(z_axis, world_z)
                    y_axis_cand = y_axis_cand / np.linalg.norm(y_axis_cand)
                
                grasp_quats = []
                
                def mat2quat(M):
                    tr = M[0,0] + M[1,1] + M[2,2]
                    if tr > 0:
                        S = np.sqrt(tr+1.0) * 2
                        qw = 0.25 * S
                        qx = (M[2,1] - M[1,2]) / S
                        qy = (M[0,2] - M[2,0]) / S
                        qz = (M[1,0] - M[0,1]) / S
                    elif (M[0,0] > M[1,1]) and (M[0,0] > M[2,2]):
                        S = np.sqrt(1.0 + M[0,0] - M[1,1] - M[2,2]) * 2
                        qw = (M[2,1] - M[1,2]) / S
                        qx = 0.25 * S
                        qy = (M[0,1] + M[1,0]) / S
                        qz = (M[0,2] + M[2,0]) / S
                    elif M[1,1] > M[2,2]:
                        S = np.sqrt(1.0 + M[1,1] - M[0,0] - M[2,2]) * 2
                        qw = (M[0,2] - M[2,0]) / S
                        qx = (M[0,1] + M[1,0]) / S
                        qy = 0.25 * S
                        qz = (M[1,2] + M[2,1]) / S
                    else:
                        S = np.sqrt(1.0 + M[2,2] - M[0,0] - M[1,1]) * 2
                        qw = (M[1,0] - M[0,1]) / S
                        qx = (M[0,2] + M[2,0]) / S
                        qy = (M[1,2] + M[2,1]) / S
                        qz = 0.25 * S
                    return [qx, qy, qz, qw]

                # Vertical Fingers: X_grip is Horizontal
                x_axes_vertical = [y_axis_cand, -y_axis_cand]
                
                for x_ax in x_axes_vertical:
                    y_ax = np.cross(z_axis, x_ax)
                    y_ax = y_ax / np.linalg.norm(y_ax)
                    x_ax_final = np.cross(y_ax, z_axis)
                    
                    R = np.eye(3)
                    R[:, 0] = x_ax_final
                    R[:, 1] = y_ax
                    R[:, 2] = z_axis
                    grasp_quats.append(mat2quat(R))

                # Hover Position
                hover_dist = 0.08 
                hover_pos = [
                    target_pos[0] - z_axis[0] * hover_dist,
                    target_pos[1] - z_axis[1] * hover_dist,
                    target_pos[2] - z_axis[2] * hover_dist
                ]
                
                # --- NEW LOGIC: Slide to Edge ---
                # Shift grasp point to the edge (negative X) to maximize slide length
                x_shift = (len_x / 2.0) - 0.03 # 0.5cm margin from edge (EXTREME)
                target_pos_edge = list(target_pos)
                target_pos_edge[0] -= x_shift

                # Try IK
                z_offsets = [0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]
                
                for i, grasp_quat in enumerate(grasp_quats):
                    for z_off in z_offsets:
                        test_hover = [hover_pos[0], hover_pos[1], hover_pos[2] + z_off]
                        test_center = [target_pos[0], target_pos[1], target_pos[2] + z_off]
                        test_edge = [target_pos_edge[0], target_pos_edge[1], target_pos_edge[2] + z_off]
                        
                        # A. Solve IK for Hover
                        path_configs_hover = self.robot.solve_ik_via_sampling(test_hover, quaternion=grasp_quat, max_configs=5, max_time_ms=100, ignore_collisions=True)
                        if path_configs_hover is None or len(path_configs_hover) == 0: 
                            continue
                        q_hover = path_configs_hover[0]

                        # VALIDATE HOVER
                        self.set_robot_conf(q_hover)
                        if self.robot.check_collision():
                            continue
                        
                        # B. Solve IK for Center (Approach)
                        path_configs_center = self.robot.solve_ik_via_sampling(test_center, quaternion=grasp_quat, max_configs=5, max_time_ms=100, ignore_collisions=True)
                        if path_configs_center is None or len(path_configs_center) == 0: 
                            continue
                        
                        # C. Solve IK for Edge (Grasp)
                        path_configs_edge = self.robot.solve_ik_via_sampling(test_edge, quaternion=grasp_quat, max_configs=5, max_time_ms=100, ignore_collisions=True)
                        if path_configs_edge is None or len(path_configs_edge) == 0: 
                            continue
                        
                        # D. Plan Hover -> Center
                        path_approach = self._get_linear_path(q_hover, test_center, grasp_quat, ignore_collisions=True)
                        if not path_approach: 
                            continue
                        
                        # E. Plan Center -> Edge
                        q_approach_end = path_approach._path_points[-7:].tolist()
                        path_slide = self._get_linear_path(q_approach_end, test_edge, grasp_quat, ignore_collisions=True)
                        if not path_slide:
                            path_approach.remove()
                            continue

                        # Validate Combined Path
                        traj_points_1 = path_approach._path_points.reshape(-1, 7).tolist()
                        traj_points_2 = path_slide._path_points.reshape(-1, 7).tolist()
                        full_traj = traj_points_1 + traj_points_2
                        
                        valid_path = True
                        for idx, q in enumerate(full_traj[:-5]):
                            self.set_robot_conf(q)
                            if self.robot.check_collision():
                                valid_path = False
                                break
                        
                        if not valid_path:
                            continue

                        q_grasp_actual = full_traj[-1]
                        
                        # Return split trajectories for precise control
                        # traj_points_1: Hover -> Center
                        # traj_points_2: Center -> Edge (Grasp)
                        return grasp_quat, q_hover, q_grasp_actual, (traj_points_1, traj_points_2)
            
            raise RuntimeError(f"Could not find valid lid grasp for any face")
            
        finally:
            self.set_robot_conf(original_conf)

    def _get_matrix_from_pose(self, pos, quat):
        # quat is [x, y, z, w] from PyRep
        # Convert to rotation matrix manually
        x, y, z, w = quat
        # Formula for quat to matrix (assuming normalized)
        # R = ...
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        mat = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
        ])
        
        T = np.eye(4)
        T[:3, :3] = mat
        T[:3, 3] = pos
        return T

    def compute_open_lid_trajectory(self, lid):
        """
        Computes the full sequence: Hover -> Grasp -> Slide -> Release -> Return -> Retreat to Hover.
        Returns: grasp_quat, q_hover_start, q_hover_end, (traj_approach, traj_slide, traj_return, traj_retreat)
        """
        # 1. Compute Grasp & Approach
        # compute_lid_grasp_trajectory returns: grasp_quat, q_hover, q_grasp, (traj_hover_to_center, traj_center_to_edge)
        grasp_quat, q_hover, q_grasp, (traj_hover_to_center, traj_center_to_edge) = self.compute_lid_grasp_trajectory(lid)
        
        # Combine for full approach
        traj_approach_full = traj_hover_to_center + traj_center_to_edge
        
        # 2. Compute Slide & Return
        # compute_slide_lid_trajectory returns: q_start, q_end, traj_slide, traj_return
        # We pass q_grasp as initial_conf
        q_slide_start, q_slide_end, traj_slide, traj_return = self.compute_slide_lid_trajectory(lid, grasp_quat, initial_conf=q_grasp)
        
        # 3. Compute Retreat (Center -> Hover)
        # After traj_return, we are at Center (because we modified compute_slide_lid_trajectory to return to Center).
        # traj_hover_to_center is Hover -> Center.
        # So reverse is Center -> Hover.
        traj_retreat = traj_hover_to_center[::-1]
        
        return grasp_quat, q_hover, q_hover, (traj_approach_full, traj_slide, traj_return, traj_retreat)

    def compute_slide_lid_trajectory(self, obj, grasp_quat, initial_conf=None):
        """
        Computes a trajectory to slide the lid open in the X direction.
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

        # 2. Determine Slide Distance
        # Use the maximum dimension of the lid to ensure we slide enough
        min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
        lid_len = max(max_x - min_x, max_y - min_y)
        
        # Try decreasing distances if full slide fails
        # User requested to extend as much as possible (at least twice the distance)
        # Range 50 down to 4 corresponds to 2.5x down to 0.2x
        fractions = [x * 0.05 for x in range(50, 3, -1)]
        
        for frac in fractions:
            slide_dist = frac * lid_len
            slide_vec = np.array([slide_dist, 0, 0])
            target_pos = start_pos + slide_vec
            
            print(f"DEBUG: Attempting Slide Lid by {slide_dist:.3f}m ({frac*100}%) in +X direction (Lid Len: {lid_len:.3f})")
            
            self.set_robot_conf(initial_conf)
            try:
                # steps=50 for smoother motion (was 15)
                path = self.robot.get_linear_path(position=target_pos, quaternion=start_quat, steps=50, ignore_collisions=True)
                
                if path:
                    traj_configs = path._path_points.reshape(-1, 7).tolist()
                    
                    # --- NEW RETURN LOGIC: Return to Center ---
                    # User Request: "return position to be... where the horizontal movement of robot is (pre-sliding motion)... like when it aligns itself to the lid face"
                    # This corresponds to the "Center" position in compute_lid_grasp_trajectory.
                    # In compute_lid_grasp_trajectory, Center = Edge + x_shift
                    # where x_shift = (len_x / 2.0) - 0.03
                    
                    x_shift = (lid_len / 2.0) - 0.03
                    return_pos = start_pos + np.array([x_shift, 0, 0])
                    
                    # Plan Return Path (Linear)
                    # We are at target_pos (end of slide). We go to return_pos.
                    # We need the config at target_pos to start the return path
                    q_slide_end = traj_configs[-1]
                    self.set_robot_conf(q_slide_end)
                    
                    path_return = self.robot.get_linear_path(position=return_pos, quaternion=start_quat, steps=30, ignore_collisions=True) # Increased to 30
                    
                    if path_return:
                        traj_return = path_return._path_points.reshape(-1, 7).tolist()
                        print(f"Slide Trajectory Computed with {len(traj_configs)} waypoints. Return path found.")
                        return traj_configs[0], traj_configs[-1], traj_configs, traj_return
                    else:
                        print("DEBUG: Return path planning failed. Using reverse slide as fallback.")
                        traj_return = traj_configs[::-1]
                        return traj_configs[0], traj_configs[-1], traj_configs, traj_return

            except Exception as e:
                print(f"DEBUG: Slide failed for fraction {frac}: {e}")
                continue

        print("DEBUG: Could not find valid slide trajectory for any fraction.")
        return initial_conf, initial_conf, [], []

    def get_camera_frames(self):
        # Capture RGB from all cameras
        frames = {}
        for name, cam in self.cams.items():
            cam.handle_explicitly() # Trigger rendering
            rgb = cam.capture_rgb()
            # PyRep returns [0,1], convert to [0,255] uint8
            rgb = (rgb * 255).astype(np.uint8)
            frames[name] = rgb
        return frames

    def find_best_placement(self, obj, region_name, count=50):
        """Find a collision-free placement in region. Returns the first valid one found."""
        region = self.regions.get(region_name)
        if not region:
            raise ValueError(f"Region {region_name} not found")
            
        # Region bounds (Local)
        r_min_x, r_max_x, r_min_y, r_max_y, r_min_z, r_max_z = region.get_bounding_box()
        
        # Region Position (World)
        rx, ry, rz = region.get_position()
        
        # World Bounds
        world_min_x = rx + r_min_x
        world_max_x = rx + r_max_x
        world_min_y = ry + r_min_y
        world_max_y = ry + r_max_y
        
        # Search range (with some padding)
        padding = 0.05
        search_min_x = world_min_x + padding
        search_max_x = world_max_x - padding
        search_min_y = world_min_y + padding
        search_max_y = world_max_y - padding
        
        # Z height: Place on table surface if possible
        table = self.regions.get('table')
        if table:
             _, _, _, _, _, table_max_z = table.get_bounding_box()
             place_z = table_max_z + 0.005
        else:
             place_z = r_max_z + 0.005
        
        original_pose = list(obj.get_pose())
        
        # Minimum distance between placed objects
        MIN_PLACEMENT_DIST = 0.12  # 12cm apart
        
        # Try random positions
        for _ in range(count):
            sample_x = np.random.uniform(search_min_x, search_max_x)
            sample_y = np.random.uniform(search_min_y, search_max_y)
            
            # Check if too close to previously placed objects
            too_close = False
            for placed_pos in self.placed_positions:
                dist = np.sqrt((sample_x - placed_pos[0])**2 + (sample_y - placed_pos[1])**2)
                if dist < MIN_PLACEMENT_DIST:
                    too_close = True
                    break
            if too_close:
                continue
            
            # Candidate pose (keep original orientation for the object)
            candidate_pose = [sample_x, sample_y, place_z] + original_pose[3:]
            
            # 1. Check Collision
            obj.set_pose(candidate_pose)
            if obj.check_collision():
                continue
                
            # 2. Check Hover Reachability (Downward gripper)
            try:
                # We just need to know if a hover config EXISTS for this spot
                self.compute_hover_config(obj, candidate_pose, hover_offset=0.18)
                # If successful, this is valid
                obj.set_pose(original_pose)
                # Track this position
                self.placed_positions.append([sample_x, sample_y, place_z])
                return candidate_pose
            except Exception:
                continue
        
        # Restore
        obj.set_pose(original_pose)
        raise RuntimeError("Could not find a valid placement in region after multiple attempts")
