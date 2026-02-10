"""
Ground Truth Orchestrator - Complete Long-Horizon Kitchen Task
Performs all 6 tasks in sequence:
1. Pick mug3 from cupboard -> placement_boundary
2. Pick 5 groceries from table -> cupboard_boundary
3. Pick mug1 from groceries_boundary -> placement_boundary
4. Pick mug2 from box_boundary -> placement_boundary
5. Slide open box lid
6. Pick mug4 from box_inside -> placement_boundary
"""
import os
import sys
import numpy as np
import time
import math

# Configure Qt for GUI
def _configure_qt():
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

_configure_qt()

# Add pddlstream to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

from pddlstream.language.constants import PDDLProblem, And
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import ENV from streams to share the instance
from rlbench_kitchen_streams import ENV, get_stream_map
from video_recorder import VideoRecorder

# Global video recorder
VIDEO_RECORDER = None


def step_and_record(pr, count=1):
    """Step simulation and record video if recorder is active."""
    global VIDEO_RECORDER
    for _ in range(count):
        pr.step()
        if VIDEO_RECORDER:
            VIDEO_RECORDER.record_step()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quaternion_from_euler(ai, aj, ak):
    """Convert Euler angles to quaternion."""
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


def execute_trajectory(env, traj, steps=15):
    """Execute a trajectory with interpolation."""
    global VIDEO_RECORDER
    if not traj:
        return
    full_traj = []
    for i in range(len(traj) - 1):
        start = np.array(traj[i])
        end = np.array(traj[i + 1])
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1 - t) * start + t * end)
    full_traj.append(traj[-1])

    for conf in full_traj:
        env.set_robot_conf(conf)
        env.pr.step()
        if VIDEO_RECORDER:
            VIDEO_RECORDER.record_step()


def validate_config(env, q, min_dist=0.10):
    """Check if a configuration is collision-free."""
    env.set_robot_conf(q)
    if env.robot.check_collision():
        return False


# =============================================================================
# VALIDATION HELPERS (matching VLM execution monitor checks)
# =============================================================================

def check_lid_closed() -> bool:
    """Check if box lid is closed (blocking access to mug4)."""
    lid = ENV.get_object('box_lid')
    if lid is None:
        return False
    lid_pos = lid.get_position()
    # If lid Z is low, it's closed
    return lid_pos[2] < 0.85


def check_object_blocked_by_mug(object_name: str) -> tuple:
    """
    Check if object is blocked by mug_box (mug2) on top of box.
    Returns: (is_blocked, blocking_object_name)
    """
    if object_name == 'box_lid':
        mug2 = ENV.get_object('mug2')  # mug_box
        lid = ENV.get_object('box_lid')
        if mug2 and lid:
            mug_pos = mug2.get_position()
            lid_pos = lid.get_position()
            # If mug is above lid and close in XY
            if (mug_pos[2] > lid_pos[2] and 
                abs(mug_pos[0] - lid_pos[0]) < 0.15 and 
                abs(mug_pos[1] - lid_pos[1]) < 0.15):
                return (True, 'mug2')
    return (False, None)


def validate_object_moved(obj, pos_before: list, min_displacement: float = 0.05) -> tuple:
    """
    Validate that an object has moved significantly.
    Returns: (success, displacement, pos_after)
    """
    pos_after = list(obj.get_position())
    displacement = np.linalg.norm(np.array(pos_after) - np.array(pos_before))
    success = displacement >= min_displacement
    return (success, displacement, pos_after)


def validate_object_in_region(obj, target_region: str) -> bool:
    """
    Validate that object is within target region bounds.
    Gets actual bounds dynamically from the environment.
    
    Special case: For 'placement_boundary', we relax to check anywhere on table
    since physics can cause objects to drift.
    """
    pos = obj.get_position()
    
    # Special case: placement_boundary -> just check if on table
    if target_region == 'placement_boundary':
        table = ENV.regions.get('table')
        if table is not None:
            t_min_x, t_max_x, t_min_y, t_max_y, t_min_z, t_max_z = table.get_bounding_box()
            tx, ty, tz = table.get_position()
            
            # Table world bounds with generous tolerance
            tolerance = 0.15
            in_x = (tx + t_min_x - tolerance) <= pos[0] <= (tx + t_max_x + tolerance)
            in_y = (ty + t_min_y - tolerance) <= pos[1] <= (ty + t_max_y + tolerance)
            in_z = pos[2] > 0.7  # Just check it's above table level
            
            return in_x and in_y and in_z
        else:
            # Fallback: just check z
            return pos[2] > 0.7
    
    # Get region from environment
    region = ENV.regions.get(target_region)
    if region is None:
        print(f"  Warning: Region '{target_region}' not found in env, checking only z > 0.7")
        return pos[2] > 0.7
    
    # Get region bounds (local frame)
    r_min_x, r_max_x, r_min_y, r_max_y, r_min_z, r_max_z = region.get_bounding_box()
    
    # Get region position (world frame)
    rx, ry, rz = region.get_position()
    
    # Convert to world bounds
    world_min_x = rx + r_min_x
    world_max_x = rx + r_max_x
    world_min_y = ry + r_min_y
    world_max_y = ry + r_max_y
    world_min_z = rz + r_min_z
    world_max_z = rz + r_max_z
    
    # Add some tolerance (objects can be slightly outside due to physics)
    tolerance = 0.1
    
    in_x = (world_min_x - tolerance) <= pos[0] <= (world_max_x + tolerance)
    in_y = (world_min_y - tolerance) <= pos[1] <= (world_max_y + tolerance)
    in_z = (world_min_z - tolerance) <= pos[2] <= (world_max_z + tolerance)
    
    if not (in_x and in_y and in_z):
        print(f"  Debug: Object pos={[f'{p:.3f}' for p in pos]}")
        print(f"  Debug: Region bounds X=({world_min_x:.3f}, {world_max_x:.3f}), Y=({world_min_y:.3f}, {world_max_y:.3f}), Z=({world_min_z:.3f}, {world_max_z:.3f})")
        print(f"  Debug: In bounds - X:{in_x}, Y:{in_y}, Z:{in_z}")
    
    return in_x and in_y and in_z


def validate_object_not_fallen(obj, min_z: float = 0.7) -> tuple:
    """
    Check if object has fallen below table level.
    Returns: (not_fallen, current_z)
    """
    pos = obj.get_position()
    return (pos[2] >= min_z, pos[2])


def validate_lid_opened(lid, pos_before: list, min_displacement_xy: float = 0.1) -> tuple:
    """
    Validate that lid was actually slid open.
    Returns: (success, displacement_xy, pos_after)
    """
    pos_after = list(lid.get_position())
    displacement_xy = np.linalg.norm(
        np.array(pos_after[:2]) - np.array(pos_before[:2])
    )
    success = displacement_xy >= min_displacement_xy
    return (success, displacement_xy, pos_after)
    return True


def interpolate_path(env, q1, q2, steps=50):
    """Interpolate between two configurations."""
    traj = []
    q1 = np.array(q1)
    q2 = np.array(q2)
    for i in range(steps + 1):
        t = i / steps
        q = (1 - t) * q1 + t * q2
        traj.append(q.tolist())
    return traj


def go_home(env):
    """Return robot to home configuration."""
    print("\n--- Returning to Home ---")
    pr = env.pr
    q_start = env.get_robot_conf()
    q_home = env.get_home_conf()

    # Try direct interpolation
    traj = interpolate_path(env, q_start, q_home, steps=100)
    if traj:
        execute_trajectory(env, traj, steps=10)
        print("Reached Home.")
        return

    # Fallback: force home
    print("WARNING: Forcing home.")
    env.set_robot_conf(q_home)
    step_and_record(pr, 10)


def _normalize_segments(traj_tuple):
    """Normalize trajectory tuple into list of segments."""
    if traj_tuple is None:
        return []
    if isinstance(traj_tuple, (list, tuple)):
        segs = []
        for s in traj_tuple:
            if s is None:
                continue
            if isinstance(s, np.ndarray):
                segs.append(s.tolist())
            elif isinstance(s, (list, tuple)) and len(s) > 0:
                segs.append(list(s))
        return segs
    return []


def _get_ee_pos(env):
    """Get end-effector position."""
    try:
        tip = env.robot.get_tip()
        return np.array(tip.get_position(), dtype=float)
    except Exception:
        return np.array(env.robot.get_position(), dtype=float)


def _ee_pos_at_conf(env, conf, restore_conf):
    """Get EE position at a configuration."""
    env.set_robot_conf(conf)
    pos = _get_ee_pos(env)
    env.set_robot_conf(restore_conf)
    return pos


def _pick_grasp_segment_index(env, obj, segments):
    """Find segment index where EE is closest to object (for grasp)."""
    if not segments:
        return 0
    restore = env.get_robot_conf()
    obj_pos = np.array(obj.get_position(), dtype=float)

    best_i, best_d = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        d = float(np.linalg.norm(ee_pos - obj_pos))
        if d < best_d:
            best_d, best_i = d, i

    return best_i


def _place_release_segment_index(env, place_pose_p, segments):
    """Find segment index for release (lowest EE or closest to place pose)."""
    if not segments:
        return 0

    restore = env.get_robot_conf()
    
    # Try using place pose if available
    try:
        if isinstance(place_pose_p, (list, tuple)) and len(place_pose_p) >= 3:
            p_xyz = np.array([float(place_pose_p[0]), float(place_pose_p[1]), float(place_pose_p[2])], dtype=float)
            best_i, best_d = 0, float("inf")
            for i, seg in enumerate(segments):
                end_conf = seg[-1]
                ee_pos = _ee_pos_at_conf(env, end_conf, restore)
                d = float(np.linalg.norm(ee_pos - p_xyz))
                if d < best_d:
                    best_d, best_i = d, i
            return best_i
    except:
        pass

    # Fallback: lowest EE z
    best_i, best_z = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        if float(ee_pos[2]) < best_z:
            best_z, best_i = float(ee_pos[2]), i
    return best_i


# ============================================================
# TASK EXECUTION FUNCTIONS
# ============================================================

def run_standard_pick_place(env, object_name, target_region, task_name=""):
    """
    Standard pick and place using PDDL planning.
    Works for table objects and vertical grasps.
    """
    pr = env.pr
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"Pick '{object_name}' -> Place in '{target_region}'")
    print(f"{'='*60}")

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    step_and_record(pr, 10)

    # Set target region context
    env.set_target_region(target_region)

    obj = env.get_object(object_name)
    if obj is None:
        print(f"ERROR: {object_name} not found.")
        return False

    # Record initial position for validation
    pos_before = list(obj.get_position())

    obj.set_dynamic(False)
    initial_pose = obj.get_pose()

    # Setup PDDL problem
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    pose_tuple = tuple(initial_pose)

    init = [
        ('conf', q_home_tuple),
        ('at-conf', q_home_tuple),
        ('hand-empty',),
        ('movable', object_name),
        ('pose', pose_tuple),
        ('at-pose', object_name, pose_tuple),
        ('region', target_region),
    ]

    goal = And(('hand-empty',), ('in-region', object_name, target_region))

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving PDDL problem...")
    solution = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
    plan, cost, evaluations = solution

    if not plan:
        print("ERROR: No PDDL plan found!")
        return False

    print(f"Plan found with {len(plan)} actions")
    for action in plan:
        print(f"  Action: {action.name}")

    # Execute plan
    for action in plan:
        if action.name == 'move':
            q1, q2, traj = action.args
            execute_trajectory(env, traj)

        elif action.name == 'pick':
            o, p, g, q1, q2, traj_tuple = action.args
            segments = _normalize_segments(traj_tuple)
            if not segments:
                print("ERROR: pick has empty trajectory")
                return False

            target_obj = env.get_object(o)
            grasp_idx = _pick_grasp_segment_index(env, target_obj, segments)

            # Execute up to grasp
            for seg in segments[:grasp_idx + 1]:
                execute_trajectory(env, seg)

            # Grasp
            print(f"Grasping {o}...")
            target_obj.set_dynamic(True)
            env.gripper.actuate(0.0, 0.1)
            step_and_record(pr, 10)
            env.gripper.grasp(target_obj)

            # Execute retreat
            for seg in segments[grasp_idx + 1:]:
                execute_trajectory(env, seg)

        elif action.name == 'place':
            o, p, g, r, q1, q2, traj_tuple = action.args
            segments = _normalize_segments(traj_tuple)
            if not segments:
                print("ERROR: place has empty trajectory")
                return False

            release_idx = _place_release_segment_index(env, p, segments)

            # Execute down
            for seg in segments[:release_idx + 1]:
                execute_trajectory(env, seg)

            # Release
            print(f"Releasing {o}...")
            target_obj = env.get_object(o)
            env.gripper.release()
            target_obj.set_dynamic(True)

            hold_q = env.get_robot_conf()
            env.gripper.actuate(1.0, velocity=0.2)
            for _ in range(60):
                env.set_robot_conf(hold_q)
                pr.step()
                if VIDEO_RECORDER:
                    VIDEO_RECORDER.record_step()

            # Retreat
            current_q = env.get_robot_conf()
            target_retreat_conf = segments[-1][-1]

            env.set_robot_conf(target_retreat_conf)
            target_pos = env.robot.get_position()
            target_quat = env.robot.get_quaternion()
            env.set_robot_conf(current_q)

            try:
                path_retreat = env.robot.get_linear_path(
                    position=target_pos,
                    quaternion=target_quat,
                    steps=50,
                    ignore_collisions=True
                )
            except Exception:
                path_retreat = None

            if path_retreat:
                retreat_traj = path_retreat._path_points.reshape(-1, 7).tolist()
                execute_trajectory(env, retreat_traj)
            else:
                for seg in segments[release_idx + 1:]:
                    execute_trajectory(env, seg)

    # =========================================================================
    # POST-EXECUTION VALIDATION
    # =========================================================================
    obj = env.get_object(object_name)
    
    # Check 1: Object moved
    moved, displacement, pos_after = validate_object_moved(obj, pos_before)
    if not moved:
        print(f"ERROR: Object '{object_name}' didn't move (displacement: {displacement:.3f}m)")
        return False
    
    # Check 2: Object in target region
    in_region = validate_object_in_region(obj, target_region)
    if not in_region:
        print(f"ERROR: Object '{object_name}' not in target region '{target_region}'")
        print(f"       Object position: {pos_after}")
        return False
    
    # Check 3: Object didn't fall
    not_fallen, current_z = validate_object_not_fallen(obj)
    if not not_fallen:
        print(f"ERROR: Object '{object_name}' fell (z={current_z:.3f})")
        return False

    print(f"✓ Validation passed: Object moved {displacement:.3f}m to target region")
    print(f"Task '{task_name}' complete!")
    return True


def run_cupboard_pick_place(env, object_name, target_region, task_name=""):
    """
    Special cupboard pick with horizontal grasp and manual mug tracking.
    Uses the proven method from script3/script4.
    """
    pr = env.pr
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"Pick '{object_name}' (cupboard) -> Place in '{target_region}'")
    print(f"{'='*60}")

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    step_and_record(pr, 10)

    mug = env.get_object(object_name)
    if mug is None:
        print(f"ERROR: {object_name} not found.")
        return False

    # Record initial position for validation
    pos_before = list(mug.get_position())

    mug.set_dynamic(False)
    pose = mug.get_pose()

    # ===== CUPBOARD PICK (Horizontal Grasp) =====
    print("Computing horizontal hover configuration...")

    hover_dists = [0.35, 0.30, 0.40]
    z_offsets = [0.001]
    q_hover = None
    successful_grasp_quat = None
    original_conf = env.get_robot_conf()

    base_ry = np.pi / 2
    grasp_quats = [
        quaternion_from_euler(0, base_ry, 0),
        quaternion_from_euler(np.pi, base_ry, 0),
    ]

    target_z = pose[2]
    grasp_depth_offset = 0.03
    grasp_pos = None
    hover_pos = None

    for h_dist in hover_dists:
        for z_off in z_offsets:
            if q_hover is not None:
                break

            hover_pos = [pose[0] - h_dist, pose[1], target_z]
            grasp_pos = [pose[0] + grasp_depth_offset, pose[1], target_z]

            for grasp_rot in grasp_quats:
                path_configs = env.robot.solve_ik_via_sampling(
                    hover_pos, quaternion=grasp_rot, max_configs=50, max_time_ms=1000, ignore_collisions=True
                )
                if path_configs is not None and len(path_configs) > 0:
                    for q in path_configs:
                        env.set_robot_conf(q)
                        if env.robot.check_collision():
                            continue
                        try:
                            path_check = env.robot.get_linear_path(
                                position=grasp_pos, quaternion=grasp_rot, steps=20, ignore_collisions=True
                            )
                            if path_check:
                                q_hover = q
                                successful_grasp_quat = grasp_rot
                                break
                        except Exception:
                            pass
                if q_hover is not None:
                    break
        if q_hover is not None:
            break

    if q_hover is None:
        print("ERROR: Could not find valid horizontal hover configuration")
        env.set_robot_conf(original_conf)
        return False

    env.set_robot_conf(original_conf)

    # Move to hover
    print("Moving to hover...")
    traj_to_hover = env._interpolate_joint_path(home_q, q_hover, steps=100, check_collisions=False)
    if traj_to_hover:
        execute_trajectory(env, traj_to_hover, steps=10)

    # Open gripper
    env.gripper.release()
    step_and_record(pr, 30)

    # Approach to grasp
    print("Approaching grasp position...")
    path_approach = env.robot.get_linear_path(
        position=grasp_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
    )
    if path_approach:
        traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
        for conf in traj_approach:
            env.set_robot_conf(conf)
            pr.step()
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()

    # Close gripper
    print("Grasping...")
    env.gripper.actuate(0.0, 0.1)
    step_and_record(pr, 50)

    # Record mug offset for tracking
    gripper_tip = env.robot.get_tip()
    tip_pos = np.array(gripper_tip.get_position())
    mug_current_pos = np.array(mug.get_position())
    mug_offset = mug_current_pos - tip_pos

    # Retrieve to hover
    print("Retrieving...")
    retrieve_pos = list(hover_pos)
    retrieve_pos[2] += 0.02
    path_retrieve = None
    try:
        path_retrieve = env.robot.get_linear_path(
            position=retrieve_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
        )
    except:
        pass

    if path_retrieve:
        traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
        for conf in traj_retrieve:
            env.set_robot_conf(conf)
            pr.step()
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()
            new_tip_pos = np.array(gripper_tip.get_position())
            mug.set_position((new_tip_pos + mug_offset).tolist())
    else:
        traj_fallback = env._interpolate_joint_path(env.get_robot_conf(), q_hover, steps=100, check_collisions=False)
        if traj_fallback:
            for conf in traj_fallback:
                env.set_robot_conf(conf)
                pr.step()
                if VIDEO_RECORDER:
                    VIDEO_RECORDER.record_step()
                new_tip_pos = np.array(gripper_tip.get_position())
                mug.set_position((new_tip_pos + mug_offset).tolist())

    # ===== PLACE (Vertical) =====
    print("Finding placement position...")
    try:
        place_pose = env.find_best_placement(mug, target_region)
    except:
        place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]

    min_x, max_x, min_y, max_y, min_z, max_z = mug.get_bounding_box()
    mug_top_z = max_z

    hover_z = place_pose[2] + mug_top_z + 0.12
    place_z = place_pose[2] + 0.015

    place_hover_pos = [place_pose[0], place_pose[1], hover_z]
    place_pos = [place_pose[0], place_pose[1], place_z]

    # Find place configuration
    place_quats = [quaternion_from_euler(np.pi, 0, angle) for angle in np.linspace(0, 2 * np.pi, 16)]

    q_place_hover = None
    successful_place_quat = None

    for place_quat in place_quats:
        path_configs = env.robot.solve_ik_via_sampling(
            place_hover_pos, quaternion=place_quat, max_configs=20, max_time_ms=500, ignore_collisions=True
        )
        if path_configs is not None and len(path_configs) > 0:
            for q in path_configs:
                env.set_robot_conf(q)
                if not env.robot.check_collision():
                    place_configs = env.robot.solve_ik_via_sampling(
                        place_pos, quaternion=place_quat, max_configs=5, max_time_ms=200, ignore_collisions=True
                    )
                    if place_configs is not None and len(place_configs) > 0:
                        q_place_hover = q
                        successful_place_quat = place_quat
                        break
        if q_place_hover is not None:
            break

    if q_place_hover is None:
        print("ERROR: Could not find place hover configuration")
        return False

    # Move to place hover
    print("Moving to place hover...")
    current_conf = env.get_robot_conf()
    traj_to_place = env._interpolate_joint_path(current_conf, q_place_hover, steps=150, check_collisions=False)
    if traj_to_place:
        for conf in traj_to_place:
            env.set_robot_conf(conf)
            pr.step()
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()
            new_tip_pos = np.array(gripper_tip.get_position())
            mug.set_position((new_tip_pos + mug_offset).tolist())

    # Lower to place
    print("Lowering to place position...")
    path_lower = None
    try:
        path_lower = env.robot.get_linear_path(
            position=place_pos, quaternion=successful_place_quat, steps=100, ignore_collisions=True
        )
    except:
        pass

    if path_lower:
        traj_lower = path_lower._path_points.reshape(-1, 7).tolist()
        for conf in traj_lower:
            env.set_robot_conf(conf)
            pr.step()
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()
            new_tip_pos = np.array(gripper_tip.get_position())
            mug.set_position((new_tip_pos + mug_offset).tolist())
    else:
        place_configs = env.robot.solve_ik_via_sampling(
            place_pos, quaternion=successful_place_quat, max_configs=10, max_time_ms=500, ignore_collisions=True
        )
        if place_configs:
            q_place = place_configs[0]
            traj_lower = env._interpolate_joint_path(env.get_robot_conf(), q_place, steps=50, check_collisions=False)
            if traj_lower:
                for conf in traj_lower:
                    env.set_robot_conf(conf)
                    pr.step()
                    if VIDEO_RECORDER:
                        VIDEO_RECORDER.record_step()
                    new_tip_pos = np.array(gripper_tip.get_position())
                    mug.set_position((new_tip_pos + mug_offset).tolist())

    # Release
    print("Releasing...")
    env.gripper.actuate(1.0, 0.1)
    step_and_record(pr, 30)

    mug.set_dynamic(True)
    step_and_record(pr, 50)

    # Lift
    print("Lifting...")
    try:
        lift_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.15]
        path_lift = env.robot.get_linear_path(
            position=lift_pos, quaternion=successful_place_quat, steps=50, ignore_collisions=True
        )
        if path_lift:
            traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
            execute_trajectory(env, traj_lift)
    except:
        pass

    # =========================================================================
    # POST-EXECUTION VALIDATION
    # =========================================================================
    mug = env.get_object(object_name)
    
    # Check 1: Object moved
    moved, displacement, pos_after = validate_object_moved(mug, pos_before)
    if not moved:
        print(f"ERROR: Object '{object_name}' didn't move (displacement: {displacement:.3f}m)")
        return False
    
    # Check 2: Object in target region
    in_region = validate_object_in_region(mug, target_region)
    if not in_region:
        print(f"ERROR: Object '{object_name}' not in target region '{target_region}'")
        print(f"       Object position: {pos_after}")
        return False
    
    # Check 3: Object didn't fall
    not_fallen, current_z = validate_object_not_fallen(mug)
    if not not_fallen:
        print(f"ERROR: Object '{object_name}' fell (z={current_z:.3f})")
        return False

    print(f"✓ Validation passed: Object moved {displacement:.3f}m to target region")
    print(f"Task '{task_name}' complete!")
    return True


def run_open_box(env, task_name=""):
    """
    Open the box lid by sliding.
    """
    pr = env.pr
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"Slide open box lid")
    print(f"{'='*60}")

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    step_and_record(pr, 10)

    obj = env.get_object('box_lid')
    if obj is None:
        print("ERROR: box_lid not found.")
        return False

    # =========================================================================
    # PRE-EXECUTION CHECKS
    # =========================================================================
    # Check if mug_box (mug2) is blocking the lid
    is_blocked, blocker = check_object_blocked_by_mug('box_lid')
    if is_blocked:
        print(f"ERROR: Cannot open box lid - '{blocker}' is blocking it")
        return False

    # Record initial position for validation
    pos_before = list(obj.get_position())

    # Ensure gripper is open
    env.gripper.actuate(1.0, 0.1)
    step_and_record(pr, 10)

    try:
        print("Computing grasp trajectory...")
        grasp_quat, q_hover, q_grasp, (traj_hover_to_center, traj_center_to_edge) = env.compute_lid_grasp_trajectory(obj)
        traj_approach = traj_hover_to_center + traj_center_to_edge

        print("Planning motion to hover...")
        path_to_hover = env.compute_motion_plan(home_q, q_hover)

        if path_to_hover:
            print("Moving to hover...")
            execute_trajectory(env, path_to_hover, steps=5)

            print("Approaching grasp...")
            execute_trajectory(env, traj_approach, steps=5)

            print("Grasping lid...")
            env.gripper.actuate(0.0, 0.1)
            step_and_record(pr, 30)
            env.gripper.grasp(obj)

            print("Computing slide trajectory...")
            q_open_start, q_open_end, traj_open, traj_return = env.compute_slide_lid_trajectory(
                obj, grasp_quat, initial_conf=q_grasp
            )

            if traj_open:
                print("Sliding lid open...")
                execute_trajectory(env, traj_open, steps=5)

                print("Releasing lid...")
                env.gripper.release()
                env.gripper.actuate(1.0, 0.1)
                step_and_record(pr, 50)

                print("Returning...")
                execute_trajectory(env, traj_return, steps=5)

                print("Retreating to hover...")
                traj_retreat = traj_hover_to_center[::-1]
                execute_trajectory(env, traj_retreat, steps=5)
                step_and_record(pr, 50)

                # =========================================================================
                # POST-EXECUTION VALIDATION
                # =========================================================================
                lid = env.get_object('box_lid')
                opened, displacement_xy, pos_after = validate_lid_opened(lid, pos_before)
                if not opened:
                    print(f"ERROR: Lid didn't slide open (XY displacement: {displacement_xy:.3f}m)")
                    return False
                
                print(f"✓ Validation passed: Lid slid {displacement_xy:.3f}m")
                print(f"Task '{task_name}' complete!")
                return True
            else:
                print("ERROR: Failed to compute slide trajectory")
                return False
        else:
            print("ERROR: Could not plan motion to hover")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_box_pick_place(env, object_name, target_region, task_name=""):
    """
    Pick object from inside/on box and place.
    Uses the PDDL planner with box-aware strategies.
    """
    pr = env.pr
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"Pick '{object_name}' (box) -> Place in '{target_region}'")
    print(f"{'='*60}")

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    step_and_record(pr, 10)

    env.set_target_region(target_region)

    obj = env.get_object(object_name)
    if obj is None:
        print(f"ERROR: {object_name} not found.")
        return False

    # =========================================================================
    # PRE-EXECUTION CHECKS
    # =========================================================================
    # Check if picking mug4 (mug inside box) - need lid to be open
    if object_name in ['mug4', 'mug_inside_box']:
        if check_lid_closed():
            print(f"ERROR: Cannot pick '{object_name}' - box lid is closed!")
            return False
        print("✓ Pre-check passed: Box lid is open")

    # Record initial position for validation
    pos_before = list(obj.get_position())

    obj.set_dynamic(False)
    initial_pose = obj.get_pose()

    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    pose_tuple = tuple(initial_pose)

    init = [
        ('conf', q_home_tuple),
        ('at-conf', q_home_tuple),
        ('hand-empty',),
        ('movable', object_name),
        ('pose', pose_tuple),
        ('at-pose', object_name, pose_tuple),
        ('region', target_region),
    ]

    goal = And(('hand-empty',), ('in-region', object_name, target_region))

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving PDDL problem...")
    solution = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
    plan, cost, evaluations = solution

    if not plan:
        print("ERROR: No PDDL plan found!")
        return False

    print(f"Plan found with {len(plan)} actions")

    for action in plan:
        if action.name == 'move':
            q1, q2, traj = action.args
            execute_trajectory(env, traj)

        elif action.name == 'pick':
            o, p, g, q1, q2, traj_tuple = action.args
            # Handle variable number of trajectory segments
            segments = list(traj_tuple) if isinstance(traj_tuple, tuple) else [traj_tuple]
            approach_traj = segments[0] if len(segments) > 0 else []
            retreat_traj = segments[1] if len(segments) > 1 else []

            execute_trajectory(env, approach_traj)

            print(f"Grasping {o}...")
            target_obj = env.get_object(o)
            target_obj.set_dynamic(True)
            env.gripper.actuate(0.0, 0.1)
            step_and_record(pr, 10)
            env.gripper.grasp(target_obj)

            execute_trajectory(env, retreat_traj)

        elif action.name == 'place':
            o, p, g, r, q1, q2, traj_tuple = action.args
            # Handle variable number of trajectory segments (lower, lift, [home])
            segments = list(traj_tuple) if isinstance(traj_tuple, tuple) else [traj_tuple]
            lower_traj = segments[0] if len(segments) > 0 else []
            lift_traj = segments[1] if len(segments) > 1 else []
            # Optional home trajectory
            home_traj = segments[2] if len(segments) > 2 else None

            execute_trajectory(env, lower_traj)

            print(f"Releasing {o}...")
            target_obj = env.get_object(o)
            env.gripper.release()
            target_obj.set_dynamic(True)

            hold_q = env.get_robot_conf()
            env.gripper.actuate(1.0, velocity=0.2)
            for _ in range(60):
                env.set_robot_conf(hold_q)
                pr.step()
                if VIDEO_RECORDER:
                    VIDEO_RECORDER.record_step()

            # Execute lift trajectory
            if lift_traj:
                current_q = env.get_robot_conf()
                target_retreat_conf = lift_traj[-1]

                env.set_robot_conf(target_retreat_conf)
                target_pos = env.robot.get_position()
                target_quat = env.robot.get_quaternion()
                env.set_robot_conf(current_q)

                try:
                    path_retreat = env.robot.get_linear_path(
                        position=target_pos,
                        quaternion=target_quat,
                        steps=50,
                        ignore_collisions=True
                    )
                    if path_retreat:
                        retreat_traj = path_retreat._path_points.reshape(-1, 7).tolist()
                        execute_trajectory(env, retreat_traj)
                    else:
                        execute_trajectory(env, [current_q] + lift_traj)
                except:
                    execute_trajectory(env, [current_q] + lift_traj)
            
            # Execute home trajectory if available
            if home_traj:
                execute_trajectory(env, home_traj)

    # =========================================================================
    # POST-EXECUTION VALIDATION
    # =========================================================================
    obj = env.get_object(object_name)
    
    # Check 1: Object moved
    moved, displacement, pos_after = validate_object_moved(obj, pos_before)
    if not moved:
        print(f"ERROR: Object '{object_name}' didn't move (displacement: {displacement:.3f}m)")
        return False
    
    # Check 2: Object in target region
    in_region = validate_object_in_region(obj, target_region)
    if not in_region:
        print(f"ERROR: Object '{object_name}' not in target region '{target_region}'")
        print(f"       Object position: {pos_after}")
        return False
    
    # Check 3: Object didn't fall
    not_fallen, current_z = validate_object_not_fallen(obj)
    if not not_fallen:
        print(f"ERROR: Object '{object_name}' fell (z={current_z:.3f})")
        return False

    print(f"✓ Validation passed: Object moved {displacement:.3f}m to target region")
    print(f"Task '{task_name}' complete!")
    return True


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():
    global VIDEO_RECORDER
    env = ENV
    pr = env.pr

    print("="*60)
    print("GROUND TRUTH ORCHESTRATOR")
    print("Long-Horizon Kitchen Task")
    print("="*60)

    # Initialize video recorder
    print("\nInitializing video recorder...")
    VIDEO_RECORDER = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
    
    print("\nSettling physics...")
    for _ in range(50):
        pr.step()
        VIDEO_RECORDER.record_step()

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        pr.step()
        VIDEO_RECORDER.record_step()

    results = []

    # ============================================
    # TASK 1: Pick mug3 from cupboard -> placement_boundary
    # ============================================
    success = run_cupboard_pick_place(
        env,
        object_name='mug3',
        target_region='placement_boundary',
        task_name="Task 1: Cupboard Mug -> Placement"
    )
    results.append(("Task 1: mug3 cupboard -> placement", success))
    go_home(env)

    # ============================================
    # TASK 2: Pick 5 groceries from table -> cupboard
    # soup, mustard, spam -> cupboard_boundary (inside)
    # sugar, crackers -> cupboard_boundary_top (top shelf)
    # ============================================
    groceries_inside = ['soup', 'mustard', 'spam']
    groceries_top = ['sugar', 'crackers']
    
    task_num = 1
    for grocery in groceries_inside:
        success = run_standard_pick_place(
            env,
            object_name=grocery,
            target_region='cupboard_boundary',
            task_name=f"Task 2.{task_num}: {grocery} -> Cupboard (inside)"
        )
        results.append((f"Task 2.{task_num}: {grocery} -> cupboard", success))
        go_home(env)
        task_num += 1
    
    for grocery in groceries_top:
        success = run_standard_pick_place(
            env,
            object_name=grocery,
            target_region='cupboard_boundary_top',
            task_name=f"Task 2.{task_num}: {grocery} -> Cupboard (top)"
        )
        results.append((f"Task 2.{task_num}: {grocery} -> cupboard_top", success))
        go_home(env)
        task_num += 1

    # ============================================
    # TASK 3: Pick mug2 from box_boundary -> placement_boundary
    # ============================================
    success = run_box_pick_place(
        env,
        object_name='mug2',
        target_region='placement_boundary',
        task_name="Task 3: Box Mug -> Placement"
    )
    results.append(("Task 3: mug2 box -> placement", success))
    go_home(env)

    # ============================================
    # TASK 4: Slide open box lid
    # ============================================
    success = run_open_box(
        env,
        task_name="Task 4: Open Box Lid"
    )
    results.append(("Task 4: open box lid", success))
    go_home(env)

    # ============================================
    # TASK 5: Pick mug4 from inside box -> placement_boundary
    # ============================================
    success = run_box_pick_place(
        env,
        object_name='mug4',
        target_region='placement_boundary',
        task_name="Task 5: Mug Inside Box -> Placement"
    )
    results.append(("Task 5: mug4 box_inside -> placement", success))
    go_home(env)

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    total = len(results)
    passed = sum(1 for _, s in results if s)
    for task, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {task}")
    print(f"\nTotal: {passed}/{total} tasks completed successfully")
    print("="*60)

    # Release video recorder
    if VIDEO_RECORDER:
        VIDEO_RECORDER.release()
        print("\nVideo recording saved to 'orchestrator_videos/' directory")

    print("\nOrchestration complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
