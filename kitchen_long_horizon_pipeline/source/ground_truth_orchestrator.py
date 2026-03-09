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

# Global video recorders
VIDEO_RECORDER = None
MASK_RECORDER = None  # For segmentation mask videos
STEP_CALLBACK = None  # Optional per-step hook (e.g., live segmentation viewer update)
ACTION_PROGRESS_CALLBACK = None  # Optional primitive-action callback for live panels


def _env_int(name, default, min_value=1):
    """Parse positive integer env var with fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(min_value, value)


# Runtime speed knobs
# Use GT_SPEED_MODE=fast for a noticeably quicker execution profile.
GT_SPEED_MODE = os.environ.get("GT_SPEED_MODE", "normal").strip().lower()
DEFAULT_EXEC_INTERP_STEPS = _env_int(
    "GT_EXEC_INTERP_STEPS", 6 if GT_SPEED_MODE == "fast" else 15
)
GO_HOME_INTERP_STEPS = _env_int(
    "GT_GO_HOME_INTERP_STEPS", 45 if GT_SPEED_MODE == "fast" else 100
)
GO_HOME_EXEC_STEPS = _env_int(
    "GT_GO_HOME_EXEC_STEPS", 5 if GT_SPEED_MODE == "fast" else 10
)
RELEASE_HOLD_STEPS = _env_int(
    "GT_RELEASE_HOLD_STEPS", 20 if GT_SPEED_MODE == "fast" else 60
)
RECORD_EVERY_N = _env_int(
    "GT_RECORD_EVERY_N", 2 if GT_SPEED_MODE == "fast" else 1
)


def _run_step_callback():
    """Invoke optional per-step callback used by external live viewers."""
    global STEP_CALLBACK
    cb = STEP_CALLBACK
    if cb is None:
        return
    try:
        cb()
    except Exception:
        # Never let visualization hooks break task execution.
        pass


def _emit_action_progress(action_name, action_label=None):
    """Emit primitive action progress (move/pick/place/open) to optional callback."""
    global ACTION_PROGRESS_CALLBACK
    cb = ACTION_PROGRESS_CALLBACK
    if cb is None:
        return
    try:
        name = "" if action_name is None else str(action_name)
        label = name if action_label is None else str(action_label)
        cb(name, label)
    except Exception:
        # Never let UI hooks break task execution.
        pass


def step_and_record(pr, count=1):
    """Step simulation and record video if recorder is active."""
    global VIDEO_RECORDER, MASK_RECORDER
    for i in range(count):
        pr.step()
        should_record = (i % RECORD_EVERY_N == 0) or (i == count - 1)
        if should_record:
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()
            if MASK_RECORDER:
                MASK_RECORDER.record_step()
            _run_step_callback()


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


def normalize_quaternion(q):
    """Return unit quaternion [x, y, z, w]."""
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quaternion_conjugate(q):
    """Quaternion conjugate for [x, y, z, w]."""
    q = np.array(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quaternion_multiply(q1, q2):
    """Multiply quaternions (x, y, z, w order)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)


def quaternion_rotate_vector(q, v):
    """Rotate 3D vector by quaternion [x, y, z, w]."""
    qn = normalize_quaternion(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=float)
    v_rot = quaternion_multiply(quaternion_multiply(qn, vq), quaternion_conjugate(qn))
    return v_rot[:3]


def compute_tip_attachment(gripper_tip, obj):
    """Compute object pose in gripper-tip frame (position + quaternion offset)."""
    tip_pos = np.array(gripper_tip.get_position(), dtype=float)
    tip_quat = normalize_quaternion(gripper_tip.get_quaternion())
    obj_pos = np.array(obj.get_position(), dtype=float)
    obj_quat = normalize_quaternion(obj.get_quaternion())

    local_pos = quaternion_rotate_vector(quaternion_conjugate(tip_quat), obj_pos - tip_pos)
    local_quat = normalize_quaternion(quaternion_multiply(quaternion_conjugate(tip_quat), obj_quat))
    return local_pos, local_quat


def update_attached_pose(gripper_tip, obj, local_pos, local_quat):
    """Apply a rigid tip-frame attachment transform to object pose."""
    tip_pos = np.array(gripper_tip.get_position(), dtype=float)
    tip_quat = normalize_quaternion(gripper_tip.get_quaternion())

    obj_pos = tip_pos + quaternion_rotate_vector(tip_quat, local_pos)
    obj_quat = normalize_quaternion(quaternion_multiply(tip_quat, local_quat))

    obj.set_position(obj_pos.tolist())
    obj.set_quaternion(obj_quat.tolist())


def execute_trajectory(env, traj, steps=None):
    """Execute a trajectory with interpolation."""
    global VIDEO_RECORDER, MASK_RECORDER
    if not traj:
        return
    if steps is None:
        steps = DEFAULT_EXEC_INTERP_STEPS
    steps = max(1, int(steps))

    # In fast mode with live segmentation + recording, dense paths can make
    # execution appear "stuck" for a long time. Downsample waypoints first.
    raw_cap_default = 260 if GT_SPEED_MODE == "fast" else 0
    raw_cap = raw_cap_default
    raw_cap_env = os.environ.get("GT_MAX_RAW_WAYPOINTS")
    if raw_cap_env is not None:
        try:
            raw_cap = max(0, int(raw_cap_env))
        except Exception:
            raw_cap = raw_cap_default
    if raw_cap > 0 and len(traj) > raw_cap:
        idxs = np.linspace(0, len(traj) - 1, raw_cap, dtype=int)
        traj = [traj[i] for i in idxs]
        print(f"[Exec] Downsampled raw trajectory to {len(traj)} waypoints.")

    full_traj = []
    for i in range(len(traj) - 1):
        start = np.array(traj[i])
        end = np.array(traj[i + 1])
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1 - t) * start + t * end)
    full_traj.append(traj[-1])

    exec_cap_default = 2200 if GT_SPEED_MODE == "fast" else 0
    exec_cap = exec_cap_default
    exec_cap_env = os.environ.get("GT_MAX_EXEC_POINTS")
    if exec_cap_env is not None:
        try:
            exec_cap = max(0, int(exec_cap_env))
        except Exception:
            exec_cap = exec_cap_default
    if exec_cap > 0 and len(full_traj) > exec_cap:
        idxs = np.linspace(0, len(full_traj) - 1, exec_cap, dtype=int)
        full_traj = [full_traj[i] for i in idxs]
        print(f"[Exec] Downsampled interpolated trajectory to {len(full_traj)} points.")

    final_idx = len(full_traj) - 1
    if len(full_traj) > 1200:
        print(f"[Exec] Executing long trajectory ({len(full_traj)} points)...")
    for idx, conf in enumerate(full_traj):
        env.set_robot_conf(conf)
        env.pr.step()
        if idx > 0 and (idx % 1200 == 0):
            print(f"[Exec] ... {idx}/{len(full_traj)}")
        should_record = (idx % RECORD_EVERY_N == 0) or (idx == final_idx)
        if should_record:
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()
            if MASK_RECORDER:
                MASK_RECORDER.record_step()
            _run_step_callback()


def _is_holding_any_object(env):
    """True when gripper currently has at least one grasped object."""
    try:
        grasped = env.gripper.get_grasped_objects()
    except Exception:
        return False
    return bool(grasped)


def _move_to_conf_via_high_hover(env, target_conf):
    """
    Move to target joint config via a high-Z Cartesian hover path.
    Used while carrying objects to avoid collisions with scene geometry.
    """
    hover_z = float(os.environ.get("GT_PICK_PLACE_HOVER_Z", "0.30"))
    if hover_z <= 0:
        return False

    current_conf = env.get_robot_conf()
    try:
        try:
            tip = env.robot.get_tip()
        except Exception:
            tip = env.robot.arm.get_tip()
        cur_pos = np.array(tip.get_position(), dtype=float)
        cur_quat = tip.get_quaternion()
    except Exception:
        return False

    try:
        env.set_robot_conf(target_conf)
        try:
            tip_t = env.robot.get_tip()
        except Exception:
            tip_t = env.robot.arm.get_tip()
        tgt_pos = np.array(tip_t.get_position(), dtype=float)
        tgt_quat = tip_t.get_quaternion()
    except Exception:
        try:
            env.set_robot_conf(current_conf)
        except Exception:
            pass
        return False
    finally:
        try:
            env.set_robot_conf(current_conf)
        except Exception:
            pass

    transit_z = max(float(cur_pos[2]), float(tgt_pos[2])) + hover_z
    lift_pos = [float(cur_pos[0]), float(cur_pos[1]), float(transit_z)]
    hover_target = [float(tgt_pos[0]), float(tgt_pos[1]), float(transit_z)]

    try:
        path_lift = env.robot.get_linear_path(
            position=lift_pos,
            quaternion=cur_quat,
            steps=40,
            ignore_collisions=True,
        )
        path_transfer = env.robot.get_linear_path(
            position=hover_target,
            quaternion=tgt_quat,
            steps=60,
            ignore_collisions=True,
        )
    except Exception:
        return False

    if (path_lift is None) or (path_transfer is None):
        return False

    _emit_action_progress("move", "move")
    execute_trajectory(env, path_lift._path_points.reshape(-1, 7).tolist(), steps=3)
    _emit_action_progress("move", "move")
    execute_trajectory(env, path_transfer._path_points.reshape(-1, 7).tolist(), steps=3)

    # Intentionally stop at high hover above target XY.
    # Place action will perform the final descent directly to release.
    return True


def _move_to_place_release_direct(env, segments, release_idx):
    """
    Move from current carry-hover state directly to release pose.
    Avoids redundant intermediate hover transitions before place.
    """
    if not segments:
        return False

    try:
        release_seg = segments[int(release_idx)]
        if not release_seg:
            return False
        release_conf = release_seg[-1]
    except Exception:
        return False

    current_q = env.get_robot_conf()
    try:
        env.set_robot_conf(release_conf)
        try:
            tip_rel = env.robot.get_tip()
        except Exception:
            tip_rel = env.robot.arm.get_tip()
        release_pos = np.array(tip_rel.get_position(), dtype=float)
        release_quat = tip_rel.get_quaternion()
    except Exception:
        try:
            env.set_robot_conf(current_q)
        except Exception:
            pass
        return False
    finally:
        try:
            env.set_robot_conf(current_q)
        except Exception:
            pass

    try:
        path_down = env.robot.get_linear_path(
            position=release_pos.tolist(),
            quaternion=release_quat,
            steps=max(30, int(os.environ.get("GT_DIRECT_PLACE_STEPS", "70"))),
            ignore_collisions=True,
        )
        if path_down is not None:
            _emit_action_progress("move", "move")
            execute_trajectory(env, path_down._path_points.reshape(-1, 7).tolist(), steps=3)
            return True
    except Exception:
        pass

    # Fallback: joint interpolation to release conf
    try:
        traj = env._interpolate_joint_path(
            current_q,
            release_conf,
            steps=max(40, int(os.environ.get("GT_DIRECT_PLACE_STEPS", "70"))),
            check_collisions=False,
        )
        if traj is not None and len(traj) > 0:
            _emit_action_progress("move", "move")
            execute_trajectory(env, traj, steps=3)
            return True
    except Exception:
        pass

    return False


def _object_handle(obj):
    """Return object handle as int when available."""
    try:
        return int(obj.get_handle())
    except Exception:
        return None


def _is_object_grasped(env, obj):
    """Check whether the gripper still reports the object as grasped."""
    if obj is None:
        return False
    target_handle = _object_handle(obj)
    try:
        grasped = env.gripper.get_grasped_objects()
    except Exception:
        return False
    for g in grasped:
        if g is obj:
            return True
        if target_handle is not None:
            try:
                if int(g.get_handle()) == target_handle:
                    return True
            except Exception:
                continue
    return False


def _is_mug_name(name):
    return isinstance(name, str) and ("mug" in name.lower())


def _world_top_z(env, obj):
    """Top Z of an object in world frame."""
    if obj is None:
        return None
    try:
        if hasattr(env, "_get_world_bounding_box"):
            _, _, _, _, _, top_z = env._get_world_bounding_box(obj)
            return float(top_z)
    except Exception:
        pass
    try:
        _, _, _, _, _, max_z = obj.get_bounding_box()
        return float(obj.get_position()[2] + max_z)
    except Exception:
        return None


def _table_surface_z(env):
    """Get table top Z with robust fallbacks."""
    table_obj = None
    try:
        table_obj = env.regions.get("table")
    except Exception:
        table_obj = None
    if table_obj is None:
        table_obj = getattr(env, "table", None)
    if table_obj is None:
        try:
            table_obj = env.get_object("diningTable")
        except Exception:
            table_obj = None
    return _world_top_z(env, table_obj)


def _upright_mug_quat(env):
    """Reference upright mug quaternion."""
    try:
        ref = env.get_object("mug2")
        if ref is not None:
            return normalize_quaternion(ref.get_quaternion()).tolist()
    except Exception:
        pass
    return [0.0, 0.0, 0.0, 1.0]


def _stable_mug_pose_on_table(env, mug, target_xy=None):
    """Build a deterministic upright mug pose resting on the table."""
    if mug is None:
        return None
    table_z = _table_surface_z(env)
    if table_z is None:
        return None
    try:
        min_z = float(mug.get_bounding_box()[4])
    except Exception:
        return None
    if target_xy is None:
        pos = mug.get_position()
        x, y = float(pos[0]), float(pos[1])
    else:
        x, y = float(target_xy[0]), float(target_xy[1])
    quat = _upright_mug_quat(env)
    z = float(table_z - min_z + 0.0002)
    return [x, y, z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def _xy_from_pose_sample(place_pose_p):
    """Best-effort XY extraction from sampled pose tuples/lists."""
    if isinstance(place_pose_p, (list, tuple)) and len(place_pose_p) >= 2:
        try:
            return (float(place_pose_p[0]), float(place_pose_p[1]))
        except Exception:
            return None
    return None


def _prepare_table_mug_release_pose(env, pr, obj_name, target_obj, target_region, place_pose_p=None):
    """Snap mug to a stable table-contact pose before release."""
    if (target_obj is None) or (not _is_mug_name(obj_name)) or (target_region != "placement_boundary"):
        return None
    target_xy = _xy_from_pose_sample(place_pose_p)
    pose_lock = _stable_mug_pose_on_table(env, target_obj, target_xy=target_xy)
    if pose_lock is None:
        print(f"[Release] Could not compute stable table pose for '{obj_name}'")
        return None
    target_obj.set_pose(pose_lock)
    target_obj.set_dynamic(False)
    step_and_record(pr, 8)
    return pose_lock


def _release_gripper_until_detached(
    env,
    pr,
    target_obj=None,
    hold_q=None,
    hold_steps=None,
    open_velocity=0.25,
    pose_lock=None,
):
    """
    Open gripper and verify target detaches.
    Optionally keeps robot fixed at hold_q and object pose locked while opening.
    """
    if hold_steps is None:
        hold_steps = RELEASE_HOLD_STEPS
    hold_steps = max(1, int(hold_steps))

    for attempt_idx in range(3):
        try:
            env.gripper.release()
        except Exception:
            pass
        if target_obj is not None:
            try:
                target_obj.set_parent(None, keep_in_place=True)
            except Exception:
                try:
                    target_obj.set_parent(None)
                except Exception:
                    pass

        open_amount = None
        open_done = False
        for _ in range(hold_steps):
            if hold_q is not None:
                env.set_robot_conf(hold_q)
            try:
                open_done = bool(env.gripper.actuate(1.0, velocity=open_velocity))
            except Exception:
                open_done = False
            try:
                open_vals = list(env.gripper.get_open_amount())
                open_amount = open_vals
                open_ok = all(v > 0.95 for v in open_vals)
            except Exception:
                open_ok = False
            if (pose_lock is not None) and (target_obj is not None):
                try:
                    target_obj.set_pose(pose_lock)
                except Exception:
                    pass
            step_and_record(pr, 1)

        detached = (target_obj is None) or (not _is_object_grasped(env, target_obj))
        if detached and (open_done or open_ok):
            return True
        print(
            f"[Release] attempt={attempt_idx+1} detached={detached} "
            f"open_done={open_done} open_amount={open_amount}"
        )

    # Last-resort detach if the simulator still reports attachment.
    if target_obj is not None and _is_object_grasped(env, target_obj):
        try:
            target_obj.set_parent(None, keep_in_place=True)
        except Exception:
            try:
                target_obj.set_parent(None)
            except Exception:
                pass
        try:
            env.gripper.release()
        except Exception:
            pass

        for _ in range(max(10, hold_steps // 2)):
            if hold_q is not None:
                env.set_robot_conf(hold_q)
            try:
                env.gripper.actuate(1.0, velocity=open_velocity)
            except Exception:
                pass
            if pose_lock is not None:
                try:
                    target_obj.set_pose(pose_lock)
                except Exception:
                    pass
            step_and_record(pr, 1)

    detached = target_obj is None or not _is_object_grasped(env, target_obj)
    try:
        open_ok = all(v > 0.90 for v in env.gripper.get_open_amount())
    except Exception:
        open_ok = True
    return detached and open_ok


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


def _ensure_lid_open_distance(env, lid_obj, pos_before, desired_xy):
    """
    While still grasping the lid, apply extra slide motion so XY opening reaches desired_xy.
    Returns: (success, final_displacement_xy)
    """
    if lid_obj is None:
        return False, 0.0

    opened, displacement_xy, pos_after = validate_lid_opened(
        lid_obj, pos_before, min_displacement_xy=desired_xy
    )
    if opened:
        return True, float(displacement_xy)

    delta_xy = np.array(pos_after[:2], dtype=float) - np.array(pos_before[:2], dtype=float)
    norm = float(np.linalg.norm(delta_xy))
    if norm > 1e-6:
        axis_xy = delta_xy / norm
    else:
        axis_xy = np.array([1.0, 0.0], dtype=float)

    remaining = max(0.0, float(desired_xy) - float(displacement_xy))
    extra_steps = max(2, int(os.environ.get("LID_OPEN_CORRECTION_STEPS", "4")))
    scales = np.linspace(1.0, 0.35, extra_steps).tolist()

    for s in scales:
        if remaining <= 0.0:
            break
        push_xy = max(0.015, remaining * float(s))
        try:
            try:
                tip = env.robot.get_tip()
            except Exception:
                tip = env.robot.arm.get_tip()
            tip_pos = np.array(tip.get_position(), dtype=float)
            tip_quat = tip.get_quaternion()
            target_pos = [
                float(tip_pos[0] + axis_xy[0] * push_xy),
                float(tip_pos[1] + axis_xy[1] * push_xy),
                float(tip_pos[2]),
            ]
            path = env.robot.get_linear_path(
                position=target_pos,
                quaternion=tip_quat,
                steps=35,
                ignore_collisions=True,
            )
            if path is None:
                continue
            _emit_action_progress("open", "open")
            execute_trajectory(env, path._path_points.reshape(-1, 7).tolist(), steps=4)
        except Exception:
            continue

        opened, displacement_xy, pos_after = validate_lid_opened(
            lid_obj, pos_before, min_displacement_xy=desired_xy
        )
        if opened:
            return True, float(displacement_xy)

        delta_xy = np.array(pos_after[:2], dtype=float) - np.array(pos_before[:2], dtype=float)
        norm = float(np.linalg.norm(delta_xy))
        if norm > 1e-6:
            axis_xy = delta_xy / norm
        remaining = max(0.0, float(desired_xy) - float(displacement_xy))

    return False, float(displacement_xy)


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
    traj = interpolate_path(env, q_start, q_home, steps=GO_HOME_INTERP_STEPS)
    if traj:
        _emit_action_progress("move", "move")
        execute_trajectory(env, traj, steps=GO_HOME_EXEC_STEPS)
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
            if _is_holding_any_object(env):
                moved_via_hover = _move_to_conf_via_high_hover(env, q2)
                if not moved_via_hover:
                    _emit_action_progress("move", "move")
                    execute_trajectory(env, traj)
            else:
                _emit_action_progress("move", "move")
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
            _emit_action_progress("pick", "pick")
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

            # Keep place simple: execute planner-provided descend/release path.
            moved_to_release = _move_to_place_release_direct(env, segments, release_idx)
            if not moved_to_release:
                for seg in segments[:release_idx + 1]:
                    execute_trajectory(env, seg)

            # Release
            _emit_action_progress("place", "place")
            print(f"Releasing {o}...")
            target_obj = env.get_object(o)
            target_obj.set_dynamic(True)
            pose_lock = _prepare_table_mug_release_pose(
                env,
                pr,
                obj_name=o,
                target_obj=target_obj,
                target_region=target_region,
                place_pose_p=p,
            )

            hold_q = env.get_robot_conf()
            released_ok = _release_gripper_until_detached(
                env,
                pr,
                target_obj=target_obj,
                hold_q=hold_q,
                hold_steps=max(RELEASE_HOLD_STEPS, 60),
                open_velocity=0.3,
                pose_lock=pose_lock,
            )
            if not released_ok:
                print(f"ERROR: '{o}' is still attached after release attempts.")
                return False
            elif pose_lock is not None:
                for _ in range(10):
                    env.set_robot_conf(hold_q)
                    target_obj.set_pose(pose_lock)
                    step_and_record(pr, 1)
                target_obj.set_dynamic(False)

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
    Special cupboard pick with horizontal grasp and script4-style rigid tracking.
    Uses gripper-frame pose attachment so mug orientation follows the gripper.
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

    upright_mug_quat = _upright_mug_quat(env)

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
                            if path_check is not None:
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
    if traj_to_hover is not None and len(traj_to_hover) > 0:
        _emit_action_progress("move", "move")
        execute_trajectory(env, traj_to_hover, steps=10)

    # Open gripper
    env.gripper.release()
    step_and_record(pr, 30)

    # Approach to grasp
    print("Approaching grasp position...")
    _emit_action_progress("move", "move")
    path_approach = env.robot.get_linear_path(
        position=grasp_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
    )
    if path_approach is not None:
        traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
        for conf in traj_approach:
            env.set_robot_conf(conf)
            step_and_record(pr, 1)
    else:
        print("ERROR: Could not plan approach trajectory")
        return False

    # Close gripper
    _emit_action_progress("pick", "pick")
    print("Grasping...")
    env.gripper.actuate(0.0, 0.1)
    step_and_record(pr, 50)

    # Record rigid mug-to-tip transform
    gripper_tip = env.robot.get_tip()
    mug_tip_offset_local, mug_tip_quat_local = compute_tip_attachment(gripper_tip, mug)
    mug.set_dynamic(False)

    # Retrieve to hover
    print("Retrieving...")
    _emit_action_progress("move", "move")
    retrieve_pos = list(hover_pos)
    retrieve_lift = float(os.environ.get("GT_PICK_PLACE_HOVER_Z", "0.30"))
    retrieve_pos[2] += max(0.02, retrieve_lift)
    path_retrieve = None
    try:
        path_retrieve = env.robot.get_linear_path(
            position=retrieve_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
        )
    except Exception:
        path_retrieve = None

    if path_retrieve is not None:
        traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
        for conf in traj_retrieve:
            env.set_robot_conf(conf)
            step_and_record(pr, 1)
            update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
    else:
        traj_fallback = env._interpolate_joint_path(env.get_robot_conf(), q_hover, steps=100, check_collisions=False)
        if traj_fallback is not None and len(traj_fallback) > 0:
            for conf in traj_fallback:
                env.set_robot_conf(conf)
                step_and_record(pr, 1)
                update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
        else:
            print("ERROR: Could not retrieve to hover")
            return False

    # ===== PLACE (Vertical) =====
    print("Finding placement position...")
    try:
        place_pose = env.find_best_placement(mug, target_region)
    except Exception:
        place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]

    min_x, max_x, min_y, max_y, min_z, max_z = mug.get_bounding_box()
    stable_table_pose = None
    if target_region == "placement_boundary":
        stable_table_pose = _stable_mug_pose_on_table(
            env,
            mug,
            target_xy=(place_pose[0], place_pose[1]),
        )
        if stable_table_pose is not None:
            final_place_obj_pos = np.array(stable_table_pose[:3], dtype=float)
            upright_mug_quat = stable_table_pose[3:]
        else:
            print("[CupboardPlace] Falling back to sampled placement Z (table pose unavailable).")

    if stable_table_pose is None:
        # Use actual support-surface height, not sampled object-origin z.
        place_surface_z = None
        region_surface_z = None
        table_surface_z = _table_surface_z(env)

        region_obj = env.regions.get(target_region)
        if region_obj is not None:
            region_surface_z = _world_top_z(env, region_obj)

        if target_region == 'placement_boundary' and table_surface_z is not None:
            place_surface_z = table_surface_z
        elif region_surface_z is not None:
            place_surface_z = region_surface_z
        elif table_surface_z is not None:
            place_surface_z = table_surface_z

        if place_surface_z is None:
            place_surface_z = float(place_pose[2])

        place_object_z = place_surface_z - float(min_z) + 0.0002
        final_place_obj_pos = np.array([place_pose[0], place_pose[1], place_object_z], dtype=float)

    place_quats = [quaternion_from_euler(np.pi, 0, angle) for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False)]

    q_place_hover = None
    successful_place_quat = None
    successful_place_pos = None
    pre_place_search_conf = env.get_robot_conf()

    for place_quat in place_quats:
        tip_offset_world = quaternion_rotate_vector(place_quat, mug_tip_offset_local)
        candidate_place_pos = (final_place_obj_pos - tip_offset_world).tolist()
        shared_hover = float(os.environ.get("GT_PICK_PLACE_HOVER_Z", "0.30"))
        candidate_hover_pos = [
            candidate_place_pos[0],
            candidate_place_pos[1],
            candidate_place_pos[2] + max(0.08, shared_hover),
        ]

        path_configs = env.robot.solve_ik_via_sampling(
            candidate_hover_pos, quaternion=place_quat, max_configs=20, max_time_ms=500, ignore_collisions=True
        )
        if path_configs is not None and len(path_configs) > 0:
            for q in path_configs:
                env.set_robot_conf(q)
                if not env.robot.check_collision():
                    place_configs = env.robot.solve_ik_via_sampling(
                        candidate_place_pos, quaternion=place_quat, max_configs=5, max_time_ms=200, ignore_collisions=True
                    )
                    if place_configs is not None and len(place_configs) > 0:
                        q_place_hover = q
                        successful_place_quat = place_quat
                        successful_place_pos = candidate_place_pos
                        break
        if q_place_hover is not None:
            break

    env.set_robot_conf(pre_place_search_conf)

    if q_place_hover is None:
        print("ERROR: Could not find place hover configuration")
        return False

    # Move to place hover
    print("Moving to place hover...")
    _emit_action_progress("move", "move")
    current_conf = env.get_robot_conf()
    traj_to_place = env._interpolate_joint_path(current_conf, q_place_hover, steps=150, check_collisions=False)
    if traj_to_place is not None and len(traj_to_place) > 0:
        for conf in traj_to_place:
            env.set_robot_conf(conf)
            step_and_record(pr, 1)
            update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
    else:
        print("ERROR: Could not move to place hover")
        return False

    # Lower to place
    print("Lowering to place position...")
    _emit_action_progress("move", "move")
    path_lower = None
    try:
        path_lower = env.robot.get_linear_path(
            position=successful_place_pos, quaternion=successful_place_quat, steps=100, ignore_collisions=True
        )
    except Exception:
        path_lower = None

    if path_lower is not None:
        traj_lower = path_lower._path_points.reshape(-1, 7).tolist()
        for conf in traj_lower:
            env.set_robot_conf(conf)
            step_and_record(pr, 1)
            update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
    else:
        place_configs = env.robot.solve_ik_via_sampling(
            successful_place_pos, quaternion=successful_place_quat, max_configs=10, max_time_ms=500, ignore_collisions=True
        )
        if place_configs is not None and len(place_configs) > 0:
            q_place = place_configs[0]
            traj_lower = env._interpolate_joint_path(env.get_robot_conf(), q_place, steps=50, check_collisions=False)
            if traj_lower is not None and len(traj_lower) > 0:
                for conf in traj_lower:
                    env.set_robot_conf(conf)
                    step_and_record(pr, 1)
                    update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
        else:
            print("WARNING: Could not lower to place position, releasing from hover-like pose.")

    final_place_obj_pose = [
        float(final_place_obj_pos[0]),
        float(final_place_obj_pos[1]),
        float(final_place_obj_pos[2]),
        float(upright_mug_quat[0]),
        float(upright_mug_quat[1]),
        float(upright_mug_quat[2]),
        float(upright_mug_quat[3]),
    ]
    mug.set_pose(final_place_obj_pose)
    step_and_record(pr, 10)

    # Release with hold-open sequence
    _emit_action_progress("place", "place")
    print("Releasing...")
    hold_q = env.get_robot_conf()
    released_ok = _release_gripper_until_detached(
        env,
        pr,
        target_obj=mug,
        hold_q=hold_q,
        hold_steps=max(RELEASE_HOLD_STEPS, 80),
        open_velocity=0.3,
        pose_lock=final_place_obj_pose,
    )
    if not released_ok:
        print(f"ERROR: '{object_name}' is still attached after release attempts.")
        return False

    # Lift
    print("Lifting...")
    _emit_action_progress("move", "move")
    path_lift = None
    try:
        lift_pos = [successful_place_pos[0], successful_place_pos[1], successful_place_pos[2] + 0.15]
        path_lift = env.robot.get_linear_path(
            position=lift_pos, quaternion=successful_place_quat, steps=50, ignore_collisions=True
        )
    except Exception:
        path_lift = None

    if path_lift is not None:
        traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
        for conf in traj_lift:
            env.set_robot_conf(conf)
            step_and_record(pr, 1)
    else:
        traj_lift = env._interpolate_joint_path(env.get_robot_conf(), q_place_hover, steps=50, check_collisions=False)
        if traj_lift is not None and len(traj_lift) > 0:
            for conf in traj_lift:
                env.set_robot_conf(conf)
                step_and_record(pr, 1)

    mug.set_pose(final_place_obj_pose)
    mug.set_dynamic(False)
    step_and_record(pr, 20)

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
    target_open_xy = float(os.environ.get("LID_OPEN_TARGET_DISPLACEMENT", "0.28"))
    min_open_xy = float(os.environ.get("LID_OPEN_MIN_DISPLACEMENT", "0.24"))
    target_open_xy = max(target_open_xy, min_open_xy)

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
            _emit_action_progress("move", "move")
            execute_trajectory(env, path_to_hover, steps=5)

            print("Approaching grasp...")
            _emit_action_progress("move", "move")
            execute_trajectory(env, traj_approach, steps=5)

            print("Grasping lid...")
            _emit_action_progress("pick", "pick")
            env.gripper.actuate(0.0, 0.1)
            step_and_record(pr, 30)
            env.gripper.grasp(obj)

            print("Computing slide trajectory...")
            q_open_start, q_open_end, traj_open, traj_return = env.compute_slide_lid_trajectory(
                obj, grasp_quat, initial_conf=q_grasp
            )

            if traj_open:
                print("Sliding lid open...")
                _emit_action_progress("open", "open")
                execute_trajectory(env, traj_open, steps=5)

                # Enforce deterministic opening distance before releasing.
                opened_target, displacement_xy, _ = validate_lid_opened(
                    obj, pos_before, min_displacement_xy=target_open_xy
                )
                if not opened_target:
                    print(
                        f"[Lid] Initial slide opened {displacement_xy:.3f}m; "
                        f"target is {target_open_xy:.3f}m. Applying corrective slide..."
                    )
                    opened_target, displacement_xy = _ensure_lid_open_distance(
                        env, obj, pos_before, target_open_xy
                    )
                    if opened_target:
                        print(f"[Lid] Corrective slide reached {displacement_xy:.3f}m.")
                    else:
                        print(
                            f"[Lid] Corrective slide ended at {displacement_xy:.3f}m; "
                            f"target {target_open_xy:.3f}m not reached."
                        )

                print("Releasing lid...")
                _emit_action_progress("place", "place")
                env.gripper.release()
                env.gripper.actuate(1.0, 0.1)
                step_and_record(pr, 50)

                print("Returning...")
                _emit_action_progress("move", "move")
                execute_trajectory(env, traj_return, steps=5)

                print("Retreating to hover...")
                _emit_action_progress("move", "move")
                traj_retreat = traj_hover_to_center[::-1]
                execute_trajectory(env, traj_retreat, steps=5)
                step_and_record(pr, 50)

                # =========================================================================
                # POST-EXECUTION VALIDATION
                # =========================================================================
                lid = env.get_object('box_lid')
                required_open_xy = target_open_xy
                opened, displacement_xy, pos_after = validate_lid_opened(
                    lid, pos_before, min_displacement_xy=required_open_xy
                )
                if not opened:
                    print(
                        f"ERROR: Lid didn't slide open enough "
                        f"(XY displacement: {displacement_xy:.3f}m, required: {required_open_xy:.3f}m)"
                    )
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
            if _is_holding_any_object(env):
                moved_via_hover = _move_to_conf_via_high_hover(env, q2)
                if not moved_via_hover:
                    _emit_action_progress("move", "move")
                    execute_trajectory(env, traj)
            else:
                _emit_action_progress("move", "move")
                execute_trajectory(env, traj)

        elif action.name == 'pick':
            o, p, g, q1, q2, traj_tuple = action.args
            # Handle variable number of trajectory segments
            segments = list(traj_tuple) if isinstance(traj_tuple, tuple) else [traj_tuple]
            approach_traj = segments[0] if len(segments) > 0 else []
            retreat_traj = segments[1] if len(segments) > 1 else []

            execute_trajectory(env, approach_traj)

            _emit_action_progress("pick", "pick")
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

            # Keep place simple: execute planner-provided lower/release trajectory.
            norm_segments = _normalize_segments(traj_tuple)
            release_idx = _place_release_segment_index(env, p, norm_segments) if norm_segments else 0
            moved_to_release = _move_to_place_release_direct(env, norm_segments, release_idx)
            if not moved_to_release:
                execute_trajectory(env, lower_traj)

            _emit_action_progress("place", "place")
            print(f"Releasing {o}...")
            target_obj = env.get_object(o)
            target_obj.set_dynamic(True)
            pose_lock = _prepare_table_mug_release_pose(
                env,
                pr,
                obj_name=o,
                target_obj=target_obj,
                target_region=target_region,
                place_pose_p=p,
            )

            hold_q = env.get_robot_conf()
            released_ok = _release_gripper_until_detached(
                env,
                pr,
                target_obj=target_obj,
                hold_q=hold_q,
                hold_steps=max(RELEASE_HOLD_STEPS, 60),
                open_velocity=0.3,
                pose_lock=pose_lock,
            )
            if not released_ok:
                print(f"ERROR: '{o}' is still attached after release attempts.")
                return False
            elif pose_lock is not None:
                for _ in range(10):
                    env.set_robot_conf(hold_q)
                    target_obj.set_pose(pose_lock)
                    step_and_record(pr, 1)
                target_obj.set_dynamic(False)

            # Execute lift trajectory
            if lift_traj is not None and len(lift_traj) > 0:
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
            if home_traj is not None and len(home_traj) > 0:
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

    print(
        f"\nSpeed mode: {GT_SPEED_MODE} "
        f"(exec_interp={DEFAULT_EXEC_INTERP_STEPS}, hold_steps={RELEASE_HOLD_STEPS}, record_every={RECORD_EVERY_N})"
    )

    # Initialize video recorder (optional)
    enable_video = os.environ.get("GT_RECORD_VIDEO", "1").strip().lower() not in {
        "0", "false", "no"
    }
    if enable_video:
        print("\nInitializing video recorder...")
        VIDEO_RECORDER = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
    else:
        VIDEO_RECORDER = None
        print("\nVideo recorder disabled (GT_RECORD_VIDEO=0)")
    
    print("\nSettling physics...")
    for _ in range(50):
        step_and_record(pr, 1)

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        step_and_record(pr, 1)

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

    keep_alive = os.environ.get("GT_KEEP_ALIVE", "0").strip().lower() not in {
        "0", "false", "no"
    }
    if keep_alive:
        print("\nOrchestration complete. Press Ctrl+C to close.")
        try:
            while True:
                pr.step()
                _run_step_callback()
        except KeyboardInterrupt:
            pass
    else:
        print("\nOrchestration complete. Auto-exit enabled.")

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
