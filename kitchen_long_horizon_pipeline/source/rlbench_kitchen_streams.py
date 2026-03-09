# rlbench_kitchen_streams.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

from pddlstream.language.generator import from_gen_fn
from pddlstream.utils import INF
from rlbench_kitchen_env_constrained import RLBenchKitchenEnvConstrained as RLBenchKitchenEnv

# One global env instance for planning
# Set headless=False to see the simulation window
# Check env var HEADLESS, default to True
headless_mode = os.environ.get("HEADLESS", "True") == "True"
ENV = RLBenchKitchenEnv(headless=headless_mode)

# ----- python functions used by streams -----

def _iter_forced_stable_poses(obj_name, region_name):
    """
    Yield forced stable poses injected by orchestrators.
    Supported keys in ENV._stable_pose_overrides:
      - (obj_name, region_name)
      - (obj_name, None)
      - (None, region_name)
      - obj_name
    Values can be a single pose (len>=7) or list of poses.
    """
    overrides = getattr(ENV, "_stable_pose_overrides", None)
    if not isinstance(overrides, dict):
        return []

    keys = [
        (obj_name, region_name),
        (obj_name, None),
        (None, region_name),
        obj_name,
    ]

    forced = []
    for key in keys:
        if key not in overrides:
            continue
        value = overrides[key]
        if value is None:
            continue

        # Single pose
        if isinstance(value, (list, tuple)) and len(value) >= 7 and not (
            isinstance(value[0], (list, tuple))
        ):
            forced.append(tuple(float(v) for v in value[:7]))
            continue

        # List of poses
        if isinstance(value, (list, tuple)):
            for pose in value:
                if isinstance(pose, (list, tuple)) and len(pose) >= 7:
                    forced.append(tuple(float(v) for v in pose[:7]))

    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for pose in forced:
        if pose in seen:
            continue
        seen.add(pose)
        deduped.append(pose)
    return deduped

def fn_sample_stable_pose(o, r):
    obj = ENV.get_object(o)
    if not obj: return

    # First emit any orchestrator-specified candidates (e.g., box slots).
    for forced_pose in _iter_forced_stable_poses(o, r):
        yield (forced_pose,)

    # Yield multiple samples to help the planner find a reachable one
    for _ in range(50):
        pose = ENV.sample_stable_pose(obj, r)
        yield (tuple(pose),)   # outputs: (?p)

def fn_sample_pick_kin(o, p):
    obj = ENV.get_object(o)
    if not obj: return

    try:
        grasp, q1, q2, traj_tuple = ENV.compute_pick_trajectory(obj, list(p))
        # traj_tuple is (t_approach, t_retreat)
        # We pass it as a single tuple of tuples to PDDLStream, which treats it as one object ?t
        yield (tuple(grasp), tuple(q1), tuple(q2), traj_tuple)
    except Exception as e:
        # print(f"DEBUG: sample-pick-kin failed with error: {e}")
        # import traceback
        # traceback.print_exc()
        # If IK/Planning fails, we just don't yield anything
        return

def fn_sample_place_kin(o, p, r):
    obj = ENV.get_object(o)
    if not obj: return

    try:
        grasp, q1, q2, traj_tuple = ENV.compute_place_trajectory(obj, list(p), region_name=r)
        # traj_tuple is (t_down, t_up)
        
        # Enforce "Retreat to Home" constraint
        q_home, traj_home = ENV.compute_retreat_to_home(q2)
        
        if traj_home:
            # Combine trajectories: (t_down, t_up, t_home)
            new_traj_tuple = (traj_tuple[0], traj_tuple[1], traj_home)
            # Yield with q_end = q_home
            yield (tuple(grasp), tuple(q1), tuple(q_home), new_traj_tuple)
    except Exception:
        return

_motion_error_count = 0

def fn_sample_motion(q1, q2):
    # Reverted resolution to default for speed, relying on better heuristics in env
    path = ENV.compute_motion_plan(q1, q2) 
    if path is None:
        return None
    yield (path,)

def fn_sample_grasp_lid(o):
    obj = ENV.get_object(o)
    if not obj: return
    try:
        # Returns grasp, q_hover, q_grasp, traj_approach
        grasp, q1, q2, traj = ENV.compute_lid_grasp_trajectory(obj)
        yield (tuple(grasp), tuple(q1), tuple(q2), traj)
    except Exception:
        return

def fn_sample_slide_lid(o, g):
    obj = ENV.get_object(o)
    if not obj: return
    try:
        # Returns q_start, q_end, traj_open, traj_return
        q1, q2, traj, traj_return = ENV.compute_slide_lid_trajectory(obj, list(g))
        # We yield the slide trajectory. The return trajectory is available if needed but we stick to the signature.
        yield (tuple(q1), tuple(q2), traj)
    except Exception:
        return

def fn_sample_open_lid(o):
    obj = ENV.get_object(o)
    if not obj: return
    try:
        # Returns grasp, q_hover_start, q_hover_end, (traj_approach, traj_slide, traj_return, traj_retreat)
        grasp, q1, q2, traj_tuple = ENV.compute_open_lid_trajectory(obj)
        yield (tuple(grasp), tuple(q1), tuple(q2), traj_tuple)
    except Exception:
        return

def get_stream_map():
    return {
        'sample-stable-pose': from_gen_fn(fn_sample_stable_pose),
        'sample-pick-kin':   from_gen_fn(fn_sample_pick_kin),
        'sample-place-kin':  from_gen_fn(fn_sample_place_kin),
        'sample-motion':     from_gen_fn(fn_sample_motion),
        'sample-open-lid':   from_gen_fn(fn_sample_open_lid),
    }
