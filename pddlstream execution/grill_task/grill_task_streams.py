# grill_task_streams.py
# Stream implementations for the grill task PDDL domain

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pddlstream'))

from pddlstream.language.generator import from_gen_fn
from pddlstream.utils import INF
from grill_task_env import GrillTaskEnv

# One global env instance for planning
# Check env var HEADLESS, default to True
headless_mode = os.environ.get("HEADLESS", "True") == "True"
ENV = GrillTaskEnv(headless=headless_mode)


# ----- Python functions used by streams -----

def fn_sample_stable_pose(o, r):
    """Sample a stable pose for object o in region r."""
    obj = ENV.get_object(o)
    if not obj:
        return

    # Yield multiple samples to help the planner find a reachable one
    for _ in range(50):
        pose = ENV.sample_stable_pose(obj, r)
        yield (tuple(pose),)  # outputs: (?p)


def fn_sample_pick_kin(o, p):
    """Sample a feasible pick trajectory for object o at pose p."""
    obj = ENV.get_object(o)
    if not obj:
        return

    try:
        grasp, q1, q2, traj_tuple = ENV.compute_pick_trajectory(obj, list(p))
        # traj_tuple is (t_approach, t_retreat)
        yield (tuple(grasp), tuple(q1), tuple(q2), traj_tuple)
    except Exception as e:
        # print(f"DEBUG: sample-pick-kin failed: {e}")
        return


def fn_sample_place_kin(o, p, r):
    """Sample a feasible place trajectory for object o at pose p in region r."""
    obj = ENV.get_object(o)
    if not obj:
        return

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


def fn_sample_motion(q1, q2):
    """Sample a motion plan between configurations q1 and q2."""
    path = ENV.compute_motion_plan(q1, q2)
    if path is None:
        return None
    yield (path,)


def fn_sample_close_grill(o):
    """Sample a trajectory to close the grill lid."""
    obj = ENV.get_object(o)
    if not obj:
        return
    try:
        grasp, q1, q2, traj = ENV.compute_close_grill_trajectory(obj)
        yield (tuple(grasp), tuple(q1), tuple(q2), traj)
    except Exception:
        return


def get_stream_map():
    """Return the stream map for PDDLStream."""
    return {
        'sample-stable-pose': from_gen_fn(fn_sample_stable_pose),
        'sample-pick-kin': from_gen_fn(fn_sample_pick_kin),
        'sample-place-kin': from_gen_fn(fn_sample_place_kin),
        'sample-motion': from_gen_fn(fn_sample_motion),
        'sample-close-grill': from_gen_fn(fn_sample_close_grill),
    }
