import os
import sys
import numpy as np
import argparse

def _configure_qt():
    """Keep Qt quiet and point it at CoppeliaSim's plugins without forcing offscreen."""
    # GUI MODE: Set headless to 0
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


def execute_trajectory(env, traj):
    if not traj:
        return
    full_traj = []
    for i in range(len(traj) - 1):
        start = np.array(traj[i])
        end = np.array(traj[i + 1])
        steps = 15  # Reduced from 30 for 2x speed
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1 - t) * start + t * end)
    full_traj.append(traj[-1])

    for conf in full_traj:
        env.set_robot_conf(conf)
        env.pr.step()


def _to_list_traj(seg):
    """Convert numpy trajs / tuples into a list-of-7D-confs."""
    if seg is None:
        return []
    if isinstance(seg, np.ndarray):
        return seg.tolist()
    # if it's a tuple of floats (single conf), wrap it
    if isinstance(seg, tuple) and len(seg) == 7 and all(isinstance(x, (float, int)) for x in seg):
        return [list(seg)]
    return list(seg)


def _normalize_segments(traj_tuple):
    """
    Normalize traj_tuple into a list of trajectory segments.
    Each segment is: list[list[float]] with shape (T,7).
    """
    if traj_tuple is None:
        return []
    if isinstance(traj_tuple, (list, tuple)):
        segs = []
        for s in traj_tuple:
            t = _to_list_traj(s)
            if t:
                segs.append(t)
        return segs
    # fallback: single segment
    t = _to_list_traj(traj_tuple)
    return [t] if t else []


def _get_ee_pos(env):
    """
    Try to get the end-effector / tip position. Works across PyRep robot wrappers.
    """
    try:
        tip = env.robot.get_tip()
        return np.array(tip.get_position(), dtype=float)
    except Exception:
        # fallback: env.robot position
        return np.array(env.robot.get_position(), dtype=float)


def _ee_pos_at_conf(env, conf, restore_conf):
    """
    Temporarily set robot conf to read EE position, then restore.
    No stepping -> should not disturb physics.
    """
    env.set_robot_conf(conf)
    pos = _get_ee_pos(env)
    env.set_robot_conf(restore_conf)
    return pos


def _pick_grasp_segment_index(env, obj, segments):
    """
    Choose the segment index at whose END the EE is closest to the object.
    Grasp should happen after executing segments[0..idx].
    """
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


def _extract_pose_xyz(p):
    """
    p is usually a 7D pose (x,y,z,qx,qy,qz,qw) in your pipeline.
    Return xyz if possible, else None.
    """
    try:
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
    except Exception:
        pass
    return None


def _place_release_segment_index(env, place_pose_p, segments):
    """
    Prefer: choose segment end whose EE is closest to placement pose p (xyz).
    Fallback: choose segment end with minimum EE z (lowest point).
    """
    if not segments:
        return 0

    restore = env.get_robot_conf()
    p_xyz = _extract_pose_xyz(place_pose_p)

    if p_xyz is not None:
        best_i, best_d = 0, float("inf")
        for i, seg in enumerate(segments):
            end_conf = seg[-1]
            ee_pos = _ee_pos_at_conf(env, end_conf, restore)
            d = float(np.linalg.norm(ee_pos - p_xyz))
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    # fallback: lowest EE z
    best_i, best_z = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        if float(ee_pos[2]) < best_z:
            best_z, best_i = float(ee_pos[2]), i
    return best_i


def run_task(target_object_name, target_region_name, close_on_finish=True):
    env = ENV
    pr = env.pr

    print("Settling physics...")
    for _ in range(50):
        pr.step()

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    env.save_conf("home", home_q)
    for _ in range(10):
        pr.step()

    # --- CONFIGURATION ---
    env.set_target_region(target_region_name)

    print(f"Target Object: {target_object_name}")
    print(f"Target Region: {target_region_name}")

    obj = env.get_object(target_object_name)
    if obj is None:
        print(f"ERROR: {target_object_name} not found.")
        if close_on_finish:
            pr.stop()
            pr.shutdown()
        return

    # Force static for stability before picking
    obj.set_dynamic(False)

    initial_pose = obj.get_pose()

    # --- PDDL PROBLEM ---
    print(f"Setting up PDDL problem: Move {target_object_name} to {target_region_name}...")

    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    pose_tuple = tuple(initial_pose)

    init = [
        ('conf', q_home_tuple),
        ('at-conf', q_home_tuple),
        ('hand-empty',),

        ('movable', target_object_name),
        ('pose', pose_tuple),
        ('at-pose', target_object_name, pose_tuple),

        ('region', target_region_name),
    ]

    goal = And(('hand-empty',), ('in-region', target_object_name, target_region_name))

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving PDDL problem...")
    solution = solve(problem, algorithm='adaptive', verbose=True, max_time=60)
    plan, cost, evaluations = solution

    try:
        if plan:
            print(f"PDDL Plan found with cost: {cost}")
            for action in plan:
                print(f"Action: {action.name} {action.args}")

                if action.name == 'move':
                    q1, q2, traj = action.args
                    execute_trajectory(env, traj)

                elif action.name == 'pick':
                    o, p, g, q1, q2, traj_tuple = action.args
                    segments = _normalize_segments(traj_tuple)
                    if not segments:
                        raise RuntimeError("pick: empty trajectory segments")

                    target_obj = env.get_object(o)
                    grasp_idx = _pick_grasp_segment_index(env, target_obj, segments)

                    # Execute up to grasp segment (inclusive)
                    for seg in segments[:grasp_idx + 1]:
                        execute_trajectory(env, seg)

                    # Grasp at the correct moment
                    print(f"Grasping {o}...")
                    target_obj.set_dynamic(True)
                    env.gripper.actuate(0.0, 0.1)
                    for _ in range(10):
                        pr.step()
                    env.gripper.grasp(target_obj)

                    # Execute remaining retreat segments
                    for seg in segments[grasp_idx + 1:]:
                        execute_trajectory(env, seg)

                elif action.name == 'place':
                    o, p, g, r, q1, q2, traj_tuple = action.args
                    segments = _normalize_segments(traj_tuple)
                    if not segments:
                        raise RuntimeError("place: empty trajectory segments")

                    # Choose where to release based on closeness to placement pose p
                    release_idx = _place_release_segment_index(env, p, segments)

                    # Execute down-to-place (and possibly intermediate segments) until release point
                    for seg in segments[:release_idx + 1]:
                        execute_trajectory(env, seg)

                    # Release
                    print(f"Releasing {o}...")
                    target_obj = env.get_object(o)

                    env.gripper.release()
                    target_obj.set_dynamic(True)

                    # ACTIVE HOLD while opening
                    hold_q = env.get_robot_conf()
                    env.gripper.actuate(1.0, velocity=0.2)

                    for _ in range(60):
                        env.set_robot_conf(hold_q)
                        pr.step()

                    # Retreat target = END of the FINAL segment (planner's intended safe state)
                    current_q = env.get_robot_conf()
                    target_retreat_conf = segments[-1][-1]

                    # Compute Cartesian target pose
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
                        print("Warning: Linear retreat replanning failed. Executing planned post-release segments.")
                        # Fallback: execute the remaining planned segments after release
                        for seg in segments[release_idx + 1:]:
                            execute_trajectory(env, seg)

        else:
            print("ERROR: No PDDL plan found!")

        if close_on_finish:
            print("Sequence complete. Press Ctrl+C to close.")
            try:
                while True:
                    pr.step()
            except KeyboardInterrupt:
                pass

    finally:
        if close_on_finish:
            pr.stop()
            pr.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Generalized Pick and Place')
    parser.add_argument('--object', type=str, default='mug_inside_box', help='Name of the object to pick')
    parser.add_argument('--region', type=str, default='placement_boundary', help='Name of the target region')
    args = parser.parse_args()

    run_task(args.object, args.region, close_on_finish=True)


if __name__ == "__main__":
    main()
