import os
import sys
import numpy as np

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

from pddlstream.language.constants import PDDLProblem
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import ENV from streams to share the instance
from rlbench_kitchen_streams import ENV, get_stream_map
# from video_recorder import VideoRecorder # Disable video recorder


def main():
    # Use the shared ENV
    env = ENV
    pr = env.pr

    # recorder = VideoRecorder(env) # Disable video recorder

    # SETTLE PHYSICS: Step simulation to let objects settle
    print("Settling physics...")
    for _ in range(50):
        pr.step()

    # Ensure we are at home and save it once
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    env.save_conf("home", home_q)
    for _ in range(10):
        pr.step()
        # recorder.record_step()

    # Get the mug object and its current pose
    mug = env.get_object("mug3")
    if mug is None:
        print("ERROR: mug3 not found in scene.")
        # recorder.release()
        pr.stop()
        pr.shutdown()
        return

    # Force the mug to be static so it doesn't jitter/slide
    mug.set_dynamic(False)

    pose = mug.get_pose()  # [x, y, z, qx, qy, qz, qw]

    # Custom Horizontal Hover Logic for Cupboard
    print("Computing horizontal hover configuration for cupboard mug...")
    
    # Hover Pose: In front of cupboard (shifted -X)
    # User requested same distance logic but horizontal
    # Added Z offset for slant pick
    hover_dist = 0.25 
    z_offset = 0.08 # 8cm higher
    hover_pos = [pose[0] - hover_dist, pose[1], pose[2] + z_offset]
    
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
    
    q_hover = None
    successful_grasp_quat = None
    original_conf = env.get_robot_conf()
    
    try:
        for grasp_rot in grasp_quats:
            # Solve IK for Hover Pose
            path_configs = env.robot.solve_ik_via_sampling(hover_pos, quaternion=grasp_rot, max_configs=5, max_time_ms=100, ignore_collisions=True)
            if path_configs is not None and len(path_configs) > 0:
                # Check collision
                for q in path_configs:
                    env.set_robot_conf(q)
                    if not env.robot.check_collision():
                        q_hover = q
                        successful_grasp_quat = grasp_rot
                        print("Found valid horizontal hover configuration.")
                        break
            if q_hover is not None:
                break
                
        if q_hover is None:
            raise RuntimeError("Could not find valid horizontal hover configuration")
            
    except Exception as e:
        print(f"ERROR: Horizontal hover computation failed: {e}")
        env.set_robot_conf(original_conf)
        # recorder.release()
        pr.stop()
        pr.shutdown()
        return
        
    env.set_robot_conf(original_conf) # Restore for planning

    if np.allclose(home_q, q_hover, atol=1e-4):
        print("WARNING: hover configuration matches home; hovering motion may be trivial.")

    # --- PDDL PLANNING START ---
    print("Setting up PDDL problem for 'move' action...")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    q_hover_tuple = tuple(q_hover)

    # Define PDDL problem
    # We want to move from home to hover.
    # We provide both configs as known 'conf' objects.
    init = [
        ('conf', q_home_tuple),
        ('conf', q_hover_tuple),
        ('at-conf', q_home_tuple),
    ]

    goal = ('at-conf', q_hover_tuple)

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving PDDL problem...")
    solution = solve(problem, algorithm='adaptive', verbose=False)
    plan, cost, evaluations = solution

    if plan:
        print(f"PDDL Plan found with cost: {cost}")
        for action in plan:
            print(f"Action: {action.name}")
            
            if action.name == 'move':
                # args: ?q1 ?q2 ?t
                q1, q2, traj = action.args
                print(f"Executing move via PDDL plan...")
                
                # Interpolate trajectory for smoother video
                full_traj = []
                # traj is a list of waypoints (tuples)
                for i in range(len(traj)-1):
                    start = np.array(traj[i])
                    end = np.array(traj[i+1])
                    steps = 60
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_traj.append((1-t)*start + t*end)
                full_traj.append(traj[-1])

                print(f"DEBUG: Executing trajectory with {len(full_traj)} points.")
                if len(full_traj) > 0:
                    print(f"DEBUG: Start Config: {full_traj[0][:3]}...")
                    print(f"DEBUG: End Config:   {full_traj[-1][:3]}...")

                for conf in full_traj:
                    env.set_robot_conf(conf)
                    pr.step()
                    # recorder.record_step()
        
        # Save hover configuration for later scripts
        env.save_conf("hover_pick", q_hover)
        print("Saved 'hover_pick' configuration after moving to hover pose.")
    else:
        print("ERROR: No PDDL plan found for move action!")
    
    # --- PDDL PLANNING END ---

    # A few extra frames to show final pose + line
    for _ in range(20):
        pr.step()
        # recorder.record_step()

    # recorder.release()

    # Leave sim running so you can inspect; close manually with Ctrl+C
    print("Hover-from-home complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
