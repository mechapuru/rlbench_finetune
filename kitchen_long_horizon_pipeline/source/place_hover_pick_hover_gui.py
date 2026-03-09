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

from pddlstream.language.constants import PDDLProblem, And
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
    mug = env.get_object("mug_box")
    if mug is None:
        print("ERROR: mug_box not found in scene.")
        # recorder.release()
        pr.stop()
        pr.shutdown()
        return

    # Force the mug to be static so it doesn't jitter/slide
    mug.set_dynamic(False)

    pose = mug.get_pose()  # [x, y, z, qx, qy, qz, qw]

    # 1. Compute Pick Hover Config
    try:
        # Increased hover offset to 0.30 to avoid collision with the box during transport
        q_pick_hover = env.compute_hover_config(mug, list(pose), hover_offset=0.30)
        print("Found valid pick-hover configuration.")
    except Exception as e:
        print(f"ERROR: compute_hover_config (pick) failed: {e}")
        pr.stop()
        pr.shutdown()
        return

    # 2. Sample Place Pose and Compute Place Hover Config
    try:
        # Find best placement in placement_boundary
        place_pose = env.find_best_placement(mug, 'placement_boundary')
        print(f"Found best place pose: {place_pose}")
        
        # Compute hover config for this place pose
        q_place_hover = env.compute_hover_config(mug, list(place_pose), hover_offset=0.125)
        print("Found valid place-hover configuration.")
    except Exception as e:
        print(f"ERROR: Placement search failed: {e}")
        pr.stop()
        pr.shutdown()
        return

    # --- PDDL PLANNING START ---
    print("Setting up PDDL problem for 'hover-pick-hover-place' sequence...")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    q_pick_hover_tuple = tuple(q_pick_hover)
    q_place_hover_tuple = tuple(q_place_hover)
    pose_tuple = tuple(pose)
    place_pose_tuple = tuple(place_pose)

    # Define PDDL problem
    init = [
        ('conf', q_home_tuple),
        ('conf', q_pick_hover_tuple),
        ('conf', q_place_hover_tuple),
        ('at-conf', q_home_tuple),
        ('hand-empty',),
        ('movable', 'mug_box'),
        ('pose', pose_tuple),
        ('pose', place_pose_tuple), # Add the target place pose
        ('at-pose', 'mug_box', pose_tuple),
        ('region', 'placement_boundary'),
        ('stable', 'mug_box', place_pose_tuple, 'placement_boundary'),
    ]

    # Goal: Mug is at the place pose and hand is empty
    goal = And(('hand-empty',), ('at-pose', 'mug_box', place_pose_tuple))

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
                
                for conf in full_traj:
                    env.set_robot_conf(conf)
                    pr.step()
            
            elif action.name == 'pick':
                # args: ?o ?p ?g ?q1 ?q2 ?t
                o, p, g, q1, q2, traj = action.args
                print(f"Executing pick via PDDL plan...")
                
                mid_idx = len(traj) // 2
                
                # Execute first half (Approach)
                approach_traj = traj[:mid_idx]
                
                full_approach = []
                for i in range(len(approach_traj)-1):
                    start = np.array(approach_traj[i])
                    end = np.array(approach_traj[i+1])
                    steps = 30
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_approach.append((1-t)*start + t*end)
                full_approach.append(approach_traj[-1])
                
                print("Approaching...")
                for conf in full_approach:
                    env.set_robot_conf(conf)
                    pr.step()
                
                # GRASP
                print("Grasping...")
                # Make mug dynamic so we can pick it up
                mug.set_dynamic(True) 
                # Close gripper
                env.gripper.actuate(0.0, 0.1) 
                # Wait for gripper to close
                for _ in range(20):
                    pr.step()
                
                # Attach object if close enough (Simulated Grasp)
                env.gripper.grasp(mug)
                
                # Execute second half (Retreat)
                retreat_traj = traj[mid_idx:]
                
                full_retreat = []
                for i in range(len(retreat_traj)-1):
                    start = np.array(retreat_traj[i])
                    end = np.array(retreat_traj[i+1])
                    steps = 30
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_retreat.append((1-t)*start + t*end)
                full_retreat.append(retreat_traj[-1])
                
                print("Retreating...")
                for conf in full_retreat:
                    env.set_robot_conf(conf)
                    pr.step()

            elif action.name == 'place':
                # args: ?o ?p ?g ?region ?q1 ?q2 ?t
                try:
                    o, p, g, region, q1, q2, traj = action.args
                except ValueError:
                    # Fallback if region is not in args (depends on domain definition)
                    o, p, g, q1, q2, traj = action.args
                
                print(f"Executing place via PDDL plan...")
                
                mid_idx = len(traj) // 2
                
                # Execute first half (Lower)
                lower_traj = traj[:mid_idx]
                
                full_lower = []
                for i in range(len(lower_traj)-1):
                    start = np.array(lower_traj[i])
                    end = np.array(lower_traj[i+1])
                    steps = 30
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_lower.append((1-t)*start + t*end)
                if lower_traj:
                    full_lower.append(lower_traj[-1])
                
                print("Lowering to place...")
                for conf in full_lower:
                    env.set_robot_conf(conf)
                    pr.step()
                
                # RELEASE
                print("Releasing...")
                # Open gripper
                env.gripper.actuate(1.0, 0.1) 
                # Wait for gripper to open
                for _ in range(20):
                    pr.step()
                
                # Detach object
                env.gripper.release()
                # Make object dynamic again so it falls/settles
                mug.set_dynamic(True)
                
                # Execute second half (Lift)
                lift_traj = traj[mid_idx:]
                
                full_lift = []
                for i in range(len(lift_traj)-1):
                    start = np.array(lift_traj[i])
                    end = np.array(lift_traj[i+1])
                    steps = 30
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_lift.append((1-t)*start + t*end)
                if lift_traj:
                    full_lift.append(lift_traj[-1])
                
                print("Lifting...")
                for conf in full_lift:
                    env.set_robot_conf(conf)
                    pr.step()

    else:
        print("ERROR: No PDDL plan found!")
    
    # --- PDDL PLANNING END ---

    # A few extra frames to show final pose
    for _ in range(20):
        pr.step()

    # Leave sim running so you can inspect; close manually with Ctrl+C
    print("Place sequence complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
