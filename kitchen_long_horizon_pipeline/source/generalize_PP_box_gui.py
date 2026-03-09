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


import argparse

def execute_trajectory(env, traj):
    if not traj: return
    full_traj = []
    for i in range(len(traj)-1):
        start = np.array(traj[i])
        end = np.array(traj[i+1])
        steps = 15 # Reduced from 30 for 2x speed
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1-t)*start + t*end)
    full_traj.append(traj[-1])
    
    for conf in full_traj:
        env.set_robot_conf(conf)
        env.pr.step()

def run_task(target_object_name, target_region_name, close_on_finish=True):
    # Use the shared ENV
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
    # Set the target region in ENV so pick strategy can adapt
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
    # Increased max_time to 60 seconds to allow for more sampling attempts
    solution = solve(problem, algorithm='adaptive', verbose=True, max_time=60)
    plan, cost, evaluations = solution

    if plan:
        print(f"PDDL Plan found with cost: {cost}")
        for action in plan:
            print(f"Action: {action.name} {action.args}")
            
            if action.name == 'move':
                q1, q2, traj = action.args
                execute_trajectory(env, traj)
            
            elif action.name == 'pick':
                o, p, g, q1, q2, traj_tuple = action.args
                # traj_tuple is (approach_traj, retreat_traj)
                approach_traj, retreat_traj = traj_tuple
                
                # Execute approach
                execute_trajectory(env, approach_traj)
                
                # Grasp
                print(f"Grasping {o}...")
                target_obj = env.get_object(o)
                target_obj.set_dynamic(True)
                env.gripper.actuate(0.0, 0.1)
                for _ in range(10): pr.step() # Reduced from 20
                env.gripper.grasp(target_obj)
                
                # Execute retreat
                execute_trajectory(env, retreat_traj)

            elif action.name == 'place':
                o, p, g, r, q1, q2, traj_tuple = action.args
                # traj_tuple is (lower_traj, lift_traj)
                lower_traj, lift_traj = traj_tuple
                
                # Execute lower/approach
                execute_trajectory(env, lower_traj)
                
                # Release
                print(f"Releasing {o}...")
                target_obj = env.get_object(o)
                
                # Detach object FIRST
                env.gripper.release()
                target_obj.set_dynamic(True)
                
                # ACTIVE HOLD: Hold position while opening gripper
                # This prevents the "loop" glitch caused by gravity drift
                hold_q = env.get_robot_conf()
                
                # Open gripper DECISIVELY (faster velocity)
                env.gripper.actuate(1.0, velocity=0.2) 
                
                # Wait for gripper to open while actively holding arm position
                for _ in range(60): # Increased to ensure release
                    env.set_robot_conf(hold_q)
                    pr.step()
                
                # RELATIVE RETREAT: Plan from ACTUAL position to intended retreat end
                # This ensures a smooth exit even if there was slight drift
                current_q = env.get_robot_conf()
                target_retreat_conf = lift_traj[-1] # The end of the planned lift
                
                # Calculate Cartesian Target for the retreat
                # We momentarily set the robot to the target config to read the pose
                env.set_robot_conf(target_retreat_conf)
                target_pos = env.robot.get_position()
                target_quat = env.robot.get_quaternion()
                env.set_robot_conf(current_q) # Restore to current
                
                # Generate a LINEAR CARTESIAN path to the retreat target
                # This avoids "weird" joint-space interpolations
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
                    # Fallback to original planned linear trajectory if dynamic replanning fails
                    # This ensures we still follow the Cartesian straight line planned originally,
                    # rather than doing a joint-space interpolation which causes collisions.
                    print("Warning: Linear retreat replanning failed. Executing original planned linear retreat.")
                    
                    # Prepend current configuration to ensure smooth start
                    # execute_trajectory will interpolate from current_q to lift_traj[0]
                    full_retreat = [current_q] + lift_traj
                    execute_trajectory(env, full_retreat)
                
    else:
        print("ERROR: No PDDL plan found!")

    if close_on_finish:
        # Keep open
        print("Sequence complete. Press Ctrl+C to close.")
        try:
            while True:
                pr.step()
        except KeyboardInterrupt:
            pass
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
