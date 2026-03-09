import os
import sys
import numpy as np
import argparse
import time
import datetime

# --- CONFIGURATION ---
LOG_FILE = "baseline_with_rules_logs.txt"
PLAN_LOG_DIR = "baseline_with_rules_plans"

if not os.path.exists(PLAN_LOG_DIR):
    os.makedirs(PLAN_LOG_DIR)

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

sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

from pddlstream.language.constants import PDDLProblem, And
from pddlstream.utils import read

os.environ["HEADLESS"] = "False"
from rlbench_kitchen_streams_constrained import ENV, get_stream_map
from solve_with_rules import solve_with_rules

# --- LOGGING HELPER ---
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def log_result(episode, status, plan_str, duration, reason=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] Episode {episode}: {status} | Time: {duration:.2f}s\n"
    if reason:
        entry += f"  Reason: {reason}\n"
    entry += f"  Plan: {plan_str}\n"
    entry += f"  Detailed Log: {PLAN_LOG_DIR}/run_{episode}.log\n"
    entry += "-"*50 + "\n"
    
    print(entry) 
    with open(LOG_FILE, "a") as f:
        f.write(entry)

def log_plan(episode, plan_str):
    filename = os.path.join(PLAN_LOG_DIR, f"plan_{episode}.txt")
    with open(filename, "w") as f:
        f.write(plan_str)

# --- EXECUTION HELPERS ---
def execute_trajectory(env, traj):
    if not traj: return
    full_traj = []
    for i in range(len(traj)-1):
        start = np.array(traj[i])
        end = np.array(traj[i+1])
        steps = 15 
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1-t)*start + t*end)
    full_traj.append(traj[-1])
    
    for conf in full_traj:
        env.set_robot_conf(conf)
        env.pr.step()

def validate_config(env, q, min_dist=0.05):
    env.set_robot_conf(q)
    if env.robot.check_collision():
        return False
    return True

def interpolate_path(env, q1, q2, steps=30):
    traj = []
    q1 = np.array(q1)
    q2 = np.array(q2)
    for i in range(steps + 1):
        t = i / steps
        q = (1 - t) * q1 + t * q2
        q_list = q.tolist()
        if not validate_config(env, q_list):
            return None
        traj.append(q_list)
    return traj

def execute_move_via_home(env, start_q, end_q):
    """Attempts to move from start -> home -> end. Returns True if successful."""
    home_q = env.get_home_conf()
    
    # 1. Start -> Home
    traj_to_home = interpolate_path(env, start_q, home_q)
    if not traj_to_home:
        # Try simple escape (lift up)
        curr_pos = env.robot.get_position()
        target_pos = [curr_pos[0], curr_pos[1], curr_pos[2] + 0.15]
        try:
            path = env.robot.get_linear_path(position=target_pos, steps=10, ignore_collisions=True)
            if path:
                escape_traj = path._path_points.reshape(-1, 7).tolist()
                execute_trajectory(env, escape_traj)
                traj_to_home = interpolate_path(env, escape_traj[-1], home_q)
        except:
            pass

    if traj_to_home:
        execute_trajectory(env, traj_to_home)
    else:
        print("  [Warn] Could not plan path to Home. Skipping Home detour.")
        return False

    # 2. Home -> End
    traj_to_end = interpolate_path(env, home_q, end_q)
    if traj_to_end:
        execute_trajectory(env, traj_to_end)
        return True
    else:
        print("  [Warn] Reached Home, but path to target blocked.")
        return False

def go_home_only(env):
    """Just goes home from current config."""
    home_q = env.get_home_conf()
    current_q = env.get_robot_conf()
    traj = interpolate_path(env, current_q, home_q)
    if traj:
        execute_trajectory(env, traj)
        return True
    return False

# --- MAIN EPISODE RUNNER ---
def run_episode(episode_num, args):
    # Start logging stdout to file
    log_path = os.path.join(PLAN_LOG_DIR, f"run_{episode_num}.log")
    tee = Tee(log_path, "w")

    try:
        env = ENV
        pr = env.pr
        
        print(f"\n--- Starting Episode {episode_num} ---")
        
        # 1. Setup Scene
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        env.gripper.release()
        
        mug_name = 'mug_box'
        lid_name = 'box_lid'
        mug_inside_name = 'mug_inside_box'
        cupboard_obj_name = args.object # e.g. 'soup'
        
        placement_region = 'placement_boundary'
        cupboard_region = 'cupboard_boundary'

        mug_obj = env.get_object(mug_name)
        lid_obj = env.get_object(lid_name)
        mug_inside_obj = env.get_object(mug_inside_name)
        cupboard_obj = env.get_object(cupboard_obj_name)
        
        if not mug_obj or not lid_obj or not mug_inside_obj or not cupboard_obj:
            log_result(episode_num, "FAILURE", "N/A", 0, "Objects not found")
            return False

        # Freeze objects for planning stability
        mug_obj.set_dynamic(False)
        mug_inside_obj.set_dynamic(False)
        cupboard_obj.set_dynamic(False)
        
        # Get Initial State
        mug_pose = tuple(mug_obj.get_pose())
        mug_inside_pose = tuple(mug_inside_obj.get_pose())
        cupboard_obj_pose = tuple(cupboard_obj.get_pose())
        current_q = tuple(env.get_robot_conf())

        # 2. Define PDDL Problem
        directory = os.path.dirname(os.path.abspath(__file__))
        domain_pddl_path = os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl')
        stream_pddl_path = os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl')

        init = [
            ('conf', current_q),
            ('at-conf', current_q),
            ('hand-empty',),
            ('is-home', current_q), # Assume start at home
            
            ('movable', mug_name),
            ('pose', mug_pose),
            ('at-pose', mug_name, mug_pose),
            ('in-region', mug_name, 'box-top'), # Explicitly state start region
            
            ('movable', mug_inside_name),
            ('pose', mug_inside_pose),
            ('at-pose', mug_inside_name, mug_inside_pose),
            
            ('movable', cupboard_obj_name),
            ('pose', cupboard_obj_pose),
            ('at-pose', cupboard_obj_name, cupboard_obj_pose),
            
            ('lid', lid_name),
            ('closed', lid_name),
            
            ('region', placement_region),
            ('region', cupboard_region),
            ('region', 'box-top'),
            
            # CRITICAL: Add 'inside' fact to trigger the domain constraint
            # We say the mug is 'inside' the lid, so that picking it requires 'opened(lid)'
            ('inside', mug_inside_name, lid_name),
        ]

        goal = And(
            ('in-region', mug_name, placement_region),
            ('opened', lid_name),
            ('in-region', mug_inside_name, cupboard_region),
            # ('in-region', cupboard_obj_name, cupboard_region), # Optional, maybe skip for now to focus on mugs
            ('hand-empty',)
        )

        # 3. Solve with Rules
        print("  Planning with Rules...")
        start_time = time.time()
        
        env.set_target_region(placement_region) 
        
        # Use the new solve_with_rules wrapper
        solution = solve_with_rules(
            domain_pddl_path=domain_pddl_path,
            stream_pddl_path=stream_pddl_path,
            init_atoms=init,
            goal_atoms=goal,
            stream_map=get_stream_map()
        )
        
        plan, cost, evaluations = solution
        planning_time = time.time() - start_time

        if plan is None:
            log_result(episode_num, "FAILURE", "None", planning_time, "No Plan Found (Timeout/Infeasible)")
            return False

        plan_str = " -> ".join([f"{a.name}" for a in plan])
        print(f"  Plan found: {plan_str}")
        log_plan(episode_num, plan_str)

        # 4. Execute with Home Injection
        last_action_name = None
        
        try:
            for action in plan:
                print(f"  Executing: {action.name}")
                
                if action.name == 'move':
                    q1, q2, traj = action.args
                    execute_trajectory(env, traj)
                
                elif action.name == 'pick':
                    o, p, g, q1, q2, traj_tuple = action.args
                    
                    # Handle potential 3-part tuple (approach, retreat, home)
                    home_traj = None
                    if len(traj_tuple) == 3:
                        approach, retreat, home_traj = traj_tuple
                    else:
                        approach, retreat = traj_tuple
                    
                    # Disable box collision if picking from inside box
                    is_mug_inside = (o == 'mug_inside_box')
                    if is_mug_inside:
                        env.set_box_collision(False)
                        
                    try:
                        execute_trajectory(env, approach)
                        env.get_object(o).set_dynamic(True)
                        env.gripper.actuate(0.0, 0.1)
                        for _ in range(20): pr.step()
                        env.gripper.grasp(env.get_object(o))
                        execute_trajectory(env, retreat)
                        if home_traj:
                            execute_trajectory(env, home_traj)
                    finally:
                        if is_mug_inside:
                            env.set_box_collision(True)

                elif action.name == 'place':
                    o, p, g, r, q1, q2, traj_tuple = action.args
                    
                    # Helper for robust release
                    def robust_release(obj_name, lift_traj):
                        print(f"  [Info] Performing Robust Release for {obj_name}...")
                        target_obj = env.get_object(obj_name)
                        
                        # 1. Detach
                        env.gripper.release()
                        target_obj.set_dynamic(True)
                        
                        # 2. Active Hold while opening
                        hold_q = env.get_robot_conf()
                        env.gripper.actuate(1.0, velocity=0.2)
                        for _ in range(60):
                            env.set_robot_conf(hold_q)
                            pr.step()
                            
                        # 3. Relative Retreat
                        current_q = env.get_robot_conf()
                        target_retreat_conf = lift_traj[-1]
                        
                        # Calculate Cartesian Target
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
                                return
                        except:
                            pass
                        
                        # Fallback
                        print("  [Warn] Relative retreat failed. Using planned lift.")
                        full_retreat = [current_q] + lift_traj
                        execute_trajectory(env, full_retreat)

                    # Check for 3-part trajectory (down, up, home)
                    if isinstance(traj_tuple, tuple) and len(traj_tuple) == 3:
                        lower, lift, home_traj = traj_tuple
                        execute_trajectory(env, lower)
                        robust_release(o, lift)
                        execute_trajectory(env, home_traj)
                    else:
                        # Fallback
                        lower, lift = traj_tuple
                        execute_trajectory(env, lower)
                        robust_release(o, lift)

                elif action.name == 'open-lid':
                    o, g, q1, q2, traj_tuple = action.args
                    
                    home_traj = None
                    if len(traj_tuple) == 5:
                        app, slide, ret, esc, home_traj = traj_tuple
                    else:
                        app, slide, ret, esc = traj_tuple

                    execute_trajectory(env, app)
                    env.gripper.actuate(0.0, 0.1)
                    for _ in range(20): pr.step()
                    env.gripper.grasp(env.get_object(o))
                    execute_trajectory(env, slide)
                    env.gripper.release()
                    env.gripper.actuate(1.0, 0.1)
                    for _ in range(30): pr.step()
                    execute_trajectory(env, ret)
                    execute_trajectory(env, esc)
                    
                    if home_traj:
                        execute_trajectory(env, home_traj)
                    else:
                        # Explicitly go home after opening lid if not part of trajectory
                        go_home_only(env)
                
                last_action_name = action.name

            log_result(episode_num, "SUCCESS", plan_str, planning_time)
            return True

        except Exception as e:
            log_result(episode_num, "FAILURE", plan_str, planning_time, f"Execution Error: {str(e)}")
            return False

    except Exception as e:
        log_result(episode_num, "FAILURE", "N/A", 0, f"Setup/Planning Error: {str(e)}")
        return False
    finally:
        # Stop logging
        del tee

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default='soup')
    parser.add_argument('--episode_num', type=int, default=1)
    args = parser.parse_args()

    run_episode(args.episode_num, args)

    ENV.pr.stop()
    ENV.pr.shutdown()

if __name__ == "__main__":
    main()
