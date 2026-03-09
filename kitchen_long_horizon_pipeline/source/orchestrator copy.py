import os
import sys
import numpy as np
import time

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

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import ENV from streams to share the instance
from rlbench_kitchen_streams import ENV

# Import tasks
from generalize_pick_place_gui import run_task as run_pick_place
from script_3_open_grasp_open_box_gui import run_task as run_open_box
from generalize_PP_box_gui import run_task as run_pick_place_box

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

def validate_config(env, q, min_dist=0.10):
    env.set_robot_conf(q)
    if env.robot.check_collision():
        return False
    if hasattr(env, 'cupboard'):
        # Check distance to cupboard
        if env.robot.check_distance(env.cupboard) < min_dist:
            return False
    return True

def interpolate_path(env, q1, q2, steps=50):
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

def go_home(env):
    print("\n--- Returning to Home ---")
    q_start = env.get_robot_conf()
    q_home = env.get_home_conf()
    
    # 1. Try Direct Interpolation with Safety Check
    traj = interpolate_path(env, q_start, q_home)
    if traj:
        execute_trajectory(env, traj)
        print("Reached Home (Direct).")
        return

    # 2. If failed, try to escape (Lift Up)
    print("Direct retreat blocked or too close. Attempting evasive maneuver...")
    curr_pos = env.robot.get_position()
    curr_quat = env.robot.get_quaternion()
    
    # Try moving UP first (safest for cupboard)
    escape_offsets = [[0, 0, 0.25], [0, 0, 0.15], [-0.1, 0, 0.1]]
    
    for off in escape_offsets:
        target_pos = [curr_pos[0]+off[0], curr_pos[1]+off[1], curr_pos[2]+off[2]]
        try:
            path = env.robot.get_linear_path(position=target_pos, quaternion=curr_quat, steps=20, ignore_collisions=True)
            if path:
                escape_traj = path._path_points.reshape(-1, 7).tolist()
                # Validate escape path
                valid_escape = True
                for q in escape_traj:
                    if not validate_config(env, q, min_dist=0.05): # Slightly relaxed for escape
                        valid_escape = False
                        break
                
                if valid_escape:
                    execute_trajectory(env, escape_traj)
                    # Try home from here
                    q_new = escape_traj[-1]
                    traj_home = interpolate_path(env, q_new, q_home)
                    if traj_home:
                        execute_trajectory(env, traj_home)
                        print("Reached Home (Escaped).")
                        return
        except:
            pass

    print("WARNING: Safe retreat failed. Forcing home.")
    env.set_robot_conf(q_home)
    for _ in range(10): env.pr.step()

def main():
    env = ENV
    pr = env.pr

    print("Settling physics...")
    for _ in range(50):
        pr.step()

    # Ensure home is set
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    
    # --- STAGE 1: Mug to Placement ---
    print("\n=== STAGE 1: Mug -> Placement ===")
    run_pick_place('mug_box', 'placement_boundary', close_on_finish=False)
    
    go_home(env)
    
    # --- STAGE 2: Open Box ---
    print("\n=== STAGE 2: Open Box ===")
    run_open_box(close_on_finish=False)
    
    go_home(env)

    # --- STAGE 2.5: Mug Inside Box -> Box Boundary ---
    print("\n=== STAGE 2.5: Mug Inside Box -> Box Boundary ===")
    run_pick_place_box('mug_inside_box', 'box_boundary', close_on_finish=False)
    
    go_home(env)
    
    # --- STAGE 3: Soup to Cupboard ---
    print("\n=== STAGE 3: Soup -> Cupboard ===")
    run_pick_place('soup', 'cupboard_boundary', close_on_finish=False)
    
    go_home(env)

    print("\nAll stages complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass
    pr.stop()
    pr.shutdown()

if __name__ == "__main__":
    main()
