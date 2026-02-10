#!/usr/bin/env python3
"""
grill_hover_pick_gui.py
GUI script to test hovering and picking meat objects (steak/chicken) in the grill task scene.
The robot will:
1. First hover over the object (like grill_hover_gui.py)
2. Then approach, grasp, and retreat with the object

Usage: python grill_hover_pick_gui.py [object_name]
       object_name can be: steak, chicken, meat1, meat2
"""

import os
import sys
import argparse
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pddlstream'))

from pddlstream.language.constants import PDDLProblem
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import ENV from streams to share the instance
from grill_task_streams import ENV, get_stream_map


def interpolate_trajectory(traj, steps_per_segment=30):
    """Interpolate a trajectory for smoother motion."""
    if not traj or len(traj) < 2:
        return traj
    
    full_traj = []
    for i in range(len(traj) - 1):
        start = np.array(traj[i])
        end = np.array(traj[i + 1])
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            full_traj.append((1 - t) * start + t * end)
    full_traj.append(traj[-1])
    return full_traj


def execute_trajectory(env, pr, traj, steps_per_segment=30):
    """Execute a trajectory with interpolation."""
    full_traj = interpolate_trajectory(traj, steps_per_segment)
    for conf in full_traj:
        env.set_robot_conf(conf)
        pr.step()


def grasp_object(env, pr, target_obj):
    """
    Properly grasp an object with correct physics handling.
    Keep object static until gripper is fully closed and attached.
    """
    # Keep object static during grasp to prevent falling
    target_obj.set_dynamic(False)
    
    # Close gripper first (while object is still static)
    env.gripper.actuate(0.0, 0.1)  # Close gripper
    
    # Wait for gripper to fully close
    for _ in range(30):
        pr.step()
    
    # Now attach the object to the gripper (simulated grasp)
    env.gripper.grasp(target_obj)
    
    # Step a few times to ensure attachment
    for _ in range(10):
        pr.step()
    
    # NOW make object dynamic - it's attached so won't fall
    target_obj.set_dynamic(True)
    
    # A few more steps to stabilize
    for _ in range(10):
        pr.step()


def main():
    parser = argparse.ArgumentParser(description='Hover and pick a meat object in the grill task scene')
    parser.add_argument('object', nargs='?', default='steak',
                        help='Object to pick: steak, chicken, meat1, meat2 (default: steak)')
    args = parser.parse_args()

    object_name = args.object
    print(f"=== Grill Task: Hover & Pick '{object_name}' ===")

    # Use the shared ENV
    env = ENV
    pr = env.pr

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

    # Get the target object and its current pose
    target_obj = env.get_object(object_name)
    if target_obj is None:
        print(f"ERROR: '{object_name}' not found in scene.")
        print("Available objects:", list(env.name_to_obj.keys()))
        pr.stop()
        pr.shutdown()
        return

    # Force the object to be static so it doesn't jitter/slide
    target_obj.set_dynamic(False)

    pose = target_obj.get_pose()  # [x, y, z, qx, qy, qz, qw]
    print(f"Object '{object_name}' is at pose: {pose[:3]}")

    # ============================================
    # STEP 1: HOVER over the object (like grill_hover_gui.py)
    # ============================================
    print("\n--- Step 1: Hover over object ---")
    
    try:
        q_hover, hover_orientation = env.compute_hover_config(target_obj, list(pose), hover_offset=0.15)
        print("Found valid hover configuration.")
    except Exception as e:
        print(f"ERROR: compute_hover_config failed: {e}")
        pr.stop()
        pr.shutdown()
        return

    # Move from home to hover position
    print("Moving from home to hover position...")
    motion_traj = env.compute_motion_plan(home_q, q_hover)
    if motion_traj:
        execute_trajectory(env, pr, motion_traj)
        print("Hover position reached!")
    else:
        print("WARNING: Motion plan to hover failed, trying direct interpolation...")
        direct_traj = []
        q1, q2 = np.array(home_q), np.array(q_hover)
        for t in np.linspace(0, 1, 100):
            direct_traj.append(((1-t)*q1 + t*q2).tolist())
        execute_trajectory(env, pr, direct_traj, steps_per_segment=1)

    # Pause at hover to show the position
    print("Pausing at hover position...")
    for _ in range(30):
        pr.step()

    # ============================================
    # STEP 2: PICK the object
    # ============================================
    print("\n--- Step 2: Pick object ---")

    # Compute pick trajectory - use the SAME orientation as hover
    current_q = env.get_robot_conf()  # This is the hover position we just reached
    
    try:
        # Pass hover_orientation so pick uses the same gripper rotation
        grasp, q_start, q_end, traj_tuple = env.compute_pick_trajectory(
            target_obj, list(pose), preferred_orientation=hover_orientation
        )
        approach_traj, retreat_traj = traj_tuple
        print("Pick trajectory computed successfully (using hover orientation).")
    except Exception as e:
        print(f"ERROR: compute_pick_trajectory failed: {e}")
        pr.stop()
        pr.shutdown()
        return

    # Override: Use the current hover config as the start for approach
    # This ensures continuity - we start from where we are (hover position)
    # The approach_traj starts from q_start, so we need to adjust
    # Simply set the robot to current position and use approach_traj as-is
    # since approach_traj[0] should be close to hover anyway
    
    # If there's a mismatch, smoothly transition from current to approach start
    if not np.allclose(current_q, approach_traj[0], atol=0.05):
        print("Smoothly transitioning to approach start...")
        # Create a short interpolation from current hover to approach start
        transition_traj = []
        q1, q2 = np.array(current_q), np.array(approach_traj[0])
        for t in np.linspace(0, 1, 20):
            transition_traj.append(((1-t)*q1 + t*q2).tolist())
        execute_trajectory(env, pr, transition_traj, steps_per_segment=1)

    # Execute approach (hover -> grasp position)
    print("Approaching object...")
    execute_trajectory(env, pr, approach_traj, steps_per_segment=20)

    # GRASP with proper physics handling
    print("Grasping object...")
    grasp_object(env, pr, target_obj)
    print("Object grasped!")

    # Execute retreat (grasp position -> hover)
    print("Retreating with object...")
    execute_trajectory(env, pr, retreat_traj, steps_per_segment=20)

    print("\n=== Hover & Pick complete! ===")

    # A few extra frames to show final pose
    for _ in range(30):
        pr.step()

    # Leave sim running so you can inspect
    print(f"Holding '{object_name}'. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
