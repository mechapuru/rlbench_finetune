#!/usr/bin/env python3
"""
grill_place_gui.py
GUI script for the complete pick & place sequence in the grill task scene.
The robot will:
1. Hover over the object
2. Pick it up
3. Move to the target region
4. Place it on the target surface (grill or plate)

Usage: python grill_place_gui.py [object_name] [region_name]
       object_name can be: steak, chicken, meat1, meat2, plate
       region_name can be: grill-top, plate-top, plate_boundary (default: grill-top)
       
Note: When placing 'plate', it will be placed upright (base facing down)
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


def grasp_object(env, pr, target_obj, is_plate=False):
    """
    Properly grasp an object with correct physics handling.
    Keep object static until gripper is fully closed and attached.
    
    Args:
        is_plate: If True, keep object static throughout to prevent physics issues
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
    
    # For plate, keep it STATIC throughout to prevent physics glitches
    # For other objects, make dynamic so they move with gripper properly
    if not is_plate:
        target_obj.set_dynamic(True)
    else:
        # Plate stays static - we'll manually update its position
        print("  Plate kept static during transport")
    
    # A few more steps to stabilize
    for _ in range(10):
        pr.step()


def release_object(env, pr, target_obj, region_name='grill-top', place_pose=None, is_plate=False):
    """
    Properly release an object at its current position.
    No teleporting - object stays where gripper releases it.
    """
    # Get current position before release
    gripper_pos = env.robot.get_tip().get_position()
    obj_pos = target_obj.get_position()
    print(f"  Gripper position at release: {gripper_pos}")
    print(f"  Object position at release: {obj_pos}")
    
    # Release the grasp
    env.gripper.release()
    
    # Open gripper
    env.gripper.actuate(1.0, velocity=0.2)
    
    # Wait for gripper to open
    for _ in range(30):
        pr.step()
    
    # Set object to STATIC so it doesn't fall through the surface
    target_obj.set_dynamic(False)
    target_obj.set_collidable(True)
    target_obj.set_respondable(True)
    
    # Object stays exactly where it was released - NO teleporting
    final_pos = target_obj.get_position()
    print(f"  Object final position: {final_pos}")
    print("  Object set to static - stays where released")
    
    # Brief pause
    for _ in range(10):
        pr.step()


def main():
    parser = argparse.ArgumentParser(description='Complete pick & place sequence on the grill or plate')
    parser.add_argument('object', nargs='?', default='steak',
                        help='Object to pick: steak, chicken, meat1, meat2, plate (default: steak)')
    parser.add_argument('region', nargs='?', default='grill-top',
                        help='Region to place on: grill-top, plate-top, plate_boundary (default: grill-top)')
    args = parser.parse_args()

    object_name = args.object
    region_name = args.region
    
    # Check if this is a plate (needs special handling)
    is_plate = object_name.lower() == 'plate'
    
    print(f"=== Grill Task: Complete Pick & Place ===")
    print(f"    Object: '{object_name}'")
    print(f"    Target: '{region_name}'")
    if is_plate:
        print("    Mode: PLATE (special handling enabled)")

    # Use the shared ENV
    env = ENV
    pr = env.pr

    # Ensure we are at home and save it once
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    env.save_conf("home", home_q)

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
    # STEP 1: HOVER over the object
    # ============================================
    print("\n" + "="*50)
    print("STEP 1: Hover over object")
    print("="*50)
    
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

    # Pause at hover
    for _ in range(20):
        pr.step()

    # ============================================
    # STEP 2: PICK the object
    # ============================================
    print("\n" + "="*50)
    print("STEP 2: Pick object")
    print("="*50)

    # Get current config (should be at hover position from Step 1)
    current_q = env.get_robot_conf()
    
    try:
        # Pass hover_orientation so pick uses the same gripper rotation
        # For plates, go DEEPER to get a good grasp
        grasp, q_start, q_end, traj_tuple = env.compute_pick_trajectory(
            target_obj, list(pose), preferred_orientation=hover_orientation, is_plate=is_plate
        )
        approach_traj, retreat_traj = traj_tuple
        print("Pick trajectory computed successfully (using hover orientation).")
    except Exception as e:
        print(f"ERROR: compute_pick_trajectory failed: {e}")
        pr.stop()
        pr.shutdown()
        return
    # Ensure smooth transition from hover to approach start
    # The hover and approach start should be the same, but handle any mismatch
    if not np.allclose(current_q, approach_traj[0], atol=0.05):
        print("Smoothly transitioning to approach start...")
        transition_traj = []
        q1, q2 = np.array(current_q), np.array(approach_traj[0])
        for t in np.linspace(0, 1, 20):
            transition_traj.append(((1-t)*q1 + t*q2).tolist())
        execute_trajectory(env, pr, transition_traj, steps_per_segment=1)

    # Execute approach
    print("Approaching object...")
    execute_trajectory(env, pr, approach_traj, steps_per_segment=20)

    # GRASP
    print("Grasping object...")
    grasp_object(env, pr, target_obj, is_plate=is_plate)
    print("Object grasped!")

    # Execute retreat
    print("Retreating with object...")
    execute_trajectory(env, pr, retreat_traj, steps_per_segment=20)

    # Pause to show picked object
    for _ in range(20):
        pr.step()

    # ============================================
    # STEP 3: MOVE to home (intermediate safe position)
    # ============================================
    print("\n" + "="*50)
    print("STEP 3: Return to home position")
    print("="*50)

    current_q = env.get_robot_conf()
    q_home, home_traj = env.compute_retreat_to_home(current_q)
    if home_traj:
        print("Moving to home position...")
        execute_trajectory(env, pr, home_traj)
    else:
        print("WARNING: Could not compute path to home")

    # Pause at home
    for _ in range(20):
        pr.step()

    # ============================================
    # STEP 4: PLACE the object on the target region
    # ============================================
    print("\n" + "="*50)
    print(f"STEP 4: Place object on '{region_name}'")
    print("="*50)

    # Sample a place pose on the target region
    place_pose = env.sample_stable_pose(target_obj, region_name)
    print(f"Target place pose: {place_pose[:3]}")

    # For plate object, ensure it's placed UPRIGHT (base facing down)
    if is_plate:
        # Upright orientation: no rotation (base down)
        # quaternion [qx, qy, qz, qw] = [0, 0, 0, 1] means no rotation
        place_pose[3] = 0.0  # qx
        place_pose[4] = 0.0  # qy
        place_pose[5] = 0.0  # qz
        place_pose[6] = 1.0  # qw
        print("  Plate will be placed UPRIGHT (base facing down)")

    try:
        grasp_p, q_start_p, q_end_p, place_traj_tuple = env.compute_place_trajectory(
            target_obj, place_pose, region_name=region_name, is_plate=is_plate
        )
        down_traj, up_traj = place_traj_tuple
        print("Place trajectory computed successfully.")
    except Exception as e:
        print(f"ERROR: compute_place_trajectory failed: {e}")
        pr.stop()
        pr.shutdown()
        return

    # Move to place hover position
    current_q = env.get_robot_conf()
    motion_traj = env.compute_motion_plan(current_q, q_start_p)
    if motion_traj:
        print("Moving to place hover position...")
        execute_trajectory(env, pr, motion_traj)
    else:
        print("WARNING: Motion plan to place hover failed")

    # Pause at place hover
    for _ in range(10):
        pr.step()

    # Execute lower
    print("Lowering object...")
    execute_trajectory(env, pr, down_traj, steps_per_segment=20)

    # RELEASE - pass place_pose so object is placed at exact position
    print("Releasing object...")
    release_object(env, pr, target_obj, region_name, place_pose, is_plate=is_plate)
    print("Object released!")

    # Execute lift
    print("Lifting gripper...")
    execute_trajectory(env, pr, up_traj, steps_per_segment=20)

    # ============================================
    # STEP 5: RETURN to home
    # ============================================
    print("\n" + "="*50)
    print("STEP 5: Return to home")
    print("="*50)

    current_q = env.get_robot_conf()
    q_home, home_traj = env.compute_retreat_to_home(current_q)
    if home_traj:
        print("Returning to home position...")
        execute_trajectory(env, pr, home_traj)
    
    # Final pause
    for _ in range(30):
        pr.step()

    print("\n" + "="*50)
    print("=== PICK & PLACE COMPLETE! ===")
    print("="*50)

    # Leave sim running so you can inspect
    print(f"\n'{object_name}' placed on '{region_name}'. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
