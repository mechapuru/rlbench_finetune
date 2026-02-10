#!/usr/bin/env python3
"""
grill_open_hover_grasp_gui.py

Step 2: Move the robot to hover near handle, then move to handle and grasp it.
Builds on grill_open_hover_gui.py
"""

import sys
import os
import argparse
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pddlstream'))

from grill_task_env import GrillTaskEnv, quaternion_from_euler
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy

# Global environment instance
ENV = GrillTaskEnv(headless=False)


def execute_trajectory(env, pr, trajectory):
    """Execute a trajectory (list of joint configs) step by step."""
    for q in trajectory:
        env.set_robot_conf(q)
        pr.step()


def compute_handle_hover_config(env):
    """
    Compute a configuration to hover in front of the grill lid handle.
    Returns: (q_hover, hover_orientation, handle_pos) or (None, None, None) if failed
    """
    # Try to get the handle position
    try:
        handle = Shape('handle_visual')
        handle_pos = handle.get_position()
        handle_orient = handle.get_orientation()
        print(f"DEBUG: Handle position from handle_visual: {handle_pos}")
        print(f"DEBUG: Handle orientation: {handle_orient}")
    except Exception:
        try:
            lid = Shape('lid_visual')
            lid_pos = lid.get_position()
            handle_pos = [lid_pos[0], lid_pos[1] - 0.1, lid_pos[2] + 0.1]
            print(f"DEBUG: Estimated handle position from lid: {handle_pos}")
        except Exception:
            print("ERROR: Could not find handle or lid")
            return None, None, None
    
    # Hover offset (same as grill_open_hover_gui.py)
    hover_offset = np.array([-0.03, 0.02, 0.06])
    hover_pos = np.array(handle_pos) + hover_offset
    
    print(f"DEBUG: Hover position: {hover_pos}")
    
    orientations = [
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.7, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, 0),
        quaternion_from_euler(np.pi - 0.5, 0, 0),
        quaternion_from_euler(np.pi - 0.4, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/4),
        quaternion_from_euler(np.pi - 0.6, 0, -np.pi/4),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/4),
        quaternion_from_euler(np.pi - 0.8, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.8, 0, 0),
    ]
    
    original_conf = env.get_robot_conf()
    
    for i, grasp_rot in enumerate(orientations):
        try:
            configs = env.robot.solve_ik_via_sampling(
                hover_pos.tolist(), quaternion=grasp_rot,
                max_configs=5, max_time_ms=100,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_hover = configs[0]
                print(f"DEBUG: Found hover config with orientation {i}")
                env.set_robot_conf(original_conf)
                return q_hover, grasp_rot, handle_pos
        except Exception as e:
            continue
    
    env.set_robot_conf(original_conf)
    print("ERROR: Could not find valid hover configuration")
    return None, None, None


def compute_grasp_config(env, handle_pos, hover_orientation):
    """
    Compute configuration to grasp the handle (move from hover to handle).
    Uses the same orientation as hover but moves closer to the handle.
    
    Returns: q_grasp or None if failed
    """
    # Grasp position: closer to the handle than hover
    # Move in the direction of the handle (positive X, negative Z from hover)
    grasp_offset = np.array([0.02, 0.0, 0.0])  # Closer to handle
    grasp_pos = np.array(handle_pos) + grasp_offset
    
    print(f"DEBUG: Grasp position: {grasp_pos}")
    
    original_conf = env.get_robot_conf()
    
    try:
        configs = env.robot.solve_ik_via_sampling(
            grasp_pos.tolist(), quaternion=hover_orientation,
            max_configs=10, max_time_ms=200,
            ignore_collisions=True
        )
        if configs is not None and len(configs) > 0:
            q_grasp = configs[0]
            print(f"DEBUG: Found grasp config")
            env.set_robot_conf(original_conf)
            return q_grasp
    except Exception as e:
        print(f"DEBUG: IK failed: {e}")
    
    env.set_robot_conf(original_conf)
    print("ERROR: Could not find valid grasp configuration")
    return None


def main():
    parser = argparse.ArgumentParser(description='Hover near handle and grasp it')
    args = parser.parse_args()

    print("=== Grill Task: Hover and Grasp Handle ===")

    env = ENV
    pr = env.pr

    # Open gripper first
    env.gripper.actuate(1.0, velocity=0.5)
    for _ in range(20):
        pr.step()

    # Step simulation a few times to stabilize
    for _ in range(10):
        pr.step()

    # Get current configuration
    current_q = env.get_robot_conf()
    print(f"Current robot config: {current_q[:3]}...")

    # ============================================
    # STEP 1: HOVER near the handle
    # ============================================
    print("\n" + "="*50)
    print("STEP 1: Hover near handle")
    print("="*50)

    q_hover, hover_orientation, handle_pos = compute_handle_hover_config(env)
    
    if q_hover is None:
        print("ERROR: Failed to compute hover configuration")
        pr.stop()
        pr.shutdown()
        return

    # Plan motion to hover position
    print("Planning motion to handle hover position...")
    motion_traj = env.compute_motion_plan(current_q, q_hover)
    
    if motion_traj is None:
        print("WARNING: Motion planning failed, trying direct move...")
        motion_traj = [q_hover]
    
    # Execute motion
    print("Executing motion to hover position...")
    execute_trajectory(env, pr, motion_traj)
    
    # Stabilize at hover position
    for _ in range(30):
        pr.step()
    
    tip = env.robot.get_tip()
    print(f"Hover position reached: {tip.get_position()}")

    # ============================================
    # STEP 2: MOVE TO GRASP POSITION
    # ============================================
    print("\n" + "="*50)
    print("STEP 2: Move to grasp position")
    print("="*50)

    q_grasp = compute_grasp_config(env, handle_pos, hover_orientation)
    
    if q_grasp is None:
        print("ERROR: Failed to compute grasp configuration")
        pr.stop()
        pr.shutdown()
        return

    # Move from hover to grasp (short linear move)
    current_q = env.get_robot_conf()
    
    # Interpolate between hover and grasp for smooth motion
    num_steps = 30
    for i in range(num_steps + 1):
        alpha = i / num_steps
        q_interp = [(1 - alpha) * current_q[j] + alpha * q_grasp[j] for j in range(len(current_q))]
        env.set_robot_conf(q_interp)
        pr.step()
    
    # Stabilize at grasp position
    for _ in range(20):
        pr.step()
    
    tip = env.robot.get_tip()
    print(f"Grasp position reached: {tip.get_position()}")

    # ============================================
    # STEP 3: CLOSE GRIPPER
    # ============================================
    print("\n" + "="*50)
    print("STEP 3: Close gripper on handle")
    print("="*50)

    print("Closing gripper...")
    env.gripper.actuate(0.0, velocity=0.2)  # Close gripper
    
    # Wait for gripper to close
    for _ in range(50):
        pr.step()
    
    print("Gripper closed on handle!")

    # Get final state
    tip = env.robot.get_tip()
    final_pos = tip.get_position()
    print(f"\nFinal gripper position: {final_pos}")
    print("Successfully grasped handle!")
    
    # Keep simulation running for visualization
    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == '__main__':
    main()
