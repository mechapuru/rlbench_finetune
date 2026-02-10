#!/usr/bin/env python3
"""
grill_open_hover_gui.py

Step 1: Move the robot to hover in front of the grill lid handle (when lid is open).
This is the first step before grasping and closing the lid.
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
    The handle is on the open lid, so we need to approach it from an angle.
    
    Returns: (q_hover, hover_orientation) or (None, None) if failed
    """
    # Try to get the handle position
    try:
        handle = Shape('handle_visual')
        handle_pos = handle.get_position()
        handle_orient = handle.get_orientation()
        print(f"DEBUG: Handle position from handle_visual: {handle_pos}")
        print(f"DEBUG: Handle orientation: {handle_orient}")
    except Exception:
        # If no handle_visual, estimate from lid position
        try:
            lid = Shape('lid_visual')
            lid_pos = lid.get_position()
            handle_pos = [lid_pos[0], lid_pos[1] - 0.1, lid_pos[2] + 0.1]
            print(f"DEBUG: Estimated handle position from lid: {handle_pos}")
        except Exception:
            print("ERROR: Could not find handle or lid")
            return None, None
    
    # The handle is bent - we need to approach perpendicular to it
    # Offset: negative X (towards robot's left), slightly forward in Y, higher Z
    hover_offset = np.array([-0.03, 0.02, 0.06])  # More negative X
    hover_pos = np.array(handle_pos) + hover_offset
    
    print(f"DEBUG: Hover position: {hover_pos}")
    
    # Orientation: gripper needs to be tilted to grab the bent handle perpendicularly
    # The handle curves, so we need the gripper tilted around X axis
    # Try orientations with tilt to match the handle angle
    
    orientations = [
        # Tilted gripper - bent forward to match handle angle
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/2),  # Tilted forward, rotated
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.7, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, 0),
        quaternion_from_euler(np.pi - 0.5, 0, 0),
        quaternion_from_euler(np.pi - 0.4, 0, np.pi/2),
        # More variations with different Z rotations
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/4),
        quaternion_from_euler(np.pi - 0.6, 0, -np.pi/4),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/4),
        # Steeper tilt
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
                return q_hover, grasp_rot
        except Exception as e:
            continue
    
    env.set_robot_conf(original_conf)
    print("ERROR: Could not find valid hover configuration")
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Hover near the grill lid handle')
    args = parser.parse_args()

    print("=== Grill Task: Hover Near Handle ===")

    env = ENV
    pr = env.pr

    # Step simulation a few times to stabilize
    for _ in range(10):
        pr.step()

    # Get current configuration
    current_q = env.get_robot_conf()
    print(f"Current robot config: {current_q[:3]}...")

    # Compute hover position near the handle
    print("\nComputing hover position near handle...")
    q_hover, hover_orientation = compute_handle_hover_config(env)
    
    if q_hover is None:
        print("ERROR: Failed to compute hover configuration")
        pr.stop()
        pr.shutdown()
        return

    # Plan motion to hover position
    print("\nPlanning motion to handle hover position...")
    motion_traj = env.compute_motion_plan(current_q, q_hover)
    
    if motion_traj is None:
        print("WARNING: Motion planning failed, trying direct move...")
        # Try direct interpolation
        motion_traj = [q_hover]
    
    # Execute motion
    print("Executing motion to hover position...")
    execute_trajectory(env, pr, motion_traj)
    
    # Stabilize at hover position
    for _ in range(30):
        pr.step()
    
    # Get final position
    tip = env.robot.get_tip()
    final_pos = tip.get_position()
    print(f"\nFinal gripper position: {final_pos}")
    print("Successfully reached handle hover position!")
    
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
