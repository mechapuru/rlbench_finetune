#!/usr/bin/env python3
"""
grill_open_hover_grasp_close_gui.py

Step 3: Complete sequence - hover, grasp handle, perform circular arc motion to close lid.
The lid rotates around a hinge (lid_joint), so the handle follows a circular arc.
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
from pyrep.objects.joint import Joint

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
    try:
        handle = Shape('handle_visual')
        handle_pos = handle.get_position()
        print(f"DEBUG: Handle position: {handle_pos}")
    except Exception:
        try:
            lid = Shape('lid_visual')
            lid_pos = lid.get_position()
            handle_pos = [lid_pos[0], lid_pos[1] - 0.1, lid_pos[2] + 0.1]
        except Exception:
            print("ERROR: Could not find handle or lid")
            return None, None, None
    
    # Hover offset: original working values
    hover_offset = np.array([-0.03, 0.02, 0.06])
    hover_pos = np.array(handle_pos) + hover_offset
    
    print(f"DEBUG: Handle position: {handle_pos}")
    print(f"DEBUG: Hover position: {hover_pos}")
    
    orientations = [
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.7, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, 0),
        quaternion_from_euler(np.pi - 0.5, 0, 0),
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
                env.set_robot_conf(original_conf)
                return q_hover, grasp_rot, handle_pos
        except Exception:
            continue
    
    env.set_robot_conf(original_conf)
    return None, None, None


def compute_grasp_config(env, handle_pos, hover_orientation):
    """Compute configuration to grasp the handle."""
    # Original working grasp offset
    grasp_offset = np.array([0.02, 0.0, 0.0])
    grasp_pos = np.array(handle_pos) + grasp_offset
    
    print(f"DEBUG: Handle position: {handle_pos}")
    print(f"DEBUG: Grasp target: {grasp_pos}")
    
    original_conf = env.get_robot_conf()
    
    try:
        configs = env.robot.solve_ik_via_sampling(
            grasp_pos.tolist(), quaternion=hover_orientation,
            max_configs=10, max_time_ms=200,
            ignore_collisions=True
        )
        if configs is not None and len(configs) > 0:
            env.set_robot_conf(original_conf)
            return configs[0]
    except Exception:
        pass
    
    env.set_robot_conf(original_conf)
    return None


def get_hinge_info(env):
    """
    Get the hinge point and axis for the lid rotation.
    Returns: (hinge_position, hinge_axis, current_angle)
    """
    try:
        lid_joint = Joint('lid_joint')
        hinge_pos = lid_joint.get_position()
        # The hinge axis is typically along the X axis for this type of lid
        # Get the joint's orientation to determine axis
        joint_orient = lid_joint.get_orientation()
        current_angle = lid_joint.get_joint_position()
        
        print(f"DEBUG: Hinge position: {hinge_pos}")
        print(f"DEBUG: Hinge orientation: {joint_orient}")
        print(f"DEBUG: Current joint angle: {current_angle}")
        
        return hinge_pos, joint_orient, current_angle
    except Exception as e:
        print(f"ERROR: Could not get hinge info: {e}")
        return None, None, None


def compute_arc_waypoints(handle_pos, hinge_pos, start_angle, end_angle, num_waypoints=25):
    """
    Compute waypoints along the circular arc that the handle follows.
    
    The handle rotates around the hinge point. We compute positions along this arc.
    
    Args:
        handle_pos: Current handle position
        hinge_pos: Position of the hinge/pivot point
        start_angle: Starting angle (current lid position)
        end_angle: Ending angle (closed position)
        num_waypoints: Number of waypoints for smooth motion
    
    Returns: List of (position, orientation_adjustment) tuples
    """
    # Vector from hinge to handle
    handle_vec = np.array(handle_pos) - np.array(hinge_pos)
    
    # The radius of the arc
    # Project onto the YZ plane (assuming hinge rotates around X axis)
    radius_yz = np.sqrt(handle_vec[1]**2 + handle_vec[2]**2)
    
    print(f"DEBUG: Handle vector from hinge: {handle_vec}")
    print(f"DEBUG: Arc radius (YZ plane): {radius_yz}")
    
    # Current angle of handle relative to hinge (in YZ plane)
    current_angle_yz = np.arctan2(handle_vec[2], handle_vec[1])
    print(f"DEBUG: Current handle angle (YZ): {np.degrees(current_angle_yz)} degrees")
    
    # For closing the lid, we rotate the handle downward
    # The arc goes from current position to closed position
    # Closing means rotating around X-axis, bringing handle down and forward
    
    # Original rotation amount that was working (~85 degrees)
    rotation_amount = -np.pi / 2.1
    
    waypoints = []
    angles = np.linspace(0, rotation_amount, num_waypoints)
    
    for i, angle_offset in enumerate(angles):
        # Rotate the handle position around the hinge (X-axis rotation)
        new_angle = current_angle_yz + angle_offset
        
        # New position in YZ plane
        new_y = hinge_pos[1] + radius_yz * np.cos(new_angle)
        new_z = hinge_pos[2] + radius_yz * np.sin(new_angle)
        
        # X stays relatively constant (maybe small adjustment)
        new_x = handle_pos[0]
        
        new_pos = np.array([new_x, new_y, new_z])
        
        # Orientation needs to change as we rotate - gripper tilts with the lid
        # The gripper needs to rotate around X-axis to match the handle orientation
        orientation_offset = angle_offset
        
        waypoints.append((new_pos, orientation_offset))
    
    return waypoints


def compute_arc_trajectory(env, waypoints, base_orientation):
    """
    Compute robot configurations for each waypoint along the arc.
    Uses the previous configuration as seed for smooth transitions.
    
    Returns: List of robot joint configurations
    """
    trajectory = []
    original_conf = env.get_robot_conf()
    prev_config = env.get_robot_conf()
    
    for i, (pos, orient_offset) in enumerate(waypoints):
        # Adjust orientation based on arc position
        # Base orientation is (np.pi - 0.6, 0, np.pi/2)
        # As we rotate, the gripper needs to follow the handle's rotation
        adjusted_orientation = quaternion_from_euler(
            np.pi - 0.6 + orient_offset,  # Tilt changes as lid closes
            0,
            np.pi/2
        )
        
        # Try to find IK solution close to previous config for smoothness
        try:
            configs = env.robot.solve_ik_via_sampling(
                pos.tolist(), quaternion=adjusted_orientation,
                max_configs=30, max_time_ms=150,  # More samples for better coverage
                ignore_collisions=True
            )
            
            if configs is not None and len(configs) > 0:
                # Find the config closest to previous for smooth motion
                best_config = None
                best_dist = float('inf')
                
                for config in configs:
                    dist = np.sum((np.array(config) - np.array(prev_config))**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_config = config
                
                if best_config is not None:
                    trajectory.append(best_config)
                    prev_config = best_config
                else:
                    print(f"WARNING: No good config at waypoint {i}, using previous")
                    trajectory.append(prev_config)
            else:
                print(f"WARNING: No IK solution at waypoint {i}, using previous")
                trajectory.append(prev_config)
                
        except Exception as e:
            print(f"WARNING: IK failed at waypoint {i}: {e}")
            trajectory.append(prev_config)
    
    env.set_robot_conf(original_conf)
    return trajectory


def smooth_trajectory(trajectory, window_size=3):
    """
    Apply smoothing to the trajectory to reduce jerkiness.
    Uses a simple moving average filter.
    """
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = []
    trajectory_array = np.array(trajectory)
    
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(trajectory), i + window_size // 2 + 1)
        smoothed.append(np.mean(trajectory_array[start_idx:end_idx], axis=0).tolist())
    
    return smoothed


def execute_smooth_trajectory(env, pr, trajectory, steps_per_waypoint=3, pause_steps=1):
    """
    Execute trajectory with interpolation between waypoints for extra smoothness.
    Added pause_steps to slow down motion.
    """
    if len(trajectory) < 2:
        for q in trajectory:
            env.set_robot_conf(q)
            for _ in range(pause_steps):
                pr.step()
        return
    
    for i in range(len(trajectory) - 1):
        q_start = np.array(trajectory[i])
        q_end = np.array(trajectory[i + 1])
        
        # Interpolate between waypoints
        for j in range(steps_per_waypoint):
            alpha = j / steps_per_waypoint
            q_interp = (1 - alpha) * q_start + alpha * q_end
            env.set_robot_conf(q_interp.tolist())
            # Multiple steps for slower motion
            for _ in range(pause_steps):
                pr.step()
    
    # Final configuration
    env.set_robot_conf(trajectory[-1])
    for _ in range(pause_steps):
        pr.step()


def main():
    parser = argparse.ArgumentParser(description='Complete grill closing sequence')
    args = parser.parse_args()

    print("=== Grill Task: Complete Lid Closing Sequence ===")

    env = ENV
    pr = env.pr

    # Open gripper first
    env.gripper.actuate(1.0, velocity=0.5)
    for _ in range(20):
        pr.step()

    for _ in range(10):
        pr.step()

    current_q = env.get_robot_conf()

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

    motion_traj = env.compute_motion_plan(current_q, q_hover)
    if motion_traj is None:
        motion_traj = [q_hover]
    
    print("Moving to hover position...")
    execute_trajectory(env, pr, motion_traj)
    
    for _ in range(30):
        pr.step()
    
    print(f"Hover reached: {env.robot.get_tip().get_position()}")

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

    current_q = env.get_robot_conf()
    
    # Smooth interpolation to grasp
    num_steps = 30
    for i in range(num_steps + 1):
        alpha = i / num_steps
        q_interp = [(1 - alpha) * current_q[j] + alpha * q_grasp[j] for j in range(len(current_q))]
        env.set_robot_conf(q_interp)
        pr.step()
    
    for _ in range(20):
        pr.step()
    
    print(f"Grasp position reached: {env.robot.get_tip().get_position()}")

    # ============================================
    # STEP 3: CLOSE GRIPPER
    # ============================================
    print("\n" + "="*50)
    print("STEP 3: Close gripper on handle")
    print("="*50)

    print("Closing gripper...")
    env.gripper.actuate(0.0, velocity=0.2)
    
    for _ in range(50):
        pr.step()
    
    print("Gripper closed!")

    # ============================================
    # STEP 4: CIRCULAR ARC MOTION TO CLOSE LID
    # ============================================
    print("\n" + "="*50)
    print("STEP 4: Circular arc motion to close lid")
    print("="*50)

    # Get hinge information
    hinge_pos, hinge_orient, current_joint_angle = get_hinge_info(env)
    
    if hinge_pos is None:
        print("WARNING: Could not get hinge info, estimating from handle")
        # Estimate hinge position (behind and below the handle)
        hinge_pos = [handle_pos[0], handle_pos[1] + 0.3, handle_pos[2] - 0.2]

    # Get current handle position (it may have moved slightly)
    try:
        handle = Shape('handle_visual')
        current_handle_pos = handle.get_position()
    except:
        current_handle_pos = handle_pos
    
    print(f"Current handle position: {current_handle_pos}")
    print(f"Hinge position: {hinge_pos}")

    # Compute arc waypoints - fewer waypoints but more interpolation for smoothness
    print("Computing arc waypoints...")
    waypoints = compute_arc_waypoints(
        current_handle_pos, hinge_pos,
        start_angle=0, end_angle=-np.pi/2,
        num_waypoints=30  # Reduced from 50 to have more reachable waypoints
    )
    
    print(f"Generated {len(waypoints)} waypoints")
    
    # Compute trajectory
    print("Computing robot trajectory along arc...")
    arc_trajectory = compute_arc_trajectory(env, waypoints, hover_orientation)
    
    # Apply smoothing
    print("Smoothing trajectory...")
    arc_trajectory = smooth_trajectory(arc_trajectory, window_size=5)
    
    # Execute the arc motion smoothly - SLOW motion with more interpolation
    print("Executing circular arc motion (slow)...")
    execute_smooth_trajectory(env, pr, arc_trajectory, steps_per_waypoint=5, pause_steps=3)
    
    # Extra stabilization
    for _ in range(30):
        pr.step()
    
    print("Arc motion complete!")
    final_arc_pos = env.robot.get_tip().get_position()
    print(f"Final gripper position: {final_arc_pos}")

    # ============================================
    # STEP 5: DETACH - Move away from handle BEFORE opening gripper
    # ============================================
    print("\n" + "="*50)
    print("STEP 5: Detach from handle (move away while still holding)")
    print("="*50)

    # Move the gripper away from the handle WHILE STILL CLOSED
    # This prevents the gripper from pushing the lid when opening
    current_tip_pos = env.robot.get_tip().get_position()
    current_q = env.get_robot_conf()
    
    # Detach offset - move away in negative X direction primarily
    # Keep gripper closed during this movement
    detach_offset = np.array([-0.15, -0.03, 0.02])  # More negative X to avoid collision
    detach_pos = np.array(current_tip_pos) + detach_offset
    
    print(f"Moving away from handle to: {detach_pos}")
    
    # Use the last orientation from arc motion
    last_arc_orient = quaternion_from_euler(np.pi - 0.6 + (-np.pi/3.0), 0, np.pi/2)
    
    try:
        configs = env.robot.solve_ik_via_sampling(
            detach_pos.tolist(), quaternion=last_arc_orient,
            max_configs=20, max_time_ms=200,
            ignore_collisions=True
        )
        if configs is not None and len(configs) > 0:
            q_detach = configs[0]
            
            # Slow, smooth movement away from handle
            print("Detaching from handle (gripper still closed)...")
            num_steps = 40
            for i in range(num_steps + 1):
                alpha = i / num_steps
                q_interp = [(1 - alpha) * current_q[j] + alpha * q_detach[j] for j in range(len(current_q))]
                env.set_robot_conf(q_interp)
                pr.step()
        else:
            print("WARNING: Could not find detach config")
    except Exception as e:
        print(f"WARNING: Detach IK failed: {e}")
    
    # Stabilize after detach
    for _ in range(30):
        pr.step()
    
    print("Detached from handle!")

    # ============================================
    # STEP 6: RELEASE GRIPPER (now safe to open)
    # ============================================
    print("\n" + "="*50)
    print("STEP 6: Release gripper (fully open)")
    print("="*50)

    print("Opening gripper fully...")
    env.gripper.actuate(1.0, velocity=0.4)  # Fully open, slightly faster
    
    # Wait for gripper to fully open
    for _ in range(100):
        pr.step()
    
    print("Gripper fully released!")

    # ============================================
    # STEP 7: MOVE TO CLOSE-HOVER POSITION
    # ============================================
    print("\n" + "="*50)
    print("STEP 7: Move to close-hover position")
    print("="*50)

    # Get current gripper position and move to hover position
    current_tip_pos = env.robot.get_tip().get_position()
    current_q = env.get_robot_conf()
    
    # Close-hover offset - move further back (negative X, positive Z)
    # This is the final safe hover position above the closed lid
    close_hover_offset = np.array([-0.08, -0.05, 0.08])
    close_hover_pos = np.array(current_tip_pos) + close_hover_offset
    
    print(f"Current tip position: {current_tip_pos}")
    print(f"Close-hover position: {close_hover_pos}")
    
    # Use an orientation suitable for the closed lid position
    # Gripper should be more vertical now since lid is closed
    close_hover_orient = quaternion_from_euler(np.pi, 0, np.pi/2)
    
    # Try multiple orientations if first fails
    orientations_to_try = [
        quaternion_from_euler(np.pi, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.3, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi, 0, 0),
    ]
    
    q_retreat = None
    for orient in orientations_to_try:
        try:
            configs = env.robot.solve_ik_via_sampling(
                close_hover_pos.tolist(), quaternion=orient,
                max_configs=10, max_time_ms=200,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_retreat = configs[0]
                break
        except Exception:
            continue
    
    if q_retreat is not None:
        # Smooth interpolation to retreat position
        print("Moving to close-hover position...")
        num_steps = 50
        for i in range(num_steps + 1):
            alpha = i / num_steps
            q_interp = [(1 - alpha) * current_q[j] + alpha * q_retreat[j] for j in range(len(current_q))]
            env.set_robot_conf(q_interp)
            pr.step()
    else:
        print("WARNING: Could not find close-hover config, staying in place")
    
    for _ in range(30):
        pr.step()
    
    print(f"Close-hover position reached: {env.robot.get_tip().get_position()}")

    print("\n" + "="*50)
    print("Lid closing sequence complete!")
    print("="*50)

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
