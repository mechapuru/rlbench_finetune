#!/usr/bin/env python3
"""
grill_open_motion_gui.py

Complete sequence: Close the grill, then open it back up.
Builds on grill_open_hover_grasp_close_gui.py by adding the reverse (opening) motion.

Uses waypoints from scene:
- waypoint29: gripper START position for closing (handle when lid is OPEN)
- waypoint23: gripper END position for closing (handle when lid is CLOSED)
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
from pyrep.objects.dummy import Dummy

# Global environment instance
ENV = GrillTaskEnv(headless=False)


def get_scene_waypoints():
    """
    Extract key waypoints from the scene.
    Returns: (waypoint29_pos, waypoint29_orient, waypoint23_pos, waypoint23_orient) or Nones
    
    waypoint29: position when lid is OPEN (start of close / end of open)
    waypoint23: position when lid is CLOSED (end of close / start of open)
    """
    wp29_pos, wp29_ori = None, None
    wp23_pos, wp23_ori = None, None
    
    try:
        wp29 = Dummy('waypoint29')
        wp29_pos = wp29.get_position()
        wp29_ori = wp29.get_orientation()
        print(f"waypoint29 (OPEN lid position):")
        print(f"  Position: {wp29_pos}")
        print(f"  Orientation: {wp29_ori} rad")
        print(f"  Orientation: {np.degrees(wp29_ori)} deg")
    except Exception as e:
        print(f"WARNING: Could not find waypoint29: {e}")
    
    try:
        wp23 = Dummy('waypoint23')
        wp23_pos = wp23.get_position()
        wp23_ori = wp23.get_orientation()
        print(f"waypoint23 (CLOSED lid position):")
        print(f"  Position: {wp23_pos}")
        print(f"  Orientation: {wp23_ori} rad")
        print(f"  Orientation: {np.degrees(wp23_ori)} deg")
    except Exception as e:
        print(f"WARNING: Could not find waypoint23: {e}")
    
    # Also look for all available waypoints to understand the scene
    print("\nSearching for all available waypoints...")
    found = []
    for i in range(50):
        try:
            wp = Dummy(f'waypoint{i}')
            pos = wp.get_position()
            ori = wp.get_orientation()
            found.append(i)
            print(f"  waypoint{i}: pos={[f'{x:.3f}' for x in pos]}, ori_deg={[f'{np.degrees(x):.1f}' for x in ori]}")
        except:
            pass
    
    if found:
        print(f"Found waypoints: {found}")
    else:
        print("No numbered waypoints found in scene")
    
    return wp29_pos, wp29_ori, wp23_pos, wp23_ori


def execute_trajectory(env, pr, trajectory):
    """Execute a trajectory (list of joint configs) step by step."""
    for q in trajectory:
        env.set_robot_conf(q)
        pr.step()


def compute_hover_config_from_waypoint(env, waypoint_pos, waypoint_orient=None):
    """
    Compute a configuration to hover using the EXACT waypoint position.
    The waypoints from the scene define the exact gripper target positions.
    
    Returns: (q_hover, hover_orientation) or (None, None) if failed
    """
    # Hover offset: just a small Z offset above the exact waypoint position
    hover_offset = np.array([0.0, 0.0, 0.04])  # Small hover above grasp point
    hover_pos = np.array(waypoint_pos) + hover_offset
    
    print(f"DEBUG: Waypoint position: {waypoint_pos}")
    print(f"DEBUG: Hover position (waypoint + offset): {hover_pos}")
    
    # Try orientations - start with waypoint orientation if provided
    orientations = []
    if waypoint_orient is not None:
        # Convert euler to quaternion if orientation is in euler angles
        orientations.append(quaternion_from_euler(waypoint_orient[0], waypoint_orient[1], waypoint_orient[2]))
    
    # Add fallback orientations
    orientations.extend([
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.7, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, 0),
        quaternion_from_euler(np.pi - 0.5, 0, 0),
    ])
    
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
                print(f"Found hover config with orientation {i}")
                return q_hover, grasp_rot
        except Exception:
            continue
    
    env.set_robot_conf(original_conf)
    return None, None


def compute_grasp_config_from_waypoint(env, waypoint_pos, hover_orientation):
    """
    Compute configuration to grasp at the EXACT waypoint position.
    
    Returns: q_grasp or None if failed
    """
    # The waypoint IS the grasp position - use it directly
    grasp_pos = np.array(waypoint_pos)
    
    print(f"DEBUG: Grasp target (waypoint): {grasp_pos}")
    
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
    
    # Hover offset: position gripper in front of and slightly above handle
    # Original working values
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
    print(f"DEBUG: Grasp target position: {grasp_pos}")
    
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
    """Get the hinge point and axis for the lid rotation."""
    try:
        lid_joint = Joint('lid_joint')
        hinge_pos = lid_joint.get_position()
        joint_orient = lid_joint.get_orientation()
        current_angle = lid_joint.get_joint_position()
        
        print(f"DEBUG: Hinge position: {hinge_pos}")
        print(f"DEBUG: Hinge orientation: {joint_orient}")
        print(f"DEBUG: Current joint angle: {current_angle}")
        
        return hinge_pos, joint_orient, current_angle
    except Exception as e:
        print(f"ERROR: Could not get hinge info: {e}")
        return None, None, None


def compute_arc_waypoints_close(handle_pos, hinge_pos, num_waypoints=25):
    """Compute waypoints for CLOSING the lid (downward arc)."""
    handle_vec = np.array(handle_pos) - np.array(hinge_pos)
    radius_yz = np.sqrt(handle_vec[1]**2 + handle_vec[2]**2)
    current_angle_yz = np.arctan2(handle_vec[2], handle_vec[1])
    
    print(f"DEBUG: Arc radius (YZ plane): {radius_yz}")
    print(f"DEBUG: Current handle angle (YZ): {np.degrees(current_angle_yz)} degrees")
    
    # Original rotation amount that was working (~85 degrees)
    rotation_amount = -np.pi / 2.1
    
    waypoints = []
    angles = np.linspace(0, rotation_amount, num_waypoints)
    
    for angle_offset in angles:
        new_angle = current_angle_yz + angle_offset
        new_y = hinge_pos[1] + radius_yz * np.cos(new_angle)
        new_z = hinge_pos[2] + radius_yz * np.sin(new_angle)
        new_x = handle_pos[0]
        new_pos = np.array([new_x, new_y, new_z])
        waypoints.append((new_pos, angle_offset))
    
    return waypoints


def compute_arc_waypoints_open(handle_pos, hinge_pos, num_waypoints=25):
    """
    Compute waypoints for OPENING the lid (upward arc - reverse of closing).
    When lid is closed, handle is near horizontal. Opening lifts it up.
    """
    handle_vec = np.array(handle_pos) - np.array(hinge_pos)
    radius_yz = np.sqrt(handle_vec[1]**2 + handle_vec[2]**2)
    current_angle_yz = np.arctan2(handle_vec[2], handle_vec[1])
    
    print(f"DEBUG: Handle vector from hinge: {handle_vec}")
    print(f"DEBUG: Arc radius (YZ plane): {radius_yz}")
    print(f"DEBUG: Current handle angle (YZ): {np.degrees(current_angle_yz)} degrees")
    
    # Opening rotation - same magnitude as closing (~85 degrees)
    rotation_amount = np.pi / 2.1
    
    waypoints = []
    angles = np.linspace(0, rotation_amount, num_waypoints)
    
    for angle_offset in angles:
        new_angle = current_angle_yz + angle_offset
        new_y = hinge_pos[1] + radius_yz * np.cos(new_angle)
        new_z = hinge_pos[2] + radius_yz * np.sin(new_angle)
        new_x = handle_pos[0]
        new_pos = np.array([new_x, new_y, new_z])
        # Orientation offset: as lid opens (upward), gripper tilts back
        waypoints.append((new_pos, angle_offset))
    
    return waypoints


def compute_arc_trajectory(env, waypoints, base_tilt, direction='close'):
    """
    Compute robot configurations for each waypoint along the arc.
    direction: 'close' or 'open' - affects orientation adjustment
    """
    trajectory = []
    original_conf = env.get_robot_conf()
    prev_config = env.get_robot_conf()
    failed_count = 0
    
    for i, (pos, orient_offset) in enumerate(waypoints):
        # For closing: tilt increases (rotates down)
        # For opening: tilt decreases (rotates up)
        current_tilt = base_tilt + orient_offset
        
        # Try multiple orientations to find a valid IK solution
        orient_variations = [0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2, 0.25, -0.25, 0.3, -0.3]
        config_found = False
        
        for var in orient_variations:
            adjusted_orientation = quaternion_from_euler(
                current_tilt + var,
                0,
                np.pi/2
            )
            
            try:
                configs = env.robot.solve_ik_via_sampling(
                    pos.tolist(), quaternion=adjusted_orientation,
                    max_configs=30, max_time_ms=150,
                    ignore_collisions=True
                )
                
                if configs is not None and len(configs) > 0:
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
                        config_found = True
                        break
            except Exception:
                continue
        
        if not config_found:
            failed_count += 1
            # Instead of using prev_config, try to interpolate toward target
            # This helps prevent the trajectory from getting "stuck"
            trajectory.append(prev_config)
    
    if failed_count > 0:
        print(f"WARNING: {failed_count}/{len(waypoints)} waypoints failed IK")
    
    env.set_robot_conf(original_conf)
    return trajectory
    
    env.set_robot_conf(original_conf)
    return trajectory


def smooth_trajectory(trajectory, window_size=3):
    """Apply smoothing to the trajectory."""
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = []
    trajectory_array = np.array(trajectory)
    
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(trajectory), i + window_size // 2 + 1)
        smoothed.append(np.mean(trajectory_array[start_idx:end_idx], axis=0).tolist())
    
    return smoothed


def execute_smooth_trajectory(env, pr, trajectory, steps_per_waypoint=8, pause_steps=2):
    """Execute trajectory with interpolation for smoothness.
    More steps and pauses to maintain contact with handle."""
    if len(trajectory) < 2:
        for q in trajectory:
            env.set_robot_conf(q)
            for _ in range(pause_steps):
                pr.step()
        return
    
    for i in range(len(trajectory) - 1):
        q_start = np.array(trajectory[i])
        q_end = np.array(trajectory[i + 1])
        
        # More interpolation steps for smoother motion
        for j in range(steps_per_waypoint):
            alpha = j / steps_per_waypoint
            q_interp = (1 - alpha) * q_start + alpha * q_end
            env.set_robot_conf(q_interp.tolist())
            for _ in range(pause_steps):
                pr.step()
    
    env.set_robot_conf(trajectory[-1])
    for _ in range(pause_steps * 2):
        pr.step()


def main():
    parser = argparse.ArgumentParser(description='Complete grill close then open sequence')
    args = parser.parse_args()

    print("=== Grill Task: Close and Open Lid Sequence ===")

    env = ENV
    pr = env.pr

    # ============================================
    # FIRST: Extract waypoints from the scene
    # ============================================
    print("\n" + "="*60)
    print("EXTRACTING SCENE WAYPOINTS")
    print("="*60)
    
    wp29_pos, wp29_ori, wp23_pos, wp23_ori = get_scene_waypoints()
    
    # Open gripper first
    env.gripper.actuate(1.0, velocity=0.5)
    for _ in range(20):
        pr.step()

    for _ in range(10):
        pr.step()

    current_q = env.get_robot_conf()

    # ============================================
    # PART A: CLOSE THE GRILL
    # ============================================
    print("\n" + "="*60)
    print("PART A: CLOSING THE GRILL")
    print("="*60)

    # ============================================
    # STEP 1: HOVER near the handle (open position)
    # Using WAYPOINT29 - the exact position when lid is OPEN
    # ============================================
    print("\n" + "="*50)
    print("STEP 1: Hover near handle (using waypoint29)")
    print("="*50)

    if wp29_pos is None:
        print("WARNING: waypoint29 not found, falling back to handle position")
        q_hover_open, hover_orientation, handle_pos_open = compute_handle_hover_config(env)
    else:
        print(f"Using waypoint29 position: {wp29_pos}")
        print(f"Using waypoint29 orientation: {wp29_ori}")
        q_hover_open, hover_orientation = compute_hover_config_from_waypoint(env, wp29_pos, wp29_ori)
        handle_pos_open = wp29_pos  # The waypoint IS the grasp position
    
    if q_hover_open is None:
        print("ERROR: Failed to compute hover configuration")
        pr.stop()
        pr.shutdown()
        return

    motion_traj = env.compute_motion_plan(current_q, q_hover_open)
    if motion_traj is None:
        motion_traj = [q_hover_open]
    
    print("Moving to hover position...")
    execute_trajectory(env, pr, motion_traj)
    
    for _ in range(30):
        pr.step()
    
    print(f"Hover reached: {env.robot.get_tip().get_position()}")

    # ============================================
    # STEP 2: MOVE TO GRASP POSITION
    # Using WAYPOINT29 - move to exact waypoint position
    # ============================================
    print("\n" + "="*50)
    print("STEP 2: Move to grasp position (waypoint29)")
    print("="*50)

    if wp29_pos is not None:
        q_grasp = compute_grasp_config_from_waypoint(env, wp29_pos, hover_orientation)
    else:
        q_grasp = compute_grasp_config(env, handle_pos_open, hover_orientation)
    
    if q_grasp is None:
        print("ERROR: Failed to compute grasp configuration")
        pr.stop()
        pr.shutdown()
        return

    current_q = env.get_robot_conf()
    
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

    print("Closing gripper tightly...")
    # Close gripper very tight (0.0 = fully closed)
    env.gripper.actuate(0.0, velocity=0.1)  # Slower closing for better grip
    
    # Wait longer for gripper to fully close and grip firmly
    for _ in range(80):
        pr.step()
    
    # Apply extra closing force
    env.gripper.actuate(0.0, velocity=0.05)
    for _ in range(40):
        pr.step()
    
    print("Gripper closed firmly!")

    # ============================================
    # STEP 4: CIRCULAR ARC MOTION TO CLOSE LID
    # ============================================
    print("\n" + "="*50)
    print("STEP 4: Circular arc motion to close lid")
    print("="*50)

    hinge_pos, hinge_orient, current_joint_angle = get_hinge_info(env)
    
    if hinge_pos is None:
        hinge_pos = [handle_pos_open[0], handle_pos_open[1] + 0.3, handle_pos_open[2] - 0.2]

    try:
        handle = Shape('handle_visual')
        current_handle_pos = handle.get_position()
    except:
        current_handle_pos = handle_pos_open
    
    print(f"Current handle position: {current_handle_pos}")
    print(f"Hinge position: {hinge_pos}")

    print("Computing arc waypoints for closing...")
    # Use fewer waypoints for larger, more stable steps
    waypoints_close = compute_arc_waypoints_close(
        current_handle_pos, hinge_pos, num_waypoints=20
    )
    
    print(f"Generated {len(waypoints_close)} waypoints")
    
    print("Computing robot trajectory along arc...")
    base_tilt = np.pi - 0.6
    arc_trajectory_close = compute_arc_trajectory(env, waypoints_close, base_tilt, direction='close')
    
    print("Smoothing trajectory...")
    arc_trajectory_close = smooth_trajectory(arc_trajectory_close, window_size=3)
    
    print("Executing circular arc motion (slow and controlled)...")
    # Slower execution: more interpolation steps, more pauses
    execute_smooth_trajectory(env, pr, arc_trajectory_close, steps_per_waypoint=10, pause_steps=2)
    
    for _ in range(30):
        pr.step()
    
    print("Arc motion complete - lid closed!")
    closed_gripper_pos = env.robot.get_tip().get_position()
    print(f"Final gripper position: {closed_gripper_pos}")

    # ============================================
    # STEP 5: DETACH from handle (move away while closed)
    # ============================================
    print("\n" + "="*50)
    print("STEP 5: Detach from handle")
    print("="*50)

    current_tip_pos = env.robot.get_tip().get_position()
    current_q = env.get_robot_conf()
    
    # Move away in negative X (back from handle)
    detach_offset = np.array([-0.10, -0.02, 0.02])
    detach_pos = np.array(current_tip_pos) + detach_offset
    
    print(f"Detaching to: {detach_pos}")
    
    # Try multiple orientations for detach (matching closing rotation π/2.1)
    detach_orientations = [
        quaternion_from_euler(np.pi - 0.6 + (-np.pi/2.1), 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.4, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.7, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.8, 0, np.pi/2),
    ]
    
    q_detach = None
    for orient in detach_orientations:
        try:
            configs = env.robot.solve_ik_via_sampling(
                detach_pos.tolist(), quaternion=orient,
                max_configs=20, max_time_ms=200,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_detach = configs[0]
                print(f"Found detach config!")
                break
        except Exception:
            continue
    
    if q_detach is not None:
        print("Detaching from handle (gripper still closed)...")
        num_steps = 40
        for i in range(num_steps + 1):
            alpha = i / num_steps
            q_interp = [(1 - alpha) * current_q[j] + alpha * q_detach[j] for j in range(len(current_q))]
            env.set_robot_conf(q_interp)
            pr.step()
    else:
        print("WARNING: Could not find detach config, skipping detach")
    
    for _ in range(30):
        pr.step()
    
    print("Detached from handle!")

    # ============================================
    # STEP 6: RELEASE GRIPPER
    # ============================================
    print("\n" + "="*50)
    print("STEP 6: Release gripper")
    print("="*50)

    print("Opening gripper fully...")
    env.gripper.actuate(1.0, velocity=0.4)
    
    for _ in range(100):
        pr.step()
    
    print("Gripper released!")

    # ============================================
    # STEP 7: MOVE TO CLOSE-HOVER POSITION
    # ============================================
    print("\n" + "="*50)
    print("STEP 7: Move to close-hover position")
    print("="*50)

    current_tip_pos = env.robot.get_tip().get_position()
    current_q = env.get_robot_conf()
    
    close_hover_offset = np.array([-0.08, -0.05, 0.08])
    close_hover_pos = np.array(current_tip_pos) + close_hover_offset
    
    print(f"Close-hover position: {close_hover_pos}")
    
    orientations_to_try = [
        quaternion_from_euler(np.pi, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.3, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi, 0, 0),
    ]
    
    q_close_hover = None
    for orient in orientations_to_try:
        try:
            configs = env.robot.solve_ik_via_sampling(
                close_hover_pos.tolist(), quaternion=orient,
                max_configs=10, max_time_ms=200,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_close_hover = configs[0]
                break
        except Exception:
            continue
    
    if q_close_hover is not None:
        print("Moving to close-hover position...")
        num_steps = 50
        for i in range(num_steps + 1):
            alpha = i / num_steps
            q_interp = [(1 - alpha) * current_q[j] + alpha * q_close_hover[j] for j in range(len(current_q))]
            env.set_robot_conf(q_interp)
            pr.step()
    
    for _ in range(30):
        pr.step()
    
    print(f"Close-hover reached: {env.robot.get_tip().get_position()}")
    
    print("\n" + "="*60)
    print("GRILL CLOSED SUCCESSFULLY!")
    print("="*60)

    # Wait 10 seconds for the grill to settle before opening
    print("\nWaiting 10 seconds for grill to settle...")
    for i in range(500):  # ~10 seconds at 50 steps/sec
        pr.step()
        if i % 50 == 0:
            print(f"  {10 - i//50} seconds remaining...")
    print("Done waiting!")

    # ============================================
    # PART B: OPEN THE GRILL
    # ============================================
    print("\n" + "="*60)
    print("PART B: OPENING THE GRILL")
    print("="*60)

    # ============================================
    # STEP 8: HOVER NEAR HANDLE (closed position)
    # Using WAYPOINT23 - the exact position when lid is CLOSED
    # ============================================
    print("\n" + "="*50)
    print("STEP 8: Hover near handle (using waypoint23)")
    print("="*50)

    if wp23_pos is None:
        print("WARNING: waypoint23 not found, falling back to handle position")
        # Fallback to getting current handle position
        try:
            handle = Shape('handle_visual')
            handle_pos_closed = handle.get_position()
            print(f"Handle position (closed): {handle_pos_closed}")
        except:
            print("ERROR: Could not find handle")
            pr.stop()
            pr.shutdown()
            return
        q_hover_closed, hover_orient_used = compute_hover_config_from_waypoint(env, handle_pos_closed, None)
    else:
        print(f"Using waypoint23 position: {wp23_pos}")
        print(f"Using waypoint23 orientation: {wp23_ori}")
        handle_pos_closed = wp23_pos  # The waypoint IS the grasp position
        q_hover_closed, hover_orient_used = compute_hover_config_from_waypoint(env, wp23_pos, wp23_ori)
    
    if q_hover_closed is None:
        print("ERROR: Could not find hover config for closed lid")
        pr.stop()
        pr.shutdown()
        return
    
    current_q = env.get_robot_conf()
    
    # Motion plan or direct interpolation
    print("Moving to hover position...")
    num_steps = 60
    for i in range(num_steps + 1):
        alpha = i / num_steps
        q_interp = [(1 - alpha) * current_q[j] + alpha * q_hover_closed[j] for j in range(len(current_q))]
        env.set_robot_conf(q_interp)
        pr.step()
    
    for _ in range(30):
        pr.step()
    
    print(f"Hover reached: {env.robot.get_tip().get_position()}")

    # ============================================
    # STEP 9: MOVE TO GRASP POSITION
    # Using WAYPOINT23 - move to exact waypoint position
    # ============================================
    print("\n" + "="*50)
    print("STEP 9: Move to grasp position (waypoint23)")
    print("="*50)

    if wp23_pos is not None:
        q_grasp_closed = compute_grasp_config_from_waypoint(env, wp23_pos, hover_orient_used)
    else:
        # Fallback with offset
        grasp_pos_closed = np.array(handle_pos_closed)
        q_grasp_closed = None
        try:
            configs = env.robot.solve_ik_via_sampling(
                grasp_pos_closed.tolist(), quaternion=hover_orient_used,
                max_configs=20, max_time_ms=300,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_grasp_closed = configs[0]
        except Exception as e:
            print(f"WARNING: Grasp IK issue: {e}")
    
    if q_grasp_closed is None:
        print("ERROR: Could not find grasp config")
        pr.stop()
        pr.shutdown()
        return
    
    current_q = env.get_robot_conf()
    
    print("Moving to grasp position...")
    num_steps = 40
    for i in range(num_steps + 1):
        alpha = i / num_steps
        q_interp = [(1 - alpha) * current_q[j] + alpha * q_grasp_closed[j] for j in range(len(current_q))]
        env.set_robot_conf(q_interp)
        pr.step()
    
    for _ in range(20):
        pr.step()
    
    print(f"Grasp position reached: {env.robot.get_tip().get_position()}")

    # ============================================
    # STEP 10: CLOSE GRIPPER ON HANDLE
    # ============================================
    print("\n" + "="*50)
    print("STEP 10: Close gripper on handle")
    print("="*50)

    print("Closing gripper tightly...")
    # Close gripper very tight (same as closing part)
    env.gripper.actuate(0.0, velocity=0.1)  # Slower closing for better grip
    
    # Wait longer for gripper to fully close and grip firmly
    for _ in range(80):
        pr.step()
    
    # Apply extra closing force
    env.gripper.actuate(0.0, velocity=0.05)
    for _ in range(40):
        pr.step()
    
    print("Gripper closed firmly on handle!")

    # ============================================
    # STEP 11: CIRCULAR ARC MOTION TO OPEN LID
    # ============================================
    print("\n" + "="*50)
    print("STEP 11: Circular arc motion to open lid")
    print("="*50)

    hinge_pos, _, _ = get_hinge_info(env)
    
    if hinge_pos is None:
        hinge_pos = [handle_pos_closed[0], handle_pos_closed[1] + 0.25, handle_pos_closed[2]]
        print(f"Estimated hinge position: {hinge_pos}")

    current_tip_pos = env.robot.get_tip().get_position()
    print(f"Current gripper position: {current_tip_pos}")
    print(f"Hinge position: {hinge_pos}")

    print("Computing arc waypoints for opening...")
    # Use same number of waypoints as closing (20) for consistency
    waypoints_open = compute_arc_waypoints_open(
        current_tip_pos, hinge_pos, num_waypoints=20
    )
    
    print(f"Generated {len(waypoints_open)} waypoints")
    
    print("Computing robot trajectory along arc...")
    # For opening, the arc starts with closed lid orientation
    # Use same tilt value that the closing arc ended with
    base_tilt_open = np.pi - 0.6 + (-np.pi/2.1)  # Closed lid tilt
    arc_trajectory_open = compute_arc_trajectory(env, waypoints_open, base_tilt_open, direction='open')
    
    print("Smoothing trajectory...")
    arc_trajectory_open = smooth_trajectory(arc_trajectory_open, window_size=3)
    
    print(f"Arc trajectory has {len(arc_trajectory_open)} configs")
    
    print("Executing circular arc motion to open (slow and controlled)...")
    # Same execution parameters as closing
    execute_smooth_trajectory(env, pr, arc_trajectory_open, steps_per_waypoint=10, pause_steps=2)
    
    for _ in range(30):
        pr.step()
    
    print("Arc motion complete - lid opened!")
    print(f"Final gripper position: {env.robot.get_tip().get_position()}")

    # ============================================
    # STEP 12: DETACH AND RELEASE
    # ============================================
    print("\n" + "="*50)
    print("STEP 12: Detach and release gripper")
    print("="*50)

    current_tip_pos = env.robot.get_tip().get_position()
    current_q = env.get_robot_conf()
    
    # Detach: move away from handle
    detach_offset = np.array([-0.12, -0.02, 0.02])
    detach_pos = np.array(current_tip_pos) + detach_offset
    
    # After opening, gripper is back to original tilt
    open_end_tilt = np.pi - 0.6
    
    detach_orientations = [
        quaternion_from_euler(open_end_tilt, 0, np.pi/2),
        quaternion_from_euler(open_end_tilt + 0.1, 0, np.pi/2),
        quaternion_from_euler(open_end_tilt - 0.1, 0, np.pi/2),
    ]
    
    q_detach = None
    for orient in detach_orientations:
        try:
            configs = env.robot.solve_ik_via_sampling(
                detach_pos.tolist(), quaternion=orient,
                max_configs=20, max_time_ms=200,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_detach = configs[0]
                break
        except Exception:
            continue
    
    if q_detach is not None:
        print("Detaching from handle...")
        num_steps = 40
        for i in range(num_steps + 1):
            alpha = i / num_steps
            q_interp = [(1 - alpha) * current_q[j] + alpha * q_detach[j] for j in range(len(current_q))]
            env.set_robot_conf(q_interp)
            pr.step()
    else:
        print("WARNING: Could not find detach config, skipping")
    
    for _ in range(20):
        pr.step()
    
    print("Opening gripper...")
    env.gripper.actuate(1.0, velocity=0.4)
    
    for _ in range(80):
        pr.step()
    
    print("Gripper released!")

    print("\n" + "="*60)
    print("COMPLETE! Grill closed and opened successfully!")
    print("="*60)

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
