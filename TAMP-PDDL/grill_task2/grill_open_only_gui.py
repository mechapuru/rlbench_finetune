#!/usr/bin/env python3
"""
Grill OPEN ONLY script - assumes grill is already CLOSED
Run grill_open_hover_grasp_close_gui.py first, then run this script.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pddlstream'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RLBench'))

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint

from grill_task_env import GrillTaskEnv


def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion [x, y, z, w]."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    ]


def execute_trajectory(env, pr, trajectory, steps_per_config=5):
    """Execute a trajectory smoothly."""
    for config in trajectory:
        env.set_robot_conf(config)
        for _ in range(steps_per_config):
            pr.step()


def get_hinge_info(env):
    """Get the hinge joint position and axis."""
    try:
        lid_joint = Joint('lid_joint')
        hinge_pos = lid_joint.get_position()
        hinge_orient = lid_joint.get_orientation()
        print(f"Found lid_joint at position: {hinge_pos}")
        print(f"lid_joint orientation: {hinge_orient}")
        return hinge_pos, hinge_orient, lid_joint
    except:
        print("WARNING: Could not find lid_joint")
        return None, None, None


def compute_arc_waypoints_open(handle_pos, hinge_pos, num_waypoints=20):
    """
    Compute waypoints for OPENING the lid (reverse of closing).
    The handle rotates around the hinge axis.
    Opening goes from closed (handle down) to open (handle up).
    """
    handle_pos = np.array(handle_pos)
    hinge_pos = np.array(hinge_pos)
    
    # Vector from hinge to handle
    radius_vec = handle_pos - hinge_pos
    radius = np.linalg.norm(radius_vec)
    
    print(f"Hinge position: {hinge_pos}")
    print(f"Handle position (closed): {handle_pos}")
    print(f"Radius: {radius}")
    
    # Opening: POSITIVE rotation (reverse of closing)
    # When opening, we go from ~0 to +rotation_amount
    rotation_amount = np.pi / 2.1  # Same angle as closing, but positive
    
    waypoints = []
    for i in range(num_waypoints + 1):
        t = i / num_waypoints
        angle = t * rotation_amount  # Positive rotation for opening
        
        # Rotation axis is along Y (hinge axis)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotate the radius vector around Y axis
        # For opening: rotate from horizontal (closed) to tilted (open)
        new_x = radius_vec[0] * cos_a + radius_vec[2] * sin_a
        new_y = radius_vec[1]
        new_z = -radius_vec[0] * sin_a + radius_vec[2] * cos_a
        
        new_pos = hinge_pos + np.array([new_x, new_y, new_z])
        waypoints.append(new_pos)
    
    return waypoints


def compute_arc_trajectory(env, waypoints, base_tilt, direction='open'):
    """Compute robot configurations for each waypoint on the arc."""
    trajectory = []
    
    for i, waypoint in enumerate(waypoints):
        progress = i / max(len(waypoints) - 1, 1)
        
        if direction == 'open':
            # Opening: gripper tilts from closed position to open position
            # Start at base_tilt, rotate back by the arc angle
            current_tilt = base_tilt + progress * (np.pi / 2.1)
        else:
            current_tilt = base_tilt - progress * (np.pi / 2.1)
        
        orientations = [
            quaternion_from_euler(current_tilt, 0, np.pi/2),
            quaternion_from_euler(current_tilt + 0.1, 0, np.pi/2),
            quaternion_from_euler(current_tilt - 0.1, 0, np.pi/2),
            quaternion_from_euler(current_tilt, 0.1, np.pi/2),
            quaternion_from_euler(current_tilt, -0.1, np.pi/2),
        ]
        
        config_found = False
        for orient in orientations:
            try:
                configs = env.robot.solve_ik_via_sampling(
                    waypoint.tolist(), quaternion=orient,
                    max_configs=5, max_time_ms=100,
                    ignore_collisions=True
                )
                if configs is not None and len(configs) > 0:
                    trajectory.append(configs[0])
                    config_found = True
                    break
            except Exception:
                continue
        
        if not config_found:
            print(f"WARNING: Could not find config for waypoint {i}")
            if len(trajectory) > 0:
                trajectory.append(trajectory[-1])
    
    return trajectory


def smooth_trajectory(trajectory, window_size=3):
    """Apply moving average smoothing to trajectory."""
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = []
    for i in range(len(trajectory)):
        start = max(0, i - window_size // 2)
        end = min(len(trajectory), i + window_size // 2 + 1)
        window = trajectory[start:end]
        
        avg_config = []
        for j in range(len(trajectory[0])):
            avg_val = sum(config[j] for config in window) / len(window)
            avg_config.append(avg_val)
        smoothed.append(avg_config)
    
    return smoothed


def execute_smooth_trajectory(env, pr, trajectory, steps_per_waypoint=3, pause_steps=1):
    """Execute trajectory with smooth interpolation between waypoints."""
    if len(trajectory) < 2:
        return
    
    for i in range(len(trajectory) - 1):
        current = trajectory[i]
        next_config = trajectory[i + 1]
        
        for step in range(steps_per_waypoint):
            alpha = step / steps_per_waypoint
            interp = [(1 - alpha) * current[j] + alpha * next_config[j] for j in range(len(current))]
            env.set_robot_conf(interp)
            pr.step()
        
        for _ in range(pause_steps):
            pr.step()


def main():
    print("="*60)
    print("GRILL OPEN ONLY SCRIPT")
    print("Assumes grill is already CLOSED")
    print("="*60)

    # Initialize PyRep
    pr = PyRep()
    scene_file = os.path.join(os.path.dirname(__file__), '..', 'grill_task.ttt')
    pr.launch(scene_file, headless=False)
    pr.start()

    # Initialize environment
    env = GrillTaskEnv(pr)

    # Let simulation settle
    for _ in range(50):
        pr.step()

    # First, CLOSE the lid manually to set up the starting state
    print("\n" + "="*50)
    print("SETUP: Closing the lid joint to simulate closed state")
    print("="*50)
    
    try:
        lid_joint = Joint('lid_joint')
        # Set joint to closed position (negative rotation)
        lid_joint.set_joint_position(-np.pi/2.1)
        for _ in range(30):
            pr.step()
        print(f"Lid joint set to closed position: {lid_joint.get_joint_position()}")
    except Exception as e:
        print(f"WARNING: Could not set lid joint: {e}")

    for _ in range(50):
        pr.step()

    # Define orientations to try
    orientations_to_try = [
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.7, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, 0),
        quaternion_from_euler(np.pi - 0.5, 0, 0),
    ]

    # ============================================
    # STEP 1: HOVER near the handle (closed position - HORIZONTAL approach)
    # ============================================
    print("\n" + "="*50)
    print("STEP 1: Hover near handle (closed lid - horizontal approach)")
    print("="*50)

    # Get current handle position (now closed - handle is horizontal/flat)
    try:
        handle = Shape('handle_visual')
        handle_pos_closed = handle.get_position()
        print(f"Handle position (closed): {handle_pos_closed}")
    except:
        print("ERROR: Could not find handle")
        pr.stop()
        pr.shutdown()
        return
    
    # When lid is closed, handle position changes
    # Approach HORIZONTALLY from the side (same style as open lid approach)
    hover_offset_closed = np.array([-0.03, 0.02, 0.06])
    hover_pos_closed = np.array(handle_pos_closed) + hover_offset_closed
    
    print(f"Hover position for closed lid: {hover_pos_closed}")
    
    # Use same orientations as closing (tilted gripper)
    orientations_closed = [
        quaternion_from_euler(np.pi - 0.6, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.5, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.7, 0, np.pi/2),
        quaternion_from_euler(np.pi - 0.6, 0, 0),
        quaternion_from_euler(np.pi - 0.5, 0, 0),
    ]
    
    q_hover_closed = None
    hover_orient_closed = None
    
    for i, orient in enumerate(orientations_closed):
        try:
            configs = env.robot.solve_ik_via_sampling(
                hover_pos_closed.tolist(), quaternion=orient,
                max_configs=10, max_time_ms=200,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_hover_closed = configs[0]
                hover_orient_closed = orient
                print(f"Found hover config for closed lid with orientation {i}")
                break
        except Exception:
            continue
    
    if q_hover_closed is None:
        print("ERROR: Could not find hover config for closed lid")
        pr.stop()
        pr.shutdown()
        return
    
    current_q = env.get_robot_conf()
    motion_traj = env.compute_motion_plan(current_q, q_hover_closed)
    if motion_traj is None:
        motion_traj = [q_hover_closed]
    
    print("Moving to hover position (closed lid)...")
    execute_trajectory(env, pr, motion_traj)
    
    for _ in range(30):
        pr.step()
    
    print(f"Hover reached: {env.robot.get_tip().get_position()}")

    # ============================================
    # STEP 2: MOVE TO GRASP POSITION (closed lid)
    # ============================================
    print("\n" + "="*50)
    print("STEP 2: Move to grasp position (closed lid)")
    print("="*50)

    # Same grasp offset as closing
    grasp_offset_closed = np.array([0.02, 0.0, 0.0])
    grasp_pos_closed = np.array(handle_pos_closed) + grasp_offset_closed
    
    print(f"Grasp position: {grasp_pos_closed}")
    
    q_grasp_closed = None
    try:
        configs = env.robot.solve_ik_via_sampling(
            grasp_pos_closed.tolist(), quaternion=hover_orient_closed,
            max_configs=15, max_time_ms=200,
            ignore_collisions=True
        )
        if configs is not None and len(configs) > 0:
            q_grasp_closed = configs[0]
    except Exception as e:
        print(f"WARNING: Grasp IK failed: {e}")
    
    if q_grasp_closed is None:
        print("ERROR: Could not find grasp config for closed lid")
        pr.stop()
        pr.shutdown()
        return
    
    current_q = env.get_robot_conf()
    
    # Smooth interpolation from hover to grasp
    print("Moving to grasp position...")
    num_steps = 30
    for i in range(num_steps + 1):
        alpha = i / num_steps
        q_interp = [(1 - alpha) * current_q[j] + alpha * q_grasp_closed[j] for j in range(len(current_q))]
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
    # STEP 4: CIRCULAR ARC MOTION TO OPEN LID
    # ============================================
    print("\n" + "="*50)
    print("STEP 4: Circular arc motion to open lid")
    print("="*50)

    hinge_pos, _, _ = get_hinge_info(env)
    
    if hinge_pos is None:
        hinge_pos = [handle_pos_closed[0], handle_pos_closed[1] + 0.3, handle_pos_closed[2] - 0.2]

    # Get the CURRENT handle position (where gripper is now)
    current_tip_pos = env.robot.get_tip().get_position()
    print(f"Current gripper/handle position: {current_tip_pos}")
    print(f"Hinge position: {hinge_pos}")

    print("Computing arc waypoints for opening...")
    waypoints_open = compute_arc_waypoints_open(
        current_tip_pos, hinge_pos, num_waypoints=30
    )
    
    print(f"Generated {len(waypoints_open)} waypoints")
    
    print("Computing robot trajectory along arc...")
    # For opening, start with same tilt as closing (np.pi - 0.6)
    base_tilt_open = np.pi - 0.6
    arc_trajectory_open = compute_arc_trajectory(env, waypoints_open, base_tilt_open, direction='open')
    
    print("Smoothing trajectory...")
    arc_trajectory_open = smooth_trajectory(arc_trajectory_open, window_size=5)
    
    print("Executing circular arc motion to open (slow)...")
    execute_smooth_trajectory(env, pr, arc_trajectory_open, steps_per_waypoint=5, pause_steps=3)
    
    for _ in range(30):
        pr.step()
    
    print("Arc motion complete - lid opened!")
    print(f"Final gripper position: {env.robot.get_tip().get_position()}")

    # ============================================
    # STEP 5: DETACH from handle
    # ============================================
    print("\n" + "="*50)
    print("STEP 5: Detach from handle")
    print("="*50)

    current_tip_pos = env.robot.get_tip().get_position()
    current_q = env.get_robot_conf()
    
    # Same detach offset as closing
    detach_offset_open = np.array([-0.15, -0.03, 0.02])
    detach_pos_open = np.array(current_tip_pos) + detach_offset_open
    
    print(f"Detaching to: {detach_pos_open}")
    
    # Use orientation from end of arc
    open_orient = quaternion_from_euler(np.pi - 0.6 + (np.pi/2.1), 0, np.pi/2)
    
    try:
        configs = env.robot.solve_ik_via_sampling(
            detach_pos_open.tolist(), quaternion=open_orient,
            max_configs=20, max_time_ms=200,
            ignore_collisions=True
        )
        if configs is not None and len(configs) > 0:
            q_detach_open = configs[0]
            
            print("Detaching from handle (gripper still closed)...")
            num_steps = 40
            for i in range(num_steps + 1):
                alpha = i / num_steps
                q_interp = [(1 - alpha) * current_q[j] + alpha * q_detach_open[j] for j in range(len(current_q))]
                env.set_robot_conf(q_interp)
                pr.step()
    except Exception as e:
        print(f"WARNING: Detach IK failed: {e}")
    
    for _ in range(30):
        pr.step()

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
    # STEP 7: MOVE TO HOVER POSITION
    # ============================================
    print("\n" + "="*50)
    print("STEP 7: Move to hover position")
    print("="*50)

    current_tip_pos = env.robot.get_tip().get_position()
    current_q = env.get_robot_conf()
    
    open_hover_offset = np.array([-0.08, -0.05, 0.08])
    open_hover_pos = np.array(current_tip_pos) + open_hover_offset
    
    print(f"Hover position: {open_hover_pos}")
    
    q_open_hover = None
    for orient in orientations_to_try:
        try:
            configs = env.robot.solve_ik_via_sampling(
                open_hover_pos.tolist(), quaternion=orient,
                max_configs=10, max_time_ms=200,
                ignore_collisions=True
            )
            if configs is not None and len(configs) > 0:
                q_open_hover = configs[0]
                break
        except Exception:
            continue
    
    if q_open_hover is not None:
        print("Moving to hover position...")
        num_steps = 50
        for i in range(num_steps + 1):
            alpha = i / num_steps
            q_interp = [(1 - alpha) * current_q[j] + alpha * q_open_hover[j] for j in range(len(current_q))]
            env.set_robot_conf(q_interp)
            pr.step()
    
    for _ in range(30):
        pr.step()
    
    print(f"Hover reached: {env.robot.get_tip().get_position()}")

    print("\n" + "="*60)
    print("COMPLETE! Grill opened successfully!")
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
