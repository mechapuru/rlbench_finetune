"""
Script 4: Pick mug from cupboard + Place on placement_boundary
Extends script3 with place functionality.
"""
import os
import sys
import numpy as np

def _configure_qt():
    """Keep Qt quiet and point it at CoppeliaSim's plugins without forcing offscreen."""
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

from pddlstream.language.constants import PDDLProblem
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

os.environ["HEADLESS"] = "False"

from rlbench_kitchen_streams import ENV, get_stream_map
import math


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk
    q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
    return q


def normalize_quaternion(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quaternion_conjugate(q):
    q = np.array(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)


def quaternion_rotate_vector(q, v):
    qn = normalize_quaternion(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=float)
    v_rot = quaternion_multiply(quaternion_multiply(qn, vq), quaternion_conjugate(qn))
    return v_rot[:3]


def compute_tip_attachment(gripper_tip, obj):
    tip_pos = np.array(gripper_tip.get_position(), dtype=float)
    tip_quat = normalize_quaternion(gripper_tip.get_quaternion())
    obj_pos = np.array(obj.get_position(), dtype=float)
    obj_quat = normalize_quaternion(obj.get_quaternion())

    local_pos = quaternion_rotate_vector(quaternion_conjugate(tip_quat), obj_pos - tip_pos)
    local_quat = normalize_quaternion(quaternion_multiply(quaternion_conjugate(tip_quat), obj_quat))
    return local_pos, local_quat


def update_attached_pose(gripper_tip, obj, local_pos, local_quat):
    tip_pos = np.array(gripper_tip.get_position(), dtype=float)
    tip_quat = normalize_quaternion(gripper_tip.get_quaternion())

    obj_pos = tip_pos + quaternion_rotate_vector(tip_quat, local_pos)
    obj_quat = normalize_quaternion(quaternion_multiply(tip_quat, local_quat))

    obj.set_position(obj_pos.tolist())
    obj.set_quaternion(obj_quat.tolist())


def main():
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

    # Get the mug object
    mug = env.get_object("mug3")
    if mug is None:
        print("ERROR: mug3 not found in scene.")
        pr.stop()
        pr.shutdown()
        return

    mug.set_dynamic(False)
    pose = mug.get_pose()

    # Use the mug on top of the box as the canonical upright orientation.
    upright_ref_mug = env.get_object("mug2")
    if upright_ref_mug is not None:
        upright_mug_quat = normalize_quaternion(upright_ref_mug.get_quaternion()).tolist()
        print(f"Using mug2 as upright orientation reference: {upright_mug_quat}")
    else:
        # Fallback: identity quaternion in world frame.
        upright_mug_quat = [0.0, 0.0, 0.0, 1.0]
        print("WARNING: mug2 not found. Falling back to identity upright quaternion.")

    # ========== PHASE 1: CUPBOARD PICK ==========
    print("\n" + "="*60)
    print("PHASE 1: CUPBOARD PICK")
    print("="*60)
    
    print("Computing horizontal hover configuration for cupboard mug...")
    
    y_shift = 0.0
    hover_dists = [0.35, 0.30, 0.40]
    z_offsets = [0.001]
    
    q_hover = None
    successful_grasp_quat = None
    original_conf = env.get_robot_conf()
    
    base_ry = np.pi/2
    grasp_quats = [
        quaternion_from_euler(0, base_ry, 0),
        quaternion_from_euler(np.pi, base_ry, 0),
    ]
    
    target_z = pose[2]
    grasp_depth_offset = 0.03
    grasp_pos = None
    hover_pos = None
    
    try:
        for h_dist in hover_dists:
            for z_off in z_offsets:
                if q_hover is not None: break
                
                print(f"Trying hover parameters: dist={h_dist}, z_off={z_off}...")
                hover_pos = [pose[0] - h_dist, pose[1] + y_shift, target_z]
                grasp_pos = [pose[0] + grasp_depth_offset, pose[1] + y_shift, target_z]

                for grasp_rot in grasp_quats:
                    path_configs = env.robot.solve_ik_via_sampling(hover_pos, quaternion=grasp_rot, max_configs=50, max_time_ms=1000, ignore_collisions=True)
                    if path_configs is not None and len(path_configs) > 0:
                        for q in path_configs:
                            env.set_robot_conf(q)
                            if env.robot.check_collision():
                                continue
                            try:
                                path_check = env.robot.get_linear_path(position=grasp_pos, quaternion=grasp_rot, steps=20, ignore_collisions=True)
                                if path_check:
                                    q_hover = q
                                    successful_grasp_quat = grasp_rot
                                    print(f"Found valid configuration with dist={h_dist}, z_off={z_off}")
                                    break
                            except Exception:
                                pass
                    if q_hover is not None: break
            if q_hover is not None: break
                
        if q_hover is None:
            raise RuntimeError("Could not find valid horizontal hover configuration")
            
    except Exception as e:
        print(f"ERROR: Horizontal hover computation failed: {e}")
        env.set_robot_conf(original_conf)
        pr.stop()
        pr.shutdown()
        return
        
    env.set_robot_conf(original_conf)

    # PDDL Planning for Move to Hover
    print("Setting up PDDL problem for 'move' action...")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    q_hover_tuple = tuple(q_hover)

    init = [
        ('conf', q_home_tuple),
        ('conf', q_hover_tuple),
        ('at-conf', q_home_tuple),
    ]

    goal = ('at-conf', q_hover_tuple)

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving PDDL problem...")
    solution = solve(problem, algorithm='adaptive', verbose=False)
    plan, cost, evaluations = solution

    gripper_tip = None
    mug_tip_offset_local = None
    mug_tip_quat_local = None

    if plan:
        print(f"PDDL Plan found with cost: {cost}")
        for action in plan:
            print(f"Action: {action.name}")
            
            if action.name == 'move':
                q1, q2, traj = action.args
                print(f"Executing move via PDDL plan...")
                
                full_traj = []
                for i in range(len(traj)-1):
                    start = np.array(traj[i])
                    end = np.array(traj[i+1])
                    steps = 60
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_traj.append((1-t)*start + t*end)
                full_traj.append(traj[-1])

                print(f"DEBUG: Executing trajectory with {len(full_traj)} points.")

                for conf in full_traj:
                    env.set_robot_conf(conf)
                    pr.step()
        
        env.save_conf("hover_pick", q_hover)
        print("Saved 'hover_pick' configuration after moving to hover pose.")
        
        # Open gripper
        print("Opening gripper at hover...")
        env.gripper.release()
        for _ in range(30):
            pr.step()
        
        # Approach to grasp
        print("Planning approach to grasp...")
        
        path_configs_grasp = env.robot.solve_ik_via_sampling(grasp_pos, quaternion=successful_grasp_quat, max_configs=20, max_time_ms=500, ignore_collisions=True)
        
        q_grasp = None
        if path_configs_grasp is not None and len(path_configs_grasp) > 0:
            path_configs_grasp = sorted(path_configs_grasp, key=lambda q: np.linalg.norm(np.array(q) - np.array(q_hover)))
            q_grasp = path_configs_grasp[0]
            print("Found valid grasp configuration.")
        else:
            print("ERROR: Could not find valid grasp configuration!")
            pr.stop()
            pr.shutdown()
            return

        path_approach = env.robot.get_linear_path(position=grasp_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True)
        
        if path_approach:
            print("Executing approach trajectory...")
            print(f"DEBUG: Mug pose: {pose[:3]}")
            print(f"DEBUG: Grasp target pos: {grasp_pos}")
            
            mug.set_dynamic(False)
            
            traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
            for conf in traj_approach:
                env.set_robot_conf(conf)
                pr.step()
            
            # Close Gripper
            print("Closing gripper...")
            env.gripper.actuate(0.0, 0.1) 
            for _ in range(50):
                pr.step()
            
            # Record offset for tracking
            print("Recording rigid mug-to-gripper transform...")
            gripper_tip = env.robot.get_tip()
            mug_tip_offset_local, mug_tip_quat_local = compute_tip_attachment(gripper_tip, mug)
            print(f"DEBUG: Mug local position offset: {mug_tip_offset_local}")
            print(f"DEBUG: Mug local quaternion offset: {mug_tip_quat_local}")
            
            mug.set_dynamic(False)
            print("Grasp successful!")
                
            # Retrieve to hover
            print("Planning retrieve motion (Grasp -> Hover)...")
            
            path_retrieve = None
            try:
                retrieve_pos = list(hover_pos)
                retrieve_pos[2] += 0.02
                path_retrieve = env.robot.get_linear_path(position=retrieve_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True)
            except Exception as e:
                print(f"Linear retrieve failed with error: {e}")
                path_retrieve = None
            
            if path_retrieve:
                print("Executing retrieve trajectory...")
                traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
                for conf in traj_retrieve:
                    env.set_robot_conf(conf)
                    pr.step()
                    update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
                
                print("Retrieve complete.")
            else:
                print("Linear retrieve failed. Executing fallback...")
                traj_fallback = env._interpolate_joint_path(env.get_robot_conf(), q_hover, steps=100, check_collisions=False)
                if traj_fallback:
                    for conf in traj_fallback:
                        env.set_robot_conf(conf)
                        pr.step()
                        update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
                    print("Retrieve complete (Fallback).")
                    
        else:
            print("ERROR: Could not plan linear approach!")
            pr.stop()
            pr.shutdown()
            return

    else:
        print("ERROR: No PDDL plan found for move action!")
        pr.stop()
        pr.shutdown()
        return

    # ========== PHASE 2: PLACE ON PLACEMENT_BOUNDARY ==========
    print("\n" + "="*60)
    print("PHASE 2: PLACE ON PLACEMENT_BOUNDARY")
    print("="*60)
    
    # Find placement position
    print("Finding placement position...")
    try:
        place_pose = env.find_best_placement(mug, 'placement_boundary')
        print(f"Found place pose: {place_pose[:3]}")
    except Exception as e:
        print(f"ERROR: Could not find placement: {e}")
        # Fallback: use a hardcoded position in placement_boundary
        place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]  # Approximate placement_boundary center
        print(f"Using fallback place pose: {place_pose[:3]}")

    if mug_tip_offset_local is None or mug_tip_quat_local is None:
        print("ERROR: Missing mug attachment transform from pick phase.")
        pr.stop()
        pr.shutdown()
        return
    
    # Compute final upright mug pose on table surface
    min_x, max_x, min_y, max_y, min_z, max_z = mug.get_bounding_box()
    place_surface_z = float(place_pose[2])
    place_object_z = place_surface_z - float(min_z) + 0.001
    final_place_obj_pos = np.array([place_pose[0], place_pose[1], place_object_z], dtype=float)
    
    # Standard vertical grasp orientation for placing (gripper pointing down)
    place_quats = []
    for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
        q = quaternion_from_euler(np.pi, 0, angle)
        place_quats.append(q)
    
    print("Computing place hover configuration...")
    
    q_place_hover = None
    successful_place_quat = None
    successful_place_pos = None
    pre_place_search_conf = env.get_robot_conf()
    
    for place_quat in place_quats:
        tip_offset_world = quaternion_rotate_vector(place_quat, mug_tip_offset_local)
        candidate_place_pos = (final_place_obj_pos - tip_offset_world).tolist()
        candidate_hover_pos = [candidate_place_pos[0], candidate_place_pos[1], candidate_place_pos[2] + 0.12]

        path_configs = env.robot.solve_ik_via_sampling(candidate_hover_pos, quaternion=place_quat, max_configs=20, max_time_ms=500, ignore_collisions=True)
        if path_configs is not None and len(path_configs) > 0:
            for q in path_configs:
                env.set_robot_conf(q)
                if not env.robot.check_collision():
                    # Also check if we can reach the place position
                    place_configs = env.robot.solve_ik_via_sampling(candidate_place_pos, quaternion=place_quat, max_configs=5, max_time_ms=200, ignore_collisions=True)
                    if place_configs is not None and len(place_configs) > 0:
                        q_place_hover = q
                        successful_place_quat = place_quat
                        successful_place_pos = candidate_place_pos
                        break
        if q_place_hover is not None:
            break

    env.set_robot_conf(pre_place_search_conf)
    
    if q_place_hover is None:
        print("ERROR: Could not find place hover configuration!")
        pr.stop()
        pr.shutdown()
        return
    
    print("Found place hover configuration.")
    
    # Move from current position to place hover
    print("Moving to place hover position...")
    current_conf = env.get_robot_conf()
    
    # Use joint space interpolation
    traj_to_place = env._interpolate_joint_path(current_conf, q_place_hover, steps=150, check_collisions=False)
    
    if traj_to_place:
        for conf in traj_to_place:
            env.set_robot_conf(conf)
            pr.step()
            update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
        print("Reached place hover position.")
    else:
        print("ERROR: Could not plan motion to place hover!")
        pr.stop()
        pr.shutdown()
        return
    
    # Lower to place position
    print("Lowering to place position...")
    
    path_lower = None
    try:
        path_lower = env.robot.get_linear_path(position=successful_place_pos, quaternion=successful_place_quat, steps=100, ignore_collisions=True)
    except:
        pass
    
    if path_lower:
        traj_lower = path_lower._path_points.reshape(-1, 7).tolist()
        for conf in traj_lower:
            env.set_robot_conf(conf)
            pr.step()
            update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
    else:
        # Fallback: try joint interpolation to place IK
        print("Linear lower failed, trying joint interpolation fallback...")
        place_configs = env.robot.solve_ik_via_sampling(successful_place_pos, quaternion=successful_place_quat, max_configs=10, max_time_ms=500, ignore_collisions=True)
        if place_configs is not None and len(place_configs) > 0:
            q_place = place_configs[0]
            traj_lower = env._interpolate_joint_path(env.get_robot_conf(), q_place, steps=50, check_collisions=False)
            if traj_lower:
                for conf in traj_lower:
                    env.set_robot_conf(conf)
                    pr.step()
                    update_attached_pose(gripper_tip, mug, mug_tip_offset_local, mug_tip_quat_local)
        else:
            print("WARNING: Could not lower to place position, releasing at hover.")
    
    print("Lowered to place position.")

    # Lock the mug upright on the support surface before release
    final_place_obj_pose = [
        float(final_place_obj_pos[0]),
        float(final_place_obj_pos[1]),
        float(final_place_obj_pos[2]),
        float(upright_mug_quat[0]),
        float(upright_mug_quat[1]),
        float(upright_mug_quat[2]),
        float(upright_mug_quat[3]),
    ]
    mug.set_pose(final_place_obj_pose)
    for _ in range(10):
        pr.step()
    
    # Release gripper
    print("Releasing gripper...")
    hold_q = env.get_robot_conf()
    env.gripper.release()
    env.gripper.actuate(1.0, 0.2)
    for _ in range(60):
        env.set_robot_conf(hold_q)
        mug.set_pose(final_place_obj_pose)
        pr.step()
    
    print("Mug placed!")
    
    # Lift back up
    print("Lifting gripper...")
    path_lift = None
    try:
        lift_pos = [successful_place_pos[0], successful_place_pos[1], successful_place_pos[2] + 0.15]
        path_lift = env.robot.get_linear_path(position=lift_pos, quaternion=successful_place_quat, steps=50, ignore_collisions=True)
    except:
        pass
    
    if path_lift:
        traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
        for conf in traj_lift:
            env.set_robot_conf(conf)
            pr.step()
    else:
        # Fallback: joint interpolation to hover
        traj_lift = env._interpolate_joint_path(env.get_robot_conf(), q_place_hover, steps=50, check_collisions=False)
        if traj_lift:
            for conf in traj_lift:
                env.set_robot_conf(conf)
                pr.step()

    # Keep final mug pose fixed and upright for this scripted demo.
    mug.set_pose(final_place_obj_pose)
    mug.set_dynamic(False)
    for _ in range(20):
        pr.step()
    
    # Return to home
    print("Returning to home...")
    traj_home = env._interpolate_joint_path(env.get_robot_conf(), home_q, steps=100, check_collisions=False)
    if traj_home:
        for conf in traj_home:
            env.set_robot_conf(conf)
            pr.step()
    
    print("\n" + "="*60)
    print("COMPLETE: Cupboard Pick + Place on Placement Boundary")
    print("="*60)
    
    # Leave sim running
    print("Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
