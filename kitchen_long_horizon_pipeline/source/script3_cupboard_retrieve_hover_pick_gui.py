import os
import sys
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

# Add pddlstream to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

from pddlstream.language.constants import PDDLProblem
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import ENV from streams to share the instance
from rlbench_kitchen_streams import ENV, get_stream_map
# from video_recorder import VideoRecorder # Disable video recorder


def main():
    # Use the shared ENV
    env = ENV
    pr = env.pr

    # recorder = VideoRecorder(env) # Disable video recorder

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
        # recorder.record_step()

    # Get the mug object and its current pose
    mug = env.get_object("mug3")
    if mug is None:
        print("ERROR: mug3 not found in scene.")
        # recorder.release()
        pr.stop()
        pr.shutdown()
        return

    # Force the mug to be static so it doesn't jitter/slide
    mug.set_dynamic(False)

    pose = mug.get_pose()  # [x, y, z, qx, qy, qz, qw]

    # Custom Horizontal Hover Logic for Cupboard
    print("Computing horizontal hover configuration for cupboard mug...")
    
    # Shift Y to grasp circumference (centered)
    y_shift = 0.0
    
    # Search grid for valid hover parameters
    # Sometimes exact coordinates fail due to singularities or minor collisions
    hover_dists = [0.35, 0.30, 0.40]
    z_offsets = [0.001] # No slant, same height as grasp
    
    q_hover = None
    successful_grasp_quat = None
    original_conf = env.get_robot_conf()
    
    # Grasp Orientation: Horizontal (Fingers Horizontal)
    # Base orientation: Ry=pi/2 (Z points +X)
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

    base_ry = np.pi/2
    grasp_quats = [
        quaternion_from_euler(0, base_ry, 0),       # Fingers Horizontal
        quaternion_from_euler(np.pi, base_ry, 0),   # Fingers Horizontal (flipped)
    ]
    
    try:
        for h_dist in hover_dists:
            for z_off in z_offsets:
                if q_hover is not None: break
                
                print(f"Trying hover parameters: dist={h_dist}, z_off={z_off}...")
                # Hover Pose: Same Z as grasp
                # Grasp Z target: pose[2] + 0.02 (slightly above bottom/center to be safe)
                target_z = pose[2]
                
                hover_pos = [pose[0] - h_dist, pose[1] + y_shift, target_z]
                
                # Grasp Pose (Target) for validation
                # Add depth offset: gripper needs to go PAST the mug center for fingers to close around it
                # For horizontal approach from -X, we need to add positive X offset
                grasp_depth_offset = 0.03  # 3cm past center
                grasp_pos = [pose[0] + grasp_depth_offset, pose[1] + y_shift, target_z] 

                for grasp_rot in grasp_quats:
                    # Solve IK for Hover Pose
                    path_configs = env.robot.solve_ik_via_sampling(hover_pos, quaternion=grasp_rot, max_configs=50, max_time_ms=1000, ignore_collisions=True)
                    if path_configs is not None and len(path_configs) > 0:
                        # Check collision AND Reachability
                        for q in path_configs:
                            env.set_robot_conf(q)
                            if env.robot.check_collision():
                                continue
                                
                            # Check if we can reach the grasp pose linearly from here
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
            raise RuntimeError("Could not find valid horizontal hover configuration that allows linear approach")
            
    except Exception as e:
        print(f"ERROR: Horizontal hover computation failed: {e}")
        env.set_robot_conf(original_conf)
        # recorder.release()
        pr.stop()
        pr.shutdown()
        return
        
    env.set_robot_conf(original_conf) # Restore for planning

    if np.allclose(home_q, q_hover, atol=1e-4):
        print("WARNING: hover configuration matches home; hovering motion may be trivial.")

    # --- PDDL PLANNING START ---
    print("Setting up PDDL problem for 'move' action...")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    q_hover_tuple = tuple(q_hover)

    # Define PDDL problem
    # We want to move from home to hover.
    # We provide both configs as known 'conf' objects.
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

    if plan:
        print(f"PDDL Plan found with cost: {cost}")
        for action in plan:
            print(f"Action: {action.name}")
            
            if action.name == 'move':
                # args: ?q1 ?q2 ?t
                q1, q2, traj = action.args
                print(f"Executing move via PDDL plan...")
                
                # Interpolate trajectory for smoother video
                full_traj = []
                # traj is a list of waypoints (tuples)
                for i in range(len(traj)-1):
                    start = np.array(traj[i])
                    end = np.array(traj[i+1])
                    steps = 60
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_traj.append((1-t)*start + t*end)
                full_traj.append(traj[-1])

                print(f"DEBUG: Executing trajectory with {len(full_traj)} points.")
                if len(full_traj) > 0:
                    print(f"DEBUG: Start Config: {full_traj[0][:3]}...")
                    print(f"DEBUG: End Config:   {full_traj[-1][:3]}...")

                for conf in full_traj:
                    env.set_robot_conf(conf)
                    pr.step()
                    # recorder.record_step()
        
        # Save hover configuration for later scripts
        env.save_conf("hover_pick", q_hover)
        print("Saved 'hover_pick' configuration after moving to hover pose.")
        
        # Ensure gripper is fully open at hover
        print("Opening gripper at hover...")
        env.gripper.release()
        for _ in range(30):
            pr.step()
        
        # --- PICK LOGIC START ---
        # We are now at q_hover (or will be after execution)
        # Let's plan the approach to grasp
        print("Planning approach to grasp...")
        
        # Target Grasp Pose: The object itself (using the same grasp_pos with offset defined earlier)
        # grasp_pos is already defined with offset
        
        # Solve IK for Grasp Pose using the SAME orientation
        # We try to find a config close to q_hover
        path_configs_grasp = env.robot.solve_ik_via_sampling(grasp_pos, quaternion=successful_grasp_quat, max_configs=20, max_time_ms=500, ignore_collisions=True)
        
        q_grasp = None
        if path_configs_grasp is not None and len(path_configs_grasp) > 0:
            # Sort by distance to q_hover
            path_configs_grasp = sorted(path_configs_grasp, key=lambda q: np.linalg.norm(np.array(q) - np.array(q_hover)))
            q_grasp = path_configs_grasp[0]
            print("Found valid grasp configuration.")
        else:
            print("ERROR: Could not find valid grasp configuration!")
            pr.stop()
            pr.shutdown()
            return

        # Plan Linear Path: Hover -> Grasp
        # We use ignore_collisions=True because we are approaching the object to grasp it
        # Slowed down: steps=200
        path_approach = env.robot.get_linear_path(position=grasp_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True)
        
        if path_approach:
            print("Executing approach trajectory...")
            print(f"DEBUG: Mug pose: {pose[:3]}")
            print(f"DEBUG: Grasp target pos: {grasp_pos}")
            
            # JUGAAD: Keep mug static during approach so it doesn't get pushed
            mug.set_dynamic(False)
            
            traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
            for conf in traj_approach:
                env.set_robot_conf(conf)
                pr.step()
            
            # Close Gripper
            print("Closing gripper...")
            
            # Close fully (0.0) with reasonable velocity for firm grasp
            env.gripper.actuate(0.0, 0.1) 
            for _ in range(50):  # Wait for gripper to close
                pr.step()
            
            # JUGAAD: Teleport mug into gripper and track manually
            print("Force-teleporting mug into gripper...")
            gripper_tip = env.robot.get_tip()
            tip_pos = np.array(gripper_tip.get_position())
            mug_current_pos = np.array(mug.get_position())
            
            print(f"DEBUG: Gripper tip position: {tip_pos}")
            print(f"DEBUG: Mug current position: {mug_current_pos}")
            
            # Calculate offset from tip to mug (mug should stay at this offset from tip)
            mug_offset = mug_current_pos - tip_pos
            print(f"DEBUG: Mug offset from tip: {mug_offset}")
            
            # Keep mug static
            mug.set_dynamic(False)
            
            print("Grasp successful (forced)!")
                
            # --- RETRIEVE LOGIC START ---
            print("Planning retrieve motion (Grasp -> Hover)...")
            
            # Try linear path first, wrapped in try-except to handle PyRep errors
            path_retrieve = None
            try:
                # User suggested adding some Z buffer to the retrieve motion
                # Let's lift slightly higher than the original hover pos to avoid scraping
                retrieve_pos = list(hover_pos)
                retrieve_pos[2] += 0.02 # Add 2cm extra lift for retrieval
                
                path_retrieve = env.robot.get_linear_path(position=retrieve_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True)
            except Exception as e:
                print(f"Linear retrieve failed with error: {e}")
                path_retrieve = None
            
            if path_retrieve:
                print("Executing retrieve trajectory...")
                traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
                for i, conf in enumerate(traj_retrieve):
                    env.set_robot_conf(conf)
                    pr.step()  # Need to step to update gripper_tip position
                    # Teleport mug to follow gripper
                    new_tip_pos = np.array(gripper_tip.get_position())
                    new_mug_pos = new_tip_pos + mug_offset
                    mug.set_position(new_mug_pos.tolist())
                    if i % 50 == 0:
                        print(f"DEBUG step {i}: tip={new_tip_pos[:2]}, mug={new_mug_pos[:2]}")
                
                print("Retrieve complete.")
                env.save_conf("hover_pick_done", traj_retrieve[-1])
            else:
                print("Linear retrieve failed. Executing fallback: Joint Space Interpolation to Hover Config...")
                # Fallback: Joint space interpolation to q_hover
                # This ignores the linear constraint and just gets the arm back to the known safe hover state
                traj_fallback = env._interpolate_joint_path(env.get_robot_conf(), q_hover, steps=100, check_collisions=False)
                if traj_fallback:
                     print("Executing fallback retrieve trajectory...")
                     for i, conf in enumerate(traj_fallback):
                        env.set_robot_conf(conf)
                        pr.step()  # Need to step to update gripper_tip position
                        # Teleport mug to follow gripper
                        new_tip_pos = np.array(gripper_tip.get_position())
                        new_mug_pos = new_tip_pos + mug_offset
                        mug.set_position(new_mug_pos.tolist())
                        if i % 20 == 0:
                            print(f"DEBUG fallback step {i}: tip={new_tip_pos[:2]}, mug={new_mug_pos[:2]}")
                     
                     print("Retrieve complete (Fallback).")
                     env.save_conf("hover_pick_done", q_hover)
                else:
                     print("CRITICAL ERROR: Fallback retrieve also failed.")
            # --- RETRIEVE LOGIC END ---
                
        else:
            print("ERROR: Could not plan linear approach!")
        # --- PICK LOGIC END ---

    else:
        print("ERROR: No PDDL plan found for move action!")
    
    # --- PDDL PLANNING END ---

    # A few extra frames to show final pose + line
    for _ in range(20):
        pr.step()
        # recorder.record_step()

    # recorder.release()

    # Leave sim running so you can inspect; close manually with Ctrl+C
    print("Hover-pick-retrieve complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
