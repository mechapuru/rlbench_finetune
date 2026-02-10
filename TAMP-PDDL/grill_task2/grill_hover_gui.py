#!/usr/bin/env python3
"""
grill_hover_gui.py
GUI script to test hovering over meat objects (steak/chicken) in the grill task scene.
Usage: python grill_hover_gui.py [object_name]
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


def main():
    parser = argparse.ArgumentParser(description='Hover over a meat object in the grill task scene')
    parser.add_argument('object', nargs='?', default='steak', 
                        help='Object to hover over: steak, chicken, meat1, meat2 (default: steak)')
    args = parser.parse_args()
    
    object_name = args.object
    print(f"=== Grill Task: Hover over '{object_name}' ===")
    
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

    # Use the environment's hover planner to get a valid hover pose
    try:
        q_hover, hover_orientation = env.compute_hover_config(target_obj, list(pose), hover_offset=0.15)
        print("Found valid hover configuration.")
    except Exception as e:
        print(f"ERROR: compute_hover_config failed: {e}")
        pr.stop()
        pr.shutdown()
        return

    if np.allclose(home_q, q_hover, atol=1e-4):
        print("WARNING: hover configuration matches home; hovering motion may be trivial.")

    # --- PDDL PLANNING START ---
    print("Setting up PDDL problem for 'move' action...")

    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl', 'grill_task_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl', 'grill_task_streams.pddl'))

    q_home_tuple = tuple(home_q)
    q_hover_tuple = tuple(q_hover)

    # Define PDDL problem
    # We want to move from home to hover position
    init = [
        ('conf', q_home_tuple),
        ('conf', q_hover_tuple),
        ('at-conf', q_home_tuple),
        ('is-home', q_home_tuple),
        ('hand-empty',),
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

                # Interpolate trajectory for smoother motion
                full_traj = []
                for i in range(len(traj) - 1):
                    start = np.array(traj[i])
                    end = np.array(traj[i + 1])
                    steps = 30
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_traj.append((1 - t) * start + t * end)
                full_traj.append(traj[-1])

                print(f"Executing trajectory with {len(full_traj)} points...")
                for conf in full_traj:
                    env.set_robot_conf(conf)
                    pr.step()

        print("Hover complete!")
    else:
        print("ERROR: No PDDL plan found!")
        print("Trying direct motion planning...")
        
        # Fallback: use direct motion planning
        traj = env.compute_motion_plan(home_q, q_hover)
        if traj:
            print(f"Direct motion plan found with {len(traj)} waypoints")
            # Interpolate for smooth motion
            full_traj = []
            for i in range(len(traj) - 1):
                start = np.array(traj[i])
                end = np.array(traj[i + 1])
                steps = 15
                for t in np.linspace(0, 1, steps, endpoint=False):
                    full_traj.append((1 - t) * start + t * end)
            full_traj.append(traj[-1])
            
            for conf in full_traj:
                env.set_robot_conf(conf)
                pr.step()
            print("Hover complete via direct planning!")
        else:
            print("Direct motion planning also failed!")

    # --- PDDL PLANNING END ---

    # A few extra frames to show final pose
    for _ in range(20):
        pr.step()

    # Leave sim running so you can inspect; close manually with Ctrl+C
    print(f"Hovering over '{object_name}' complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
