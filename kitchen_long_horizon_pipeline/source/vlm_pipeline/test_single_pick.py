#!/usr/bin/env python3
"""
Test Motion Planning for Single Object
=======================================
Loads environment and tests picking a single grocery item.
Can test with object upright or knocked over.

Usage:
  python -m vlm_pipeline.test_single_pick
"""

import os
import sys
import time
import numpy as np

# Configure Qt
os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ["HEADLESS"] = "False"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlbench_kitchen_streams import ENV


def get_object_info(env, obj_name):
    """Get current position and orientation of object."""
    obj = env.get_object(obj_name)
    if obj is None:
        return None
    pos = obj.get_position()
    ori = obj.get_orientation()
    return {"name": obj_name, "position": pos, "orientation": ori}


def knock_over(env, obj_name):
    """Knock object on its side."""
    obj = env.get_object(obj_name)
    if obj is None:
        print(f"Object '{obj_name}' not found")
        return
    
    pos = obj.get_position()
    # Rotate 90 degrees around X axis (tips it over)
    obj.set_orientation([1.57, 0, 0])
    # Lower it slightly since it's now on its side
    obj.set_position([pos[0], pos[1], pos[2] - 0.03])
    
    for _ in range(100):
        env.pr.step()
    
    new_pos = obj.get_position()
    print(f"Knocked over '{obj_name}'")
    print(f"  Old Z: {pos[2]:.3f}, New Z: {new_pos[2]:.3f}")


def stand_up(env, obj_name):
    """Stand object back up."""
    obj = env.get_object(obj_name)
    if obj is None:
        print(f"Object '{obj_name}' not found")
        return
    
    pos = obj.get_position()
    obj.set_orientation([0, 0, 0])
    obj.set_position([pos[0], pos[1], 0.8])  # Reset to table height
    
    for _ in range(100):
        env.pr.step()
    
    print(f"Stood up '{obj_name}'")


def test_pick_place_via_executor(env, obj_name, target_region='placement_boundary'):
    """
    Test picking using the VLMExecutorV2 which handles everything correctly.
    """
    from vlm_pipeline.vlm_executor_v2 import VLMExecutorV2
    from vlm_pipeline.vlm_planner import ActionSkeleton
    
    print(f"\n{'='*60}")
    print(f"Testing pick-place for: {obj_name}")
    print(f"{'='*60}")
    
    # Get object info before
    info = get_object_info(env, obj_name)
    if info:
        print(f"Object position: {info['position']}")
        print(f"Object orientation: {info['orientation']}")
    
    # Create executor
    executor = VLMExecutorV2()
    executor.set_env(env)
    
    # Create pick+place actions
    pick = ActionSkeleton('pick', (obj_name,))
    place = ActionSkeleton('place', (obj_name, target_region))
    
    print(f"\nExecuting: {pick} + {place}")
    
    result = executor.execute_pick_place_pair(pick, place)
    
    print(f"\nResult: {result.status.value}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    # Get object info after
    info_after = get_object_info(env, obj_name)
    if info_after:
        print(f"\nFinal position: {info_after['position']}")
    
    return result


def main():
    print("\n" + "=" * 60)
    print("SINGLE OBJECT PICK TEST")
    print("=" * 60)
    
    env = ENV
    
    # Settle physics
    print("\nSettling physics...")
    for _ in range(50):
        env.pr.step()
    
    # Go to home
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(20):
        env.pr.step()
    
    # Test object
    test_obj = 'soup'
    
    print("\n" + "-" * 60)
    print(f"Test object: {test_obj}")
    print("-" * 60)
    
    info = get_object_info(env, test_obj)
    if info:
        print(f"Initial position: {info['position']}")
        print(f"Initial orientation: {info['orientation']}")
    
    print("\n" + "-" * 60)
    print("Commands:")
    print("  1 - Pick object (upright)")
    print("  2 - Knock over object")
    print("  3 - Pick object (after knock)")
    print("  4 - Reset to home")
    print("  5 - Show object info")
    print("  6 - Stand object back up")
    print("  q - Quit")
    print("-" * 60)
    
    while True:
        try:
            cmd = input("\n>>> ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == '1':
                test_pick_place_via_executor(env, test_obj)
            elif cmd == '2':
                knock_over(env, test_obj)
            elif cmd == '3':
                test_pick_place_via_executor(env, test_obj)
            elif cmd == '4':
                print("Returning to home...")
                env.set_robot_conf(home_q)
                env.robot.gripper.actuate(1, 0.04)  # Open gripper
                for _ in range(50):
                    env.pr.step()
                print("Done.")
            elif cmd == '5':
                info = get_object_info(env, test_obj)
                if info:
                    print(f"Position: {info['position']}")
                    print(f"Orientation: {info['orientation']}")
            elif cmd == '6':
                stand_up(env, test_obj)
            else:
                print("Unknown command")
            
            # Step simulation
            for _ in range(10):
                env.pr.step()
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nExiting...")
    try:
        env.pr.stop()
        env.pr.shutdown()
    except:
        pass


if __name__ == "__main__":
    main()
