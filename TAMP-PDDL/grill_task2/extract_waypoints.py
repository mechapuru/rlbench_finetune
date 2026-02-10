#!/usr/bin/env python3
"""
Script to extract all waypoint positions from the grill_task scene.
This will help identify the correct positions for grill lid operations.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RLBench'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyrep import PyRep
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
import numpy as np

def main():
    pr = PyRep()
    
    # Load the scene
    scene_path = os.path.join(os.path.dirname(__file__), '..', 'grill_task.ttt')
    pr.launch(scene_path, headless=True)
    pr.start()
    
    print("=" * 60)
    print("Extracting waypoints from grill_task.ttt")
    print("=" * 60)
    
    # Try to find all waypoints (typically named waypoint0, waypoint1, etc.)
    found_waypoints = []
    for i in range(50):  # Check waypoint0 through waypoint49
        try:
            wp = Dummy(f'waypoint{i}')
            pos = wp.get_position()
            ori = wp.get_orientation()
            print(f"\nwaypoint{i}:")
            print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"  Orientation (rad): [{ori[0]:.4f}, {ori[1]:.4f}, {ori[2]:.4f}]")
            print(f"  Orientation (deg): [{np.degrees(ori[0]):.1f}, {np.degrees(ori[1]):.1f}, {np.degrees(ori[2]):.1f}]")
            found_waypoints.append(i)
        except:
            pass
    
    print("\n" + "=" * 60)
    print(f"Found {len(found_waypoints)} waypoints: {found_waypoints}")
    print("=" * 60)
    
    # Also print key object positions for reference
    print("\nKey object positions for reference:")
    print("-" * 40)
    
    objects_to_check = [
        'handle_visual',
        'lid_visual', 
        'grill_visual',
        'lid_joint',
        'steak_visual',
        'chicken_visual'
    ]
    
    for obj_name in objects_to_check:
        try:
            obj = Shape(obj_name)
            pos = obj.get_position()
            ori = obj.get_orientation()
            print(f"\n{obj_name}:")
            print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"  Orientation (deg): [{np.degrees(ori[0]):.1f}, {np.degrees(ori[1]):.1f}, {np.degrees(ori[2]):.1f}]")
        except Exception as e:
            print(f"\n{obj_name}: Not found or error ({e})")
    
    # Try to find lid_joint as Joint object
    try:
        from pyrep.objects.joint import Joint
        lid_joint = Joint('lid_joint')
        pos = lid_joint.get_position()
        print(f"\nlid_joint (as Joint):")
        print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"  Joint position: {lid_joint.get_joint_position():.4f} rad ({np.degrees(lid_joint.get_joint_position()):.1f} deg)")
    except Exception as e:
        print(f"\nlid_joint (as Joint): Error ({e})")
    
    pr.stop()
    pr.shutdown()
    print("\nDone!")

if __name__ == '__main__':
    main()
