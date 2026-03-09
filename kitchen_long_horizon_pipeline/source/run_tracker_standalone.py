"""
Standalone Camera Viewer with Object Tracking

This script shows live camera feeds from the simulation with object
visibility overlay. All GUI updates happen on the main thread to avoid Qt issues.

Usage:
    python run_tracker_standalone.py
"""
import os
import sys
import time

# Configure Qt for GUI FIRST before any other imports
def _configure_qt():
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

# Set HEADLESS env var to False
os.environ["HEADLESS"] = "False"

import numpy as np

# Import simulation environment FIRST (initializes Qt properly)
from rlbench_kitchen_env_constrained import RLBenchKitchenEnvConstrained as RLBenchKitchenEnv

# NOW import cv2 after Qt is initialized by CoppeliaSim
import cv2

from live_object_tracker import ObjectVisibilityTracker


def create_info_panel(visible_objects, box_open, width=250, height=400):
    """Create an info panel showing visible objects."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(panel, "Visible Objects", (10, 25), font, 0.6, (255, 255, 255), 1)
    cv2.line(panel, (5, 35), (width - 5, 35), (100, 100, 100), 1)
    
    # Box status
    box_color = (100, 255, 100) if box_open else (100, 100, 255)
    box_text = "OPEN" if box_open else "CLOSED"
    cv2.putText(panel, f"Box: {box_text}", (10, 55), font, 0.5, box_color, 1)
    
    # mug4 status
    mug4_visible = 'mug4' in visible_objects or 'mug_inside_box' in visible_objects
    mug4_color = (100, 255, 100) if mug4_visible else (100, 100, 255)
    mug4_text = "VISIBLE" if mug4_visible else "HIDDEN"
    cv2.putText(panel, f"mug4: {mug4_text}", (10, 75), font, 0.5, mug4_color, 1)
    
    cv2.line(panel, (5, 85), (width - 5, 85), (100, 100, 100), 1)
    
    # List objects
    y = 105
    for obj in sorted(visible_objects):
        if y > height - 20:
            cv2.putText(panel, "...", (10, y), font, 0.4, (150, 150, 150), 1)
            break
        
        # Color by category
        if 'mug' in obj.lower():
            color = (255, 100, 100)
        elif any(g in obj.lower() for g in ['soup', 'mustard', 'spam', 'sugar', 'crackers']):
            color = (100, 255, 100)
        elif 'box' in obj.lower() or 'lid' in obj.lower():
            color = (100, 100, 255)
        else:
            color = (200, 200, 200)
        
        cv2.circle(panel, (15, y - 4), 4, color, -1)
        cv2.putText(panel, obj[:20], (25, y), font, 0.4, (255, 255, 255), 1)
        y += 18
    
    return panel


def main():
    print("="*70)
    print("STANDALONE CAMERA VIEWER WITH OBJECT TRACKING")
    print("="*70)
    print("\nControls:")
    print("  q - Quit")
    print("  s - Print summary to console")
    print("="*70 + "\n")
    
    # Initialize environment
    print("Loading simulation environment...")
    env = RLBenchKitchenEnv(headless=False)
    pr = env.pr
    
    # Settle physics
    print("Settling physics...")
    for _ in range(50):
        pr.step()
    
    # Move robot to home
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        pr.step()
    
    # Initialize tracker
    print("Initializing object tracker...")
    tracker = ObjectVisibilityTracker(env)
    
    # Print initial state
    print("\nInitial objects in scene:")
    for name in sorted(env.name_to_obj.keys()):
        obj = env.get_object(name)
        if obj:
            pos = obj.get_position()
            print(f"  {name}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print("\n" + "="*70)
    print("Viewer running. Press 'q' to quit.")
    print("="*70 + "\n")
    
    # Create windows
    cv2.namedWindow("Camera: Front", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Camera: Overhead", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Object Tracker", cv2.WINDOW_AUTOSIZE)
    
    cv2.moveWindow("Camera: Front", 0, 0)
    cv2.moveWindow("Camera: Overhead", 660, 0)
    cv2.moveWindow("Object Tracker", 1320, 0)
    
    frame_count = 0
    
    try:
        while True:
            # Step simulation
            pr.step()
            frame_count += 1
            
            # Update tracker
            tracker.update()
            visible = tracker.known_objects
            box_open = tracker.check_box_opened()
            
            # Get camera frames
            frames = env.get_camera_frames()
            
            # Display front camera
            if 'front' in frames:
                front = cv2.cvtColor(frames['front'], cv2.COLOR_RGB2BGR)
                front = cv2.resize(front, (640, 480))
                cv2.imshow("Camera: Front", front)
            
            # Display overhead camera
            if 'overhead' in frames:
                overhead = cv2.cvtColor(frames['overhead'], cv2.COLOR_RGB2BGR)
                overhead = cv2.resize(overhead, (640, 480))
                cv2.imshow("Camera: Overhead", overhead)
            
            # Display info panel
            panel = create_info_panel(visible, box_open)
            cv2.imshow("Object Tracker", panel)
            
            # Print periodic summary
            if frame_count % 150 == 0:
                mug4_vis = 'mug4' in visible or 'mug_inside_box' in visible
                print(f"[Frame {frame_count}] Visible: {len(visible)} | Box: {'OPEN' if box_open else 'CLOSED'} | mug4: {'YES' if mug4_vis else 'NO'}")
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                print(f"\n[SUMMARY] Visible objects: {sorted(visible)}")
                print(f"  Box open: {box_open}")
                print(f"  mug4 visible: {'mug4' in visible or 'mug_inside_box' in visible}\n")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    # Cleanup
    print("Cleaning up...")
    cv2.destroyAllWindows()
    pr.stop()
    pr.shutdown()
    
    print("Done.")


if __name__ == "__main__":
    main()
