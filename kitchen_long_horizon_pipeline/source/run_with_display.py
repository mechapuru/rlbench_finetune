"""
Run Ground Truth Orchestrator with Segmentation Display

Saves camera views and detected objects to images that update live.
Opens an image viewer to watch the output.

Usage:
    python run_with_display.py
"""
import os
import sys
import subprocess
import time

# Configure Qt for GUI - MUST be before any Qt imports
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

import numpy as np

# Add pddlstream to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

# Set HEADLESS env var to False
os.environ["HEADLESS"] = "False"

# Import environment and streams
from rlbench_kitchen_streams import ENV, get_stream_map
from video_recorder import VideoRecorder

# Import orchestrator functions
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import (
    step_and_record, go_home, 
    run_standard_pick_place, run_cupboard_pick_place, 
    run_box_pick_place, run_open_box
)

# Import display (PIL-based, no Qt)
from segmentation_display import SegmentationDisplay


def open_image_viewer(image_path):
    """Open an auto-refreshing image viewer."""
    # Try different viewers
    viewers = [
        ['feh', '--reload', '1', image_path],  # feh with auto-reload
        ['eog', image_path],  # Eye of GNOME
        ['gpicview', image_path],  # GPicView
        ['xdg-open', image_path],  # Default viewer
    ]
    
    for viewer in viewers:
        try:
            # Start viewer in background
            proc = subprocess.Popen(
                viewer, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            print(f"[Viewer] Opened with: {viewer[0]}")
            return proc
        except FileNotFoundError:
            continue
    
    print("[Viewer] No image viewer found. View images manually:")
    print(f"         {image_path}")
    return None


class DisplayOrchestrator:
    """Orchestrator with live image display."""
    
    def __init__(self, env):
        self.env = env
        self.pr = env.pr
        self.display = SegmentationDisplay(env, output_dir="tracker_output", save_interval=5)
        self.video_recorder = None
        self.viewer_proc = None
        
    def step_with_display(self, count=1):
        """Step simulation and update display."""
        for _ in range(count):
            self.pr.step()
            if self.video_recorder:
                self.video_recorder.record_step()
        self.display.update()
    
    def run(self):
        """Run orchestration with live display."""
        env = self.env
        pr = self.pr
        
        print("=" * 60)
        print("GROUND TRUTH ORCHESTRATOR WITH SEGMENTATION DISPLAY")
        print("=" * 60)
        print("\nImages will be saved to: tracker_output/")
        print("An image viewer will open showing live camera feeds.")
        print("=" * 60)
        
        # Initialize video recorder
        self.video_recorder = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
        gt_orch.VIDEO_RECORDER = self.video_recorder
        
        # Create initial image
        self.display.update()
        self.display.save_snapshot("initial")
        
        # Open image viewer
        image_path = os.path.abspath("tracker_output/current_view.png")
        print(f"\nOpening viewer for: {image_path}")
        self.viewer_proc = open_image_viewer(image_path)
        
        # Settle physics
        print("\nSettling physics...")
        for i in range(50):
            pr.step()
            self.video_recorder.record_step()
            if i % 10 == 0:
                self.display.update()
        
        # Home position
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        self.step_with_display(10)
        
        results = []
        
        try:
            # === TASK 1 ===
            print("\n" + "=" * 50)
            print("TASK 1: Cupboard Mug -> Placement")
            print("=" * 50)
            self.display.print_status()
            
            success = run_cupboard_pick_place(
                env, object_name='mug3',
                target_region='placement_boundary',
                task_name="Task 1"
            )
            results.append(("Task 1: mug3", success))
            self.display.save_snapshot("after_task1")
            go_home(env)
            self.step_with_display(10)
            
            # === TASK 2: Groceries ===
            groceries = [
                ('soup', 'cupboard_boundary'),
                ('mustard', 'cupboard_boundary'),
                ('spam', 'cupboard_boundary'),
                ('sugar', 'cupboard_boundary_top'),
                ('crackers', 'cupboard_boundary_top'),
            ]
            
            for i, (grocery, region) in enumerate(groceries, 1):
                print(f"\n--- Task 2.{i}: {grocery} ---")
                success = run_standard_pick_place(
                    env, object_name=grocery,
                    target_region=region,
                    task_name=f"Task 2.{i}"
                )
                results.append((f"Task 2.{i}: {grocery}", success))
                go_home(env)
                self.step_with_display(10)
            
            # === TASK 3: mug2 from box ===
            print("\n" + "=" * 50)
            print("TASK 3: Box Mug -> Placement")
            print("=" * 50)
            
            success = run_box_pick_place(
                env, object_name='mug2',
                target_region='placement_boundary',
                task_name="Task 3"
            )
            results.append(("Task 3: mug2", success))
            self.display.save_snapshot("after_task3")
            go_home(env)
            self.step_with_display(10)
            
            # === TASK 4: Open Box ===
            print("\n" + "=" * 50)
            print("TASK 4: OPENING BOX")
            print(">>> mug4 will become visible after this!")
            print("=" * 50)
            
            self.display.save_snapshot("before_box_open")
            self.display.print_status()
            
            success = run_open_box(env, task_name="Task 4: Open Box")
            results.append(("Task 4: open box", success))
            
            # This is the key moment!
            self.display.save_snapshot("after_box_open")
            self.display.print_status()
            
            print("\n*** mug4 should now be visible! Check the image viewer! ***")
            
            go_home(env)
            self.step_with_display(10)
            
            # === TASK 5: mug4 from inside box ===
            print("\n" + "=" * 50)
            print("TASK 5: Mug Inside Box -> Placement")
            print("=" * 50)
            
            success = run_box_pick_place(
                env, object_name='mug4',
                target_region='placement_boundary',
                task_name="Task 5"
            )
            results.append(("Task 5: mug4", success))
            self.display.save_snapshot("after_task5")
            go_home(env)
            self.step_with_display(10)
            
        except Exception as e:
            print(f"\n!!! ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # === SUMMARY ===
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        total = len(results)
        passed = sum(1 for _, s in results if s)
        for task, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {task}")
        print(f"\nTotal: {passed}/{total} tasks completed")
        print("=" * 60)
        
        # Save final state
        self.display.print_status()
        self.display.close()
        
        # Save videos
        if self.video_recorder:
            self.video_recorder.release()
            print("\nVideos saved to orchestrator_videos/")
        
        print("\nImages saved to tracker_output/")
        print("Key snapshots:")
        print("  - initial.png")
        print("  - before_box_open.png")
        print("  - after_box_open.png (mug4 should appear here!)")
        print("  - final_view.png")
        
        # Keep simulation running
        print("\nPress Ctrl+C to close simulation.")
        try:
            while True:
                pr.step()
                self.display.update()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
        # Cleanup
        if self.viewer_proc:
            self.viewer_proc.terminate()
        pr.stop()
        pr.shutdown()


def main():
    orchestrator = DisplayOrchestrator(ENV)
    orchestrator.run()


if __name__ == "__main__":
    main()
