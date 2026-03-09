"""
Run Ground Truth Orchestrator with REAL Segmentation Masks

This uses actual RLBench segmentation masks to detect objects.
Objects are detected based on what's VISIBLE in the camera masks,
not from a hardcoded list.

How it works:
1. RLBench has "mask cameras" (e.g., cam_front_mask) that render each pixel
   as RGB-encoded object handles
2. We decode: handle = R + G*256 + B*256*256  
3. We map handles to object names using the scene graph
4. Only objects that appear in the mask pixels are reported as "detected"

This means:
- mug4 inside a closed box will NOT be detected (occluded by lid)
- When box opens, mug4 WILL appear in the mask and be detected

Usage:
    python run_with_segmentation.py
"""
import os
import sys
import subprocess
import time

# Configure Qt
def _configure_qt():
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    for candidate in [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
    ]:
        if os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break

_configure_qt()

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))
os.environ["HEADLESS"] = "False"

from rlbench_kitchen_streams import ENV, get_stream_map
from video_recorder import VideoRecorder
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import (
    go_home, run_standard_pick_place, run_cupboard_pick_place,
    run_box_pick_place, run_open_box
)

from segmentation_mask_viewer import SegmentationMaskViewer


def open_viewer(image_path):
    """Open image viewer with auto-refresh."""
    viewers = [
        ['feh', '-R', '1', image_path],  # feh with 1-second reload
        ['eog', image_path],
    ]
    for cmd in viewers:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Viewer] Opened: {cmd[0]}")
            return proc
        except FileNotFoundError:
            continue
    print(f"[Viewer] Manual view: {image_path}")
    return None


class SegmentationOrchestrator:
    """Orchestrator with real segmentation-based object detection."""
    
    def __init__(self, env):
        self.env = env
        self.pr = env.pr
        self.viewer = SegmentationMaskViewer(env, output_dir="segmentation_output")
        self.video_recorder = None
        self.viewer_proc = None
        
        # Track what we've discovered via segmentation
        self.known_objects = set()
        self.newly_discovered = set()
    
    def update_detection(self, event_name=""):
        """Update segmentation and check for new objects."""
        self.viewer.update()
        
        current_detected = self.viewer.get_detected_objects()
        self.newly_discovered = current_detected - self.known_objects
        self.known_objects.update(current_detected)
        
        if self.newly_discovered:
            print(f"\n{'='*50}")
            print(f"[{event_name}] NEW OBJECTS DISCOVERED VIA SEGMENTATION:")
            for obj in sorted(self.newly_discovered):
                print(f"  ★ {obj}")
            print(f"{'='*50}\n")
        
        return current_detected
    
    def step(self, count=1):
        """Step simulation and update."""
        for _ in range(count):
            self.pr.step()
            if self.video_recorder:
                self.video_recorder.record_step()
    
    def run(self):
        env = self.env
        pr = self.pr
        
        print("=" * 70)
        print("GROUND TRUTH ORCHESTRATOR WITH SEGMENTATION MASKS")
        print("=" * 70)
        print("\nThis uses ACTUAL segmentation masks from RLBench cameras.")
        print("Objects are detected based on what's VISIBLE in the masks,")
        print("not from hardcoded lists.")
        print("\nWatch for mug4 - it should NOT be detected until box opens!")
        print("=" * 70)
        
        # Initialize
        self.video_recorder = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
        gt_orch.VIDEO_RECORDER = self.video_recorder
        
        # Initial capture
        print("\n[Segmentation] Initial capture...")
        self.viewer.update()
        self.viewer.save_snapshot("01_initial")
        
        # Open viewer
        image_path = os.path.abspath("segmentation_output/segmentation_view.png")
        self.viewer_proc = open_viewer(image_path)
        
        # Settle
        print("\nSettling physics...")
        for _ in range(50):
            pr.step()
            self.video_recorder.record_step()
        
        env.set_robot_conf(env.get_home_conf())
        self.step(10)
        
        # Initial detection
        print("\n" + "=" * 50)
        print("INITIAL SEGMENTATION DETECTION")
        print("=" * 50)
        detected = self.update_detection("INITIAL")
        print(f"Detected {len(detected)} objects in camera masks")
        self.viewer.print_detected()
        
        # Check if mug4 is detected initially (it shouldn't be!)
        if 'mug4' in detected:
            print("\n[WARNING] mug4 detected initially - box might be open?")
        else:
            print("\n[OK] mug4 NOT detected - correctly occluded by closed box lid")
        
        self.viewer.save_snapshot("02_after_settle")
        
        results = []
        
        try:
            # === TASK 1: Cupboard mug ===
            print("\n" + "=" * 50)
            print("TASK 1: Cupboard Mug -> Placement")
            print("=" * 50)
            success = run_cupboard_pick_place(env, 'mug3', 'placement_boundary', "Task 1")
            results.append(("Task 1: mug3", success))
            self.update_detection("AFTER_TASK_1")
            go_home(env)
            self.step(5)
            
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
                success = run_standard_pick_place(env, grocery, region, f"Task 2.{i}")
                results.append((f"Task 2.{i}: {grocery}", success))
                self.update_detection(f"AFTER_TASK_2_{i}")
                go_home(env)
                self.step(5)
            
            # === TASK 3: mug2 from box top ===
            print("\n" + "=" * 50)
            print("TASK 3: Box Mug (mug2) -> Placement")
            print("=" * 50)
            success = run_box_pick_place(env, 'mug2', 'placement_boundary', "Task 3")
            results.append(("Task 3: mug2", success))
            self.update_detection("AFTER_TASK_3")
            go_home(env)
            self.step(5)
            
            # === TASK 4: OPEN BOX - THE KEY MOMENT ===
            print("\n" + "=" * 70)
            print("TASK 4: OPENING BOX")
            print("=" * 70)
            print("\n>>> WATCH THE SEGMENTATION OUTPUT!")
            print(">>> mug4 should APPEAR in the detected objects after this!")
            print()
            
            # Capture BEFORE opening
            self.viewer.save_snapshot("03_before_box_open")
            detected_before = self.viewer.get_detected_objects()
            print(f"BEFORE opening: {len(detected_before)} objects detected")
            print(f"mug4 detected: {'YES' if 'mug4' in detected_before else 'NO'}")
            
            # Open the box
            success = run_open_box(env, "Task 4: Open Box")
            results.append(("Task 4: open box", success))
            
            # Wait for physics to settle
            self.step(20)
            
            # Capture AFTER opening
            self.viewer.save_snapshot("04_after_box_open")
            detected_after = self.update_detection("AFTER_BOX_OPEN")
            print(f"\nAFTER opening: {len(detected_after)} objects detected")
            print(f"mug4 detected: {'YES' if 'mug4' in detected_after else 'NO'}")
            
            # Check what's new
            new_objects = detected_after - detected_before
            if new_objects:
                print(f"\n★★★ NEWLY VISIBLE OBJECTS: {sorted(new_objects)} ★★★")
                if 'mug4' in new_objects:
                    print("★★★ mug4 is now visible! Segmentation correctly detected it! ★★★")
            
            go_home(env)
            self.step(5)
            
            # === TASK 5: Pick mug4 (now visible) ===
            print("\n" + "=" * 50)
            print("TASK 5: mug4 (inside box) -> Placement")
            print("=" * 50)
            success = run_box_pick_place(env, 'mug4', 'placement_boundary', "Task 5")
            results.append(("Task 5: mug4", success))
            self.update_detection("AFTER_TASK_5")
            go_home(env)
            self.step(5)
            
        except Exception as e:
            print(f"\n!!! ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # === SUMMARY ===
        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        for task, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {task}")
        print(f"\nTotal: {sum(1 for _, s in results if s)}/{len(results)}")
        
        # Final segmentation state
        print("\n" + "=" * 70)
        print("FINAL SEGMENTATION STATE")
        print("=" * 70)
        self.viewer.print_detected()
        self.viewer.save_snapshot("05_final")
        
        # Save video
        if self.video_recorder:
            self.video_recorder.release()
        
        print("\nImages saved to: segmentation_output/")
        print("Key images:")
        print("  - 01_initial.png")
        print("  - 03_before_box_open.png")
        print("  - 04_after_box_open.png (mug4 should appear here!)")
        print("  - 05_final.png")
        
        # Keep running
        print("\nPress Ctrl+C to exit.")
        try:
            while True:
                pr.step()
                self.viewer.update()
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        
        self.viewer.close()
        if self.viewer_proc:
            self.viewer_proc.terminate()
        pr.stop()
        pr.shutdown()


def main():
    orch = SegmentationOrchestrator(ENV)
    orch.run()


if __name__ == "__main__":
    main()
