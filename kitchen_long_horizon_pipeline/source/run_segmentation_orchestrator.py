"""
Ground Truth Orchestrator with Segmentation-Based Object Detection

Object list comes from segmentation masks, NOT hardcoded RLBench list.
Only visible objects are known to the planner.

Key difference from regular orchestrator:
- Objects are detected via vision (segmentation masks)
- mug4 inside closed box is NOT visible → NOT in PDDL init
- When box opens and mug4 appears in masks → NEW_OBJECT_DETECTED → replan

Usage:
    python run_segmentation_orchestrator.py
"""
import os
import sys

# Qt setup
os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
os.environ.pop("QT_PLUGIN_PATH", None)
coppelia = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
for p in [os.path.join(coppelia, "platforms"), os.path.join(coppelia, "Qt/plugins/platforms")]:
    if os.path.isdir(p):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", p)
        break

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))
os.environ["HEADLESS"] = "False"

from rlbench_kitchen_streams import ENV
from video_recorder import VideoRecorder
from segmentation_video_recorder import SegmentationVideoRecorder
from segmentation_object_detector import SegmentationObjectDetector
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import (
    go_home, run_standard_pick_place, run_cupboard_pick_place,
    run_box_pick_place, run_open_box
)


class SegmentationOrchestrator:
    """
    Orchestrator that uses segmentation for object detection.
    """
    
    def __init__(self, env):
        self.env = env
        self.pr = env.pr
        
        # Object detector from segmentation
        self.detector = SegmentationObjectDetector(env)
        
        # Video recorders
        self.rgb_recorder = None
        self.mask_recorder = None
        
        # Task queue (what we need to do)
        self.pending_tasks = []
        self.completed_tasks = []
        
    def step(self, n=1):
        for _ in range(n):
            self.pr.step()
            if self.rgb_recorder:
                self.rgb_recorder.record_step()
            if self.mask_recorder:
                self.mask_recorder.record_step()
    
    def update_vision(self):
        """Update segmentation and check for new objects."""
        self.detector.update()
        
        if self.detector.check_for_new_objects():
            new_objs = self.detector.get_newly_detected()
            print(f"\n{'='*60}")
            print(f"★ NEW OBJECTS DETECTED VIA SEGMENTATION ★")
            for obj in sorted(new_objs):
                print(f"  → {obj}")
            print(f"{'='*60}\n")
            return True, new_objs
        
        return False, set()
    
    def get_visible_objects(self):
        """Get current visible object list from segmentation."""
        return self.detector.get_visible_objects()
    
    def run(self):
        print("=" * 70)
        print("SEGMENTATION-BASED ORCHESTRATOR")
        print("=" * 70)
        print("\nObject list comes from SEGMENTATION MASKS, not hardcoded.")
        print("mug4 inside closed box = NOT visible = NOT in plan")
        print("When box opens → mug4 appears in masks → REPLAN")
        print("=" * 70)
        
        # Initialize recorders
        self.rgb_recorder = VideoRecorder(self.env, output_dir="orchestrator_videos", fps=30)
        self.mask_recorder = SegmentationVideoRecorder(self.env, output_dir="segmentation_videos", fps=30)
        gt_orch.VIDEO_RECORDER = self.rgb_recorder
        
        # Settle physics
        print("\nSettling physics...")
        for _ in range(50):
            self.step()
        
        self.env.set_robot_conf(self.env.get_home_conf())
        self.step(10)
        
        # Initial detection
        print("\n" + "=" * 60)
        print("INITIAL OBJECT DETECTION (from segmentation)")
        print("=" * 60)
        self.detector.update()
        visible = self.get_visible_objects()
        
        print(f"\nVisible objects ({len(visible)}):")
        for obj in sorted(visible):
            print(f"  ✓ {obj}")
        
        # Check if mug4 is visible (it shouldn't be!)
        if 'mug4' in visible:
            print("\n[WARNING] mug4 is visible! Box might be open?")
        else:
            print("\n[OK] mug4 NOT visible (hidden inside closed box)")
            print("     Planner does NOT know mug4 exists yet!")
        
        results = []
        
        try:
            # === PHASE 1: Tasks with initially visible objects ===
            print("\n" + "=" * 60)
            print("PHASE 1: Tasks with VISIBLE objects only")
            print("=" * 60)
            
            # Task 1: mug3 from cupboard (visible)
            if 'mug3' in visible:
                print("\n--- Task 1: mug3 (visible in cupboard) ---")
                ok = run_cupboard_pick_place(self.env, 'mug3', 'placement_boundary', "T1-mug3")
                results.append(("mug3", ok))
                go_home(self.env)
                self.step(5)
                self.update_vision()
            
            # Task 2: Groceries (all visible)
            groceries = ['soup', 'mustard', 'spam', 'sugar', 'crackers']
            for item in groceries:
                if item in visible:
                    dest = 'cupboard_boundary_top' if item in ['sugar', 'crackers'] else 'cupboard_boundary'
                    print(f"\n--- Task 2: {item} (visible) ---")
                    ok = run_standard_pick_place(self.env, item, dest, f"T2-{item}")
                    results.append((item, ok))
                    go_home(self.env)
                    self.step(5)
                    self.update_vision()
            
            # Task 3: mug2 from box top (visible)
            if 'mug2' in visible:
                print("\n--- Task 3: mug2 (visible on box) ---")
                ok = run_box_pick_place(self.env, 'mug2', 'placement_boundary', "T3-mug2")
                results.append(("mug2", ok))
                go_home(self.env)
                self.step(5)
                self.update_vision()
            
            # === PHASE 2: Open box (reveals mug4) ===
            print("\n" + "=" * 60)
            print("PHASE 2: Opening box")
            print("=" * 60)
            
            visible_before = self.get_visible_objects()
            print(f"\nBefore opening: {len(visible_before)} objects visible")
            print(f"mug4 visible: {'YES' if 'mug4' in visible_before else 'NO'}")
            
            print("\n--- Task 4: Open box ---")
            ok = run_open_box(self.env, "T4-open-box")
            results.append(("open_box", ok))
            
            # Let physics settle
            for _ in range(30):
                self.step()
            
            # Check for new objects!
            new_found, new_objs = self.update_vision()
            visible_after = self.get_visible_objects()
            
            print(f"\nAfter opening: {len(visible_after)} objects visible")
            print(f"mug4 visible: {'YES' if 'mug4' in visible_after else 'NO'}")
            
            if 'mug4' in new_objs:
                print("\n" + "★" * 60)
                print("★ mug4 NOW VISIBLE! It was hidden inside closed box!")
                print("★ In a real system, this would trigger REPLANNING")
                print("★" * 60)
            
            go_home(self.env)
            self.step(5)
            
            # === PHASE 3: Now mug4 is visible, can plan for it ===
            print("\n" + "=" * 60)
            print("PHASE 3: mug4 now visible, can execute")
            print("=" * 60)
            
            if 'mug4' in visible_after:
                print("\n--- Task 5: mug4 (NOW visible after box open) ---")
                ok = run_box_pick_place(self.env, 'mug4', 'placement_boundary', "T5-mug4")
                results.append(("mug4", ok))
                go_home(self.env)
                self.step(5)
            else:
                print("\n[ERROR] mug4 still not visible! Cannot plan for it.")
                results.append(("mug4", False))
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        for name, ok in results:
            print(f"  [{'OK' if ok else 'FAIL'}] {name}")
        print(f"\nTotal: {sum(1 for _, ok in results if ok)}/{len(results)}")
        
        # Final vision state
        print("\n" + "=" * 60)
        print("FINAL VISIBLE OBJECTS")
        print("=" * 60)
        self.detector.update()
        final_visible = self.get_visible_objects()
        print(f"Count: {len(final_visible)}")
        for obj in sorted(final_visible):
            print(f"  - {obj}")
        
        # Release
        self.rgb_recorder.release()
        self.mask_recorder.release()
        
        print("\nVideos saved to:")
        print("  RGB:   orchestrator_videos/")
        print("  Masks: segmentation_videos/")
        
        self.pr.stop()
        self.pr.shutdown()


def main():
    orch = SegmentationOrchestrator(ENV)
    orch.run()


if __name__ == "__main__":
    main()
