"""
Run Ground Truth Orchestrator with LIVE Segmentation View

Uses multiprocessing to display segmentation masks in real-time
without Qt conflicts with CoppeliaSim.

Also records:
- RGB camera videos to orchestrator_videos/
- Segmentation mask videos (all 5 cams) to segmentation_videos/

Usage:
    python run_live_segmentation.py
"""
import os
import sys
import time
from datetime import datetime

# Configure Qt BEFORE any imports
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

import multiprocessing as mp
try:
    # Only set if unset. Prefer forkserver: avoids heavy __main__ re-import
    # while being safer than raw fork for GUI backends.
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('forkserver')
except Exception:
    # Keep platform default if unavailable.
    pass

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))
os.environ["HEADLESS"] = "False"

from rlbench_kitchen_streams import ENV, get_stream_map
from video_recorder import VideoRecorder
from segmentation_video_recorder import SegmentationVideoRecorder
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import (
    go_home, run_standard_pick_place, run_cupboard_pick_place,
    run_box_pick_place, run_open_box
)

from live_segmentation_viewer import LiveSegmentationViewer
from tkinter_segmentation_viewer import TkinterSegmentationViewer


class LiveSegmentationOrchestrator:
    """Orchestrator with live segmentation display."""
    
    def __init__(self, env):
        self.env = env
        self.pr = env.pr
        self.viewer = None
        self.viewer_backend = "none"
        self.video_recorder = None
        self.mask_recorder = None
        
        # Track discoveries
        self.known_objects = set()
        self.last_visible_count = -1
        self.last_view_update_ts = 0.0
        self.min_view_update_period_s = 0.08

        # Per-run output directories
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.rgb_output_dir = os.path.join("orchestrator_videos_live_seg", run_id)
        self.mask_output_dir = os.path.join("segmentation_videos_live_seg", run_id)
    
    def update_view(self, event=""):
        """Update the live view and check for new objects."""
        if self.viewer:
            # Safety: if viewer process died unexpectedly, don't crash orchestration.
            proc = getattr(self.viewer, "viewer_proc", None)
            if proc is not None and not proc.is_alive():
                print("[Live Viewer] Viewer process is not alive.")
                return set()

            detected = self.viewer.update()
            current_count = len(detected)
            if event or current_count != self.last_visible_count:
                label = event if event else "LIVE"
                print(f"[{label}] Combined visible object count (all cameras): {current_count}")
                self.last_visible_count = current_count
            
            new_objects = detected - self.known_objects
            if new_objects:
                print(f"\n{'='*50}")
                print(f"[{event}] NEW OBJECTS DISCOVERED:")
                for obj in sorted(new_objects):
                    print(f"  ★ {obj}")
                print(f"{'='*50}\n")
            
            self.known_objects.update(detected)
            return detected
        return set()

    def _on_world_step(self):
        """Per-simulation-step hook used by ground_truth_orchestrator."""
        if not self.viewer:
            return
        now = time.time()
        if (now - self.last_view_update_ts) < self.min_view_update_period_s:
            return
        self.last_view_update_ts = now
        self.update_view()
    
    def step(self, count=1):
        """Step simulation and update view."""
        for _ in range(count):
            self.pr.step()
            if self.video_recorder:
                self.video_recorder.record_step()
            if self.mask_recorder:
                self.mask_recorder.record_step()
            self._on_world_step()
    
    def run(self):
        env = self.env
        pr = self.pr
        
        print("=" * 70)
        print("GROUND TRUTH ORCHESTRATOR - LIVE SEGMENTATION VIEW")
        print("=" * 70)
        print("\nThis displays LIVE segmentation masks from RLBench cameras.")
        print("Combined visible-object count is shown in the panel and console.")
        print("Watch for mug4 to appear when the box opens!")
        print("\nPress 'q' in the viewer window to close it.")
        print("=" * 70)
        
        # Initialize RGB + mask recorders
        self.video_recorder = VideoRecorder(env, output_dir=self.rgb_output_dir, fps=30)
        self.mask_recorder = SegmentationVideoRecorder(env, output_dir=self.mask_output_dir, fps=30)
        gt_orch.VIDEO_RECORDER = self.video_recorder
        gt_orch.MASK_RECORDER = self.mask_recorder
        gt_orch.STEP_CALLBACK = self._on_world_step
        
        # Start live viewer.
        # Set LIVE_SEG_VIEWER_BACKEND=tkinter to bypass OpenCV/Qt.
        print("\n[Live Viewer] Starting...")
        backend_pref = os.environ.get("LIVE_SEG_VIEWER_BACKEND", "auto").strip().lower()

        def _start_backend(name):
            try:
                if name == "tkinter":
                    self.viewer = TkinterSegmentationViewer(env, width=1260, height=480)
                    self.viewer_backend = "tkinter"
                else:
                    self.viewer = LiveSegmentationViewer(env, width=1260, height=480)
                    self.viewer_backend = "opencv"
                self.viewer.start()
                proc = getattr(self.viewer, "viewer_proc", None)
                if proc is not None:
                    time.sleep(0.7)
                    return proc.is_alive()
                # In-process viewers (e.g., Tk backend) have no child process.
                return True
            except Exception as e:
                print(f"[Live Viewer] Failed to start {name} backend: {e}")
                self.viewer = None
                self.viewer_backend = "none"
                return False

        live_ok = False
        if backend_pref == "tkinter":
            live_ok = _start_backend("tkinter")
            if not live_ok:
                print("[Live Viewer] Falling back to OpenCV backend.")
                live_ok = _start_backend("opencv")
        elif backend_pref == "opencv":
            live_ok = _start_backend("opencv")
        else:
            live_ok = _start_backend("opencv")
            if not live_ok:
                print("[Live Viewer] OpenCV/Qt backend failed. Falling back to Tkinter viewer.")
                try:
                    self.viewer.stop()
                except Exception:
                    pass
                live_ok = _start_backend("tkinter")

        if not live_ok:
            print("[Live Viewer] Viewer backend failed. Continuing without live window.")
            self.viewer = None
            self.viewer_backend = "none"
        print(f"[Live Viewer] Active backend: {self.viewer_backend}")
        
        # Settle physics
        print("\nSettling physics...")
        self.step(50)
        
        env.set_robot_conf(env.get_home_conf())
        self.step(10)
        
        # Initial detection
        print("\n" + "=" * 50)
        print("INITIAL DETECTION")
        print("=" * 50)
        detected = self.update_view("INITIAL")
        print(f"Detected {len(detected)} objects")
        
        if 'mug4' in detected:
            print("[WARNING] mug4 visible initially - box may be open?")
        else:
            print("[OK] mug4 NOT visible (occluded by closed box)")
        
        results = []
        
        try:
            # === TASK 1: Cupboard mug ===
            print("\n" + "=" * 50)
            print("TASK 1: Cupboard Mug -> Placement")
            print("=" * 50)
            success = run_cupboard_pick_place(env, 'mug3', 'placement_boundary', "Task 1")
            results.append(("Task 1: mug3", success))
            self.update_view("AFTER_TASK_1")
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
                self.update_view(f"AFTER_TASK_2_{i}")
                go_home(env)
                self.step(5)
            
            # === TASK 3: mug2 from box top ===
            print("\n" + "=" * 50)
            print("TASK 3: Box Mug (mug2) -> Placement")
            print("=" * 50)
            success = run_box_pick_place(env, 'mug2', 'placement_boundary', "Task 3")
            results.append(("Task 3: mug2", success))
            self.update_view("AFTER_TASK_3")
            go_home(env)
            self.step(5)
            
            # === TASK 4: OPEN BOX - KEY MOMENT ===
            print("\n" + "=" * 70)
            print("TASK 4: OPENING BOX")
            print("=" * 70)
            print("\n>>> WATCH THE LIVE VIEW!")
            print(">>> mug4 should APPEAR after this!\n")
            
            detected_before = self.viewer.get_detected_objects()
            print(f"BEFORE opening: mug4 visible = {'YES' if 'mug4' in detected_before else 'NO'}")
            
            success = run_open_box(env, "Task 4: Open Box")
            results.append(("Task 4: open box", success))
            
            # Wait for physics and update view multiple times
            self.step(20)
            if self.viewer:
                self.viewer.update()
            
            detected_after = self.viewer.get_detected_objects()
            print(f"\nAFTER opening: mug4 visible = {'YES' if 'mug4' in detected_after else 'NO'}")
            
            new_objects = detected_after - detected_before
            if new_objects:
                print(f"\n★★★ NEWLY VISIBLE: {sorted(new_objects)} ★★★")
            
            go_home(env)
            self.step(5)
            
            # === TASK 5: mug4 from inside box ===
            print("\n" + "=" * 50)
            print("TASK 5: mug4 (inside box) -> Placement")
            print("=" * 50)
            success = run_box_pick_place(env, 'mug4', 'placement_boundary', "Task 5")
            results.append(("Task 5: mug4", success))
            self.update_view("AFTER_TASK_5")
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
        
        # Final state
        print("\n" + "=" * 50)
        print("FINAL DETECTED OBJECTS")
        print("=" * 50)
        final = self.viewer.get_detected_objects()
        print(f"Count: {len(final)}")
        for obj in sorted(final):
            print(f"  - {obj}")
        
        # Save video
        if self.video_recorder:
            self.video_recorder.release()
            self.video_recorder = None
        if self.mask_recorder:
            self.mask_recorder.release()
            self.mask_recorder = None
        print("\nSaved videos:")
        print(f"  - {self.rgb_output_dir}/ (RGB)")
        print(f"  - {self.mask_output_dir}/ (mask_left/right/overhead/wrist/front)")
        
        # Keep running
        print("\nKeeping view open. Press Ctrl+C to exit.")
        try:
            while True:
                pr.step()
                self._on_world_step()
        except KeyboardInterrupt:
            pass

        gt_orch.STEP_CALLBACK = None
        if self.viewer:
            self.viewer.stop()
        pr.stop()
        pr.shutdown()


def main():
    orch = LiveSegmentationOrchestrator(ENV)
    orch.run()


if __name__ == "__main__":
    main()
