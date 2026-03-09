#!/usr/bin/env python3
"""
Live segmentation run for:
1) pick mug2 from top of box and place it
2) slide open the box lid
3) check count increase before vs after opening
"""

import os
import sys
import time
from datetime import datetime


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
sys.path.append(os.path.join(os.path.dirname(__file__), "pddlstream"))
os.environ["HEADLESS"] = "False"

from rlbench_kitchen_streams import ENV
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import go_home, run_box_pick_place, run_open_box
from video_recorder import VideoRecorder
from segmentation_video_recorder import SegmentationVideoRecorder
from live_segmentation_viewer import LiveSegmentationViewer
from tkinter_segmentation_viewer import TkinterSegmentationViewer


class Mug2ThenOpenLidLiveCheck:
    def __init__(self, env):
        self.env = env
        self.pr = env.pr
        self.viewer = None
        self.viewer_backend = "none"
        self.video_recorder = None
        self.mask_recorder = None
        self.last_visible_count = -1
        self.last_view_update_ts = 0.0
        self.min_view_update_period_s = 0.08

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.rgb_output_dir = os.path.join("orchestrator_videos_live_seg", run_id)
        self.mask_output_dir = os.path.join("segmentation_videos_live_seg", run_id)

    def update_view(self, event=""):
        if self.viewer is None:
            return set()
        proc = getattr(self.viewer, "viewer_proc", None)
        if proc is not None and not proc.is_alive():
            print("[Live Viewer] Viewer process is not alive.")
            return set()
        detected = self.viewer.update()
        count = len(detected)
        if event or count != self.last_visible_count:
            label = event if event else "LIVE"
            print(f"[{label}] Combined visible object count: {count}")
            self.last_visible_count = count
        return detected

    def _on_world_step(self):
        if self.viewer is None:
            return
        now = time.time()
        if (now - self.last_view_update_ts) < self.min_view_update_period_s:
            return
        self.last_view_update_ts = now
        self.update_view()

    def step(self, count=1):
        for _ in range(count):
            self.pr.step()
            if self.video_recorder:
                self.video_recorder.record_step()
            if self.mask_recorder:
                self.mask_recorder.record_step()
            self._on_world_step()

    def _start_backend(self, name):
        try:
            if name == "tkinter":
                self.viewer = TkinterSegmentationViewer(self.env, width=1260, height=480)
                self.viewer_backend = "tkinter"
            else:
                self.viewer = LiveSegmentationViewer(self.env, width=1260, height=480)
                self.viewer_backend = "opencv"
            self.viewer.start()
            proc = getattr(self.viewer, "viewer_proc", None)
            if proc is not None:
                time.sleep(0.7)
                return proc.is_alive()
            return True
        except Exception as exc:
            print(f"[Live Viewer] Failed to start {name} backend: {exc}")
            self.viewer = None
            self.viewer_backend = "none"
            return False

    def _start_viewer(self):
        backend_pref = os.environ.get("LIVE_SEG_VIEWER_BACKEND", "auto").strip().lower()
        if backend_pref == "tkinter":
            ok = self._start_backend("tkinter")
            if not ok:
                print("[Live Viewer] Tkinter failed. Trying OpenCV.")
                ok = self._start_backend("opencv")
            return ok
        if backend_pref == "opencv":
            ok = self._start_backend("opencv")
            if not ok:
                print("[Live Viewer] OpenCV failed. Falling back to Tkinter.")
                ok = self._start_backend("tkinter")
            return ok

        ok = self._start_backend("opencv")
        if not ok:
            print("[Live Viewer] OpenCV failed. Falling back to Tkinter.")
            ok = self._start_backend("tkinter")
        return ok

    def run(self):
        print("=" * 70)
        print("LIVE CHECK: PICK MUG2 -> OPEN BOX LID")
        print("=" * 70)
        print("Flow: pick mug2 from box top, place it, then slide open lid.")
        print("Window: live segmentation panel like run_live_segmentation.py")
        print("=" * 70)

        self.video_recorder = VideoRecorder(self.env, output_dir=self.rgb_output_dir, fps=30)
        self.mask_recorder = SegmentationVideoRecorder(self.env, output_dir=self.mask_output_dir, fps=30)
        gt_orch.VIDEO_RECORDER = self.video_recorder
        gt_orch.MASK_RECORDER = self.mask_recorder
        gt_orch.STEP_CALLBACK = self._on_world_step

        print("\n[Live Viewer] Starting...")
        live_ok = self._start_viewer()
        if not live_ok:
            print("[Live Viewer] Failed to start any backend. Continuing without live window.")
            self.viewer = None
            self.viewer_backend = "none"
        print(f"[Live Viewer] Active backend: {self.viewer_backend}")

        print("\nSettling physics...")
        self.step(50)
        self.env.set_robot_conf(self.env.get_home_conf())
        self.step(10)

        self.update_view("INITIAL")

        results = []
        try:
            print("\n" + "=" * 50)
            print("TASK 1: Pick mug2 (box top) -> placement_boundary")
            print("=" * 50)
            ok_mug2 = run_box_pick_place(self.env, "mug2", "placement_boundary", "Task 1: mug2 move")
            results.append(("Task 1: mug2 move", ok_mug2))
            self.update_view("AFTER_MUG2")
            go_home(self.env)
            self.step(5)

            before_open = self.viewer.get_detected_objects() if self.viewer else set()
            print(f"\nBEFORE opening: count={len(before_open)}, mug4={'YES' if 'mug4' in before_open else 'NO'}")

            print("\n" + "=" * 50)
            print("TASK 2: Slide open box lid")
            print("=" * 50)
            ok_open = run_open_box(self.env, "Task 2: open box")
            results.append(("Task 2: open box", ok_open))

            self.step(20)
            self.update_view("AFTER_OPEN")
            after_open = self.viewer.get_detected_objects() if self.viewer else set()
            print(f"AFTER opening : count={len(after_open)}, mug4={'YES' if 'mug4' in after_open else 'NO'}")

            delta = len(after_open) - len(before_open)
            print("\n" + "=" * 70)
            print("COUNT CHECK (around open-lid step)")
            print("=" * 70)
            print(f"Before count: {len(before_open)}")
            print(f"After count : {len(after_open)}")
            print(f"Delta       : {delta:+d}")
            print(f"Increased?  : {'YES' if delta > 0 else 'NO'}")
            print(f"New objects : {sorted(after_open - before_open)}")
            print("=" * 70)

        except Exception as exc:
            print(f"\nERROR: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            for task, ok in results:
                print(f"[{'PASS' if ok else 'FAIL'}] {task}")
            print(f"Videos:")
            print(f"  RGB : {self.rgb_output_dir}")
            print(f"  MASK: {self.mask_output_dir}")

            print("\nKeeping view open. Press Ctrl+C to exit.")
            try:
                while True:
                    self.pr.step()
                    self._on_world_step()
            except KeyboardInterrupt:
                pass

            gt_orch.STEP_CALLBACK = None
            gt_orch.VIDEO_RECORDER = None
            gt_orch.MASK_RECORDER = None
            if self.video_recorder:
                self.video_recorder.release()
            if self.mask_recorder:
                self.mask_recorder.release()
            if self.viewer:
                self.viewer.stop()
            self.pr.stop()
            self.pr.shutdown()


def main():
    runner = Mug2ThenOpenLidLiveCheck(ENV)
    runner.run()


if __name__ == "__main__":
    main()
