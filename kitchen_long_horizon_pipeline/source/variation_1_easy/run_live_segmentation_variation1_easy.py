"""
Run variation_1_easy ground-truth with LIVE segmentation masks + object count.

Usage:
    python variation_1_easy/run_live_segmentation_variation1_easy.py

Optional:
    LIVE_SEG_VIEWER_BACKEND=tkinter python variation_1_easy/run_live_segmentation_variation1_easy.py
"""
import os
import re
import sys
import time
from datetime import datetime


def _env_int(name, default, min_value=1):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(min_value, value)



def _configure_qt():
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    for candidate in [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
        os.path.join(coppelia_root, "qt", "plugins", "platforms"),
    ]:
        if os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break


_configure_qt()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
SCENE_PATH = os.path.join(ROOT_DIR, "task1_variation1.ttt")

os.environ["KITCHEN_SCENE_FILE"] = SCENE_PATH
os.environ["HEADLESS"] = "False"
os.environ.setdefault("GT_KEEP_ALIVE", "0")
os.environ.setdefault("GT_SPEED_MODE", "fast")
os.environ.setdefault("GT_RECORD_EVERY_N", "3")
os.environ.setdefault("GT_EXEC_INTERP_STEPS", "5")
os.environ.setdefault("GT_GO_HOME_INTERP_STEPS", "30")
os.environ.setdefault("GT_GO_HOME_EXEC_STEPS", "4")
os.environ.setdefault("GT_RELEASE_HOLD_STEPS", "16")
os.environ.setdefault("BOX_REGION_SAMPLE_PADDING", "0.01")
os.environ.setdefault("BOX_REGION_SAMPLE_GRID", "5")
os.environ.setdefault("BOX_REGION_SAMPLE_Z_OFFSET", "0.01")
os.environ.setdefault("BOX_REGION_OCCUPANCY_PAD_XY", "0.02")
os.environ.setdefault("BOX_REGION_OCCUPANCY_PAD_Z", "0.08")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import ground_truth_orchestrator_variation1_easy as v1_easy
from segmentation_video_recorder import SegmentationVideoRecorder
from live_panel_video_recorder import LivePanelVideoRecorder
from live_segmentation_viewer import LiveSegmentationViewer
from tkinter_segmentation_viewer import TkinterSegmentationViewer


class Variation1EasyLiveSegRunner:
    ACTION_SEQUENCE = [
        "Mug on box -> placement",
        "Cupboard mug -> placement",
        "Open box lid",
        "Grocery in box -> cupboard",
        "Grocery on table -> cupboard",
        "Table mug -> box",
        "Table mug -> box",
    ]

    def __init__(self):
        self.env = v1_easy.base.ENV
        self.viewer = None
        self.viewer_backend = "none"
        self.mask_recorder = None
        self.panel_recorder = None

        self.known_objects = set()
        self.last_visible_count = -1
        self._viewer_dead_reported = False
        self.last_view_update_ts = 0.0
        self.min_view_update_period_s = 0.12

        self.action_sequence = list(self.ACTION_SEQUENCE)
        self.current_action_index = -1
        self._task_re = re.compile(r"Task\s+(\d+)")

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.mask_output_dir = os.path.join(THIS_DIR, "segmentation_videos_live_seg", run_id)
        self.panel_video_path = os.path.join(self.mask_output_dir, "live_panel.mp4")
        self.mask_fps = _env_int("LIVE_SEG_MASK_FPS", 20)
        self.panel_fps = _env_int("LIVE_PANEL_FPS", 10)
        self.enable_panel_record = os.environ.get("LIVE_PANEL_RECORD", "1").strip().lower() not in {
            "0", "false", "no"
        }

    def _extract_task_name(self, args, kwargs):
        task_name = kwargs.get("task_name")
        if task_name is not None:
            return str(task_name)
        for arg in args:
            if isinstance(arg, str) and self._task_re.search(arg):
                return arg
        return None

    def _set_current_subtask(self, task_name):
        if self.viewer is None or not task_name:
            return
        match = self._task_re.search(task_name)
        if not match:
            return
        idx = int(match.group(1)) - 1
        idx = max(0, min(idx, len(self.action_sequence) - 1))
        self.current_action_index = idx
        label = self.action_sequence[idx]
        try:
            self.viewer.set_current_action(action_index=idx, action_label=label)
        except Exception:
            pass

    def _install_task_hooks(self):
        runner = self
        base_mod = v1_easy.base
        originals = {}

        def _wrap(name):
            original = getattr(base_mod, name)
            originals[name] = original

            def wrapped(*args, **kwargs):
                task_name = runner._extract_task_name(args, kwargs)
                runner._set_current_subtask(task_name)
                return original(*args, **kwargs)

            setattr(base_mod, name, wrapped)

        for fn_name in (
            "run_standard_pick_place",
            "run_cupboard_pick_place",
            "run_box_pick_place",
            "run_open_box",
        ):
            _wrap(fn_name)
        return originals

    def _restore_task_hooks(self, originals):
        base_mod = v1_easy.base
        for name, fn in originals.items():
            setattr(base_mod, name, fn)

    def _record_live_panel_frame(self):
        if self.viewer is None or self.panel_recorder is None:
            return
        try:
            frame = self.viewer.get_latest_frame()
        except Exception:
            frame = None
        self.panel_recorder.record_frame(frame)

    def update_view(self, event=""):
        if self.viewer is None:
            return set()

        proc = getattr(self.viewer, "viewer_proc", None)
        if proc is not None and not proc.is_alive():
            if not self._viewer_dead_reported:
                print("[Live Viewer] Viewer process died. Disabling live window and continuing.")
                self._viewer_dead_reported = True
            try:
                self.viewer.stop()
            except Exception:
                pass
            self.viewer = None
            self.viewer_backend = "none"
            return set()

        detected = self.viewer.update()
        if detected is None:
            detected = set()

        self._record_live_panel_frame()

        current_count = len(detected)
        if event or current_count != self.last_visible_count:
            label = event if event else "LIVE"
            print(f"[{label}] Combined visible object count (all cameras): {current_count}")
            self.last_visible_count = current_count

        new_objects = detected - self.known_objects
        if new_objects:
            print(f"[DISCOVERY] New: {sorted(new_objects)}")

        self.known_objects.update(detected)
        return detected

    def _on_world_step(self):
        if self.viewer is None:
            return
        now = time.time()
        if (now - self.last_view_update_ts) < self.min_view_update_period_s:
            return
        self.last_view_update_ts = now
        self.update_view()

    def _start_viewer(self):
        pref = os.environ.get("LIVE_SEG_VIEWER_BACKEND", "auto").strip().lower()

        def _start(name):
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
                    alive = proc.is_alive()
                    if alive:
                        self._viewer_dead_reported = False
                    return alive
                self._viewer_dead_reported = False
                return True
            except Exception as e:
                print(f"[Live Viewer] Failed to start {name} backend: {e}")
                self.viewer = None
                self.viewer_backend = "none"
                return False

        if pref == "tkinter":
            ok = _start("tkinter")
            return ok or _start("opencv")
        if pref == "opencv":
            return _start("opencv")

        ok = _start("tkinter")
        if ok:
            return True
        print("[Live Viewer] Tkinter backend failed. Falling back to OpenCV.")
        return _start("opencv")

    def run(self):
        print("=" * 70)
        print("VARIATION 1 EASY - LIVE SEGMENTATION")
        print("=" * 70)
        print(f"Scene: {SCENE_PATH}")
        print("Showing segmentation masks + subtask action sequence.")

        self.mask_recorder = SegmentationVideoRecorder(
            self.env, output_dir=self.mask_output_dir, fps=self.mask_fps
        )
        if self.enable_panel_record:
            self.panel_recorder = LivePanelVideoRecorder(self.panel_video_path, fps=self.panel_fps)
        else:
            self.panel_recorder = None

        v1_easy.base.MASK_RECORDER = self.mask_recorder
        v1_easy.base.STEP_CALLBACK = self._on_world_step
        v1_easy.base.ACTION_PROGRESS_CALLBACK = None

        live_ok = self._start_viewer()
        if not live_ok:
            print("[Live Viewer] Could not start a live viewer. Continuing without window.")
            self.viewer = None
            self.viewer_backend = "none"
        print(f"[Live Viewer] Active backend: {self.viewer_backend}")

        if self.viewer is not None:
            try:
                self.viewer.set_action_sequence(self.action_sequence)
                self.viewer.set_current_action(action_index=None, action_label=None)
            except Exception:
                pass

        self.update_view("INITIAL")
        originals = self._install_task_hooks()

        try:
            v1_easy.main()
        finally:
            self._restore_task_hooks(originals)
            v1_easy.base.STEP_CALLBACK = None
            v1_easy.base.MASK_RECORDER = None
            v1_easy.base.ACTION_PROGRESS_CALLBACK = None

            if self.mask_recorder:
                self.mask_recorder.release()
                self.mask_recorder = None
                print(f"Mask videos saved to: {self.mask_output_dir}")

            if self.panel_recorder:
                self.panel_recorder.release()
                self.panel_recorder = None

            if self.viewer:
                try:
                    self.viewer.stop()
                except Exception:
                    pass
                self.viewer = None


def main():
    runner = Variation1EasyLiveSegRunner()
    runner.run()


if __name__ == "__main__":
    main()
