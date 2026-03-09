#!/usr/bin/env python3
"""
Live segmentation run with discovery-triggered VLM replanning.

Flow:
1) Pick mug on top of box (mug2 / mug_box) -> placement_boundary
2) Slide open box lid
3) AFTER open action completes, detect newly visible objects from segmentation
4) Raise failure code NEW_OBJECT_INTRODUCED_IN_SCENE
5) Call VLM (local/mock/remote) to replan remaining actions
6) Execute replanned actions (e.g., mug_inside_box -> placement_boundary)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Set


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

from vlm_pipeline.vlm_context_aggregator import VLMContextAggregator
from vlm_pipeline.vlm_executor_v2 import VLMExecutorV2, ExecutorConfig, ExecutionStatus
from vlm_pipeline.vlm_planner import VLMPlanner, MockVLMPlanner, ActionSkeleton


DISCOVERY_FAILURE_CODE = "NEW_OBJECT_INTRODUCED_IN_SCENE"

SCENE_TO_PDDL = {
    "mug1": "mug_table",
    "mug2": "mug_box",
    "mug3": "mug_cupboard",
    "mug4": "mug_inside_box",
    "soup": "soup",
    "mustard": "mustard",
    "spam": "spam",
    "sugar": "sugar",
    "crackers": "crackers",
    "box_lid": "box_lid",
}

MOVABLE_PDDL = {
    "mug_box",
    "mug_inside_box",
    "mug_table",
    "mug_cupboard",
    "soup",
    "mustard",
    "spam",
    "sugar",
    "crackers",
}


class OpenSlideDiscoveryReplanRunner:
    def __init__(
        self,
        env,
        use_remote_vlm: bool = False,
        remote_vlm_url: str = "",
        use_mock_vlm: bool = False,
        use_vision: bool = True,
        vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        vlm_4bit: bool = False,
    ):
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

        self.use_remote_vlm = use_remote_vlm
        self.remote_vlm_url = remote_vlm_url
        self.use_mock_vlm = use_mock_vlm
        self.use_vision = use_vision
        self.vlm_model = vlm_model
        self.vlm_4bit = vlm_4bit

        self.context_aggregator = VLMContextAggregator(
            visible_objects_only=True,
            enable_live_segmentation_view=False,
        )
        self.context_aggregator.set_env(self.env)

        self.executor = VLMExecutorV2(config=ExecutorConfig(pause_between_actions=0.1))
        self.executor.set_env(self.env)
        self.executor.set_step_callback(self._on_world_step)
        self.planner = None

    def _ensure_planner_loaded(self) -> bool:
        if self.planner is not None:
            return True

        if self.use_mock_vlm:
            print("[VLM] Using MOCK planner")
            self.planner = MockVLMPlanner()
            return self.planner.load_model()

        if self.use_remote_vlm:
            from vlm_pipeline.vlm_client import RemoteVLMPlanner

            url = self.remote_vlm_url or os.environ.get("VLM_SERVER_URL", "http://localhost:8000")
            print(f"[VLM] Using REMOTE planner: {url}")
            self.planner = RemoteVLMPlanner(server_url=url, use_vision=self.use_vision)
            return self.planner.load_model()

        print(f"[VLM] Using LOCAL planner: {self.vlm_model}")
        self.planner = VLMPlanner(model_name=self.vlm_model, use_4bit=self.vlm_4bit)
        return self.planner.load_model()

    def update_view(self, event: str = "") -> Set[str]:
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

    def step(self, count: int = 1):
        for _ in range(count):
            self.pr.step()
            if self.video_recorder:
                self.video_recorder.record_step()
            if self.mask_recorder:
                self.mask_recorder.record_step()
            self._on_world_step()

    def _start_backend(self, name: str) -> bool:
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

    def _start_viewer(self) -> bool:
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

    def _scene_to_pddl(self, scene_names: Set[str]) -> List[str]:
        mapped = []
        for name in sorted(scene_names):
            mapped_name = SCENE_TO_PDDL.get(name, name)
            mapped.append(mapped_name)
        return mapped

    def _build_replan_prompt(
        self, newly_visible_pddl: List[str], state_text: str, completed_actions: List[ActionSkeleton]
    ) -> str:
        completed_str = "\n".join([f"  {i+1}. {a}" for i, a in enumerate(completed_actions)])
        target_str = ", ".join(newly_visible_pddl)
        return f"""=== REPLANNING REQUIRED ===

FAILURE CODE:
  {DISCOVERY_FAILURE_CODE}

WHAT HAPPENED:
  open-lid action has completed successfully.
  Segmentation detected NEW object(s) becoming visible: {target_str}

ALREADY COMPLETED ACTIONS:
{completed_str}

CURRENT STATE SNAPSHOT:
{state_text}

TASK FOR THIS REPLAN:
Generate ONLY the remaining actions required to move the newly visible object(s) to placement_boundary.
Do not repeat completed actions. Do not add unrelated object moves.

Output ONLY numbered action lines:
1. pick(object)
2. place(object, placement_boundary)
..."""

    def _run_vlm_replan_for_new_objects(
        self, newly_visible_scene: Set[str], completed_actions: List[ActionSkeleton]
    ) -> bool:
        newly_visible_pddl_all = self._scene_to_pddl(newly_visible_scene)
        targets = [name for name in newly_visible_pddl_all if name in MOVABLE_PDDL]

        print("\n" + "=" * 70)
        print("FAILURE DETECTED")
        print("=" * 70)
        print(f"Code: {DISCOVERY_FAILURE_CODE}")
        print(f"Newly visible (scene): {sorted(newly_visible_scene)}")
        print(f"Newly visible (pddl) : {newly_visible_pddl_all}")
        print(f"Replan targets        : {targets}")
        print("=" * 70)

        if not targets:
            print("[Replan] No movable newly visible objects. Skipping VLM replan.")
            return True

        if not self._ensure_planner_loaded():
            print("[Replan] ERROR: Could not load VLM planner.")
            return False

        replan_goal = (
            "Move the newly visible object(s) to placement_boundary. "
            f"Targets: {', '.join(targets)}."
        )

        self.context_aggregator.update_visible_objects(event="replan-trigger")
        bundle = self.context_aggregator.create_prompt_bundle(replan_goal)
        replan_prompt = self._build_replan_prompt(targets, bundle.state_text, completed_actions)

        print("\n[Replan] Querying VLM...")
        if self.use_vision:
            plan_result = self.planner.generate_plan(
                image=bundle.composite_image,
                system_prompt=bundle.system_prompt,
                user_prompt=replan_prompt,
            )
        else:
            if hasattr(self.planner, "generate_plan_text_only"):
                plan_result = self.planner.generate_plan_text_only(
                    system_prompt=bundle.system_prompt,
                    user_prompt=replan_prompt,
                )
            else:
                plan_result = self.planner.generate_plan(
                    image=bundle.composite_image,
                    system_prompt=bundle.system_prompt,
                    user_prompt=replan_prompt,
                )

        if not plan_result.success or not plan_result.skeleton:
            print(f"[Replan] ERROR: VLM replan failed: {plan_result.error_message}")
            print(f"[Replan] Raw output:\n{plan_result.raw_output}")
            return False

        print(f"[Replan] VLM produced {len(plan_result.skeleton)} actions:")
        for i, action in enumerate(plan_result.skeleton, 1):
            print(f"  {i}. {action}")

        if hasattr(self.planner, "validate_plan"):
            is_valid, errors = self.planner.validate_plan(plan_result.skeleton)
            if not is_valid:
                print("[Replan] WARNING: Plan validation reported issues:")
                for err in errors:
                    print(f"  - {err}")

        print("\n[Replan] Executing replanned actions...")
        exec_result = self.executor.execute_plan(plan_result.skeleton)
        if exec_result.status != ExecutionStatus.SUCCESS:
            print(f"[Replan] ERROR: Execution failed: {exec_result.error_message}")
            return False

        print("[Replan] SUCCESS: Replanned actions executed.")
        return True

    def run(self):
        print("=" * 70)
        print("LIVE CHECK + DISCOVERY FAILURE + VLM REPLAN")
        print("=" * 70)
        print("Flow: mug2->placement, open-lid, detect new object, replan, execute.")
        print("Failure code on discovery: NEW_OBJECT_INTRODUCED_IN_SCENE")
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
        completed_actions: List[ActionSkeleton] = []

        try:
            print("\n" + "=" * 50)
            print("TASK 1: Pick mug2 (box top) -> placement_boundary")
            print("=" * 50)
            ok_mug2 = run_box_pick_place(self.env, "mug2", "placement_boundary", "Task 1: mug2 move")
            results.append(("Task 1: mug2 move", ok_mug2))
            if ok_mug2:
                completed_actions.extend(
                    [
                        ActionSkeleton("pick", ("mug_box",)),
                        ActionSkeleton("place", ("mug_box", "placement_boundary")),
                    ]
                )
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
            if ok_open:
                completed_actions.append(ActionSkeleton("open-lid", ("box_lid",)))

            self.step(20)
            self.update_view("AFTER_OPEN")
            after_open = self.viewer.get_detected_objects() if self.viewer else set()
            print(f"AFTER opening : count={len(after_open)}, mug4={'YES' if 'mug4' in after_open else 'NO'}")

            delta = len(after_open) - len(before_open)
            new_scene_objects = after_open - before_open
            print("\n" + "=" * 70)
            print("COUNT CHECK (around open-lid step)")
            print("=" * 70)
            print(f"Before count: {len(before_open)}")
            print(f"After count : {len(after_open)}")
            print(f"Delta       : {delta:+d}")
            print(f"Increased?  : {'YES' if delta > 0 else 'NO'}")
            print(f"New objects : {sorted(new_scene_objects)}")
            print("=" * 70)

            if ok_open and new_scene_objects:
                ok_replan = self._run_vlm_replan_for_new_objects(new_scene_objects, completed_actions)
                results.append(("Task 3: VLM replan for discovered object(s)", ok_replan))
                self.update_view("AFTER_REPLAN_EXEC")
            else:
                print("\n[Replan] Discovery trigger not activated.")
                if not ok_open:
                    print("[Replan] Reason: open-lid failed.")
                elif not new_scene_objects:
                    print("[Replan] Reason: no new objects appeared after open.")

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
            print("Videos:")
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

            try:
                self.context_aggregator.shutdown()
            except Exception:
                pass
            if self.video_recorder:
                self.video_recorder.release()
            if self.mask_recorder:
                self.mask_recorder.release()
            if self.viewer:
                self.viewer.stop()
            self.pr.stop()
            self.pr.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Open-lid discovery trigger + VLM replan (live)")
    parser.add_argument("--remote", action="store_true", help="Use remote VLM server")
    parser.add_argument("--remote-url", type=str, default="", help="Remote VLM server URL")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM (no server/GPU)")
    parser.add_argument("--text-only", action="store_true", help="Disable vision input to planner")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Local model name")
    parser.add_argument("--vlm-4bit", action="store_true", help="Enable 4-bit quantization for local VLM")
    args = parser.parse_args()

    runner = OpenSlideDiscoveryReplanRunner(
        ENV,
        use_remote_vlm=args.remote,
        remote_vlm_url=args.remote_url,
        use_mock_vlm=args.mock,
        use_vision=not args.text_only,
        vlm_model=args.vlm_model,
        vlm_4bit=args.vlm_4bit,
    )
    runner.run()


if __name__ == "__main__":
    main()
