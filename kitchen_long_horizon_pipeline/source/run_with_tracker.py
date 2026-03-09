"""
Run Ground Truth Orchestrator with Live Object Tracker

This script runs the ground truth orchestrator alongside visibility tracking
that shows which objects are visible during execution.

The tracker detects when hidden objects (like mug4 inside a closed box) 
become visible during task execution.

Usage:
    python run_with_tracker.py
"""
import os
import sys
import time

# Configure Qt for GUI
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

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import components
from rlbench_kitchen_streams import ENV, get_stream_map
from video_recorder import VideoRecorder
from live_object_tracker import ObjectVisibilityTracker

# Import ground truth orchestrator functions
import ground_truth_orchestrator as gt_orchestrator
from ground_truth_orchestrator import (
    step_and_record, go_home, 
    run_standard_pick_place, run_cupboard_pick_place, 
    run_box_pick_place, run_open_box
)


class TrackedOrchestrator:
    """
    Orchestrator that runs tasks while tracking visible objects.
    Prints visibility updates to console.
    """
    
    def __init__(self, env):
        self.env = env
        self.pr = env.pr
        
        # Initialize visibility tracker (console-based)
        self.visibility_tracker = ObjectVisibilityTracker(env)
        
        # Video recorder
        self.video_recorder = None
        
        # Track execution state
        self.visibility_log = []
        self.last_known = set()
        
    def log_visibility(self, event_name):
        """Log current visibility state at key events."""
        self.visibility_tracker.update()
        newly_visible = self.visibility_tracker.newly_visible
        
        log_entry = {
            'event': event_name,
            'time': time.time(),
            'known_objects': sorted(self.visibility_tracker.known_objects),
            'newly_visible': sorted(newly_visible) if newly_visible else [],
        }
        self.visibility_log.append(log_entry)
        
        # Print status
        box_open = self.visibility_tracker.check_box_opened()
        mug4_visible = 'mug4' in self.visibility_tracker.known_objects or 'mug_inside_box' in self.visibility_tracker.known_objects
        
        print(f"\n[{event_name}] Known: {len(self.visibility_tracker.known_objects)} objects | Box open: {box_open} | mug4 visible: {mug4_visible}")
        
        if newly_visible:
            print(f"  ★ Newly discovered: {', '.join(sorted(newly_visible))}")
        
        return log_entry
    
    def print_visibility_summary(self):
        """Print summary of object visibility throughout execution."""
        print("\n" + "="*70)
        print("OBJECT VISIBILITY LOG")
        print("="*70)
        
        for entry in self.visibility_log:
            event = entry['event']
            known = entry['known_objects']
            newly = entry['newly_visible']
            
            print(f"\n[{event}]")
            print(f"  Total known objects: {len(known)}")
            if newly:
                print(f"  ★ Newly discovered: {', '.join(newly)}")
        
        print("\n" + "="*70)
        
        # Final summary
        final_known = self.visibility_tracker.known_objects
        all_objects = set(self.env.name_to_obj.keys())
        still_hidden = all_objects - final_known
        
        print(f"\nFinal visibility state:")
        print(f"  Known objects: {len(final_known)}")
        print(f"  Still hidden: {len(still_hidden)}")
        if still_hidden:
            print(f"  Hidden objects: {', '.join(sorted(still_hidden))}")
        print("="*70)
    
    def run(self):
        """Run the full orchestration with tracking."""
        env = self.env
        pr = self.pr
        
        print("="*70)
        print("GROUND TRUTH ORCHESTRATOR WITH OBJECT VISIBILITY TRACKING")
        print("="*70)
        print("\nThis tracks which objects are visible during execution.")
        print("Hidden objects (like mug4 inside closed box) will be detected")
        print("when they become visible (e.g., after opening the box).\n")
        
        # Initialize video recorder
        print("Initializing video recorder...")
        self.video_recorder = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
        gt_orchestrator.VIDEO_RECORDER = self.video_recorder
        
        # Initial visibility check
        self.log_visibility("INITIAL_STATE")
        
        print("\nSettling physics...")
        for _ in range(50):
            pr.step()
            self.video_recorder.record_step()
        
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        for _ in range(10):
            pr.step()
            self.video_recorder.record_step()
        
        # Log after settling
        self.log_visibility("AFTER_SETTLING")
        
        results = []
        
        try:
            # ============================================
            # TASK 1: Pick mug3 from cupboard -> placement_boundary
            # ============================================
            self.log_visibility("BEFORE_TASK_1")
            success = run_cupboard_pick_place(
                env,
                object_name='mug3',
                target_region='placement_boundary',
                task_name="Task 1: Cupboard Mug -> Placement"
            )
            results.append(("Task 1: mug3 cupboard -> placement", success))
            self.log_visibility("AFTER_TASK_1")
            go_home(env)
            
            # ============================================
            # TASK 2: Pick 5 groceries from table -> cupboard
            # ============================================
            groceries_inside = ['soup', 'mustard', 'spam']
            groceries_top = ['sugar', 'crackers']
            
            task_num = 1
            for grocery in groceries_inside:
                self.log_visibility(f"BEFORE_TASK_2_{task_num}")
                success = run_standard_pick_place(
                    env,
                    object_name=grocery,
                    target_region='cupboard_boundary',
                    task_name=f"Task 2.{task_num}: {grocery} -> Cupboard (inside)"
                )
                results.append((f"Task 2.{task_num}: {grocery} -> cupboard", success))
                self.log_visibility(f"AFTER_TASK_2_{task_num}")
                go_home(env)
                task_num += 1
            
            for grocery in groceries_top:
                self.log_visibility(f"BEFORE_TASK_2_{task_num}")
                success = run_standard_pick_place(
                    env,
                    object_name=grocery,
                    target_region='cupboard_boundary_top',
                    task_name=f"Task 2.{task_num}: {grocery} -> Cupboard (top)"
                )
                results.append((f"Task 2.{task_num}: {grocery} -> cupboard_top", success))
                self.log_visibility(f"AFTER_TASK_2_{task_num}")
                go_home(env)
                task_num += 1
            
            # ============================================
            # TASK 3: Pick mug2 from box_boundary -> placement_boundary
            # ============================================
            self.log_visibility("BEFORE_TASK_3")
            success = run_box_pick_place(
                env,
                object_name='mug2',
                target_region='placement_boundary',
                task_name="Task 3: Box Mug -> Placement"
            )
            results.append(("Task 3: mug2 box -> placement", success))
            self.log_visibility("AFTER_TASK_3")
            go_home(env)
            
            # ============================================
            # TASK 4: Slide open box lid
            # This is where mug4 becomes visible!
            # ============================================
            self.log_visibility("BEFORE_TASK_4_BOX_OPEN")
            print("\n" + "*"*60)
            print("* OPENING BOX - Watch for mug4 becoming visible!")
            print("*"*60 + "\n")
            
            success = run_open_box(
                env,
                task_name="Task 4: Open Box Lid"
            )
            results.append(("Task 4: open box lid", success))
            
            # This should detect mug4 becoming visible
            self.log_visibility("AFTER_TASK_4_BOX_OPENED")
            go_home(env)
            
            # ============================================
            # TASK 5: Pick mug4 from inside box -> placement_boundary
            # ============================================
            self.log_visibility("BEFORE_TASK_5")
            success = run_box_pick_place(
                env,
                object_name='mug4',
                target_region='placement_boundary',
                task_name="Task 5: Mug Inside Box -> Placement"
            )
            results.append(("Task 5: mug4 box_inside -> placement", success))
            self.log_visibility("AFTER_TASK_5")
            go_home(env)
            
        except Exception as e:
            print(f"\nERROR during execution: {e}")
            import traceback
            traceback.print_exc()
        
        # ============================================
        # SUMMARY
        # ============================================
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        total = len(results)
        passed = sum(1 for _, s in results if s)
        for task, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {status} - {task}")
        print(f"\nTotal: {passed}/{total} tasks completed successfully")
        
        # Print visibility summary
        self.print_visibility_summary()
        
        # Release resources
        if self.video_recorder:
            self.video_recorder.release()
            print("\nVideo recording saved to 'orchestrator_videos/' directory")
        
        print("\n" + "="*70)
        print("Execution complete.")
        print("="*70)
        
        print("\nPress Ctrl+C to close simulation.")
        try:
            while True:
                pr.step()
        except KeyboardInterrupt:
            pass
        
        # Cleanup
        pr.stop()
        pr.shutdown()


def main():
    """Run the tracked orchestrator."""
    env = ENV
    orchestrator = TrackedOrchestrator(env)
    orchestrator.run()


if __name__ == "__main__":
    main()
