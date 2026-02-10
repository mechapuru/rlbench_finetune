#!/usr/bin/env python3
"""
Demo Replanning Script
======================
Demonstrates 3 replanning scenarios in a specific order:

INITIAL PLAN (deliberately wrong - doesn't account for blockers):
- Put groceries in cupboard
- Slide open box
- Put mugs on table

REPLAN 1: mug_cupboard blocks cupboard_boundary
- Move mug_cupboard to table first, then continue groceries

REPLAN 2: mug_box blocks box_lid  
- Move mug_box aside first, then slide open box

REPLAN 3: Discovery of mug_inside_box (unknown until lid opened)
- After opening box, discover new mug, add to plan

Run: python -m vlm_pipeline.demo_replanning
     python -m vlm_pipeline.demo_replanning --record  # With video recording
"""

import os
import sys

# Configure Qt for GUI BEFORE any other imports
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

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.vlm_planner import ActionSkeleton
from vlm_pipeline.vlm_executor_v2 import VLMExecutorV2, ExecutionResult, ExecutionStatus

# Video recorder
try:
    from video_recorder import VideoRecorder
    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False
    print("Warning: video_recorder not available")


@dataclass
class DemoReplanCycle:
    """Information about a replan cycle."""
    cycle_number: int
    plan: List[str]
    failure_action: Optional[str]
    failure_reason: Optional[str]
    replan_reason: Optional[str]
    completed_actions: List[str]
    success: bool


class DemoReplanningPipeline:
    """
    Demo pipeline showing 3 specific replanning scenarios.
    """
    
    def __init__(self, record_video: bool = False, video_dir: str = "demo_videos"):
        self.executor = VLMExecutorV2()
        self.env = None
        self.cycles: List[DemoReplanCycle] = []
        self.all_completed: List[str] = []
        
        # Video recording
        self.record_video = record_video
        self.video_dir = video_dir
        self.video_recorder = None
        
    def initialize(self) -> bool:
        """Initialize the environment."""
        print("\n" + "=" * 70)
        print("DEMO REPLANNING PIPELINE")
        print("=" * 70)
        print("This demo shows 3 replanning scenarios:\n")
        print("  REPLAN 1: mug_cupboard blocks grocery placement in cupboard")
        print("  REPLAN 2: mug_box blocks sliding open the box lid")
        print("  REPLAN 3: Discover mug_inside_box after opening lid (unknown object)")
        if self.record_video:
            print(f"\n  📹 VIDEO RECORDING ENABLED -> {self.video_dir}/")
        print("=" * 70)
        
        try:
            # Import ENV from streams (this creates it properly in main thread)
            from rlbench_kitchen_streams import ENV
            print("\n[Demo] Using shared RLBench environment from streams...")
            self.env = ENV
            print("[Demo] Settling physics...")
            for _ in range(50):
                self.env.pr.step()
            print("[Demo] Environment ready!\n")
            
            # Initialize video recorder
            if self.record_video and HAS_VIDEO:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_subdir = os.path.join(self.video_dir, f"demo_{timestamp}")
                self.video_recorder = VideoRecorder(
                    self.env, 
                    output_dir=video_subdir,
                    fps=20
                )
                print(f"[Demo] Video recorder initialized -> {video_subdir}/")
            
            # Set env in executor
            self.executor.set_env(self.env)
            return True
            
        except Exception as e:
            print(f"[Demo] Error initializing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def shutdown(self):
        """Shutdown environment and save videos."""
        # Save videos first
        if self.video_recorder:
            print("\n[Demo] Saving videos...")
            self.video_recorder.release()
            self.video_recorder = None  # Prevent further recording
        
        if self.env:
            try:
                self.env.pr.stop()
                self.env.pr.shutdown()
            except:
                pass
            print("[Demo] Environment shutdown.")
    
    def record_frame(self):
        """Record a single frame if video recording is enabled."""
        if self.video_recorder:
            self.video_recorder.record_step()
    
    def record_frames(self, n: int = 10):
        """Record multiple frames (for pauses/transitions)."""
        for _ in range(n):
            if self.env:
                self.env.pr.step()
            self.record_frame()
    
    def print_plan(self, plan: List[ActionSkeleton], title: str):
        """Pretty print a plan."""
        print(f"\n{'─' * 60}")
        print(f"📋 {title}")
        print(f"{'─' * 60}")
        for i, action in enumerate(plan, 1):
            print(f"  {i:2d}. {action}")
        print()
    
    def print_failure(self, action: ActionSkeleton, reason: str):
        """Print failure message."""
        print(f"\n{'!' * 60}")
        print(f"❌ EXECUTION FAILED")
        print(f"{'!' * 60}")
        print(f"  Action: {action}")
        print(f"  Reason: {reason}")
        print()
    
    def print_replan(self, replan_num: int, reason: str, new_plan: List[ActionSkeleton]):
        """Print replanning message."""
        print(f"\n{'═' * 60}")
        print(f"🔄 REPLAN #{replan_num}")
        print(f"{'═' * 60}")
        print(f"  Trigger: {reason}")
        print(f"  New plan ({len(new_plan)} actions):")
        for i, action in enumerate(new_plan, 1):
            print(f"    {i:2d}. {action}")
        print()
    
    def execute_actions(self, actions: List[ActionSkeleton]) -> tuple:
        """
        Execute actions until failure or completion.
        Returns: (completed_actions, failed_action, failure_reason)
        """
        completed = []
        i = 0
        
        while i < len(actions):
            action = actions[i]
            
            # Record frames before action
            self.record_frames(5)
            
            # Check preconditions for demo failures
            failure = self._check_preconditions(action)
            if failure:
                self.record_frames(20)  # Record the failure moment
                return completed, action, failure
            
            # Handle pick+place pairs
            if action.action_name == 'pick':
                if i + 1 < len(actions) and actions[i + 1].action_name == 'place':
                    pick_action = action
                    place_action = actions[i + 1]
                    
                    print(f"  [Executing] {pick_action} + {place_action}...")
                    
                    # Check place preconditions too
                    failure = self._check_preconditions(place_action)
                    if failure:
                        self.record_frames(20)
                        return completed, place_action, failure
                    
                    try:
                        result = self.executor.execute_pick_place_pair(pick_action, place_action)
                        
                        # Record frames during/after execution
                        self.record_frames(30)
                        
                        if result.status != ExecutionStatus.SUCCESS:
                            return completed, pick_action, result.error_message or "Execution failed"
                        
                        completed.append(str(pick_action))
                        completed.append(str(place_action))
                        self.all_completed.append(str(pick_action))
                        self.all_completed.append(str(place_action))
                        print(f"  [Success] {pick_action} + {place_action}")
                        i += 2
                        continue
                        
                    except Exception as e:
                        return completed, pick_action, str(e)
                else:
                    return completed, action, "Pick without following place"
            
            # Handle open-lid
            elif action.action_name in ['open-lid', 'open_lid']:
                print(f"  [Executing] {action}...")
                try:
                    result = self.executor.execute_single_action(action)
                    
                    # Record frames during/after execution
                    self.record_frames(30)
                    
                    if result.status != ExecutionStatus.SUCCESS:
                        return completed, action, result.error_message or "Execution failed"
                    
                    completed.append(str(action))
                    self.all_completed.append(str(action))
                    print(f"  [Success] {action}")
                    i += 1
                    
                except Exception as e:
                    return completed, action, str(e)
            
            else:
                i += 1
        
        return completed, None, None
    
    def _check_preconditions(self, action: ActionSkeleton) -> Optional[str]:
        """
        Check preconditions for demo-specific failures.
        Returns failure reason or None if OK.
        
        We check BEFORE picking if the subsequent place would fail.
        This way we fail before even starting the pick action.
        """
        # FAILURE 1: Picking a grocery that will be placed in cupboard_boundary
        # when mug_cupboard is still inside (check on PICK, not place)
        if action.action_name == 'pick':
            obj = action.args[0]
            groceries = ['soup', 'mustard', 'spam']  # These go to cupboard_boundary
            
            if obj in groceries:
                # Check if mug_cupboard has been moved
                if 'pick(mug_cupboard)' not in self.all_completed and 'pick(mug3)' not in self.all_completed:
                    # Check actual position of mug3
                    try:
                        mug3_pos = self.env.objects['mug3'].get_position()
                        cupboard_region = self.env.regions.get('cupboard_boundary')
                        if cupboard_region:
                            bbox = cupboard_region.get_bounding_box()
                            if bbox[0] <= mug3_pos[0] <= bbox[3] and bbox[1] <= mug3_pos[1] <= bbox[4]:
                                return f"BLOCKED: Cannot pick {obj} for cupboard_boundary - mug_cupboard is inside cupboard and must be moved first"
                    except:
                        # Fallback: just check if not completed
                        return f"BLOCKED: Cannot pick {obj} for cupboard_boundary - mug_cupboard is inside cupboard and must be moved first"
        
        # FAILURE 2: Opening lid when mug_box is on top
        if action.action_name == 'open-lid':
            lid = action.args[0]
            if lid == 'box_lid':
                # Check if mug_box has been moved
                if 'pick(mug_box)' not in self.all_completed and 'pick(mug2)' not in self.all_completed:
                    # Check actual position of mug2
                    try:
                        mug2_pos = self.env.objects['mug2'].get_position()
                        box_region = self.env.regions.get('box_boundary')
                        if box_region:
                            bbox = box_region.get_bounding_box()
                            if bbox[0] <= mug2_pos[0] <= bbox[3] and bbox[1] <= mug2_pos[1] <= bbox[4]:
                                return "BLOCKED: mug_box is on top of box_lid - must move it first"
                    except:
                        # Fallback
                        return "BLOCKED: mug_box is on top of box_lid - must move it first"
        
        return None
    
    def run(self):
        """Run the demo with 3 replanning scenarios."""
        
        print("\n" + "=" * 70)
        print("STARTING DEMO EXECUTION")
        print("=" * 70)
        
        # ================================================================
        # INITIAL PLAN (deliberately wrong - doesn't account for blockers)
        # ================================================================
        # sugar/crackers go to TOP SHELF first (will succeed)
        # Then soup to cupboard_boundary (will FAIL - mug_cupboard blocks)
        # Note: mug_inside_box is NOT in this plan (unknown until lid opens)
        
        initial_plan = [
            # Top shelf items FIRST - these succeed
            ActionSkeleton('pick', ('sugar',)),
            ActionSkeleton('place', ('sugar', 'cupboard_boundary_top')),
            ActionSkeleton('pick', ('crackers',)),
            ActionSkeleton('place', ('crackers', 'cupboard_boundary_top')),
            # Then try groceries for cupboard_boundary - WILL FAIL at pick(soup)
            ActionSkeleton('pick', ('soup',)),
            ActionSkeleton('place', ('soup', 'cupboard_boundary')),
            ActionSkeleton('pick', ('mustard',)),
            ActionSkeleton('place', ('mustard', 'cupboard_boundary')),
            ActionSkeleton('pick', ('spam',)),
            ActionSkeleton('place', ('spam', 'cupboard_boundary')),
            # Then open box (will fail - mug_box blocks)
            ActionSkeleton('open-lid', ('box_lid',)),
            # Then mugs on table
            ActionSkeleton('pick', ('mug_cupboard',)),
            ActionSkeleton('place', ('mug_cupboard', 'placement_boundary')),
            ActionSkeleton('pick', ('mug_box',)),
            ActionSkeleton('place', ('mug_box', 'placement_boundary')),
            # Note: mug_inside_box NOT included (unknown)
        ]
        
        self.print_plan(initial_plan, "INITIAL PLAN (VLM output - has errors)")
        input("\n>>> Press Enter to start execution...")
        
        # Execute initial plan
        print("\n[Cycle 1] Executing initial plan...")
        completed, failed, reason = self.execute_actions(initial_plan)
        
        if failed:
            self.print_failure(failed, reason)
            self.cycles.append(DemoReplanCycle(
                cycle_number=1,
                plan=[str(a) for a in initial_plan],
                failure_action=str(failed),
                failure_reason=reason,
                replan_reason=None,
                completed_actions=completed,
                success=False
            ))
            
            # ================================================================
            # REPLAN 1: Move mug_cupboard first, then continue with remaining groceries
            # ================================================================
            input("\n>>> Press Enter to trigger REPLAN 1...")
            
            replan1 = [
                # First move the blocking mug to table
                ActionSkeleton('pick', ('mug_cupboard',)),
                ActionSkeleton('place', ('mug_cupboard', 'placement_boundary')),
                # Then continue with remaining groceries (soup, mustard, spam)
                # Note: sugar and crackers already done in initial plan
                ActionSkeleton('pick', ('soup',)),
                ActionSkeleton('place', ('soup', 'cupboard_boundary')),
                ActionSkeleton('pick', ('mustard',)),
                ActionSkeleton('place', ('mustard', 'cupboard_boundary')),
                ActionSkeleton('pick', ('spam',)),
                ActionSkeleton('place', ('spam', 'cupboard_boundary')),
                # Then try to open box (will fail - mug_box blocks)
                ActionSkeleton('open-lid', ('box_lid',)),
            ]
            
            self.print_replan(1, "mug_cupboard blocking cupboard_boundary", replan1)
            input("\n>>> Press Enter to execute REPLAN 1...")
            
            completed, failed, reason = self.execute_actions(replan1)
            
            self.cycles.append(DemoReplanCycle(
                cycle_number=2,
                plan=[str(a) for a in replan1],
                failure_action=str(failed) if failed else None,
                failure_reason=reason,
                replan_reason="mug_cupboard blocking cupboard_boundary",
                completed_actions=completed,
                success=failed is None
            ))
            
            if failed:
                self.print_failure(failed, reason)
                
                # ================================================================
                # REPLAN 2: Move mug_box first, then open lid
                # ================================================================
                input("\n>>> Press Enter to trigger REPLAN 2...")
                
                replan2 = [
                    # First move the blocking mug
                    ActionSkeleton('pick', ('mug_box',)),
                    ActionSkeleton('place', ('mug_box', 'placement_boundary')),
                    # Then open the lid
                    ActionSkeleton('open-lid', ('box_lid',)),
                ]
                
                self.print_replan(2, "mug_box blocking box_lid", replan2)
                input("\n>>> Press Enter to execute REPLAN 2...")
                
                completed, failed, reason = self.execute_actions(replan2)
                
                self.cycles.append(DemoReplanCycle(
                    cycle_number=3,
                    plan=[str(a) for a in replan2],
                    failure_action=str(failed) if failed else None,
                    failure_reason=reason,
                    replan_reason="mug_box blocking box_lid",
                    completed_actions=completed,
                    success=failed is None
                ))
                
                if not failed:
                    # ================================================================
                    # REPLAN 3: Discovered mug_inside_box after opening lid!
                    # ================================================================
                    print("\n" + "=" * 60)
                    print("🔍 NEW OBJECT DISCOVERED!")
                    print("=" * 60)
                    print("  After opening box lid, detected: mug_inside_box")
                    print("  This object was NOT in the original plan!")
                    print("  Triggering replan to include it...")
                    
                    input("\n>>> Press Enter to trigger REPLAN 3...")
                    
                    replan3 = [
                        # Add the newly discovered mug
                        ActionSkeleton('pick', ('mug_inside_box',)),
                        ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
                    ]
                    
                    self.print_replan(3, "Discovered mug_inside_box after opening lid", replan3)
                    input("\n>>> Press Enter to execute REPLAN 3...")
                    
                    completed, failed, reason = self.execute_actions(replan3)
                    
                    self.cycles.append(DemoReplanCycle(
                        cycle_number=4,
                        plan=[str(a) for a in replan3],
                        failure_action=str(failed) if failed else None,
                        failure_reason=reason,
                        replan_reason="Discovered mug_inside_box (unknown until lid opened)",
                        completed_actions=completed,
                        success=failed is None
                    ))
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal cycles: {len(self.cycles)}")
        print(f"Total replans: {len(self.cycles) - 1}")
        print(f"Total actions completed: {len(self.all_completed)}")
        
        print("\n📋 All completed actions:")
        for i, action in enumerate(self.all_completed, 1):
            print(f"  {i:2d}. {action}")
        
        print("\n🔄 Replan triggers:")
        print("  1. mug_cupboard blocking cupboard_boundary")
        print("  2. mug_box blocking box_lid")
        print("  3. Discovered mug_inside_box (unknown until lid opened)")
        
        # Save log
        log = {
            "timestamp": datetime.now().isoformat(),
            "total_cycles": len(self.cycles),
            "total_replans": len(self.cycles) - 1,
            "total_actions_completed": len(self.all_completed),
            "completed_actions": self.all_completed,
            "cycles": [asdict(c) for c in self.cycles]
        }
        
        os.makedirs("demo_logs", exist_ok=True)
        log_file = f"demo_logs/demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"\n[Demo] Log saved to: {log_file}")
        
        return log


def main():
    parser = argparse.ArgumentParser(description="Demo Replanning Pipeline")
    parser.add_argument("--record", action="store_true",
                        help="Record video from all 5 cameras")
    parser.add_argument("--video-dir", type=str, default="demo_videos",
                        help="Directory to save videos (default: demo_videos)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Run without pausing for user input")
    args = parser.parse_args()
    
    pipeline = DemoReplanningPipeline(
        record_video=args.record,
        video_dir=args.video_dir
    )
    
    try:
        if not pipeline.initialize():
            print("Failed to initialize!")
            return
        
        # Record initial scene
        if pipeline.video_recorder:
            print("[Demo] Recording initial scene...")
            pipeline.record_frames(60)  # 3 seconds at 20fps
        
        result = pipeline.run()
        
        # Record final scene
        if pipeline.video_recorder:
            print("[Demo] Recording final scene...")
            pipeline.record_frames(60)  # 3 seconds
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        
        if pipeline.video_recorder:
            print(f"\n📹 Videos saved to: {pipeline.video_dir}/")
        
        # Keep simulation open (but stop recording - videos already saved)
        print("\nPress Ctrl+C to exit...")
        try:
            while True:
                pipeline.env.pr.step()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            pipeline.shutdown()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        pipeline.shutdown()


if __name__ == "__main__":
    main()
