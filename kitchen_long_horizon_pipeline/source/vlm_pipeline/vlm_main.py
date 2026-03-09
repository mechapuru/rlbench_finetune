"""
VLM Main Pipeline (Entry Point)
===============================
Main loop that wires together:
- Module 1: Context Aggregator (captures frames + state)
- Module 2: VLM Planner (generates action skeleton)
- Module 2.5: Skeleton to PDDL translator
- Module 3: Executor (runs actions, monitors for failures)

Supports:
- Initial planning from visual context
- Placeholder for manual interruption/replanning
- Object tracking (expected vs visible)
"""

import os
import sys
import time
import json
import argparse
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure Qt for GUI before any imports
def _configure_qt():
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    candidate_dirs = [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
    ]
    for candidate in candidate_dirs:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break

_configure_qt()

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pddlstream'))

# Import VLM pipeline modules
from vlm_pipeline.vlm_context_aggregator import VLMContextAggregator, PromptBundle
from vlm_pipeline.vlm_planner import VLMPlanner, MockVLMPlanner, ActionSkeleton, PlanResult
from vlm_pipeline.skeleton_to_pddl import SkeletonToPDDL
# Use V2 executor with PDDL-based execution (like ground_truth_orchestrator)
try:
    from vlm_pipeline.vlm_executor_v2 import VLMExecutorV2 as VLMExecutor, ExecutionStatus, ExecutionResult
    print("[Import] Using VLMExecutorV2 with PDDL execution")
except ImportError:
    from vlm_pipeline.vlm_executor import VLMExecutor, ExecutionStatus, ExecutionResult
    print("[Import] Fallback to original VLMExecutor")


@dataclass
class PipelineConfig:
    """Configuration for the VLM pipeline."""
    use_mock_vlm: bool = False  # Use mock planner (no GPU needed)
    use_remote_vlm: bool = False  # Use remote VLM server
    remote_vlm_url: str = ""  # URL of remote VLM server
    use_vision: bool = True  # Use image input for planning
    visible_objects_only: bool = False  # Use segmentation-visible objects only in context
    live_segmentation_view: bool = False  # Show live segmentation masks with object list
    use_real_env: bool = True   # Use real RLBench environment
    headless: bool = False      # Run simulation headless
    max_replan_attempts: int = 3
    save_logs: bool = True
    log_dir: str = "vlm_pipeline_logs"
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_4bit: bool = False


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    current_plan: List[str]
    executed_actions: List[str]
    replan_count: int = 0
    is_interrupted: bool = False
    interrupt_reason: Optional[str] = None
    unknown_objects: List[str] = None
    
    def __post_init__(self):
        if self.unknown_objects is None:
            self.unknown_objects = []


class VLMPipeline:
    """
    Main VLM pipeline that orchestrates all modules.
    
    Flow:
    1. Capture visual context (5 cameras + state)
    2. Query VLM for action skeleton
    3. Translate skeleton to executable format
    4. Execute actions, monitoring for failures
    5. On interruption/failure, trigger replanning
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize modules
        self.context_aggregator = VLMContextAggregator(
            visible_objects_only=self.config.visible_objects_only,
            enable_live_segmentation_view=self.config.live_segmentation_view
        )
        self.translator = SkeletonToPDDL()
        self.executor = VLMExecutor()
        
        # Initialize planner based on config
        if self.config.use_mock_vlm:
            print("[Pipeline] Using MOCK VLM planner")
            self.planner = MockVLMPlanner()
        elif self.config.use_remote_vlm:
            from vlm_pipeline.vlm_client import RemoteVLMPlanner
            url = self.config.remote_vlm_url or os.environ.get("VLM_SERVER_URL", "http://localhost:8000")
            print(f"[Pipeline] Using REMOTE VLM: {url}")
            if not self.config.use_vision:
                print("[Pipeline] Remote planner in TEXT-ONLY mode (vision disabled)")
            self.planner = RemoteVLMPlanner(server_url=url, use_vision=self.config.use_vision)
        else:
            print(f"[Pipeline] Using real VLM: {self.config.vlm_model}")
            self.planner = VLMPlanner(
                model_name=self.config.vlm_model,
                use_4bit=self.config.vlm_4bit
            )
        
        # Environment (loaded later)
        self.env = None
        
        # Pipeline state
        self.state = PipelineState(
            current_plan=[],
            executed_actions=[]
        )
        
        # Logging
        if self.config.save_logs:
            os.makedirs(self.config.log_dir, exist_ok=True)
        
        self.run_log = {
            "start_time": None,
            "config": asdict(self.config),
            "cycles": [],
            "final_result": None
        }
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("VLM PIPELINE INITIALIZATION")
        print("="*60)
        
        # Load VLM
        if not self.planner.load_model():
            print("[Pipeline] ERROR: Failed to load VLM")
            return False
        
        # Load environment if requested
        if self.config.use_real_env:
            print("[Pipeline] Loading RLBench environment...")
            try:
                os.environ["HEADLESS"] = "False" if not self.config.headless else "True"
                
                # IMPORTANT: Use the ENV from rlbench_kitchen_streams to ensure
                # PDDL planning works correctly (streams use this global ENV)
                from rlbench_kitchen_streams import ENV
                self.env = ENV
                
                # Share env with modules
                self.context_aggregator.set_env(self.env)
                self.context_aggregator.configure_visibility(
                    visible_objects_only=self.config.visible_objects_only,
                    enable_live_segmentation_view=self.config.live_segmentation_view
                )
                self.executor.set_env(self.env)

                if self.config.live_segmentation_view and hasattr(self.executor, "set_step_callback"):
                    self.executor.set_step_callback(self.context_aggregator.update_live_segmentation_view)
                
                # Settle physics
                print("[Pipeline] Settling physics...")
                for _ in range(50):
                    self.env.pr.step()
                
                # Go to home configuration
                home_q = self.env.get_home_conf()
                self.env.set_robot_conf(home_q)
                for _ in range(10):
                    self.env.pr.step()

                if self.config.visible_objects_only or self.config.live_segmentation_view:
                    self.context_aggregator.update_visible_objects(event="initial")
                
                print("[Pipeline] Environment ready")
            except Exception as e:
                print(f"[Pipeline] ERROR loading environment: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        self.run_log["start_time"] = datetime.now().isoformat()
        print("[Pipeline] Initialization complete!")
        return True
    
    def plan(self, goal: str) -> PlanResult:
        """
        Generate an action plan for the given goal.
        
        Args:
            goal: Natural language goal description
            
        Returns:
            PlanResult from VLM
        """
        print(f"\n[Pipeline] Planning for goal: {goal}")
        
        # Create prompt bundle
        if self.env:
            bundle = self.context_aggregator.create_prompt_bundle(goal)
        else:
            bundle = self.context_aggregator.create_prompt_bundle_offline(goal)
            
        print("[Pipeline] Prompt bundle created.")
        if not self.config.use_real_env:
             print("[Pipeline] Running in offline mode (dummy images).")
        
        # Query VLM
        if self.config.use_vision:
            result = self.planner.generate_plan(
                image=bundle.composite_image,
                system_prompt=bundle.system_prompt,
                user_prompt=bundle.user_prompt
            )
        else:
            if hasattr(self.planner, "generate_plan_text_only"):
                result = self.planner.generate_plan_text_only(
                    system_prompt=bundle.system_prompt,
                    user_prompt=bundle.user_prompt
                )
            else:
                # Remote planner uses generate_plan() and carries text-only mode internally.
                result = self.planner.generate_plan(
                    image=bundle.composite_image,
                    system_prompt=bundle.system_prompt,
                    user_prompt=bundle.user_prompt
                )
        
        print(f"[Pipeline] VLM inference time: {result.inference_time:.2f}s")
        print(f"[Pipeline] Raw output:\n{result.raw_output}")
        
        if result.success:
            print(f"[Pipeline] Parsed {len(result.skeleton)} actions:")
            for i, a in enumerate(result.skeleton):
                print(f"  {i+1}. {a}")
            
            # Validate
            if hasattr(self.planner, "validate_plan"):
                is_valid, errors = self.planner.validate_plan(result.skeleton)
                if not is_valid:
                    print(f"[Pipeline] WARNING: Plan validation errors:")
                    for e in errors:
                        print(f"  - {e}")
            else:
                print("[Pipeline] Skipping validation (planner has no validate_plan)")
        
        return result
    
    def replan(self, 
               goal: str,
               failure_reason: str,
               executed_actions: List[ActionSkeleton]) -> PlanResult:
        """
        Generate a new plan after failure/interruption.
        
        Args:
            goal: Original goal
            failure_reason: Why replanning is needed
            executed_actions: Actions already completed
            
        Returns:
            New PlanResult
        """
        print(f"\n[Pipeline] REPLANNING (attempt {self.state.replan_count + 1})")
        print(f"[Pipeline] Reason: {failure_reason}")
        print(f"[Pipeline] Completed actions: {[str(a) for a in executed_actions]}")
        
        # Create replan bundle
        if self.env:
            bundle = self.context_aggregator.trigger_replan(
                failure_reason=failure_reason,
                previous_plan=[str(a) for a in executed_actions + self.executor.get_remaining_plan()],
                goal=goal
            )
        else:
            bundle = self.context_aggregator.create_prompt_bundle_offline(goal)
            bundle.is_replan = True
            bundle.failure_context = failure_reason
            bundle.previous_plan = [str(a) for a in executed_actions]
        
        # Query VLM
        result = self.planner.generate_plan(
            image=bundle.composite_image,
            system_prompt=bundle.system_prompt,
            user_prompt=bundle.user_prompt
        )
        
        self.state.replan_count += 1
        
        return result
    
    def execute(self, skeleton: List[ActionSkeleton]) -> ExecutionResult:
        """
        Execute an action skeleton.
        
        Args:
            skeleton: Actions to execute
            
        Returns:
            ExecutionResult
        """
        if not self.env:
            print("[Pipeline] No environment loaded (offline mode). Skipping execution.")
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                completed_actions=0,
                total_actions=len(skeleton),
                trajectory_executed=False
            )

        print(f"\n[Pipeline] Executing plan with {len(skeleton)} actions")
        
        self.state.current_plan = [str(a) for a in skeleton]
        
        result = self.executor.execute_plan(skeleton)
        
        self.state.executed_actions = [str(a) for a in self.executor.get_executed_plan()]
        
        return result
    
    def run(self, goal: str) -> Dict[str, Any]:
        """
        Run the complete pipeline for a goal.
        
        Args:
            goal: Natural language goal
            
        Returns:
            Run summary dictionary
        """
        print("\n" + "="*60)
        print("VLM PIPELINE RUN")
        print("="*60)
        print(f"Goal: {goal}")
        
        self.executor.reset()
        self.state = PipelineState(current_plan=[], executed_actions=[])
        
        # SINGLE RUN - no automatic replanning
        # Planning phase
        plan_result = self.plan(goal)
        
        cycle_log = {
            "cycle": 0,
            "type": "initial",
            "plan_result": {
                "success": plan_result.success,
                "skeleton": [str(a) for a in plan_result.skeleton],
                "inference_time": plan_result.inference_time
            },
            "execution_result": None
        }
        
        if not plan_result.success:
            print("[Pipeline] Planning failed!")
            self.run_log["cycles"].append(cycle_log)
            return self._generate_summary()
        
        # Execution phase
        exec_result = self.execute(plan_result.skeleton)
        
        cycle_log["execution_result"] = {
            "status": exec_result.status.value,
            "completed": exec_result.completed_actions,
            "total": exec_result.total_actions,
            "error": exec_result.error_message
        }
        
        self.run_log["cycles"].append(cycle_log)
        
        if exec_result.status == ExecutionStatus.SUCCESS:
            print("\n[Pipeline] SUCCESS! Goal achieved.")
        else:
            print("\n" + "="*60)
            print("EXECUTION STOPPED - ERROR ENCOUNTERED")
            print("="*60)
            print(f"\n❌ FAILED at action {exec_result.completed_actions + 1}/{exec_result.total_actions}")
            print(f"   Error: {exec_result.error_message}")
            
            # Show completed actions
            executed = self.executor.get_executed_plan()
            if executed:
                print(f"\n✓ COMPLETED ACTIONS ({len(executed)}):")
                for i, action in enumerate(executed, 1):
                    print(f"   {i}. {action}")
            else:
                print("\n✓ COMPLETED ACTIONS: None")
            
            # Show remaining actions
            remaining = self.executor.get_remaining_plan()
            if remaining:
                print(f"\n✗ NOT EXECUTED ({len(remaining)}):")
                for i, action in enumerate(remaining, 1):
                    print(f"   {i}. {action}")
            
            # Show failed action details
            if exec_result.current_action:
                print(f"\n⚠ FAILED ACTION: {exec_result.current_action}")
            
            print("\n" + "="*60)
            print("Replanning DISABLED - execution stopped.")
            print("="*60)
        
        # Summary
        summary = {
            "goal": goal,
            "success": exec_result.status == ExecutionStatus.SUCCESS,
            "total_cycles": 1,
            "final_status": exec_result.status.value,
            "actions_completed": exec_result.completed_actions,
            "actions_total": exec_result.total_actions
        }
        
        self.run_log["final_result"] = summary
        
        # Save log
        if self.config.save_logs:
            self._save_log()
        
        return summary
    
    def interrupt(self, reason: str = "Unknown object detected"):
        """
        Manually interrupt the pipeline (e.g., when unknown object appears).
        
        Args:
            reason: Reason for interruption
        """
        print(f"\n[Pipeline] INTERRUPT TRIGGERED: {reason}")
        self.state.is_interrupted = True
        self.state.interrupt_reason = reason
        self.executor.interrupt(reason)
    
    def add_unknown_object(self, object_name: str):
        """
        Register that an unknown object has been detected.
        
        Args:
            object_name: Name/description of the unknown object
        """
        print(f"[Pipeline] Unknown object registered: {object_name}")
        self.state.unknown_objects.append(object_name)
    
    def _save_log(self):
        """Save run log to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.config.log_dir, f"run_{timestamp}.json")
        
        with open(log_path, 'w') as f:
            json.dump(self.run_log, f, indent=2, default=str)
        
        print(f"[Pipeline] Log saved to: {log_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary when planning fails (before execution)."""
        summary = {
            "goal": self.state.current_plan[0] if self.state.current_plan else "unknown",
            "success": False,
            "total_cycles": 1,
            "final_status": "planning_failed",
            "actions_completed": 0,
            "actions_total": 0,
            "error": "VLM planning failed - no valid plan generated"
        }
        
        self.run_log["final_result"] = summary
        
        # Save log
        if self.config.save_logs:
            self._save_log()
        
        return summary
    
    def shutdown(self):
        """Clean up resources."""
        print("\n[Pipeline] Shutting down...")
        try:
            self.context_aggregator.shutdown()
        except Exception:
            pass
        if self.env:
            try:
                self.env.pr.stop()
                self.env.pr.shutdown()
            except:
                pass
        print("[Pipeline] Shutdown complete")


# ============================================================================
# DEFAULT KITCHEN TASK GOALS
# ============================================================================

DEFAULT_GOALS = {
    "full_task": """Move mug_box from box-top to placement_boundary.
Open the box lid to access mug_inside_box.
Move mug_inside_box to placement_boundary.
Move soup to cupboard_boundary.""",

    "simple_pick_place": "Move mug_box to placement_boundary.",
    
    "open_and_retrieve": """Open the box lid and move mug_inside_box to placement_boundary.""",
    
    "soup_to_cupboard": "Move soup to cupboard_boundary."
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Pipeline for Kitchen Task")
    parser.add_argument("--goal", type=str, default="full_task",
                        help="Goal name or custom goal string")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock VLM (no GPU required)")
    parser.add_argument("--remote", action="store_true",
                        help="Use remote VLM server (set VLM_SERVER_URL env var)")
    parser.add_argument("--remote-url", type=str, default="",
                        help="Remote VLM server URL")
    parser.add_argument("--text-only", action="store_true",
                        help="Disable vision input; use text-only planning")
    parser.add_argument("--visible-only", action="store_true",
                        help="Use segmentation-visible objects only in context state")
    parser.add_argument("--live-masks", action="store_true",
                        help="Show live segmentation mask window with visible object list")
    parser.add_argument("--no-env", action="store_true",
                        help="Run without RLBench environment")
    parser.add_argument("--headless", action="store_true",
                        help="Run simulation headless")
    parser.add_argument("--list-goals", action="store_true",
                        help="List available default goals")
    
    args = parser.parse_args()
    
    if args.list_goals:
        print("\nAvailable default goals:")
        for name, goal in DEFAULT_GOALS.items():
            print(f"\n  {name}:")
            print(f"    {goal[:80]}..." if len(goal) > 80 else f"    {goal}")
        return
    
    # Get goal
    goal = DEFAULT_GOALS.get(args.goal, args.goal)
    
    # Configure pipeline
    config = PipelineConfig(
        use_mock_vlm=args.mock,
        use_remote_vlm=args.remote,
        remote_vlm_url=args.remote_url,
        use_vision=not args.text_only,
        visible_objects_only=args.visible_only,
        live_segmentation_view=args.live_masks,
        use_real_env=not args.no_env,
        headless=args.headless
    )
    
    # Create and run pipeline
    pipeline = VLMPipeline(config)
    
    try:
        if not pipeline.initialize():
            print("Failed to initialize pipeline!")
            return
        
        result = pipeline.run(goal)
        
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        
        # Keep simulation running if not headless
        if not args.headless and pipeline.env:
            print("\nPress Ctrl+C to exit...")
            try:
                while True:
                    pipeline.env.pr.step()
            except KeyboardInterrupt:
                pass
        
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
