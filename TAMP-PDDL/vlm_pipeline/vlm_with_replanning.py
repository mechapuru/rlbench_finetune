#!/usr/bin/env python3
"""
VLM Pipeline with Replanning
=============================
Runs VLM planning with automatic replanning on failures.

When execution fails:
1. Captures completed actions
2. Captures error message
3. Re-prompts VLM with failure context
4. Continues with new plan

Run: python -m vlm_pipeline.vlm_with_replanning --goal "your goal here"
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.vlm_planner import VLMPlanner, ActionSkeleton, PlanResult, MockVLMPlanner
from vlm_pipeline.vlm_executor_v2 import VLMExecutorV2, ExecutionResult, ExecutionStatus
from vlm_pipeline.vlm_context_aggregator import VLMContextAggregator, PromptBundle


@dataclass
class ReplanConfig:
    """Configuration for replanning."""
    max_replans: int = 3
    use_mock_vlm: bool = False
    use_remote_vlm: bool = False  # Use remote VLM server
    remote_vlm_url: str = ""  # URL of remote VLM server
    headless: bool = False
    save_logs: bool = True
    log_dir: str = "vlm_replan_logs"


@dataclass
class ReplanCycle:
    """Information about a single plan-execute cycle."""
    cycle_number: int
    is_replan: bool
    plan: List[str]
    completed_actions: List[str]
    failed_action: Optional[str]
    error_message: Optional[str]
    success: bool


class VLMReplanningPipeline:
    """
    VLM Pipeline with replanning capability.
    
    When execution fails, re-prompts the VLM with:
    - Original task description
    - Actions completed so far
    - Error message explaining failure
    - Request for corrected remaining actions
    """
    
    def __init__(self, config: ReplanConfig = None):
        self.config = config or ReplanConfig()
        
        # Initialize components
        self.context_aggregator = VLMContextAggregator()
        self.executor = VLMExecutorV2()
        
        # Initialize planner
        if self.config.use_mock_vlm:
            print("[ReplanPipeline] Using MOCK VLM")
            self.planner = MockVLMPlanner()
        elif self.config.use_remote_vlm:
            from vlm_pipeline.vlm_client import RemoteVLMPlanner
            url = self.config.remote_vlm_url or os.environ.get("VLM_SERVER_URL", "http://localhost:8000")
            print(f"[ReplanPipeline] Using REMOTE VLM: {url}")
            self.planner = RemoteVLMPlanner(server_url=url)
        else:
            print("[ReplanPipeline] Using real VLM: Qwen/Qwen2-VL-2B-Instruct")
            self.planner = VLMPlanner(
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                use_4bit=False
            )
        
        self.env = None
        
        # Tracking
        self.all_completed_actions: List[ActionSkeleton] = []
        self.cycles: List[ReplanCycle] = []
        self.original_goal = ""
        
        # Logging
        if self.config.save_logs:
            os.makedirs(self.config.log_dir, exist_ok=True)
    
    def initialize(self) -> bool:
        """Initialize the pipeline."""
        print("\n" + "=" * 60)
        print("VLM REPLANNING PIPELINE")
        print("=" * 60)
        
        # Load VLM
        if not self.planner.load_model():
            print("[ReplanPipeline] ERROR: Failed to load VLM")
            return False
        
        # Load environment
        print("[ReplanPipeline] Loading environment...")
        try:
            os.environ["HEADLESS"] = "True" if self.config.headless else "False"
            
            from rlbench_kitchen_streams import ENV
            self.env = ENV
            
            self.context_aggregator.set_env(self.env)
            self.executor.set_env(self.env)
            
            # Settle physics
            print("[ReplanPipeline] Settling physics...")
            for _ in range(50):
                self.env.pr.step()
            
            # Go home
            home_q = self.env.get_home_conf()
            self.env.set_robot_conf(home_q)
            for _ in range(10):
                self.env.pr.step()
            
            print("[ReplanPipeline] Environment ready")
            
        except Exception as e:
            print(f"[ReplanPipeline] ERROR loading environment: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("[ReplanPipeline] Initialization complete!")
        return True
    
    def build_replan_prompt(self, 
                            original_goal: str,
                            completed_actions: List[ActionSkeleton],
                            failed_action: ActionSkeleton,
                            error_message: str,
                            remaining_actions: List[ActionSkeleton]) -> str:
        """
        Build a replanning prompt with failure context.
        """
        completed_str = "\n".join([f"  {i+1}. {a}" for i, a in enumerate(completed_actions)]) if completed_actions else "  (none)"
        remaining_str = "\n".join([f"  - {a}" for a in remaining_actions]) if remaining_actions else "  (none)"
        
        prompt = f"""=== REPLANNING REQUIRED ===

ORIGINAL TASK:
{original_goal}

ACTIONS COMPLETED SUCCESSFULLY:
{completed_str}

FAILED ACTION:
  ❌ {failed_action}

ERROR MESSAGE:
  {error_message}

ACTIONS THAT WERE NOT EXECUTED:
{remaining_str}

---

Please provide a CORRECTED action sequence to complete the remaining task.
Consider:
1. The actions already completed (don't repeat them)
2. The error message explains what went wrong
3. You may need to add actions to resolve the blockage/issue

Output ONLY the remaining actions needed, starting from action 1:
1. action(args)
2. action(args)
..."""
        
        return prompt
    
    def plan_initial(self, goal: str) -> PlanResult:
        """Generate initial plan."""
        print(f"\n[ReplanPipeline] Initial planning for: {goal}")
        
        # Capture context
        bundle = self.context_aggregator.create_prompt_bundle(goal)
        
        # Query VLM
        result = self.planner.generate_plan(
            image=bundle.composite_image,
            system_prompt=bundle.system_prompt,
            user_prompt=bundle.user_prompt
        )
        
        print(f"[ReplanPipeline] VLM inference time: {result.inference_time:.2f}s")
        
        if result.success:
            print(f"[ReplanPipeline] Initial plan ({len(result.skeleton)} actions):")
            for i, action in enumerate(result.skeleton, 1):
                print(f"  {i}. {action}")
        else:
            print(f"[ReplanPipeline] Planning failed: {result.error_message}")
        
        return result
    
    def plan_replan(self, 
                    original_goal: str,
                    completed_actions: List[ActionSkeleton],
                    failed_action: ActionSkeleton,
                    error_message: str,
                    remaining_actions: List[ActionSkeleton]) -> PlanResult:
        """Generate a replan after failure."""
        print(f"\n[ReplanPipeline] REPLANNING...")
        print(f"[ReplanPipeline] Completed: {len(completed_actions)} actions")
        print(f"[ReplanPipeline] Failed: {failed_action}")
        print(f"[ReplanPipeline] Error: {error_message}")
        
        # Capture fresh context (scene may have changed)
        bundle = self.context_aggregator.create_prompt_bundle(original_goal)
        
        # Build replan user prompt
        replan_user_prompt = self.build_replan_prompt(
            original_goal=original_goal,
            completed_actions=completed_actions,
            failed_action=failed_action,
            error_message=error_message,
            remaining_actions=remaining_actions
        )
        
        print(f"\n[ReplanPipeline] Replan prompt:\n{replan_user_prompt[:500]}...")
        
        # Query VLM with replan prompt
        result = self.planner.generate_plan(
            image=bundle.composite_image,
            system_prompt=bundle.system_prompt,
            user_prompt=replan_user_prompt
        )
        
        print(f"[ReplanPipeline] VLM inference time: {result.inference_time:.2f}s")
        
        if result.success:
            print(f"[ReplanPipeline] Replan ({len(result.skeleton)} actions):")
            for i, action in enumerate(result.skeleton, 1):
                print(f"  {i}. {action}")
        else:
            print(f"[ReplanPipeline] Replanning failed: {result.error_message}")
        
        return result
    
    def execute_plan(self, skeleton: List[ActionSkeleton]) -> ExecutionResult:
        """Execute a plan."""
        print(f"\n[ReplanPipeline] Executing {len(skeleton)} actions...")
        return self.executor.execute_plan(skeleton)
    
    def run(self, goal: str) -> Dict[str, Any]:
        """
        Run the full pipeline with replanning.
        
        Returns summary of execution.
        """
        print("\n" + "=" * 60)
        print("STARTING EXECUTION WITH REPLANNING")
        print("=" * 60)
        print(f"Goal: {goal}")
        print(f"Max replans: {self.config.max_replans}")
        
        self.original_goal = goal
        self.all_completed_actions = []
        self.cycles = []
        
        # Initial planning
        plan_result = self.plan_initial(goal)
        
        if not plan_result.success:
            return self._generate_summary(success=False, reason="Initial planning failed")
        
        current_plan = plan_result.skeleton
        replan_count = 0
        
        while replan_count <= self.config.max_replans:
            cycle_num = len(self.cycles) + 1
            is_replan = replan_count > 0
            
            print(f"\n{'='*60}")
            print(f"EXECUTION CYCLE {cycle_num} {'(REPLAN)' if is_replan else '(INITIAL)'}")
            print(f"{'='*60}")
            
            # Execute current plan
            exec_result = self.execute_plan(current_plan)
            
            # Get what was executed
            executed_this_cycle = self.executor.get_executed_plan()
            remaining = self.executor.get_remaining_plan()
            
            # Track completed actions globally
            self.all_completed_actions.extend(executed_this_cycle)
            
            # Log this cycle
            cycle = ReplanCycle(
                cycle_number=cycle_num,
                is_replan=is_replan,
                plan=[str(a) for a in current_plan],
                completed_actions=[str(a) for a in executed_this_cycle],
                failed_action=str(exec_result.current_action) if exec_result.current_action else None,
                error_message=exec_result.error_message,
                success=exec_result.status == ExecutionStatus.SUCCESS
            )
            self.cycles.append(cycle)
            
            # Check if successful
            if exec_result.status == ExecutionStatus.SUCCESS:
                print("\n" + "=" * 60)
                print("✓ EXECUTION SUCCESSFUL!")
                print("=" * 60)
                return self._generate_summary(success=True)
            
            # Failed - check if we can replan
            if replan_count >= self.config.max_replans:
                print("\n" + "=" * 60)
                print(f"✗ MAX REPLANS REACHED ({self.config.max_replans})")
                print("=" * 60)
                return self._generate_summary(success=False, reason="Max replans exceeded")
            
            # Attempt replan
            print("\n" + "-" * 60)
            print("INITIATING REPLAN...")
            print("-" * 60)
            
            replan_result = self.plan_replan(
                original_goal=goal,
                completed_actions=self.all_completed_actions,
                failed_action=exec_result.current_action,
                error_message=exec_result.error_message or "Unknown error",
                remaining_actions=remaining
            )
            
            if not replan_result.success or not replan_result.skeleton:
                print("[ReplanPipeline] Replan failed - no valid plan generated")
                return self._generate_summary(success=False, reason="Replan failed")
            
            # Update for next iteration
            current_plan = replan_result.skeleton
            replan_count += 1
            
            # Reset executor for new plan
            self.executor.reset()
        
        return self._generate_summary(success=False, reason="Unexpected exit")
    
    def _generate_summary(self, success: bool, reason: str = None) -> Dict[str, Any]:
        """Generate execution summary."""
        summary = {
            "goal": self.original_goal,
            "success": success,
            "total_cycles": len(self.cycles),
            "total_replans": sum(1 for c in self.cycles if c.is_replan),
            "total_actions_completed": len(self.all_completed_actions),
            "completed_actions": [str(a) for a in self.all_completed_actions],
            "cycles": [asdict(c) for c in self.cycles],
            "failure_reason": reason if not success else None
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Goal: {self.original_goal}")
        print(f"Success: {'✓ YES' if success else '✗ NO'}")
        print(f"Total cycles: {len(self.cycles)}")
        print(f"Replans used: {summary['total_replans']}/{self.config.max_replans}")
        print(f"Actions completed: {len(self.all_completed_actions)}")
        
        if self.all_completed_actions:
            print("\nCompleted actions:")
            for i, a in enumerate(self.all_completed_actions, 1):
                print(f"  {i}. {a}")
        
        if not success and reason:
            print(f"\nFailure reason: {reason}")
        
        print("=" * 60)
        
        # Save log
        if self.config.save_logs:
            self._save_log(summary)
        
        return summary
    
    def _save_log(self, summary: Dict):
        """Save execution log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.config.log_dir, f"replan_{timestamp}.json")
        
        with open(log_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"[ReplanPipeline] Log saved to: {log_path}")
    
    def shutdown(self):
        """Clean up."""
        print("\n[ReplanPipeline] Shutting down...")
        if self.env:
            try:
                self.env.pr.stop()
                self.env.pr.shutdown()
            except:
                pass
        print("[ReplanPipeline] Done")


def main():
    parser = argparse.ArgumentParser(description="VLM Pipeline with Replanning")
    parser.add_argument("--goal", type=str, required=True,
                        help="Task goal in natural language")
    parser.add_argument("--max-replans", type=int, default=3,
                        help="Maximum number of replan attempts (default: 3)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock VLM (no GPU required)")
    parser.add_argument("--remote", action="store_true",
                        help="Use remote VLM server (set VLM_SERVER_URL env var)")
    parser.add_argument("--remote-url", type=str, default="",
                        help="Remote VLM server URL")
    parser.add_argument("--headless", action="store_true",
                        help="Run simulation headless")
    
    args = parser.parse_args()
    
    config = ReplanConfig(
        max_replans=args.max_replans,
        use_mock_vlm=args.mock,
        use_remote_vlm=args.remote,
        remote_vlm_url=args.remote_url,
        headless=args.headless
    )
    
    pipeline = VLMReplanningPipeline(config)
    
    try:
        if not pipeline.initialize():
            print("Failed to initialize pipeline!")
            return
        
        result = pipeline.run(args.goal)
        
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
        
        # Keep simulation open
        print("\nPress Ctrl+C to exit...")
        while True:
            pipeline.env.pr.step()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
