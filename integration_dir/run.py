#!/usr/bin/env python3
"""Main entry point for COAST + RLBench integration.

Usage:
    python run.py --task LongHorizonGrillTask --planner fast_downward
    python run.py --task LongHorizonGrillTask --planner llm --llm-model gpt-4
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for coast import
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add symbolic directory to path to ensure correct package resolution
sys.path.insert(0, str(Path(__file__).parent.parent / "coast/symbolic"))

from config import CoastConfig
from world import RLBenchWorld
from planners import TaskPlanner, FastDownwardPlanner
from planners.llm_planner import LLMPlanner
from streams import get_streams, GEOMETRIC_PREDICATES
from actions import get_actions


def create_planner(config: CoastConfig) -> TaskPlanner:
    """Create task planner based on config."""
    if config.planner == "fast_downward":
        return FastDownwardPlanner(
            fd_path=config.fd_path if os.path.exists(config.fd_path) else None
        )
    elif config.planner == "llm":
        return LLMPlanner(model="gpt-4")
    else:
        raise ValueError(f"Unknown planner: {config.planner}")


def run_coast_planning(
    world: RLBenchWorld,
    config: CoastConfig,
    planner: TaskPlanner
) -> Optional[dict]:
    """Run COAST planning for the task.
    
    Args:
        world: RLBenchWorld instance
        config: CoastConfig instance
        planner: TaskPlanner instance
        
    Returns:
        Plan result dict if successful
    """
    print(f"\n{'='*60}")
    print(f"COAST + RLBench Planning")
    print(f"Task: {config.task_name}")
    print(f"Planner: {planner.name}")
    print(f"{'='*60}\n")
    
    # Get initial state
    stream_state = world.get_stream_state()
    objects = world.get_coast_objects()
    
    print(f"Initial stream state: {stream_state}")
    print(f"Objects: {[o['name'] for o in objects]}")
    
    # Get streams and actions
    streams = get_streams(world)
    actions = get_actions(world)
    
    print(f"Loaded {len(streams)} streams, {len(actions)} actions")
    
    try:
        # Import coast module
        import coast
        from coast.constraint import Constraint
        
        # Run COAST planning
        print("\n[COAST] Starting planning...")
        start_time = time.time()
        
        plan = coast.plan(
            domain_pddl=config.domain_pddl,
            problem_pddl=config.problem_pddl,
            streams_pddl=config.streams_pddl,
            streams_py=str(Path(__file__).parent / "streams.py"),
            algorithm=config.algorithm,
            max_level=config.max_level,
            search_sample_ratio=config.search_sample_ratio,
            timeout=config.planner_timeout,
            experiment=1,
            stream_state=stream_state,
            objects=[coast.Object(o['name'], o['type'], o.get('value')) for o in objects],
            random_seed=0,
            world=world,
            constraint_cls=Constraint,
            use_cache=False
        )
        
        planning_time = time.time() - start_time
        print(f"\n[COAST] Planning completed in {planning_time:.2f}s")
        
        if plan.action_plan is not None:
            print(f"\n[COAST] Plan found with {len(plan.action_plan)} actions:")
            for i, action in enumerate(plan.action_plan):
                print(f"  {i+1}. {action}")
            
            return {
                'success': True,
                'plan': plan,
                'planning_time': planning_time
            }
        else:
            print("\n[COAST] No plan found")
            return {'success': False, 'error': 'No plan found'}
            
    except ImportError as e:
        print(f"\n[ERROR] Could not import coast module: {e}")
        print("Make sure COAST is installed: pip install -e ../coast")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        print(f"\n[ERROR] Planning failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def execute_plan(world: RLBenchWorld, plan) -> bool:
    """Execute the plan in simulation.
    
    Args:
        world: RLBenchWorld instance
        plan: COAST plan object
        
    Returns:
        True if execution successful
    """
    print(f"\n{'='*60}")
    print("Executing Plan")
    print(f"{'='*60}\n")
    
    if plan.action_plan is None or plan.objects is None:
        print("[ERROR] No plan to execute")
        return False
    
    for i, action in enumerate(plan.action_plan):
        print(f"\n[Execute] Step {i+1}: {action}")
        
        try:
            success = action.execute(plan.objects)
            if not success:
                print(f"[ERROR] Action {action} failed")
                return False
            
            # Let simulation settle
            for _ in range(5):
                world.step()
                
        except Exception as e:
            print(f"[ERROR] Execution error: {e}")
            return False
    
    print("\n[Execute] Plan execution complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="COAST + RLBench Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Task options
    parser.add_argument(
        "--task", "-t",
        default="LongHorizonGrillTask",
        help="RLBench task name"
    )
    
    # Planner options
    parser.add_argument(
        "--planner", "-p",
        choices=["fast_downward", "llm"],
        default="fast_downward",
        help="Task planner to use"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4",
        help="LLM model for LLM planner"
    )
    parser.add_argument(
        "--fd-path",
        default=None,
        help="Path to Fast Downward"
    )
    
    # Algorithm options
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Planning timeout in seconds"
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=6,
        help="COAST max level"
    )
    
    # Execution options
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute plan after planning"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = CoastConfig(
        task_name=args.task,
        planner=args.planner,
        planner_timeout=args.timeout,
        max_level=args.max_level,
        headless=args.headless
    )
    
    if args.fd_path:
        config.fd_path = args.fd_path
    
    print(f"\nConfiguration:")
    print(f"  Task: {config.task_name}")
    print(f"  Planner: {config.planner}")
    print(f"  Timeout: {config.planner_timeout}s")
    print(f"  Headless: {config.headless}")
    
    # Initialize world
    print("\n[Init] Loading RLBench environment...")
    try:
        world = RLBenchWorld.from_task_name(
            config.task_name,
            headless=config.headless
        )
        print(f"[Init] Loaded task: {config.task_name}")
        print(f"[Init] Objects: {list(world.objects.keys())}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize world: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        # Create planner
        planner = create_planner(config)
        
        # Run planning
        result = run_coast_planning(world, config, planner)
        
        if result and result.get('success') and args.execute:
            # Execute the plan
            success = execute_plan(world, result['plan'])
            
            if success:
                print("\n[SUCCESS] Task completed!")
                input("Press Enter to close...")
            else:
                print("\n[FAILED] Task execution failed")
                
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        # Cleanup
        print("\n[Cleanup] Shutting down environment...")
        world.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
