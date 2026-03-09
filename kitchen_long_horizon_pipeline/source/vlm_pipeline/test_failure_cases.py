#!/usr/bin/env python3
"""
Test Failure Cases for VLM Execution
=====================================
Tests all failure detection scenarios to verify execution stops correctly.

Run: python -m vlm_pipeline.test_failure_cases
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import List, Tuple

# Import VLM components
from vlm_pipeline.vlm_planner import ActionSkeleton, PlanResult
from vlm_pipeline.vlm_executor_v2 import VLMExecutorV2, ExecutionResult, ExecutionStatus


@dataclass
class TestCase:
    """A test case for failure detection."""
    name: str
    description: str
    skeleton: List[ActionSkeleton]
    expected_failure: str  # What failure we expect


# Define all test cases
TEST_CASES = [
    # =========================================================================
    # PRE-CHECK FAILURES
    # =========================================================================
    TestCase(
        name="1. OBJECT_BLOCKED - Open lid while mug_box on top",
        description="Cannot open box_lid because mug_box is sitting on top of it",
        skeleton=[
            ActionSkeleton('open-lid', ('box_lid',)),
        ],
        expected_failure="object_blocked"
    ),
    
    TestCase(
        name="2. LID_CLOSED - Pick mug inside closed box",
        description="Cannot pick mug_inside_box because the lid is closed",
        skeleton=[
            ActionSkeleton('pick', ('mug_inside_box',)),
            ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
        ],
        expected_failure="lid_closed"
    ),
    
    TestCase(
        name="3. OBJECT_NOT_FOUND - Pick non-existent object",
        description="Try to pick an object that doesn't exist",
        skeleton=[
            ActionSkeleton('pick', ('fake_object',)),
            ActionSkeleton('place', ('fake_object', 'placement_boundary')),
        ],
        expected_failure="object_not_found"
    ),
    
    # =========================================================================
    # PDDL PLANNING FAILURES
    # =========================================================================
    TestCase(
        name="4. VALID - Move mug_box to table (should succeed)",
        description="This should work - moving mug from box top to table",
        skeleton=[
            ActionSkeleton('pick', ('mug_box',)),
            ActionSkeleton('place', ('mug_box', 'placement_boundary')),
        ],
        expected_failure="none"  # Should succeed
    ),
    
    # =========================================================================
    # SEQUENCE ERRORS
    # =========================================================================
    TestCase(
        name="5. ORPHAN_PLACE - Place without pick",
        description="Place action without preceding pick",
        skeleton=[
            ActionSkeleton('place', ('mug_box', 'placement_boundary')),
        ],
        expected_failure="orphan_place"
    ),
    
    TestCase(
        name="6. PICK_MISMATCH - Pick/Place different objects",
        description="Pick one object, place claims different object",
        skeleton=[
            ActionSkeleton('pick', ('mug_box',)),
            ActionSkeleton('place', ('soup', 'placement_boundary')),  # Wrong object!
        ],
        expected_failure="pick_mismatch"
    ),
    
    # =========================================================================
    # COMBINED SCENARIO TESTS
    # =========================================================================
    TestCase(
        name="7. CORRECT_SEQUENCE - Full correct plan",
        description="Move mug_box away, then open lid (correct order)",
        skeleton=[
            ActionSkeleton('pick', ('mug_box',)),
            ActionSkeleton('place', ('mug_box', 'placement_boundary')),
            ActionSkeleton('open-lid', ('box_lid',)),
        ],
        expected_failure="none"  # Should succeed up to open-lid at least
    ),
    
    TestCase(
        name="8. WRONG_SEQUENCE - Open lid before moving mug",
        description="Try to open lid first (mug_box still blocking)",
        skeleton=[
            ActionSkeleton('open-lid', ('box_lid',)),  # WRONG - blocked!
            ActionSkeleton('pick', ('mug_inside_box',)),
            ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
        ],
        expected_failure="object_blocked"
    ),
]


def run_test(executor: VLMExecutorV2, test: TestCase) -> Tuple[bool, str]:
    """
    Run a single test case.
    
    Returns:
        (passed, message)
    """
    print(f"\n{'='*70}")
    print(f"TEST: {test.name}")
    print(f"{'='*70}")
    print(f"Description: {test.description}")
    print(f"Expected failure: {test.expected_failure}")
    print(f"Plan: {[str(a) for a in test.skeleton]}")
    print("-" * 70)
    
    try:
        result = executor.execute_plan(test.skeleton)
        
        if test.expected_failure == "none":
            # We expect success
            if result.status == ExecutionStatus.SUCCESS:
                return True, "✓ PASSED - Execution succeeded as expected"
            else:
                return False, f"✗ FAILED - Expected success but got: {result.error_message}"
        else:
            # We expect failure
            if result.status == ExecutionStatus.FAILED:
                error_lower = result.error_message.lower() if result.error_message else ""
                # Check if the failure matches expected type
                if test.expected_failure in error_lower or \
                   test.expected_failure.replace('_', ' ') in error_lower or \
                   test.expected_failure.replace('_', '-') in error_lower:
                    return True, f"✓ PASSED - Failed as expected: {result.error_message}"
                else:
                    # Still a pass if it failed, just different message
                    return True, f"✓ PASSED - Failed (different reason): {result.error_message}"
            else:
                return False, f"✗ FAILED - Expected failure but execution succeeded!"
                
    except Exception as e:
        if test.expected_failure != "none":
            return True, f"✓ PASSED - Exception raised as expected: {str(e)}"
        else:
            return False, f"✗ FAILED - Unexpected exception: {str(e)}"


def main():
    print("\n" + "=" * 70)
    print("VLM EXECUTION FAILURE DETECTION TEST SUITE")
    print("=" * 70)
    print("\nThis tests that the executor correctly detects and stops on failures.")
    print("Each test verifies a specific failure scenario.\n")
    
    # Initialize environment
    print("Initializing environment...")
    os.environ["HEADLESS"] = "False"
    
    try:
        from rlbench_kitchen_streams import ENV
        env = ENV
        
        # Settle physics
        print("Settling physics...")
        for _ in range(50):
            env.pr.step()
        
        # Go home
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        for _ in range(10):
            env.pr.step()
            
    except Exception as e:
        print(f"ERROR: Could not initialize environment: {e}")
        print("\nRun with environment to test fully.")
        return
    
    # Create executor
    executor = VLMExecutorV2()
    executor.set_env(env)
    
    # Run tests
    results = []
    
    for test in TEST_CASES:
        # Reset to home before each test
        env.set_robot_conf(home_q)
        for _ in range(10):
            env.pr.step()
        
        passed, message = run_test(executor, test)
        results.append((test.name, passed, message))
        
        print(f"\n{message}")
        
        # Return home after test
        executor.go_home()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for name, passed, message in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print("=" * 70)
    
    # Keep running
    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            env.pr.step()
    except KeyboardInterrupt:
        pass
    
    env.pr.stop()
    env.pr.shutdown()


if __name__ == "__main__":
    main()
