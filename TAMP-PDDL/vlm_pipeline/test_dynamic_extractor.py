#!/usr/bin/env python3
"""
Test Dynamic State Extractor and Failure Recovery
=================================================
Executes hardcoded, deliberate failure sequences to test if the 
DynamicStateExtractor and ExecutionMonitor properly catch physical 
blocks and generate useful semantic failure contexts for the LLM.
"""

import os
import sys
import time

# Configure Qt for GUI BEFORE imports
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.dynamic_state_extractor import DynamicStateExtractor
from vlm_pipeline.pddl_executor import PDDLExecutor, ExecutionStatus
from vlm_pipeline.llm_planner import ActionSkeleton

def run_test():
    print("\n" + "="*60)
    print("TESTING DYNAMIC EXTRACTOR AND RECOVERY (NO LLM)")
    print("="*60)
    
    # 1. Initialize Environment
    try:
        from rlbench_kitchen_streams import ENV
        env = ENV
        for _ in range(50):
            env.pr.step()
        env.set_robot_conf(env.get_home_conf())
        for _ in range(10):
            env.pr.step()
    except Exception as e:
        print(f"ERROR: Could not initialize environment: {e}")
        return
        
    executor = PDDLExecutor()
    executor.set_env(env)
    extractor = DynamicStateExtractor(env)
    
    # 2. Extract Initial State
    print("\n--- INITIAL STATE EXTRACTION ---")
    state = extractor.create_prompt_bundle("none").state_text
    print(state)
    
    # 3. Deliberate Failure Scenario
    # We will try to open the box_lid while mug_box is deliberately sitting on top of it
    print("\n--- EXECUTING BAD PLAN (EXPECTED TO FAIL) ---")
    bad_plan = [
        ActionSkeleton('open-lid', ('box_lid',))
    ]
    
    print(f"Attempting to execute: {bad_plan[0]}")
    result = executor.execute_plan(bad_plan)
    
    # 4. Analyze Failure and Extract Context
    print("\n--- FAILURE RECOVERY TRIGGERED ---")
    if result.status != ExecutionStatus.SUCCESS:
        print(f"Executor correctly failed: {result.error_message}")
    
        if executor.monitor:
            print("\nGenerating LLM Semantic Failure Context...")
            llm_context = executor.monitor.get_llm_failure_context()
            print("\n" + "*"*60)
            print("CONTEXT THAT WOULD BE SENT TO LLM:")
            print("*"*60)
            print(llm_context)
            print("*"*60)
        else:
            print("ExecutionMonitor not attached to Executor!")
    else:
        print("Test failed! The executor was supposed to fail because the lid is blocked.")
        
    print("\nTest Complete! Press Ctrl+C to exit simulation.")
    try:
        while True:
            env.pr.step()
    except KeyboardInterrupt:
        pass
    finally:
        env.pr.stop()
        env.pr.shutdown()


if __name__ == "__main__":
    run_test()
