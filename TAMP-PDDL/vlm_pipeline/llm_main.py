#!/usr/bin/env python3
"""
LLM-Guided Physics-Aware TAMP Pipeline
======================================
Main closed-loop pipeline for semantic planning and execution.
1. Extracts dynamic state via Inverse Kinematics and physics
2. Generates purely semantic plans using an LLM
3. Executes plans using PDDL action implementations
4. Monitors failures and feeds structured text feedback back to the LLM for replanning
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import List, Dict, Any

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

# Import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.dynamic_state_extractor import DynamicStateExtractor
from vlm_pipeline.llm_planner import LLMPlanner, PlanResult
from vlm_pipeline.llm_client import RemoteLLMPlanner
from vlm_pipeline.pddl_executor import PDDLExecutor, ExecutionStatus


class LLMPipeline:
    """
    Main pipeline orchestrating the TAMP loop.
    """
    def __init__(self, use_remote: bool = False, max_retries: int = 4):
        self.use_remote = use_remote
        self.max_retries = max_retries
        self.env = None
        self.extractor = None
        self.planner = None
        self.executor = None
        
    def initialize(self) -> bool:
        print("\n" + "="*60)
        print("LLM TAMP PIPELINE INITIALIZATION")
        print("="*60)
        
        # Load environment via shared streams config
        try:
            from rlbench_kitchen_streams import ENV
            self.env = ENV
            print("[Pipeline] Settling physics...")
            for _ in range(50):
                self.env.pr.step()
            
            # Go to home configuration
            self.env.set_robot_conf(self.env.get_home_conf())
            for _ in range(10):
                self.env.pr.step()
        except Exception as e:
            print(f"[Pipeline] ERROR loading environment: {e}")
            return False

        # Initialize Components
        self.extractor = DynamicStateExtractor(self.env)
        if self.use_remote:
            self.planner = RemoteLLMPlanner()
        else:
            self.planner = LLMPlanner()
            self.planner.load_model()
            
        self.executor = PDDLExecutor()
        self.executor.set_env(self.env)
        
        print("[Pipeline] Initialization complete!")
        return True

    def run(self, goal: str) -> bool:
        print("\n" + "="*60)
        print(f"Goal: {goal}")
        print("="*60)
        
        failure_context = ""
        past_plans = []
        
        # Main Replanning Loop
        for attempt in range(1, self.max_retries + 1):
            print(f"\n--- ATTEMPT {attempt}/{self.max_retries} ---")
            
            # 1. State Extraction
            print("[1. State Extraction] Analyzing physical scene via PyRep...")
            state_description = self.extractor.get_full_scene_description()
            print("Extracted State:\n", state_description)
            
            # 2. Prompt Construction
            system_prompt = (
                "You are an expert robot task planner. You generate action skeletons to reach a goal.\n"
                "Allowed actions are: pick(object), place(object, region), open-lid(object).\n"
                "Only output the sequence of actions as a numbered list. Example:\n"
                "1. pick(mug_box)\n"
                "2. place(mug_box, placement_boundary)\n"
                "3. open-lid(box_lid)"
            )
            
            user_prompt = f"GOAL: {goal}\n\nCURRENT PHYSICAL STATE:\n{state_description}\n"
            
            if failure_context:
                user_prompt += f"\nPREVIOUS FAILURE:\n{failure_context}\n"
                user_prompt += f"PAST PLANS ATTEMPTED: {past_plans}\n"
                user_prompt += "Please generate a NEW plan that resolves this failure by changing the physical state."
                
            # 3. LLM Planning
            print("[2. LLM Planning] Querying LLM...")
            plan_result = self.planner.generate_plan(system_prompt, user_prompt)
            print(f"[LLM Output]\n{plan_result.raw_output}")
            
            if not plan_result.success or not plan_result.skeleton:
                print("Failed to generate a valid plan.")
                failure_context = "LLM failed to output a parseable plan."
                continue
                
            past_plans.append([str(a) for a in plan_result.skeleton])
            
            # 4. Execution
            print("[3. Execution] Handing to PDDL Action Executor...")
            exec_result = self.executor.execute_plan(plan_result.skeleton)
            
            # 5. Monitor & Loop back
            if exec_result.status == ExecutionStatus.SUCCESS:
                print("\n" + "="*60)
                print("✓ GOAL ACHIEVED SUCCESSFULLY")
                print("="*60)
                return True
            else:
                print("\n" + "="*60)
                print(f"✗ EXECUTION FAILED: {exec_result.error_message}")
                print("="*60)
                
                # Extract semantic reasoning for the failure
                if self.executor.monitor:
                    failure_context = self.executor.monitor.get_llm_failure_context()
                else:
                    failure_context = f"Failed at action: {exec_result.current_action}. Reason: {exec_result.error_message}"
                    
                print(f"Feedback to LLM:\n{failure_context}")
                
        print("\n" + "="*60)
        print("✗ MAXIMUM RETRIES REACHED. PIPELINE FAILED.")
        print("="*60)
        return False

    def shutdown(self):
        if self.env:
            try:
                self.env.pr.stop()
                self.env.pr.shutdown()
            except: pass


def main():
    parser = argparse.ArgumentParser(description="LLM Physics-Aware TAMP Pipeline")
    parser.add_argument("--goal", type=str, default="Move the mug from the table to the cupboard.",
                        help="Goal instruction for the robot.")
    parser.add_argument("--remote", action="store_true",
                        help="Use remote LLM server (set LLM_SERVER_URL)")
    args = parser.parse_args()

    pipeline = LLMPipeline(use_remote=args.remote)
    
    if not pipeline.initialize():
        return
        
    try:
        pipeline.run(args.goal)
        print("\nPress Ctrl+C to exit...")
        while True:
            pipeline.env.pr.step()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.shutdown()

if __name__ == "__main__":
    main()
