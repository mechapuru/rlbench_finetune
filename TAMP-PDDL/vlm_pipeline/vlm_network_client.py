#!/usr/bin/env python3
"""
ZMQ Network Client for LLM-Guided TAMP Pipeline
===============================================
Runs the LLM Planner locally and queries the physics server over ZMQ.
Completely decouples the heavy PyTorch LLM from the CoppeliaSim/Qt thread.
"""

import os
import sys
import argparse
import time
import json
import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.inference_client import InferenceClient
import re
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ActionSkeleton:
    action_name: str
    args: Tuple[str, ...]

class RemoteLLMWrapper:
    def __init__(self, server_url: str):
        self.client = InferenceClient(server_url)
        
    def load_model(self) -> bool:
        try:
            print(f"[RemoteLLMWrapper] Pinging Inference Server at {self.client.server_url}")
            health = self.client.health()
            print(f"[RemoteLLMWrapper] Server health response: {health}")
            return health.get("status") in ["ok", "healthy"]
        except Exception as e:
            print(f"[RemoteLLMWrapper] ERROR connecting to inference server at {self.client.server_url}: {type(e).__name__} - {str(e)}")
            return False
            
    def generate_plan(self, system_prompt: str, user_prompt: str):
        class PlanResult:
            def __init__(self, actions, raw):
                self.success = len(actions) > 0
                self.skeleton = actions
                self.raw_output = raw
                
        try:
            # We explicitly ask for "qwen" but the inference server usually defaults if not provided
            res = self.client.planning(prompt=user_prompt, system_prompt=system_prompt)
            
            actions = []
            pattern = r'(pick|place|open-lid|open_lid)\s*\(\s*([^)]+)\s*\)'
            matches = re.findall(pattern, res.output, re.IGNORECASE)
            
            for action_name, args_str in matches:
                action_name = action_name.lower().replace('_', '-')
                args = [a.strip() for a in args_str.split(',')]
                actions.append(ActionSkeleton(action_name, tuple(args)))
                
            return PlanResult(actions, res.output)
        except Exception as e:
            return PlanResult([], str(e))

class NetworkLLMPipeline:
    def __init__(self, port: int = 5555, max_retries: int = 4, llm_url: str = "http://localhost:8000"):
        self.port = port
        self.max_retries = max_retries
        self.planner = RemoteLLMWrapper(llm_url)
        
        # Setup ZMQ Context
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        print(f"Connecting to RLBench ZMQ Server at localhost:{self.port}...")
        self.socket.connect(f"tcp://localhost:{self.port}")
        
    def initialize(self) -> bool:
        print("\n" + "="*60)
        print("NETWORKED LLM TAMP PIPELINE INITIALIZATION")
        print("="*60)
        
        # 1. Test Server Connection
        try:
            print(f"[Pipeline] Pinging ZMQ Server...")
            self.socket.send_string(json.dumps({"type": "PING"}))
            reply = self.socket.recv_string()
            data = json.loads(reply)
            if data.get("status") == "success":
                print("[Pipeline] Connected to Server successfully!")
            else:
                print("[Pipeline] Failed to ping server.")
                return False
        except Exception as e:
            print(f"[Pipeline] ERROR connecting to server: {e}")
            return False

        # 2. Ping Remote LLM Planner
        if not self.planner.load_model():
            print("[Pipeline] Failed to connect to Remote LLM Inference Server.")
            return False
            
        print("[Pipeline] Initialization complete!")
        return True

    def get_state_from_server(self, goal: str) -> str:
        req = {
            "type": "GET_STATE",
            "goal": goal
        }
        self.socket.send_string(json.dumps(req))
        reply = self.socket.recv_string()
        data = json.loads(reply)
        if data.get("status") == "success":
            return data.get("state_text", "")
        else:
            print(f"[ERROR] Failed to get state: {data.get('message', 'Unknown')}")
            return "ERROR_EXTRACTING_STATE"

    def execute_plan_on_server(self, skeleton_list) -> dict:
        actions = []
        for skel in skeleton_list:
            actions.append({
                "name": skel.action_name,
                "args": skel.args
            })
            
        req = {
            "type": "EXECUTE_ACTION",
            "actions": actions
        }
        
        self.socket.send_string(json.dumps(req))
        # This will block until the server finishes the entire plan execution!
        reply = self.socket.recv_string()
        return json.loads(reply)

    def run(self, goal: str) -> bool:
        print("\n" + "="*60)
        print(f"Goal: {goal}")
        print("="*60)
        
        failure_context = ""
        past_plans = []
        
        # Create debugging log directory
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debugging_log")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[Pipeline] Saving debug logs to: {debug_dir}")
        
        # Main Replanning Loop
        for attempt in range(1, self.max_retries + 1):
            print(f"\n--- ATTEMPT {attempt}/{self.max_retries} ---")
            
            # 1. State Extraction (Network)
            print("[1. State Extraction] Requesting physical scene state from server...")
            state_description = self.get_state_from_server(goal)
            if state_description == "ERROR_EXTRACTING_STATE":
                return False
                
            print("Extracted State:\n", state_description)
            with open(os.path.join(debug_dir, f"attempt_{attempt}_01_state.txt"), "w") as f:
                f.write(state_description)
            
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
                
            with open(os.path.join(debug_dir, f"attempt_{attempt}_02_prompt.txt"), "w") as f:
                f.write("=== SYSTEM PROMPT ===\n" + system_prompt + "\n\n=== USER PROMPT ===\n" + user_prompt)
                
            # 3. LLM Planning (Local inference)
            print("[2. LLM Planning] Querying LLM...")
            plan_result = self.planner.generate_plan(system_prompt, user_prompt)
            print(f"[LLM Output]\n{plan_result.raw_output}")
            
            with open(os.path.join(debug_dir, f"attempt_{attempt}_03_llm_raw_response.txt"), "w") as f:
                f.write(plan_result.raw_output)
            
            if not plan_result.success or not plan_result.skeleton:
                print("Failed to generate a valid plan.")
                failure_context = "LLM failed to output a parseable plan."
                with open(os.path.join(debug_dir, f"attempt_{attempt}_04_parsed_plan.txt"), "w") as f:
                    f.write("FAILED TO PARSE PLAN\n")
                continue
                
            parsed_str = "\n".join([str(a) for a in plan_result.skeleton])
            with open(os.path.join(debug_dir, f"attempt_{attempt}_04_parsed_plan.txt"), "w") as f:
                f.write(parsed_str)
                
            past_plans.append([str(a) for a in plan_result.skeleton])
            
            # 4. Execution (Network)
            print("[3. Execution] Handing to PDDL Action Executor on Server...")
            exec_result = self.execute_plan_on_server(plan_result.skeleton)
            
            with open(os.path.join(debug_dir, f"attempt_{attempt}_05_execution_result.txt"), "w") as f:
                f.write(json.dumps(exec_result, indent=2))
            
            # 5. Monitor & Loop back
            if exec_result.get("status") == "success":
                print("\n" + "="*60)
                print("✓ GOAL ACHIEVED SUCCESSFULLY")
                print("="*60)
                return True
            else:
                print("\n" + "="*60)
                print(f"✗ EXECUTION FAILED: {exec_result.get('error')}")
                print("="*60)
                
                # Extract semantic reasoning for the failure
                # Since the Executor is on the Server, the Server returns the semantic failure reason
                failure_reason = exec_result.get("reason")
                action_failed = exec_result.get("failed_action", "unknown")
                
                if failure_reason:
                    failure_context = failure_reason
                else:
                    failure_context = f"Failed at action: {action_failed}. Error: {exec_result.get('error')}"
                    
                print(f"Feedback to LLM:\n{failure_context}")
                
        print("\n" + "="*60)
        print("✗ MAXIMUM RETRIES REACHED. PIPELINE FAILED.")
        print("="*60)
        return False

def main():
    parser = argparse.ArgumentParser(description="Networked LLM Physics-Aware TAMP Pipeline")
    parser.add_argument("--goal", type=str, default="Move the mug from the table to the cupboard.",
                        help="Goal instruction for the robot.")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ Port of RLBench Server")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8000", help="URL of Inference Server")
    args = parser.parse_args()

    pipeline = NetworkLLMPipeline(port=args.port, llm_url=args.llm_url)
    
    if not pipeline.initialize():
        return
        
    try:
        pipeline.run(args.goal)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

if __name__ == "__main__":
    main()
