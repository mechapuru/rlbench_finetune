import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import select

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Parse arguments FIRST so we can set the globally expected SCENE_FILE
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=int, choices=[0, 1], default=1, 
                    help="0 for grill_task, 1 for kitchen_task (default: kitchen_task)")
parser.add_argument("--port", type=int, default=5555, help="ZMQ Port")
args = parser.parse_args()

scenes = [
    "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/grill_task/grill_task.ttt",
    "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/kitchen_task/task_design_proposal_variation_1.ttt"
]
target_scene = scenes[args.scene]

import rlbench_kitchen_env as env_module
env_module.SCENE_FILE = target_scene

# NOW import streams which will initialize PyRep under the hood (Singleton, exactly ONCE)
from rlbench_kitchen_streams import ENV, RLBenchKitchenEnv
from vlm_pipeline.dynamic_state_extractor import DynamicStateExtractor
from vlm_pipeline.pddl_executor import PDDLExecutor
from vlm_pipeline.llm_planner import ActionSkeleton

import zmq
import select

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RLBenchZMQServer:
    def __init__(self, scene_file: str, port: int):
        self.port = port
        self.scene_file = scene_file
        
        logging.info(f"Using RLBench environment with scene: {self.scene_file}")
        
        # USE THE PRE-INITIALIZED SINGLETON TO PREVENT QT CRASHES
        self.env = ENV
        
        # Settle physics
        logging.info("Settling physics engine...")
        for _ in range(20):
            self.env.pr.step()
            
        # Hook up Extractor and Executor natively
        self.extractor = DynamicStateExtractor(self.env)
        self.executor = PDDLExecutor(self.env)
        
        # Initialize ZMQ Server
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        
        logging.info(f"ZMQ Server listening on port {self.port}...")

    def handle_request(self, message):
        """Parse incoming JSON request and route to appropriate handler."""
        try:
            req = json.loads(message)
            req_type = req.get("type", "UNKNOWN")
            
            if req_type == "PING":
                return {"status": "success", "message": "PONG"}
                
            elif req_type == "GET_STATE":
                # Option C: Extractor runs on the server natively
                goal = req.get("goal", "none")
                bundle = self.extractor.create_prompt_bundle(goal)
                return {
                    "status": "success", 
                    "state_text": bundle.state_text,
                    "scene": self.scene_file
                }
                
            elif req_type == "EXECUTE_ACTION":
                action_dicts = req.get("actions", [])
                
                # Convert back to dataclass
                skeletons = []
                for a in action_dicts:
                    skel = ActionSkeleton(action_name=a['name'], args=a['args'])
                    skeletons.append(skel)
                    
                logging.info(f"Executing plan: {[skel.action_name for skel in skeletons]}")
                
                # Group pick and place actions into pairs for proper trajectory solving
                i = 0
                while i < len(skeletons):
                    skel = skeletons[i]
                    
                    if skel.action_name == 'open-lid' or skel.action_name == 'open_lid':
                        result = self.executor.execute_single_action(skel)
                        i += 1
                        
                    elif skel.action_name == 'pick':
                        # Look ahead for a matching place action
                        if i + 1 < len(skeletons) and skeletons[i+1].action_name == 'place':
                            place_skel = skeletons[i+1]
                            logging.info(f"Executing macro: pick({skel.args}) -> place({place_skel.args})")
                            result = self.executor.execute_pick_place_pair(skel, place_skel)
                            i += 2  # Consume both
                        else:
                            # It's an isolated pick, use single action
                            result = self.executor.execute_single_action(skel)
                            i += 1
                    else:
                        result = self.executor.execute_single_action(skel)
                        i += 1
                        
                    if result.status.value != "success":
                        logging.warning(f"Action failed: {result.error_message}")
                        return {
                            "status": "failed",
                            "failed_action": skel.action_name,
                            "error": result.error_message,
                            "reason": getattr(result, 'failure_reason', None)
                        }
                        
                return {"status": "success", "message": "Plan executed."}
                
            else:
                return {"status": "error", "message": f"Unknown request type {req_type}"}
                
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format"}
        except Exception as e:
            logging.error(f"Error handling request: {e}")
            return {"status": "error", "message": str(e)}

    def run(self):
        """Main Loop: Keep PyRep ticking and check for non-blocking ZMQ messages."""
        try:
            # We must use a poller to do non-blocking reads without burning CPU
            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)
            
            while True:
                # Step the environment constantly to keep physics and GUI responsive
                self.env.pr.step()
                
                # Check for messages (non-blocking, timeout in milliseconds)
                # 10ms timeout lets us run physics at ~100Hz
                socks = dict(poller.poll(10))
                
                if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                    message = self.socket.recv()
                    logging.info(f"Received request: {message.decode('utf-8')[:100]}")
                    
                    # Handle it
                    response_data = self.handle_request(message)
                    
                    # Send response back
                    self.socket.send_string(json.dumps(response_data))
                    
        except KeyboardInterrupt:
            logging.info("Shutting down server...")
        finally:
            self.context.term()

if __name__ == "__main__":
    # Note: args logic moved to top of file to catch before PyRep initialize
    server = RLBenchZMQServer(scene_file=target_scene, port=args.port)
    server.run()
