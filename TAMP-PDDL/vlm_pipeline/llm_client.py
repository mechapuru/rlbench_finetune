import os
import sys
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.llm_planner import ActionSkeleton, PlanResult


class RemoteLLMPlanner:
    """
    LLM Planner that connects to remote server for inference without images.
    """
    
    def __init__(self, server_url: str = None):
        self.server_url = server_url or os.environ.get("LLM_SERVER_URL", os.environ.get("VLM_SERVER_URL", "http://localhost:8000"))
        self.loaded = False
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install with: pip install requests")
    
    def load_model(self) -> bool:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                self.loaded = response.json().get("model_loaded", False)
                return self.loaded
            return False
        except Exception as e:
            print(f"[RemoteLLM] Error checking server: {e}")
            return False
    
    def generate_plan(self,
                      system_prompt: str,
                      user_prompt: str,
                      max_new_tokens: int = 1024,
                      temperature: float = 0.1) -> PlanResult:
        start_time = time.time()
        
        try:
            request_data = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature
            }
            
            response = requests.post(f"{self.server_url}/plan_text", json=request_data, timeout=120)
            
            if response.status_code != 200:
                # Fallback to the old /plan endpoint which might expect image_base64
                if response.status_code == 404:
                    request_data["image_base64"] = ""  # Empty image for VLM compatibility
                    response = requests.post(f"{self.server_url}/plan", json=request_data, timeout=120)
                
                if response.status_code != 200:
                    return PlanResult(False, [], "", time.time() - start_time, f"Server error: {response.text}")
            
            data = response.json()
            skeleton = [
                ActionSkeleton(action_name=a["action_name"], args=tuple(a["args"]))
                for a in data.get("actions", [])
            ]
            
            return PlanResult(
                success=data.get("success", False),
                skeleton=skeleton,
                raw_output=data.get("raw_output", ""),
                inference_time=data.get("inference_time", time.time() - start_time)
            )
            
        except Exception as e:
            return PlanResult(False, [], "", time.time() - start_time, str(e))
