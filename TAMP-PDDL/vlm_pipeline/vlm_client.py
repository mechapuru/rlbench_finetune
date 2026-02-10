#!/usr/bin/env python3
"""
VLM Remote Client
=================
Connects to remote VLM server for inference.

Usage:
    export VLM_SERVER_URL="http://localhost:8000"  # or remote server
    
    from vlm_pipeline.vlm_client import RemoteVLMPlanner
    planner = RemoteVLMPlanner()
    result = planner.generate_plan(image, system_prompt, user_prompt)
"""

import os
import sys
import time
import base64
import numpy as np
from io import BytesIO
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. Run: pip install requests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.vlm_planner import ActionSkeleton, PlanResult


class RemoteVLMPlanner:
    """
    VLM Planner that connects to remote server for inference.
    Drop-in replacement for VLMPlanner.
    """
    
    def __init__(self, server_url: str = None):
        """
        Initialize remote planner.
        
        Args:
            server_url: URL of VLM server. Defaults to VLM_SERVER_URL env var.
        """
        self.server_url = server_url or os.environ.get("VLM_SERVER_URL", "http://localhost:8000")
        self.loaded = False
        
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install with: pip install requests")
        
        print(f"[RemoteVLM] Server URL: {self.server_url}")
    
    def load_model(self) -> bool:
        """Check if remote server is ready."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.loaded = data.get("model_loaded", False)
                print(f"[RemoteVLM] Server status: {data.get('status')}")
                print(f"[RemoteVLM] Model: {data.get('model_name')}")
                print(f"[RemoteVLM] GPU available: {data.get('gpu_available')}")
                return self.loaded
            else:
                print(f"[RemoteVLM] Server returned {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"[RemoteVLM] Cannot connect to {self.server_url}")
            print("[RemoteVLM] Is the server running?")
            return False
        except Exception as e:
            print(f"[RemoteVLM] Error checking server: {e}")
            return False
    
    def generate_plan(self,
                      image: np.ndarray,
                      system_prompt: str,
                      user_prompt: str,
                      max_new_tokens: int = 1024,
                      temperature: float = 0.1) -> PlanResult:
        """
        Generate plan by calling remote server.
        
        Args:
            image: Composite image (numpy array)
            system_prompt: System prompt for VLM
            user_prompt: User prompt with state and goal
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            PlanResult with action skeleton
        """
        start_time = time.time()
        
        try:
            # Encode image as base64
            buffer = BytesIO()
            np.save(buffer, image, allow_pickle=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Build request
            request_data = {
                "image_base64": image_base64,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "goal": "",  # Extracted from user_prompt on server
                "max_new_tokens": max_new_tokens,
                "temperature": temperature
            }
            
            print(f"[RemoteVLM] Sending request to {self.server_url}/plan...")
            
            # Call server
            response = requests.post(
                f"{self.server_url}/plan",
                json=request_data,
                timeout=120  # VLM inference can be slow
            )
            
            if response.status_code != 200:
                return PlanResult(
                    success=False,
                    skeleton=[],
                    raw_output="",
                    inference_time=time.time() - start_time,
                    error_message=f"Server error: {response.status_code} - {response.text}"
                )
            
            # Parse response
            data = response.json()
            
            # Convert actions back to ActionSkeleton
            skeleton = [
                ActionSkeleton(action_name=a["action_name"], args=tuple(a["args"]))
                for a in data.get("actions", [])
            ]
            
            print(f"[RemoteVLM] Received {len(skeleton)} actions")
            print(f"[RemoteVLM] Raw output: {data.get('raw_output', '')[:200]}...")
            
            return PlanResult(
                success=data.get("success", False),
                skeleton=skeleton,
                raw_output=data.get("raw_output", ""),
                inference_time=data.get("inference_time", time.time() - start_time),
                error_message=data.get("error_message")
            )
            
        except requests.exceptions.Timeout:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=time.time() - start_time,
                error_message="Request timeout - VLM inference took too long"
            )
        except requests.exceptions.ConnectionError:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=time.time() - start_time,
                error_message=f"Cannot connect to server at {self.server_url}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=time.time() - start_time,
                error_message=str(e)
            )


def test_connection(server_url: str = None):
    """Test connection to VLM server."""
    url = server_url or os.environ.get("VLM_SERVER_URL", "http://localhost:8000")
    
    print("=" * 60)
    print("TESTING VLM SERVER CONNECTION")
    print("=" * 60)
    print(f"Server URL: {url}")
    
    try:
        response = requests.get(f"{url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Server Status: {data.get('status')}")
            print(f"Model Loaded: {data.get('model_loaded')}")
            print(f"Model Name: {data.get('model_name')}")
            print(f"GPU Available: {data.get('gpu_available')}")
            print("=" * 60)
            print("✓ Connection successful!")
        else:
            print(f"Server returned error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server")
        print("  Make sure:")
        print("  1. VPN is connected")
        print("  2. SSH tunnel is active (if needed)")
        print("  3. Server is running: python -m vlm_pipeline.vlm_server")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="Server URL to test")
    args = parser.parse_args()
    
    test_connection(args.url)
