#!/usr/bin/env python3
"""
VLM Inference Server
====================
Runs on remote GPU server, exposes VLM planning via HTTP API.

Usage on SERVER:
    python -m vlm_pipeline.vlm_server --port 8000

Usage on CLIENT (your local machine):
    # Set environment variable or pass to client
    export VLM_SERVER_URL="http://gvlab2.iiit.ac.in:8000"
    python -m vlm_pipeline.vlm_main --goal "..." --remote
"""

import os
import sys
import json
import base64
import argparse
import numpy as np
from io import BytesIO
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

# FastAPI for HTTP server
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.vlm_planner import VLMPlanner, ActionSkeleton, PlanResult


# ============================================================
# PYDANTIC MODELS FOR API
# ============================================================

class PlanRequest(BaseModel):
    """Request to generate a plan."""
    image_base64: str  # Base64 encoded composite image
    system_prompt: str
    user_prompt: str
    goal: str
    max_new_tokens: int = 1024
    temperature: float = 0.1


class ActionResponse(BaseModel):
    """Single action in response."""
    action_name: str
    args: List[str]


class PlanResponse(BaseModel):
    """Response with generated plan."""
    success: bool
    actions: List[ActionResponse]
    raw_output: str
    inference_time: float
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    gpu_available: bool


# ============================================================
# VLM SERVER
# ============================================================

class VLMServer:
    """VLM inference server."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.planner: Optional[VLMPlanner] = None
        self.loaded = False
        
    def load_model(self) -> bool:
        """Load the VLM model."""
        print(f"[VLMServer] Loading model: {self.model_name}")
        self.planner = VLMPlanner(
            model_name=self.model_name,
            use_4bit=self.use_4bit
        )
        self.loaded = self.planner.load_model()
        if self.loaded:
            print("[VLMServer] Model loaded successfully!")
        else:
            print("[VLMServer] Failed to load model!")
        return self.loaded
    
    def generate_plan(self, request: PlanRequest) -> PlanResponse:
        """Generate a plan from the request."""
        if not self.loaded:
            return PlanResponse(
                success=False,
                actions=[],
                raw_output="",
                inference_time=0,
                error_message="Model not loaded"
            )
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(request.image_base64)
            image_array = np.load(BytesIO(image_bytes), allow_pickle=True)
            
            # Call VLM
            result = self.planner.generate_plan(
                image=image_array,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature
            )
            
            # Convert to response
            actions = [
                ActionResponse(action_name=a.action_name, args=list(a.args))
                for a in result.skeleton
            ]
            
            return PlanResponse(
                success=result.success,
                actions=actions,
                raw_output=result.raw_output,
                inference_time=result.inference_time,
                error_message=result.error_message
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PlanResponse(
                success=False,
                actions=[],
                raw_output="",
                inference_time=0,
                error_message=str(e)
            )


# ============================================================
# FASTAPI APP
# ============================================================

def create_app(model_name: str = "Qwen/Qwen2-VL-7B-Instruct", use_4bit: bool = True) -> FastAPI:
    """Create the FastAPI application."""
    
    app = FastAPI(
        title="VLM Planning Server",
        description="Remote VLM inference for robot task planning",
        version="1.0.0"
    )
    
    # CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize server
    server = VLMServer(model_name=model_name, use_4bit=use_4bit)
    
    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        server.load_model()
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check server health."""
        import torch
        return HealthResponse(
            status="ok" if server.loaded else "model_not_loaded",
            model_loaded=server.loaded,
            model_name=server.model_name,
            gpu_available=torch.cuda.is_available()
        )
    
    @app.post("/plan", response_model=PlanResponse)
    async def generate_plan(request: PlanRequest):
        """Generate a plan from the given context."""
        if not server.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return server.generate_plan(request)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "VLM Planning Server",
            "endpoints": {
                "/health": "GET - Check server health",
                "/plan": "POST - Generate action plan"
            }
        }
    
    return app


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model name")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()
    
    if not HAS_FASTAPI:
        print("ERROR: FastAPI not installed.")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    print("=" * 60)
    print("VLM INFERENCE SERVER")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"4-bit quantization: {not args.no_4bit}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    
    app = create_app(model_name=args.model, use_4bit=not args.no_4bit)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
