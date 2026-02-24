"""
Inference Server for Robotic Manipulation Pipeline
===================================================

A FastAPI-based inference server that provides VLM/LLM inference for the
robotic manipulation pipeline. Designed for remote inference where RLBench
runs on a local machine and sends requests to this server for model inference.

FEATURES
--------
- Smart model caching: Keeps models loaded in GPU memory, swaps only when needed
- Phase-specific endpoints: /discovery, /planning, /execution
- Generic endpoint: /infer for custom model selection
- Base64 image support for transmitting images over HTTP

REQUIREMENTS
------------
    pip install fastapi uvicorn python-multipart pillow

USAGE
-----
1. Start the server on the remote machine (with GPU):
   
       python inference_server.py
   
   Or with custom settings:
   
       python inference_server.py --host 0.0.0.0 --port 8000
   
   For development with auto-reload:
   
       python inference_server.py --reload

2. The server will be available at http://<server-ip>:8000

3. API documentation is auto-generated at:
   - Swagger UI: http://<server-ip>:8000/docs
   - ReDoc: http://<server-ip>:8000/redoc

ENDPOINTS
---------
POST /discovery
    Run discovery phase (VLM) - identify objects in scene images.
    Request body: {images: [base64...], prompt: str, system_prompt: str}

POST /planning  
    Run planning phase (LLM) - generate action plan from context.
    Request body: {context: {...}, prompt: str, system_prompt: str}

POST /execution
    Run execution phase (LLM) - convert plan to robotic actions.
    Request body: {plan: {...}, prompt: str, system_prompt: str}

POST /infer
    Generic inference with any model.
    Request body: {model: str, images: [base64...], prompt: str, system_prompt: str}

GET /models
    List available models and currently loaded model.

GET /health
    Health check endpoint.

CONFIGURATION
-------------
Models are configured in the MODELS dict. Phase-to-model mapping is in PHASE_MODELS.
Modify these to change which models are used for each phase.

EXAMPLE CLIENT USAGE
--------------------
    import requests
    import base64
    
    # Encode image
    with open("scene.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    # Send discovery request
    response = requests.post(
        "http://server:8000/discovery",
        json={
            "images": [img_b64],
            "prompt": "Identify all objects in the scene",
            "system_prompt": "You are a scene analysis system..."
        }
    )
    
    result = response.json()
    print(result["objects"])
"""

import argparse
import base64
import gc
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from rich.console import Console

from adapters import LanguageModelAdapter, VisionLanguageModelAdapter

console = Console(log_time=False, log_path=False)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODELS = {
    # VLMs (Vision-Language Models)
    "qwen-vl": {
        "path": "Qwen/Qwen3-VL-8B-Thinking",
        "type": "vlm",
        "description": "Qwen3 Vision-Language Model for scene understanding",
    },
    "gsarch": {
        "path": "gsarch/ViGoRL-MCTS-SFT-7b-Spatial",
        "type": "vlm", 
        "description": "Spatial reasoning VLM optimized for robotic tasks",
    },
    # LLMs (Language Models)
    "qwen": {
        "path": "Qwen/Qwen3-8B",
        "type": "llm",
        "description": "Qwen3 Language Model for planning and execution",
    },
    "selene": {
        "path": "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
        "type": "llm",
        "description": "Selene Mini for planning validation",
    },
}

# Default model for each phase
PHASE_MODELS = {
    "discovery": "qwen-vl",
    "planning": "qwen",
    "execution": "qwen",
}


# =============================================================================
# MODEL MANAGER (Smart Caching)
# =============================================================================

class ModelManager:
    """
    Manages model loading/unloading with smart caching.
    Keeps one model loaded at a time to maximize GPU memory efficiency.
    """
    
    def __init__(self):
        self.current_model_key: Optional[str] = None
        self.current_adapter = None
        self.load_times: dict[str, float] = {}
    
    def get_model(self, model_key: str):
        """
        Get a model, loading it if necessary.
        Unloads current model if switching to a different one.
        """
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        # Already loaded
        if self.current_model_key == model_key and self.current_adapter is not None:
            console.log(f"[green]using cached model[/] {model_key}")
            return self.current_adapter
        
        # Need to swap
        if self.current_adapter is not None:
            console.log(f"[yellow]unloading[/] {self.current_model_key}")
            self.current_adapter.unload()
            self.current_adapter = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load new model
        model_config = MODELS[model_key]
        console.log(f"[cyan]loading[/] {model_key} ({model_config['path']})")
        
        start = time.time()
        if model_config["type"] == "vlm":
            self.current_adapter = VisionLanguageModelAdapter(model_config["path"])
        else:
            self.current_adapter = LanguageModelAdapter(model_config["path"])
        
        self.current_adapter.load()
        load_time = time.time() - start
        self.load_times[model_key] = load_time
        
        console.log(f"[green]loaded[/] {model_key} in {load_time:.1f}s")
        self.current_model_key = model_key
        
        return self.current_adapter
    
    def get_for_phase(self, phase: str, model_override: Optional[str] = None):
        """Get the appropriate model for a pipeline phase."""
        model_key = model_override or PHASE_MODELS.get(phase)
        if not model_key:
            raise ValueError(f"Unknown phase: {phase}")
        return self.get_model(model_key)
    
    def status(self) -> dict:
        """Get current model manager status."""
        return {
            "current_model": self.current_model_key,
            "available_models": list(MODELS.keys()),
            "phase_defaults": PHASE_MODELS,
            "load_times": self.load_times,
        }
    
    def unload_all(self):
        """Unload all models and free GPU memory."""
        if self.current_adapter is not None:
            console.log(f"[yellow]unloading[/] {self.current_model_key}")
            self.current_adapter.unload()
            self.current_adapter = None
            self.current_model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Global model manager instance
model_manager = ModelManager()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class DiscoveryRequest(BaseModel):
    images: list[str]  # Base64 encoded images
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # Override default model

class PlanningRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None

class ExecutionRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None

class InferRequest(BaseModel):
    model: str
    images: Optional[list[str]] = None  # Base64 encoded images (for VLMs)
    prompt: str
    system_prompt: Optional[str] = None

class InferResponse(BaseModel):
    output: str
    duration: float
    model: str
    model_was_cached: bool


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_images(base64_images: list[str]) -> list[Image.Image]:
    """Decode base64 strings to PIL Images."""
    images = []
    for b64 in base64_images:
        try:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(img)
        except Exception as e:
            console.log(f"[red]failed to decode image: {e}[/]")
    return images


def run_inference(
    model_key: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    images: Optional[list[Image.Image]] = None,
) -> InferResponse:
    """Run inference with the specified model."""
    was_cached = model_manager.current_model_key == model_key
    adapter = model_manager.get_model(model_key)
    model_config = MODELS[model_key]
    
    start = time.time()
    
    if model_config["type"] == "vlm":
        output = adapter.generate(
            images=images or [],
            prompt=prompt,
            system_prompt=system_prompt,
        )
    else:
        output = adapter.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )
    
    duration = time.time() - start
    
    return InferResponse(
        output=output,
        duration=duration,
        model=model_key,
        model_was_cached=was_cached,
    )


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    console.log("[bold green]inference server starting[/]")
    console.log(f"[dim]available models: {list(MODELS.keys())}[/]")
    yield
    console.log("[bold yellow]inference server shutting down[/]")
    model_manager.unload_all()
    console.log("[green]cleanup complete[/]")


app = FastAPI(
    title="Robotic Pipeline Inference Server",
    description="VLM/LLM inference server for robotic manipulation pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}


@app.get("/models")
async def list_models():
    """List available models and current status."""
    return model_manager.status()


@app.post("/discovery", response_model=InferResponse)
async def discovery(request: DiscoveryRequest):
    """
    Discovery phase: Identify objects and surfaces in scene images.
    Uses VLM by default.
    """
    try:
        images = decode_images(request.images)
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        model_key = request.model or PHASE_MODELS["discovery"]
        
        return run_inference(
            model_key=model_key,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            images=images,
        )
    except Exception as e:
        console.log(f"[red]discovery error: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/planning", response_model=InferResponse)
async def planning(request: PlanningRequest):
    """
    Planning phase: Generate action plan from context.
    Uses LLM by default.
    """
    try:
        model_key = request.model or PHASE_MODELS["planning"]
        
        return run_inference(
            model_key=model_key,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
        )
    except Exception as e:
        console.log(f"[red]planning error: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execution", response_model=InferResponse)
async def execution(request: ExecutionRequest):
    """
    Execution phase: Convert plan to robotic actions.
    Uses LLM by default.
    """
    try:
        model_key = request.model or PHASE_MODELS["execution"]
        
        return run_inference(
            model_key=model_key,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
        )
    except Exception as e:
        console.log(f"[red]execution error: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Generic inference endpoint.
    Specify model explicitly for custom workflows.
    """
    try:
        if request.model not in MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown model: {request.model}. Available: {list(MODELS.keys())}"
            )
        
        images = None
        if request.images:
            images = decode_images(request.images)
        
        return run_inference(
            model_key=request.model,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            images=images,
        )
    except HTTPException:
        raise
    except Exception as e:
        console.log(f"[red]infer error: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload():
    """Unload current model to free GPU memory."""
    model_manager.unload_all()
    return {"status": "ok", "message": "Model unloaded"}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference server for robotic manipulation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_server.py                    # Start with defaults
  python inference_server.py --port 8080        # Custom port
  python inference_server.py --host 0.0.0.0     # Listen on all interfaces
  python inference_server.py --reload           # Development mode with auto-reload
        """,
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    console.log(f"[bold]Starting server on {args.host}:{args.port}[/]")
    
    uvicorn.run(
        "inference_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
