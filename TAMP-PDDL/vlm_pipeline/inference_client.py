"""
Inference Client for Robotic Manipulation Pipeline
===================================================

A Python client library for communicating with the inference server.
Use this on your local machine (where RLBench runs) to send inference
requests to the remote GPU server.

REQUIREMENTS
------------
    pip install requests pillow

USAGE
-----
1. As a library in your code:

    from inference_client import InferenceClient
    
    client = InferenceClient("http://server-ip:8000")
    
    # Discovery with images
    result = client.discovery(
        images=["scene1.jpg", "scene2.jpg"],
        prompt="Identify all objects",
        system_prompt="You are a scene analysis system..."
    )
    print(result.objects)
    
    # Planning with text
    result = client.planning(
        prompt="Create a plan to organize the objects",
        system_prompt="You are a planning system..."
    )
    print(result.output)

2. As a command-line tool:

    # Discovery
    python inference_client.py discovery \\
        --server http://server:8000 \\
        --images scene1.jpg scene2.jpg \\
        --prompt "Identify objects" \\
        --system-prompt prompts/discovery.md
    
    # Planning
    python inference_client.py planning \\
        --server http://server:8000 \\
        --prompt "Create a plan" \\
        --system-prompt prompts/planning.md
    
    # Generic inference with specific model
    python inference_client.py infer \\
        --server http://server:8000 \\
        --model qwen-vl \\
        --images scene.jpg \\
        --prompt "Analyze this scene"

3. Test connection:

    python inference_client.py health --server http://server:8000

API REFERENCE
-------------
InferenceClient(server_url, timeout=300)
    Main client class. timeout is in seconds (default 5 min for large models).

.health() -> dict
    Check server health and GPU availability.

.models() -> dict
    List available models and current status.

.discovery(images, prompt, system_prompt=None, model=None) -> InferenceResult
    Run discovery phase with images.

.planning(prompt, system_prompt=None, model=None) -> InferenceResult
    Run planning phase with text.

.execution(prompt, system_prompt=None, model=None) -> InferenceResult
    Run execution phase with text.

.infer(model, prompt, system_prompt=None, images=None) -> InferenceResult
    Generic inference with any model.

InferenceResult
    .output: str          - Raw model output
    .duration: float      - Inference time in seconds
    .model: str           - Model used
    .model_was_cached: bool - Whether model was already loaded
"""

import argparse
import base64
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: requests library not installed. Run: pip install requests")
    sys.exit(1)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InferenceResult:
    """Result from an inference request."""
    output: str
    duration: float
    model: str
    model_was_cached: bool
    
    # Parsed fields (populated by parse methods)
    objects: list = None
    surfaces: list = None
    scene_summary: str = None
    spatial_relationships: str = None
    steps: list = None
    actions: list = None
    
    def __post_init__(self):
        self.objects = []
        self.surfaces = []
        self.steps = []
        self.actions = []
        self._parse_output()
    
    def _parse_output(self):
        """Parse structured output from model response."""
        # Extract objects
        self.objects = self._extract_json_tags("object")
        # Extract surfaces
        self.surfaces = self._extract_json_tags("surface")
        # Extract steps (for planning)
        self.steps = self._extract_json_tags("step")
        # Extract actions (for execution)
        self.actions = self._extract_json_tags("action")
        # Extract text sections
        self.scene_summary = self._extract_text_tag("scene_summary")
        self.spatial_relationships = self._extract_text_tag("spatial_relationships")
    
    def _extract_json_tags(self, tag: str) -> list:
        """Extract JSON objects from XML-style tags."""
        pattern = f"<{tag}>\\s*({{.*?}})\\s*</{tag}>"
        matches = re.findall(pattern, self.output, re.DOTALL)
        results = []
        for m in matches:
            try:
                results.append(json.loads(m))
            except json.JSONDecodeError:
                pass
        return results
    
    def _extract_text_tag(self, tag: str) -> Optional[str]:
        """Extract text content from XML-style tags."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, self.output, re.DOTALL)
        return match.group(1).strip() if match else None


# =============================================================================
# CLIENT CLASS
# =============================================================================

class InferenceClient:
    """Client for the inference server."""
    
    def __init__(self, server_url: str, timeout: int = 300):
        """
        Initialize client.
        
        Args:
            server_url: Full URL of inference server (e.g., http://server:8000)
            timeout: Request timeout in seconds (default 5 minutes for large models)
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _encode_images(self, image_paths: list) -> list:
        """Encode multiple images to base64."""
        return [self._encode_image(p) for p in image_paths]
    
    def _load_prompt(self, prompt_or_path: str) -> str:
        """Load prompt from file if it's a valid path, otherwise return as-is."""
        try:
            if Path(prompt_or_path).exists():
                with open(prompt_or_path) as f:
                    return f.read()
        except OSError:
            # Reached if prompt is a massive literal string (Errno 36 File name too long)
            pass
        return prompt_or_path
    
    def _post(self, endpoint: str, data: dict) -> dict:
        """Make POST request to server."""
        url = f"{self.server_url}/{endpoint}"
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to server at {self.server_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json().get("detail", str(e))
            raise RuntimeError(f"Server error: {error_detail}")
    
    def _get(self, endpoint: str) -> dict:
        """Make GET request to server."""
        url = f"{self.server_url}/{endpoint}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to server at {self.server_url}")
    
    def health(self) -> dict:
        """Check server health."""
        return self._get("health")
    
    def models(self) -> dict:
        """List available models and status."""
        return self._get("models")
    
    def discovery(
        self,
        images: list,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run discovery phase.
        
        Args:
            images: List of image file paths
            prompt: User prompt (or path to prompt file)
            system_prompt: System prompt (or path to prompt file)
            model: Override default model
            
        Returns:
            InferenceResult with parsed objects and surfaces
        """
        data = {
            "images": self._encode_images(images),
            "prompt": self._load_prompt(prompt),
        }
        if system_prompt:
            data["system_prompt"] = self._load_prompt(system_prompt)
        if model:
            data["model"] = model
        
        result = self._post("discovery", data)
        return InferenceResult(**result)
    
    def planning(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run planning phase.
        
        Args:
            prompt: User prompt with context (or path to prompt file)
            system_prompt: System prompt (or path to prompt file)
            model: Override default model
            
        Returns:
            InferenceResult with parsed steps
        """
        data = {"prompt": self._load_prompt(prompt)}
        if system_prompt:
            data["system_prompt"] = self._load_prompt(system_prompt)
        if model:
            data["model"] = model
        
        result = self._post("planning", data)
        return InferenceResult(**result)
    
    def execution(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run execution phase.
        
        Args:
            prompt: User prompt with plan (or path to prompt file)
            system_prompt: System prompt (or path to prompt file)
            model: Override default model
            
        Returns:
            InferenceResult with parsed actions
        """
        data = {"prompt": self._load_prompt(prompt)}
        if system_prompt:
            data["system_prompt"] = self._load_prompt(system_prompt)
        if model:
            data["model"] = model
        
        result = self._post("execution", data)
        return InferenceResult(**result)
    
    def infer(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[list] = None,
    ) -> InferenceResult:
        """
        Generic inference with any model.
        
        Args:
            model: Model key (e.g., "qwen-vl", "qwen")
            prompt: User prompt (or path to prompt file)
            system_prompt: System prompt (or path to prompt file)
            images: Optional list of image file paths (for VLMs)
            
        Returns:
            InferenceResult
        """
        data = {
            "model": model,
            "prompt": self._load_prompt(prompt),
        }
        if system_prompt:
            data["system_prompt"] = self._load_prompt(system_prompt)
        if images:
            data["images"] = self._encode_images(images)
        
        result = self._post("infer", data)
        return InferenceResult(**result)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference client for robotic manipulation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--server",
        "-s",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check server health")
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    
    # Discovery command
    discovery_parser = subparsers.add_parser("discovery", help="Run discovery phase")
    discovery_parser.add_argument("--images", "-i", nargs="+", required=True, help="Image files")
    discovery_parser.add_argument("--prompt", "-p", required=True, help="Prompt or prompt file")
    discovery_parser.add_argument("--system-prompt", help="System prompt or file")
    discovery_parser.add_argument("--model", "-m", help="Override model")
    discovery_parser.add_argument("--output", "-o", help="Save result to JSON file")
    
    # Planning command
    planning_parser = subparsers.add_parser("planning", help="Run planning phase")
    planning_parser.add_argument("--prompt", "-p", required=True, help="Prompt or prompt file")
    planning_parser.add_argument("--system-prompt", help="System prompt or file")
    planning_parser.add_argument("--model", "-m", help="Override model")
    planning_parser.add_argument("--output", "-o", help="Save result to JSON file")
    
    # Execution command
    execution_parser = subparsers.add_parser("execution", help="Run execution phase")
    execution_parser.add_argument("--prompt", "-p", required=True, help="Prompt or prompt file")
    execution_parser.add_argument("--system-prompt", help="System prompt or file")
    execution_parser.add_argument("--model", "-m", help="Override model")
    execution_parser.add_argument("--output", "-o", help="Save result to JSON file")
    
    # Generic infer command
    infer_parser = subparsers.add_parser("infer", help="Generic inference")
    infer_parser.add_argument("--model", "-m", required=True, help="Model to use")
    infer_parser.add_argument("--images", "-i", nargs="*", help="Image files (for VLMs)")
    infer_parser.add_argument("--prompt", "-p", required=True, help="Prompt or prompt file")
    infer_parser.add_argument("--system-prompt", help="System prompt or file")
    infer_parser.add_argument("--output", "-o", help="Save result to JSON file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = InferenceClient(args.server)
    
    try:
        if args.command == "health":
            result = client.health()
            print(json.dumps(result, indent=2))
        
        elif args.command == "models":
            result = client.models()
            print(json.dumps(result, indent=2))
        
        elif args.command == "discovery":
            result = client.discovery(
                images=args.images,
                prompt=args.prompt,
                system_prompt=args.system_prompt,
                model=args.model,
            )
            _print_result(result, args.output)
        
        elif args.command == "planning":
            result = client.planning(
                prompt=args.prompt,
                system_prompt=args.system_prompt,
                model=args.model,
            )
            _print_result(result, args.output)
        
        elif args.command == "execution":
            result = client.execution(
                prompt=args.prompt,
                system_prompt=args.system_prompt,
                model=args.model,
            )
            _print_result(result, args.output)
        
        elif args.command == "infer":
            result = client.infer(
                model=args.model,
                prompt=args.prompt,
                system_prompt=args.system_prompt,
                images=args.images,
            )
            _print_result(result, args.output)
    
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _print_result(result: InferenceResult, output_path: Optional[str] = None):
    """Print result and optionally save to file."""
    print(f"\n{'='*60}")
    print(f"Model: {result.model} (cached: {result.model_was_cached})")
    print(f"Duration: {result.duration:.2f}s")
    print(f"{'='*60}\n")
    
    print("OUTPUT:")
    print(result.output)
    
    if result.objects:
        print(f"\nPARSED OBJECTS ({len(result.objects)}):")
        for obj in result.objects:
            print(f"  - {json.dumps(obj)}")
    
    if result.surfaces:
        print(f"\nPARSED SURFACES ({len(result.surfaces)}):")
        for surf in result.surfaces:
            print(f"  - {json.dumps(surf)}")
    
    if result.steps:
        print(f"\nPARSED STEPS ({len(result.steps)}):")
        for step in result.steps:
            print(f"  - {json.dumps(step)}")
    
    if result.actions:
        print(f"\nPARSED ACTIONS ({len(result.actions)}):")
        for action in result.actions:
            print(f"  - {json.dumps(action)}")
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump({
                "output": result.output,
                "duration": result.duration,
                "model": result.model,
                "model_was_cached": result.model_was_cached,
                "objects": result.objects,
                "surfaces": result.surfaces,
                "steps": result.steps,
                "actions": result.actions,
                "scene_summary": result.scene_summary,
            }, f, indent=2)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
