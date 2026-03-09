"""
VLM Pipeline Package
====================
Vision-Language Model based task planner for kitchen manipulation.

Modules:
- vlm_context_aggregator: Captures camera frames and builds state context
- vlm_planner: Qwen2-VL based action skeleton generator
- skeleton_to_pddl: Translates skeletons to PDDLStream format
- vlm_executor: Executes plans and monitors for failures
- vlm_main: Main pipeline orchestration

Usage:
    # Mock mode (no GPU)
    python vlm_main.py --mock --goal full_task
    
    # Real VLM with environment
    python vlm_main.py --goal full_task
"""

__all__ = []

# Intentionally keep package init side-effect free.
# Import symbols from submodules directly, e.g.:
# from vlm_pipeline.vlm_client import RemoteVLMPlanner
