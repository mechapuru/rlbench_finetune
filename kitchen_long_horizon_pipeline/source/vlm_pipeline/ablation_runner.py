"""
Ablation Runner - VLM planning with different constraint levels.
NO CoppeliaSim/RLBench imports. Pure offline.

Levels:
  - Level 0: Domain file actions (pick/place/open_lid preconditions)
  - Level 1: Domain actions + Spatial blocking annotations in state

Usage:
    python -m vlm_pipeline.ablation_runner --all --goal "Move all mugs inside box"
    python -m vlm_pipeline.ablation_runner --level 0 --goal "..." --vlm
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List

# NO environment imports here - this is offline only
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.constraints import ConstraintLevel, LEVEL_NONE, LEVEL_SPATIAL
from vlm_pipeline.context_aggregator_v2 import SceneAgnosticContextAggregator, PromptBundle


def print_bundle(bundle: PromptBundle):
    print("\n" + "=" * 70)
    print(f"CONSTRAINT LEVEL: {bundle.constraint_level.name} (Level {bundle.constraint_level.value})")
    print("=" * 70)
    print("\n--- SYSTEM PROMPT ---")
    print(bundle.system_prompt)
    print("\n--- USER PROMPT ---")
    print(bundle.user_prompt)
    print("\n--- STATE TEXT ---")
    print(bundle.state_text)


def run_level(level: ConstraintLevel, goal: str, use_vlm: bool = False, save: bool = True, text_only: bool = False, planner=None) -> Dict[str, Any]:
    print(f"\n{'#' * 70}")
    print(f"# RUNNING ABLATION: Level {level.value} ({level.name})")
    print(f"{'#' * 70}")
    
    result = {
        "level": level.value,
        "level_name": level.name,
        "goal": goal,
        "timestamp": datetime.now().isoformat(),
        "text_only": text_only,
    }
    
    aggregator = SceneAgnosticContextAggregator(constraint_level=level, env=None)
    bundle = aggregator.create_prompt_bundle(goal)
    
    # Create a placeholder image (gray with text) instead of black
    import numpy as np
    placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray
    bundle.composite_image = np.concatenate([
        np.concatenate([placeholder, placeholder, placeholder], axis=1),
        np.concatenate([placeholder, placeholder, placeholder], axis=1)
    ], axis=0)
    
    result["system_prompt"] = bundle.system_prompt
    result["user_prompt"] = bundle.user_prompt
    result["state_text"] = bundle.state_text
    
    print_bundle(bundle)
    
    if use_vlm:
        # Use shared planner if provided, otherwise create new one
        if planner is None:
            from vlm_pipeline.vlm_planner import VLMPlanner
            print("\n--- LOADING VLM ---")
            planner = VLMPlanner(use_4bit=False)
            if not planner.load_model():
                result["error"] = "Failed to load VLM"
                return result
        
        print("\n--- QUERYING VLM ---")
        start = time.time()
        
        # Use text-only mode if specified (no image in prompt)
        if text_only:
            plan_result = planner.generate_plan_text_only(
                system_prompt=bundle.system_prompt,
                user_prompt=bundle.user_prompt,
            )
        else:
            plan_result = planner.generate_plan(
                image=bundle.composite_image,
                system_prompt=bundle.system_prompt,
                user_prompt=bundle.user_prompt,
            )
        elapsed = time.time() - start
        
        result["vlm_output"] = plan_result.raw_output
        result["parsed_plan"] = [str(a) for a in plan_result.skeleton]
        result["inference_time"] = elapsed
        result["success"] = plan_result.success
        
        print(f"\n--- VLM OUTPUT ({elapsed:.1f}s) ---")
        print(plan_result.raw_output)
        print("\n--- PARSED PLAN ---")
        for i, a in enumerate(plan_result.skeleton, 1):
            print(f"{i}. {a}")
    else:
        result["vlm_output"] = "(VLM not queried - dry run)"
    
    if save:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VLM_outputs")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"level{level.value}_{level.name}.json"
        filepath = os.path.join(out_dir, fname)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Also save as readable text
        txt_fname = f"level{level.value}_{level.name}.txt"
        txt_path = os.path.join(out_dir, txt_fname)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"LEVEL {level.value} ({level.name})\n")
            f.write("=" * 60 + "\n\n")
            f.write("GOAL: " + result["goal"] + "\n\n")
            f.write("--- SYSTEM PROMPT ---\n")
            f.write(result["system_prompt"] + "\n\n")
            f.write("--- USER PROMPT ---\n")
            f.write(result["user_prompt"] + "\n\n")
            f.write("--- VLM OUTPUT ---\n")
            f.write(result.get("vlm_output", "(not run)") + "\n\n")
            if "parsed_plan" in result:
                f.write("--- PARSED PLAN ---\n")
                for i, a in enumerate(result["parsed_plan"], 1):
                    f.write(f"{i}. {a}\n")
                f.write(f"\nInference time: {result.get('inference_time', 0):.1f}s\n")
        
        print(f"\n[Saved to {filepath}]")
        print(f"[Saved to {txt_path}]")
        print(f"\n[Saved to {os.path.join(out_dir, fname)}]")
    
    return result


def run_all(goal: str, use_vlm: bool = False, text_only: bool = False) -> List[Dict]:
    results = []
    
    # Load VLM ONCE and share across all levels to avoid memory issues
    shared_planner = None
    if use_vlm:
        from vlm_pipeline.vlm_planner import VLMPlanner
        print("\n--- LOADING VLM (shared across all levels) ---")
        shared_planner = VLMPlanner(use_4bit=False)
        if not shared_planner.load_model():
            print("ERROR: Failed to load VLM")
            return results
    
    for level in [LEVEL_NONE, LEVEL_SPATIAL]:
        results.append(run_level(level, goal, use_vlm=use_vlm, text_only=text_only, planner=shared_planner))
    
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"\nLevel {r['level']} ({r['level_name']}):")
        print(f"  System prompt: {len(r['system_prompt'])} chars")
        print(f"  User prompt: {len(r['user_prompt'])} chars")
        if "parsed_plan" in r:
            print(f"  Plan: {len(r['parsed_plan'])} actions, {r.get('inference_time', 0):.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VLM constraint ablation")
    parser.add_argument("--level", type=int, choices=[0, 1])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--goal", type=str, default="Move all mugs to placement_boundary")
    parser.add_argument("--vlm", action="store_true", help="Query VLM (slow)")
    parser.add_argument("--text-only", action="store_true", help="Text-only mode (no image)")
    parser.add_argument("--no-save", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VLM CONSTRAINT ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"Goal: {args.goal}")
    print(f"VLM Query: {'Yes' if args.vlm else 'No (dry run)'}")
    print(f"Text-only: {'Yes' if args.text_only else 'No (with image)'}")
    print("\nLevels:")
    print("  0 = Domain actions only (pick/place/open_lid preconditions)")
    print("  1 = Domain actions + Spatial blocking annotations")
    
    if args.all:
        run_all(args.goal, use_vlm=args.vlm, text_only=args.text_only)
    elif args.level is not None:
        run_level(ConstraintLevel(args.level), args.goal, use_vlm=args.vlm, save=not args.no_save, text_only=args.text_only)
    else:
        print("\nUsage:")
        print("  python -m vlm_pipeline.ablation_runner --level 0 --goal '...'")
        print("  python -m vlm_pipeline.ablation_runner --all --goal '...'")
        print("  python -m vlm_pipeline.ablation_runner --all --vlm --text-only")
if __name__ == "__main__":
    main()
