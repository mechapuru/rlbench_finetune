"""
Experiment 4: VLM-based Visual Planner
======================================
Benchmarks the VLM pipeline against baselines.

Compares:
- VLM planning accuracy
- Execution success rate
- Replanning effectiveness
- Time performance
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'pddlstream'))
sys.path.insert(0, os.path.join(BASE_DIR, 'vlm_pipeline'))

# Experiment config
EXPERIMENT_NAME = "exp4_vlm_planner"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", EXPERIMENT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ground truth from baseline experiments
GROUND_TRUTH_ACTIONS = [
    ('pick', 'mug_box'),
    ('place', 'mug_box', 'placement_boundary'),
    ('open-lid', 'box_lid'),
    ('pick', 'mug_inside_box'),
    ('place', 'mug_inside_box', 'placement_boundary'),
    ('pick', 'soup'),
    ('place', 'soup', 'cupboard_boundary'),
]

GROUND_TRUTH_SEQUENCE = [
    'pick(mug_box)',
    'place(mug_box, placement_boundary)',
    'open-lid(box_lid)',
    'pick(mug_inside_box)',
    'place(mug_inside_box, placement_boundary)',
    'pick(soup)',
    'place(soup, cupboard_boundary)',
]


class Experiment4:
    """VLM Planner experiment."""
    
    def __init__(self, 
                 use_mock: bool = False,
                 use_env: bool = False,
                 headless: bool = True,
                 num_trials: int = 5):
        """
        Initialize experiment.
        
        Args:
            use_mock: Use mock VLM (no GPU)
            use_env: Use real RLBench environment
            headless: Run headless
            num_trials: Number of trials to run
        """
        self.use_mock = use_mock
        self.use_env = use_env
        self.headless = headless
        self.num_trials = num_trials
        
        self.results = {
            "experiment": EXPERIMENT_NAME,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "use_mock": use_mock,
                "use_env": use_env,
                "num_trials": num_trials
            },
            "trials": [],
            "summary": {}
        }
    
    def compare_plans(self, 
                      generated: List[str], 
                      ground_truth: List[str]) -> Dict[str, Any]:
        """
        Compare generated plan against ground truth.
        
        Args:
            generated: Generated action strings
            ground_truth: Ground truth action strings
            
        Returns:
            Comparison metrics
        """
        # Exact match
        exact_match = generated == ground_truth
        
        # Action-level accuracy
        correct = 0
        for i, (gen, gt) in enumerate(zip(generated, ground_truth)):
            if gen == gt:
                correct += 1
        
        action_accuracy = correct / len(ground_truth) if ground_truth else 0
        
        # Check critical ordering constraints
        constraints_satisfied = True
        constraint_errors = []
        
        # Constraint 1: pick(mug_box) before open-lid(box_lid)
        pick_mug_idx = -1
        open_lid_idx = -1
        for i, a in enumerate(generated):
            if 'pick(mug_box)' in a:
                pick_mug_idx = i
            if 'open-lid(box_lid)' in a or 'open_lid(box_lid)' in a:
                open_lid_idx = i
        
        if open_lid_idx >= 0 and pick_mug_idx >= 0:
            if open_lid_idx < pick_mug_idx:
                constraints_satisfied = False
                constraint_errors.append("open-lid before picking mug_box (should remove from top first)")
        
        # Constraint 2: place(mug_box, ...) before open-lid(box_lid)
        place_mug_idx = -1
        for i, a in enumerate(generated):
            if 'place(mug_box' in a:
                place_mug_idx = i
                break
        
        if open_lid_idx >= 0 and place_mug_idx >= 0:
            if open_lid_idx < place_mug_idx:
                constraints_satisfied = False
                constraint_errors.append("open-lid before placing mug_box")
        
        # Constraint 3: open-lid before pick(mug_inside_box)
        pick_inside_idx = -1
        for i, a in enumerate(generated):
            if 'pick(mug_inside_box)' in a:
                pick_inside_idx = i
                break
        
        if pick_inside_idx >= 0 and open_lid_idx >= 0:
            if pick_inside_idx < open_lid_idx:
                constraints_satisfied = False
                constraint_errors.append("pick mug_inside_box before opening lid")
        elif pick_inside_idx >= 0 and open_lid_idx < 0:
            constraints_satisfied = False
            constraint_errors.append("pick mug_inside_box without opening lid")
        
        return {
            "exact_match": exact_match,
            "action_accuracy": action_accuracy,
            "generated_length": len(generated),
            "ground_truth_length": len(ground_truth),
            "constraints_satisfied": constraints_satisfied,
            "constraint_errors": constraint_errors
        }
    
    def run_planning_trial(self, trial_num: int) -> Dict[str, Any]:
        """
        Run a single planning trial (no execution).
        
        Args:
            trial_num: Trial number
            
        Returns:
            Trial results
        """
        print(f"\n{'='*60}")
        print(f"TRIAL {trial_num}")
        print(f"{'='*60}")
        
        trial_result = {
            "trial": trial_num,
            "planning": {},
            "comparison": {},
            "errors": []
        }
        
        try:
            # Import here to avoid loading at module level
            from vlm_pipeline.vlm_context_aggregator import VLMContextAggregator
            from vlm_pipeline.vlm_planner import VLMPlanner, MockVLMPlanner
            
            # Initialize modules
            aggregator = VLMContextAggregator()
            
            if self.use_mock:
                planner = MockVLMPlanner()
            else:
                planner = VLMPlanner(use_4bit=True)
                if not planner.load_model():
                    trial_result["errors"].append("Failed to load VLM model")
                    return trial_result
            
            # Define goal
            goal = """Move mug_box to placement_boundary.
Open the box lid.
Move mug_inside_box to placement_boundary.
Move soup to cupboard_boundary."""
            
            # Create prompt bundle
            bundle = aggregator.create_prompt_bundle_offline(goal)
            
            # Generate plan
            start_time = time.time()
            result = planner.generate_plan(
                image=bundle.composite_image,
                system_prompt=bundle.system_prompt,
                user_prompt=bundle.user_prompt
            )
            planning_time = time.time() - start_time
            
            trial_result["planning"] = {
                "success": result.success,
                "time": planning_time,
                "raw_output": result.raw_output,
                "parsed_actions": [str(a) for a in result.skeleton]
            }
            
            if result.success:
                # Validate plan
                is_valid, errors = planner.validate_plan(result.skeleton)
                trial_result["planning"]["valid"] = is_valid
                trial_result["planning"]["validation_errors"] = errors
                
                # Compare to ground truth
                generated = [str(a) for a in result.skeleton]
                trial_result["comparison"] = self.compare_plans(
                    generated, GROUND_TRUTH_SEQUENCE
                )
            else:
                trial_result["errors"].append(f"Planning failed: {result.error_message}")
            
        except Exception as e:
            trial_result["errors"].append(str(e))
            import traceback
            traceback.print_exc()
        
        return trial_result
    
    def run_full_trial(self, trial_num: int) -> Dict[str, Any]:
        """
        Run a full trial with execution.
        
        Args:
            trial_num: Trial number
            
        Returns:
            Trial results
        """
        print(f"\n{'='*60}")
        print(f"FULL TRIAL {trial_num}")
        print(f"{'='*60}")
        
        trial_result = {
            "trial": trial_num,
            "planning": {},
            "execution": {},
            "comparison": {},
            "errors": []
        }
        
        try:
            from vlm_pipeline.vlm_main import VLMPipeline, PipelineConfig
            
            config = PipelineConfig(
                use_mock_vlm=self.use_mock,
                use_real_env=self.use_env,
                headless=self.headless,
                save_logs=False
            )
            
            pipeline = VLMPipeline(config)
            
            if not pipeline.initialize():
                trial_result["errors"].append("Failed to initialize pipeline")
                return trial_result
            
            goal = """Move mug_box to placement_boundary.
Open the box lid.
Move mug_inside_box to placement_boundary.
Move soup to cupboard_boundary."""
            
            # Run pipeline
            start_time = time.time()
            result = pipeline.run(goal)
            total_time = time.time() - start_time
            
            trial_result["execution"] = {
                "success": result["success"],
                "total_time": total_time,
                "cycles": result["total_cycles"],
                "actions_completed": result["actions_completed"],
                "actions_total": result["actions_total"]
            }
            
            # Get planning info from log
            if pipeline.run_log["cycles"]:
                first_cycle = pipeline.run_log["cycles"][0]
                trial_result["planning"] = first_cycle.get("plan_result", {})
                
                if trial_result["planning"].get("skeleton"):
                    trial_result["comparison"] = self.compare_plans(
                        trial_result["planning"]["skeleton"],
                        GROUND_TRUTH_SEQUENCE
                    )
            
            pipeline.shutdown()
            
        except Exception as e:
            trial_result["errors"].append(str(e))
            import traceback
            traceback.print_exc()
        
        return trial_result
    
    def run(self, full_execution: bool = False):
        """
        Run all trials.
        
        Args:
            full_execution: Whether to run with execution or planning-only
        """
        print("\n" + "="*60)
        print(f"EXPERIMENT 4: VLM PLANNER")
        print(f"Mode: {'Full Execution' if full_execution else 'Planning Only'}")
        print(f"Trials: {self.num_trials}")
        print(f"Mock VLM: {self.use_mock}")
        print("="*60)
        
        for i in range(1, self.num_trials + 1):
            if full_execution:
                trial_result = self.run_full_trial(i)
            else:
                trial_result = self.run_planning_trial(i)
            
            self.results["trials"].append(trial_result)
            
            # Print trial summary
            print(f"\nTrial {i} Summary:")
            if trial_result.get("comparison"):
                comp = trial_result["comparison"]
                print(f"  Exact match: {comp['exact_match']}")
                print(f"  Action accuracy: {comp['action_accuracy']:.1%}")
                print(f"  Constraints satisfied: {comp['constraints_satisfied']}")
            if trial_result.get("errors"):
                print(f"  Errors: {trial_result['errors']}")
        
        # Compute summary
        self.compute_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def compute_summary(self):
        """Compute summary statistics."""
        trials = self.results["trials"]
        n = len(trials)
        
        if n == 0:
            return
        
        # Planning metrics
        planning_successes = sum(1 for t in trials 
                                  if t.get("planning", {}).get("success", False))
        
        planning_times = [t["planning"]["time"] for t in trials 
                          if t.get("planning", {}).get("time")]
        
        # Comparison metrics
        exact_matches = sum(1 for t in trials 
                           if t.get("comparison", {}).get("exact_match", False))
        
        constraints_ok = sum(1 for t in trials 
                             if t.get("comparison", {}).get("constraints_satisfied", False))
        
        accuracies = [t["comparison"]["action_accuracy"] for t in trials 
                      if t.get("comparison", {}).get("action_accuracy") is not None]
        
        # Execution metrics (if available)
        exec_successes = sum(1 for t in trials 
                             if t.get("execution", {}).get("success", False))
        
        self.results["summary"] = {
            "total_trials": n,
            "planning_success_rate": planning_successes / n,
            "exact_match_rate": exact_matches / n,
            "constraint_satisfaction_rate": constraints_ok / n,
            "mean_action_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "mean_planning_time": sum(planning_times) / len(planning_times) if planning_times else 0,
            "execution_success_rate": exec_successes / n if any(t.get("execution") for t in trials) else None
        }
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        for k, v in self.results["summary"].items():
            if v is not None:
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
    
    def save_results(self):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
        
        # Also save latest
        latest_path = os.path.join(RESULTS_DIR, "latest.json")
        with open(latest_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: VLM Planner Benchmark")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock VLM")
    parser.add_argument("--execute", action="store_true",
                        help="Run full execution (not just planning)")
    parser.add_argument("--env", action="store_true",
                        help="Use real RLBench environment")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless")
    
    args = parser.parse_args()
    
    experiment = Experiment4(
        use_mock=args.mock,
        use_env=args.env,
        headless=args.headless,
        num_trials=args.trials
    )
    
    experiment.run(full_execution=args.execute)


if __name__ == "__main__":
    main()
