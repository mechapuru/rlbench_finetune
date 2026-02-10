"""
Compare Results from all seeds across Experiment 1 and Experiment 2
===================================================================
Aggregates results from all seed runs and generates comparison report.
"""

import os
import json
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

GROUND_TRUTH_ACTIONS = [
    'pick', 'place', 'open-lid', 'pick', 'place', 'pick', 'place'
]


def load_all_summaries(experiment_name):
    """Load all seed summaries for an experiment."""
    pattern = os.path.join(RESULTS_DIR, experiment_name, "seed_*_summary.json")
    summaries = []
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath) as f:
            summaries.append(json.load(f))
    return summaries


def aggregate_results(summaries):
    """Aggregate results across all seeds."""
    if not summaries:
        return None
    
    total_episodes = sum(s['total_episodes'] for s in summaries)
    total_successes = sum(s['successes'] for s in summaries)
    total_gt_matches = sum(s.get('ground_truth_matches', 0) for s in summaries)
    
    planning_times = [s['avg_planning_time'] for s in summaries if s['avg_planning_time'] > 0]
    plan_lengths = [s.get('avg_plan_length', 0) for s in summaries if s.get('avg_plan_length', 0) > 0]
    
    return {
        'num_seeds': len(summaries),
        'total_episodes': total_episodes,
        'total_successes': total_successes,
        'total_failures': total_episodes - total_successes,
        'success_rate': total_successes / total_episodes if total_episodes > 0 else 0,
        'ground_truth_matches': total_gt_matches,
        'ground_truth_match_rate': total_gt_matches / total_episodes if total_episodes > 0 else 0,
        'avg_planning_time': np.mean(planning_times) if planning_times else 0,
        'std_planning_time': np.std(planning_times) if planning_times else 0,
        'avg_plan_length': np.mean(plan_lengths) if plan_lengths else 0,
    }


def generate_comparison():
    """Generate comparison report between experiments."""
    
    exp1_summaries = load_all_summaries("exp1_pure_pddl")
    exp2_summaries = load_all_summaries("exp2_coast")
    
    exp1 = aggregate_results(exp1_summaries)
    exp2 = aggregate_results(exp2_summaries)
    
    if not exp1 and not exp2:
        print("No experiment results found. Run experiments first.")
        return
    
    report = []
    report.append("=" * 80)
    report.append("BENCHMARK COMPARISON REPORT: PDDLStream vs COAST")
    report.append("=" * 80)
    report.append("")
    report.append(f"Ground Truth Action Sequence: {' -> '.join(GROUND_TRUTH_ACTIONS)}")
    report.append("")
    
    # Header
    report.append(f"{'Metric':<40} {'Exp1 (Pure)':<18} {'Exp2 (COAST)':<18}")
    report.append("-" * 80)
    
    def fmt(val):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)
    
    # Metrics
    metrics = [
        ('Seeds Run', 'num_seeds'),
        ('Total Episodes', 'total_episodes'),
        ('Successes', 'total_successes'),
        ('Failures', 'total_failures'),
        ('Success Rate', 'success_rate'),
        ('Ground Truth Matches', 'ground_truth_matches'),
        ('GT Match Rate', 'ground_truth_match_rate'),
        ('Avg Planning Time (s)', 'avg_planning_time'),
        ('Std Planning Time (s)', 'std_planning_time'),
        ('Avg Plan Length', 'avg_plan_length'),
    ]
    
    for label, key in metrics:
        v1 = exp1.get(key) if exp1 else None
        v2 = exp2.get(key) if exp2 else None
        report.append(f"{label:<40} {fmt(v1):<18} {fmt(v2):<18}")
    
    # COAST-specific
    if exp2_summaries:
        report.append("")
        report.append("-" * 80)
        report.append("COAST-Specific Metrics:")
        avg_constraints = np.mean([s.get('avg_constraints', 0) for s in exp2_summaries])
        avg_replanning = np.mean([s.get('avg_replanning', 0) for s in exp2_summaries])
        report.append(f"  Avg Constraints Used: {avg_constraints:.2f}")
        report.append(f"  Avg Replanning Cycles: {avg_replanning:.2f}")
    
    report.append("")
    report.append("=" * 80)
    report.append("")
    
    # Per-seed breakdown
    report.append("PER-SEED BREAKDOWN:")
    report.append("-" * 80)
    report.append(f"{'Seed':<8} {'Exp1 Status':<15} {'Exp1 GT Match':<15} {'Exp2 Status':<15} {'Exp2 GT Match':<15}")
    report.append("-" * 80)
    
    max_seeds = max(len(exp1_summaries) if exp1_summaries else 0, 
                    len(exp2_summaries) if exp2_summaries else 0)
    
    for i in range(max_seeds):
        e1 = exp1_summaries[i] if exp1_summaries and i < len(exp1_summaries) else None
        e2 = exp2_summaries[i] if exp2_summaries and i < len(exp2_summaries) else None
        
        e1_status = f"{e1['successes']}/{e1['total_episodes']}" if e1 else "N/A"
        e1_gt = f"{e1.get('ground_truth_matches', 0)}/{e1['total_episodes']}" if e1 else "N/A"
        e2_status = f"{e2['successes']}/{e2['total_episodes']}" if e2 else "N/A"
        e2_gt = f"{e2.get('ground_truth_matches', 0)}/{e2['total_episodes']}" if e2 else "N/A"
        
        seed = e1.get('seed', i) if e1 else (e2.get('seed', i) if e2 else i)
        report.append(f"{seed:<8} {e1_status:<15} {e1_gt:<15} {e2_status:<15} {e2_gt:<15}")
    
    report.append("")
    report.append("=" * 80)
    
    # Print and save
    report_text = "\n".join(report)
    print(report_text)
    
    report_path = os.path.join(RESULTS_DIR, "benchmark_comparison.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    
    # Save as JSON too
    json_report = {
        'ground_truth': GROUND_TRUTH_ACTIONS,
        'exp1_pure_pddl': exp1,
        'exp2_coast': exp2,
        'exp1_per_seed': exp1_summaries,
        'exp2_per_seed': exp2_summaries,
    }
    json_path = os.path.join(RESULTS_DIR, "benchmark_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    generate_comparison()
