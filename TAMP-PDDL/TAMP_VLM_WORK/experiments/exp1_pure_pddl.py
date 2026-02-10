"""
Experiment 1: Pure PDDLStream Baseline
======================================
No logical constraints, no ordering rules, completely static domain.
PDDLStream handles backtracking internally but learns nothing across episodes.

Task: 4-step kitchen manipulation
1. mug_box → placement_boundary
2. open box_lid
3. mug_inside_box → placement_boundary  
4. soup → cupboard_boundary
"""

import os
import sys
import numpy as np
import argparse
import time
import datetime
import json
import random

# --- CONFIGURATION ---
EXPERIMENT_NAME = "exp1_pure_pddl"

# Ground truth action sequence (expected order)
GROUND_TRUTH_ACTIONS = [
    'pick',      # pick mug_box (from box top)
    'place',     # place mug_box on placement_boundary
    'open-lid',  # open box_lid
    'pick',      # pick mug_inside_box
    'place',     # place mug_inside_box on placement_boundary
    'pick',      # pick soup
    'place',     # place soup in cupboard_boundary
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", EXPERIMENT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

def _configure_qt():
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    candidate_dirs = [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
        os.path.join(coppelia_root, "qt", "plugins", "platforms"),
    ]
    for candidate in candidate_dirs:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break

_configure_qt()

# Add paths
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', 'pddlstream'))

from pddlstream.language.constants import PDDLProblem, And
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

os.environ["HEADLESS"] = "False"
from rlbench_kitchen_streams import ENV, get_stream_map


def extract_action_sequence(plan):
    """Extract just action names from plan (excluding move/retreat)."""
    if not plan:
        return []
    return [a.name for a in plan if a.name not in ['move', 'retreat']]


def compare_to_ground_truth(plan):
    """Compare plan to ground truth, return similarity metrics."""
    actual = extract_action_sequence(plan)
    expected = GROUND_TRUTH_ACTIONS
    
    # Exact match
    exact_match = actual == expected
    
    # Action count match
    count_match = len(actual) == len(expected)
    
    # Sequence similarity (Levenshtein-like)
    matches = sum(1 for a, e in zip(actual, expected) if a == e)
    similarity = matches / max(len(actual), len(expected)) if actual else 0
    
    return {
        'exact_match': exact_match,
        'count_match': count_match,
        'similarity': similarity,
        'actual_sequence': actual,
        'expected_sequence': expected,
    }


class ExperimentLogger:
    """Structured logging for experiment results."""
    
    def __init__(self, experiment_name, results_dir, seed):
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.seed = seed
        self.episode_data = []
        
    def log_episode(self, episode_num, status, plan, planning_time, 
                    execution_time=0, failure_reason=""):
        """Log results of a single episode."""
        
        # Compare to ground truth
        gt_comparison = compare_to_ground_truth(plan) if plan else None
        
        # Full plan with all actions
        full_plan = [a.name for a in plan] if plan else []
        
        entry = {
            'seed': self.seed,
            'episode': episode_num,
            'timestamp': datetime.datetime.now().isoformat(),
            'status': status,
            'plan_full': full_plan,
            'plan_actions_only': gt_comparison['actual_sequence'] if gt_comparison else [],
            'plan_length': len(plan) if plan else 0,
            'planning_time_sec': planning_time,
            'execution_time_sec': execution_time,
            'total_time_sec': planning_time + execution_time,
            'failure_reason': failure_reason,
            'ground_truth_comparison': gt_comparison,
        }
        self.episode_data.append(entry)
        
        self._save_episode_log(episode_num, entry)
        self._save_plan(episode_num, plan)
        self._save_summary()
        
    def _save_episode_log(self, episode_num, entry):
        filepath = os.path.join(self.results_dir, f"seed_{self.seed:02d}_episode_{episode_num:03d}.json")
        with open(filepath, 'w') as f:
            json.dump(entry, f, indent=2)
    
    def _save_plan(self, episode_num, plan):
        """Save plan to separate text file for easy viewing."""
        filepath = os.path.join(self.results_dir, f"seed_{self.seed:02d}_episode_{episode_num:03d}_plan.txt")
        with open(filepath, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Episode: {episode_num}\n")
            f.write(f"="*50 + "\n\n")
            
            f.write("GROUND TRUTH:\n")
            for i, action in enumerate(GROUND_TRUTH_ACTIONS, 1):
                f.write(f"  {i}. {action}\n")
            
            f.write(f"\nGENERATED PLAN (full):\n")
            if plan:
                for i, action in enumerate(plan, 1):
                    f.write(f"  {i}. {action.name} {action.args[0] if action.args else ''}\n")
            else:
                f.write("  No plan found\n")
            
            f.write(f"\nGENERATED PLAN (actions only, no move/retreat):\n")
            if plan:
                actions_only = [a.name for a in plan if a.name not in ['move', 'retreat']]
                for i, action in enumerate(actions_only, 1):
                    f.write(f"  {i}. {action}\n")
            
            f.write(f"\nMATCH: {extract_action_sequence(plan) == GROUND_TRUTH_ACTIONS if plan else False}\n")
            
    def _save_summary(self):
        successes = sum(1 for e in self.episode_data if e['status'] == 'SUCCESS')
        gt_matches = sum(1 for e in self.episode_data 
                        if (e.get('ground_truth_comparison') or {}).get('exact_match', False))
        
        summary = {
            'experiment': self.experiment_name,
            'seed': self.seed,
            'total_episodes': len(self.episode_data),
            'successes': successes,
            'failures': len(self.episode_data) - successes,
            'success_rate': successes / len(self.episode_data) if self.episode_data else 0,
            'ground_truth_matches': gt_matches,
            'ground_truth_match_rate': gt_matches / len(self.episode_data) if self.episode_data else 0,
            'avg_planning_time': float(np.mean([e['planning_time_sec'] for e in self.episode_data])) if self.episode_data else 0,
            'avg_plan_length': float(np.mean([e['plan_length'] for e in self.episode_data if e['plan_length'] > 0])) if self.episode_data else 0,
            'ground_truth_sequence': GROUND_TRUTH_ACTIONS,
            'episodes': self.episode_data,
        }
        filepath = os.path.join(self.results_dir, f"seed_{self.seed:02d}_summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


# --- EXECUTION HELPERS ---
def execute_trajectory(env, traj):
    """Execute a trajectory with interpolation."""
    if not traj: 
        return
    full_traj = []
    for i in range(len(traj)-1):
        start = np.array(traj[i])
        end = np.array(traj[i+1])
        steps = 15 
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1-t)*start + t*end)
    full_traj.append(traj[-1])
    
    for conf in full_traj:
        env.set_robot_conf(conf)
        env.pr.step()


def validate_config(env, q):
    """Check if configuration is collision-free."""
    env.set_robot_conf(q)
    return not env.robot.check_collision()


def interpolate_path(env, q1, q2, steps=30):
    """Create interpolated path between two configurations."""
    traj = []
    q1 = np.array(q1)
    q2 = np.array(q2)
    for i in range(steps + 1):
        t = i / steps
        q = (1 - t) * q1 + t * q2
        q_list = q.tolist()
        if not validate_config(env, q_list):
            return None
        traj.append(q_list)
    return traj


def go_home(env):
    """Return robot to home configuration."""
    home_q = env.get_home_conf()
    current_q = env.get_robot_conf()
    traj = interpolate_path(env, current_q, home_q)
    if traj:
        execute_trajectory(env, traj)
        return True
    else:
        env.set_robot_conf(home_q)
        for _ in range(10): 
            env.pr.step()
        return False


def execute_plan(env, plan):
    """Execute a PDDLStream plan."""
    pr = env.pr
    
    for action in plan:
        print(f"  Executing: {action.name}")
        
        if action.name == 'move':
            q1, q2, traj = action.args
            execute_trajectory(env, traj)
        
        elif action.name == 'pick':
            o, p, g, q1, q2, traj_tuple = action.args
            approach, retreat = traj_tuple
            execute_trajectory(env, approach)
            env.get_object(o).set_dynamic(True)
            env.gripper.actuate(0.0, 0.1)
            for _ in range(20): 
                pr.step()
            env.gripper.grasp(env.get_object(o))
            execute_trajectory(env, retreat)

        elif action.name == 'place':
            o, p, g, r, q1, q2, traj_tuple = action.args
            # Stream returns (t_down, t_up, t_home) - 3 trajectories
            if len(traj_tuple) == 3:
                lower, lift, home_traj = traj_tuple
            else:
                lower, lift = traj_tuple
                home_traj = None
            execute_trajectory(env, lower)
            env.gripper.release()
            env.get_object(o).set_dynamic(True)
            env.gripper.actuate(1.0, 0.1)
            for _ in range(30): 
                pr.step()
            execute_trajectory(env, lift)
            if home_traj:
                execute_trajectory(env, home_traj)

        elif action.name == 'open-lid':
            o, g, q1, q2, traj_tuple = action.args
            app, slide, ret, esc = traj_tuple
            execute_trajectory(env, app)
            env.gripper.actuate(0.0, 0.1)
            for _ in range(20): 
                pr.step()
            env.gripper.grasp(env.get_object(o))
            execute_trajectory(env, slide)
            env.gripper.release()
            env.gripper.actuate(1.0, 0.1)
            for _ in range(30): 
                pr.step()
            execute_trajectory(env, ret)
            execute_trajectory(env, esc)
        
        elif action.name == 'retreat':
            o, q1, q2, traj = action.args
            execute_trajectory(env, traj)
    
    go_home(env)


def run_episode(episode_num, args, logger):
    """Run a single episode of pure PDDLStream planning."""
    
    env = ENV
    pr = env.pr
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Pure PDDLStream - Episode {episode_num}")
    print(f"{'='*60}")
    
    # --- SETUP ---
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    env.gripper.release()
    
    # Object names
    mug_name = 'mug_box'
    lid_name = 'box_lid'
    mug_inside_name = 'mug_inside_box'
    cupboard_obj_name = args.object
    
    placement_region = 'placement_boundary'
    cupboard_region = 'cupboard_boundary'

    # Get objects
    mug_obj = env.get_object(mug_name)
    lid_obj = env.get_object(lid_name)
    mug_inside_obj = env.get_object(mug_inside_name)
    cupboard_obj = env.get_object(cupboard_obj_name)
    
    if not all([mug_obj, lid_obj, mug_inside_obj, cupboard_obj]):
        logger.log_episode(episode_num, "FAILURE", None, 0, 
                          failure_reason="Objects not found in scene")
        return False

    # Freeze objects for planning
    mug_obj.set_dynamic(False)
    mug_inside_obj.set_dynamic(False)
    cupboard_obj.set_dynamic(False)
    
    # Get poses
    mug_pose = tuple(mug_obj.get_pose())
    mug_inside_pose = tuple(mug_inside_obj.get_pose())
    cupboard_obj_pose = tuple(cupboard_obj.get_pose())
    current_q = tuple(env.get_robot_conf())

    # --- PDDL PROBLEM ---
    project_dir = os.path.join(BASE_DIR, '..')
    domain_pddl = read(os.path.join(project_dir, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(project_dir, 'pddl/rlbench_kitchen_streams.pddl'))

    init = [
        ('conf', current_q),
        ('at-conf', current_q),
        ('hand-empty',),
        
        ('movable', mug_name),
        ('pose', mug_pose),
        ('at-pose', mug_name, mug_pose),
        
        ('movable', mug_inside_name),
        ('pose', mug_inside_pose),
        ('at-pose', mug_inside_name, mug_inside_pose),
        
        ('movable', cupboard_obj_name),
        ('pose', cupboard_obj_pose),
        ('at-pose', cupboard_obj_name, cupboard_obj_pose),
        
        ('lid', lid_name),
        ('closed', lid_name),
        
        ('region', placement_region),
        ('region', cupboard_region),
        ('region', 'box-top'),
        
        ('in-region', mug_name, 'box-top'),
    ]

    goal = And(
        ('in-region', mug_name, placement_region),
        ('opened', lid_name),
        ('in-region', mug_inside_name, placement_region),
        ('in-region', cupboard_obj_name, cupboard_region),
        ('hand-empty',)
    )

    # --- PLANNING ---
    print("\n  [Planning] Pure PDDLStream with static domain...")
    start_time = time.time()
    
    env.set_target_region(placement_region)
    
    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    solution = solve(problem, algorithm='adaptive', verbose=True, max_time=180)
    plan, cost, evaluations = solution
    planning_time = time.time() - start_time

    if plan is None:
        print(f"  [FAILURE] No plan found in {planning_time:.2f}s")
        logger.log_episode(episode_num, "FAILURE", None, planning_time,
                          failure_reason="No plan found")
        return False

    print(f"\n  [SUCCESS] Plan found in {planning_time:.2f}s")
    print(f"  Plan: {' -> '.join([a.name for a in plan])}")
    
    # --- EXECUTION ---
    print("\n  [Executing plan...]")
    exec_start = time.time()
    
    try:
        execute_plan(env, plan)
        execution_time = time.time() - exec_start
        
        logger.log_episode(episode_num, "SUCCESS", plan, planning_time, 
                          execution_time=execution_time)
        print(f"  [DONE] Execution completed in {execution_time:.2f}s")
        return True
        
    except Exception as e:
        execution_time = time.time() - exec_start
        logger.log_episode(episode_num, "FAILURE", plan, planning_time,
                          execution_time=execution_time,
                          failure_reason=f"Execution error: {str(e)}")
        print(f"  [FAILURE] Execution error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Pure PDDLStream Baseline")
    parser.add_argument('--object', type=str, default='soup', 
                       help='Object to place in cupboard')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run')
    parser.add_argument('--start_episode', type=int, default=1,
                       help='Starting episode number')
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger = ExperimentLogger(EXPERIMENT_NAME, RESULTS_DIR, args.seed)
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT 1: Pure PDDLStream (No Constraints)")
    print(f"# Seed: {args.seed}")
    print(f"# Episodes: {args.start_episode} to {args.start_episode + args.episodes - 1}")
    print(f"# Results: {RESULTS_DIR}")
    print(f"{'#'*60}")
    print(f"\nGround Truth: {' -> '.join(GROUND_TRUTH_ACTIONS)}")
    
    successes = 0
    for i in range(args.episodes):
        episode_num = args.start_episode + i
        if run_episode(episode_num, args, logger):
            successes += 1
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1 COMPLETE: {successes}/{args.episodes} episodes succeeded")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")

    ENV.pr.stop()
    ENV.pr.shutdown()


if __name__ == "__main__":
    main()
