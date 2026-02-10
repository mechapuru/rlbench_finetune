"""
Experiment 2: COAST - COnstraints And STreams
=============================================
Automatic constraint learning from stream failures.
When a stream exhaustively fails, we inject failure predicates 
into the PDDL domain to prevent the planner from proposing that action again.
"""

import os
import sys
import numpy as np
import argparse
import time
import datetime
import json
import random
from pathlib import Path

# --- CONFIGURATION ---
EXPERIMENT_NAME = "exp2_coast"

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


def extract_action_sequence(plan):
    """Extract just action names from plan (excluding move/retreat)."""
    if not plan:
        return []
    return [a.name for a in plan if a.name not in ['move', 'retreat']]


def compare_to_ground_truth(plan):
    """Compare plan to ground truth, return similarity metrics."""
    actual = extract_action_sequence(plan)
    expected = GROUND_TRUTH_ACTIONS
    
    exact_match = actual == expected
    count_match = len(actual) == len(expected)
    matches = sum(1 for a, e in zip(actual, expected) if a == e)
    similarity = matches / max(len(actual), len(expected)) if actual else 0
    
    return {
        'exact_match': exact_match,
        'count_match': count_match,
        'similarity': similarity,
        'actual_sequence': actual,
        'expected_sequence': expected,
    }
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

sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', 'pddlstream'))

from pddlstream.language.constants import PDDLProblem, And
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

os.environ["HEADLESS"] = "False"
from rlbench_kitchen_streams import ENV, get_stream_map


# =============================================================================
# COAST: Constraint Learning System
# =============================================================================

class FailureConstraint:
    """Represents a learned constraint from stream failure."""
    
    def __init__(self, stream_name, inputs, failure_type, timestamp):
        self.stream_name = stream_name
        self.inputs = inputs
        self.failure_type = failure_type
        self.timestamp = timestamp
        self.attempts = 1
        
    def __repr__(self):
        return f"FailureConstraint({self.stream_name}, {self.inputs}, attempts={self.attempts})"


class COASTConstraintManager:
    """Manages learned constraints from stream failures."""
    
    def __init__(self, base_domain_path, results_dir):
        self.base_domain_path = base_domain_path
        self.results_dir = results_dir
        self.constraints = []
        self.constraint_history = []
        self.domain_version = 0
        
    def add_failure_constraint(self, stream_name, inputs, failure_type='exhausted'):
        """Record a stream failure as a constraint."""
        constraint = FailureConstraint(
            stream_name=stream_name,
            inputs=inputs,
            failure_type=failure_type,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        for existing in self.constraints:
            if existing.stream_name == stream_name and existing.inputs == inputs:
                existing.attempts += 1
                return existing
        
        self.constraints.append(constraint)
        self.constraint_history.append(constraint)
        print(f"  [COAST] New constraint: {constraint}")
        return constraint
    
    def generate_constrained_domain(self):
        """Generate PDDL domain with failure predicates."""
        self.domain_version += 1
        domain_text = Path(self.base_domain_path).read_text()
        
        # Inject failure predicates
        failure_predicates = self._generate_failure_predicates()
        if failure_predicates:
            domain_text = self._inject_predicates(domain_text, failure_predicates)
            domain_text = self._inject_action_preconditions(domain_text)
        
        output_path = os.path.join(self.results_dir, f"domain_v{self.domain_version:03d}.pddl")
        Path(output_path).write_text(domain_text)
        
        print(f"  [COAST] Generated domain v{self.domain_version}, {len(self.constraints)} constraints")
        return output_path, domain_text
    
    def _generate_failure_predicates(self):
        predicates = set()
        for c in self.constraints:
            if c.stream_name == 'sample-pick-kin':
                predicates.add("(fail-pick ?o)")
            elif c.stream_name == 'sample-place-kin':
                predicates.add("(fail-place ?o ?r)")
            elif c.stream_name == 'sample-open-lid':
                predicates.add("(fail-open-lid ?l)")
        return list(predicates)
    
    def _inject_predicates(self, domain_text, predicates):
        pred_marker = "(:predicates"
        idx = domain_text.find(pred_marker)
        if idx == -1:
            return domain_text
        insertion_point = idx + len(pred_marker)
        pred_str = "\n      ;; COAST failure predicates\n      " + "\n      ".join(predicates)
        return domain_text[:insertion_point] + pred_str + domain_text[insertion_point:]
    
    def _inject_action_preconditions(self, domain_text):
        pick_constraints = [c for c in self.constraints if c.stream_name == 'sample-pick-kin']
        if pick_constraints:
            domain_text = self._inject_pick_precondition(domain_text)
        
        place_constraints = [c for c in self.constraints if c.stream_name == 'sample-place-kin']
        if place_constraints:
            domain_text = self._inject_place_precondition(domain_text)
            
        return domain_text
    
    def _inject_pick_precondition(self, domain_text):
        action_start = domain_text.find("(:action pick")
        if action_start == -1:
            return domain_text
        precond_idx = domain_text.find(":precondition", action_start)
        and_idx = domain_text.find("(and", precond_idx)
        if and_idx == -1:
            return domain_text
        insertion = and_idx + 4
        new_cond = "\n        (not (fail-pick ?o))"
        return domain_text[:insertion] + new_cond + domain_text[insertion:]
    
    def _inject_place_precondition(self, domain_text):
        action_start = domain_text.find("(:action place")
        if action_start == -1:
            return domain_text
        precond_idx = domain_text.find(":precondition", action_start)
        and_idx = domain_text.find("(and", precond_idx)
        if and_idx == -1:
            return domain_text
        insertion = and_idx + 4
        new_cond = "\n        (not (fail-place ?o ?r))"
        return domain_text[:insertion] + new_cond + domain_text[insertion:]
    
    def get_init_atoms_for_constraints(self):
        atoms = []
        for c in self.constraints:
            if c.stream_name == 'sample-pick-kin':
                obj = c.inputs.get('object')
                if obj:
                    atoms.append(('fail-pick', obj))
            elif c.stream_name == 'sample-place-kin':
                obj = c.inputs.get('object')
                region = c.inputs.get('region')
                if obj and region:
                    atoms.append(('fail-place', obj, region))
        return atoms
    
    def clear_constraints(self):
        self.constraints = []
        
    def save_constraint_log(self):
        log = {
            'total_constraints_learned': len(self.constraint_history),
            'active_constraints': len(self.constraints),
            'domain_versions': self.domain_version,
            'constraints': [
                {
                    'stream': c.stream_name,
                    'inputs': {k: str(v) for k, v in c.inputs.items()},
                    'type': c.failure_type,
                    'timestamp': c.timestamp,
                    'attempts': c.attempts
                }
                for c in self.constraint_history
            ]
        }
        filepath = os.path.join(self.results_dir, "constraint_log.json")
        with open(filepath, 'w') as f:
            json.dump(log, f, indent=2)


class COASTStreamWrapper:
    """Wraps raw generator functions to track failures before from_gen_fn."""
    
    def __init__(self, constraint_manager, max_attempts=1):
        self.constraint_manager = constraint_manager
        self.max_attempts = max_attempts  # 1 = single failure triggers constraint (PDDLStream doesn't retry)
        self.attempt_counts = {}
        self.failure_counts = {}  # Track consecutive failures per object
        
    def get_wrapped_stream_map(self):
        """Create stream map with wrapped generator functions."""
        from pddlstream.language.generator import from_gen_fn
        from rlbench_kitchen_streams import (
            fn_sample_stable_pose, fn_sample_pick_kin, fn_sample_place_kin,
            fn_sample_motion, fn_sample_open_lid
        )
        
        return {
            'sample-stable-pose': from_gen_fn(fn_sample_stable_pose),
            'sample-pick-kin':    from_gen_fn(self._wrap_gen('sample-pick-kin', fn_sample_pick_kin)),
            'sample-place-kin':   from_gen_fn(self._wrap_gen('sample-place-kin', fn_sample_place_kin)),
            'sample-motion':      from_gen_fn(fn_sample_motion),
            'sample-open-lid':    from_gen_fn(fn_sample_open_lid),
        }
    
    def _wrap_gen(self, stream_name, gen_fn):
        """Wrap a generator function to track its outputs."""
        wrapper = self
        
        def wrapped_gen(*args):
            obj_key = args[0] if args else 'unknown'
            input_key = (stream_name, str(obj_key))
            
            # Track call
            if input_key not in wrapper.attempt_counts:
                wrapper.attempt_counts[input_key] = 0
            wrapper.attempt_counts[input_key] += 1
            
            # Call original generator
            gen = gen_fn(*args)
            if gen is None:
                # Generator returned None directly (some functions do this)
                wrapper._record_failure(stream_name, obj_key, args)
                return
            
            # Iterate through generator and track results
            yielded_anything = False
            for result in gen:
                yielded_anything = True
                # Reset failure count on success
                if input_key in wrapper.failure_counts:
                    wrapper.failure_counts[input_key] = 0
                yield result
            
            if not yielded_anything:
                # Generator yielded nothing = failure
                wrapper._record_failure(stream_name, obj_key, args)
        
        return wrapped_gen
    
    def _record_failure(self, stream_name, obj_key, args):
        input_key = (stream_name, str(obj_key))
        
        if input_key not in self.failure_counts:
            self.failure_counts[input_key] = 0
        self.failure_counts[input_key] += 1
        
        print(f"  [COAST] Stream FAILED: {stream_name}({obj_key}) - failure #{self.failure_counts[input_key]}")
        
        if self.failure_counts[input_key] >= self.max_attempts:
            print(f"  [COAST] >>> EXHAUSTED: {stream_name} for {obj_key}")
            self._report_constraint(stream_name, obj_key, args)
    
    def _report_constraint(self, stream_name, obj_key, args):
        inputs = {'object': obj_key}
        if stream_name == 'sample-place-kin' and len(args) >= 3:
            inputs['region'] = args[2]
        self.constraint_manager.add_failure_constraint(stream_name, inputs, 'exhausted')
    
    def reset_counts(self):
        self.attempt_counts = {}
        self.failure_counts = {}


class ExperimentLogger:
    def __init__(self, experiment_name, results_dir, seed):
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.seed = seed
        self.episode_data = []
        
    def log_episode(self, episode_num, status, plan, planning_time, 
                    execution_time=0, failure_reason="", 
                    constraints_used=0, replanning_cycles=0):
        
        gt_comparison = compare_to_ground_truth(plan) if plan else None
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
            'constraints_used': constraints_used,
            'replanning_cycles': replanning_cycles,
            'ground_truth_comparison': gt_comparison,
        }
        self.episode_data.append(entry)
        self._save_episode_log(episode_num, entry)
        self._save_plan(episode_num, plan, constraints_used, replanning_cycles)
        self._save_summary()
        
    def _save_episode_log(self, episode_num, entry):
        filepath = os.path.join(self.results_dir, f"seed_{self.seed:02d}_episode_{episode_num:03d}.json")
        with open(filepath, 'w') as f:
            json.dump(entry, f, indent=2)
    
    def _save_plan(self, episode_num, plan, constraints_used, replanning_cycles):
        """Save plan to separate text file."""
        filepath = os.path.join(self.results_dir, f"seed_{self.seed:02d}_episode_{episode_num:03d}_plan.txt")
        with open(filepath, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Episode: {episode_num}\n")
            f.write(f"Constraints Used: {constraints_used}\n")
            f.write(f"Replanning Cycles: {replanning_cycles}\n")
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
            'avg_constraints': float(np.mean([e['constraints_used'] for e in self.episode_data])) if self.episode_data else 0,
            'avg_replanning': float(np.mean([e['replanning_cycles'] for e in self.episode_data])) if self.episode_data else 0,
            'ground_truth_sequence': GROUND_TRUTH_ACTIONS,
            'episodes': self.episode_data,
        }
        filepath = os.path.join(self.results_dir, f"seed_{self.seed:02d}_summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


# --- EXECUTION HELPERS ---
def execute_trajectory(env, traj):
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
    env.set_robot_conf(q)
    return not env.robot.check_collision()


def interpolate_path(env, q1, q2, steps=30):
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


def run_episode_with_coast(episode_num, args, logger, constraint_manager):
    """Run episode with COAST constraint learning."""
    
    env = ENV
    pr = env.pr
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: COAST - Episode {episode_num}")
    print(f"{'='*60}")
    
    # --- SETUP ---
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    env.gripper.release()
    
    mug_name = 'mug_box'
    lid_name = 'box_lid'
    mug_inside_name = 'mug_inside_box'
    cupboard_obj_name = args.object
    
    placement_region = 'placement_boundary'
    cupboard_region = 'cupboard_boundary'

    mug_obj = env.get_object(mug_name)
    lid_obj = env.get_object(lid_name)
    mug_inside_obj = env.get_object(mug_inside_name)
    cupboard_obj = env.get_object(cupboard_obj_name)
    
    if not all([mug_obj, lid_obj, mug_inside_obj, cupboard_obj]):
        logger.log_episode(episode_num, "FAILURE", None, 0, failure_reason="Objects not found")
        return False

    mug_obj.set_dynamic(False)
    mug_inside_obj.set_dynamic(False)
    cupboard_obj.set_dynamic(False)
    
    mug_pose = tuple(mug_obj.get_pose())
    mug_inside_pose = tuple(mug_inside_obj.get_pose())
    cupboard_obj_pose = tuple(cupboard_obj.get_pose())
    current_q = tuple(env.get_robot_conf())

    # --- COAST PLANNING LOOP ---
    project_dir = os.path.join(BASE_DIR, '..')
    base_domain_path = os.path.join(project_dir, 'pddl/rlbench_kitchen_domain.pddl')
    stream_pddl = read(os.path.join(project_dir, 'pddl/rlbench_kitchen_streams.pddl'))
    
    max_replanning_cycles = 5
    replanning_cycle = 0
    total_planning_time = 0
    plan = None
    
    # New interface: no stream_map arg, wraps raw generators internally
    # max_attempts=1 because PDDLStream doesn't retry failed streams
    stream_wrapper = COASTStreamWrapper(constraint_manager, max_attempts=1)
    
    while replanning_cycle < max_replanning_cycles:
        replanning_cycle += 1
        print(f"\n  [COAST] Planning cycle {replanning_cycle}/{max_replanning_cycles}")
        
        if constraint_manager.constraints:
            domain_path, domain_text = constraint_manager.generate_constrained_domain()
            domain_pddl = domain_text
        else:
            domain_pddl = read(base_domain_path)
        
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
        
        failure_atoms = constraint_manager.get_init_atoms_for_constraints()
        init.extend(failure_atoms)

        goal = And(
            ('in-region', mug_name, placement_region),
            ('opened', lid_name),
            ('in-region', mug_inside_name, placement_region),
            ('in-region', cupboard_obj_name, cupboard_region),
            ('hand-empty',)
        )

        env.set_target_region(placement_region)
        stream_wrapper.reset_counts()
        
        problem = PDDLProblem(
            domain_pddl=domain_pddl,
            constant_map={},
            stream_pddl=stream_pddl,
            stream_map=stream_wrapper.get_wrapped_stream_map(),
            init=init,
            goal=goal,
        )

        print(f"  [Planning...]")
        start_time = time.time()
        solution = solve(problem, algorithm='adaptive', verbose=True, max_time=120)
        plan, cost, evaluations = solution
        cycle_time = time.time() - start_time
        total_planning_time += cycle_time

        if plan is not None:
            print(f"  [SUCCESS] Plan found in cycle {replanning_cycle}")
            break
        else:
            print(f"  [COAST] No plan. Constraints: {len(constraint_manager.constraints)}")
            if len(constraint_manager.constraints) == 0:
                break

    if plan is None:
        logger.log_episode(episode_num, "FAILURE", None, total_planning_time,
            failure_reason="No plan after COAST replanning",
            constraints_used=len(constraint_manager.constraints),
            replanning_cycles=replanning_cycle)
        constraint_manager.save_constraint_log()
        return False

    print(f"\n  [COAST] Final Plan: {' -> '.join([a.name for a in plan])}")
    
    # --- EXECUTION ---
    print("\n  [Executing plan...]")
    exec_start = time.time()
    
    try:
        execute_plan(env, plan)
        execution_time = time.time() - exec_start
        
        logger.log_episode(episode_num, "SUCCESS", plan, total_planning_time,
            execution_time=execution_time,
            constraints_used=len(constraint_manager.constraints),
            replanning_cycles=replanning_cycle)
        constraint_manager.save_constraint_log()
        return True
        
    except Exception as e:
        execution_time = time.time() - exec_start
        logger.log_episode(episode_num, "FAILURE", plan, total_planning_time,
            execution_time=execution_time,
            failure_reason=f"Execution error: {str(e)}",
            constraints_used=len(constraint_manager.constraints),
            replanning_cycles=replanning_cycle)
        constraint_manager.save_constraint_log()
        return False


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: COAST")
    parser.add_argument('--object', type=str, default='soup')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--start_episode', type=int, default=1)
    parser.add_argument('--persist_constraints', action='store_true')
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger = ExperimentLogger(EXPERIMENT_NAME, RESULTS_DIR, args.seed)
    
    project_dir = os.path.join(BASE_DIR, '..')
    base_domain_path = os.path.join(project_dir, 'pddl/rlbench_kitchen_domain.pddl')
    constraint_manager = COASTConstraintManager(base_domain_path, RESULTS_DIR)
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT 2: COAST (Constraint Learning)")
    print(f"# Seed: {args.seed}")
    print(f"# Episodes: {args.start_episode} to {args.start_episode + args.episodes - 1}")
    print(f"# Persist constraints: {args.persist_constraints}")
    print(f"# Results: {RESULTS_DIR}")
    print(f"{'#'*60}")
    print(f"\nGround Truth: {' -> '.join(GROUND_TRUTH_ACTIONS)}")
    
    successes = 0
    for i in range(args.episodes):
        episode_num = args.start_episode + i
        if not args.persist_constraints:
            constraint_manager.clear_constraints()
        if run_episode_with_coast(episode_num, args, logger, constraint_manager):
            successes += 1
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2 COMPLETE (Seed {args.seed}): {successes}/{args.episodes} succeeded")
    print(f"Total constraints learned: {len(constraint_manager.constraint_history)}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*60}")

    ENV.pr.stop()
    ENV.pr.shutdown()


if __name__ == "__main__":
    main()
