"""
VLM Executor (Module 3)
=======================
Executes VLM-generated action skeletons by orchestrating robust
Task and Motion Planning (TAMP) scripts.

Refactored to match `orchestrator.py` logic:
- Uses `generalize_pick_place_gui.py` for standard pick/place.
- Uses `generalize_PP_box_gui.py` for constrained box picking.
- Uses `script_3_open_grasp_open_box_gui.py` for lid opening.
- Implements robust "Home" return with collision avoidance/escape.
"""

import os
import sys
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import VLM pipeline modules
from vlm_pipeline.vlm_planner import ActionSkeleton, PlanResult

# Import Task Scripts
# We need to make sure these are in the path. The VLM pipeline is in vlm_pipeline/, 
# but the scripts are in the root. The parent dir is already inserted above.
try:
    from generalize_pick_place_gui import run_task as run_pick_place
    from script_3_open_grasp_open_box_gui import run_task as run_open_box
    from generalize_PP_box_gui import run_task as run_pick_place_box
except ImportError:
    print("Warning: Could not import task scripts. Execution will fail.")

class ExecutionStatus(Enum):
    """Status of action execution."""
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    COLLISION = "collision"
    IK_FAILURE = "ik_failure"
    UNKNOWN_OBJECT = "unknown_object"


@dataclass
class ExecutionResult:
    """Result of executing an action or plan."""
    status: ExecutionStatus
    completed_actions: int
    total_actions: int
    current_action: Optional[ActionSkeleton] = None
    error_message: Optional[str] = None
    trajectory_executed: bool = False
    
    # For replanning
    failure_reason: Optional[str] = None
    blocking_object: Optional[str] = None


@dataclass
class ExecutorConfig:
    """Configuration for the executor."""
    interpolation_steps: int = 15
    min_collision_distance: float = 0.10
    escape_lift_heights: List[float] = field(default_factory=lambda: [0.25, 0.15, 0.10])
    pause_between_actions: float = 0.5  # seconds
    enable_unknown_object_detection: bool = True


class VLMExecutor:
    """
    Executes VLM action skeletons through robust TAMP scripts.
    """
    
    def __init__(self, 
                 env=None,
                 config: ExecutorConfig = None,
                 unknown_object_callback: Callable = None):
        self.env = env
        self.config = config or ExecutorConfig()
        self.unknown_object_callback = unknown_object_callback
        
        # Execution state
        self.current_plan: List[ActionSkeleton] = []
        self.executed_actions: List[ActionSkeleton] = []
        self.current_action_idx: int = 0
        self.is_paused: bool = False
        self.is_interrupted: bool = False
        
    def set_env(self, env):
        """Set the environment instance."""
        self.env = env
        
    def reset(self):
        """Reset executor state."""
        self.current_plan = []
        self.current_action_idx = 0
        self.is_paused = False
        self.is_interrupted = False
        self.executed_actions = []
        
    # =========================================================================
    # HELPER FUNCTIONS (From orchestrator.py)
    # =========================================================================
    
    def _execute_raw_trajectory(self, traj: List[List[float]]):
        """Low-level trajectory execution with interpolation (from orchestrator.py)."""
        if not traj or not self.env: return
        full_traj = []
        for i in range(len(traj)-1):
            start = np.array(traj[i])
            end = np.array(traj[i+1])
            steps = self.config.interpolation_steps 
            for t in np.linspace(0, 1, steps, endpoint=False):
                full_traj.append((1-t)*start + t*end)
        full_traj.append(traj[-1])
        
        for conf in full_traj:
            self.env.set_robot_conf(conf)
            self.env.pr.step()

    def _validate_config(self, q: List[float], min_dist: float = 0.10) -> bool:
        """Check collision and cupboard distance."""
        self.env.set_robot_conf(q)
        if self.env.robot.check_collision():
            return False
        if hasattr(self.env, 'cupboard'):
            # Check distance to cupboard
            if self.env.robot.check_distance(self.env.cupboard) < min_dist:
                return False
        return True

    def _interpolate_path(self, q1, q2, steps=50):
        """Linear configuration space interpolation."""
        traj = []
        q1 = np.array(q1)
        q2 = np.array(q2)
        for i in range(steps + 1):
            t = i / steps
            q = (1 - t) * q1 + t * q2
            q_list = q.tolist()
            if not self._validate_config(q_list):
                return None
            traj.append(q_list)
        return traj

    def go_home(self) -> bool:
        """
        Robust return to home configuration.
        Implements orchestrator.py's Safe Home logic with escape maneuvers.
        """
        if not self.env: return False
        
        print("\n  [Executor] Returning to Home...")
        q_start = self.env.get_robot_conf()
        q_home = self.env.get_home_conf()
        
        # 1. Try Direct Interpolation
        traj = self._interpolate_path(q_start, q_home)
        if traj:
            self._execute_raw_trajectory(traj)
            print("  [Executor] Reached Home (Direct).")
            return True

        # 2. Escape Maneuver
        print("  [Executor] Direct retreat blocked. Attempting evasive maneuver...")
        curr_pos = self.env.robot.get_position()
        curr_quat = self.env.robot.get_quaternion()
        
        # Try moving UP first (safest for cupboard), then BACK
        escape_offsets = [[0, 0, 0.25], [0, 0, 0.15], [-0.1, 0, 0.1]]
        
        for off in escape_offsets:
            target_pos = [curr_pos[0]+off[0], curr_pos[1]+off[1], curr_pos[2]+off[2]]
            try:
                path = self.env.robot.get_linear_path(
                    position=target_pos, quaternion=curr_quat, 
                    steps=20, ignore_collisions=True
                )
                if path:
                    escape_traj = path._path_points.reshape(-1, 7).tolist()
                    
                    # Validate escape path
                    valid_escape = True
                    for q in escape_traj:
                        # Slightly relaxed dist for escape
                        if not self._validate_config(q, min_dist=0.05):
                            valid_escape = False
                            break
                    
                    if valid_escape:
                        self._execute_raw_trajectory(escape_traj)
                        # Try home from here
                        q_new = escape_traj[-1]
                        traj_home = self._interpolate_path(q_new, q_home)
                        if traj_home:
                            self._execute_raw_trajectory(traj_home)
                            print("  [Executor] Reached Home (Escaped).")
                            return True
            except:
                continue

        # 3. Forced Home (Last Resort)
        print("  [Executor] WARNING: Safe retreat failed. Forcing home.")
        self.env.set_robot_conf(q_home)
        for _ in range(10): self.env.pr.step()
        return True

    # =========================================================================
    # PLAN EXECUTION
    # =========================================================================
    
    def execute_plan(self, skeleton: List[ActionSkeleton]) -> ExecutionResult:
        """
        Execute the action skeleton using specific Task Scripts.
        """
        self.current_plan = skeleton
        self.current_action_idx = 0
        self.is_interrupted = False
        self.executed_actions = []
        
        print(f"\n[Executor] Starting plan execution ({len(skeleton)} actions)")
        print(f"[Executor] Plan: {[str(a) for a in skeleton]}")
        
        # Ensure we start at home
        if self.env:
            self.go_home()
        
        while self.current_action_idx < len(skeleton):
            if self.is_interrupted:
                return ExecutionResult(
                    status=ExecutionStatus.INTERRUPTED,
                    completed_actions=self.current_action_idx,
                    total_actions=len(skeleton),
                    failure_reason="Execution interrupted"
                )
            
            action = skeleton[self.current_action_idx]
            print(f"\n[Executor] Processing Action {self.current_action_idx + 1}/{len(skeleton)}: {action}")
            
            try:
                # -----------------------------------------------------------
                # PATTERN: open-lid(lid)
                # -----------------------------------------------------------
                if action.action_name == 'open-lid' or action.action_name == 'open_lid':
                    print("  [Executor] Delegating to: script_3_open_grasp_open_box_gui")
                    
                    # Run the script
                    run_open_box(close_on_finish=False)
                    
                    self.executed_actions.append(action)
                    self.current_action_idx += 1
                
                # -----------------------------------------------------------
                # PATTERN: pick(obj) -> place(obj, region)
                # -----------------------------------------------------------
                elif action.action_name == 'pick':
                    # Look ahead for the place action
                    if self.current_action_idx + 1 >= len(skeleton):
                        raise Exception("Found 'pick' without following 'place'")
                        
                    next_action = skeleton[self.current_action_idx + 1]
                    
                    if next_action.action_name != 'place':
                         raise Exception(f"Expected 'place' after 'pick', found {next_action.action_name}")
                    
                    obj_name = action.args[0]
                    target_obj = next_action.args[0]
                    target_region = next_action.args[1]
                    
                    if obj_name != target_obj:
                        raise Exception(f"Pick object ({obj_name}) does not match Place object ({target_obj})")
                    
                    print(f"  [Executor] Merging Pick+Place for '{obj_name}' -> '{target_region}'")
                    
                    # Choose script based on object
                    if obj_name == 'mug_inside_box':
                         print("  [Executor] Delegating to: generalize_PP_box_gui (Constrained)")
                         run_pick_place_box(obj_name, target_region, close_on_finish=False)
                    else:
                         print("  [Executor] Delegating to: generalize_pick_place_gui (Standard)")
                         run_pick_place(obj_name, target_region, close_on_finish=False)
                    
                    # Mark BOTH as executed
                    self.executed_actions.append(action)
                    self.executed_actions.append(next_action)
                    self.current_action_idx += 2
                    
                # -----------------------------------------------------------
                # UNKNOWN
                # -----------------------------------------------------------
                else:
                    # Logic for 'place' is handled by the 'pick' block looking ahead.
                    # If we hit a 'place' here, it means it wasn't paired, which is an error in our logic
                    # or the plan is weird ("place" without "pick").
                    if action.action_name == 'place':
                         raise Exception("Orphaned 'place' action found (not preceded by 'pick')")
                    else:
                         raise Exception(f"Unknown action type: {action.action_name}")
                
                # Post-Action Homing
                if self.env:
                    success_home = self.go_home()
                    if not success_home:
                        print("  [Executor] Warning: Failed to return home safely.")
                
                time.sleep(self.config.pause_between_actions)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    completed_actions=self.current_action_idx,
                    total_actions=len(skeleton),
                    current_action=action,
                    error_message=str(e),
                    failure_reason=str(e)
                )
        
        print(f"\n[Executor] Plan completed successfully!")
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            completed_actions=len(skeleton),
            total_actions=len(skeleton),
            trajectory_executed=True
        )

    def get_remaining_plan(self) -> List[ActionSkeleton]:
        return self.current_plan[self.current_action_idx:]
    
    def get_executed_plan(self) -> List[ActionSkeleton]:
        return self.executed_actions.copy()
