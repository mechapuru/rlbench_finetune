#!/usr/bin/env python3
"""
VLM Execute Pipeline
====================
Integrated pipeline that:
1. Uses existing VLM planner (Qwen2-VL) to generate action plans
2. Executes actions using ground_truth_orchestrator execution functions
3. Supports action-by-action execution for replanning

This bridges the VLM pipeline with the proven execution code.
"""

import os
import sys
import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Set HEADLESS env var BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pddlstream'))

from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, And
from pddlstream.utils import read

# Import ENV from streams
from rlbench_kitchen_streams import ENV, get_stream_map


# ============================================================
# OBJECT NAME MAPPING (VLM names <-> Environment names)
# ============================================================

VLM_TO_ENV_OBJECT = {
    # Mugs
    'mug_box': 'mug2',           # On top of box
    'mug_inside_box': 'mug4',    # Inside the box
    'mug_table': 'mug1',         # On table
    'mug_cupboard': 'mug3',      # In cupboard
    
    # Can also use direct names
    'mug1': 'mug1',
    'mug2': 'mug2',
    'mug3': 'mug3',
    'mug4': 'mug4',
    
    # Groceries (same names)
    'soup': 'soup',
    'mustard': 'mustard',
    'spam': 'spam',
    'sugar': 'sugar',
    'crackers': 'crackers',
    
    # Lid
    'box_lid': 'box_lid',
    'lid': 'box_lid',
}

VLM_TO_ENV_REGION = {
    'box_top': 'box_boundary',
    'box_inside': 'box_boundary',
    'box_boundary': 'box_boundary',
    'table': 'placement_boundary',
    'placement_boundary': 'placement_boundary',
    'cupboard_boundary': 'cupboard_boundary',
    'cupboard_boundary_top': 'cupboard_boundary_top',
    'cupboard': 'cupboard_boundary',
    'groceries_boundary': 'cupboard_boundary',
}

# Objects that require special handling
CUPBOARD_OBJECTS = {'mug3', 'mug_cupboard'}
BOX_OBJECTS = {'mug2', 'mug4', 'mug_box', 'mug_inside_box'}


# ============================================================
# ACTION CLASSES
# ============================================================

class ActionType(Enum):
    PICK_PLACE = "pick_place"
    CUPBOARD_PICK_PLACE = "cupboard_pick_place"
    BOX_PICK_PLACE = "box_pick_place"
    OPEN_LID = "open_lid"


@dataclass
class ParsedAction:
    """A parsed action from VLM output."""
    action_type: ActionType
    object_name: str  # Environment object name
    target_region: str = None  # Environment region name
    original_text: str = ""
    
    def __repr__(self):
        if self.action_type == ActionType.OPEN_LID:
            return f"OPEN_LID()"
        return f"{self.action_type.value}({self.object_name}, {self.target_region})"


# ============================================================
# VLM PLAN PARSER
# ============================================================

def parse_vlm_plan(vlm_output: str) -> List[ParsedAction]:
    """
    Parse VLM output into executable actions.
    
    Handles formats:
    - 1. pick(obj)\n2. place(obj, region)
    - PICK_PLACE(obj, region)
    - open_lid(lid)
    """
    actions = []
    lines = vlm_output.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check for open_lid
        match = re.search(r'open[_-]?lid\s*\(\s*([^)]*)\s*\)', line, re.IGNORECASE)
        if match:
            actions.append(ParsedAction(
                action_type=ActionType.OPEN_LID,
                object_name='box_lid',
                original_text=line
            ))
            i += 1
            continue
        
        # Check for pick(obj)
        pick_match = re.search(r'pick\s*\(\s*([^)]+)\s*\)', line, re.IGNORECASE)
        if pick_match:
            vlm_obj = pick_match.group(1).strip().lower()
            env_obj = VLM_TO_ENV_OBJECT.get(vlm_obj, vlm_obj)
            
            # Look ahead for place
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                place_match = re.search(r'place\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', next_line, re.IGNORECASE)
                if place_match:
                    vlm_region = place_match.group(2).strip().lower()
                    env_region = VLM_TO_ENV_REGION.get(vlm_region, vlm_region)
                    
                    # Determine action type based on object location
                    if vlm_obj in CUPBOARD_OBJECTS or env_obj == 'mug3':
                        action_type = ActionType.CUPBOARD_PICK_PLACE
                    elif vlm_obj in BOX_OBJECTS or env_obj in ['mug2', 'mug4']:
                        action_type = ActionType.BOX_PICK_PLACE
                    else:
                        action_type = ActionType.PICK_PLACE
                    
                    actions.append(ParsedAction(
                        action_type=action_type,
                        object_name=env_obj,
                        target_region=env_region,
                        original_text=f"{line} + {next_line}"
                    ))
                    i += 2
                    continue
            
            # pick without place - skip
            print(f"Warning: pick without place: {line}")
            i += 1
            continue
        
        # Check for combined PICK_PLACE format
        match = re.search(r'PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
        if match:
            vlm_obj = match.group(1).strip().lower()
            vlm_region = match.group(2).strip().lower()
            env_obj = VLM_TO_ENV_OBJECT.get(vlm_obj, vlm_obj)
            env_region = VLM_TO_ENV_REGION.get(vlm_region, vlm_region)
            
            actions.append(ParsedAction(
                action_type=ActionType.PICK_PLACE,
                object_name=env_obj,
                target_region=env_region,
                original_text=line
            ))
            i += 1
            continue
        
        # Check for CUPBOARD_PICK_PLACE
        match = re.search(r'CUPBOARD_PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
        if match:
            vlm_obj = match.group(1).strip().lower()
            vlm_region = match.group(2).strip().lower()
            env_obj = VLM_TO_ENV_OBJECT.get(vlm_obj, vlm_obj)
            env_region = VLM_TO_ENV_REGION.get(vlm_region, vlm_region)
            
            actions.append(ParsedAction(
                action_type=ActionType.CUPBOARD_PICK_PLACE,
                object_name=env_obj,
                target_region=env_region,
                original_text=line
            ))
            i += 1
            continue
        
        # Check for BOX_PICK_PLACE
        match = re.search(r'BOX_PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
        if match:
            vlm_obj = match.group(1).strip().lower()
            vlm_region = match.group(2).strip().lower()
            env_obj = VLM_TO_ENV_OBJECT.get(vlm_obj, vlm_obj)
            env_region = VLM_TO_ENV_REGION.get(vlm_region, vlm_region)
            
            actions.append(ParsedAction(
                action_type=ActionType.BOX_PICK_PLACE,
                object_name=env_obj,
                target_region=env_region,
                original_text=line
            ))
            i += 1
            continue
        
        # Skip unrecognized lines
        i += 1
    
    return actions


# ============================================================
# HELPER FUNCTIONS (from ground_truth_orchestrator)
# ============================================================

def quaternion_from_euler(ai, aj, ak):
    """Convert Euler angles to quaternion."""
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk
    q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
    return q


def execute_trajectory(env, traj, steps=15):
    """Execute a trajectory with interpolation."""
    if not traj:
        return
    full_traj = []
    for i in range(len(traj) - 1):
        start = np.array(traj[i])
        end = np.array(traj[i + 1])
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1 - t) * start + t * end)
    full_traj.append(traj[-1])

    for conf in full_traj:
        env.set_robot_conf(conf)
        env.pr.step()


def interpolate_path(env, q1, q2, steps=50):
    """Interpolate between two configurations."""
    traj = []
    q1 = np.array(q1)
    q2 = np.array(q2)
    for i in range(steps + 1):
        t = i / steps
        q = (1 - t) * q1 + t * q2
        traj.append(q.tolist())
    return traj


def go_home(env):
    """Return robot to home configuration."""
    print("  Returning to home...")
    pr = env.pr
    q_start = env.get_robot_conf()
    q_home = env.get_home_conf()
    traj = interpolate_path(env, q_start, q_home, steps=100)
    if traj:
        execute_trajectory(env, traj, steps=10)
    else:
        env.set_robot_conf(q_home)
        for _ in range(10):
            pr.step()


def _normalize_segments(traj_tuple):
    """Normalize trajectory tuple into list of segments."""
    if traj_tuple is None:
        return []
    if isinstance(traj_tuple, (list, tuple)):
        segs = []
        for s in traj_tuple:
            if s is None:
                continue
            if isinstance(s, np.ndarray):
                segs.append(s.tolist())
            elif isinstance(s, (list, tuple)) and len(s) > 0:
                segs.append(list(s))
        return segs
    return []


def _get_ee_pos(env):
    """Get end-effector position."""
    try:
        tip = env.robot.get_tip()
        return np.array(tip.get_position(), dtype=float)
    except Exception:
        return np.array(env.robot.get_position(), dtype=float)


def _ee_pos_at_conf(env, conf, restore_conf):
    """Get EE position at a configuration."""
    env.set_robot_conf(conf)
    pos = _get_ee_pos(env)
    env.set_robot_conf(restore_conf)
    return pos


def _pick_grasp_segment_index(env, obj, segments):
    """Find segment index where EE is closest to object (for grasp)."""
    if not segments:
        return 0
    restore = env.get_robot_conf()
    obj_pos = np.array(obj.get_position(), dtype=float)
    best_i, best_d = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        d = float(np.linalg.norm(ee_pos - obj_pos))
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _place_release_segment_index(env, place_pose_p, segments):
    """Find segment index for release."""
    if not segments:
        return 0
    restore = env.get_robot_conf()
    try:
        if isinstance(place_pose_p, (list, tuple)) and len(place_pose_p) >= 3:
            p_xyz = np.array([float(place_pose_p[0]), float(place_pose_p[1]), float(place_pose_p[2])], dtype=float)
            best_i, best_d = 0, float("inf")
            for i, seg in enumerate(segments):
                end_conf = seg[-1]
                ee_pos = _ee_pos_at_conf(env, end_conf, restore)
                d = float(np.linalg.norm(ee_pos - p_xyz))
                if d < best_d:
                    best_d, best_i = d, i
            return best_i
    except:
        pass
    best_i, best_z = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        if float(ee_pos[2]) < best_z:
            best_z, best_i = float(ee_pos[2]), i
    return best_i


# ============================================================
# ACTION EXECUTORS
# ============================================================

def execute_pick_place(env, object_name: str, target_region: str) -> Tuple[bool, str]:
    """Execute standard pick and place using PDDL planning."""
    pr = env.pr
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        pr.step()

    env.set_target_region(target_region)

    obj = env.get_object(object_name)
    if obj is None:
        return False, f"Object '{object_name}' not found"

    obj.set_dynamic(False)
    initial_pose = obj.get_pose()

    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    pose_tuple = tuple(initial_pose)

    init = [
        ('conf', q_home_tuple),
        ('at-conf', q_home_tuple),
        ('hand-empty',),
        ('movable', object_name),
        ('pose', pose_tuple),
        ('at-pose', object_name, pose_tuple),
        ('region', target_region),
    ]

    goal = And(('hand-empty',), ('in-region', object_name, target_region))

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    solution = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
    plan, cost, evaluations = solution

    if not plan:
        return False, "No PDDL plan found"

    # Execute plan
    for action in plan:
        if action.name == 'move':
            q1, q2, traj = action.args
            execute_trajectory(env, traj)

        elif action.name == 'pick':
            o, p, g, q1, q2, traj_tuple = action.args
            segments = _normalize_segments(traj_tuple)
            if not segments:
                return False, "Pick has empty trajectory"

            target_obj = env.get_object(o)
            grasp_idx = _pick_grasp_segment_index(env, target_obj, segments)

            for seg in segments[:grasp_idx + 1]:
                execute_trajectory(env, seg)

            target_obj.set_dynamic(True)
            env.gripper.actuate(0.0, 0.1)
            for _ in range(10):
                pr.step()
            env.gripper.grasp(target_obj)

            for seg in segments[grasp_idx + 1:]:
                execute_trajectory(env, seg)

        elif action.name == 'place':
            o, p, g, r, q1, q2, traj_tuple = action.args
            segments = _normalize_segments(traj_tuple)
            if not segments:
                return False, "Place has empty trajectory"

            release_idx = _place_release_segment_index(env, p, segments)

            for seg in segments[:release_idx + 1]:
                execute_trajectory(env, seg)

            target_obj = env.get_object(o)
            env.gripper.release()
            target_obj.set_dynamic(True)

            hold_q = env.get_robot_conf()
            env.gripper.actuate(1.0, velocity=0.2)
            for _ in range(60):
                env.set_robot_conf(hold_q)
                pr.step()

            if len(segments) > release_idx + 1:
                for seg in segments[release_idx + 1:]:
                    execute_trajectory(env, seg)

    return True, "Success"


def execute_cupboard_pick_place(env, object_name: str, target_region: str) -> Tuple[bool, str]:
    """Execute cupboard pick with horizontal grasp and place."""
    pr = env.pr
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        pr.step()

    mug = env.get_object(object_name)
    if mug is None:
        return False, f"Object '{object_name}' not found"

    mug.set_dynamic(False)
    pose = mug.get_pose()

    # Find horizontal hover configuration
    hover_dists = [0.35, 0.30, 0.40]
    q_hover = None
    successful_grasp_quat = None
    original_conf = env.get_robot_conf()

    base_ry = np.pi / 2
    grasp_quats = [
        quaternion_from_euler(0, base_ry, 0),
        quaternion_from_euler(np.pi, base_ry, 0),
    ]

    target_z = pose[2]
    grasp_depth_offset = 0.03
    grasp_pos = None
    hover_pos = None

    for h_dist in hover_dists:
        if q_hover is not None:
            break
        hover_pos = [pose[0] - h_dist, pose[1], target_z]
        grasp_pos = [pose[0] + grasp_depth_offset, pose[1], target_z]

        for grasp_rot in grasp_quats:
            path_configs = env.robot.solve_ik_via_sampling(
                hover_pos, quaternion=grasp_rot, max_configs=50, max_time_ms=1000, ignore_collisions=True
            )
            if path_configs is not None and len(path_configs) > 0:
                for q in path_configs:
                    env.set_robot_conf(q)
                    if env.robot.check_collision():
                        continue
                    try:
                        path_check = env.robot.get_linear_path(
                            position=grasp_pos, quaternion=grasp_rot, steps=20, ignore_collisions=True
                        )
                        if path_check:
                            q_hover = q
                            successful_grasp_quat = grasp_rot
                            break
                    except:
                        pass
            if q_hover is not None:
                break

    if q_hover is None:
        env.set_robot_conf(original_conf)
        return False, "Could not find valid horizontal hover configuration"

    env.set_robot_conf(original_conf)

    # Move to hover
    traj_to_hover = env._interpolate_joint_path(home_q, q_hover, steps=100, check_collisions=False)
    if traj_to_hover:
        execute_trajectory(env, traj_to_hover, steps=10)

    env.gripper.release()
    for _ in range(30):
        pr.step()

    # Approach to grasp
    path_approach = env.robot.get_linear_path(
        position=grasp_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
    )
    if path_approach:
        traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
        for conf in traj_approach:
            env.set_robot_conf(conf)
            pr.step()

    # Grasp
    env.gripper.actuate(0.0, 0.1)
    for _ in range(50):
        pr.step()

    # Track mug
    gripper_tip = env.robot.get_tip()
    tip_pos = np.array(gripper_tip.get_position())
    mug_current_pos = np.array(mug.get_position())
    mug_offset = mug_current_pos - tip_pos

    # Retrieve
    retrieve_pos = list(hover_pos)
    retrieve_pos[2] += 0.02
    try:
        path_retrieve = env.robot.get_linear_path(
            position=retrieve_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
        )
    except:
        path_retrieve = None

    if path_retrieve:
        traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
        for conf in traj_retrieve:
            env.set_robot_conf(conf)
            pr.step()
            new_tip_pos = np.array(gripper_tip.get_position())
            mug.set_position((new_tip_pos + mug_offset).tolist())
    else:
        traj_fallback = env._interpolate_joint_path(env.get_robot_conf(), q_hover, steps=100, check_collisions=False)
        if traj_fallback:
            for conf in traj_fallback:
                env.set_robot_conf(conf)
                pr.step()
                new_tip_pos = np.array(gripper_tip.get_position())
                mug.set_position((new_tip_pos + mug_offset).tolist())

    # Place
    try:
        place_pose = env.find_best_placement(mug, target_region)
    except:
        place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]

    min_x, max_x, min_y, max_y, min_z, max_z = mug.get_bounding_box()
    mug_top_z = max_z

    hover_z = place_pose[2] + mug_top_z + 0.12
    place_z = place_pose[2] + 0.015

    place_hover_pos = [place_pose[0], place_pose[1], hover_z]
    place_pos = [place_pose[0], place_pose[1], place_z]

    place_quats = [quaternion_from_euler(np.pi, 0, angle) for angle in np.linspace(0, 2 * np.pi, 16)]

    q_place_hover = None
    successful_place_quat = None

    for place_quat in place_quats:
        path_configs = env.robot.solve_ik_via_sampling(
            place_hover_pos, quaternion=place_quat, max_configs=20, max_time_ms=500, ignore_collisions=True
        )
        if path_configs is not None and len(path_configs) > 0:
            for q in path_configs:
                env.set_robot_conf(q)
                if not env.robot.check_collision():
                    place_configs = env.robot.solve_ik_via_sampling(
                        place_pos, quaternion=place_quat, max_configs=5, max_time_ms=200, ignore_collisions=True
                    )
                    if place_configs is not None and len(place_configs) > 0:
                        q_place_hover = q
                        successful_place_quat = place_quat
                        break
        if q_place_hover is not None:
            break

    if q_place_hover is None:
        return False, "Could not find place hover configuration"

    # Move to place hover
    current_conf = env.get_robot_conf()
    traj_to_place = env._interpolate_joint_path(current_conf, q_place_hover, steps=150, check_collisions=False)
    if traj_to_place:
        for conf in traj_to_place:
            env.set_robot_conf(conf)
            pr.step()
            new_tip_pos = np.array(gripper_tip.get_position())
            mug.set_position((new_tip_pos + mug_offset).tolist())

    # Lower to place
    try:
        path_lower = env.robot.get_linear_path(
            position=place_pos, quaternion=successful_place_quat, steps=100, ignore_collisions=True
        )
    except:
        path_lower = None

    if path_lower:
        traj_lower = path_lower._path_points.reshape(-1, 7).tolist()
        for conf in traj_lower:
            env.set_robot_conf(conf)
            pr.step()
            new_tip_pos = np.array(gripper_tip.get_position())
            mug.set_position((new_tip_pos + mug_offset).tolist())

    # Release
    env.gripper.actuate(1.0, 0.1)
    for _ in range(30):
        pr.step()

    mug.set_dynamic(True)
    for _ in range(50):
        pr.step()

    # Lift
    try:
        lift_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.15]
        path_lift = env.robot.get_linear_path(
            position=lift_pos, quaternion=successful_place_quat, steps=50, ignore_collisions=True
        )
        if path_lift:
            traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
            execute_trajectory(env, traj_lift)
    except:
        pass

    return True, "Success"


def execute_box_pick_place(env, object_name: str, target_region: str) -> Tuple[bool, str]:
    """Execute pick from box area and place."""
    return execute_pick_place(env, object_name, target_region)


def execute_open_lid(env) -> Tuple[bool, str]:
    """Open the box lid by sliding."""
    pr = env.pr
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        pr.step()

    obj = env.get_object('box_lid')
    if obj is None:
        return False, "box_lid not found"

    env.gripper.actuate(1.0, 0.1)
    for _ in range(10):
        pr.step()

    try:
        grasp_quat, q_hover, q_grasp, (traj_hover_to_center, traj_center_to_edge) = env.compute_lid_grasp_trajectory(obj)
        traj_approach = traj_hover_to_center + traj_center_to_edge

        path_to_hover = env.compute_motion_plan(home_q, q_hover)

        if path_to_hover:
            execute_trajectory(env, path_to_hover, steps=5)
            execute_trajectory(env, traj_approach, steps=5)

            env.gripper.actuate(0.0, 0.1)
            for _ in range(30):
                pr.step()
            env.gripper.grasp(obj)

            q_open_start, q_open_end, traj_open, traj_return = env.compute_slide_lid_trajectory(
                obj, grasp_quat, initial_conf=q_grasp
            )

            if traj_open:
                execute_trajectory(env, traj_open, steps=5)

                env.gripper.release()
                env.gripper.actuate(1.0, 0.1)
                for _ in range(50):
                    pr.step()

                execute_trajectory(env, traj_return, steps=5)
                traj_retreat = traj_hover_to_center[::-1]
                execute_trajectory(env, traj_retreat, steps=5)

                for _ in range(50):
                    pr.step()

                return True, "Success"
            else:
                return False, "Failed to compute slide trajectory"
        else:
            return False, "Could not plan motion to hover"

    except Exception as e:
        return False, f"Error: {e}"


# ============================================================
# MAIN PIPELINE
# ============================================================

class VLMExecutePipeline:
    """
    Pipeline that takes VLM output and executes it.
    """
    
    def __init__(self, env, use_vlm: bool = False):
        self.env = env
        self.use_vlm = use_vlm
        self.execution_log = []
        self.vlm_planner = None
        
        if use_vlm:
            try:
                from vlm_pipeline.vlm_planner import VLMPlanner
                self.vlm_planner = VLMPlanner(
                    model_name="Qwen/Qwen2-VL-7B-Instruct",
                    use_4bit=True
                )
            except ImportError:
                print("Warning: VLM planner not available")
    
    def execute_action(self, action: ParsedAction, action_num: int) -> Tuple[bool, str]:
        """Execute a single parsed action."""
        print(f"\n{'='*60}")
        print(f"ACTION {action_num}: {action}")
        print(f"{'='*60}")
        
        try:
            if action.action_type == ActionType.PICK_PLACE:
                success, msg = execute_pick_place(self.env, action.object_name, action.target_region)
            elif action.action_type == ActionType.CUPBOARD_PICK_PLACE:
                success, msg = execute_cupboard_pick_place(self.env, action.object_name, action.target_region)
            elif action.action_type == ActionType.BOX_PICK_PLACE:
                success, msg = execute_box_pick_place(self.env, action.object_name, action.target_region)
            elif action.action_type == ActionType.OPEN_LID:
                success, msg = execute_open_lid(self.env)
            else:
                success, msg = False, f"Unknown action type: {action.action_type}"
            
            go_home(self.env)
            
            self.execution_log.append({
                'action_num': action_num,
                'action': str(action),
                'success': success,
                'message': msg
            })
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"\nResult: {status} - {msg}")
            
            return success, msg
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def execute_vlm_output(self, vlm_output: str, stop_on_failure: bool = True) -> Tuple[int, int]:
        """
        Parse and execute VLM output.
        
        Returns (successful_count, total_count)
        """
        print("\n" + "#"*60)
        print("PARSING VLM OUTPUT")
        print("#"*60)
        print(vlm_output)
        print("#"*60)
        
        actions = parse_vlm_plan(vlm_output)
        
        print(f"\nParsed {len(actions)} actions:")
        for i, action in enumerate(actions):
            print(f"  {i+1}. {action}")
        
        print("\n" + "#"*60)
        print("EXECUTING PLAN")
        print("#"*60)
        
        successful = 0
        for i, action in enumerate(actions):
            success, msg = self.execute_action(action, i + 1)
            
            if success:
                successful += 1
            elif stop_on_failure:
                print(f"\n⚠️  Stopping at action {i+1} for potential replanning")
                break
        
        return successful, len(actions)
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        
        for entry in self.execution_log:
            status = "✓" if entry['success'] else "✗"
            print(f"  {status} Action {entry['action_num']}: {entry['action']}")
            if not entry['success']:
                print(f"      Reason: {entry['message']}")
        
        successful = sum(1 for e in self.execution_log if e['success'])
        total = len(self.execution_log)
        print(f"\nTotal: {successful}/{total} actions successful")
        print("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Execute VLM-generated plans")
    parser.add_argument(
        "--vlm-output-file",
        type=str,
        help="File containing VLM output to execute"
    )
    parser.add_argument(
        "--vlm-output",
        type=str,
        help="Direct VLM output string to execute"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock VLM output for testing"
    )
    args = parser.parse_args()
    
    env = ENV
    pr = env.pr
    
    print("Initializing environment...")
    for _ in range(50):
        pr.step()
    
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        pr.step()
    
    # Get VLM output
    if args.vlm_output_file:
        with open(args.vlm_output_file, 'r') as f:
            vlm_output = f.read()
    elif args.vlm_output:
        vlm_output = args.vlm_output
    elif args.use_mock:
        # Mock output matching the ground truth task
        vlm_output = """
1. pick(mug_cupboard)
2. place(mug_cupboard, placement_boundary)
3. pick(soup)
4. place(soup, cupboard_boundary)
5. pick(mustard)
6. place(mustard, cupboard_boundary)
7. pick(spam)
8. place(spam, cupboard_boundary)
9. pick(sugar)
10. place(sugar, cupboard_boundary_top)
11. pick(crackers)
12. place(crackers, cupboard_boundary_top)
13. pick(mug_box)
14. place(mug_box, placement_boundary)
15. open_lid(box_lid)
16. pick(mug_inside_box)
17. place(mug_inside_box, placement_boundary)
"""
    else:
        print("Error: Please provide --vlm-output-file, --vlm-output, or --use-mock")
        return 1
    
    # Execute
    pipeline = VLMExecutePipeline(env)
    successful, total = pipeline.execute_vlm_output(vlm_output, stop_on_failure=False)
    
    pipeline.print_summary()
    
    print("\nExecution complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass
    
    pr.stop()
    pr.shutdown()
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
