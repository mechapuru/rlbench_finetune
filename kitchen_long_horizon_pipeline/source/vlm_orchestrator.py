#!/usr/bin/env python3
"""
VLM Orchestrator - Execute robot tasks based on VLM-generated plans.

This script:
1. Captures scene image from simulation
2. Queries VLM for action plan
3. Parses VLM output into executable actions
4. Executes actions one-by-one (allowing replanning after failures)
"""

import os
import sys
import re
import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Set HEADLESS env var BEFORE importing streams
os.environ["HEADLESS"] = "False"

from pddlstream.algorithms.algorithm import solve
from pddlstream.language.generator import from_fn
from pddlstream.utils import read
from pddlstream.language.constants import And

# Import ENV from streams
from rlbench_kitchen_streams import ENV, get_stream_map


# ============================================================
# ACTION TYPES
# ============================================================

class Action:
    """Base class for executable actions."""
    def __init__(self, action_type: str, params: Dict[str, Any]):
        self.action_type = action_type
        self.params = params
    
    def __repr__(self):
        return f"Action({self.action_type}, {self.params})"


# ============================================================
# VLM INTERFACE
# ============================================================

def capture_scene_image(env, camera="front") -> np.ndarray:
    """Capture current scene image from simulation."""
    frames = env.get_camera_frames()
    if camera in frames:
        return frames[camera]
    # Fallback to first available camera
    return list(frames.values())[0]


def query_vlm_for_plan(image: np.ndarray, task_description: str, available_objects: List[str], available_regions: List[str]) -> str:
    """
    Query VLM for an action plan.
    
    Returns raw VLM text output.
    """
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import torch
        from PIL import Image
        
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        print("Loading VLM model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Convert numpy image to PIL
        pil_image = Image.fromarray(image)
        
        # Build prompt
        prompt = f"""You are a robot task planner. Given an image of a kitchen scene, generate a sequence of actions to complete the task.

TASK: {task_description}

AVAILABLE OBJECTS: {', '.join(available_objects)}

AVAILABLE REGIONS/LOCATIONS:
- placement_boundary: The table area for placing items
- cupboard_boundary: Inside the cupboard (lower shelf)
- cupboard_boundary_top: Top shelf of the cupboard
- box_boundary: On top of the box

OUTPUT FORMAT:
Output a numbered list of actions. Each action should be one of:
1. PICK_PLACE(object, target_region) - Pick an object and place it in a region
2. CUPBOARD_PICK_PLACE(object, target_region) - Pick from cupboard with horizontal grasp
3. BOX_PICK_PLACE(object, target_region) - Pick from box area
4. OPEN_BOX() - Slide open the box lid

Example output:
1. PICK_PLACE(soup, cupboard_boundary)
2. CUPBOARD_PICK_PLACE(mug3, placement_boundary)
3. OPEN_BOX()
4. BOX_PICK_PLACE(mug4, placement_boundary)

Generate the action sequence now:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        print("Generating plan from VLM...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        output_text = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # Clean up GPU memory
        del model, processor
        torch.cuda.empty_cache()
        
        return output_text
        
    except ImportError as e:
        print(f"VLM not available: {e}")
        print("Using mock VLM response for testing...")
        return get_mock_vlm_response(task_description)


def get_mock_vlm_response(task_description: str) -> str:
    """Return a mock VLM response for testing without actual VLM."""
    # This simulates what a VLM might output for the kitchen task
    return """Based on the kitchen scene, here is the action sequence:

1. CUPBOARD_PICK_PLACE(mug3, placement_boundary)
2. PICK_PLACE(soup, cupboard_boundary)
3. PICK_PLACE(mustard, cupboard_boundary)
4. PICK_PLACE(spam, cupboard_boundary)
5. PICK_PLACE(sugar, cupboard_boundary_top)
6. PICK_PLACE(crackers, cupboard_boundary_top)
7. BOX_PICK_PLACE(mug2, placement_boundary)
8. OPEN_BOX()
9. BOX_PICK_PLACE(mug4, placement_boundary)
"""


# ============================================================
# PLAN PARSING
# ============================================================

def parse_vlm_output(vlm_output: str) -> List[Action]:
    """
    Parse VLM text output into a list of Action objects.
    
    Handles formats like:
    - 1. PICK_PLACE(soup, cupboard_boundary)
    - CUPBOARD_PICK_PLACE(mug3, placement_boundary)
    - OPEN_BOX()
    """
    actions = []
    
    # Pattern to match action calls
    patterns = [
        r'PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        r'CUPBOARD_PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        r'BOX_PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        r'OPEN_BOX\s*\(\s*\)',
    ]
    
    lines = vlm_output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try each pattern
        # PICK_PLACE
        match = re.search(r'PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
        if match:
            obj = match.group(1).strip().lower()
            region = match.group(2).strip().lower()
            actions.append(Action('PICK_PLACE', {'object': obj, 'region': region}))
            continue
        
        # CUPBOARD_PICK_PLACE
        match = re.search(r'CUPBOARD_PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
        if match:
            obj = match.group(1).strip().lower()
            region = match.group(2).strip().lower()
            actions.append(Action('CUPBOARD_PICK_PLACE', {'object': obj, 'region': region}))
            continue
        
        # BOX_PICK_PLACE
        match = re.search(r'BOX_PICK_PLACE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
        if match:
            obj = match.group(1).strip().lower()
            region = match.group(2).strip().lower()
            actions.append(Action('BOX_PICK_PLACE', {'object': obj, 'region': region}))
            continue
        
        # OPEN_BOX
        match = re.search(r'OPEN_BOX\s*\(\s*\)', line, re.IGNORECASE)
        if match:
            actions.append(Action('OPEN_BOX', {}))
            continue
    
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
    # Fallback: lowest EE z
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
    """
    Execute standard pick and place using PDDL planning.
    Returns (success, message).
    """
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

    # Setup PDDL problem
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

            # Retreat
            if len(segments) > release_idx + 1:
                for seg in segments[release_idx + 1:]:
                    execute_trajectory(env, seg)

    return True, "Success"


def execute_cupboard_pick_place(env, object_name: str, target_region: str) -> Tuple[bool, str]:
    """
    Execute cupboard pick with horizontal grasp and place.
    Returns (success, message).
    """
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
    z_offsets = [0.001]
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
        for z_off in z_offsets:
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
                        except Exception:
                            pass
                if q_hover is not None:
                    break
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
    """
    Execute pick from box area and place.
    Returns (success, message).
    """
    # This uses the same PDDL approach as standard pick_place
    # The streams handle box-specific logic
    return execute_pick_place(env, object_name, target_region)


def execute_open_box(env) -> Tuple[bool, str]:
    """
    Open the box lid by sliding.
    Returns (success, message).
    """
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
# MAIN ORCHESTRATOR
# ============================================================

# Import PDDLProblem
from pddlstream.language.stream import PDDLProblem


class VLMOrchestrator:
    """Orchestrates VLM-driven task execution with replanning support."""
    
    def __init__(self, env, use_vlm: bool = True):
        self.env = env
        self.use_vlm = use_vlm
        self.execution_log = []
        
        # Available objects and regions
        self.available_objects = [
            'mug1', 'mug2', 'mug3', 'mug4',
            'soup', 'mustard', 'spam', 'sugar', 'crackers',
            'box_lid'
        ]
        self.available_regions = [
            'placement_boundary',
            'cupboard_boundary',
            'cupboard_boundary_top',
            'box_boundary'
        ]
    
    def get_plan_from_vlm(self, task_description: str) -> List[Action]:
        """Get action plan from VLM."""
        print("\n" + "="*60)
        print("QUERYING VLM FOR PLAN")
        print("="*60)
        print(f"Task: {task_description}")
        
        # Capture scene image
        image = capture_scene_image(self.env, "front")
        
        # Query VLM
        if self.use_vlm:
            vlm_output = query_vlm_for_plan(
                image,
                task_description,
                self.available_objects,
                self.available_regions
            )
        else:
            vlm_output = get_mock_vlm_response(task_description)
        
        print("\nVLM Output:")
        print("-" * 40)
        print(vlm_output)
        print("-" * 40)
        
        # Parse output
        actions = parse_vlm_output(vlm_output)
        
        print(f"\nParsed {len(actions)} actions:")
        for i, action in enumerate(actions):
            print(f"  {i+1}. {action}")
        
        return actions
    
    def execute_action(self, action: Action, action_num: int) -> Tuple[bool, str]:
        """
        Execute a single action.
        Returns (success, message).
        """
        print(f"\n{'='*60}")
        print(f"EXECUTING ACTION {action_num}: {action.action_type}")
        print(f"{'='*60}")
        
        try:
            if action.action_type == 'PICK_PLACE':
                obj = action.params['object']
                region = action.params['region']
                print(f"  Object: {obj}")
                print(f"  Target: {region}")
                success, msg = execute_pick_place(self.env, obj, region)
                
            elif action.action_type == 'CUPBOARD_PICK_PLACE':
                obj = action.params['object']
                region = action.params['region']
                print(f"  Object: {obj}")
                print(f"  Target: {region}")
                success, msg = execute_cupboard_pick_place(self.env, obj, region)
                
            elif action.action_type == 'BOX_PICK_PLACE':
                obj = action.params['object']
                region = action.params['region']
                print(f"  Object: {obj}")
                print(f"  Target: {region}")
                success, msg = execute_box_pick_place(self.env, obj, region)
                
            elif action.action_type == 'OPEN_BOX':
                print(f"  Opening box lid")
                success, msg = execute_open_box(self.env)
                
            else:
                success, msg = False, f"Unknown action type: {action.action_type}"
            
            # Return home after each action
            go_home(self.env)
            
            # Log result
            self.execution_log.append({
                'action_num': action_num,
                'action': str(action),
                'success': success,
                'message': msg
            })
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"\n  Result: {status} - {msg}")
            
            return success, msg
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = f"Exception: {e}"
            self.execution_log.append({
                'action_num': action_num,
                'action': str(action),
                'success': False,
                'message': msg
            })
            return False, msg
    
    def execute_plan(self, actions: List[Action], stop_on_failure: bool = True) -> Tuple[int, int]:
        """
        Execute a list of actions.
        
        Args:
            actions: List of actions to execute
            stop_on_failure: If True, stop execution on first failure (for replanning)
        
        Returns:
            (successful_count, total_count)
        """
        print("\n" + "="*60)
        print("EXECUTING PLAN")
        print(f"Total actions: {len(actions)}")
        print("="*60)
        
        successful = 0
        
        for i, action in enumerate(actions):
            success, msg = self.execute_action(action, i + 1)
            
            if success:
                successful += 1
            elif stop_on_failure:
                print(f"\n⚠️  Stopping at action {i+1} for potential replanning")
                print(f"   Failed action: {action}")
                print(f"   Reason: {msg}")
                break
        
        return successful, len(actions)
    
    def run(self, task_description: str, max_replan_attempts: int = 3) -> bool:
        """
        Run the full VLM orchestration with optional replanning.
        
        Returns True if all actions completed successfully.
        """
        print("\n" + "#"*60)
        print("VLM ORCHESTRATOR")
        print("#"*60)
        print(f"Task: {task_description}")
        
        # Settle physics
        pr = self.env.pr
        for _ in range(50):
            pr.step()
        
        home_q = self.env.get_home_conf()
        self.env.set_robot_conf(home_q)
        for _ in range(10):
            pr.step()
        
        replan_count = 0
        remaining_actions = None
        
        while replan_count <= max_replan_attempts:
            if remaining_actions is None:
                # Get initial plan from VLM
                actions = self.get_plan_from_vlm(task_description)
            else:
                # Use remaining actions from failed plan
                actions = remaining_actions
                remaining_actions = None
            
            if not actions:
                print("\n❌ No actions to execute!")
                return False
            
            # Execute plan
            successful, total = self.execute_plan(actions, stop_on_failure=True)
            
            if successful == total:
                print("\n" + "="*60)
                print("✅ ALL ACTIONS COMPLETED SUCCESSFULLY")
                print("="*60)
                self.print_summary()
                return True
            
            # Plan failed - check for replanning
            if replan_count < max_replan_attempts:
                replan_count += 1
                print(f"\n🔄 REPLANNING (attempt {replan_count}/{max_replan_attempts})")
                
                # For now, we'll requery VLM with updated state
                # In production, you'd capture new image and describe current state
                remaining_actions = None  # Force new VLM query
            else:
                print(f"\n❌ Max replan attempts ({max_replan_attempts}) reached")
                break
        
        self.print_summary()
        return False
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        
        if not self.execution_log:
            print("  No actions executed")
            return
        
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
    
    parser = argparse.ArgumentParser(description="VLM-driven robot task orchestrator")
    parser.add_argument(
        "--task",
        default="Complete all kitchen tasks: move items to cupboard, retrieve mugs, and organize the workspace.",
        help="Task description for VLM"
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        default=False,
        help="Use actual VLM (requires model). Default uses mock response."
    )
    parser.add_argument(
        "--max-replan",
        type=int,
        default=0,
        help="Maximum replanning attempts on failure"
    )
    args = parser.parse_args()
    
    env = ENV
    pr = env.pr
    
    print("Initializing environment...")
    for _ in range(50):
        pr.step()
    
    orchestrator = VLMOrchestrator(env, use_vlm=args.use_vlm)
    success = orchestrator.run(args.task, max_replan_attempts=args.max_replan)
    
    print("\nOrchestration complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass
    
    pr.stop()
    pr.shutdown()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
