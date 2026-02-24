import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object

@dataclass
class ObjectState:
    """Represents the state of an object in the scene."""
    name: str
    pddl_name: str
    obj_type: str
    location: str
    region: Optional[str] = None
    is_pickable: bool = True
    blocked_by: Optional[str] = None
    state: Optional[str] = None  # For lids: 'open' or 'closed'


@dataclass
class SceneState:
    """Complete scene state for LLM context."""
    objects: List[ObjectState]
    regions: List[Dict[str, str]]
    robot_gripper_state: str = "empty"
    robot_holding: Optional[str] = None
    lid_state: str = "closed"
    expected_objects: List[str] = field(default_factory=list)
    missing_objects: List[str] = field(default_factory=list)


@dataclass  
class PromptBundle:
    """Bundle of all context for LLM query."""
    composite_image: None  # Legacy support, set to None
    individual_frames: None # Legacy support
    state_text: str  
    goal_text: str   
    system_prompt: str
    user_prompt: str
    is_replan: bool = False
    failure_context: Optional[str] = None
    unknown_object_crop: None = None
    previous_plan: Optional[List[str]] = None


class DynamicStateExtractor:
    """
    Replaces VLMContextAggregator.
    Uses pure PyRep backend handles to extract bounding boxes and IK blockages
    to build an open-world semantic prompt.
    """
    
    def __init__(self, env=None):
        self.env = env
        
        # We define core regions that act as containers or surfaces
        self.regions = [
            {'name': 'table', 'description': 'main dining table surface'},
            {'name': 'box-top', 'description': 'top surface of the closed box'},
            {'name': 'box-inside', 'description': 'interior of the box (accessible when lid is open)'},
            {'name': 'placement_boundary', 'description': 'target area for placing mugs'},
            {'name': 'cupboard_boundary', 'description': 'inside the cupboard shelf'},
        ]
        
    def set_env(self, env):
        self.env = env
        
    def _is_dynamic_shape(self, handle: int) -> bool:
        """Check if a PyRep handle is a movable shape (not a wall/floor)."""
        try:
            obj_type = sim.simGetObjectType(handle)
            if obj_type != sim.sim_object_shape_type:
                return False
            # If static is 0, it's movable. If 1, it's static (wall, table)
            is_static = sim.simGetObjectInt32Parameter(handle, sim.sim_shapeintparam_static)
            return is_static == 0
        except:
            return False

    def get_scene_objects(self) -> List[Tuple[str, Object]]:
        """Scans the open world for all dynamic/interactable objects."""
        found_objects = []
        try:
            handles = sim.simGetObjects(sim.sim_handle_all)
            for h in handles:
                if self._is_dynamic_shape(h):
                    obj = Object.get_object(h)
                    name = obj.get_name()
                    # Filter out purely structural pieces, visual markers, or robot components
                    name_lower = name.lower()
                    if "visual" not in name_lower and "respondable" not in name_lower and "panda" not in name_lower and "contact" not in name_lower and "force" not in name_lower and "prox_sensor" not in name_lower:
                        obj = Object.get_object(h)
                        found_objects.append((name, obj))
            
            # Also always include the 'box_lid' even if it's considered kinematic/static
            lid = self.env.get_object('box_lid')
            if lid and not any(n == 'box_lid' for n, _ in found_objects):
                found_objects.append(('box_lid', lid))
                
        except Exception as e:
            print(f"[StateExtractor] Error discovering objects: {e}")
            
        return found_objects

    def check_ik_accessibility(self, obj: Object) -> Tuple[bool, Optional[str]]:
        """
        Tests if the robot can physically reach an object.
        Returns: (is_accessible, blocking_object_name)
        """
        # A true implementation would call self.env.robot.arm.solve_ik(...)
        # For performance, we use geometric bounding boxes as a proxy for the IK collision failure.
        try:
            obj_pos = obj.get_position()
            
            # Hardcoded check for the known lid obstruction logic for safety,
            # but expands naturally to other objects
            if "mug4" in obj.get_name() or "mug_inside_box" in obj.get_name():
                lid = self.env.get_object('box_lid')
                if lid:
                    lid_pos = lid.get_position()
                    if lid_pos[2] < 0.85: # Lid is closed and physically blocking
                        return False, "box_lid"
            
            # Check if any OTHER object is directly on top of it
            all_objs = self.get_scene_objects()
            for other_name, other_obj in all_objs:
                if other_name != obj.get_name() and other_name != 'box_lid':
                    other_pos = other_obj.get_position()
                    # If another object is directly above it (Z > obj_Z) and close in XY
                    if other_pos[2] > obj_pos[2] and \
                       abs(other_pos[0] - obj_pos[0]) < 0.1 and \
                       abs(other_pos[1] - obj_pos[1]) < 0.1:
                        return False, other_name
                        
            return True, None
        except Exception as e:
            # If math fails, default to accessible to let standard planning try
            return True, None

    def get_scene_state(self) -> SceneState:
        objects = []
        
        # Assess lid state
        lid_state = "closed"
        try:
            lid_obj = self.env.get_object('box_lid')
            box_base = self.env.get_object('box_base')
            if lid_obj and box_base:
                if abs(lid_obj.get_position()[0] - box_base.get_position()[0]) > 0.10:
                    lid_state = "open"
        except:
            pass

        # Dynamically discover all objects
        scene_objs = self.get_scene_objects()
        
        for name, pyrep_obj in scene_objs:
            if name == 'box_lid':
                objects.append(ObjectState(
                    name=name, pddl_name=name, obj_type='lid',
                    location='on box', is_pickable=False, state=lid_state
                ))
                continue
                
            # Classify location roughly
            pos = pyrep_obj.get_position()
            location = "on table"
            region = "table"
            
            # Simple geometric region mapping
            if pos[2] > 0.85 and 0.0 < pos[0] < 0.4: # Example box coordinates
                location = "on closed box"
                region = "box-top"
            elif pos[2] < 0.85 and 0.0 < pos[0] < 0.4:
                location = "inside box"
                region = "box-inside"
                
            # Check Physics / IK blockages
            is_reachable, blocker = self.check_ik_accessibility(pyrep_obj)
            
            objects.append(ObjectState(
                name=name,
                pddl_name=name,
                obj_type="item",
                location=location,
                region=region,
                is_pickable=is_reachable,
                blocked_by=blocker
            ))

        return SceneState(
            objects=objects,
            regions=self.regions,
            robot_gripper_state="empty",
            robot_holding=None,
            lid_state=lid_state
        )
    
    def state_to_pddl_text(self, state: SceneState) -> str:
        lines = ["=== CURRENT SEMANTIC STATE ===", ""]
        
        lines.append("## Robot Status:")
        lines.append(f"- gripper: {state.robot_gripper_state}")
        lines.append("")
        
        lines.append("## Dynamically Discovered Objects:")
        for obj in state.objects:
            status = []
            if obj.state: status.append(f"state={obj.state}")
            status.append(f"location={obj.location}")
            if not obj.is_pickable:
                status.append(f"STATUS=BLOCKED_BY_{obj.blocked_by}")
            
            lines.append(f"- {obj.name}: {', '.join(status)}")
        lines.append("")
        
        lines.append("## Valid Target Regions:")
        for r in state.regions:
            lines.append(f"- {r['name']}: {r['description']}")
            
        return "\n".join(lines)
    
    def build_system_prompt(self) -> str:
        return """You are a stateless linguistic task planner for a deterministic robotic manipulation system.
You do not use vision. You rely purely on the provided semantic state mathematically extracted from the PyRep physics engine.

AVAILABLE ACTIONS:
1. pick(object_name)
2. place(object_name, region_name)
3. open-lid(lid_name)

CRITICAL RULES:
1. If an object's STATUS is BLOCKED_BY_[X], you CANNOT pick it. You must MUST move [X] first.
2. A sequence MUST be completely valid logically. E.g., open-lid must occur before picking an object blocked by the lid.

OUTPUT FORMAT:
Output ONLY a numbered list of actions, one per line. Do not include explanations."""

    def build_user_prompt(self, state: SceneState, goal: str, 
                          is_replan: bool = False,
                          failure_context: Optional[str] = None,
                          previous_plan: Optional[List[str]] = None) -> str:
        
        state_text = self.state_to_pddl_text(state)
        
        if is_replan:
            return f"""=== REPLANNING TRIGGERED BY EXECUTION MONITOR ===

## PREVIOUS PLAN EXECUTED:
{chr(10).join(previous_plan) if previous_plan else 'Unknown'}

## CRITICAL EXECUTION FAILURE:
The previous plan halted midway due to the following hardware/physics failure:
>> {failure_context} <<

{state_text}

## GOAL:
{goal}

Provide a completely NEW action sequence to achieve the goal, avoiding the previous failure."""
        else:
            return f"""{state_text}

## GOAL:
{goal}

Provide the optimal sequence of actions to achieve the goal."""
    
    def create_prompt_bundle(self, goal: str,
                             is_replan: bool = False,
                             failure_context: Optional[str] = None,
                             previous_plan: Optional[List[str]] = None,
                             unknown_object_crop: Any = None) -> PromptBundle:
        state = self.get_scene_state()
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(state, goal, is_replan, failure_context, previous_plan)
        state_text = self.state_to_pddl_text(state)
        
        return PromptBundle(
            composite_image=None,
            individual_frames=None,
            state_text=state_text,
            goal_text=goal,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            is_replan=is_replan,
            failure_context=failure_context,
            previous_plan=previous_plan
        )
