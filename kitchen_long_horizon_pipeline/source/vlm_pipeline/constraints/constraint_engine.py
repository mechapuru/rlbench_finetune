"""
Generic Constraint System
=========================
GENERIC spatial/physical rules. NO object-specific logic.

Ablation Levels:
- Level 0 (NONE): Domain file actions only (pick/place/open_lid preconditions)
- Level 1 (SPATIAL): Domain file + spatial blocking constraints
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ConstraintLevel(Enum):
    NONE = 0       # Domain actions only
    SPATIAL = 1    # + Spatial constraints


def load_domain_file() -> str:
    """Load the PDDL domain file and extract action definitions."""
    domain_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'pddl',
        'rlbench_kitchen_domain.pddl'
    )
    
    if not os.path.exists(domain_path):
        return "(Domain file not found)"
    
    with open(domain_path, 'r') as f:
        content = f.read()
    
    return content


def get_domain_actions_text() -> str:
    """Load actual PDDL domain file and format it for the VLM prompt."""
    pddl_content = load_domain_file()
    
    return f"""=== PDDL DOMAIN FILE (rlbench_kitchen_domain.pddl) ===

{pddl_content}

=== END DOMAIN FILE ===

CRITICAL RULES FROM DOMAIN:
1. You MUST pick(X) before you can place(X) - the robot can only hold ONE object at a time
2. Robot gripper must be empty to pick or open_lid"""


@dataclass
class ObjectState:
    name: str
    obj_type: str
    location: str
    spatial_relation: Optional[Dict] = None
    state: Optional[str] = None
    blocked_by: Optional[str] = None
    is_accessible: bool = True
    blocking: List[str] = field(default_factory=list)


@dataclass
class SceneState:
    objects: Dict[str, ObjectState]
    regions: Dict[str, str]
    containers: Dict[str, Dict]
    robot_holding: Optional[str] = None
    constraint_level: ConstraintLevel = ConstraintLevel.NONE


class ConstraintEngine(ABC):
    @abstractmethod
    def apply(self, state: SceneState) -> SceneState:
        pass
    
    @abstractmethod
    def get_constraint_description(self) -> str:
        pass


class NoConstraints(ConstraintEngine):
    """Level 0: Domain actions only - no spatial blocking in state."""
    def apply(self, state: SceneState) -> SceneState:
        for obj in state.objects.values():
            obj.blocked_by = None
            obj.is_accessible = True
            obj.blocking = []
        state.constraint_level = ConstraintLevel.NONE
        return state
    
    def get_constraint_description(self) -> str:
        # Always include domain actions
        return get_domain_actions_text()


class SpatialConstraints(ConstraintEngine):
    """Level 1: Domain actions + spatial blocking annotations in state."""
    def apply(self, state: SceneState) -> SceneState:
        state.constraint_level = ConstraintLevel.SPATIAL
        
        for obj in state.objects.values():
            obj.blocked_by = None
            obj.is_accessible = True
            obj.blocking = []
        
        # Rule 1: ON_TOP_OF blocks the lid beneath
        for obj_name, obj in state.objects.items():
            if obj.spatial_relation and obj.spatial_relation.get('type') == 'on_top_of':
                target = obj.spatial_relation.get('target')
                if target and target in state.containers:
                    lid_name = state.containers[target].get('lid')
                    if lid_name and lid_name in state.objects:
                        state.objects[lid_name].blocked_by = obj_name
                        state.objects[lid_name].is_accessible = False
                        obj.blocking.append(lid_name)
        
        # Rule 2: CLOSED container blocks INSIDE objects
        for obj_name, obj in state.objects.items():
            if obj.spatial_relation and obj.spatial_relation.get('type') == 'inside':
                target = obj.spatial_relation.get('target')
                if target and target in state.containers:
                    lid_name = state.containers[target].get('lid')
                    if lid_name and lid_name in state.objects:
                        lid_obj = state.objects[lid_name]
                        if lid_obj.state == 'closed':
                            obj.blocked_by = lid_name
                            obj.is_accessible = False
        
        return state
    
    def get_constraint_description(self) -> str:
        # Domain actions + spatial constraints
        return get_domain_actions_text() + """

IMPORTANT SPATIAL RULES:
1. CHECK THE OBJECT LIST FOR "BLOCKED_BY=X".
2. If an object is "BLOCKED_BY=X", you CANNOT pick it or open it. It is physically impossible.
3. You MUST move the blocking object X first.
4. EXAMPLE: If 'box_lid' is 'BLOCKED_BY=mug', you CANNOT 'open_lid(box_lid)'. You MUST 'pick(mug)' and 'place' it somewhere else first.
5. "NOT_ACCESSIBLE" means the action will fail constraints."""


def get_constraint_engine(level: ConstraintLevel) -> ConstraintEngine:
    return {
        ConstraintLevel.NONE: NoConstraints(),
        ConstraintLevel.SPATIAL: SpatialConstraints(),
    }.get(level, NoConstraints())


LEVEL_NONE = ConstraintLevel.NONE
LEVEL_SPATIAL = ConstraintLevel.SPATIAL
