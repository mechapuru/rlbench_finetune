"""
Scene-Agnostic Context Aggregator
=================================
NO hardcoded object logic. Everything from config + generic constraints.
"""

import os
import sys
import yaml
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.constraints import (
    ConstraintLevel, 
    get_constraint_engine,
    ObjectState,
    SceneState,
)


@dataclass
class PromptBundle:
    composite_image: np.ndarray
    individual_frames: Dict[str, np.ndarray]
    state_text: str
    goal_text: str
    system_prompt: str
    user_prompt: str
    constraint_level: ConstraintLevel
    is_replan: bool = False
    failure_context: Optional[str] = None
    previous_plan: Optional[List[str]] = None


class SceneAgnosticContextAggregator:
    def __init__(self, 
                 config_path: str = None,
                 constraint_level: ConstraintLevel = ConstraintLevel.NONE,
                 env=None):
        self.env = env
        self.constraint_level = constraint_level
        self.constraint_engine = get_constraint_engine(constraint_level)
        
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                'configs', 
                'scene_config.yaml'
            )
        self.config = self._load_config(config_path)
        self.camera_names = ['left', 'right', 'overhead', 'wrist', 'front']
    
    def _load_config(self, path: str) -> Dict:
        if not os.path.exists(path):
            return {'objects': {}, 'regions': {}, 'containers': {}}
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def set_env(self, env):
        self.env = env
    
    def set_constraint_level(self, level: ConstraintLevel):
        self.constraint_level = level
        self.constraint_engine = get_constraint_engine(level)
    
    def capture_frames(self) -> Dict[str, np.ndarray]:
        if self.env is None:
            raise RuntimeError("Environment not set.")
        frames = {}
        for name in self.camera_names:
            if name in self.env.cams:
                cam = self.env.cams[name]
                cam.handle_explicitly()
                img = cam.capture_rgb()
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                frames[name] = img
        return frames
    
    def stitch_frames(self, frames: Dict[str, np.ndarray], layout: str = "grid") -> np.ndarray:
        ordered = [frames[n] for n in self.camera_names if n in frames]
        if not ordered:
            raise ValueError("No frames")
        h, w = ordered[0].shape[:2]
        resized = []
        for f in ordered:
            if f.shape[:2] != (h, w):
                f = np.array(Image.fromarray(f).resize((w, h)))
            resized.append(f)
        if layout == "grid":
            while len(resized) < 6:
                resized.append(np.zeros_like(resized[0]))
            row1 = np.concatenate(resized[:3], axis=1)
            row2 = np.concatenate(resized[3:6], axis=1)
            return np.concatenate([row1, row2], axis=0)
        return np.concatenate(resized, axis=1)
    
    def _get_lid_state_from_env(self) -> str:
        if not self.env:
            return "closed"
        try:
            lid = self.env.get_object('box_lid')
            box = self.env.get_object('box_base')
            if lid and box:
                offset = abs(lid.get_position()[0] - box.get_position()[0])
                return "open" if offset > 0.10 else "closed"
        except:
            pass
        return "closed"
    
    def get_scene_state(self) -> SceneState:
        objects = {}
        lid_state = self._get_lid_state_from_env()
        
        for name, cfg in self.config.get('objects', {}).items():
            state = None
            if cfg.get('initial_state'):
                state = lid_state if name == 'box_lid' else cfg.get('initial_state')
            
            objects[name] = ObjectState(
                name=name,
                obj_type=cfg.get('type', 'unknown'),
                location=cfg.get('initial_location', 'unknown'),
                spatial_relation=cfg.get('spatial_relation'),
                state=state,
            )
        
        regions = {}
        for rname, rcfg in self.config.get('regions', {}).items():
            regions[rname] = rcfg.get('description', rname) if isinstance(rcfg, dict) else str(rcfg)
        
        state = SceneState(
            objects=objects,
            regions=regions,
            containers=self.config.get('containers', {}),
            robot_holding=None,
            constraint_level=self.constraint_level,
        )
        return self.constraint_engine.apply(state)
    
    def state_to_text(self, state: SceneState) -> str:
        lines = ["=== SCENE STATE ===", "", "## Robot:"]
        lines.append(f"- Holding: {state.robot_holding}" if state.robot_holding else "- Gripper: empty")
        lines.extend(["", "## Objects:"])
        
        for name, obj in state.objects.items():
            parts = [f"{name}: type={obj.obj_type}, location={obj.location}"]
            if obj.state:
                parts.append(f"state={obj.state}")
            if obj.spatial_relation:
                rel = obj.spatial_relation
                parts.append(f"relation={rel.get('type', '')}({rel.get('target', '')})")
            if state.constraint_level != ConstraintLevel.NONE:
                if obj.blocked_by:
                    parts.append(f"BLOCKED_BY={obj.blocked_by}")
                if not obj.is_accessible:
                    parts.append("NOT_ACCESSIBLE")
            lines.append(f"- {', '.join(parts)}")
        
        lines.extend(["", "## Regions:"])
        for rname, desc in state.regions.items():
            lines.append(f"- {rname}: {desc}")
        
        return "\n".join(lines)
    
    def build_system_prompt(self) -> str:
        base = """You are a robot task planner. You control a robot arm in a kitchen environment.

Your job is to output a sequence of actions to achieve the given goal.

AVAILABLE ACTIONS:
1. pick(object) - Pick up an object
2. place(object, region) - Place held object in a region
3. open_lid(lid) - Open a lid/cover

OUTPUT FORMAT:
Output ONLY a numbered list of actions:
1. action(args)
2. action(args)
...

Do NOT include explanations."""
        
        constraint_desc = self.constraint_engine.get_constraint_description()
        return base + ("\n\n" + constraint_desc if constraint_desc else "")
    
    def build_user_prompt(self, state: SceneState, goal: str,
                          is_replan: bool = False,
                          failure_context: Optional[str] = None,
                          previous_plan: Optional[List[str]] = None) -> str:
        state_text = self.state_to_text(state)
        
        if is_replan:
            return f"""=== REPLANNING REQUIRED ===

PREVIOUS PLAN:
{chr(10).join(previous_plan) if previous_plan else 'None'}

FAILURE: {failure_context}

{state_text}

## GOAL: {goal}

Provide a NEW action sequence."""
        
        return f"""{state_text}

## GOAL: {goal}

## VISUAL CONTEXT:
The image shows 5 camera views of the scene.

Provide the action sequence to achieve the goal."""
    
    def create_prompt_bundle(self, goal: str,
                             is_replan: bool = False,
                             failure_context: Optional[str] = None,
                             previous_plan: Optional[List[str]] = None) -> PromptBundle:
        if self.env:
            frames = self.capture_frames()
        else:
            frames = {n: np.zeros((480, 640, 3), dtype=np.uint8) for n in self.camera_names}
        
        composite = self.stitch_frames(frames)
        state = self.get_scene_state()
        
        return PromptBundle(
            composite_image=composite,
            individual_frames=frames,
            state_text=self.state_to_text(state),
            goal_text=goal,
            system_prompt=self.build_system_prompt(),
            user_prompt=self.build_user_prompt(state, goal, is_replan, failure_context, previous_plan),
            constraint_level=self.constraint_level,
            is_replan=is_replan,
            failure_context=failure_context,
            previous_plan=previous_plan,
        )
