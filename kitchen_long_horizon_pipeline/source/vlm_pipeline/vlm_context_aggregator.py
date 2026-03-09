"""
VLM Context Aggregator (Module 1)
=================================
Captures 5 camera views, builds object state from environment,
and creates prompt bundles for the VLM planner.

Tracks objects that are expected but not yet visible in frames.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ObjectState:
    """Represents the state of an object in the scene."""
    name: str
    pddl_name: str  # Name used in PDDL (e.g., 'mug_box')
    obj_type: str   # 'mug', 'can', 'lid', etc.
    location: str   # 'on table', 'inside box', 'on box-top', etc.
    region: Optional[str] = None  # PDDL region name
    is_pickable: bool = True
    blocked_by: Optional[str] = None
    state: Optional[str] = None  # For lids: 'open' or 'closed'
    visible_in_frames: bool = True  # Whether detected in current frames


@dataclass
class SceneState:
    """Complete scene state for VLM context."""
    objects: List[ObjectState]
    regions: List[Dict[str, str]]
    robot_gripper_state: str = "empty"
    robot_holding: Optional[str] = None
    lid_state: str = "closed"  # 'open' or 'closed'
    
    # Track objects not yet visible
    expected_objects: List[str] = field(default_factory=list)
    missing_objects: List[str] = field(default_factory=list)
    visibility_mode: str = "full-scene"  # 'full-scene' or 'visible-only'
    visible_objects: List[str] = field(default_factory=list)
    newly_visible_objects: List[str] = field(default_factory=list)


@dataclass  
class PromptBundle:
    """Bundle of all context for VLM query."""
    composite_image: np.ndarray  # Stitched 5-view image
    individual_frames: Dict[str, np.ndarray]  # Original frames
    state_text: str  # PDDL-style state description
    goal_text: str   # User's goal
    system_prompt: str
    user_prompt: str
    
    # For replanning
    is_replan: bool = False
    failure_context: Optional[str] = None
    unknown_object_crop: Optional[np.ndarray] = None
    previous_plan: Optional[List[str]] = None


class VLMContextAggregator:
    """
    Aggregates visual and state context for the VLM planner.
    """
    
    def __init__(self,
                 env=None,
                 visible_objects_only: bool = False,
                 enable_live_segmentation_view: bool = False):
        """
        Initialize the context aggregator.
        
        Args:
            env: RLBenchKitchenEnv instance (optional, can be set later)
            visible_objects_only: If True, include only segmentation-visible objects in state text
            enable_live_segmentation_view: If True, show live segmentation masks + visible list panel
        """
        self.env = env
        
        # Define expected objects in the scene
        self.expected_objects = {
            'mug_box': {'type': 'mug', 'scene_name': 'mug2'},
            'mug_inside_box': {'type': 'mug', 'scene_name': 'mug4'},
            'mug_table': {'type': 'mug', 'scene_name': 'mug1'},
            'mug_cupboard': {'type': 'mug', 'scene_name': 'mug3'},
            'soup': {'type': 'can', 'scene_name': 'soup'},
            'mustard': {'type': 'bottle', 'scene_name': 'mustard'},
            'spam': {'type': 'tin', 'scene_name': 'spam'},
            'sugar': {'type': 'box', 'scene_name': 'sugar'},
            'crackers': {'type': 'cereal', 'scene_name': 'crackers'},
            'box_lid': {'type': 'lid', 'scene_name': 'box_lid'},
        }
        
        # Define regions
        self.regions = [
            {'name': 'table', 'description': 'main dining table surface'},
            {'name': 'box-top', 'description': 'top surface of the closed box'},
            {'name': 'box-inside', 'description': 'interior of the box (accessible when lid is open)'},
            {'name': 'placement_boundary', 'description': 'target area for placing mugs'},
            {'name': 'cupboard_boundary', 'description': 'inside the cupboard shelf'},
            {'name': 'groceries_boundary', 'description': 'groceries storage area'},
        ]
        
        # Camera names
        self.camera_names = ['left', 'right', 'overhead', 'wrist', 'front']

        # Visibility-mode settings (opt-in; default behavior remains unchanged)
        self.visible_objects_only = visible_objects_only
        self.enable_live_segmentation_view = enable_live_segmentation_view

        # Segmentation components (lazy initialized)
        self.segmentation_detector = None
        self.live_segmentation_viewer = None
        self.live_viewer_backend = "none"
        self._visibility_init_attempted = False
        self._visibility_warning_printed = False

        # Track visible/new objects (scene names and mapped PDDL names)
        self.visible_scene_objects: Set[str] = set()
        self.visible_pddl_objects: Set[str] = set()
        self.newly_visible_scene_objects: Set[str] = set()
        self.newly_visible_pddl_objects: Set[str] = set()
        self.all_seen_scene_objects: Set[str] = set()
        self.all_seen_pddl_objects: Set[str] = set()

        # Scene-name -> PDDL-name mapping
        self.scene_to_pddl = {
            info['scene_name']: pddl_name
            for pddl_name, info in self.expected_objects.items()
        }

        if self.env is not None:
            self._initialize_visibility_tools()
        
    def set_env(self, env):
        """Set the environment instance."""
        self.env = env
        self._initialize_visibility_tools()

    def configure_visibility(self,
                             visible_objects_only: Optional[bool] = None,
                             enable_live_segmentation_view: Optional[bool] = None):
        """
        Configure segmentation-backed visibility mode.
        """
        if visible_objects_only is not None:
            self.visible_objects_only = visible_objects_only
        if enable_live_segmentation_view is not None:
            self.enable_live_segmentation_view = enable_live_segmentation_view
        self._initialize_visibility_tools()

    def _initialize_visibility_tools(self):
        """
        Lazy-init segmentation detector/viewer only when visibility features are enabled.
        """
        if self.env is None:
            return
        if not (self.visible_objects_only or self.enable_live_segmentation_view):
            return

        if self.segmentation_detector is None:
            try:
                from segmentation_object_detector import SegmentationObjectDetector
                self.segmentation_detector = SegmentationObjectDetector(self.env)
                print("[Context Aggregator] Segmentation detector initialized")
            except Exception as e:
                if not self._visibility_warning_printed:
                    print(f"[Context Aggregator] WARNING: Segmentation detector unavailable: {e}")
                    self._visibility_warning_printed = True
                self.segmentation_detector = None

        if self.enable_live_segmentation_view and self.live_segmentation_viewer is None:
            try:
                backend_pref = os.environ.get("LIVE_SEG_VIEWER_BACKEND", "auto").strip().lower()

                def _start_backend(name: str) -> bool:
                    if name == "tkinter":
                        from tkinter_segmentation_viewer import TkinterSegmentationViewer
                        viewer = TkinterSegmentationViewer(self.env, width=1260, height=480)
                        backend = "tkinter"
                    else:
                        from live_segmentation_viewer import LiveSegmentationViewer
                        viewer = LiveSegmentationViewer(self.env, width=1260, height=480)
                        backend = "opencv"

                    viewer.start()
                    proc = getattr(viewer, "viewer_proc", None)
                    if proc is not None:
                        time.sleep(0.7)
                        if not proc.is_alive():
                            try:
                                viewer.stop()
                            except Exception:
                                pass
                            return False

                    self.live_segmentation_viewer = viewer
                    self.live_viewer_backend = backend
                    return True

                ok = False
                if backend_pref == "tkinter":
                    ok = _start_backend("tkinter")
                    if not ok:
                        print("[Context Aggregator] Tkinter viewer failed, trying OpenCV viewer")
                        ok = _start_backend("opencv")
                elif backend_pref == "opencv":
                    ok = _start_backend("opencv")
                    if not ok:
                        print("[Context Aggregator] OpenCV viewer failed, trying Tkinter viewer")
                        ok = _start_backend("tkinter")
                else:
                    ok = _start_backend("opencv")
                    if not ok:
                        print("[Context Aggregator] OpenCV viewer failed, falling back to Tkinter viewer")
                        ok = _start_backend("tkinter")

                if ok:
                    print(f"[Context Aggregator] Live segmentation viewer started (backend: {self.live_viewer_backend})")
                else:
                    self.live_segmentation_viewer = None
                    self.live_viewer_backend = "none"
                    raise RuntimeError("No live segmentation viewer backend could be started")
            except Exception as e:
                if not self._visibility_warning_printed:
                    print(f"[Context Aggregator] WARNING: Live segmentation viewer unavailable: {e}")
                    self._visibility_warning_printed = True
                self.live_segmentation_viewer = None
                self.live_viewer_backend = "none"

        self._visibility_init_attempted = True

    def _scene_name_to_pddl_name(self, scene_name: str) -> str:
        """
        Map scene object name to PDDL object name used by prompts/plans.
        """
        return self.scene_to_pddl.get(scene_name, scene_name)

    def update_live_segmentation_view(self) -> Set[str]:
        """
        Update live segmentation window (if enabled).
        Returns currently detected scene names from the viewer.
        """
        if self.live_segmentation_viewer is None:
            return set()
        try:
            proc = getattr(self.live_segmentation_viewer, "viewer_proc", None)
            if proc is not None and not proc.is_alive():
                print("[Context Aggregator] WARNING: Live segmentation viewer process is not alive")
                return set()
            detected = self.live_segmentation_viewer.update()
            return set(detected) if detected is not None else set()
        except Exception:
            return set()

    def update_visible_objects(self, event: str = "") -> Dict[str, List[str]]:
        """
        Refresh visible object sets from segmentation and optionally update live viewer.
        """
        self._initialize_visibility_tools()

        # Keep viewer updating even if detector is unavailable
        self.update_live_segmentation_view()

        if self.segmentation_detector is None:
            return {
                "visible_scene": sorted(self.visible_scene_objects),
                "visible_pddl": sorted(self.visible_pddl_objects),
                "newly_visible_scene": [],
                "newly_visible_pddl": [],
            }

        try:
            current_scene = set(self.segmentation_detector.update())
        except Exception as e:
            if not self._visibility_warning_printed:
                print(f"[Context Aggregator] WARNING: Segmentation update failed: {e}")
                self._visibility_warning_printed = True
            current_scene = set()

        expected_names = set(self.expected_objects.keys())
        current_pddl = {
            self._scene_name_to_pddl_name(name)
            for name in current_scene
            if self._scene_name_to_pddl_name(name) in expected_names
        }

        newly_scene = current_scene - self.all_seen_scene_objects
        newly_pddl = current_pddl - self.all_seen_pddl_objects

        self.visible_scene_objects = current_scene
        self.visible_pddl_objects = current_pddl
        self.newly_visible_scene_objects = newly_scene
        self.newly_visible_pddl_objects = newly_pddl
        self.all_seen_scene_objects.update(current_scene)
        self.all_seen_pddl_objects.update(current_pddl)

        label = f" ({event})" if event else ""
        print(f"[Context Aggregator] Visible objects{label}: {len(current_pddl)} -> {sorted(current_pddl)}")
        if newly_pddl:
            print(f"[Context Aggregator] NEW object(s) entered scene{label}: {sorted(newly_pddl)}")

        return {
            "visible_scene": sorted(current_scene),
            "visible_pddl": sorted(current_pddl),
            "newly_visible_scene": sorted(newly_scene),
            "newly_visible_pddl": sorted(newly_pddl),
        }

    def shutdown(self):
        """Cleanup for live viewer process."""
        if self.live_segmentation_viewer is not None:
            try:
                self.live_segmentation_viewer.stop()
            except Exception:
                pass
            self.live_segmentation_viewer = None
            self.live_viewer_backend = "none"
        
    def capture_frames(self) -> Dict[str, np.ndarray]:
        """
        Capture frames from all 5 cameras.
        
        Returns:
            Dictionary mapping camera name to RGB numpy array
        """
        if self.env is None:
            raise RuntimeError("Environment not set. Call set_env() first.")
        
        frames = {}
        for name in self.camera_names:
            if name in self.env.cams:
                cam = self.env.cams[name]
                # Trigger capture
                cam.handle_explicitly()
                # Get RGB image
                img = cam.capture_rgb()
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                frames[name] = img
        
        return frames
    
    def stitch_frames(self, frames: Dict[str, np.ndarray], 
                      layout: str = "grid") -> np.ndarray:
        """
        Stitch multiple camera frames into a single composite image.
        
        Args:
            frames: Dictionary of camera frames
            layout: 'grid' (2x3) or 'horizontal' (1x5)
            
        Returns:
            Composite image as numpy array
        """
        # Get frames in order
        ordered_frames = []
        for name in self.camera_names:
            if name in frames:
                ordered_frames.append(frames[name])
        
        if not ordered_frames:
            raise ValueError("No frames to stitch")
        
        # Ensure all frames have same size
        target_h, target_w = ordered_frames[0].shape[:2]
        resized = []
        for f in ordered_frames:
            if f.shape[:2] != (target_h, target_w):
                from PIL import Image
                pil_img = Image.fromarray(f)
                pil_img = pil_img.resize((target_w, target_h))
                f = np.array(pil_img)
            resized.append(f)
        
        if layout == "grid":
            # 2x3 grid (pad with black if needed)
            while len(resized) < 6:
                resized.append(np.zeros_like(resized[0]))
            
            row1 = np.concatenate(resized[:3], axis=1)
            row2 = np.concatenate(resized[3:6], axis=1)
            composite = np.concatenate([row1, row2], axis=0)
        else:
            # Horizontal strip
            composite = np.concatenate(resized, axis=1)
        
        return composite
    
    def get_scene_state(self) -> SceneState:
        """
        Extract current scene state from the environment.
        
        Returns:
            SceneState object with all object and region information
        """
        objects = []
        missing = []

        # Visible-only mode: include only objects currently visible in segmentation masks
        visible_filter: Optional[Set[str]] = None
        visibility_mode = "full-scene"
        if self.visible_objects_only:
            self._initialize_visibility_tools()
            if self.segmentation_detector is not None:
                snapshot = self.update_visible_objects(event="scene-state")
                visible_filter = set(snapshot.get("visible_pddl", []))
                missing = sorted(set(self.expected_objects.keys()) - visible_filter)
                visibility_mode = "visible-only"
            else:
                # Safety fallback to full-scene mode if detector is unavailable
                if not self._visibility_warning_printed:
                    print("[Context Aggregator] WARNING: visible-only requested but segmentation detector is unavailable. Falling back to full-scene state.")
                    self._visibility_warning_printed = True
                visible_filter = None

        def include_object(pddl_name: str) -> bool:
            return visible_filter is None or pddl_name in visible_filter
        
        # Check lid state first
        lid_state = "closed"
        if self.env:
            try:
                lid_obj = self.env.get_object('box_lid')
                box_base = self.env.get_object('box_base')
                if lid_obj and box_base:
                    lid_pos = lid_obj.get_position()
                    box_pos = box_base.get_position()
                    x_offset = abs(lid_pos[0] - box_pos[0])
                    lid_state = "open" if x_offset > 0.10 else "closed"
            except:
                pass
        
        processed_keys = set()
        
        # 1. SPECIAL OBJECTS (Constraints)
        
        # mug_box - starts on top of the box
        if include_object('mug_box'):
            objects.append(ObjectState(
                name='mug_box',
                pddl_name='mug_box',
                obj_type='mug',
                location='on top of closed box',
                region='box-top',
                is_pickable=True,
                blocked_by=None
            ))
        processed_keys.add('mug_box')
        
        # mug_inside_box - inside the box, blocked by lid
        if include_object('mug_inside_box'):
            objects.append(ObjectState(
                name='mug_inside_box',
                pddl_name='mug_inside_box',
                obj_type='mug',
                location='inside the box',
                region='box-inside',
                is_pickable=(lid_state == "open"),
                blocked_by='box_lid' if lid_state == "closed" else None
            ))
        processed_keys.add('mug_inside_box')
        
        # box_lid
        # will handle blocked_by at the end
        if include_object('box_lid'):
            objects.append(ObjectState(
                name='box_lid',
                pddl_name='box_lid',
                obj_type='lid',
                location='on box',
                region=None,
                is_pickable=False,
                blocked_by=None,
                state=lid_state
            ))
        processed_keys.add('box_lid')
        
        # 2. GENERIC OBJECTS
        for name, info in self.expected_objects.items():
            if name in processed_keys:
                continue
            if not include_object(name):
                continue
            
            # Default location
            region = 'table'
            location = 'on table'
            
            # Specific defaults for known items
            if name == 'mug_cupboard':
                region = 'cupboard_boundary'
                location = 'inside cupboard'
            
            objects.append(ObjectState(
                name=name,
                pddl_name=name,
                obj_type=info['type'],
                location=location,
                region=region,
                is_pickable=True,
                blocked_by=None
            ))
            processed_keys.add(name)
            
        # 3. UPDATE CONSTRAINTS
        # Allow box_lid to be blocked if mug_box is on box-top
        mug_on_top = any(o.name == 'mug_box' and o.region == 'box-top' for o in objects)
        for o in objects:
            if o.name == 'box_lid':
                o.blocked_by = 'mug_box' if mug_on_top else None
        
        return SceneState(
            objects=objects,
            regions=self.regions,
            robot_gripper_state="empty",
            robot_holding=None,
            lid_state=lid_state,
            expected_objects=list(self.expected_objects.keys()),
            missing_objects=missing,
            visibility_mode=visibility_mode,
            visible_objects=sorted(list(self.visible_pddl_objects)) if visibility_mode == "visible-only" else sorted([o.pddl_name for o in objects]),
            newly_visible_objects=sorted(list(self.newly_visible_pddl_objects)) if visibility_mode == "visible-only" else []
        )
    
    def state_to_pddl_text(self, state: SceneState) -> str:
        """
        Convert scene state to PDDL-style text description.
        
        Args:
            state: SceneState object
            
        Returns:
            Formatted text describing state in PDDL terms
        """
        lines = []
        lines.append("=== CURRENT STATE (PDDL-style) ===")
        lines.append("")
        
        # Robot state
        lines.append("## Robot:")
        lines.append(f"- gripper: {state.robot_gripper_state}")
        if state.robot_holding:
            lines.append(f"- holding: {state.robot_holding}")
        lines.append(f"- (hand-empty): {state.robot_gripper_state == 'empty'}")
        lines.append("")
        
        # Objects
        lines.append("## Objects:")
        for obj in state.objects:
            blocked = f", BLOCKED BY {obj.blocked_by}" if obj.blocked_by else ""
            state_str = f", state={obj.state}" if obj.state else ""
            region_str = f", in-region={obj.region}" if obj.region else ""
            lines.append(f"- {obj.pddl_name}: type={obj.obj_type}, location={obj.location}{region_str}{state_str}{blocked}")
        lines.append("")

        if state.visibility_mode == "visible-only":
            lines.append("## Visibility:")
            lines.append("- mode: segmentation visible-objects-only")
            lines.append(f"- visible-count: {len(state.visible_objects)}")
            lines.append(f"- visible-objects: {', '.join(state.visible_objects) if state.visible_objects else '(none)'}")
            if state.newly_visible_objects:
                lines.append(f"- newly-visible: {', '.join(state.newly_visible_objects)}")
            lines.append("")
        
        # Lid state
        lines.append("## Lid State:")
        lines.append(f"- (lid-closed box_lid): {state.lid_state == 'closed'}")
        lines.append(f"- (lid-opened box_lid): {state.lid_state == 'open'}")
        lines.append("")
        
        # Regions
        lines.append("## Target Regions:")
        for r in state.regions:
            lines.append(f"- {r['name']}: {r['description']}")
        
        return "\n".join(lines)
    
    def build_system_prompt(self) -> str:
        """Build the system prompt for the VLM."""
        return """You are a robot task planner for a kitchen manipulation task. You control a Panda robot arm.

Your job is to output a sequence of actions to achieve the given goal. You MUST use ONLY these actions:

AVAILABLE ACTIONS:
1. pick(object) - Pick up an object.
2. place(object, region) - Place the held object in a target region.
3. open-lid(lid) - Open a lid to access objects inside.

EXAMPLE - Moving two objects:
1. pick(object1)
2. place(object1,placeament1)
3. pick(object2)
4. place(object2, placement2)

OUTPUT FORMAT:
Output ONLY a numbered list of actions, one per line. Each pick must be followed by a place for that same object.

MINIMALITY RULES:
- Include only actions needed for the stated goal.
- Do not move unrelated objects.
- Do not open lids/containers unless required for the goal object.

Do NOT include any explanation, just the action sequence."""
    
    def build_user_prompt(self, state: SceneState, goal: str, 
                          is_replan: bool = False,
                          failure_context: Optional[str] = None,
                          previous_plan: Optional[List[str]] = None) -> str:
        """
        Build the user prompt with state and goal.
        
        Args:
            state: Current scene state
            goal: Natural language goal
            is_replan: Whether this is a replanning query
            failure_context: Description of what failed (for replanning)
            previous_plan: The plan that was being executed (for replanning)
            
        Returns:
            User prompt string
        """
        state_text = self.state_to_pddl_text(state)
        
        if is_replan:
            prompt = f"""=== REPLANNING REQUIRED ===

PREVIOUS PLAN WAS:
{chr(10).join(previous_plan) if previous_plan else 'Unknown'}

FAILURE REASON:
{failure_context}

The attached image shows the current scene with the obstruction/unknown object highlighted.

{state_text}

## GOAL:
{goal}

Please provide a NEW action sequence that avoids the failure. Consider alternative paths or object orderings."""
        else:
            visibility_clause = ""
            if state.visibility_mode == "visible-only":
                visibility_clause = """
## IMPORTANT OBSERVABILITY RULE:
Only objects currently visible in segmentation masks are known/available for planning now.
If additional objects appear later, the system will replan."""

            prompt = f"""{state_text}

## GOAL:
{goal}
{visibility_clause}

## VISUAL CONTEXT:
The attached image shows 5 camera views of the scene:
- Top row: left shoulder, right shoulder, overhead
- Bottom row: wrist camera, front view

Based on the visual scene and state description, provide the action sequence to achieve the goal."""
        
        return prompt
    
    def create_prompt_bundle(self, goal: str,
                             is_replan: bool = False,
                             failure_context: Optional[str] = None,
                             previous_plan: Optional[List[str]] = None,
                             unknown_object_crop: Optional[np.ndarray] = None) -> PromptBundle:
        """
        Create complete prompt bundle for VLM query.
        
        Args:
            goal: Natural language goal
            is_replan: Whether this is a replanning query
            failure_context: What failed (for replanning)
            previous_plan: Previous plan being executed
            unknown_object_crop: Cropped image of unknown object
            
        Returns:
            PromptBundle ready for VLM
        """
        # Keep live viewer moving even when not using visible-only state mode.
        if self.enable_live_segmentation_view and not self.visible_objects_only:
            event = "replan-prompt" if is_replan else "plan-prompt"
            self.update_visible_objects(event=event)

        # Capture frames
        frames = self.capture_frames()
        
        # Stitch into composite
        composite = self.stitch_frames(frames)
        
        # Get scene state
        state = self.get_scene_state()
        
        # Build prompts
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(
            state, goal, is_replan, failure_context, previous_plan
        )
        state_text = self.state_to_pddl_text(state)
        
        print(f"[Context Aggregator] Captured {len(frames)} frames")
        print(f"[Context Aggregator] State text preview:\n{state_text}")
        print(f"[Context Aggregator] User prompt length: {len(user_prompt)}")
        
        return PromptBundle(
            composite_image=composite,
            individual_frames=frames,
            state_text=state_text,
            goal_text=goal,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            is_replan=is_replan,
            failure_context=failure_context,
            unknown_object_crop=unknown_object_crop,
            previous_plan=previous_plan
        )

    
    def create_prompt_bundle_offline(self, goal: str, 
                                      frames: Optional[Dict[str, np.ndarray]] = None,
                                      state: Optional[SceneState] = None) -> PromptBundle:
        """
        Create prompt bundle without live environment (for testing).
        
        Args:
            goal: Natural language goal
            frames: Pre-captured frames (or None for dummy)
            state: Pre-built state (or None for default)
            
        Returns:
            PromptBundle
        """
        # Use provided frames or create dummy
        if frames is None:
            frames = {name: np.zeros((480, 640, 3), dtype=np.uint8) 
                     for name in self.camera_names}
        
        # Stitch
        composite = self.stitch_frames(frames)
        
        # Use provided state or default
        if state is None:
            state = self.get_scene_state()
        
        # Build prompts
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(state, goal)
        state_text = self.state_to_pddl_text(state)
        
        return PromptBundle(
            composite_image=composite,
            individual_frames=frames,
            state_text=state_text,
            goal_text=goal,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            is_replan=False,
            failure_context=None,
            unknown_object_crop=None,
            previous_plan=None
        )
    
    def trigger_replan(self, failure_reason: str,
                       previous_plan: List[str],
                       goal: str,
                       unknown_object_crop: Optional[np.ndarray] = None) -> PromptBundle:
        """
        Create a replanning prompt bundle after a failure/interruption.
        
        This is the placeholder for the interruption system.
        Call this when:
        - An unknown object is detected in the scene
        - Execution fails (collision, IK failure, etc.)
        - User manually triggers replanning
        
        Args:
            failure_reason: Description of why replanning is needed
            previous_plan: The plan that was being executed
            goal: The original goal
            unknown_object_crop: Cropped image of unknown/blocking object
            
        Returns:
            PromptBundle for replanning query
        """
        return self.create_prompt_bundle(
            goal=goal,
            is_replan=True,
            failure_context=failure_reason,
            previous_plan=previous_plan,
            unknown_object_crop=unknown_object_crop
        )


# ============================================================================
# TESTING
# ============================================================================

def test_offline():
    """Test the context aggregator without environment."""
    print("Testing VLMContextAggregator (offline mode)...")
    
    aggregator = VLMContextAggregator()
    
    goal = "Move mug_box and mug_inside_box to placement_boundary. Move soup to cupboard_boundary."
    
    bundle = aggregator.create_prompt_bundle_offline(goal)
    
    print("\n=== SYSTEM PROMPT ===")
    print(bundle.system_prompt)
    print("\n=== USER PROMPT ===")
    print(bundle.user_prompt)
    print("\n=== COMPOSITE IMAGE SHAPE ===")
    print(bundle.composite_image.shape)
    
    # Test replan prompt
    print("\n" + "="*50)
    print("Testing REPLAN prompt...")
    
    replan_bundle = aggregator.create_prompt_bundle_offline(
        goal=goal,
        state=aggregator.get_scene_state()
    )
    
    # Manually set replan context
    replan_bundle.is_replan = True
    replan_bundle.failure_context = "Unknown object detected blocking path to box_lid"
    replan_bundle.previous_plan = ["1. pick(mug_box)", "2. place(mug_box, placement_boundary)"]
    
    print("Replan bundle created successfully.")


if __name__ == "__main__":
    test_offline()
