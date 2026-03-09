"""
Live Object Tracker - Displays visible objects from segmentation masks for each camera.

This module creates a live visualization that shows which objects are visible
from each camera in the simulation, using RLBench/CoppeliaSim segmentation masks.

The idea is to track only objects that are VISIBLE in camera views, so hidden
objects (like a mug inside a closed box) won't be detected until they become visible.
"""

import os
import sys
import numpy as np
import threading
import time
from collections import defaultdict

# cv2 imported lazily to avoid Qt conflicts - import after CoppeliaSim init
cv2 = None

def _ensure_cv2():
    """Lazy import of cv2 to avoid Qt initialization conflicts."""
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    return cv2

from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode


class LiveObjectTracker:
    """
    Tracks visible objects using segmentation masks from cameras.
    
    Uses OpenGL3_ColorCoded render mode to capture object handles encoded in RGB,
    then decodes them to identify which objects are visible in each camera view.
    """
    
    def __init__(self, env, update_interval=0.1, display_scale=1.0):
        """
        Initialize the tracker.
        
        Args:
            env: RLBenchKitchenEnv instance with cameras
            update_interval: How often to update displays (seconds)
            display_scale: Scale factor for display windows
        """
        # Ensure cv2 is loaded (lazy import)
        _ensure_cv2()
        
        self.env = env
        self.update_interval = update_interval
        self.display_scale = display_scale
        self.running = False
        self.update_thread = None
        
        # Build mapping from object handle to name
        self.handle_to_name = {}
        self.tracked_objects = set()
        self._build_handle_mapping()
        
        # Create mask cameras (same position as RGB cameras but with different render mode)
        self.mask_cams = {}
        self._setup_mask_cameras()
        
        # Track visible objects per camera
        self.visible_objects = {name: set() for name in self.env.cams.keys()}
        
        # For display
        self.window_names = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Colors for different object categories
        self.category_colors = {
            'mug': (255, 100, 100),      # Red-ish
            'grocery': (100, 255, 100),   # Green-ish
            'box': (100, 100, 255),       # Blue-ish
            'boundary': (200, 200, 100),  # Yellow-ish
            'other': (200, 200, 200),     # Gray
        }
        
    def _build_handle_mapping(self):
        """Build mapping from object handles to object names."""
        # Get all shapes in the scene
        try:
            handles = sim.simGetObjectsInTree(
                sim.sim_handle_scene, 
                sim.sim_object_shape_type, 
                0
            )
            
            for h in handles:
                try:
                    name = sim.simGetObjectName(h)
                    self.handle_to_name[h] = name
                except:
                    continue
                    
            print(f"[ObjectTracker] Found {len(self.handle_to_name)} shapes in scene")
            
            # Also register known objects from env
            for name, obj in self.env.name_to_obj.items():
                try:
                    handle = obj.get_handle()
                    self.handle_to_name[handle] = name
                    self.tracked_objects.add(name)
                except:
                    continue
                    
            print(f"[ObjectTracker] Tracking {len(self.tracked_objects)} named objects")
            
        except Exception as e:
            print(f"[ObjectTracker] Warning: Could not enumerate scene objects: {e}")
    
    def _setup_mask_cameras(self):
        """
        For each RGB camera, we'll use the same camera but switch render modes
        or create paired mask cameras if they exist in the scene.
        """
        # Check if mask cameras exist in the scene (RLBench style)
        mask_camera_names = {
            'left': 'cam_over_shoulder_left_mask',
            'right': 'cam_over_shoulder_right_mask',
            'overhead': 'cam_overhead_mask',
            'wrist': 'cam_wrist_mask',
            'front': 'cam_front_mask',
        }
        
        for cam_name, mask_cam_name in mask_camera_names.items():
            try:
                mask_cam = VisionSensor(mask_cam_name)
                mask_cam.set_explicit_handling(1)
                mask_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)  # For handle-coded output
                self.mask_cams[cam_name] = mask_cam
                print(f"[ObjectTracker] Found mask camera: {mask_cam_name}")
            except Exception as e:
                # Mask camera doesn't exist, we'll use the RGB camera with render mode switch
                print(f"[ObjectTracker] Mask camera {mask_cam_name} not found, will compute from RGB camera")
                self.mask_cams[cam_name] = None
    
    def _rgb_handles_to_mask(self, rgb):
        """
        Convert RGB-coded handles to a mask of object handles.
        
        In CoppeliaSim's handle-coded rendering mode, each pixel's RGB value
        encodes the object handle: handle = R + G*256 + B*256*256
        
        Args:
            rgb: numpy array of shape (H, W, 3) with values in [0, 1] or [0, 255]
            
        Returns:
            numpy array of shape (H, W) with integer object handles
        """
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        
        # Decode handle from RGB
        handles = (rgb[:, :, 0].astype(np.int32) + 
                   rgb[:, :, 1].astype(np.int32) * 256 + 
                   rgb[:, :, 2].astype(np.int32) * 256 * 256)
        
        return handles
    
    def _get_visible_objects_from_camera(self, cam_name):
        """
        Get list of visible objects from a camera using geometric checks.
        
        This approximates visibility by checking:
        1. Is the object above table level (not fallen)?
        2. Is the object within camera viewing distance?
        3. Is the object occluded by a container (e.g., mug4 inside closed box)?
        
        Args:
            cam_name: Name of the camera ('left', 'right', 'overhead', 'wrist', 'front')
            
        Returns:
            set of object names visible in this camera
        """
        visible = set()
        
        try:
            cam = self.env.cams.get(cam_name)
            if cam is None:
                return visible
            
            # Capture RGB - we'll extract object info from it
            cam.handle_explicitly()
            
            # Get camera parameters
            cam_pos = np.array(cam.get_position())
            
            # Check if box is open (for visibility of mug4)
            box_is_open = self._check_box_open()
            
            # For each tracked object, check if it's in the camera's view
            for obj_name in self.tracked_objects:
                obj = self.env.get_object(obj_name)
                if obj is None:
                    continue
                    
                try:
                    obj_pos = np.array(obj.get_position())
                    
                    # Simple visibility check: is object in front of camera?
                    to_obj = obj_pos - cam_pos
                    distance = np.linalg.norm(to_obj)
                    
                    # Check if within reasonable viewing distance
                    if distance > 0.1 and distance < 3.0:
                        # Check if object is above table level (not fallen)
                        if obj_pos[2] > 0.5:
                            # Special handling for mug4 (inside box)
                            # Only visible if box is open
                            if obj_name in ['mug4', 'mug_inside_box']:
                                if box_is_open:
                                    visible.add(obj_name)
                                # else: mug4 is hidden inside closed box
                            else:
                                visible.add(obj_name)
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"[ObjectTracker] Error getting visible objects from {cam_name}: {e}")
        
        return visible
    
    def _check_box_open(self):
        """
        Check if the box lid has been opened.
        The lid slides in the X direction when opened.
        
        Returns:
            True if box is open, False otherwise
        """
        try:
            lid = self.env.get_object('box_lid')
            box_base = self.env.get_object('box_base') or self.env.get_object('box')
            
            if lid is None or box_base is None:
                return True  # Can't determine, assume open
            
            lid_pos = lid.get_position()
            box_pos = box_base.get_position()
            
            # Lid slides in X direction
            lid_offset = abs(lid_pos[0] - box_pos[0])
            
            # Threshold for considering lid "open"
            LID_OPEN_THRESHOLD = 0.08
            
            return lid_offset >= LID_OPEN_THRESHOLD
            
        except Exception as e:
            return True  # Error, assume open
        
        return visible
    
    def _get_visible_objects_from_mask(self, cam_name):
        """
        Get visible objects using actual mask camera if available.
        """
        visible = set()
        mask_cam = self.mask_cams.get(cam_name)
        
        if mask_cam is None:
            # Fall back to approximation
            return self._get_visible_objects_from_camera(cam_name)
        
        try:
            mask_cam.handle_explicitly()
            rgb = mask_cam.capture_rgb()
            handle_mask = self._rgb_handles_to_mask(rgb)
            
            # Get unique handles in the image
            unique_handles = np.unique(handle_mask)
            
            for h in unique_handles:
                if h == 0:
                    continue  # Background
                    
                name = self.handle_to_name.get(h)
                if name and name in self.tracked_objects:
                    visible.add(name)
                    
        except Exception as e:
            print(f"[ObjectTracker] Error with mask camera {cam_name}: {e}")
            return self._get_visible_objects_from_camera(cam_name)
        
        return visible
    
    def get_all_visible_objects(self):
        """
        Get all objects visible from any camera.
        
        Returns:
            dict mapping camera names to sets of visible object names
        """
        result = {}
        for cam_name in self.env.cams.keys():
            result[cam_name] = self._get_visible_objects_from_camera(cam_name)
        return result
    
    def _categorize_object(self, name):
        """Categorize an object for coloring."""
        name_lower = name.lower()
        if 'mug' in name_lower:
            return 'mug'
        elif any(g in name_lower for g in ['soup', 'mustard', 'spam', 'sugar', 'crackers', 'cereal', 'bottle', 'can', 'tin', 'food']):
            return 'grocery'
        elif 'box' in name_lower or 'lid' in name_lower:
            return 'box'
        elif 'boundary' in name_lower:
            return 'boundary'
        else:
            return 'other'
    
    def _create_display_image(self, cam_name, visible_objects, rgb_frame=None):
        """
        Create a display image showing camera view and visible objects list.
        """
        # Base dimensions
        panel_width = 300
        panel_height = 400
        
        if rgb_frame is not None:
            # Scale RGB frame
            h, w = rgb_frame.shape[:2]
            scale = min(panel_width / w, panel_height / h) * self.display_scale
            new_w, new_h = int(w * scale), int(h * scale)
            rgb_scaled = cv2.resize(rgb_frame, (new_w, new_h))
            rgb_bgr = cv2.cvtColor(rgb_scaled, cv2.COLOR_RGB2BGR)
            
            # Create combined image
            total_height = max(new_h, panel_height)
            img = np.zeros((total_height, new_w + panel_width, 3), dtype=np.uint8)
            
            # Place RGB frame
            img[:new_h, :new_w] = rgb_bgr
            
            # Panel offset
            panel_x = new_w
        else:
            img = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panel_x = 0
        
        # Draw panel background
        img[:, panel_x:] = (40, 40, 40)  # Dark gray
        
        # Draw title
        title = f"Camera: {cam_name.upper()}"
        cv2.putText(img, title, (panel_x + 10, 25), self.font, 0.6, (255, 255, 255), 1)
        
        # Draw separator line
        cv2.line(img, (panel_x + 5, 35), (panel_x + panel_width - 5, 35), (100, 100, 100), 1)
        
        # Draw object count
        count_text = f"Visible Objects: {len(visible_objects)}"
        cv2.putText(img, count_text, (panel_x + 10, 55), self.font, 0.5, (200, 200, 200), 1)
        
        # List visible objects
        y_offset = 80
        sorted_objects = sorted(visible_objects)
        
        for obj_name in sorted_objects:
            if y_offset > img.shape[0] - 20:
                break
                
            category = self._categorize_object(obj_name)
            color = self.category_colors.get(category, (200, 200, 200))
            
            # Draw colored bullet
            cv2.circle(img, (panel_x + 15, y_offset - 4), 4, color, -1)
            
            # Draw object name
            display_name = obj_name[:25] if len(obj_name) > 25 else obj_name
            cv2.putText(img, display_name, (panel_x + 25, y_offset), 
                       self.font, 0.4, (255, 255, 255), 1)
            
            y_offset += 18
        
        # Draw legend at bottom
        legend_y = img.shape[0] - 60
        cv2.line(img, (panel_x + 5, legend_y - 10), (panel_x + panel_width - 5, legend_y - 10), (100, 100, 100), 1)
        cv2.putText(img, "Legend:", (panel_x + 10, legend_y), self.font, 0.4, (180, 180, 180), 1)
        
        legend_items = [('Mug', 'mug'), ('Grocery', 'grocery'), ('Box', 'box')]
        x_off = panel_x + 10
        legend_y += 18
        for label, cat in legend_items:
            color = self.category_colors[cat]
            cv2.circle(img, (x_off + 5, legend_y - 4), 4, color, -1)
            cv2.putText(img, label, (x_off + 15, legend_y), self.font, 0.35, (180, 180, 180), 1)
            x_off += 70
        
        return img
    
    def update_display(self):
        """Update all camera displays with current visible objects."""
        for cam_name in self.env.cams.keys():
            # Get visible objects
            visible = self._get_visible_objects_from_camera(cam_name)
            self.visible_objects[cam_name] = visible
            
            # Get RGB frame
            try:
                cam = self.env.cams[cam_name]
                cam.handle_explicitly()
                rgb = cam.capture_rgb()
                rgb = (rgb * 255).astype(np.uint8)
            except:
                rgb = None
            
            # Create display
            display_img = self._create_display_image(cam_name, visible, rgb)
            
            # Show window
            window_name = f"Object Tracker - {cam_name}"
            if window_name not in self.window_names:
                self.window_names.append(window_name)
                
            cv2.imshow(window_name, display_img)
        
        cv2.waitKey(1)
    
    def _update_loop(self):
        """Background thread for updating displays."""
        while self.running:
            try:
                self.update_display()
            except Exception as e:
                print(f"[ObjectTracker] Update error: {e}")
            time.sleep(self.update_interval)
    
    def start(self, threaded=True):
        """
        Start the live object tracker.
        
        Args:
            threaded: If True, run updates in background thread
        """
        print("[ObjectTracker] Starting live object tracking...")
        self.running = True
        
        # Position windows
        positions = {
            'left': (0, 0),
            'right': (650, 0),
            'overhead': (0, 450),
            'wrist': (650, 450),
            'front': (1300, 0),
        }
        
        for cam_name, (x, y) in positions.items():
            window_name = f"Object Tracker - {cam_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(window_name, x, y)
            self.window_names.append(window_name)
        
        if threaded:
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
        else:
            # Single update
            self.update_display()
    
    def stop(self):
        """Stop the tracker and close windows."""
        print("[ObjectTracker] Stopping...")
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        for window_name in self.window_names:
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
        
        cv2.destroyAllWindows()
    
    def get_summary(self):
        """Get a summary of all visible objects across all cameras."""
        all_visible = set()
        for cam_name, objects in self.visible_objects.items():
            all_visible.update(objects)
        
        return {
            'total_visible': len(all_visible),
            'visible_objects': sorted(all_visible),
            'per_camera': {k: sorted(v) for k, v in self.visible_objects.items()}
        }
    
    def print_summary(self):
        """Print a formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("VISIBLE OBJECTS SUMMARY")
        print("="*60)
        print(f"Total unique visible objects: {summary['total_visible']}")
        print(f"All visible: {', '.join(summary['visible_objects'])}")
        print("-"*60)
        
        for cam_name, objects in summary['per_camera'].items():
            print(f"  {cam_name:10s}: ({len(objects)}) {', '.join(objects[:5])}" + 
                  ("..." if len(objects) > 5 else ""))
        
        print("="*60 + "\n")


class ObjectVisibilityTracker:
    """
    A simpler tracker that just maintains a set of currently visible objects.
    This can be used for replanning when new objects become visible.
    """
    
    def __init__(self, env):
        self.env = env
        self.known_objects = set()
        self.newly_visible = set()
        
        # Build handle mapping
        self.handle_to_name = {}
        self._build_handle_mapping()
    
    def _build_handle_mapping(self):
        """Build mapping from handles to names."""
        for name, obj in self.env.name_to_obj.items():
            try:
                handle = obj.get_handle()
                self.handle_to_name[handle] = name
            except:
                continue
    
    def update(self):
        """
        Update visibility state.
        
        Returns:
            set of newly visible objects (not seen before)
        """
        currently_visible = set()
        box_is_open = self.check_box_opened()
        
        for name, obj in self.env.name_to_obj.items():
            try:
                pos = obj.get_position()
                # Object is considered "visible" if it's above table level
                # and not inside a closed container
                if pos[2] > 0.5:
                    # Special case: mug4 is only visible if box is open
                    if name in ['mug4', 'mug_inside_box']:
                        if box_is_open:
                            currently_visible.add(name)
                    else:
                        currently_visible.add(name)
            except:
                continue
        
        # Find newly visible objects
        self.newly_visible = currently_visible - self.known_objects
        
        # Update known objects
        self.known_objects.update(currently_visible)
        
        return self.newly_visible
    
    def check_box_opened(self):
        """
        Check if the box has been opened (lid moved).
        When box opens, mug4 inside becomes visible.
        """
        try:
            lid = self.env.get_object('box_lid')
            box_base = self.env.get_object('box_base') or self.env.get_object('box')
            
            if lid and box_base:
                lid_pos = lid.get_position()
                box_pos = box_base.get_position()
                
                # Lid slides in X direction
                lid_offset = abs(lid_pos[0] - box_pos[0])
                
                # Threshold for considering lid "open"
                LID_OPEN_THRESHOLD = 0.08
                
                return lid_offset >= LID_OPEN_THRESHOLD
        except:
            pass
        return False
    
    def get_hidden_objects(self):
        """
        Get objects that exist but are not yet visible.
        These are typically inside containers.
        """
        all_objects = set(self.env.name_to_obj.keys())
        return all_objects - self.known_objects


# Demo / standalone test
if __name__ == "__main__":
    print("Live Object Tracker - Standalone Demo")
    print("This needs to be run alongside the ground_truth_orchestrator")
    print("See: run_with_tracker.py")
