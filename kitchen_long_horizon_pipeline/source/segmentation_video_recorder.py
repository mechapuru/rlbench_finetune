"""
Record SEGMENTATION MASK videos for all cameras.
Just like video_recorder.py but records masks instead of RGB.

Output: segmentation_videos/mask_left.mp4, mask_right.mp4, etc.
"""

import os
import cv2
import numpy as np
import colorsys
import re
from pyrep.backend import sim


class SegmentationVideoRecorder:
    """Records segmentation mask videos for all cameras."""
    
    def __init__(self, env, output_dir="segmentation_videos", fps=30, resolution=(640, 480)):
        self.env = env
        self.output_dir = output_dir
        self.fps = fps
        self.resolution = resolution
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Build handle -> name/color mapping
        self.handle_to_name = {}
        self.handle_to_color = {}
        self.handle_to_task_name = {}
        self.task_colors = {
            "mug1": (70, 70, 255),      # BGR
            "mug2": (40, 160, 255),
            "mug3": (0, 235, 255),
            "mug4": (255, 80, 230),
            "soup": (220, 220, 0),
            "mustard": (0, 255, 170),
            "spam": (255, 150, 70),
            "sugar": (190, 100, 255),
            "crackers": (255, 80, 120),
            "box_lid": (255, 255, 255),
            "cupboard": (160, 160, 160),
        }
        self.env_aliases = {
            "mug1": ["mug1", "mug_table"],
            "mug2": ["mug2", "mug_box"],
            "mug3": ["mug3", "mug_cupboard"],
            "mug4": ["mug4", "mug_inside_box"],
            "soup": ["soup", "can"],
            "mustard": ["mustard", "bottle"],
            "spam": ["spam", "tin"],
            "sugar": ["sugar", "food_box"],
            "crackers": ["crackers", "cereal"],
            "box_lid": ["box_lid"],
            "cupboard": ["cupboard"],
        }
        self._build_handles()
        
        # Mask cameras
        self.mask_cams = {}
        self._setup_mask_cameras()
        
        # Video writers
        self.writers = {}
        self._setup_writers()
        
        self.frame_count = 0
        print(f"[SegmentationRecorder] Recording to: {os.path.abspath(output_dir)}")
    
    def _build_handles(self):
        """Build handle -> name mapping from scene."""
        try:
            handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_shape_type, 0)
            n = len(handles)
            for i, h in enumerate(handles):
                try:
                    name = sim.simGetObjectName(h)
                    self.handle_to_name[h] = name
                    task_name = self._canonical_task_name(name)
                    if task_name is not None:
                        self.handle_to_task_name[h] = task_name
                        self.handle_to_color[h] = self.task_colors.get(task_name, (255, 255, 255))
                        continue
                    # Generate color
                    hue = i / max(n, 1)
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    self.handle_to_color[h] = (int(b*255), int(g*255), int(r*255))  # BGR for OpenCV
                except:
                    pass
        except Exception as e:
            print(f"[SegmentationRecorder] Handle scan error: {e}")

        # Prefer mapping by env object hierarchy for robust mug/cupboard IDs.
        try:
            for task_name, aliases in self.env_aliases.items():
                root_obj = None
                for alias in aliases:
                    try:
                        root_obj = self.env.get_object(alias)
                        if root_obj is not None:
                            break
                    except Exception:
                        continue
                if root_obj is None:
                    continue
                try:
                    root_handle = root_obj.get_handle()
                    shape_handles = sim.simGetObjectsInTree(
                        int(root_handle),
                        sim.sim_object_shape_type,
                        0,
                    )
                    for handle in shape_handles:
                        self.handle_to_task_name[int(handle)] = task_name
                        self.handle_to_color[int(handle)] = self.task_colors.get(task_name, (255, 255, 255))
                except Exception:
                    continue
        except Exception:
            pass

    def _canonical_task_name(self, scene_name):
        """Map low-level shape names to task objects."""
        if not scene_name:
            return None
        n = scene_name.lower()
        if 'mug_inside_box' in n or re.search(r'\bmug4\b', n):
            return 'mug4'
        if 'mug_cupboard' in n or re.search(r'\bmug3\b', n):
            return 'mug3'
        if 'mug_box' in n or re.search(r'\bmug2\b', n):
            return 'mug2'
        if 'mug_table' in n or re.search(r'\bmug1\b', n):
            return 'mug1'
        if 'box_lid' in n or n == 'lid':
            return 'box_lid'
        if 'cupboard' in n:
            return 'cupboard'
        for name in ('soup', 'mustard', 'spam', 'sugar', 'crackers'):
            if name in n:
                return name
        return None
    
    def _setup_mask_cameras(self):
        """Find mask cameras in scene or prepare to use render mode switching."""
        from pyrep.objects.vision_sensor import VisionSensor
        
        cam_mapping = {
            'left': 'cam_over_shoulder_left_mask',
            'right': 'cam_over_shoulder_right_mask', 
            'overhead': 'cam_overhead_mask',
            'wrist': 'cam_wrist_mask',
            'front': 'cam_front_mask',
        }
        
        for name, mask_name in cam_mapping.items():
            try:
                cam = VisionSensor(mask_name)
                cam.set_explicit_handling(1)
                self.mask_cams[name] = ('mask', cam)
                print(f"[SegmentationRecorder] Found mask camera: {mask_name}")
            except:
                # Use render mode switching on RGB camera
                if name in self.env.cams:
                    self.mask_cams[name] = ('render_mode', self.env.cams[name])
                    print(f"[SegmentationRecorder] Using render mode switch for '{name}'")
    
    def _setup_writers(self):
        """Create video writers for each camera."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        for name in ['left', 'right', 'overhead', 'wrist', 'front']:
            path = os.path.join(self.output_dir, f"mask_{name}.mp4")
            writer = cv2.VideoWriter(path, fourcc, self.fps, self.resolution)
            self.writers[name] = writer
            print(f"[SegmentationRecorder] Writer: {path}")
    
    def _decode_mask(self, rgb):
        """Decode RGB to object handles."""
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        handles = (
            rgb[:, :, 0].astype(np.int32) +
            rgb[:, :, 1].astype(np.int32) * 256 +
            rgb[:, :, 2].astype(np.int32) * 256 * 256
        )
        return handles
    
    def _colorize(self, handle_mask):
        """Convert handle mask to colored image."""
        h, w = handle_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[:, :] = (18, 18, 18)
        
        for handle in np.unique(handle_mask):
            if handle <= 0:
                continue
            mask = handle_mask == handle
            color = self.handle_to_color.get(handle, (128, 128, 128))
            colored[mask] = color

        # Make boundaries explicit so tiny table objects are easier to spot.
        edges = np.zeros((h, w), dtype=bool)
        edges[1:, :] |= handle_mask[1:, :] != handle_mask[:-1, :]
        edges[:, 1:] |= handle_mask[:, 1:] != handle_mask[:, :-1]
        colored[edges] = (255, 255, 255)
        
        return colored
    
    def _capture_mask(self, cam_name):
        """Capture and colorize mask from camera."""
        from pyrep.const import RenderMode
        
        entry = self.mask_cams.get(cam_name)
        if entry is None:
            return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        mode, cam = entry
        
        if mode == 'mask':
            # Dedicated mask camera
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            handle_mask = self._decode_mask(rgb)
            colored = self._colorize(handle_mask)
            return colored
        
        elif mode == 'render_mode':
            # Switch render mode to get handles
            try:
                original = cam.get_render_mode()
                cam.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
                cam.handle_explicitly()
                rgb = cam.capture_rgb()
                # IMMEDIATELY restore to avoid red tint in CoppeliaSim viewport
                cam.set_render_mode(original)
                cam.handle_explicitly()  # Re-render with normal mode
                
                handle_mask = self._decode_mask(rgb)
                colored = self._colorize(handle_mask)
                return colored
            except Exception as e:
                # Fallback to RGB
                try:
                    cam.set_render_mode(RenderMode.OPENGL3)  # Force restore
                except:
                    pass
                cam.handle_explicitly()
                rgb = cam.capture_rgb()
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
    
    def record_step(self):
        """Record one frame from all cameras."""
        for name, writer in self.writers.items():
            frame = self._capture_mask(name)
            
            # Resize if needed
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
            
            writer.write(frame)
        
        self.frame_count += 1
    
    def release(self):
        """Close all video writers."""
        for name, writer in self.writers.items():
            writer.release()
        
        print(f"\n[SegmentationRecorder] Saved {self.frame_count} frames")
        print(f"[SegmentationRecorder] Videos in: {os.path.abspath(self.output_dir)}")
        for name in self.writers:
            print(f"  - mask_{name}.mp4")


if __name__ == "__main__":
    print("Import this and use SegmentationVideoRecorder")
    print("Example:")
    print("  recorder = SegmentationVideoRecorder(env)")
    print("  recorder.record_step()  # call every frame")
    print("  recorder.release()      # when done")
