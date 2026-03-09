"""
Proper Segmentation Mask Viewer

This module captures ACTUAL segmentation masks from RLBench cameras,
decodes object handles from the RGB-encoded mask data, and displays
which objects are truly visible in each camera view.

How RLBench segmentation works:
- Cameras can render in "handle-coded" mode where each pixel's RGB value
  encodes the object handle: handle = R + G*256 + B*256*256
- By decoding these, we know exactly which objects are visible
- Objects occluded (like mug4 inside closed box) won't appear in the mask!

This is TRUE vision-based object detection, not hardcoded lists.
"""

import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.const import RenderMode
import colorsys
import time


class SegmentationMaskViewer:
    """
    Captures and displays actual segmentation masks from cameras.
    Detects objects based on what's VISIBLE in masks, not hardcoded lists.
    """
    
    def __init__(self, env, output_dir="segmentation_output"):
        self.env = env
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Build handle -> name mapping by scanning the scene
        self.handle_to_name = {}
        self.name_to_handle = {}
        self._build_handle_mapping()
        
        # Generate unique colors for each object (for visualization)
        self.handle_to_color = {}
        self._generate_colors()
        
        # Store original render modes to restore later
        self.original_render_modes = {}
        
        # Cache for mask cameras (avoid repeated lookups)
        self._mask_camera_cache = {}
        
        # Warning flags (avoid repeated warnings)
        self._warn_once = {}
        
        # Camera names
        self.camera_names = ['left', 'right', 'overhead', 'wrist', 'front']
        
        # Track what we've seen
        self.detected_objects_per_camera = {name: set() for name in self.camera_names}
        self.all_detected_objects = set()
        
        # Font for labels
        self.font = None
        self.font_small = None
        self._load_fonts()
        
        print(f"[SegmentationMaskViewer] Found {len(self.handle_to_name)} objects in scene")
        print(f"[SegmentationMaskViewer] Output: {os.path.abspath(output_dir)}")
    
    def _load_fonts(self):
        """Load fonts for text rendering."""
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def _build_handle_mapping(self):
        """
        Build mapping from object handles to names by scanning the scene.
        This is how we know what object each handle in the mask represents.
        """
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
                    self.name_to_handle[name] = h
                except:
                    continue
            
            print(f"[SegmentationMaskViewer] Scanned {len(self.handle_to_name)} shapes from scene")
            
        except Exception as e:
            print(f"[SegmentationMaskViewer] Error scanning scene: {e}")
    
    def _generate_colors(self):
        """Generate distinct colors for each object handle."""
        n_objects = len(self.handle_to_name)
        for i, handle in enumerate(self.handle_to_name.keys()):
            # Use HSV to generate distinct colors
            hue = i / max(n_objects, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            self.handle_to_color[handle] = (int(r * 255), int(g * 255), int(b * 255))
    
    def _decode_mask(self, rgb_image):
        """
        Decode RGB-encoded object handles from a segmentation mask.
        
        In RLBench/PyRep mask rendering:
        - Each pixel's RGB encodes an object handle
        - handle = R + G*256 + B*256*256
        
        Args:
            rgb_image: numpy array (H, W, 3) with values in [0, 1] or [0, 255]
            
        Returns:
            handle_mask: numpy array (H, W) of integer object handles
        """
        # Ensure uint8 [0, 255]
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        elif rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        
        # Decode handles
        handle_mask = (
            rgb_image[:, :, 0].astype(np.int32) +
            rgb_image[:, :, 1].astype(np.int32) * 256 +
            rgb_image[:, :, 2].astype(np.int32) * 256 * 256
        )
        
        return handle_mask
    
    def _get_objects_from_mask(self, handle_mask):
        """
        Extract unique objects from a decoded handle mask.
        
        Returns:
            set of object names visible in the mask
        """
        unique_handles = np.unique(handle_mask)
        
        detected = set()
        for h in unique_handles:
            if h <= 0:
                continue  # Background or invalid
            
            name = self.handle_to_name.get(h)
            if name:
                detected.add(name)
        
        return detected
    
    def _filter_interesting_objects(self, objects):
        """
        Filter detected objects to only show interesting task-related ones.
        Removes robot parts, environment fixtures, etc.
        
        Args:
            objects: set of object names
            
        Returns:
            set of filtered object names
        """
        # Keywords that indicate task-relevant objects
        interesting_keywords = [
            'mug', 'soup', 'mustard', 'spam', 'sugar', 'crackers', 
            'box', 'cupboard', 'table', 'bottle', 'can', 'food',
            'grocery', 'lid', 'boundary', 'placement'
        ]
        
        # Keywords to exclude (robot parts, fixtures, etc.)
        exclude_keywords = [
            'panda', 'gripper', 'link', 'joint', 'collision', 'respondable',
            'floor', 'wall', 'camera', 'cam', 'light', 'dummy', 'connection',
            'visible', 'visual_', '_visual', 'base_link', 'finger'
        ]
        
        filtered = set()
        for name in objects:
            name_lower = name.lower()
            
            # Skip if it matches an exclude pattern
            if any(kw in name_lower for kw in exclude_keywords):
                continue
            
            # Include if it matches an interesting pattern OR is in env.name_to_obj
            if any(kw in name_lower for kw in interesting_keywords):
                filtered.add(name)
            elif hasattr(self.env, 'name_to_obj') and name in self.env.name_to_obj:
                filtered.add(name)
        
        return filtered
    
    def _colorize_mask(self, handle_mask):
        """
        Convert a handle mask to a colorized visualization.
        Each object gets a unique color.
        
        Returns:
            PIL Image with colored segmentation
        """
        h, w = handle_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        unique_handles = np.unique(handle_mask)
        
        for handle in unique_handles:
            if handle <= 0:
                continue
            
            mask = handle_mask == handle
            color = self.handle_to_color.get(handle, (128, 128, 128))
            colored[mask] = color
        
        return Image.fromarray(colored)
    
    def _get_mask_camera_name(self, cam_name):
        """Get the expected mask camera name for a given camera."""
        if cam_name in ['left', 'right']:
            return f'cam_over_shoulder_{cam_name}_mask'
        else:
            return f'cam_{cam_name}_mask'
    
    def _try_get_mask_camera(self, cam_name):
        """Try to get a mask camera if it exists in the scene."""
        mask_cam_name = self._get_mask_camera_name(cam_name)
        
        if mask_cam_name not in self._mask_camera_cache:
            try:
                from pyrep.objects.vision_sensor import VisionSensor
                mask_cam = VisionSensor(mask_cam_name)
                mask_cam.set_explicit_handling(1)
                self._mask_camera_cache[mask_cam_name] = mask_cam
                print(f"[SegmentationMaskViewer] Found mask camera: {mask_cam_name}")
            except Exception:
                self._mask_camera_cache[mask_cam_name] = None
                
        return self._mask_camera_cache.get(mask_cam_name)
    
    def capture_segmentation_mask(self, cam_name):
        """
        Capture segmentation mask from a camera.
        
        Tries these methods in order:
        1. Use dedicated mask camera if it exists (e.g., cam_front_mask)
        2. Switch camera render mode to capture handles (OPENGL_COLOR_CODED)
        3. Fall back to RGB visualization if neither works
        
        Returns:
            (handle_mask, detected_objects, colorized_image)
        """
        cam = self.env.cams.get(cam_name)
        if cam is None:
            return None, set(), None
        
        try:
            # Method 1: Try dedicated mask camera
            mask_cam = self._try_get_mask_camera(cam_name)
            
            if mask_cam is not None:
                # Use actual mask camera!
                mask_cam.handle_explicitly()
                mask_rgb = mask_cam.capture_rgb()
                
                # Decode handles
                handle_mask = self._decode_mask(mask_rgb)
                detected = self._get_objects_from_mask(handle_mask)
                colorized = self._colorize_mask(handle_mask)
                
                return handle_mask, detected, colorized
            
            # Method 2: Try switching render mode
            # RenderMode values: OPENGL, OPENGL3, POV_RAY, EXTERNAL, OPENGL_COLOR_CODED
            # OPENGL_COLOR_CODED gives handle-coded output
            try:
                original_mode = cam.get_render_mode()
                
                # Try switching to handle-coded render mode
                from pyrep.const import RenderMode
                cam.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
                cam.handle_explicitly()
                mask_rgb = cam.capture_rgb()
                
                # Restore original mode
                cam.set_render_mode(original_mode)
                
                # Check if we got valid mask data (not all zeros/ones)
                if mask_rgb.max() > 0.01:
                    handle_mask = self._decode_mask(mask_rgb)
                    unique_handles = np.unique(handle_mask)
                    
                    # If we have multiple handles, this worked!
                    if len(unique_handles) > 1:
                        detected = self._get_objects_from_mask(handle_mask)
                        colorized = self._colorize_mask(handle_mask)
                        return handle_mask, detected, colorized
            except Exception as e:
                # RenderMode switch didn't work
                pass
            
            # Method 3: Fall back to RGB
            if self._warn_once.get(cam_name) is None:
                print(f"[Warning] No mask camera and render mode switch failed for '{cam_name}' - using RGB")
                self._warn_once[cam_name] = True
            
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            return None, set(), Image.fromarray(rgb_uint8)
            
        except Exception as e:
            print(f"[Error] Capturing mask from {cam_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, set(), None
    
    def capture_all_cameras(self):
        """
        Capture segmentation from all cameras.
        
        Returns:
            dict with camera_name -> (handle_mask, detected_objects, colorized_image)
        """
        results = {}
        all_detected = set()
        all_detected_raw = set()
        
        for cam_name in self.camera_names:
            handle_mask, detected, colorized = self.capture_segmentation_mask(cam_name)
            
            # Filter to interesting objects only
            detected_filtered = self._filter_interesting_objects(detected)
            
            results[cam_name] = {
                'mask': handle_mask,
                'detected': detected_filtered,
                'detected_raw': detected,  # Keep raw for debugging
                'colorized': colorized,
            }
            self.detected_objects_per_camera[cam_name] = detected_filtered
            all_detected.update(detected_filtered)
            all_detected_raw.update(detected)
        
        self.all_detected_objects = all_detected
        self.all_detected_objects_raw = all_detected_raw
        return results
    
    def create_visualization(self):
        """
        Create a combined visualization showing all camera masks and detected objects.
        """
        results = self.capture_all_cameras()
        
        # Image dimensions
        cam_w, cam_h = 320, 240
        panel_w = 300
        
        # Grid: 2 rows x 3 cols for cameras, plus panel
        grid_w = cam_w * 3
        grid_h = cam_h * 2
        total_w = grid_w + panel_w
        total_h = grid_h
        
        # Create canvas
        canvas = Image.new('RGB', (total_w, total_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)
        
        # Place camera images
        positions = {
            'left': (0, 0),
            'overhead': (cam_w, 0),
            'right': (cam_w * 2, 0),
            'front': (0, cam_h),
            'wrist': (cam_w, cam_h),
        }
        
        for cam_name, pos in positions.items():
            data = results.get(cam_name, {})
            colorized = data.get('colorized')
            
            if colorized:
                # Resize
                colorized = colorized.resize((cam_w, cam_h), Image.LANCZOS)
                canvas.paste(colorized, pos)
                
                # Add label
                draw.rectangle([pos[0], pos[1], pos[0] + 90, pos[1] + 20], fill=(0, 0, 0, 180))
                draw.text((pos[0] + 5, pos[1] + 2), cam_name.upper(), fill=(255, 255, 255), font=self.font_small)
                
                # Show count of detected objects
                count = len(data.get('detected', set()))
                draw.text((pos[0] + 5, pos[1] + cam_h - 18), f"{count} objects", fill=(200, 200, 200), font=self.font_small)
        
        # Create object panel
        panel_x = grid_w
        draw.rectangle([panel_x, 0, total_w, total_h], fill=(40, 40, 40))
        
        # Title
        draw.text((panel_x + 10, 10), "DETECTED OBJECTS", fill=(255, 255, 255), font=self.font)
        draw.text((panel_x + 10, 30), "(from segmentation masks)", fill=(150, 150, 150), font=self.font_small)
        
        # Separator
        draw.line([(panel_x + 10, 50), (total_w - 10, 50)], fill=(100, 100, 100))
        
        # Total count
        draw.text((panel_x + 10, 60), f"Total visible: {len(self.all_detected_objects)}", fill=(200, 200, 200), font=self.font_small)
        
        # List objects
        y = 85
        sorted_objects = sorted(self.all_detected_objects)
        
        for name in sorted_objects:
            if y > total_h - 20:
                draw.text((panel_x + 15, y), "...", fill=(150, 150, 150), font=self.font_small)
                break
            
            # Get handle and color
            handle = self.name_to_handle.get(name, 0)
            color = self.handle_to_color.get(handle, (150, 150, 150))
            
            # Draw colored bullet
            draw.ellipse([panel_x + 15, y + 2, panel_x + 23, y + 10], fill=color)
            
            # Draw name
            display_name = name[:25] if len(name) > 25 else name
            draw.text((panel_x + 28, y), display_name, fill=(255, 255, 255), font=self.font_small)
            
            y += 16
        
        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        draw.text((total_w - 70, total_h - 18), timestamp, fill=(100, 100, 100), font=self.font_small)
        
        return canvas
    
    def update(self):
        """Update visualization and save to file."""
        try:
            canvas = self.create_visualization()
            
            # Save current view
            output_path = os.path.join(self.output_dir, "segmentation_view.png")
            canvas.save(output_path)
            
            return True
        except Exception as e:
            print(f"[SegmentationMaskViewer] Update error: {e}")
            return False
    
    def save_snapshot(self, name):
        """Save a named snapshot."""
        canvas = self.create_visualization()
        path = os.path.join(self.output_dir, f"{name}.png")
        canvas.save(path)
        print(f"[SegmentationMaskViewer] Saved: {path}")
        return path
    
    def print_detected(self):
        """Print currently detected objects."""
        print(f"\n[Segmentation] Detected {len(self.all_detected_objects)} objects:")
        for cam_name in self.camera_names:
            objects = self.detected_objects_per_camera.get(cam_name, set())
            print(f"  {cam_name}: {len(objects)} objects")
        print(f"  Combined: {sorted(self.all_detected_objects)}")
    
    def get_detected_objects(self):
        """Get set of all detected object names."""
        return self.all_detected_objects.copy()
    
    def close(self):
        """Cleanup."""
        self.save_snapshot("final_segmentation")


# Test if run directly
if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use: python run_with_segmentation.py")
