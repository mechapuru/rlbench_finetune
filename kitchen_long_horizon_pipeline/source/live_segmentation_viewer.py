"""
Live Segmentation Viewer using Multiprocessing

Runs the visualization in a separate process to avoid Qt conflicts.
Uses shared memory to pass image data between processes.
"""

import multiprocessing as mp
import numpy as np
import time
import os
import re
from multiprocessing import shared_memory
import signal
import sys


def _pick_mp_context():
    """Pick a stable process context for GUI viewer subprocesses."""
    for method in ("forkserver", "spawn", "fork"):
        try:
            return mp.get_context(method)
        except Exception:
            continue
    return mp.get_context()


# Viewer process - runs completely isolated from CoppeliaSim
def viewer_process(shm_name, shape, stop_event, title="Live Segmentation"):
    """
    Viewer process that displays images from shared memory.
    Uses OpenCV in its own process (no Qt conflicts).
    """
    import cv2
    
    # Connect to shared memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        img_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        print(f"[Viewer] Failed to connect to shared memory: {e}")
        return
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, shape[1], shape[0])
    
    print(f"[Viewer] Started - Press 'q' to quit")
    
    last_update = time.time()
    while not stop_event.is_set():
        try:
            # Read image from shared memory
            frame = img_array.copy()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Display
            cv2.imshow(title, frame_bgr)
            
            # Check for quit
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            
        except Exception as e:
            print(f"[Viewer] Error: {e}")
            break
    
    cv2.destroyAllWindows()
    shm.close()
    print("[Viewer] Closed")


class LiveSegmentationViewer:
    """
    Manages a live segmentation viewer in a separate process.
    
    Usage:
        viewer = LiveSegmentationViewer(env)
        viewer.start()
        
        # In your loop:
        viewer.update()
        
        # When done:
        viewer.stop()
    """
    
    def __init__(self, env, width=1260, height=480):
        self.env = env
        self.width = width
        self.height = height
        self.shape = (height, width, 3)
        
        # Shared memory for image data
        self.shm = None
        self.img_array = None
        
        # Viewer process
        self.viewer_proc = None
        self.stop_event = None
        self._mp_ctx = _pick_mp_context()
        
        # Segmentation helpers
        self.handle_to_name = {}
        self.name_to_handle = {}
        self.handle_to_color = {}
        self.handle_to_task_name = {}
        self._mask_camera_cache = {}
        self._warn_once = {}

        self.camera_names = ['left', 'right', 'overhead', 'wrist', 'front']
        self.detected_objects = set()
        self.object_camera_hits = {}
        self.object_pixel_counts = {}
        self.frame_idx = 0
        self.last_seen_frame = {}
        self.latest_composite_frame = None

        # Camera-mask fusion settings
        self.min_pixels_per_camera = int(os.environ.get("LIVE_SEG_MIN_PIXELS", "10"))
        self.persistence_frames = int(os.environ.get("LIVE_SEG_PERSIST_FRAMES", "10"))
        self.action_sequence = []
        self.current_action_index = -1
        self.current_action_label = ""

        # Task-object palette: vivid and high-contrast for table objects.
        self.task_objects = [
            "mug1", "mug2", "mug3", "mug4",
            "soup", "mustard", "spam", "sugar", "crackers",
            "box_lid", "cupboard",
        ]
        self.task_colors = {
            "mug1": (255, 70, 70),
            "mug2": (255, 160, 40),
            "mug3": (255, 235, 0),
            "mug4": (230, 80, 255),
            "soup": (0, 220, 220),
            "mustard": (170, 255, 0),
            "spam": (70, 150, 255),
            "sugar": (255, 100, 190),
            "crackers": (120, 80, 255),
            "box_lid": (255, 255, 255),
            "cupboard": (160, 160, 160),
        }
        self.display_names = {
            "mug1": "mug1 (table)",
            "mug2": "mug2 (box-top)",
            "mug3": "mug3 (cupboard)",
            "mug4": "mug4 (inside-box)",
            "soup": "soup",
            "mustard": "mustard",
            "spam": "spam",
            "sugar": "sugar",
            "crackers": "crackers",
            "box_lid": "box_lid",
            "cupboard": "cupboard",
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
        
        self._build_handle_mapping()
        self._build_task_handle_mapping()
        self._generate_colors()

    def set_action_sequence(self, actions):
        """Set full ordered action sequence displayed in the side panel."""
        if actions is None:
            self.action_sequence = []
            return
        self.action_sequence = [str(a) for a in actions]
        if self.current_action_index >= len(self.action_sequence):
            self.current_action_index = -1

    def set_current_action(self, action_index=None, action_label=None):
        """Set currently active action index (0-based) and optional label."""
        if action_index is None:
            self.current_action_index = -1
        else:
            try:
                self.current_action_index = int(action_index)
            except Exception:
                self.current_action_index = -1
        self.current_action_label = "" if action_label is None else str(action_label)
        
    def _build_handle_mapping(self):
        """Build handle -> name mapping from scene."""
        from pyrep.backend import sim
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
        except Exception as e:
            print(f"[LiveViewer] Error scanning scene: {e}")
    
    def _generate_colors(self):
        """Generate colors with task-object emphasis."""
        for handle in self.handle_to_name.keys():
            task_name = self.handle_to_task_name.get(handle)
            if task_name is not None:
                self.handle_to_color[handle] = self.task_colors.get(task_name, (255, 255, 255))
            else:
                self.handle_to_color[handle] = (45, 45, 45)

    def _canonical_task_name(self, scene_name):
        """Map low-level scene shape name to a stable task object name."""
        if not scene_name:
            return None

        n = scene_name.lower()
        if "mug_inside_box" in n or re.search(r"\bmug4\b", n):
            return "mug4"
        if "mug_cupboard" in n or re.search(r"\bmug3\b", n):
            return "mug3"
        if "mug_box" in n or re.search(r"\bmug2\b", n):
            return "mug2"
        if "mug_table" in n or re.search(r"\bmug1\b", n):
            return "mug1"
        if "box_lid" in n or n == "lid":
            return "box_lid"
        if "cupboard" in n:
            return "cupboard"
        for name in ("soup", "mustard", "spam", "sugar", "crackers"):
            if name in n:
                return name
        return None

    def _build_task_handle_mapping(self):
        """Pre-map every shape handle to a canonical task object (if any)."""
        # Primary mapping: gather all shape descendants of known env objects.
        try:
            from pyrep.backend import sim
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
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback mapping from raw shape names.
        for handle, scene_name in self.handle_to_name.items():
            if handle in self.handle_to_task_name:
                continue
            task_name = self._canonical_task_name(scene_name)
            if task_name is not None:
                self.handle_to_task_name[handle] = task_name

        counts = {name: 0 for name in self.task_objects}
        for task_name in self.handle_to_task_name.values():
            if task_name in counts:
                counts[task_name] += 1
        print(f"[LiveViewer] Task-handle mapping: {counts}")
    
    def start(self):
        """Start the viewer process."""
        # Create shared memory
        nbytes = int(np.prod(self.shape))
        self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self.img_array = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.img_array.fill(30)  # Dark gray background
        
        # Start viewer process
        self.stop_event = self._mp_ctx.Event()
        self.stop_event.clear()
        self.viewer_proc = self._mp_ctx.Process(
            target=viewer_process,
            args=(self.shm.name, self.shape, self.stop_event, "Live Segmentation Masks")
        )
        self.viewer_proc.start()
        
        print(f"[LiveViewer] Started (shared memory: {self.shm.name})")
        time.sleep(0.5)  # Let viewer initialize
    
    def stop(self):
        """Stop the viewer process."""
        if self.viewer_proc:
            self.stop_event.set()
            self.viewer_proc.join(timeout=2)
            if self.viewer_proc.is_alive():
                self.viewer_proc.terminate()
            self.viewer_proc = None
        
        if self.shm:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
        
        print("[LiveViewer] Stopped")
    
    def _decode_mask(self, rgb_image):
        """Decode RGB-encoded handles."""
        if rgb_image.dtype != np.uint8:
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image = rgb_image.astype(np.uint8)
        
        return (
            rgb_image[:, :, 0].astype(np.int32) +
            rgb_image[:, :, 1].astype(np.int32) * 256 +
            rgb_image[:, :, 2].astype(np.int32) * 256 * 256
        )
    
    def _get_objects_from_mask(self, handle_mask):
        """Extract canonical task objects and pixels from a handle mask."""
        detected_pixels = {}
        unique, counts = np.unique(handle_mask, return_counts=True)
        for h, pix in zip(unique, counts):
            if h <= 0:
                continue
            task_name = self.handle_to_task_name.get(int(h))
            if task_name is None:
                continue
            if int(pix) < self.min_pixels_per_camera:
                continue
            detected_pixels[task_name] = detected_pixels.get(task_name, 0) + int(pix)
        return detected_pixels
    
    def _colorize_mask(self, handle_mask):
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

        # White edges make small table objects visually obvious.
        edges = np.zeros((h, w), dtype=bool)
        edges[1:, :] |= handle_mask[1:, :] != handle_mask[:-1, :]
        edges[:, 1:] |= handle_mask[:, 1:] != handle_mask[:, :-1]
        colored[edges] = (255, 255, 255)
        
        return colored
    
    def _try_get_mask_camera(self, cam_name):
        """Try to get mask camera."""
        from pyrep.objects.vision_sensor import VisionSensor
        
        if cam_name in ['left', 'right']:
            mask_name = f'cam_over_shoulder_{cam_name}_mask'
        else:
            mask_name = f'cam_{cam_name}_mask'
        
        if mask_name not in self._mask_camera_cache:
            try:
                cam = VisionSensor(mask_name)
                cam.set_explicit_handling(1)
                self._mask_camera_cache[mask_name] = cam
            except:
                self._mask_camera_cache[mask_name] = None
        
        return self._mask_camera_cache.get(mask_name)
    
    def _capture_camera(self, cam_name):
        """Capture segmentation from one camera."""
        cam = self.env.cams.get(cam_name)
        if cam is None:
            return None, {}
        
        try:
            # Try mask camera first
            mask_cam = self._try_get_mask_camera(cam_name)
            
            if mask_cam:
                mask_cam.handle_explicitly()
                mask_rgb = mask_cam.capture_rgb()
                handle_mask = self._decode_mask(mask_rgb)
                detected = self._get_objects_from_mask(handle_mask)
                colorized = self._colorize_mask(handle_mask)
                return colorized, detected
            
            # Try render mode switch
            try:
                from pyrep.const import RenderMode
                original = cam.get_render_mode()
                cam.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
                cam.handle_explicitly()
                mask_rgb = cam.capture_rgb()
                cam.set_render_mode(original)
                
                if mask_rgb.max() > 0.01:
                    handle_mask = self._decode_mask(mask_rgb)
                    if len(np.unique(handle_mask)) > 1:
                        detected = self._get_objects_from_mask(handle_mask)
                        colorized = self._colorize_mask(handle_mask)
                        return colorized, detected
            except:
                pass
            
            # Fallback to RGB
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            rgb_uint8 = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
            return rgb_uint8, {}
            
        except Exception as e:
            return None, {}
    
    def _filter_objects(self, objects):
        """Compatibility shim; objects are already canonical task names."""
        return {o for o in objects if o in self.task_objects}
    
    def update(self):
        """Capture all cameras and update the shared image."""
        if self.img_array is None:
            return set()
        
        from PIL import Image, ImageDraw, ImageFont
        
        # Camera layout
        cam_w, cam_h = 240, 180
        panel_w = 620
        obj_panel_w = 300
        action_panel_x = cam_w * 3 + obj_panel_w
        
        # Capture all cameras and fuse object evidence.
        self.frame_idx += 1
        images = {}
        fused_pixels = {}
        fused_hits = {}

        for cam_name in self.camera_names:
            colorized, detected_pixels = self._capture_camera(cam_name)
            images[cam_name] = colorized

            for obj_name, pixels in detected_pixels.items():
                fused_pixels[obj_name] = fused_pixels.get(obj_name, 0) + int(pixels)
                fused_hits.setdefault(obj_name, []).append(cam_name)
                self.last_seen_frame[obj_name] = self.frame_idx

        # Keep objects for a short time even if briefly occluded.
        fused_visible = set()
        for obj_name in self.task_objects:
            if obj_name in fused_pixels:
                fused_visible.add(obj_name)
                continue
            last = self.last_seen_frame.get(obj_name)
            if last is not None and (self.frame_idx - last) <= self.persistence_frames:
                fused_visible.add(obj_name)

        self.detected_objects = self._filter_objects(fused_visible)
        self.object_camera_hits = fused_hits
        self.object_pixel_counts = fused_pixels
        
        # Build composite image
        grid_w = cam_w * 3
        grid_h = cam_h * 2
        total_w = grid_w + panel_w
        total_h = grid_h
        
        canvas = Image.new('RGB', (total_w, total_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font = font_small = ImageFont.load_default()
        
        # Place camera images
        positions = {
            'left': (0, 0),
            'overhead': (cam_w, 0),
            'right': (cam_w * 2, 0),
            'front': (0, cam_h),
            'wrist': (cam_w, cam_h),
        }
        
        for cam_name, pos in positions.items():
            img = images.get(cam_name)
            if img is not None:
                pil_img = Image.fromarray(img).resize((cam_w, cam_h), Image.LANCZOS)
                canvas.paste(pil_img, pos)
                
                # Label
                draw.rectangle([pos[0], pos[1], pos[0]+70, pos[1]+18], fill=(0,0,0,180))
                draw.text((pos[0]+4, pos[1]+2), cam_name.upper(), fill=(255,255,255), font=font_small)
        
        # Panel
        px = grid_w
        draw.rectangle([px, 0, total_w, total_h], fill=(40, 40, 40))
        draw.text((px+10, 10), "VISIBLE TASK OBJECTS", fill=(255,255,255), font=font)
        draw.text((px+10, 30), "(fused from all 5 masks)", fill=(150,150,150), font=font_small)
        draw.line([(px+10, 50), (px + obj_panel_w - 10, 50)], fill=(100,100,100))
        draw.text((px+10, 55), f"Count: {len(self.detected_objects)}", fill=(200,200,200), font=font_small)
        draw.text((px+10, 68), "Cams: L O R F W", fill=(130, 130, 130), font=font_small)
        
        # List objects
        y = 85
        for name in self.task_objects:
            if name not in self.detected_objects:
                continue
            if y > total_h - 20:
                draw.text((px+15, y), "...", fill=(150,150,150), font=font_small)
                break
            
            color = self.task_colors.get(name, (150, 150, 150))
            draw.ellipse([px+12, y+3, px+20, y+11], fill=color)
            
            hits = self.object_camera_hits.get(name, [])
            hit_flags = {
                "L": "L" if "left" in hits else ".",
                "O": "O" if "overhead" in hits else ".",
                "R": "R" if "right" in hits else ".",
                "F": "F" if "front" in hits else ".",
                "W": "W" if "wrist" in hits else ".",
            }
            cams_text = f"[{hit_flags['L']}{hit_flags['O']}{hit_flags['R']}{hit_flags['F']}{hit_flags['W']}]"
            seen_now = name in self.object_pixel_counts
            text_color = (255, 255, 255) if seen_now else (160, 160, 160)
            label = self.display_names.get(name, name)
            draw.text((px+25, y), f"{label:<16} {cams_text}", fill=text_color, font=font_small)
            y += 15

        # Divider between object panel and action panel.
        draw.line([(action_panel_x, 8), (action_panel_x, total_h - 8)], fill=(95, 95, 95), width=1)

        # Action sequence panel
        ax = action_panel_x + 10
        draw.text((ax, 10), "ACTION SEQUENCE", fill=(255, 255, 255), font=font)
        if self.current_action_label:
            draw.text((ax, 30), f"Now: {self.current_action_label}", fill=(185, 230, 185), font=font_small)
        else:
            draw.text((ax, 30), "Now: (idle)", fill=(150, 150, 150), font=font_small)
        draw.line([(ax, 50), (total_w - 10, 50)], fill=(100, 100, 100))

        ay = 60
        if self.action_sequence:
            for idx, action in enumerate(self.action_sequence):
                if ay > total_h - 18:
                    draw.text((ax, ay), "...", fill=(150, 150, 150), font=font_small)
                    break
                is_current = (idx == self.current_action_index)
                if is_current:
                    draw.rectangle(
                        [ax - 3, ay - 1, total_w - 12, ay + 13],
                        fill=(52, 95, 52),
                        outline=(95, 140, 95),
                    )
                text = f"{idx + 1}. {action}"
                tcolor = (255, 255, 255) if is_current else (210, 210, 210)
                draw.text((ax, ay), text, fill=tcolor, font=font_small)
                ay += 15
        else:
            draw.text((ax, ay), "No action sequence set.", fill=(150, 150, 150), font=font_small)
        
        # Timestamp
        ts = time.strftime("%H:%M:%S")
        draw.text((total_w-70, total_h-18), ts, fill=(100,100,100), font=font_small)
        
        # Copy to shared memory
        result = np.array(canvas.resize((self.width, self.height), Image.LANCZOS))
        self.latest_composite_frame = result.copy()
        np.copyto(self.img_array, result)
        
        return self.detected_objects
    
    def get_detected_objects(self):
        """Get current detected objects."""
        return self.detected_objects.copy()

    def get_latest_frame(self):
        """Get the most recently rendered composite panel frame (RGB)."""
        if self.latest_composite_frame is None:
            return None
        return self.latest_composite_frame.copy()


if __name__ == "__main__":
    print("This module should be imported.")
    print("Use: python run_live_segmentation.py")
