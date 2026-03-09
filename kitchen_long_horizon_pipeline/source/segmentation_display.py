"""
Segmentation Display - Shows camera feeds and detected objects.

Saves visualization images to a folder that can be viewed with any image viewer.
Also creates a simple tkinter window (no Qt conflicts) for live display.

Usage:
    viewer = SegmentationDisplay(env)
    viewer.update()  # Call in simulation loop
"""

import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import time


class SegmentationDisplay:
    """
    Displays camera views and detected objects.
    Saves images to output folder for viewing.
    """
    
    def __init__(self, env, output_dir="tracker_output", save_interval=10):
        """
        Args:
            env: RLBenchKitchenEnv instance
            output_dir: Directory to save output images
            save_interval: Save images every N updates
        """
        self.env = env
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.update_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track visible objects
        self.visible_objects = set()
        self.box_is_open = False
        
        # Camera order for grid
        self.camera_order = ['left', 'overhead', 'right', 'front', 'wrist']
        
        # Colors for categories (RGB)
        self.colors = {
            'mug': (255, 100, 100),
            'grocery': (100, 255, 100),
            'box': (100, 100, 255),
            'other': (200, 200, 200),
        }
        
        # Try to load a font
        self.font = None
        self.font_small = None
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
                self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 14)
            except:
                self.font = ImageFont.load_default()
                self.font_small = ImageFont.load_default()
        
        print(f"[SegmentationDisplay] Output directory: {os.path.abspath(output_dir)}")
        print(f"[SegmentationDisplay] View images with: eog {output_dir}/current_view.png")
        
    def _categorize(self, name):
        """Categorize object for coloring."""
        name_lower = name.lower()
        if 'mug' in name_lower:
            return 'mug'
        elif any(x in name_lower for x in ['soup', 'mustard', 'spam', 'sugar', 'crackers', 
                                            'bottle', 'can', 'tin', 'food', 'cereal']):
            return 'grocery'
        elif 'box' in name_lower or 'lid' in name_lower:
            return 'box'
        return 'other'
    
    def _check_box_open(self):
        """Check if box lid is open."""
        try:
            lid = self.env.get_object('box_lid')
            box = self.env.get_object('box_base')
            if lid and box:
                lid_pos = lid.get_position()
                box_pos = box.get_position()
                offset = abs(lid_pos[0] - box_pos[0])
                return offset >= 0.08
        except:
            pass
        return False
    
    def _get_visible_objects(self):
        """Get currently visible objects."""
        visible = set()
        self.box_is_open = self._check_box_open()
        
        for name, obj in self.env.name_to_obj.items():
            try:
                pos = obj.get_position()
                if pos[2] > 0.5:
                    if name in ['mug4', 'mug_inside_box']:
                        if self.box_is_open:
                            visible.add(name)
                    else:
                        visible.add(name)
            except:
                continue
        
        self.visible_objects = visible
        return visible
    
    def _capture_camera_pil(self, cam_name):
        """Capture camera image as PIL Image."""
        try:
            cam = self.env.cams.get(cam_name)
            if cam is None:
                return None
            
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            rgb = (rgb * 255).astype(np.uint8)
            
            # Convert to PIL
            img = Image.fromarray(rgb)
            
            # Resize
            new_size = (320, 240)
            img = img.resize(new_size, Image.LANCZOS)
            
            # Add label
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 80, 22], fill=(0, 0, 0, 180))
            draw.text((5, 2), cam_name.upper(), fill=(255, 255, 255), font=self.font_small)
            
            return img
            
        except Exception as e:
            # Return blank image
            return Image.new('RGB', (320, 240), (50, 50, 50))
    
    def _create_camera_grid(self):
        """Create a grid of all camera views."""
        frames = []
        for cam_name in self.camera_order:
            frame = self._capture_camera_pil(cam_name)
            if frame:
                frames.append(frame)
        
        if not frames:
            return Image.new('RGB', (960, 480), (30, 30, 30))
        
        # Grid: 2 rows, 3 columns (320x240 each)
        w, h = 320, 240
        grid = Image.new('RGB', (w * 3, h * 2), (30, 30, 30))
        
        positions = [(0, 0), (w, 0), (w*2, 0), (0, h), (w, h)]
        for i, frame in enumerate(frames[:5]):
            grid.paste(frame, positions[i])
        
        return grid
    
    def _create_object_panel(self):
        """Create a panel showing detected objects."""
        width, height = 300, 480
        panel = Image.new('RGB', (width, height), (40, 40, 40))
        draw = ImageDraw.Draw(panel)
        
        # Title
        draw.text((10, 10), "DETECTED OBJECTS", fill=(255, 255, 255), font=self.font)
        
        # Box status
        y = 40
        box_status = "OPEN" if self.box_is_open else "CLOSED"
        box_color = (100, 255, 100) if self.box_is_open else (255, 100, 100)
        draw.text((10, y), f"Box: {box_status}", fill=box_color, font=self.font_small)
        
        # Separator
        y += 25
        draw.line([(10, y), (width - 10, y)], fill=(100, 100, 100))
        
        # Count
        y += 10
        draw.text((10, y), f"Visible: {len(self.visible_objects)} objects", 
                 fill=(180, 180, 180), font=self.font_small)
        
        # Group by category
        y += 30
        by_category = {'mug': [], 'grocery': [], 'box': [], 'other': []}
        for name in sorted(self.visible_objects):
            cat = self._categorize(name)
            by_category[cat].append(name)
        
        for category in ['mug', 'grocery', 'box', 'other']:
            objects = by_category[category]
            if not objects:
                continue
            
            if y > height - 60:
                break
            
            color = self.colors[category]
            draw.text((10, y), f"{category.upper()}S:", fill=color, font=self.font_small)
            y += 18
            
            for name in objects:
                if y > height - 40:
                    draw.text((20, y), "...", fill=(150, 150, 150), font=self.font_small)
                    break
                
                # Bullet
                draw.ellipse([15, y + 2, 23, y + 10], fill=color)
                draw.text((30, y), name, fill=(255, 255, 255), font=self.font_small)
                y += 16
            
            y += 8
        
        # Hidden objects
        if not self.box_is_open:
            y = height - 50
            draw.line([(10, y - 5), (width - 10, y - 5)], fill=(100, 100, 100))
            draw.text((10, y), "HIDDEN (box closed):", fill=(150, 100, 100), font=self.font_small)
            draw.text((15, y + 18), "• mug4 (inside box)", fill=(150, 100, 100), font=self.font_small)
        
        return panel
    
    def _create_combined_image(self):
        """Create combined visualization."""
        camera_grid = self._create_camera_grid()
        object_panel = self._create_object_panel()
        
        # Combine side by side
        total_width = camera_grid.width + object_panel.width
        total_height = max(camera_grid.height, object_panel.height)
        
        combined = Image.new('RGB', (total_width, total_height), (30, 30, 30))
        combined.paste(camera_grid, (0, 0))
        combined.paste(object_panel, (camera_grid.width, 0))
        
        # Add timestamp
        draw = ImageDraw.Draw(combined)
        timestamp = time.strftime("%H:%M:%S")
        draw.text((total_width - 80, total_height - 20), timestamp, 
                 fill=(150, 150, 150), font=self.font_small)
        
        return combined
    
    def update(self):
        """Update visualization and save to file."""
        self.update_count += 1
        
        # Get visible objects
        self._get_visible_objects()
        
        # Only save periodically to avoid disk IO overhead
        if self.update_count % self.save_interval == 0:
            combined = self._create_combined_image()
            
            # Save current view
            output_path = os.path.join(self.output_dir, "current_view.png")
            combined.save(output_path)
            
            # Also save timestamped version periodically
            if self.update_count % (self.save_interval * 10) == 0:
                ts_path = os.path.join(self.output_dir, f"view_{self.update_count:06d}.png")
                combined.save(ts_path)
        
        return True
    
    def save_snapshot(self, name="snapshot"):
        """Save a named snapshot."""
        self._get_visible_objects()
        combined = self._create_combined_image()
        
        output_path = os.path.join(self.output_dir, f"{name}.png")
        combined.save(output_path)
        print(f"[SegmentationDisplay] Saved: {output_path}")
        
        return output_path
    
    def get_status(self):
        """Get current visibility status as dict."""
        return {
            'box_open': self.box_is_open,
            'visible_count': len(self.visible_objects),
            'visible_objects': sorted(self.visible_objects),
            'mug4_visible': 'mug4' in self.visible_objects or 'mug_inside_box' in self.visible_objects,
        }
    
    def print_status(self):
        """Print current status to console."""
        status = self.get_status()
        print(f"\n[Visibility] Box: {'OPEN' if status['box_open'] else 'CLOSED'} | "
              f"Visible: {status['visible_count']} | "
              f"mug4: {'YES' if status['mug4_visible'] else 'NO (hidden)'}")
    
    def close(self):
        """Cleanup."""
        # Save final view
        self.save_snapshot("final_view")
        print(f"[SegmentationDisplay] Final images saved to: {self.output_dir}/")
