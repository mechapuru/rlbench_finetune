"""
Segmentation Viewer - Shows camera feeds and detected objects in separate windows.

This module displays:
1. A combined window with all 5 camera views (RGB + object overlays)
2. A separate window listing all detected objects

Updates synchronously - call update() from the main simulation loop.
"""

import numpy as np
import os

# Force OpenCV to use GTK backend instead of Qt to avoid conflicts with CoppeliaSim
os.environ["QT_QPA_PLATFORM"] = ""  # Disable Qt platform
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Defer cv2 import to avoid Qt conflicts
_cv2 = None

def _get_cv2():
    global _cv2
    if _cv2 is None:
        # Try to use cv2 with non-Qt backend
        import cv2
        _cv2 = cv2
    return _cv2


class SegmentationViewer:
    """
    Displays camera views and detected objects in OpenCV windows.
    Must be updated from the main thread (call update() in simulation loop).
    """
    
    def __init__(self, env, window_scale=0.6):
        """
        Args:
            env: RLBenchKitchenEnv instance
            window_scale: Scale factor for camera images
        """
        self.env = env
        self.window_scale = window_scale
        self.initialized = False
        
        # Track visible objects
        self.visible_objects = set()
        self.box_is_open = False
        
        # Camera layout: 2 rows, 3 columns
        # Row 1: left, overhead, right
        # Row 2: front, wrist, (object list)
        self.camera_order = ['left', 'overhead', 'right', 'front', 'wrist']
        
        # Colors for object categories (BGR for OpenCV)
        self.colors = {
            'mug': (100, 100, 255),      # Red
            'grocery': (100, 255, 100),   # Green
            'box': (255, 100, 100),       # Blue
            'other': (200, 200, 200),     # Gray
        }
        
        # Font settings
        self.font = None  # Set after cv2 import
        self.font_scale = 0.5
        self.font_thickness = 1
        
    def _init_windows(self):
        """Initialize OpenCV windows."""
        cv2 = _get_cv2()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Create windows
        cv2.namedWindow("Camera Views", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        
        # Position windows
        cv2.moveWindow("Camera Views", 50, 50)
        cv2.moveWindow("Detected Objects", 1000, 50)
        
        self.initialized = True
    
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
        """Get currently visible objects based on geometric checks."""
        visible = set()
        self.box_is_open = self._check_box_open()
        
        for name, obj in self.env.name_to_obj.items():
            try:
                pos = obj.get_position()
                # Object visible if above table level
                if pos[2] > 0.5:
                    # mug4 only visible if box is open
                    if name in ['mug4', 'mug_inside_box']:
                        if self.box_is_open:
                            visible.add(name)
                    else:
                        visible.add(name)
            except:
                continue
        
        self.visible_objects = visible
        return visible
    
    def _capture_camera(self, cam_name):
        """Capture and process image from a camera."""
        cv2 = _get_cv2()
        
        try:
            cam = self.env.cams.get(cam_name)
            if cam is None:
                return None
            
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            rgb = (rgb * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Resize
            h, w = bgr.shape[:2]
            new_w = int(w * self.window_scale)
            new_h = int(h * self.window_scale)
            bgr = cv2.resize(bgr, (new_w, new_h))
            
            # Add camera label
            cv2.putText(bgr, cam_name.upper(), (10, 25), 
                       self.font, 0.7, (255, 255, 255), 2)
            cv2.putText(bgr, cam_name.upper(), (10, 25), 
                       self.font, 0.7, (0, 0, 0), 1)
            
            return bgr
            
        except Exception as e:
            # Return blank frame on error
            return np.zeros((int(480 * self.window_scale), 
                           int(640 * self.window_scale), 3), dtype=np.uint8)
    
    def _create_camera_grid(self):
        """Create a grid of all camera views."""
        cv2 = _get_cv2()
        
        frames = []
        for cam_name in self.camera_order:
            frame = self._capture_camera(cam_name)
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Get frame dimensions
        h, w = frames[0].shape[:2]
        
        # Create grid: 2 rows, 3 columns
        # Pad to 6 frames if needed
        while len(frames) < 6:
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        # Build rows
        row1 = np.hstack(frames[0:3])
        row2 = np.hstack(frames[3:6])
        
        grid = np.vstack([row1, row2])
        
        return grid
    
    def _create_object_panel(self):
        """Create a panel showing detected objects."""
        cv2 = _get_cv2()
        
        # Panel dimensions
        width = 350
        height = 600
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Title
        cv2.putText(panel, "DETECTED OBJECTS", (10, 30), 
                   self.font, 0.8, (255, 255, 255), 2)
        
        # Box status
        box_status = "OPEN" if self.box_is_open else "CLOSED"
        box_color = (0, 255, 0) if self.box_is_open else (0, 0, 255)
        cv2.putText(panel, f"Box: {box_status}", (10, 60), 
                   self.font, 0.6, box_color, 1)
        
        # Separator
        cv2.line(panel, (10, 75), (width - 10, 75), (100, 100, 100), 1)
        
        # Count
        cv2.putText(panel, f"Visible: {len(self.visible_objects)} objects", 
                   (10, 100), self.font, 0.5, (180, 180, 180), 1)
        
        # List objects by category
        y = 130
        
        # Group by category
        by_category = {'mug': [], 'grocery': [], 'box': [], 'other': []}
        for name in sorted(self.visible_objects):
            cat = self._categorize(name)
            by_category[cat].append(name)
        
        for category in ['mug', 'grocery', 'box', 'other']:
            objects = by_category[category]
            if not objects:
                continue
            
            if y > height - 40:
                break
            
            # Category header
            color = self.colors[category]
            cv2.putText(panel, f"{category.upper()}S:", (10, y), 
                       self.font, 0.5, color, 1)
            y += 20
            
            for name in objects:
                if y > height - 20:
                    cv2.putText(panel, "  ...", (20, y), 
                               self.font, 0.4, (150, 150, 150), 1)
                    break
                
                # Bullet point
                cv2.circle(panel, (25, y - 4), 4, color, -1)
                cv2.putText(panel, name, (35, y), 
                           self.font, 0.45, (255, 255, 255), 1)
                y += 18
            
            y += 10
        
        # Hidden objects notice (mug4 when box is closed)
        if not self.box_is_open:
            y = height - 60
            cv2.line(panel, (10, y - 10), (width - 10, y - 10), (100, 100, 100), 1)
            cv2.putText(panel, "HIDDEN (box closed):", (10, y + 5), 
                       self.font, 0.45, (100, 100, 200), 1)
            cv2.putText(panel, "  - mug4 (inside box)", (10, y + 25), 
                       self.font, 0.4, (100, 100, 200), 1)
        
        return panel
    
    def update(self):
        """
        Update the display windows.
        Call this from the main simulation loop.
        Returns True if windows are still open, False if user closed them.
        """
        cv2 = _get_cv2()
        
        if not self.initialized:
            self._init_windows()
        
        # Get visible objects
        self._get_visible_objects()
        
        # Create displays
        camera_grid = self._create_camera_grid()
        object_panel = self._create_object_panel()
        
        # Show windows
        cv2.imshow("Camera Views", camera_grid)
        cv2.imshow("Detected Objects", object_panel)
        
        # Process events (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        
        # Return False if 'q' pressed or windows closed
        if key == ord('q'):
            return False
        
        return True
    
    def close(self):
        """Close all windows."""
        cv2 = _get_cv2()
        cv2.destroyAllWindows()


def create_viewer(env):
    """Factory function to create a SegmentationViewer."""
    return SegmentationViewer(env)
