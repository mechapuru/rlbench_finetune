import time
import argparse
import sys
import os

# Ensure CoppeliaSim ZMQ Remote API is available
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("Error: coppeliasim_zmqremoteapi_client not installed.")
    print("Please run: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)


class ZMQDynamicExtractor:
    """
    Connects to a running RLBench/CoppeliaSim simulation via ZeroMQ 
    and extracts the semantic open-world state dynamically.
    """
    def __init__(self, host='localhost', port=23000):
        print(f"Connecting to CoppeliaSim ZMQ API at {host}:{port}...")
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require('sim')
        print("Connected successfully.")
        
        self.regions = [
            {'name': 'table', 'description': 'main dining table surface'},
            {'name': 'box-top', 'description': 'top surface of the closed box'},
            {'name': 'box-inside', 'description': 'interior of the box (accessible when lid is open)'},
            {'name': 'placement_boundary', 'description': 'target area for placing mugs'},
            {'name': 'cupboard_boundary', 'description': 'inside the cupboard shelf'},
        ]

    def _is_dynamic_shape(self, handle: int) -> bool:
        """Check if a PyRep handle is a movable shape (not a wall/floor)."""
        try:
            obj_type = self.sim.getObjectType(handle)
            if obj_type != self.sim.object_shape_type:
                return False
            # Return true if shape is NOT static
            is_static = self.sim.getObjectInt32Param(handle, self.sim.shapeintparam_static)
            return is_static == 0
        except:
            return False

    def get_scene_objects(self):
        found_objects = []
        try:
            # -1 gets all objects
            handles = self.sim.getObjects(self.sim.handle_all)
            for h in handles:
                if self._is_dynamic_shape(h):
                    name = self.sim.getObjectName(h)
                    name_lower = name.lower()
                    
                    # Filter out robot parts and visual artifacts
                    if "visual" not in name_lower and "respondable" not in name_lower and "panda" not in name_lower and "contact" not in name_lower and "force" not in name_lower and "prox_sensor" not in name_lower:
                        found_objects.append((name, h))
            
            # Explicitly add box lid if present
            try:
                lid_handle = self.sim.getObject('/box_lid')
                if lid_handle != -1 and not any(n == 'box_lid' for n, _ in found_objects):
                    found_objects.append(('box_lid', lid_handle))
            except:
                pass
                
        except Exception as e:
            print(f"Error discovering objects: {e}")
            
        return found_objects

    def check_ik_accessibility(self, handle, pos):
        """Simple heuristic Z-barrier check (similar to PyRep version)"""
        try:
            obj_name = self.sim.getObjectName(handle)
            if "mug4" in obj_name or "mug_inside_box" in obj_name:
                try:
                    lid_h = self.sim.getObject('/box_lid')
                    if lid_h != -1:
                        lid_pos = self.sim.getObjectPosition(lid_h, -1)
                        if lid_pos[2] < 0.85: # Lid is closed
                            return False, "box_lid"
                except:
                    pass
            
            # Check if any OTHER object is directly on top of it
            all_objs = self.get_scene_objects()
            for other_name, other_h in all_objs:
                if other_h != handle and other_name != 'box_lid':
                    other_pos = self.sim.getObjectPosition(other_h, -1)
                    if other_pos[2] > pos[2] and \
                       abs(other_pos[0] - pos[0]) < 0.1 and \
                       abs(other_pos[1] - pos[1]) < 0.1:
                        return False, other_name
                        
            return True, None
        except:
            return True, None

    def extract_state(self):
        print("\n--- EXTRACTING SCENE STATE ---")
        objects_data = []
        
        # Assess lid state
        lid_state = "closed"
        try:
            lid_h = self.sim.getObject('/box_lid')
            box_h = self.sim.getObject('/box_base')
            if lid_h != -1 and box_h != -1:
                l_pos = self.sim.getObjectPosition(lid_h, -1)
                b_pos = self.sim.getObjectPosition(box_h, -1)
                if abs(l_pos[0] - b_pos[0]) > 0.10:
                    lid_state = "open"
        except:
            pass

        scene_objs = self.get_scene_objects()
        
        for name, handle in scene_objs:
            if name == 'box_lid':
                objects_data.append(f"- {name}: state={lid_state}, location=on box, STATUS=BLOCKED_BY_None")
                continue
                
            pos = self.sim.getObjectPosition(handle, -1)
            location = "on table"
            
            if pos[2] > 0.85 and 0.0 < pos[0] < 0.4:
                location = "on closed box"
            elif pos[2] < 0.85 and 0.0 < pos[0] < 0.4:
                location = "inside box"
                
            is_reachable, blocker = self.check_ik_accessibility(handle, pos)
            status = "BLOCKED_BY_None" if is_reachable else f"BLOCKED_BY_{blocker}"
            
            objects_data.append(f"- {name}: location={location}, STATUS={status}")

        # Format output
        lines = ["=== CURRENT SEMANTIC STATE ===", ""]
        lines.append("## Robot Status:")
        lines.append("- gripper: empty\n")
        
        lines.append("## Dynamically Discovered Objects:")
        lines.extend(objects_data)
        lines.append("")
        
        lines.append("## Valid Target Regions:")
        for r in self.regions:
            lines.append(f"- {r['name']}: {r['description']}")
            
        result = "\n".join(lines)
        print(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Continuously monitor state")
    args = parser.parse_args()
    
    extractor = ZMQDynamicExtractor()
    
    if args.loop:
        try:
            while True:
                os.system('clear')
                extractor.extract_state()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting ZMQ Monitor.")
    else:
        extractor.extract_state()
