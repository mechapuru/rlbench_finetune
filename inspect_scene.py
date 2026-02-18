import os
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy

# Path to the scene file causing issues
SCENE_FILE = "/home/paddy/rrc/RLBench/RLBench/TAMP-PDDL/grill_task.ttt"

def inspect_scene():
    print(f"Launching PyRep with scene: {SCENE_FILE}")
    pr = PyRep()
    try:
        pr.launch(SCENE_FILE, headless=True)
        pr.start()
        
        print("\n--- Listing All Objects (Robust Loop) ---")
        
        # Iterate via handles
        i = 0
        while True:
            # Try getting object handle by index
            # simGetObjects(int index, int objectType)
            # sim_handle_all is -1 or similar constant
            
            try:
                # Try getting handle by index from the general list
                # Note: simGetObjects behavior varies by version. 
                # If we use sim_handle_all, it might return a LIST of handles?
                # or it iterates?
                # The modern PyRep backend wrapper maps simGetObjects to sim.simGetObjects
                # In standard CoppeliaSim API: 
                # sim.getObjectHandle(name) -> handle
                # sim.getObjects(index, type) -> handle (deprecated?)
                
                # Let's try the modern way if available: sim.simGetObjects(type) -> list[handle]
                pass
            except:
                pass

            i += 1
            if i > 5: break # Placeholder for loop logic check

        # CORRECT WAY: use sim.simGetObjects(sim.sim_handle_all) -> list of handles
        try:
            handles = sim.simGetObjects(sim.sim_handle_all)
            print(f"Found {len(handles)} objects in total.")
            
            sorted_objects = []
            for h in handles:
                try:
                    name = sim.simGetObjectAlias(h)
                    # Get position to help identify
                    pos = sim.simGetObjectPosition(h, -1)
                    pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                    sorted_objects.append((name, h, pos_str))
                except:
                    sorted_objects.append(("Unknown", h, "Unknown"))
            
            # Sort by name for readable output
            sorted_objects.sort(key=lambda x: x[0])
            
            print(f"{'Name':<30} | {'Handle':<10} | {'Position'}")
            print("-" * 60)
            for name, h, pos in sorted_objects:
                print(f"{name:<30} | {h:<10} | {pos}")
                
        except Exception as e:
            print(f"Error listing objects via simGetObjects: {e}")

        print("\n--- Testing Panda Class Instantiation ---")
        from pyrep.robots.arms.panda import Panda
        try:
            robot = Panda()
            print(f"Success! Panda object created: {robot}")
            print(f"Joints: {robot.get_joint_count()}")
        except Exception as e:
            print(f"FAILURE: Could not create Panda() object: {e}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pr.shutdown()

if __name__ == "__main__":
    inspect_scene()
