from pyrep import PyRep
from pyrep.backend import sim

SCENE_FILE = "/home/naren/iiith/Long_Horizon/TAMP-PDDL/task_design_proposal_variation_1.ttt"

def main():
    pr = PyRep()
    pr.launch(SCENE_FILE, headless=True)
    pr.start()
    
    print("--- Shapes in Scene ---")
    # sim_object_shape_type is 1
    handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_shape_type, 0)
    for h in handles:
        name = sim.simGetObjectName(h)
        print(f"Shape: {name}")
        
    print("\n--- Dummies in Scene ---")
    # sim_object_dummy_type is 6
    handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_dummy_type, 0)
    for h in handles:
        name = sim.simGetObjectName(h)
        print(f"Dummy: {name}")

    pr.stop()
    pr.shutdown()

if __name__ == "__main__":
    main()
