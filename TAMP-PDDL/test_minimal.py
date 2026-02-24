import numpy as np

# NO Custom Qt Environment Variables Here!

from pyrep import PyRep

def test_launch():
    print("Launching PyRep exactly like RLBench does natively...")
    pr = PyRep()
    # Try to launch the kitchen scene
    SCENE_FILE = "/home/paddy/rrc/RLBench/RLBench/pddlstream execution/kitchen_task/task_design_proposal_variation_1.ttt"
    pr.launch(SCENE_FILE, headless=False)
    pr.start()
    
    print("PyRep is running. Stepping 50 times...")
    for i in range(50):
        pr.step()
        
    print("Done. Shutting down...")
    pr.shutdown()
    
if __name__ == "__main__":
    test_launch()
