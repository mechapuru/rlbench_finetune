"""
Run orchestrator with LIVE Tkinter segmentation view.
Tkinter uses Tcl/Tk - completely separate from Qt!

Usage:
    python run_tkinter_segmentation.py
"""
import os
import sys

# Qt config
os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
os.environ.pop("QT_PLUGIN_PATH", None)
coppelia = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
for p in [os.path.join(coppelia, "platforms"), os.path.join(coppelia, "Qt/plugins/platforms")]:
    if os.path.isdir(p):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", p)
        break

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))
os.environ["HEADLESS"] = "False"

from rlbench_kitchen_streams import ENV
from video_recorder import VideoRecorder
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import go_home, run_standard_pick_place, run_cupboard_pick_place, run_box_pick_place, run_open_box

from tkinter_segmentation_viewer import TkinterSegmentationViewer


def main():
    env = ENV
    pr = env.pr
    
    print("="*60)
    print("LIVE SEGMENTATION VIEW (Tkinter)")
    print("="*60)
    
    # Video
    video = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
    gt_orch.VIDEO_RECORDER = video
    
    # Live viewer
    viewer = TkinterSegmentationViewer(env, width=1020, height=360)
    viewer.start()
    
    # Settle
    for _ in range(50):
        pr.step()
        video.record_step()
    env.set_robot_conf(env.get_home_conf())
    for _ in range(10):
        pr.step()
        video.record_step()
    
    viewer.update()
    
    print("\nInitial detection:")
    detected = viewer.get_detected_objects()
    print(f"  {len(detected)} objects: {sorted(detected)}")
    print(f"  mug4 visible: {'YES' if 'mug4' in detected else 'NO'}")
    
    results = []
    update_counter = 0
    
    def step(n=1):
        nonlocal update_counter
        for _ in range(n):
            pr.step()
            video.record_step()
            update_counter += 1
            if update_counter >= 3:
                viewer.update()
                update_counter = 0
    
    try:
        # Task 1
        print("\n--- Task 1: mug3 ---")
        ok = run_cupboard_pick_place(env, 'mug3', 'placement_boundary', "T1")
        results.append(("mug3", ok))
        go_home(env); step(5)
        
        # Task 2: Groceries
        for item, dest in [('soup','cupboard_boundary'), ('mustard','cupboard_boundary'), 
                           ('spam','cupboard_boundary'), ('sugar','cupboard_boundary_top'),
                           ('crackers','cupboard_boundary_top')]:
            print(f"\n--- Task 2: {item} ---")
            ok = run_standard_pick_place(env, item, dest, f"T2-{item}")
            results.append((item, ok))
            go_home(env); step(5)
        
        # Task 3
        print("\n--- Task 3: mug2 ---")
        ok = run_box_pick_place(env, 'mug2', 'placement_boundary', "T3")
        results.append(("mug2", ok))
        go_home(env); step(5)
        
        # Task 4: Open box
        print("\n" + "="*60)
        print("OPENING BOX - watch for mug4!")
        print("="*60)
        before = viewer.get_detected_objects()
        print(f"Before: mug4 = {'YES' if 'mug4' in before else 'NO'}")
        
        ok = run_open_box(env, "T4-OpenBox")
        results.append(("open_box", ok))
        
        for _ in range(20):
            pr.step()
            video.record_step()
        viewer.update()
        
        after = viewer.get_detected_objects()
        print(f"After: mug4 = {'YES' if 'mug4' in after else 'NO'}")
        new = after - before
        if new:
            print(f"★ NEW: {sorted(new)}")
        
        go_home(env); step(5)
        
        # Task 5
        print("\n--- Task 5: mug4 ---")
        ok = run_box_pick_place(env, 'mug4', 'placement_boundary', "T5")
        results.append(("mug4", ok))
        go_home(env); step(5)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for name, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    
    video.release()
    
    print("\nDone. Close the viewer window or Ctrl+C to exit.")
    try:
        while True:
            pr.step()
            viewer.update()
    except KeyboardInterrupt:
        pass
    
    viewer.stop()
    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
