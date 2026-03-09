"""
Run orchestrator and record SEGMENTATION MASK videos.

Output:
  - orchestrator_videos/    (RGB videos)
  - segmentation_videos/    (Mask videos)

Usage:
    python run_record_segmentation.py
"""
import os
import sys

# Qt setup
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
from segmentation_video_recorder import SegmentationVideoRecorder
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import go_home, run_standard_pick_place, run_cupboard_pick_place, run_box_pick_place, run_open_box


def main():
    env = ENV
    pr = env.pr
    
    print("=" * 60)
    print("RECORDING RGB + SEGMENTATION MASK VIDEOS")
    print("=" * 60)
    
    # RGB video recorder - set as global in orchestrator
    rgb_recorder = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
    gt_orch.VIDEO_RECORDER = rgb_recorder
    
    # Segmentation mask recorder - set as global in orchestrator
    mask_recorder = SegmentationVideoRecorder(env, output_dir="segmentation_videos", fps=30)
    gt_orch.MASK_RECORDER = mask_recorder
    
    # Settle
    print("\nSettling physics...")
    for _ in range(50):
        pr.step()
        rgb_recorder.record_step()
        mask_recorder.record_step()
    
    env.set_robot_conf(env.get_home_conf())
    gt_orch.step_and_record(pr, 10)
    
    results = []
    
    try:
        # Task 1: Cupboard mug
        print("\n--- Task 1: mug3 (cupboard) ---")
        ok = run_cupboard_pick_place(env, 'mug3', 'placement_boundary', "T1")
        results.append(("mug3", ok))
        go_home(env)
        gt_orch.step_and_record(pr, 5)
        
        # Task 2: Groceries
        for item, dest in [('soup', 'cupboard_boundary'), 
                           ('mustard', 'cupboard_boundary'),
                           ('spam', 'cupboard_boundary'), 
                           ('sugar', 'cupboard_boundary_top'),
                           ('crackers', 'cupboard_boundary_top')]:
            print(f"\n--- Task 2: {item} ---")
            ok = run_standard_pick_place(env, item, dest, f"T2-{item}")
            results.append((item, ok))
            go_home(env)
            gt_orch.step_and_record(pr, 5)
        
        # Task 3: mug2
        print("\n--- Task 3: mug2 (box top) ---")
        ok = run_box_pick_place(env, 'mug2', 'placement_boundary', "T3")
        results.append(("mug2", ok))
        go_home(env)
        gt_orch.step_and_record(pr, 5)
        
        # Task 4: Open box
        print("\n--- Task 4: Open box ---")
        ok = run_open_box(env, "T4")
        results.append(("open_box", ok))
        gt_orch.step_and_record(pr, 20)
        go_home(env)
        gt_orch.step_and_record(pr, 5)
        
        # Task 5: mug4
        print("\n--- Task 5: mug4 (inside box) ---")
        ok = run_box_pick_place(env, 'mug4', 'placement_boundary', "T5")
        results.append(("mug4", ok))
        go_home(env)
        gt_orch.step_and_record(pr, 5)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"\nTotal: {sum(1 for _, ok in results if ok)}/{len(results)}")
    
    # Release recorders
    rgb_recorder.release()
    mask_recorder.release()
    
    print("\n" + "=" * 60)
    print("VIDEOS SAVED")
    print("=" * 60)
    print("RGB videos:  orchestrator_videos/")
    print("Mask videos: segmentation_videos/")
    print("  - mask_left.mp4")
    print("  - mask_right.mp4")
    print("  - mask_overhead.mp4")
    print("  - mask_wrist.mp4")
    print("  - mask_front.mp4")
    
    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
