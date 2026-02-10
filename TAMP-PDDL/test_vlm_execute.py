#!/usr/bin/env python3
"""
Test VLM Execute Pipeline
=========================
Tests the integrated VLM -> Execution pipeline.

Usage:
    python test_vlm_execute.py --mock   # Test with mock VLM output
    python test_vlm_execute.py --file vlm_output.txt  # Test with file
"""

import os
import sys
import argparse

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parser():
    """Test the VLM output parser."""
    from vlm_execute_pipeline import parse_vlm_plan, ActionType
    
    print("="*60)
    print("TEST: VLM Output Parser")
    print("="*60)
    
    # Test 1: Standard pick-place format
    vlm_output1 = """
1. pick(mug_cupboard)
2. place(mug_cupboard, placement_boundary)
3. pick(soup)
4. place(soup, cupboard_boundary)
"""
    actions = parse_vlm_plan(vlm_output1)
    print(f"\nTest 1 - Standard format:")
    print(f"Input: {vlm_output1.strip()}")
    print(f"Parsed {len(actions)} actions:")
    for a in actions:
        print(f"  - {a}")
    assert len(actions) == 2, f"Expected 2 actions, got {len(actions)}"
    assert actions[0].object_name == 'mug3', f"Expected 'mug3', got '{actions[0].object_name}'"
    print("✓ PASS")
    
    # Test 2: With open_lid
    vlm_output2 = """
1. pick(mug_box)
2. place(mug_box, table)
3. open_lid(box_lid)
4. pick(mug_inside_box)
5. place(mug_inside_box, table)
"""
    actions = parse_vlm_plan(vlm_output2)
    print(f"\nTest 2 - With open_lid:")
    print(f"Input: {vlm_output2.strip()}")
    print(f"Parsed {len(actions)} actions:")
    for a in actions:
        print(f"  - {a}")
    assert len(actions) == 3, f"Expected 3 actions, got {len(actions)}"
    assert actions[1].action_type == ActionType.OPEN_LID
    print("✓ PASS")
    
    # Test 3: Object name mapping
    vlm_output3 = """
1. pick(mug_table)
2. place(mug_table, box_boundary)
"""
    actions = parse_vlm_plan(vlm_output3)
    print(f"\nTest 3 - Object mapping:")
    print(f"Input: {vlm_output3.strip()}")
    assert actions[0].object_name == 'mug1', f"Expected 'mug1', got '{actions[0].object_name}'"
    assert actions[0].target_region == 'box_boundary', f"Expected 'box_boundary', got '{actions[0].target_region}'"
    print("✓ PASS")
    
    print("\n" + "="*60)
    print("ALL PARSER TESTS PASSED")
    print("="*60)


def test_full_execution(vlm_file=None, use_mock=False):
    """Test full execution with environment."""
    
    print("\n" + "="*60)
    print("TEST: Full Execution Pipeline")
    print("="*60)
    
    # Get VLM output
    if vlm_file:
        with open(vlm_file, 'r') as f:
            vlm_output = f.read()
    elif use_mock:
        # Simple test case: pick mug1 and place on box
        vlm_output = """
1. pick(mug_table)
2. place(mug_table, box_boundary)
"""
    else:
        print("ERROR: Please provide --file or --mock")
        return
    
    print(f"VLM Output to execute:")
    print(vlm_output)
    
    # Import and execute
    os.environ["HEADLESS"] = "False"
    from vlm_execute_pipeline import VLMExecutePipeline
    from rlbench_kitchen_streams import ENV
    
    env = ENV
    pr = env.pr
    
    # Initialize
    print("\nInitializing environment...")
    for _ in range(50):
        pr.step()
    
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        pr.step()
    
    # Create and run pipeline
    pipeline = VLMExecutePipeline(env)
    successful, total = pipeline.execute_vlm_output(vlm_output, stop_on_failure=False)
    
    pipeline.print_summary()
    
    print(f"\nResult: {successful}/{total} actions successful")
    
    # Keep window open
    print("\nKeep window open for inspection. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass
    
    pr.stop()
    pr.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Test VLM Execute Pipeline")
    parser.add_argument("--parser-only", action="store_true", help="Only test parser (no env)")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM output for testing")
    parser.add_argument("--file", type=str, help="VLM output file to execute")
    args = parser.parse_args()
    
    if args.parser_only:
        test_parser()
    else:
        test_parser()
        test_full_execution(vlm_file=args.file, use_mock=args.mock)


if __name__ == "__main__":
    main()
