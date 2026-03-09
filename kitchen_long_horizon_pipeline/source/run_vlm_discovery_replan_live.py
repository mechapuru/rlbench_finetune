#!/usr/bin/env python3
"""
Run VLM replanning pipeline with discovery-triggered failure/replan in live view mode.

Defaults are tuned for:
- visible objects only from segmentation
- live segmentation window enabled
- replan when new object appears after open-lid completes
- explicit failure code NEW_OBJECT_INTRODUCED_IN_SCENE
"""

import argparse
import os


def _configure_qt():
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    for candidate in [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
    ]:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break


def main():
    _configure_qt()

    from vlm_pipeline.vlm_with_replanning import (
        ReplanConfig,
        VLMReplanningPipeline,
        DISCOVERY_FAILURE_CODE_DEFAULT,
    )

    parser = argparse.ArgumentParser(description="Live VLM discovery-triggered replanning runner")
    parser.add_argument(
        "--goal",
        type=str,
        default=(
            "Using only currently visible objects from segmentation, move all visible mugs to "
            "placement_boundary and move visible groceries to cupboard regions. If any new object "
            "becomes visible later, replan and continue until the task is complete."
        ),
        help="Natural language goal for VLM",
    )
    parser.add_argument("--max-replans", type=int, default=3, help="Maximum replan attempts")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM")
    parser.add_argument("--remote", action="store_true", help="Use remote VLM server")
    parser.add_argument("--remote-url", type=str, default="", help="Remote VLM URL")
    parser.add_argument("--text-only", action="store_true", help="Disable vision input to VLM")
    parser.add_argument(
        "--discovery-targets",
        type=str,
        default="",
        help="Comma-separated discovery targets (empty = any newly visible object)",
    )
    parser.add_argument("--live-mask-stride", type=int, default=5, help="Refresh live masks every N sim steps")
    parser.add_argument("--headless", action="store_true", help="Run simulation headless")
    parser.add_argument(
        "--discovery-failure-code",
        type=str,
        default=DISCOVERY_FAILURE_CODE_DEFAULT,
        help="Failure code for discovery-triggered replan",
    )
    args = parser.parse_args()

    config = ReplanConfig(
        max_replans=args.max_replans,
        use_mock_vlm=args.mock,
        use_remote_vlm=args.remote,
        remote_vlm_url=args.remote_url,
        use_vision=not args.text_only,
        visible_objects_only=True,
        live_segmentation_view=True,
        replan_on_discovery=True,
        discovery_targets=[s.strip() for s in args.discovery_targets.split(",") if s.strip()],
        discovery_failure_code=args.discovery_failure_code,
        replan_on_discovery_after_plan_complete=True,
        live_view_update_stride=max(1, args.live_mask_stride),
        headless=args.headless,
    )

    pipeline = VLMReplanningPipeline(config)
    try:
        if not pipeline.initialize():
            print("Failed to initialize pipeline")
            return

        result = pipeline.run(args.goal)
        print(result)

        print("\nPress Ctrl+C to exit...")
        while True:
            pipeline.env.pr.step()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
