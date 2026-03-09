import argparse
import os
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ORDER = [
    "video_left.mp4",
    "video_right.mp4",
    "video_overhead.mp4",
    "video_wrist.mp4",
    "video_front.mp4",
]

VIEW_LABELS = [
    "LEFT",
    "RIGHT", 
    "OVERHEAD",
    "WRIST",
    "FRONT",
]

BORDER_COLOR = (50, 50, 50)  # Dark gray
BORDER_WIDTH = 4
LABEL_BG_COLOR = (30, 30, 30)  # Darker background for labels
LABEL_TEXT_COLOR = (255, 255, 255)  # White text


def _add_label(frame, label, position="top-left"):
    """Add a label with background to the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Add padding
    padding = 8
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2 + baseline
    
    # Draw background rectangle
    cv2.rectangle(frame, (0, 0), (box_w, box_h), LABEL_BG_COLOR, -1)
    
    # Draw border around label
    cv2.rectangle(frame, (0, 0), (box_w, box_h), BORDER_COLOR, 2)
    
    # Draw text
    cv2.putText(frame, label, (padding, text_h + padding), font, font_scale, LABEL_TEXT_COLOR, thickness)
    
    return frame


def _add_border(frame, color=BORDER_COLOR, width=BORDER_WIDTH):
    """Add border around a frame."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, width)
    return frame


def _open_captures(input_dir, order):
    caps = []
    for name in order:
        path = os.path.join(input_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open: {path}")
        caps.append(cap)
    return caps


def _get_min_frame_count(caps):
    counts = []
    for cap in caps:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            counts.append(count)
    return min(counts) if counts else 0


def _get_source_fps(caps, default_fps=30):
    for cap in caps:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return float(fps)
    return float(default_fps)


def _tile_2x3(frames, target_size, speed_factor=1.0, original_duration_secs=0):
    w, h = target_size
    resized = []
    for i, f in enumerate(frames):
        frame = cv2.resize(f, (w, h))
        # Add border
        _add_border(frame)
        # Add label
        _add_label(frame, VIEW_LABELS[i])
        resized.append(frame)
    
    # Create info panel for 6th slot
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    blank[:] = (20, 20, 20)  # Dark background
    _add_border(blank)
    
    # Add info text to panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    title = "GROUND TRUTH"
    subtitle = "ORCHESTRATOR"
    (tw, _), _ = cv2.getTextSize(title, font, 0.9, 2)
    (sw, _), _ = cv2.getTextSize(subtitle, font, 0.7, 2)
    
    y_offset = h // 4
    cv2.putText(blank, title, ((w - tw) // 2, y_offset), font, 0.9, (100, 220, 100), 2)
    cv2.putText(blank, subtitle, ((w - sw) // 2, y_offset + 40), font, 0.7, (100, 220, 100), 2)
    
    # Divider line
    cv2.line(blank, (40, y_offset + 70), (w - 40, y_offset + 70), (80, 80, 80), 2)
    
    # Stats
    stats_y = y_offset + 110
    
    # Original duration
    orig_mins = int(original_duration_secs // 60)
    orig_secs = int(original_duration_secs % 60)
    orig_text = f"Original: {orig_mins}m {orig_secs}s"
    cv2.putText(blank, orig_text, (50, stats_y), font, 0.6, (200, 200, 200), 2)
    
    # Speed factor
    speed_text = f"Speed: {speed_factor:.0f}x"
    cv2.putText(blank, speed_text, (50, stats_y + 40), font, 0.7, (255, 200, 100), 2)
    
    # Task info
    cv2.putText(blank, "Tasks: 9/9 Completed", (50, stats_y + 90), font, 0.55, (150, 150, 150), 1)
    cv2.putText(blank, "Long-Horizon Kitchen", (50, stats_y + 120), font, 0.5, (120, 120, 120), 1)
    
    # Order: left, right, overhead / wrist, front, blank
    top = np.hstack([resized[0], resized[1], resized[2]])
    bottom = np.hstack([resized[3], resized[4], blank])
    
    # Add horizontal divider line
    combined = np.vstack([top, bottom])
    cv2.line(combined, (0, h), (w * 3, h), BORDER_COLOR, BORDER_WIDTH)
    
    return combined


def main():
    parser = argparse.ArgumentParser(description="Combine 5 camera videos into a tiled view and speed up.")
    parser.add_argument(
        "--input-dir",
        default="orchestrator_videos",
        help="Directory containing camera mp4 files",
    )
    parser.add_argument(
        "--output",
        default="orchestrator_videos/combined_5view_fast.mp4",
        help="Output mp4 file path",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=60.0,
        help="Target duration in seconds for the output video",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=30.0,
        help="Output FPS",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of each view in the tiled output",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of each view in the tiled output",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_path = os.path.abspath(args.output)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    caps = _open_captures(input_dir, DEFAULT_ORDER)
    try:
        min_frames = _get_min_frame_count(caps)
        source_fps = _get_source_fps(caps)

        desired_frames = max(1, int(round(args.target_duration * args.output_fps)))
        stride = max(1, int(round(min_frames / desired_frames))) if min_frames > 0 else 1

        # Calculate original duration and speed factor
        original_duration_secs = min_frames / source_fps if source_fps > 0 else 0
        speed_factor = stride  # Each stride skips frames, so speed = stride

        print("Input dir:", input_dir)
        print("Output:", output_path)
        print("Source FPS:", source_fps)
        print("Min frames:", min_frames)
        print(f"Original duration: {original_duration_secs:.1f}s ({original_duration_secs/60:.1f} min)")
        print("Target duration (s):", args.target_duration)
        print("Output FPS:", args.output_fps)
        print(f"Speed factor: {speed_factor}x")
        print("Stride:", stride)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = args.width * 3
        out_h = args.height * 2
        writer = cv2.VideoWriter(output_path, fourcc, args.output_fps, (out_w, out_h))

        frame_index = 0
        written = 0
        while True:
            frames = []
            for cap in caps:
                ok, frame = cap.read()
                if not ok:
                    frames = None
                    break
                frames.append(frame)
            if frames is None:
                break

            if frame_index % stride == 0:
                tiled = _tile_2x3(frames, (args.width, args.height), speed_factor, original_duration_secs)
                writer.write(tiled)
                written += 1

            # Skip next frames in stride
            for _ in range(stride - 1):
                for cap in caps:
                    if not cap.grab():
                        frames = None
                        break
                frame_index += 1
                if frames is None:
                    break

            frame_index += 1

        writer.release()
        print(f"Wrote {written} frames -> {output_path}")

    finally:
        for cap in caps:
            cap.release()


if __name__ == "__main__":
    main()
