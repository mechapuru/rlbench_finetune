"""
Record the live segmentation panel window (composited viewer frame) to video.
"""

import os

try:
    import cv2
except ImportError:
    cv2 = None


class LivePanelVideoRecorder:
    """Writes RGB panel frames to an MP4 video."""

    def __init__(self, output_path, fps=12):
        self.output_path = output_path
        self.fps = int(fps)
        self.writer = None
        self.frame_count = 0

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def record_frame(self, frame_rgb):
        """Record one RGB frame (numpy uint8 HxWx3)."""
        if cv2 is None or frame_rgb is None:
            return
        if len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
            return

        h, w = frame_rgb.shape[:2]
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)
        self.frame_count += 1

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.frame_count > 0:
            print(f"Live panel video saved: {self.output_path} ({self.frame_count} frames)")
