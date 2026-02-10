import os
try:
    import cv2
except ImportError:
    cv2 = None


class VideoRecorder:
    def __init__(self, env, output_dir="videos", fps=20, cameras=None):
        """
        Initialize video recorder.
        
        Args:
            env: The environment with get_camera_frames() method
            output_dir: Directory to save videos (created if not exists)
            fps: Frames per second for output videos
            cameras: List of camera names to record (None = all cameras)
        """
        self.env = env
        self.fps = fps
        self.output_dir = output_dir
        self.cameras = cameras  # None means all cameras
        self.writers = {}
        self.initialized = False
        self.frame_count = 0
        
        # Create output directory
        if cv2 is not None:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Video output directory: {os.path.abspath(output_dir)}")

    def _init_writers(self, frames):
        for name, frame in frames.items():
            # Skip cameras not in the filter list
            if self.cameras is not None and name not in self.cameras:
                continue
            height, width, _ = frame.shape
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out_file = os.path.join(self.output_dir, f'video_{name}.mp4')
            print(f"Creating video writer for {name} ({width}x{height}) -> {out_file}")
            self.writers[name] = cv2.VideoWriter(out_file, codec, self.fps, (width, height))
        self.initialized = True

    def record_step(self):
        if cv2 is None:
            return
        frames = self.env.get_camera_frames()
        if not self.initialized:
            self._init_writers(frames)
        for name, frame in frames.items():
            if name not in self.writers:
                continue
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writers[name].write(bgr)
        self.frame_count += 1

    def release(self):
        if cv2 is None:
            return
        print(f"Saving {self.frame_count} frames to videos...")
        for name, writer in self.writers.items():
            writer.release()
            print(f"  Saved: {os.path.join(self.output_dir, f'video_{name}.mp4')}")
        print("Videos saved successfully.")
