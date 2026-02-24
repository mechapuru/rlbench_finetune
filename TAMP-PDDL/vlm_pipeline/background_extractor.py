import time
import threading
from typing import Optional

class BackgroundStateExtractor:
    """
    Runs the existing DynamicStateExtractor safely in a background thread 
    inside the PyRep/RLBench process, allowing continuous extraction 
    without blocking physics and without Qt OpenGL crashes.
    """
    def __init__(self, rlbench_env, task_name="Unknown Task"):
        self.env = rlbench_env
        self.task_name = task_name
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_state = None
        
        # We must import inside so PyRep handles are ready
        from dynamic_state_extractor import DynamicStateExtractor
        self.extractor = DynamicStateExtractor(self.env)
        
    def _extractor_loop(self):
        print(f"\n[Dynamic Extractor] Background monitoring started for {self.task_name}.")
        while self._running:
            try:
                # We acquire the PyRep lock naturally by reading handles
                # If extraction is heavy, we might want to throttle it
                bundle = self.extractor.create_prompt_bundle(f"Executing {self.task_name}")
                self._latest_state = bundle.state_text
                
                # Print live updates cleanly
                print(f"\r[Dynamic State] Extracted {bundle.state_text.count('-')} objects.    ", end="", flush=True)
                
            except Exception as e:
                # Suppress transient physics errors during rapid resets
                pass
                
            time.sleep(1.0) # Update once per second
            
    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._extractor_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        if not self._running: return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("\n[Dynamic Extractor] Stopped.")
        
    def get_latest_state(self):
        """Returns the most recently computed PDDL text state"""
        return self._latest_state
