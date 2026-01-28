"""Configuration for COAST + RLBench integration."""

from dataclasses import dataclass, field
from typing import Optional
import os

# Map task names to their PDDL folder names
TASK_PDDL_FOLDERS = {
    "LongHorizonGrillTask": "long_horizon_grill_pddl_files",
    # Add more tasks here:
    # "LongHorizonGroceryTask": "long_horizon_grocery_pddl_files",
}

@dataclass
class CoastConfig:
    """Configuration for COAST TAMP algorithm."""
    
    # Planner settings
    planner: str = "fast_downward"
    planner_timeout: int = 1200  # seconds
    task_horizon: int = 50
    
    # Algorithm settings
    algorithm: str = "improved"
    max_level: int = 6
    search_sample_ratio: float = 10.0
    
    # Paths (set dynamically based on task)
    domain_pddl: str = ""
    problem_pddl: str = ""
    streams_pddl: str = ""
    
    # Fast Downward path
    fd_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "..", "coast", "external", "downward", "fast-downward.py"
    ))
    
    # RLBench settings
    task_name: str = "LongHorizonGrillTask"
    headless: bool = False
    
    def __post_init__(self):
        """Set PDDL paths based on task name."""
        pkg_dir = os.path.dirname(__file__)
        
        # Get PDDL folder for this task
        pddl_folder = TASK_PDDL_FOLDERS.get(self.task_name, "long_horizon_grill_pddl_files")
        pddl_dir = os.path.join(pkg_dir, pddl_folder)
        
        if not self.domain_pddl:
            self.domain_pddl = os.path.join(pddl_dir, "domain.pddl")
        if not self.problem_pddl:
            self.problem_pddl = os.path.join(pddl_dir, "problem.pddl")
        if not self.streams_pddl:
            self.streams_pddl = os.path.join(pddl_dir, "streams.pddl")

