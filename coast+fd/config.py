"""Configuration for COAST + RLBench integration."""

from dataclasses import dataclass, field
from typing import Optional
import os

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
    
    # Paths
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
        """Set default paths based on package location."""
        pkg_dir = os.path.dirname(__file__)
        if not self.domain_pddl:
            self.domain_pddl = os.path.join(pkg_dir, "domain.pddl")
        if not self.problem_pddl:
            self.problem_pddl = os.path.join(pkg_dir, "problem.pddl")
        if not self.streams_pddl:
            self.streams_pddl = os.path.join(pkg_dir, "streams.pddl")
