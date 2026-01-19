"""Abstract base class for task planners.

This module defines the interface that all task planners must implement,
enabling easy swapping between different planning backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, NamedTuple, Set
from dataclasses import dataclass


class PlannerNode(NamedTuple):
    """Represents a step in a task plan."""
    action: str
    state: Set[str]


@dataclass
class PlanResult:
    """Result of a planning operation."""
    success: bool
    plan: Optional[List[PlannerNode]] = None
    error_message: Optional[str] = None
    planning_time: float = 0.0


class TaskPlanner(ABC):
    """Abstract interface for task planners.
    
    All task planners (Fast Downward, LLM-based, custom) must implement
    this interface to be used with COAST.
    
    Example usage:
        planner = FastDownwardPlanner()
        result = planner.plan(domain_pddl, problem_str, pddl)
        if result.success:
            for step in result.plan:
                print(f"Action: {step.action}")
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this planner."""
        pass
    
    @abstractmethod
    def plan(
        self,
        domain_pddl: str,
        problem_state: str,
        pddl,  # symbolic.Pddl
        task_horizon: int = 50
    ) -> PlanResult:
        """Generate a task plan from PDDL domain and problem.
        
        Args:
            domain_pddl: Path to domain PDDL file
            problem_state: PDDL problem state as string
            pddl: Parsed PDDL object (symbolic.Pddl)
            task_horizon: Maximum plan length
            
        Returns:
            PlanResult with success status and plan (if successful)
        """
        pass
    
    def validate_domain(self, domain_pddl: str) -> bool:
        """Validate a PDDL domain file.
        
        Args:
            domain_pddl: Path to domain PDDL file
            
        Returns:
            True if domain is valid, False otherwise
        """
        try:
            with open(domain_pddl, 'r') as f:
                content = f.read()
            return '(define' in content and ':action' in content
        except Exception:
            return False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
