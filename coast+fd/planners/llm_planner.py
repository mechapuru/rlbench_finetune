"""LLM-based task planner (placeholder).

This module provides a template for LLM-based task planners.
Implement the plan() method to use GPT-4, Gemini, or other LLMs.
"""

from typing import List, Optional
from .base import TaskPlanner, PlannerNode, PlanResult


class LLMPlanner(TaskPlanner):
    """LLM-based task planner (placeholder implementation).
    
    This is a template for integrating LLM-based planners.
    Subclass this or implement the plan() method to use your preferred LLM.
    
    Example:
        class GPT4Planner(LLMPlanner):
            def __init__(self, api_key: str):
                super().__init__(model="gpt-4")
                self.client = OpenAI(api_key=api_key)
            
            def plan(self, domain_pddl, problem_state, pddl, task_horizon=50):
                # Call GPT-4 API with PDDL domain and problem
                ...
    """
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize LLM planner.
        
        Args:
            model: LLM model name (e.g., "gpt-4", "gemini-pro")
            api_key: API key for the LLM service
        """
        self.model = model
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return f"LLM ({self.model})"
    
    def plan(
        self,
        domain_pddl: str,
        problem_state: str,
        pddl,
        task_horizon: int = 50
    ) -> PlanResult:
        """Generate a task plan using an LLM.
        
        This is a placeholder implementation. Override this method
        to integrate with your preferred LLM.
        """
        return PlanResult(
            success=False,
            error_message=(
                f"LLM planner ({self.model}) not yet implemented. "
                "Please subclass LLMPlanner and implement the plan() method."
            )
        )
    
    def _build_prompt(self, domain_pddl: str, problem_state: str) -> str:
        """Build prompt for LLM.
        
        Override this to customize the prompt format.
        """
        return f"""You are a PDDL planning assistant. Given the following domain and problem,
generate a valid plan as a sequence of actions.

DOMAIN:
{domain_pddl}

PROBLEM:
{problem_state}

Generate a plan as a list of actions, one per line, in the format:
(action-name arg1 arg2 ...)

PLAN:
"""
    
    def _parse_llm_response(self, response: str, pddl) -> List[PlannerNode]:
        """Parse LLM response into PlannerNode list.
        
        Override this to handle different response formats.
        """
        plan = [PlannerNode("", pddl.initial_state)]
        state = pddl.initial_state
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse (action arg1 arg2) format
            if line.startswith('(') and line.endswith(')'):
                parts = line[1:-1].split()
                action = f"{parts[0]}({','.join(parts[1:])})"
                try:
                    state = pddl.next_state(state, action)
                    plan.append(PlannerNode(action, state))
                except Exception:
                    pass
        
        return plan
