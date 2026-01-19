"""Fast Downward task planner implementation.

Wraps the Fast Downward PDDL planner for use with COAST.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Set

from .base import TaskPlanner, PlannerNode, PlanResult


class FastDownwardPlanner(TaskPlanner):
    """Fast Downward PDDL planner wrapper.
    
    Uses Fast Downward with FF heuristic for task planning.
    """
    
    def __init__(
        self,
        fd_path: Optional[str] = None,
        temp_dir: str = "/tmp/coast_fd",
        search_config: str = "ff"
    ):
        """Initialize Fast Downward planner.
        
        Args:
            fd_path: Path to fast-downward.py script
            temp_dir: Directory for temporary files
            search_config: Search configuration ("ff", "lama", "lama_first")
        """
        self._fd_path = fd_path or self._find_fd_path()
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._search_config = search_config
    
    @property
    def name(self) -> str:
        return "Fast Downward"
    
    def _find_fd_path(self) -> str:
        """Find Fast Downward installation."""
        # Check common locations
        candidates = [
            Path(__file__).parent.parent.parent / "coast" / "external" / "downward" / "fast-downward.py",
            Path.home() / "downward" / "fast-downward.py",
            Path("/usr/local/bin/fast-downward.py"),
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        raise FileNotFoundError(
            "Fast Downward not found. Please install it or provide fd_path."
        )
    
    def _get_search_config(self, task_horizon: int) -> List[str]:
        """Get search configuration arguments."""
        configs = {
            "ff": [
                '--evaluator',
                "h=ff(transform=adapt_costs(cost_type=PLUSONE))",
                '--search',
                f"eager(alt([single(sum([g(), weight(h,2)])), "
                f"single(sum([g(),weight(h,2)]),pref_only=true)]), "
                f"preferred=[h],cost_type=PLUSONE,bound={task_horizon})"
            ],
            "lama_first": [
                "--evaluator",
                "hlm=lmcount(lm_factory=lm_reasonable_orders_hps(lm_rhw()),"
                "transform=adapt_costs(one),pref=false)",
                "--evaluator",
                "hff=ff(transform=adapt_costs(one))",
                "--search",
                f"lazy_greedy([hff,hlm],preferred=[hff,hlm],"
                f"cost_type=one,reopen_closed=false,bound={task_horizon})",
            ],
        }
        return configs.get(self._search_config, configs["ff"])
    
    def plan(
        self,
        domain_pddl: str,
        problem_state: str,
        pddl,  # symbolic.Pddl
        task_horizon: int = 50
    ) -> PlanResult:
        """Generate a task plan using Fast Downward.
        
        Args:
            domain_pddl: Path to domain PDDL file
            problem_state: PDDL problem state as string
            pddl: Parsed PDDL object
            task_horizon: Maximum plan length
            
        Returns:
            PlanResult with success status and plan
        """
        start_time = time.time()
        
        # Write problem to temp file
        problem_path = self._temp_dir / "problem.pddl"
        plan_path = self._temp_dir / "plan.txt"
        
        with open(problem_path, 'w') as f:
            f.write(problem_state)
        
        # Remove old plan file if exists
        if plan_path.exists():
            plan_path.unlink()
        
        # Build command
        cmd = [
            "python3", self._fd_path,
            "--plan-file", str(plan_path),
            "--log-level", "warning",
            domain_pddl,
            str(problem_path),
        ]
        cmd.extend(self._get_search_config(task_horizon))
        
        # Run Fast Downward
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=300,  # 5 minute timeout
                check=False
            )
        except subprocess.TimeoutExpired:
            return PlanResult(
                success=False,
                error_message="Planning timed out",
                planning_time=time.time() - start_time
            )
        except Exception as e:
            return PlanResult(
                success=False,
                error_message=str(e),
                planning_time=time.time() - start_time
            )
        
        # Check if plan was found
        if not plan_path.exists():
            return PlanResult(
                success=False,
                error_message="No plan found",
                planning_time=time.time() - start_time
            )
        
        # Parse plan
        plan = self._parse_plan(plan_path, pddl)
        
        return PlanResult(
            success=True,
            plan=plan,
            planning_time=time.time() - start_time
        )
    
    def _parse_plan(self, plan_path: Path, pddl) -> List[PlannerNode]:
        """Parse Fast Downward plan output."""
        with open(plan_path, 'r') as f:
            actions = f.readlines()
        
        state = pddl.initial_state
        plan = [PlannerNode("", state)]
        
        for line in actions:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            # Parse action: (action-name arg1 arg2 ...) -> action-name(arg1,arg2,...)
            parts = line[1:-1].split()  # Remove parentheses
            action = f"{parts[0]}({','.join(parts[1:])})"
            
            try:
                state = pddl.next_state(state, action)
            except Exception:
                # If state transition fails, use previous state
                pass
            
            plan.append(PlannerNode(action, state))
        
        return plan
