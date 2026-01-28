# COAST + RLBench Integration

Integrates COAST (COnstraints And STreams) TAMP algorithm with RLBench simulation, featuring a pluggable task planner interface.

## Quick Start

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlbench
source import_commands.txt

# Run planning for LongHorizonGrillTask
cd coast+fd
python run.py --task LongHorizonGrillTask --planner fast_downward

# With execution
python run.py --task LongHorizonGrillTask --execute
```

## Directory Structure

```
coast+fd/
├── long_horizon_grill_pddl_files/   # PDDL files for LongHorizonGrillTask
│   ├── domain.pddl
│   ├── problem.pddl
│   └── streams.pddl
├── planners/                         # Pluggable task planners
│   ├── base.py                       # TaskPlanner ABC
│   ├── fast_downward.py              # Fast Downward wrapper
│   └── llm_planner.py                # LLM planner template
├── actions.py                        # Geometric action implementations
├── config.py                         # Configuration (auto-selects PDDL folder)
├── run.py                            # Main CLI entry point
├── streams.py                        # Stream implementations
├── world.py                          # RLBench/PyRep interface
└── README.md
```

## Adding a New Task

1. Create a new PDDL folder: `my_task_pddl_files/`
2. Add `domain.pddl`, `problem.pddl`, `streams.pddl`
3. Register in `config.py`:
   ```python
   TASK_PDDL_FOLDERS = {
       "LongHorizonGrillTask": "long_horizon_grill_pddl_files",
       "MyNewTask": "my_task_pddl_files",  # Add this
   }
   ```
4. Run: `python run.py --task MyNewTask`

## How to Swap Planners

```python
from planners.base import TaskPlanner, PlanResult

class MyPlanner(TaskPlanner):
    @property
    def name(self): return "My Planner"
    
    def plan(self, domain_pddl, problem_state, pddl, task_horizon=50):
        return PlanResult(success=True, plan=plan_nodes)
```

## LongHorizonGrillTask

**Task Flow:**
1. `pick(chicken, grill_side)` → `place(chicken, grill_cooking_area)`
2. `close_lid` (constrained trajectory)
3. `pick(plate, dish_rack)` → `place(plate, plate_target)`
4. `open_lid` (constrained trajectory)
5. `pick(chicken, grill_cooking_area)` → `place(chicken, plate_target)`

**Locations:**
- `grill_side` - Where chicken starts
- `grill_cooking_area` - Where chicken cooks (under lid)
- `dish_rack` - Where plate starts
- `plate_target` - Where plate ends up

## CLI Options

```
python run.py --help

Options:
  --task, -t        RLBench task name (default: LongHorizonGrillTask)
  --planner, -p     Planner: fast_downward, llm (default: fast_downward)
  --timeout         Planning timeout in seconds (default: 1200)
  --max-level       COAST max level (default: 6)
  --execute         Execute plan after planning
  --headless        Run without GUI
```
