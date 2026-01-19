# COAST + RLBench Integration

Integrates COAST (COnstraints And STreams) TAMP algorithm with RLBench simulation, featuring a pluggable task planner interface.

## Quick Start

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlbench
source import_commands.txt

# Run planning
cd coast+fd
python run.py --task LongHorizonGrillTask --planner fast_downward

# With execution
python run.py --task LongHorizonGrillTask --execute
```

## Directory Structure

```
coast+fd/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclass
├── world.py             # RLBench/PyRep world interface
├── streams.py           # 5 stream classes for motion planning
├── actions.py           # 6 action classes (Pick, Place, etc.)
├── run.py               # Main CLI entry point
├── domain.pddl          # PDDL domain for LongHorizonGrillTask
├── problem.pddl         # PDDL problem file
├── streams.pddl         # Stream declarations
├── planners/
│   ├── base.py          # TaskPlanner ABC (pluggable interface)
│   ├── fast_downward.py # Fast Downward wrapper
│   └── llm_planner.py   # LLM planner template
└── README.md
```

## How to Swap Planners

The system uses a pluggable `TaskPlanner` interface:

```python
# planners/base.py
class TaskPlanner(ABC):
    @abstractmethod
    def plan(self, domain_pddl, problem_state, pddl, task_horizon=50):
        pass
```

**To add a new planner:**

1. Create `planners/my_planner.py`:
```python
from .base import TaskPlanner, PlanResult

class MyPlanner(TaskPlanner):
    @property
    def name(self): return "My Planner"
    
    def plan(self, domain_pddl, problem_state, pddl, task_horizon=50):
        # Your planning logic here
        return PlanResult(success=True, plan=plan_nodes)
```

2. Register in `planners/__init__.py`
3. Use via CLI: `python run.py --planner my_planner`

## Components

### Streams (`streams.py`)
| Stream | Description |
|--------|-------------|
| `SampleGrasp` | Sample grasp poses for objects |
| `SamplePose` | Sample placement poses on surfaces |
| `SampleIK` | Solve inverse kinematics |
| `SampleMotion` | Plan collision-free motion |
| `CheckCollision` | Validate trajectories |

### Actions (`actions.py`)
| Action | Description |
|--------|-------------|
| `Pick` | Grasp object from location |
| `Place` | Place held object |
| `PlaceOnGrill` | Place object on grill surface |
| `OpenGrill` | Open grill lid |
| `CloseGrill` | Close grill lid |
| `Cook` | Simulate cooking |

### World Interface (`world.py`)
Provides RLBench/PyRep integration:
- `get_robot_config()` / `set_robot_config()` - Joint configuration
- `plan_to_pose()` - Motion planning via PyRep
- `solve_ik()` - Inverse kinematics
- `grasp()` / `release()` - Gripper control
- `check_collision()` - Collision checking

## Prerequisites

1. **RLBench environment**: `conda activate rlbench`
2. **CoppeliaSim**: Set `COPPELIASIM_ROOT`
3. **COAST**: `pip install -e ../coast`
4. **Fast Downward**: See `coast/README.md` for installation

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

## Citation

Based on the COAST paper:
```bibtex
@article{vu2024coast,
  title={COAST: Constraints And Streams for Task and Motion Planning},
  author={Vu, Brandon and Migimatsu, Toki and Bohg, Jeannette},
  booktitle={2024 IEEE ICRA},
  year={2024}
}
```
