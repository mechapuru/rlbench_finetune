# Kitchen Task - PDDLStream Execution

Long-horizon manipulation task in a kitchen environment using PDDLStream for task and motion planning.

## Task Description
The robot must:
1. Open the cupboard door
2. Retrieve an object from inside the cupboard
3. Open a box (slide the lid)
4. Place the object inside the box

## Files

### Core
| File | Description |
|------|-------------|
| `rlbench_kitchen_env.py` | Environment class - robot control, object handling, motion planning |
| `rlbench_kitchen_streams.py` | PDDLStream integration - stream functions for sampling poses/trajectories |
| `planning_rules.py` | Logical constraints and rules for valid action sequences |

### Execution Scripts
| File | Description |
|------|-------------|
| `orchestrator.py` | **Main entry point** - Runs the full task in correct logical order |
| `script3_cupboard_retrieve_hover_pick_gui.py` | Cupboard open + retrieve object |
| `script_3_open_grasp_open_box_gui.py` | Box lid open (slide) |
| `generalize_PP_box_gui.py` | Generalized pick-and-place for putting objects in the box |

### PDDL
| File | Description |
|------|-------------|
| `pddl/rlbench_kitchen_domain.pddl` | PDDL domain - actions, predicates |
| `pddl/rlbench_kitchen_streams.pddl` | Stream definitions for PDDLStream |

### Scene Files
| File | Description |
|------|-------------|
| `task_design_proposal_variation_1.ttt` | CoppeliaSim scene (variation 1) |
| `task_design_proposal_variation_2.ttt` | CoppeliaSim scene (variation 2) |

### Data
| File | Description |
|------|-------------|
| `data_states/*.npy` | Saved robot joint configurations (home, hover positions) |

## Why Orchestrator?

Running PDDL directly doesn't guarantee a **logically correct** order. For example, PDDL might try to slide open the box lid before moving the object sitting on top of it aside. The `orchestrator.py` enforces the correct sequence:

1. Clear any objects blocking the box lid
2. Open the box
3. Open cupboard
4. Retrieve object from cupboard
5. Place object in the box

## Requirements

- CoppeliaSim (with `COPPELIASIM_ROOT` set)
- PyRep
- PDDLStream (see `pddlstream/` folder in repo root)
- Python 3.8+

## Usage

```bash
# Run the full orchestrated task
python orchestrator.py

# Or run individual scripts for testing
python script3_cupboard_retrieve_hover_pick_gui.py
python script_3_open_grasp_open_box_gui.py
```

## Notes

- The scene file path in `rlbench_kitchen_env.py` may need to be updated to match your local setup
- Set `COPPELIASIM_ROOT` environment variable before running
