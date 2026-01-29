# Grill Task - PDDLStream Execution

Long-horizon manipulation task where the robot must cook meat on a grill using PDDLStream for task and motion planning.

## Task Description
The robot must:
1. Pick up meat (steak or chicken) from the table
2. Open the grill lid (hinged rotation)
3. Place the meat on the grill surface
4. Close the grill lid

## Files

### Core
| File | Description |
|------|-------------|
| `grill_task_env.py` | Environment class - robot control, object handling, motion planning, grill lid kinematics |
| `grill_task_streams.py` | PDDLStream integration - stream functions for sampling poses/trajectories |

### Execution Scripts
| File | Description |
|------|-------------|
| `grill_place_gui.py` | **Pick & place** - picks an object and places it on grill-top or plate |
| `grill_open_hover_grasp_close_gui.py` | **Grill lid operation** - open/close the hinged grill lid |

### PDDL
| File | Description |
|------|-------------|
| `pddl/grill_task_domain.pddl` | PDDL domain - actions: move, pick, place, open-grill, close-grill |
| `pddl/grill_task_streams.pddl` | Stream definitions for PDDLStream |

### Scene File
| File | Description |
|------|-------------|
| `grill_task.ttt` | CoppeliaSim scene with grill, meats, plate, dish rack |

### Data
| File | Description |
|------|-------------|
| `data_states/home.npy` | Saved robot home joint configuration |

## Scene Objects

- **Meats**: `steak`, `chicken` (aliases: `meat1`, `meat2`)
- **Grill**: hinged lid with `lid_joint`, `grill_boundary` (placement surface)
- **Plate**: `plate_visual`, `plate_boundary`
- **Robot**: Franka Panda with gripper

## PDDL Actions

| Action | Description |
|--------|-------------|
| `move` | Move robot between configurations |
| `pick` | Pick up a movable object |
| `place` | Place object on a region (grill-top, plate-top) |
| `open-grill` | Open the hinged grill lid |
| `close-grill` | Close the hinged grill lid |

## Requirements

- CoppeliaSim (with `COPPELIASIM_ROOT` set)
- PyRep
- PDDLStream (see `pddlstream/` folder)
- Python 3.8+

## Usage

```bash
# Pick steak and place on grill
python grill_place_gui.py steak grill-top

# Pick chicken and place on plate
python grill_place_gui.py chicken plate-top

# Open/close grill lid
python grill_open_hover_grasp_close_gui.py
```

### Arguments for grill_place_gui.py
- **object**: `steak`, `chicken`, `meat1`, `meat2`, `plate`
- **region**: `grill-top`, `plate-top`, `plate_boundary`

## Notes

- The grill lid uses **hinged rotation** (circular arc motion), not linear motion
- The scene file path in `grill_task_env.py` may need to be updated to match your local setup
- Set `COPPELIASIM_ROOT` environment variable before running
