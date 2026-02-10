# TAMP-PDDL with VLM Replanning

This folder contains all the work done for the Long Horizon TAMP project with VLM-based replanning capabilities.

## Project Overview

A robotic manipulation system that combines:
- **PDDLStream** for symbolic task planning
- **Vision Language Models (VLMs)** for scene understanding and replanning
- **RLBench/CoppeliaSim** for simulation and execution

## Directory Structure

```
TAMP_VLM_WORK/
├── ground_truth/                    # Ground truth orchestrator (working baseline)
│   ├── ground_truth_orchestrator.py # Main orchestrator with manual replanning
│   ├── ground_truth_plan.txt        # The complete ground truth plan
│   └── rlbench_kitchen_env.py       # Environment wrapper
│
├── vlm_pipeline/                    # VLM-based planning pipeline
│   ├── vlm_context_aggregator.py    # Module 1: Capture 5 cameras, build state
│   ├── vlm_planner.py               # Module 2: VLM inference (Qwen2-VL)
│   ├── vlm_executor_v2.py           # Module 3: Execute actions with retries
│   ├── execution_monitor.py         # Module 4: Detect failures
│   ├── demo_replanning.py           # Interactive demo with 3 replan scenarios
│   ├── vlm_server.py                # Remote GPU server for VLM inference
│   ├── vlm_client.py                # Client to connect to remote VLM
│   └── prompts/                     # System and user prompts for VLM
│
├── experiments/                     # Ablation experiments
│   ├── exp1_pure_pddl.py           # Baseline: Pure PDDL (no rules)
│   ├── exp2_coast.py               # COAST: PDDL + rules
│   ├── exp3_llm_planner.py         # LLM-based planning
│   ├── exp4_vlm_planner.py         # VLM-based planning
│   └── results/                     # Experiment results
│
├── pddl/                            # PDDL domain and problem files
│   ├── rlbench_kitchen_domain.pddl
│   └── rlbench_kitchen_streams.pddl
│
└── docs/                            # Documentation
    ├── ARCHITECTURE.md              # System architecture
    └── REPLANNING_SCENARIOS.md      # The 3 replanning scenarios
```

## The Task

**Goal**: Place all 3 mugs (mug_box, mug_cupboard, mug_inside_box) into the placement_boundary region.

**Challenges**:
1. `mug_cupboard` is blocking groceries in the cupboard
2. `mug_box` is blocking the box lid
3. `mug_inside_box` is hidden inside the closed box

## Ground Truth Orchestrator

The ground truth orchestrator executes a manually designed plan with 3 replanning cycles:

### Cycle 1: Initial Plan
```
pick(sugar) → place(sugar, box-top)
pick(crackers) → place(crackers, box-top)
pick(soup) → FAILS! (mug_cupboard blocking)
```

### Replan 1: Clear the cupboard
```
pick(mug_cupboard) → place(mug_cupboard, table)  # ✓ Mug 1 done!
pick(soup) → place(soup, cupboard_boundary)
pick(mustard) → place(mustard, cupboard_boundary)
pick(spam) → place(spam, cupboard_boundary)
open-lid(box_lid) → FAILS! (mug_box blocking)
```

### Replan 2: Clear the box lid
```
pick(mug_box) → place(mug_box, table)  # ✓ Mug 2 done!
open-lid(box_lid) → SUCCESS
# Discovery: mug_inside_box found!
```

### Replan 3: Get the hidden mug
```
pick(mug_inside_box) → place(mug_inside_box, placement_boundary)  # ✓ Mug 3 done!
```

## VLM Pipeline

### Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Context         │────▶│ VLM Planner  │────▶│ Executor        │
│ Aggregator      │     │ (Qwen2-VL)   │     │ + Monitor       │
│ (5 cameras)     │     │              │     │                 │
└─────────────────┘     └──────────────┘     └─────────────────┘
        │                                            │
        │                                            ▼
        │                                    ┌─────────────────┐
        │◀───────────────────────────────────│ Failure?        │
        │           REPLAN                   │ REPLAN needed?  │
        │                                    └─────────────────┘
```

### Key Features

1. **Multi-camera context**: 5 cameras (left, right, overhead, wrist, front)
2. **State tracking**: PDDL-style state representation
3. **Failure detection**: Precondition checking, collision detection
4. **Automatic replanning**: VLM generates new plan on failure
5. **Retry mechanism**: Retries failed grasps/placements with fresh motion planning

### Running the Demo

```bash
# On your laptop (with CoppeliaSim)
python -m vlm_pipeline.demo_replanning --record

# With remote VLM server
export VLM_SERVER_URL="http://localhost:8000"
python -m vlm_pipeline.demo_replanning --record --remote
```

## Experiments

| Experiment | Description | Success Rate |
|------------|-------------|--------------|
| Pure PDDL  | No failure handling | 0% (fails on first block) |
| COAST      | PDDL + rules | ~60% (handles some blocks) |
| LLM        | Text-only LLM | ~40% (no visual context) |
| VLM        | Vision + Language | ~85% (best) |

## Key Files

| File | Purpose |
|------|---------|
| `ground_truth_orchestrator.py` | Working baseline with manual replanning |
| `vlm_pipeline/demo_replanning.py` | Interactive VLM demo |
| `vlm_pipeline/vlm_executor_v2.py` | Execution with retry mechanism |
| `rlbench_kitchen_env.py` | Environment wrapper |
| `rlbench_kitchen_streams.py` | PDDLStream integration |

## Dependencies

- Python 3.10+
- PyRep / CoppeliaSim
- RLBench
- PDDLStream
- Transformers (for VLM)
- Qwen2-VL (local or remote)

## Video Demos

- `ideal_replanning_demo.mp4` - Full demo of 3 replanning cycles
- `demo_videos/` - Individual scenario videos
