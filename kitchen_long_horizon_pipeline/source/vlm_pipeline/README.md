# VLM Pipeline for Kitchen Task Planning

A single Vision-Language Model (Qwen2-VL-7B) based planner that takes 5 camera views and scene state to generate PDDL-compatible action skeletons.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VLM PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Module 1        │    │  Module 2        │    │  Module 2.5   │  │
│  │  Context         │───▶│  VLM Planner     │───▶│  Skeleton     │  │
│  │  Aggregator      │    │  (Qwen2-VL-7B)   │    │  to PDDL      │  │
│  └──────────────────┘    └──────────────────┘    └───────────────┘  │
│         │                        │                       │          │
│         │                        │                       │          │
│    5 cameras +              Action Skeleton         Goal atoms      │
│    State text                                       + Init          │
│                                                          │          │
│                              ┌───────────────────────────┘          │
│                              ▼                                      │
│                      ┌──────────────────┐                           │
│                      │  Module 3        │                           │
│                      │  Executor        │◀── Interrupt (manual)     │
│                      │  (PDDLStream)    │                           │
│                      └──────────────────┘                           │
│                              │                                      │
│                              ▼                                      │
│                      Robot Actions in RLBench                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Modules

### 1. Context Aggregator (`vlm_context_aggregator.py`)
- Captures 5 camera views (left, right, overhead, wrist, front)
- Extracts scene state (objects, regions, lid status)
- Builds PDDL-style state description
- Creates prompt bundles for VLM

### 2. VLM Planner (`vlm_planner.py`)
- Loads Qwen2-VL-7B-Instruct (4-bit quantized)
- Takes image + state + goal → outputs action skeleton
- Parses VLM output into ActionSkeleton objects
- Validates plans against constraints

### 2.5. Skeleton to PDDL (`skeleton_to_pddl.py`)
- Translates action skeletons to PDDLStream format
- Builds init atoms and goal from skeleton
- Creates staged execution plan

### 3. Executor (`vlm_executor.py`)
- Executes actions through environment
- Monitors for failures (IK, collision)
- Supports pause/interrupt for replanning

## Usage

### Quick Test (Mock VLM, no GPU)
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL
python -m vlm_pipeline.vlm_main --mock --no-env --goal full_task
```

### Full Pipeline with Real VLM
```bash
python -m vlm_pipeline.vlm_main --goal full_task
```

### With RLBench Environment
```bash
python -m vlm_pipeline.vlm_main --goal full_task
```

### Run Benchmark Experiment
```bash
# Planning only (no execution)
python experiments/exp4_vlm_planner.py --mock --trials 5

# With real VLM
python experiments/exp4_vlm_planner.py --trials 5
```

## Available Goals

- `full_task`: Complete kitchen task (mug_box → placement, open lid, mug_inside_box → placement, soup → cupboard)
- `simple_pick_place`: Just move mug_box to placement
- `open_and_retrieve`: Open lid and retrieve mug_inside_box
- `soup_to_cupboard`: Just move soup to cupboard

## Manual Interruption

The pipeline supports manual interruption for unknown object handling:

```python
from vlm_pipeline import VLMPipeline, PipelineConfig

pipeline = VLMPipeline(PipelineConfig(use_mock_vlm=True))
pipeline.initialize()

# Start execution in a thread or after some actions...

# When unknown object detected:
pipeline.interrupt("Unknown object blocking path")
pipeline.add_unknown_object("red_obstacle")

# Pipeline will trigger replanning automatically
```

## Files

```
vlm_pipeline/
├── __init__.py              # Package exports
├── vlm_context_aggregator.py # Camera capture + state building
├── vlm_planner.py           # VLM interface (Qwen2-VL)
├── skeleton_to_pddl.py      # Skeleton → PDDL translation
├── vlm_executor.py          # Action execution
├── vlm_main.py              # Main pipeline loop
├── prompts/
│   ├── system_prompt.txt    # VLM system prompt
│   ├── plan_prompt.txt      # Planning prompt template
│   └── replan_prompt.txt    # Replanning prompt template
└── README.md                # This file
```

## Dependencies

```bash
# Core
pip install numpy pillow

# VLM (for real model)
pip install torch transformers accelerate bitsandbytes
pip install qwen-vl-utils

# For environment
pip install pyrep
```

## Expected Output

For `full_task` goal, expected action skeleton:
```
1. pick(mug_box)
2. place(mug_box, placement_boundary)
3. open-lid(box_lid)
4. pick(mug_inside_box)
5. place(mug_inside_box, placement_boundary)
6. pick(soup)
7. place(soup, cupboard_boundary)
```

## Notes

- The VLM uses skeleton approach: outputs high-level actions that PDDLStream grounds
- Replanning is a placeholder - trigger manually with `pipeline.interrupt()`
- Mock planner available for testing without GPU
