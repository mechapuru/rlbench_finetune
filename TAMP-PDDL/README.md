# TAMP-PDDL: Task and Motion Planning with VLM Replanning

A robotic manipulation system combining **PDDLStream** for symbolic task planning with **Vision Language Models (VLMs)** for scene understanding and dynamic replanning.

## 🎯 Project Overview

This project implements a long-horizon manipulation task in a kitchen environment where a robot must:
- **Goal**: Place 3 mugs into a target boundary region
- **Challenge**: Objects block each other, requiring intelligent replanning
- **Solution**: VLM-guided replanning that detects failures and generates new plans

### The Task Flow

```
Initial State                     Goal State
┌─────────────────────┐          ┌─────────────────────┐
│  Cupboard           │          │  Cupboard           │
│  ├─ mug_cupboard ←──┼── blocks │  └─ (groceries)     │
│  └─ groceries       │          │                     │
│                     │          │  Placement Boundary │
│  Box (closed)       │          │  ├─ mug_cupboard ✓  │
│  ├─ mug_box ←───────┼── blocks │  ├─ mug_box ✓       │
│  └─ mug_inside_box  │   lid    │  └─ mug_inside_box ✓│
└─────────────────────┘          └─────────────────────┘
```

## 📁 Directory Structure

```
TAMP-PDDL/
│
├── 🎮 CORE EXECUTION
│   ├── ground_truth_orchestrator.py   # Main orchestrator with 3 replanning cycles
│   ├── ground_truth_plan.txt          # Complete ground truth action plan
│   ├── rlbench_kitchen_env.py         # RLBench environment wrapper (69KB)
│   ├── rlbench_kitchen_streams.py     # PDDLStream integration
│   └── video_recorder.py              # Multi-camera video recording
│
├── 🤖 VLM PIPELINE (vlm_pipeline/)
│   ├── vlm_context_aggregator.py      # Module 1: Capture 5 cameras, build state
│   ├── vlm_planner.py                 # Module 2: Qwen2-VL inference
│   ├── vlm_executor_v2.py             # Module 3: Execute with retry mechanism
│   ├── execution_monitor.py           # Module 4: Failure detection
│   ├── demo_replanning.py             # Interactive demo with video recording
│   ├── vlm_server.py                  # Remote GPU server for VLM
│   ├── vlm_client.py                  # Client for remote inference
│   ├── prompts/                       # System/user prompts for VLM
│   │   ├── system_prompt.txt
│   │   ├── plan_prompt.txt
│   │   └── replan_prompt.txt
│   └── configs/                       # Scene configuration
│
├── 🧪 EXPERIMENTS (experiments/)
│   ├── exp1_pure_pddl.py              # Baseline: Pure PDDL (no rules)
│   ├── exp2_coast.py                  # COAST: PDDL + constraint rules
│   ├── exp3_llm_planner.py            # LLM-based planning (text only)
│   ├── exp4_vlm_planner.py            # VLM-based planning (vision + text)
│   ├── compare_results.py             # Result comparison script
│   └── results/                       # Experiment results and logs
│
├── 📋 PDDL DOMAIN (pddl/)
│   ├── rlbench_kitchen_domain.pddl    # Action definitions
│   └── rlbench_kitchen_streams.pddl   # Stream definitions
│
├── 📊 LOGS & OUTPUTS
│   ├── vlm_pipeline_logs/             # VLM inference logs
│   ├── vlm_replan_logs/               # Replanning session logs
│   ├── demo_logs/                     # Demo run logs
│   ├── demo_videos/                   # Recorded demo videos (5 cameras)
│   ├── orchestrator_videos/           # Ground truth execution videos
│   ├── VLM_outputs/                   # Raw VLM outputs
│   └── ablation_results/              # Ablation study results
│
├── 📚 DOCUMENTATION
│   ├── TAMP_VLM_WORK/                 # Organized work summary
│   │   ├── README.md
│   │   ├── ground_truth/              # Ground truth files
│   │   ├── vlm_pipeline/              # VLM pipeline files
│   │   ├── experiments/               # Experiment files
│   │   └── docs/                      # Additional documentation
│   ├── COAST-PDDL/                    # COAST analysis documents
│   └── Failure_checks_list.txt        # Failure detection checklist
│
├── 🔧 UTILITIES
│   ├── combine_camera_videos.py       # Combine multi-camera views
│   ├── list_objects.py                # List scene objects
│   └── domain_patcher.py              # PDDL domain modification
│
├── 🎮 GUI SCRIPTS (curriculum learning)
│   ├── generalize_pick_place_gui.py   # Generalized pick-place
│   ├── script4_cupboard_pick_place_gui.py
│   └── curriculum_scripts/            # Step-by-step learning scripts
│
├── 🔥 GRILL TASK (grill_task2/)
│   ├── grill_task_env.py              # Grill task environment
│   ├── grill_task_streams.py          # Grill task streams
│   └── pddl/                          # Grill task PDDL files
│
└── 📦 DEPENDENCIES
    └── pddlstream/                    # PDDLStream library
```

## 🔄 The 3 Replanning Cycles

### Cycle 1: Initial Plan (Fails)
```
pick(sugar) → place(sugar, box-top) ✓
pick(crackers) → place(crackers, box-top) ✓
pick(soup) → ❌ FAILS! (mug_cupboard blocking access)
```

### Replan 1: Clear the Cupboard
```
pick(mug_cupboard) → place(mug_cupboard, placement_boundary) ✓  # Mug 1!
pick(soup) → place(soup, cupboard_boundary) ✓
pick(mustard) → place(mustard, cupboard_boundary) ✓
pick(spam) → place(spam, cupboard_boundary) ✓
open-lid(box_lid) → ❌ FAILS! (mug_box blocking lid)
```

### Replan 2: Clear the Box Lid
```
pick(mug_box) → place(mug_box, placement_boundary) ✓  # Mug 2!
open-lid(box_lid) → ✓
🔍 DISCOVERY: mug_inside_box found inside!
```

### Replan 3: Get Hidden Mug
```
pick(mug_inside_box) → place(mug_inside_box, placement_boundary) ✓  # Mug 3!
🎉 TASK COMPLETE!
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VLM PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Context    │───▶│  VLM Planner │───▶│    Executor      │  │
│  │  Aggregator  │    │  (Qwen2-VL)  │    │   + Monitor      │  │
│  │  (5 cameras) │    │              │    │                  │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│         ▲                                          │            │
│         │            REPLAN LOOP                   ▼            │
│         │         ┌──────────────────────────────────┐         │
│         └─────────│  Failure Detected? New Object?   │         │
│                   └──────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PDDLStream + RLBench                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   PDDL      │───▶│   Motion    │───▶│    CoppeliaSim      │  │
│  │  Planning   │    │  Planning   │    │    Execution        │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Run Ground Truth Demo
```bash
cd TAMP-PDDL
python ground_truth_orchestrator.py
```

### Run VLM Pipeline Demo
```bash
# Local VLM (requires GPU)
python -m vlm_pipeline.demo_replanning --record

# With remote VLM server
export VLM_SERVER_URL="http://localhost:8000"
python -m vlm_pipeline.demo_replanning --record --remote
```

### Run Experiments
```bash
cd experiments
python exp1_pure_pddl.py    # Pure PDDL baseline
python exp2_coast.py        # COAST with rules
python exp3_llm_planner.py  # LLM planner
python exp4_vlm_planner.py  # VLM planner
python compare_results.py   # Compare all
```

## 📊 Experiment Results

| Method | Success Rate | Handles Blocking | Discovers Hidden |
|--------|--------------|------------------|------------------|
| Pure PDDL | 0% | ❌ | ❌ |
| COAST (PDDL + rules) | ~60% | ✓ | ❌ |
| LLM Planner | ~40% | ✓ | ❌ |
| **VLM Planner** | **~85%** | ✓ | ✓ |

## 🔧 Key Components

### VLM Planner (`vlm_pipeline/vlm_planner.py`)
- Uses Qwen2-VL-7B-Instruct (or 2B for faster inference)
- Multi-camera input (5 views: left, right, overhead, wrist, front)
- PDDL-style action output parsing

### Executor with Retry (`vlm_pipeline/vlm_executor_v2.py`)
- Executes actions via PDDLStream
- Retry mechanism for GRASP_FAILED / PLACEMENT_FAILED
- Queries current object pose before each retry

### Execution Monitor (`vlm_pipeline/execution_monitor.py`)
- Detects precondition failures
- Collision detection
- New object discovery

## 📝 PDDL Actions

```pddl
(:action pick
  :parameters (?obj - object)
  :precondition (and (on-table ?obj) (not-blocked ?obj))
  :effect (holding ?obj))

(:action place
  :parameters (?obj - object ?region - region)
  :precondition (holding ?obj)
  :effect (and (at ?obj ?region) (not (holding ?obj))))

(:action open-lid
  :parameters (?lid - lid)
  :precondition (and (lid-closed ?lid) (not-blocked ?lid))
  :effect (lid-open ?lid))
```

## 📦 Dependencies

- Python 3.10+
- PyRep / CoppeliaSim 4.6+
- RLBench
- PDDLStream
- Transformers (for VLM)
- Qwen2-VL models

## 📹 Demo Videos

Videos are stored locally (too large for GitHub):
- `ideal_replanning_demo.mp4` - Full 3-cycle demo
- `orchestrator_videos/combined_5view_labeled.mp4` - Multi-camera view
- `demo_videos/` - Individual demo recordings

## 👤 Author

Narendhiran V - IIIT Hyderabad

## 📄 License

Research use only.
