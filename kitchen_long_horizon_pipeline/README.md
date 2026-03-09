# Kitchen Long-Horizon Pipeline Snapshot

## Working commands used in the final pipeline

These are the current commands worth preserving from the recent working setup. This section intentionally excludes outdated intermediate/prototype runs.

Important:

- these commands were run from the original local kitchen project root, not from this archival snapshot,
- the original root was the `TAMP-PDDL` workspace,
- some commands require the RLBench/CoppeliaSim environment and the project Python environment to already be configured.

### Ground-truth orchestrator

Run the full trusted kitchen execution:

```bash
python ground_truth_orchestrator.py
```

### Main VLM pipeline

Run direct VLM planning/execution for the kitchen goal:

```bash
python -m vlm_pipeline.vlm_main --goal "Keep all groceries in cupboard and all mugs inside the box"
```

### Failure-triggered replanning pipeline

Run replanning when execution failures occur:

```bash
python -m vlm_pipeline.vlm_with_replanning --goal "Keep all groceries in the cupboard and mugs inside the box"
```

Run the same flow with the mock planner for demonstration/debugging:

```bash
python -m vlm_pipeline.vlm_with_replanning --goal "Keep all groceries in the cupboard and mugs inside the box" --mock
```

### Failure-case test suite

Run the explicit failure checks used for interruption and replan prompting:

```bash
python -m vlm_pipeline.test_failure_cases
```

### Live segmentation runner

Run the ground-truth pipeline with live segmentation:

```bash
python run_live_segmentation.py
```

Optional Tkinter backend for the live viewer:

```bash
LIVE_SEG_VIEWER_BACKEND=tkinter python run_live_segmentation.py
```

### Segmentation-based execution / discovery checks

Run segmentation-backed orchestration:

```bash
python run_with_segmentation.py
```

Run the segmentation orchestrator:

```bash
python run_segmentation_orchestrator.py
```

Run the lid-opening visibility count check:

```bash
python run_open_slide_count_check.py
```

### Live VLM discovery + replanning demos

Run the live discovery/replan pipeline:

```bash
python run_vlm_discovery_replan_live.py
```

Run the focused open-lid -> discovery -> replan demo:

```bash
python run_open_slide_vlm_replan_live.py
```

### Remote VLM server

Start the remote inference server on the GPU machine:

```bash
python -m vlm_pipeline.vlm_server --port 8000
```

Run the lighter model variant:

```bash
python -m vlm_pipeline.vlm_server --port 8000 --model Qwen/Qwen2-VL-2B-Instruct --no-4bit
```

Test the remote connection from the client side:

```bash
python -m vlm_pipeline.vlm_client --url http://localhost:8000
```

### Variation-specific ground-truth runs

Run the dedicated variation executors:

```bash
python variation_1_easy/ground_truth_orchestrator_variation1_easy.py
python variation_2/ground_truth_orchestrator_variation2.py
python variation_3_hard/ground_truth_orchestrator_variation3_hard.py
```

Run their live segmentation variants:

```bash
python variation_1_easy/run_live_segmentation_variation1_easy.py
python variation_2/run_live_segmentation_variation2.py
python variation_3_hard/run_live_segmentation_variation3_hard.py
```

## What this folder is

This folder is a curated snapshot of the kitchen-task work that lives in the local `TAMP-PDDL` project. It was added into this repository to preserve, document, and publish the long-horizon kitchen pipeline without disturbing the original local workspace.

This snapshot focuses on:

- the kitchen-task environment and ground-truth execution stack,
- the VLM planning and replanning pipeline,
- the remote VLM server/client path,
- the segmentation and live-visibility subsystem,
- the failure-handling and replanning logic,
- the kitchen-task scene variations,
- the legacy/prototype scripts that led to the current pipeline.

The source files copied here are stored under `source/`.
A raw copy manifest is stored in `COPIED_FILES.txt`.

---

## Why this snapshot exists

The local project had grown into a large experimental workspace containing:

- stable execution code,
- prototype scripts,
- multiple task variations,
- VLM planning code,
- segmentation viewers and recorders,
- remote serving support,
- baselines and constrained planning variants,
- failure-case tests and replanning experiments.

The purpose of this snapshot is to make that work legible in one place and to keep the relevant files together inside the GitHub repository.

---

## High-level project summary

The kitchen pipeline is a long-horizon manipulation stack built on top of RLBench, PyRep, and PDDLStream. The broad workflow is:

1. load a kitchen scene in RLBench/CoppeliaSim,
2. represent the scene through environment wrappers and object/region mappings,
3. optionally capture segmentation-based visibility instead of assuming full scene knowledge,
4. build a prompt and visual context for a VLM,
5. obtain a symbolic action skeleton such as `pick`, `place`, and `open-lid`,
6. translate that skeleton into executable PDDL/PDDLStream actions,
7. execute actions in simulation,
8. stop on failure or on discovery of newly visible objects,
9. replan using the updated state.

The project therefore combines:

- symbolic planning,
- VLM-driven task planning,
- segmentation-grounded perception,
- long-horizon execution,
- failure-triggered replanning,
- discovery-triggered replanning,
- ground-truth scripted orchestration.

---

## Architectural layers

### 1. Scene and environment layer
This layer loads the kitchen scene, exposes named objects and regions, and provides robot/execution helpers.

Main files:

- `source/rlbench_kitchen_env.py`
- `source/rlbench_kitchen_env_constrained.py`
- `source/rlbench_kitchen_streams.py`
- `source/rlbench_kitchen_streams_constrained.py`
- `source/pddl/rlbench_kitchen_domain.pddl`
- `source/pddl/rlbench_kitchen_domain_constrained.pddl`
- `source/pddl/rlbench_kitchen_streams.pddl`

### 2. Ground-truth execution layer
This layer executes the kitchen task with hand-authored, trusted action logic.

Main file:

- `source/ground_truth_orchestrator.py`

### 3. VLM planning layer
This layer builds prompts, queries a VLM, parses action skeletons, and executes them.

Main files:

- `source/vlm_pipeline/vlm_main.py`
- `source/vlm_pipeline/vlm_context_aggregator.py`
- `source/vlm_pipeline/vlm_planner.py`
- `source/vlm_pipeline/skeleton_to_pddl.py`
- `source/vlm_pipeline/vlm_executor_v2.py`

### 4. Replanning and monitoring layer
This layer stops execution on failure or discovery events and requests corrected plans.

Main files:

- `source/vlm_pipeline/vlm_with_replanning.py`
- `source/vlm_pipeline/execution_monitor.py`
- `source/vlm_pipeline/test_failure_cases.py`
- `source/Failure_checks_list.txt`

### 5. Segmentation and live visibility layer
This layer detects visible objects from segmentation masks and supports live visualization.

Main files:

- `source/segmentation_object_detector.py`
- `source/live_segmentation_viewer.py`
- `source/tkinter_segmentation_viewer.py`
- `source/segmentation_mask_viewer.py`
- `source/segmentation_display.py`
- `source/live_object_tracker.py`

### 6. Remote inference layer
This layer allows the VLM to be hosted on a separate GPU server and accessed over HTTP.

Main files:

- `source/vlm_pipeline/vlm_server.py`
- `source/vlm_pipeline/vlm_client.py`
- `source/vlm_pipeline/REMOTE_VLM_SETUP.md`
- `source/vlm_pipeline/start_vlm_tunnel.sh`
- `source/vlm_pipeline/sync_to_server.sh`

---

## Kitchen-task variations created in this project

The kitchen work includes three explicit scene variations, plus proposal scenes used during earlier task design.

### Variation 1: Easy

Scene and runner files:

- `source/task1_variation1.ttt`
- `source/variation_1_easy/README.md`
- `source/variation_1_easy/ground_truth_orchestrator_variation1_easy.py`
- `source/variation_1_easy/run_live_segmentation_variation1_easy.py`
- `source/variation_1_easy/ground_truth_plan.txt`

Distinguishing sequence:

1. mug on box -> placement boundary,
2. mug in cupboard -> placement boundary,
3. open box lid,
4. grocery in box -> cupboard,
5. grocery on table -> cupboard,
6. first table mug -> box,
7. second table mug -> box.

Why it matters:

- this is the simplest multi-stage kitchen variant,
- it already includes box access, cupboard interaction, and object relocation,
- it is useful for validating the pipeline before moving to denser scenes.

### Variation 2

Scene and runner files:

- `source/task1_variation2.ttt`
- `source/variation_2/README.md`
- `source/variation_2/ground_truth_orchestrator_variation2.py`
- `source/variation_2/run_live_segmentation_variation2.py`
- `source/variation_2/ground_truth_plan.txt`

Distinguishing sequence:

1. cupboard mug -> placement boundary,
2. table grocery -> cupboard,
3. mug on box -> placement boundary,
4. open box lid,
5. grocery in box -> cupboard,
6. both table mugs -> box with non-overlapping slot logic.

Why it matters:

- this introduces more runtime object selection,
- it needs box-slot management,
- it mixes cupboard, box-top, and inside-box reasoning.

### Variation 3: Hard

Scene and runner files:

- `source/task1_variation3.ttt`
- `source/variation_3_hard/README.md`
- `source/variation_3_hard/ground_truth_orchestrator_variation3_hard.py`
- `source/variation_3_hard/run_live_segmentation_variation3_hard.py`
- `source/variation_3_hard/ground_truth_plan.txt`

Distinguishing sequence:

1. cupboard mug -> placement boundary,
2. table grocery -> cupboard,
3. mug on box -> placement boundary,
4. open box lid,
5. grocery in box -> cupboard,
6. all table mugs -> box with lid-open checks and non-overlapping placement.

Why it matters:

- this is the densest kitchen version among the three,
- it stresses repeated box access,
- it makes object discovery and stable placement more important.

### Earlier proposal scenes and task-design assets

- `source/task_design_proposal_variation_1.ttt`
- `source/task_design_proposal_variation_1_modified.ttt`
- `source/task_design_proposal_variation_2.ttt`

These are relevant because they document the design trajectory that led to the finalized task variations.

---

## Current pipeline versus older scripts

### Current canonical stack

The most important current files are:

- `source/ground_truth_orchestrator.py`
- `source/vlm_pipeline/vlm_main.py`
- `source/vlm_pipeline/vlm_with_replanning.py`
- `source/vlm_pipeline/vlm_executor_v2.py`
- `source/segmentation_object_detector.py`
- `source/run_vlm_discovery_replan_live.py`
- `source/run_open_slide_vlm_replan_live.py`

These represent the consolidated kitchen pipeline.

### Older monolithic or transitional scripts

The following are important for project history and implementation evolution:

- `source/vlm_orchestrator.py`
- `source/vlm_execute_pipeline.py`
- `source/orchestrator.py`
- `source/orchestrator copy.py`
- `source/orchestrator_working_till_cupboard_pick.py`

These are preserved because they show how the current modular pipeline emerged from earlier direct orchestration scripts.

### Primitive prototypes and GUI bring-up scripts

These scripts are not the final pipeline, but they are relevant because they capture the step-by-step development of grasps, placements, and box/cupboard interactions:

- `source/generalize_pick_place_gui.py`
- `source/generalize_PP_box_gui.py`
- `source/place_hover_pick_hover_gui.py`
- `source/script_3_open_grasp_open_box_gui.py`
- `source/script1_cupboard_hover_gui.py`
- `source/script2_cupboard_hover_pick_gui.py`
- `source/script3_cupboard_retrieve_hover_pick_gui.py`
- `source/script4_cupboard_pick_place_gui.py`
- `source/test_crackers_pick_place.py`
- `source/test_spam_pick_place.py`

---

## Failure handling in this project

The explicit failure cases captured for VLM interruption and replanning are documented in:

- `source/Failure_checks_list.txt`
- `source/vlm_pipeline/test_failure_cases.py`

Failures already considered:

1. `OBJECT_BLOCKED`
2. `LID_CLOSED`
3. `OBJECT_NOT_FOUND`
4. valid success path
5. `ORPHAN_PLACE`
6. `PICK_MISMATCH`

Additional possible failures called out in project notes:

- `PLACEMENT_FAILED`
- `GRASP_FAILED`

Why this matters:

- the execution pipeline does not simply continue after an invalid plan,
- failures are meant to stop execution,
- the failure code and message are passed back into the replanning prompt,
- this is one of the core behaviors that distinguishes the current pipeline from direct one-shot VLM execution.

---

## Segmentation and visibility design

A major contribution of this project is that planning can be restricted to what segmentation actually sees.

That means the system can move from a naive mode:

- assume full scene knowledge,

into a grounded mode:

- only expose currently visible objects to the planner,
- detect newly visible objects after an action,
- trigger replanning if a hidden-but-relevant object appears.

This is especially important for inside-box objects that become visible only after the lid is opened.

Key files for this logic:

- `source/segmentation_object_detector.py`
- `source/live_segmentation_viewer.py`
- `source/tkinter_segmentation_viewer.py`
- `source/segmentation_mask_viewer.py`
- `source/vlm_pipeline/vlm_context_aggregator.py`
- `source/vlm_pipeline/vlm_with_replanning.py`
- `source/run_open_slide_count_check.py`
- `source/run_open_slide_vlm_replan_live.py`
- `source/run_vlm_discovery_replan_live.py`

---

## Remote VLM server design

The remote VLM server path exists because the main environment and visualization may run locally while model inference runs on a stronger GPU machine.

Core files:

- `source/vlm_pipeline/vlm_server.py`
- `source/vlm_pipeline/vlm_client.py`
- `source/vlm_pipeline/REMOTE_VLM_SETUP.md`
- `source/vlm_pipeline/start_vlm_tunnel.sh`
- `source/vlm_pipeline/sync_to_server.sh`

This supports:

- serving a Qwen VLM over HTTP,
- health checks,
- plan requests with image and prompt payloads,
- SSH tunneling from a local machine,
- remote GPU use without moving the whole simulation loop.

---

## File-by-file inventory and what each file contributes

This section is intentionally exhaustive.

### A. Core execution and environment files

- `source/ground_truth_orchestrator.py`  
  Canonical hand-authored long-horizon kitchen executor. It performs the trusted end-to-end action sequence, owns recording callbacks, and contains the robust motion/execution utilities that much of the rest of the stack builds on.

- `source/ground_truth_plan.txt`  
  Human-written reference plan for the canonical kitchen task.

- `source/rlbench_kitchen_env.py`  
  Main kitchen environment wrapper. It centralizes object lookup, region definitions, camera access, and robot/environment helpers.

- `source/rlbench_kitchen_env_constrained.py`  
  Constraint-aware environment variant used for stricter execution behavior and rule-aware planning.

- `source/rlbench_kitchen_streams.py`  
  Main PDDLStream bridge. Exposes stream samplers and motion/planning hooks used during symbolic execution.

- `source/rlbench_kitchen_streams_constrained.py`  
  Constrained stream variant that works with rule-patched planning.

- `source/video_recorder.py`  
  RGB execution recording utility.

- `source/combine_camera_videos.py`  
  Utility that combines multiple camera outputs into a single labeled panel video.

### B. PDDL and rule files

- `source/pddl/rlbench_kitchen_domain.pddl`  
  Base kitchen domain specification.

- `source/pddl/rlbench_kitchen_domain_constrained.pddl`  
  Rule-augmented kitchen domain.

- `source/pddl/rlbench_kitchen_streams.pddl`  
  Stream declaration file for the kitchen domain.

- `source/domain_patcher.py`  
  Utility that patches or rewrites the domain to inject rule constraints.

- `source/planning_rules.py`  
  Rule definitions and data structures used for constrained planning.

- `source/kitchen_rules_manual.py`  
  Manually authored kitchen commonsense constraints.

- `source/constraints_and_helpers.txt`  
  Important design memo describing task sequence, home-reset expectations, region mapping, ghost mode, lid behavior, and clearance constraints.

- `source/solve_with_rules.py`  
  Rule-aware solve path that applies the constrained planning setup.

### C. Baseline files

- `source/baseline_0_just_PDDL.py`  
  Baseline using plain PDDL/PDDLStream-style planning without the full VLM/replanning stack.

- `source/baseline_1_with_rules.py`  
  Baseline that incorporates manual rules and constraints.

- `source/run_baseline.sh`  
  Shell entry point for baseline execution.

- `source/run_baseline_experiment.sh`  
  Shell entry point for repeated or experiment-style baseline runs.

### D. Segmentation and perception files

- `source/segmentation_object_detector.py`  
  Segmentation-backed visible-object detector based on RLBench mask cameras and object-handle decoding.

- `source/live_segmentation_viewer.py`  
  OpenCV-based multiprocessing live segmentation viewer with separate process isolation.

- `source/tkinter_segmentation_viewer.py`  
  Tkinter fallback viewer used when the OpenCV/Qt path is unstable.

- `source/segmentation_mask_viewer.py`  
  File/image-style segmentation visualizer for inspecting masks and detected objects.

- `source/segmentation_display.py`  
  Additional live display utility for segmentation and object status visualization.

- `source/segmentation_video_recorder.py`  
  Recorder for segmentation mask videos.

- `source/live_panel_video_recorder.py`  
  Recorder for the live panel visualization output.

- `source/segmentation_viewer.py`  
  Older or alternate segmentation visualization utility retained for completeness.

- `source/live_object_tracker.py`  
  Tracks currently visible objects during execution.

### E. Segmentation-enabled runner scripts

- `source/run_live_segmentation.py`  
  Launches the main ground-truth execution with live segmentation viewing.

- `source/run_with_segmentation.py`  
  Uses true segmentation-derived visibility instead of hardcoded object assumptions.

- `source/run_segmentation_orchestrator.py`  
  Integrates segmentation with orchestration and visibility-change logic.

- `source/run_record_segmentation.py`  
  Records segmentation outputs during execution.

- `source/run_with_display.py`  
  Runs with a display-focused visualization path.

- `source/run_with_tracker.py`  
  Runs with object tracking enabled.

- `source/run_tracker_standalone.py`  
  Standalone object-tracker utility.

- `source/run_tkinter_segmentation.py`  
  Tkinter-based live segmentation runner.

- `source/run_open_slide_count_check.py`  
  Focused utility to check visibility count changes around lid opening.

### F. Root-level VLM scripts and artifacts

- `source/vlm_execute_pipeline.py`  
  Older root-level bridge from VLM output to execution.

- `source/vlm_orchestrator.py`  
  Older direct VLM orchestration path before the packaged `vlm_pipeline` layout became primary.

- `source/VLM_replan_mock_action_sequence.txt`  
  Mock action sequence used for replanning demonstrations.

- `source/test_vlm_execute.py`  
  Test harness for root-level VLM execution.

- `source/test_vlm_output.txt`  
  Sample VLM output used for testing and parser validation.

### G. Packaged `vlm_pipeline` files

- `source/vlm_pipeline/__init__.py`  
  Package exports.

- `source/vlm_pipeline/README.md`  
  Original package-level documentation for the VLM pipeline.

- `source/vlm_pipeline/ALL_SCRIPTS.txt`  
  Informal usage index and experiment notes.

- `source/vlm_pipeline/REMOTE_VLM_SETUP.md`  
  End-to-end remote VLM hosting instructions.

- `source/vlm_pipeline/vlm_main.py`  
  Primary entry point for the current VLM pipeline.

- `source/vlm_pipeline/vlm_with_replanning.py`  
  Replanning-enabled VLM entry point.

- `source/vlm_pipeline/vlm_context_aggregator.py`  
  Builds the state text, captures images, and supports segmentation-visible-object mode.

- `source/vlm_pipeline/context_aggregator_v2.py`  
  Alternate or generalized context-aggregation path.

- `source/vlm_pipeline/vlm_planner.py`  
  Local VLM loader, planner, parser, and validator.

- `source/vlm_pipeline/vlm_executor.py`  
  Earlier executor implementation.

- `source/vlm_pipeline/vlm_executor_v2.py`  
  Current executor implementation with PDDL-based pick/place/open-lid execution and retry logic.

- `source/vlm_pipeline/skeleton_to_pddl.py`  
  Converts VLM action skeletons into PDDL-oriented execution structures.

- `source/vlm_pipeline/execution_monitor.py`  
  Runtime execution monitor and failure classifier.

- `source/vlm_pipeline/vlm_client.py`  
  Remote planner client for HTTP-based inference.

- `source/vlm_pipeline/vlm_server.py`  
  FastAPI server for remote VLM planning.

- `source/vlm_pipeline/demo_replanning.py`  
  Demonstration script for staged replanning behavior.

- `source/vlm_pipeline/ablation_runner.py`  
  Runner for ablations and comparison experiments.

- `source/vlm_pipeline/test_failure_cases.py`  
  Failure-case test suite for the VLM execution pipeline.

- `source/vlm_pipeline/test_single_pick.py`  
  Small executor/planner smoke test.

- `source/vlm_pipeline/MOCK_REPLAN_SEQUENCE.txt`  
  Mock sequence used in replanning experiments.

- `source/vlm_pipeline/example1_vlm_full`  
  Example artifact for full VLM execution.

- `source/vlm_pipeline/example2_vlm_full`  
  Example artifact for full VLM execution.

- `source/vlm_pipeline/example3_vlm_replanning.txt`  
  Example artifact for replanning.

- `source/vlm_pipeline/start_vlm_tunnel.sh`  
  SSH tunnel helper for remote inference.

- `source/vlm_pipeline/sync_to_server.sh`  
  Sync helper for moving pipeline code to the remote server.

- `source/vlm_pipeline/prompts/`  
  Prompt templates used for planning and replanning.

- `source/vlm_pipeline/configs/`  
  Configuration assets for generalized pipeline behavior.

- `source/vlm_pipeline/constraints/`  
  Constraint definitions for scene-agnostic or structured prompt/state logic.

### H. Legacy orchestration and prototype files

- `source/orchestrator.py`  
  Earlier orchestrator that chains stages in a more manual way.

- `source/orchestrator copy.py`  
  Backup copy of an older orchestrator.

- `source/orchestrator_working_till_cupboard_pick.py`  
  Intermediate orchestrator snapshot from earlier development.

- `source/generalize_pick_place_gui.py`  
  Generic pick-place prototype.

- `source/generalize_PP_box_gui.py`  
  Box-focused pick-place prototype.

- `source/place_hover_pick_hover_gui.py`  
  Hover-based place/pick motion prototype.

- `source/script_3_open_grasp_open_box_gui.py`  
  Dedicated box-lid opening prototype.

- `source/script1_cupboard_hover_gui.py`  
  Early cupboard hover prototype.

- `source/script2_cupboard_hover_pick_gui.py`  
  Early cupboard hover-and-pick prototype.

- `source/script3_cupboard_retrieve_hover_pick_gui.py`  
  Early cupboard retrieval prototype.

- `source/script4_cupboard_pick_place_gui.py`  
  Early cupboard pick-and-place prototype.

- `source/test_crackers_pick_place.py`  
  Focused grocery pick/place test for crackers.

- `source/test_spam_pick_place.py`  
  Focused grocery pick/place test for spam.

### I. Variation-specific files

#### Variation 1 files

- `source/variation_1_easy/README.md`  
  Variation-specific usage note.

- `source/variation_1_easy/ground_truth_orchestrator_variation1_easy.py`  
  Dedicated runner for the easy variation.

- `source/variation_1_easy/ground_truth_plan.txt`  
  Reference plan for the easy variation.

- `source/variation_1_easy/run_live_segmentation_variation1_easy.py`  
  Live segmentation runner for the easy variation.

#### Variation 2 files

- `source/variation_2/README.md`  
  Variation-specific usage note.

- `source/variation_2/ground_truth_orchestrator_variation2.py`  
  Dedicated runner for variation 2.

- `source/variation_2/ground_truth_plan.txt`  
  Reference plan for variation 2.

- `source/variation_2/run_live_segmentation_variation2.py`  
  Live segmentation runner for variation 2.

#### Variation 3 files

- `source/variation_3_hard/README.md`  
  Variation-specific usage note.

- `source/variation_3_hard/ground_truth_orchestrator_variation3_hard.py`  
  Dedicated runner for the hard variation.

- `source/variation_3_hard/ground_truth_plan.txt`  
  Reference plan for the hard variation.

- `source/variation_3_hard/run_live_segmentation_variation3_hard.py`  
  Live segmentation runner for the hard variation.

### J. Scene assets

- `source/task1_variation1.ttt`  
  Variation 1 scene.

- `source/task1_variation2.ttt`  
  Variation 2 scene.

- `source/task1_variation3.ttt`  
  Variation 3 scene.

- `source/task_design_proposal_variation_1.ttt`  
  Earlier design scene proposal.

- `source/task_design_proposal_variation_1_modified.ttt`  
  Modified proposal scene.

- `source/task_design_proposal_variation_2.ttt`  
  Additional design proposal scene.

### K. Notes and failure artifacts

- `source/Failure_checks_list.txt`  
  Enumerates explicit failure checks for replanning.

- `source/constraints_and_helpers.txt`  
  Consolidated design notes for constraints and helpers.

---

## What was actually added to this GitHub branch

This branch adds a self-contained documentation-and-source snapshot under:

- `kitchen_long_horizon_pipeline/`

That folder contains:

- `README.md` — this documentation,
- `COPIED_FILES.txt` — manifest of copied files,
- `source/` — curated file snapshot from the local kitchen project.

The intent is archival and documentation-first:

- preserve the current kitchen work,
- make the system understandable to a reader,
- avoid mixing the snapshot into unrelated parts of the repository.

---

## Recommended reading order

For a new reader, the best order is:

1. `README.md` in this folder,
2. `source/ground_truth_orchestrator.py`,
3. `source/rlbench_kitchen_env.py`,
4. `source/vlm_pipeline/vlm_main.py`,
5. `source/vlm_pipeline/vlm_context_aggregator.py`,
6. `source/vlm_pipeline/vlm_executor_v2.py`,
7. `source/vlm_pipeline/vlm_with_replanning.py`,
8. `source/segmentation_object_detector.py`,
9. variation-specific runners under `source/variation_1_easy/`, `source/variation_2/`, and `source/variation_3_hard/`.

---

## Closing summary

This kitchen snapshot captures the evolution from:

- direct scripted execution,
- to rule-aware symbolic planning,
- to VLM planning,
- to failure-aware replanning,
- to segmentation-grounded object discovery,
- to variation-specific long-horizon task execution.

It includes both the current stable paths and the historical prototype scripts that explain how the final pipeline was assembled.
