# LLM-Guided Physics-Aware TAMP Pipeline (Distributed Architecture)

This document explains the architecture of the distributed LLM-Guided Task and Motion Planning (TAMP) pipeline. 

To overcome the inherent Qt/multithreading segmentation faults of PyRep/CoppeliaSim and the massive GPU VRAM requirements of cutting-edge Vision-Language Models (VLMs), the system has been refactored into a **Distributed 3-Node Architecture**.

---

## The Distributed Architecture

The system relies on a strict separation of concerns across physical processes and network boundaries.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          LOCAL WORKSTATION                           │
│                                                                      │
│  ┌────────────────────────┐         ┌─────────────────────────────┐  │
│  │   NODE 1 (Physics)     │         │      NODE 2 (Client)        │  │
│  │ rlbench_zmq_server.py  │←─ZMQ───→│   vlm_network_client.py     │  │
│  │ ---------------------- │ (5555)  │  -------------------------  │  │
│  │ - CoppeliaSim GUI      │         │ - Orchestrates loop         │  │
│  │ - PDDL Executor        │         │ - Maintains failure history │  │
│  │ - Extractor / Monitor  │         │ - Parses Action Skeletons   │  │
│  └────────────────────────┘         └──────────────┬──────────────┘  │
└────────────────────────────────────────────────────┼────────────────┘
                                                     │ HTTP JSON (800x)
                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       REMOTE GPU SERVER (gvlab2)                     │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                        NODE 3 (Brain)                         │   │
│  │                     inference_server.py                       │   │
│  │  -----------------------------------------------------------  │   │
│  │  - FastAPI REST Server                                        │   │
│  │  - Qwen2.5-7B (or Qwen2-VL)                                   │   │
│  │  - Generates plans from prompt context                        │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules and Files Used

The architecture is divided cleanly between the PyRep Physics Node, the Orchestration Client Node, and the Inference Node.

### NODE 1: The Physics Server (`rlbench_zmq_server.py`)
This process **must run natively on the desktop** as it requires the X11/Qt graphical display to launch CoppeliaSim. It listens on ZeroMQ port `5555`.
*   **`rlbench_zmq_server.py`**: The main entry point. Initializes PyRep safely on the main thread and runs a non-blocking `zmq.Poller` loop to step the physics engine while awaiting network queries.
*   **`dynamic_state_extractor.py`**: Runs inside the server. Uses `simGetObjects` to discover dynamic objects and perform IK reachability checks. Formats the data into a structured semantic prompt.
*   **`pddl_executor.py`**: Runs inside the server. Takes the discrete Action Skeletons (e.g., `pick(mug) -> place(mug, table)`) sent over network, translates them into continuous PDDL problems, solves for joint kinematics, and moves the simulated arm.
*   **`execution_monitor.py`**: Monitors the simulation for dropped objects or mathematical IK failures, returning causal strings back to the network client.

### NODE 2: The Orchestration Client (`vlm_network_client.py`)
This is a lightweight Python script that acts as the "Manager". It does **not** import PyRep, Qt, Torch, or Transformers.
*   **`vlm_network_client.py`**: Contains `NetworkLLMPipeline`. Runs the main `while(retries)` replanning loop.
    1. Sends `GET_STATE` over ZMQ to Node 1.
    2. Packages the state, goal, and past failures into a text prompt.
    3. Calls Node 3 (Inference) to generate the next action plan.
    4. Parses the text into an `ActionSkeleton` list.
    5. Sends `EXECUTE_ACTION` over ZMQ to Node 1.
*   **`inference_client.py`**: A helper module used by Node 2 to cleanly format REST HTTP requests and parse the outputs from the remote FastAPI Inference Server.

### NODE 3: The Inference Server (`inference_server.py`)
This process runs on a remote high-GPU machine (e.g., `gvlab2.iiit.ac.in`) and is usually accessed via an SSH tunnel on port `800x`.
*   **`inference_server.py`**: A FastAPI web server that loads heavy Transformer ML models (`Qwen2.5` or `Qwen2-VL`) into GPU VRAM. It simply receives text (and optionally images), runs inference, and returns the raw string response.

---

## Detailed Walkthrough: "Put all groceries in cupboard"

This section traces exactly what happens across the distributed pipeline when a complex semantic query is initiated.

**Scenario**: The user wants the robot to move multiple grocery items (`spam`, `sugar`, `crackers`, `mustard`) from the table into the cupboard.

### Step 1: User Initiation (`vlm_network_client.py`)
The pipeline runs autonomously from the client node. The user passes the natural language task query as a command-line argument:
```bash
python -m vlm_pipeline.vlm_network_client --goal "Put all groceries in cupboard" --llm-url http://10.10.16.68:8001
```

### Step 2: State Request (`vlm_network_client.py` ➔ `rlbench_zmq_server.py`)
At the start of the `while(retries)` loop, the client sends a `{"type": "GET_STATE", "goal": ...}` JSON payload over the ZeroMQ socket to the physics server (Node 1).

### Step 3: Scene Extraction (`dynamic_state_extractor.py`)
The physics server receives the request and pauses the PyRep simulation visually. It triggers the `DynamicStateExtractor`.
*   It sweeps the PyRep memory block using `simGetObjects`.
*   It finds the interactive items: `spam`, `sugar`, `crackers`, `mustard`, `mug1`, `mug3`, `mug4`.
*   It performs a 50ms mathematical Inverse Kinematics (IK) check on each object to ensure the robot can physically reach them.
*   **Actual Output Sent Over Network:**
```text
 === CURRENT SEMANTIC STATE ===

## Robot Status:
- gripper: empty

## Dynamically Discovered Objects:
- spam: location=on table
- sugar: location=inside box
- crackers: location=on table
- mustard: location=on table
- box_lid: state=closed, location=on box, STATUS=BLOCKED_BY_None
- mug1: location=inside box
- mug3: location=on table
- mug4: location=inside box

## Valid Target Regions:
- table: main dining table surface
- box-top: top surface of the closed box
- box-inside: interior of the box (accessible when lid is open)
- placement_boundary: target area for placing mugs
- cupboard_boundary: inside the cupboard shelf
```

### Step 4: Prompt Construction & LLM Query (`vlm_network_client.py` ➔ `inference_server.py`)
The client receives the text state. It concatenates the original `--goal`, the textual physical state, and the system prompt constraints (e.g., "allowed actions are pick, place, open-lid") into a text payload.
*   It uses `inference_client.py` to fire an HTTP POST request to the remote GPU server at `10.10.16.68:8001`.

### Step 5: AI Reasoning (`inference_server.py`)
The Fast-API server (Node 3) feeds the prompt into the VRAM-loaded Qwen model. The LLM reads the constraints, outputs a chain-of-thought, and concludes with a formatted sequence.
*   **Actual Output Received from LLM:**
```text
<think>
Okay, let's tackle this problem. The goal is to put all groceries in the cupboard and mugs inside the box. First, I need to understand the current state.
Looking at the objects: spam, sugar, crackers, mustard are on the table or inside the box. The box_lid is closed, so the box-inside is inaccessible...
...
So the sequence would be:
1. open-lid(box_lid)
2. pick(sugar)
3. place(sugar, cupboard_boundary)
4. pick(spam)
...
</think>

1. open-lid(box_lid)  
2. pick(sugar)  
3. place(sugar, cupboard_boundary)  
4. pick(spam)  
5. place(spam, cupboard_boundary)  
6. pick(crackers)  
7. place(crackers, cupboard_boundary)  
8. pick(mustard)  
9. place(mustard, cupboard_boundary)  
10. pick(mug3)  
11. place(mug3, box-inside)
```

### Step 6: Action Parsing (`vlm_network_client.py`)
The client receives the HTTP response text from the GPU. It uses regular expressions to extract `pick`, `place`, and `open-lid` calls and bundles them into Python `ActionSkeleton` dataclasses.
*   **Actual Python Array Generated:**
```python
[
    ActionSkeleton(action_name='open-lid', args=('box_lid',)),
    ActionSkeleton(action_name='pick', args=('sugar',)),
    ActionSkeleton(action_name='place', args=('sugar', 'cupboard_boundary')),
    ActionSkeleton(action_name='pick', args=('spam',)),
    ActionSkeleton(action_name='place', args=('spam', 'cupboard_boundary')),
    ActionSkeleton(action_name='pick', args=('crackers',)),
    ActionSkeleton(action_name='place', args=('crackers', 'cupboard_boundary')),
    ActionSkeleton(action_name='pick', args=('mustard',)),
    ActionSkeleton(action_name='place', args=('mustard', 'cupboard_boundary')),
    ActionSkeleton(action_name='pick', args=('mug3',)),
    ActionSkeleton(action_name='place', args=('mug3', 'box-inside'))
]
```
*   This array is converted to a JSON payload `{"type": "EXECUTE_ACTION", "actions": [...]}` and sent over ZMQ to the Physics server.

### Step 7: Physics Execution (`pddl_executor.py`)
The physics server (Node 1) receives the action payload.
*   It groups `pick(spam)` and `place(spam, cupboard_boundary)` into an `execute_pick_place_pair` macro.
*   It passes this semantic requirement to PDDLStream (`streams.pddl`). PDDLStream uses RRT-Connect to find a collision-free 7-DOF joint angle trajectory that reaches the spam, closes the gripper, and moves to the inside of the cupboard without hitting the doors.
*   The PyRep engine interpolates the mathematical path and the robot physically moves on your screen.

### Step 8: Autonomous Looping & Completion
The execution succeeds. Node 1 returns `"status": "success"` to Node 2.
*   The Client loop restarts at Step 2.
*   This time, the Extractor inherently sees that `spam` is now `location=inside cupboard_boundary`.
*   The LLM prompt is updated. The LLM sees the new state, realizes `sugar`, `crackers`, and `mustard` are still on the table, and generates the next plan (e.g., `pick(crackers)`).
*   The loop continues until the LLM sees all items are in the cupboard and outputs an empty plan, completing the task.

---

## Benefits of this Architecture

1. **No Qt Segmentation Faults:** By isolating heavy libraries (`torch`, `fastapi`, `transformers`) entirely away from the script that launches PyRep, we eliminate the `QObject::setParent` threading crashes.
2. **No Hardware Bottlenecks:** The machine running the 3D physics simulation (usually limited by CPU/OpenGL) doesn't have to share its RAM/VRAM with a 7B parameter LLM.
3. **Hot-Swappable Brains:** The `inference_client.py` can be pointed to any IP address. We can instantly switch between a local tiny model, a massive remote server running `Qwen3-8B`, or even OpenAI's GPT-4o API without restarting the physics simulation.

---

## Deep Dive: How It Works Internally

### 1. Scene Files & Environment Definition
The physical environments are built on the **CoppeliaSim/RLBench** engine. 
*   **Location:** The core scenes are typically loaded from base `.ttt` (CoppeliaSim scene files) and configured dynamically using RLBench task scripts (which load `.ttm` task models).
*   **Interaction:** The system acts directly on the simulated PyRep objects rendered in these geometric scenes.

### 2. PDDL Domains & Streams
While the LLM provides high-level logic, the actual validity and collision bounds of those actions are defined in rigid PDDL format.
*   **Location:** `pddl/rlbench_kitchen_domain.pddl` and `pddl/rlbench_kitchen_streams.pddl`.
*   **Usage:** The domain file defines the abstract semantic actions (like `pick`, `place`, `open-lid`) and their mathematical requirements. The `streams.pddl` acts as the bridge, linking PyRep Inverse Kinematics (IK) solvers to the logical requirements. For example, before a `pick(X)` is valid, the stream solver must find a collision-free `Traj` (joint trajectory) array to grasp `X`.

### 3. Dynamic Object Extraction mechanism (`dynamic_state_extractor.py`)
We do not hardcode object states. Instead, we use a global extraction approach:
1.  **Discovery:** The system calls `sim.simGetObjects(sim.sim_handle_all)` directly on the physics engine memory.
2.  **Naming:** It strips structural artifacts (walls, lights) and uses `sim.simGetObjectAlias(handle)` to extract human-readable semantic names (`mug3`, `spam`, `box_lid`).
3.  **Physical Reachability:** To see if a mug inside a closed box is physically grabbable, it requests a 50ms dummy IK solve from the robot (`self.env.robot.solve_ik()`). If the math fails due to the closed lid, the Extractor accurately labels the object as `"STATUS=BLOCKED"`.

### 4. Action Execution mechanism (`pddl_executor.py`)
When the ZMQ server receives `pick(mug3) -> place(mug3, table)`:
1.  It groups them into an `execute_pick_place_pair` macro.
2.  **Trajectory Generation:** It asks PDDLStream to find a valid 7-Degree-Of-Freedom path for the robot arm using `solve_ik_via_sampling`.
3.  **Interpolation:** If mathematically sound, it retrieves the `waypoint` angles.
4.  **Actuation:** PyRep linearly interpolates the simulated robotic arm through those joint angles. The `gripper` state is toggled using `get_grasped_objects` manipulation hooks.

### 5. Autonomous Recovery Logic
If the Execution logic (Step 4) fails, the system does not crash. It leverages the LLM's vast contextual intelligence:
1.  **Failure Capture:** `vlm_network_client.py` captures the specific error across the ZMQ socket (e.g. `"Failed at action open-lid. Reason: object_blocked."`).
2.  **Prompt Engineering:** It appends a `PREVIOUS FAILURE:` context block to the user's prompt. It lists the exact actions that were attempted and specifically why the physical environment rejected them.
3.  **Corrective Replanning:** The payload is sent back to the Remote GPU (Node 3). The LLM reads the failure, recognizes it missed a prerequisite (like taking the book off the box first), and generates an entirely new, corrected chronological `ActionSkeleton` sequence for the next attempt.
