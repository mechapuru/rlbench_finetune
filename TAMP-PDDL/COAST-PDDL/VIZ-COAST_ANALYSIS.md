# VIZ-COAST & RLBench Integration - Complete Analysis

## 📄 Paper Summary: "Using VLM Reasoning to Constrain Task and Motion Planning"

### **Core Concept**
The paper introduces **VIZ-COAST** (Vision-Language Model + COnstraints And STreams), a method that uses Vision-Language Models (VLMs) to identify geometric constraints *a priori* in Task and Motion Planning (TAMP) problems, rather than discovering them through expensive trial-and-error during planning.

### **Key Problem Being Solved**
Traditional TAMP approaches suffer from:
- **Poor downward refinability**: High-level task plans that look valid but fail during motion planning
- **Expensive search**: Time wasted exploring infeasible branches
- **Repeated failures**: Learning constraints only after motion planning failures

### **VIZ-COAST Solution**
1. **Pre-planning Constraint Identification**: Use VLMs to analyze scene images and domain descriptions
2. **Spatial Reasoning**: Leverage VLM common-sense knowledge to identify geometric issues before planning
3. **Constraint Injection**: Add identified constraints to PDDL domain to prune infeasible plans early
4. **Result**: Dramatically reduced planning time and fewer downward refinement failures

### **Technical Architecture**
```
Scene Image + Domain Description
           ↓
     VLM Analysis (GPT-4V, etc.)
           ↓
  Spatial Constraints Extracted
           ↓
   Augmented PDDL Domain
           ↓
    COAST TAMP Algorithm
           ↓
  Feasible Plan (Fewer Failures)
```

### **Key Innovation**
Instead of adding constraints *reactively* (after failures), VIZ-COAST adds them *proactively* (before planning) using VLM spatial reasoning.

---

## 🏗️ Repository Structure Analysis

### **Repository: mechapuru/rlbench_long_horizon**
GitHub: https://github.com/mechapuru/rlbench_long_horizon/tree/main/integration_dir

### **High-Level Overview**
This is an implementation of COAST TAMP algorithm integrated with RLBench simulation environment, featuring a **pluggable task planner interface** that supports multiple planners (Fast Downward, LLM-based, etc.).

---

## 📁 Directory Structure Breakdown

```
integration_dir/
├── long_horizon_grill_pddl_files/    # PDDL files for the grill cooking task
│   ├── domain.pddl                   # TAMP domain definition
│   ├── problem.pddl                  # Initial state & goal
│   └── streams.pddl                  # Stream declarations
│
├── planners/                         # Pluggable task planner interface
│   ├── base.py                       # TaskPlanner ABC
│   ├── fast_downward.py              # Fast Downward wrapper
│   └── llm_planner.py                # LLM planner template
│
├── run.py                            # Main CLI entry point
├── world.py                          # RLBench/PyRep interface
├── actions.py                        # Geometric action implementations
├── streams.py                        # Stream implementations
├── config.py                         # Configuration management
├── __init__.py                       # Package initialization
└── README.md                         # Documentation
```

---

## 🔍 Detailed Component Analysis

### **1. world.py - RLBench World Interface**
**Purpose**: Bridge between COAST and RLBench/PyRep simulation

**Key Classes:**
- `ObjectInfo`: Metadata about scene objects
- `RLBenchWorld`: Main interface with methods for:
  - Object pose queries
  - Robot configuration management
  - Motion planning (via PyRep)
  - Collision checking
  - Gripper control
  - COAST integration helpers

**Key Methods:**
```python
# Robot control
get_robot_config() -> np.ndarray
set_robot_config(config: np.ndarray)
get_gripper_pose() -> np.ndarray

# Motion planning
plan_to_pose(position, orientation) -> Optional[Path]
solve_ik(position, orientation) -> Optional[config]

# COAST integration
get_stream_state() -> Set[str]  # Initial certified facts
get_coast_objects() -> List[Dict]  # Objects for planning
```

---

### **2. streams.py - Stream Implementations**
**Purpose**: Geometric samplers that ground symbolic actions

**Stream Classes:**
1. **SampleGrasp**: Generate grasp poses for objects
2. **SamplePose**: Generate placement poses on surfaces
3. **SampleIK**: Inverse kinematics solver
4. **SampleMotion**: Collision-free motion planning
5. **CheckCollision**: Collision detection test

**Data Classes:**
- `GraspPose`: Represents grasp configuration
- `PlacementPose`: Represents placement configuration
- `Trajectory`: Robot trajectory waypoints

**Pattern:**
```python
class SampleMotion(Stream):
    name = "sample-motion"
    inputs = [Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?t", "traj")]
    
    def sample(self, inputs, fluents) -> Optional[Dict]:
        # Use PyRep to plan collision-free path
        # Return trajectory or None if planning fails
```

---

### **3. actions.py - Geometric Action Implementations**
**Purpose**: Execute symbolic actions in simulation

**Action Classes:**
1. **Pick**: Grasp object from location
2. **Place**: Release object at target
3. **PlaceOnGrill**: Specialized placement on grill
4. **OpenLid**: Open grill lid (constrained trajectory)
5. **CloseLid**: Close grill lid (constrained trajectory)
6. **Cook**: Cooking action (wait + state change)

**Pattern:**
```python
class Pick(Action):
    def execute(self, inputs: Dict, outputs: Dict) -> bool:
        # Extract trajectory from stream outputs
        # Execute in simulation
        # Return success/failure
```

---

### **4. run.py - Main Entry Point**
**Purpose**: CLI interface for COAST + RLBench

**Workflow:**
1. Parse command-line arguments
2. Initialize RLBench world
3. Load task-specific PDDL files
4. Create task planner
5. Run COAST planning algorithm
6. (Optional) Execute plan in simulation

**Key Functions:**
```python
run_coast_planning(world, config, planner) -> dict
execute_plan(world, plan) -> bool
main() -> int
```

---

### **5. config.py - Configuration Management**
**Purpose**: Task-specific configuration

**Key Features:**
```python
TASK_PDDL_FOLDERS = {
    "LongHorizonGrillTask": "long_horizon_grill_pddl_files",
    # Easy to add new tasks
}

@dataclass
class CoastConfig:
    planner: str = "fast_downward"
    planner_timeout: int = 1200
    max_level: int = 6
    task_name: str = "LongHorizonGrillTask"
    # Auto-sets PDDL paths based on task
```

---

### **6. planners/ - Pluggable Planner Interface**

#### **base.py**
Defines abstract interface for task planners:
```python
class TaskPlanner(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def plan(self, domain_pddl, problem_state, pddl, task_horizon) -> PlanResult:
        """Returns PlanResult(success=bool, plan=nodes)"""
```

#### **fast_downward.py**
Wrapper for Fast Downward classical planner

#### **llm_planner.py**
Template for LLM-based task planning

**Design Pattern**: Strategy Pattern - easy to swap planners without changing COAST code

---

## 🎯 Example Task: LongHorizonGrillTask

### **Task Flow:**
1. `pick(chicken, grill_side)` → `place(chicken, grill_cooking_area)`
2. `close_lid` (constrained trajectory to avoid collisions)
3. `pick(plate, dish_rack)` → `place(plate, plate_target)`
4. `open_lid` (constrained trajectory)
5. `pick(chicken, grill_cooking_area)` → `place(chicken, plate_target)`

### **Locations:**
- `grill_side`: Starting position of chicken
- `grill_cooking_area`: Where chicken cooks (under lid)
- `dish_rack`: Starting position of plate
- `plate_target`: Final position for serving

### **Challenges:**
- Lid must be closed while chicken cooks
- Constrained trajectories to avoid hitting lid/objects
- Multi-object manipulation
- Long-horizon planning (5+ steps)

---

## 🔧 Technical Implementation Details

### **COAST Algorithm Integration**
The code integrates with the COAST library (COnstraints And STreams):

```python
import coast

plan = coast.plan(
    domain_pddl=config.domain_pddl,
    problem_pddl=config.problem_pddl,
    streams_pddl=config.streams_pddl,
    streams_py="streams.py",  # Python stream implementations
    algorithm="improved",
    max_level=6,
    search_sample_ratio=10.0,
    timeout=1200,
    world=world,  # RLBench world interface
    ...
)
```

### **Stream-Action Coordination**
1. **Streams** generate geometric parameters (grasps, poses, trajectories)
2. **Actions** execute using those parameters
3. COAST searches over symbolic + geometric space simultaneously

### **PDDL Structure**
```
domain.pddl:     Predicates, actions (symbolic)
problem.pddl:    Initial state, goal (symbolic)
streams.pddl:    Stream declarations (symbolic)
streams.py:      Stream implementations (geometric)
actions.py:      Action implementations (geometric)
```

---

## 🚀 How to Use

### **Basic Planning:**
```bash
cd integration_dir
python run.py --task LongHorizonGrillTask --planner fast_downward
```

### **Planning + Execution:**
```bash
python run.py --task LongHorizonGrillTask --execute
```

### **With GUI:**
```bash
python run.py --task LongHorizonGrillTask --execute  # GUI by default
```

### **Headless Mode:**
```bash
python run.py --task LongHorizonGrillTask --headless
```

### **CLI Options:**
```
--task, -t        RLBench task name
--planner, -p     Planner: fast_downward, llm
--timeout         Planning timeout (default: 1200s)
--max-level       COAST max level (default: 6)
--execute         Execute plan after planning
--headless        Run without GUI
```

---

## 🔌 Extensibility

### **Adding New Tasks:**
1. Create `my_task_pddl_files/` with domain.pddl, problem.pddl, streams.pddl
2. Register in `config.py`:
   ```python
   TASK_PDDL_FOLDERS["MyTask"] = "my_task_pddl_files"
   ```
3. Run: `python run.py --task MyTask`

### **Adding New Planners:**
1. Inherit from `TaskPlanner` in `planners/base.py`
2. Implement `name` property and `plan()` method
3. Register in `run.py`'s `create_planner()` function

### **Adding New Streams:**
1. Create class inheriting from `Stream` in `streams.py`
2. Define `name`, `inputs`, `outputs`, `certified`
3. Implement `sample()` method
4. Add to `get_streams()` function

### **Adding New Actions:**
1. Create class inheriting from `Action` in `actions.py`
2. Implement `execute()` method
3. Add to `get_actions()` function

---

## 🔗 Dependencies

### **Core Dependencies:**
- **RLBench**: Robotics simulation environment
- **PyRep**: Python interface to CoppeliaSim
- **COAST**: COnstraints And STreams TAMP library
- **Fast Downward**: Classical task planner
- **NumPy**: Numerical computing

### **Optional:**
- **OpenAI API**: For LLM planner
- **pdftotext**: For paper analysis

---

## 📊 Key Differences from Your Current Code

### **Your Workspace (TAMP-PDDL):**
- More experimental scripts
- Baseline implementations
- Manual rule definitions
- Custom orchestration

### **Collaborator's Repo (integration_dir):**
- Clean modular architecture
- Pluggable planner interface
- COAST library integration
- Production-ready structure
- Extensible design patterns

---

## 🎓 Learning Points

### **1. Separation of Concerns:**
- **World**: Simulation interface
- **Streams**: Geometric sampling
- **Actions**: Execution
- **Planners**: Task planning
- **Config**: Settings management

### **2. Abstract Interfaces:**
- `TaskPlanner` ABC allows swapping planners
- Stream/Action base classes enable extensibility

### **3. PDDL + Python Hybrid:**
- PDDL for symbolic reasoning
- Python for geometric computations
- COAST bridges the gap

### **4. Constraint-Based Planning:**
- Streams generate candidates
- Constraints prune infeasible options
- Incremental refinement

---

## 🔄 Connection to VIZ-COAST Paper

The repository implements the **COAST** part of VIZ-COAST:
- ✅ COAST TAMP algorithm
- ✅ Stream-based geometric sampling
- ✅ Constraint handling
- ❌ VLM-based constraint generation (not in this repo)

**The paper adds VLM layer on top:**
```
Your Collaborator's Repo:  COAST + RLBench
Paper's Contribution:       VLM → Constraints → COAST + RLBench
```

The VLM component would:
1. Analyze scene images
2. Generate spatial constraints
3. Augment PDDL domain
4. Feed into existing COAST implementation

---

## 📋 Clone & Setup Plan

### **Directory Structure After Clone:**
```
TAMP-PDDL/
├── [your existing files]
├── rlbench_long_horizon/           # Cloned repo
│   ├── integration_dir/            # Main implementation
│   │   ├── run.py
│   │   ├── world.py
│   │   ├── streams.py
│   │   ├── actions.py
│   │   ├── config.py
│   │   └── ...
│   ├── pddlstream execution/       # PDDLStream examples
│   ├── examples/                   # RLBench examples
│   └── ...
```

### **Setup Steps:**

#### **1. Clone Repository**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL
git clone https://github.com/mechapuru/rlbench_long_horizon.git
cd rlbench_long_horizon
```

#### **2. Install Dependencies**
```bash
# Activate conda environment (if exists)
conda activate rlbench  # or create new: conda create -n rlbench python=3.8

# Install RLBench
pip install -e RLBench/

# Install PyRep (requires CoppeliaSim)
# Download CoppeliaSim first from: https://www.coppeliarobotics.com/downloads
pip install pyrep

# Install COAST (if available)
# May need to clone COAST separately or install from package
pip install coast-tamp  # or pip install -e path/to/coast

# Install other requirements
pip install numpy
```

#### **3. Configure Environment**
```bash
cd integration_dir

# Check config.py to ensure paths are correct
# May need to update Fast Downward path in config.py

# Test basic import
python -c "from world import RLBenchWorld; print('Success!')"
```

#### **4. Test Run**
```bash
# Simple planning test (headless)
python run.py --task LongHorizonGrillTask --headless

# With GUI (if CoppeliaSim configured)
python run.py --task LongHorizonGrillTask
```

#### **5. Integration with Your Code**
```bash
# Option 1: Use as standalone
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/integration_dir
python run.py --task YourTask

# Option 2: Import modules in your scripts
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL
# In your Python scripts:
import sys
sys.path.append('rlbench_long_horizon/integration_dir')
from world import RLBenchWorld
from streams import get_streams
from actions import get_actions
```

---

## ⚠️ Potential Setup Challenges

### **1. CoppeliaSim Configuration**
- Needs proper installation and PATH setup
- PyRep requires CoppeliaSim libraries

### **2. COAST Library**
- May not be publicly available
- Might need to contact authors or find alternative

### **3. Fast Downward**
- Requires compilation
- Path configuration in config.py

### **4. RLBench Tasks**
- Custom tasks might need additional setup
- Task files must match structure

---

## 🎯 Next Steps Recommendations

1. **Clone the repository** to a separate directory in TAMP-PDDL
2. **Set up environment** following the steps above
3. **Run example task** to verify installation
4. **Compare architectures** between your code and collaborator's
5. **Identify reusable components** (world interface, streams, actions)
6. **Integrate incrementally** starting with world.py interface
7. **Test with your tasks** from existing PDDL files

---

## 📝 Summary

**What is this?**
A production-ready implementation of COAST TAMP algorithm integrated with RLBench simulation, featuring modular architecture and pluggable components.

**Key Strengths:**
- Clean separation of concerns
- Extensible design (easy to add tasks/planners)
- Well-documented code
- Ready for research experiments

**Connection to Paper:**
Implements the COAST foundation that VIZ-COAST builds upon. The VLM constraint generation layer would augment this implementation.

**Value for Your Work:**
- Reference architecture for TAMP systems
- Reusable components (world interface, streams)
- Best practices for PDDLStream integration
- Template for adding new tasks/domains

---

**Document Created:** January 29, 2026
**Analysis Based On:**
- Paper: "Using VLM Reasoning to Constrain Task and Motion Planning" (arXiv:2510.25548)
- Repository: mechapuru/rlbench_long_horizon (integration_dir branch)
