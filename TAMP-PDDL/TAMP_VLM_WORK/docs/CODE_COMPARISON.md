# Comparison: Your TAMP-PDDL vs Collaborator's RLBench Integration

## 📊 Overview Comparison

| Aspect | Your TAMP-PDDL | Collaborator's integration_dir |
|--------|----------------|-------------------------------|
| **Structure** | Experimental scripts | Production-ready modules |
| **Architecture** | Monolithic with baselines | Modular with clear separation |
| **Extensibility** | Manual additions | Plugin-based architecture |
| **Planning** | Direct PDDLStream calls | Abstracted planner interface |
| **Focus** | Research experimentation | Reusable framework |

---

## 🗂️ File Structure Comparison

### **Your TAMP-PDDL Directory:**
```
TAMP-PDDL/
├── baseline_0_just_PDDL.py                    # Baseline experiments
├── baseline_1_with_rules.py                   
├── orchestrator.py                            # Main orchestration
├── rlbench_kitchen_env.py                     # Environment setup
├── rlbench_kitchen_streams.py                 # Stream implementations
├── planning_rules.py                          # Manual rules
├── generalize_pick_place_gui.py               # Task-specific scripts
├── script_3_open_grasp_open_box_gui.py        
├── pddl/                                      # PDDL files
│   ├── rlbench_kitchen_domain.pddl
│   └── rlbench_kitchen_streams.pddl
├── curriculum_scripts/                        # Learning curriculum
└── grill_task2/                              # Grill task variant
```

**Characteristics:**
- Task-specific scripts
- Baseline comparisons
- Experimental iterations
- Manual rule encoding

### **Collaborator's integration_dir:**
```
integration_dir/
├── run.py                                     # CLI entry point
├── world.py                                   # Clean world interface
├── streams.py                                 # Generic streams
├── actions.py                                 # Generic actions
├── config.py                                  # Configuration management
├── planners/                                  # Pluggable planners
│   ├── base.py                               # Abstract interface
│   ├── fast_downward.py                      # FD implementation
│   └── llm_planner.py                        # LLM implementation
└── long_horizon_grill_pddl_files/            # Task-specific PDDL
    ├── domain.pddl
    ├── problem.pddl
    └── streams.pddl
```

**Characteristics:**
- Generic, reusable components
- Clear separation of concerns
- Plugin architecture
- Easy task addition

---

## 🔍 Component-by-Component Analysis

### **1. Environment/World Interface**

#### **Your Code:**
```python
# rlbench_kitchen_env.py
class RLBenchKitchenEnv:
    def __init__(self, headless=True):
        # Setup code mixed with task-specific logic
        self.env = Environment(...)
        self.task = self.env.get_task(...)
        # Direct manipulation
        
# Usage:
ENV = RLBenchKitchenEnv(headless=headless_mode)  # Global instance
```

**Pros:**
- Direct, straightforward
- Quick prototyping

**Cons:**
- Global state
- Task-specific coupling
- Hard to swap environments

#### **Collaborator's Code:**
```python
# world.py
@dataclass
class RLBenchWorld:
    """Generic RLBench interface"""
    env: Environment
    task: Task
    
    @classmethod
    def from_task_name(cls, task_name, headless=False):
        # Task-agnostic initialization
        ...
    
    def plan_to_pose(self, position, orientation):
        # Clean interface methods
        ...
```

**Pros:**
- Task-agnostic design
- Clear interface
- Easy to test/mock
- Factory pattern for flexibility

**Cons:**
- More initial setup

---

### **2. Streams Implementation**

#### **Your Code:**
```python
# rlbench_kitchen_streams.py
def fn_sample_stable_pose(o, r):
    """Direct function implementation"""
    # Task-specific logic
    ...

def fn_sample_motion(q1, q2):
    """Direct function with global ENV"""
    global _motion_error_count
    path = ENV.plan_trajectory(q1, q2)
    ...

# Stream map
stream_map = {
    's_sample_stable_pose': from_gen_fn(fn_sample_stable_pose),
    's_sample_motion': from_gen_fn(fn_sample_motion),
}
```

**Pros:**
- Simple, direct
- Easy to debug

**Cons:**
- Global state (_motion_error_count, ENV)
- Function-based (harder to extend)
- Mixed concerns

#### **Collaborator's Code:**
```python
# streams.py
class SampleMotion(Stream):
    """Object-oriented stream"""
    name = "sample-motion"
    inputs = [Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?t", "traj")]
    
    def __init__(self, world):
        self.world = world  # No global state
    
    def sample(self, inputs, fluents):
        """Clean interface"""
        q1 = inputs["?q1"]
        q2 = inputs["?q2"]
        path = self.world.plan_to_pose(...)
        return {"?t": Trajectory(path)}

# Factory function
def get_streams(world):
    return [
        SampleGrasp(world),
        SampleMotion(world),
        ...
    ]
```

**Pros:**
- No global state
- OOP design (easy to extend)
- Clear dependencies
- Type hints
- Reusable across tasks

**Cons:**
- More boilerplate

---

### **3. Actions Implementation**

#### **Your Code:**
```python
# Mixed in with orchestrator.py and task scripts
def execute_trajectory(env, traj):
    """Function-based execution"""
    for waypoint in traj:
        env.robot.arm.set_joint_positions(waypoint)
        env.step()

# Inline in orchestrator
for action in plan:
    if action.name == "pick":
        execute_trajectory(ENV, action.traj)
        ENV.close_gripper()
    elif action.name == "place":
        execute_trajectory(ENV, action.traj)
        ENV.open_gripper()
```

**Pros:**
- Straightforward
- Direct control

**Cons:**
- Scattered logic
- Hard to maintain
- No clear interface

#### **Collaborator's Code:**
```python
# actions.py
class Pick(Action):
    """Object-oriented action"""
    def __init__(self, world):
        self.world = world
    
    def execute(self, inputs, outputs):
        """Standardized interface"""
        try:
            traj = outputs.get("t")
            # Execute trajectory
            for waypoint in traj.waypoints:
                self.world.set_robot_config(waypoint)
                self.world.step()
            # Grasp
            self.world.close_gripper()
            return True
        except Exception as e:
            print(f"[Pick] Failed: {e}")
            return False

# Factory function
def get_actions(world):
    return [Pick(world), Place(world), ...]
```

**Pros:**
- Encapsulated logic
- Error handling
- Clear success/failure
- Easy to add actions
- Testable

**Cons:**
- More code upfront

---

### **4. Configuration Management**

#### **Your Code:**
```python
# Scattered across files
headless_mode = os.environ.get("HEADLESS", "True") == "True"
ENV = RLBenchKitchenEnv(headless=headless_mode)

# Hard-coded paths
domain_pddl = "pddl/rlbench_kitchen_domain.pddl"
streams_pddl = "pddl/rlbench_kitchen_streams.pddl"
```

**Pros:**
- Simple
- No extra files

**Cons:**
- Hard to change
- No validation
- Scattered configuration

#### **Collaborator's Code:**
```python
# config.py
TASK_PDDL_FOLDERS = {
    "LongHorizonGrillTask": "long_horizon_grill_pddl_files",
    # Easy to add new tasks
}

@dataclass
class CoastConfig:
    planner: str = "fast_downward"
    planner_timeout: int = 1200
    task_name: str = "LongHorizonGrillTask"
    headless: bool = False
    
    def __post_init__(self):
        # Auto-set paths based on task
        pddl_folder = TASK_PDDL_FOLDERS[self.task_name]
        self.domain_pddl = f"{pddl_folder}/domain.pddl"
        ...
```

**Pros:**
- Centralized configuration
- Type checking
- Auto-path resolution
- Easy to extend
- Validation

**Cons:**
- Extra file

---

### **5. Planner Integration**

#### **Your Code:**
```python
# Direct PDDLStream calls
from pddlstream.algorithms.meta import solve

solution = solve(
    problem,
    algorithm='incremental',
    unit_costs=True,
    planner='ff-wastar1',
    # Hard-coded parameters
)
```

**Pros:**
- Direct control
- Simple

**Cons:**
- Hard to swap planners
- No abstraction
- Tight coupling

#### **Collaborator's Code:**
```python
# planners/base.py
class TaskPlanner(ABC):
    @abstractmethod
    def plan(self, domain_pddl, problem_state, pddl):
        """Abstract interface"""
        pass

# planners/fast_downward.py
class FastDownwardPlanner(TaskPlanner):
    def plan(self, ...):
        # FD-specific implementation
        ...

# planners/llm_planner.py
class LLMPlanner(TaskPlanner):
    def plan(self, ...):
        # LLM-specific implementation
        ...

# Usage in run.py
planner = create_planner(config)  # Factory
result = planner.plan(...)  # Polymorphic call
```

**Pros:**
- Strategy pattern
- Easy to swap planners
- Consistent interface
- Testable
- Extensible

**Cons:**
- More abstraction layers

---

### **6. Main Entry Point**

#### **Your Code:**
```python
# Multiple scripts:
# - orchestrator.py
# - baseline_0_just_PDDL.py
# - baseline_1_with_rules.py
# - generalize_pick_place_gui.py
# - etc.

# Each script has its own main()
def main():
    # Inline setup
    env = RLBenchKitchenEnv(headless=False)
    # Inline planning
    solution = solve(...)
    # Inline execution
    for action in plan:
        execute_trajectory(env, action)
```

**Pros:**
- Self-contained experiments
- Easy to understand each script

**Cons:**
- Code duplication
- Hard to maintain consistency
- No unified interface

#### **Collaborator's Code:**
```python
# run.py - Single entry point
def main():
    # Parse CLI arguments
    args = parser.parse_args()
    
    # Create config
    config = CoastConfig(
        task_name=args.task,
        planner=args.planner,
        ...
    )
    
    # Initialize world
    world = RLBenchWorld.from_task_name(config.task_name)
    
    # Create planner
    planner = create_planner(config)
    
    # Run planning
    result = run_coast_planning(world, config, planner)
    
    # Execute (if requested)
    if args.execute:
        execute_plan(world, result['plan'])

# Clean CLI interface
python run.py --task MyTask --planner fast_downward --execute
```

**Pros:**
- Single entry point
- Consistent interface
- Easy to use
- Command-line friendly
- No code duplication

**Cons:**
- Less flexibility for experiments

---

## 🎯 Key Architectural Differences

### **1. Design Pattern Usage**

| Pattern | Your Code | Collaborator's Code |
|---------|-----------|-------------------|
| **Factory** | ❌ Not used | ✅ `from_task_name()`, `create_planner()` |
| **Strategy** | ❌ Not used | ✅ Pluggable planners |
| **Dataclass** | ❌ Not used | ✅ Extensive use |
| **OOP** | ⚠️ Minimal | ✅ Heavy use |
| **Global State** | ⚠️ Common (ENV) | ✅ Avoided |

### **2. Code Organization**

| Aspect | Your Code | Collaborator's Code |
|--------|-----------|-------------------|
| **Files** | Many task-specific scripts | Few generic modules |
| **Lines per file** | Variable | Consistent |
| **Coupling** | High (shared globals) | Low (dependency injection) |
| **Cohesion** | Mixed concerns | Single responsibility |

### **3. Extensibility**

| Task | Your Code | Collaborator's Code |
|------|-----------|-------------------|
| **Add new task** | Create new script, copy-paste setup | Add PDDL folder, register in config |
| **Add new planner** | Modify orchestrator | Implement `TaskPlanner` interface |
| **Add new stream** | Add function, update stream_map | Create class, add to `get_streams()` |
| **Add new action** | Modify orchestrator | Create class, add to `get_actions()` |

---

## 🔄 Integration Recommendations

### **Option 1: Incremental Adoption**
Gradually adopt collaborator's patterns in your code:

1. **Phase 1: World Interface**
   - Extract `RLBenchWorld` pattern
   - Remove global `ENV` variable
   - Use dependency injection

2. **Phase 2: Streams as Classes**
   - Convert functions to classes
   - Add factory function
   - Pass world as dependency

3. **Phase 3: Actions as Classes**
   - Encapsulate execution logic
   - Standardize interface
   - Add error handling

4. **Phase 4: Configuration**
   - Create `config.py`
   - Centralize settings
   - Auto-resolve paths

5. **Phase 5: Planner Abstraction**
   - Create `TaskPlanner` interface
   - Wrap existing planners
   - Enable swapping

### **Option 2: Side-by-Side**
Keep your experimental code, use collaborator's for production:

```
TAMP-PDDL/
├── [your experimental scripts]  # Keep for research
├── production/                  # Adapted from collaborator
│   ├── world.py
│   ├── streams.py
│   ├── actions.py
│   ├── run.py
│   └── config.py
└── rlbench_long_horizon/        # Cloned repo for reference
```

### **Option 3: Full Migration**
Restructure your code to match collaborator's architecture:

1. Create module structure
2. Migrate streams to classes
3. Migrate actions to classes
4. Create unified entry point
5. Add configuration management
6. Port existing tasks to new structure

---

## 📈 Pros & Cons of Each Approach

### **Your Current Code**

**✅ Pros:**
- Easy to experiment
- Quick prototyping
- Self-contained scripts
- Direct control

**❌ Cons:**
- Hard to maintain
- Code duplication
- Tight coupling
- Global state issues
- Difficult to extend

### **Collaborator's Code**

**✅ Pros:**
- Production-ready
- Easy to extend
- Clean architecture
- No code duplication
- Testable
- Reusable

**❌ Cons:**
- More initial setup
- Steeper learning curve
- More abstraction

---

## 💡 Specific Improvements You Can Adopt

### **1. World Interface**
```python
# Instead of:
ENV = RLBenchKitchenEnv(headless=True)  # Global

# Do:
world = RLBenchWorld.from_task_name("MyTask", headless=True)
# Pass world to functions/classes
```

### **2. Stream Classes**
```python
# Instead of:
def fn_sample_motion(q1, q2):
    global ENV
    return ENV.plan(q1, q2)

# Do:
class SampleMotion(Stream):
    def __init__(self, world):
        self.world = world
    def sample(self, inputs, fluents):
        return self.world.plan(inputs["?q1"], inputs["?q2"])
```

### **3. Configuration**
```python
# Instead of:
domain_pddl = "pddl/domain.pddl"  # Hard-coded

# Do:
@dataclass
class Config:
    task_name: str = "MyTask"
    def __post_init__(self):
        self.domain_pddl = f"pddl/{self.task_name}_domain.pddl"
```

### **4. Entry Point**
```python
# Instead of multiple scripts, create:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    
    world = create_world(args.task)
    result = plan(world, args)
    if args.execute:
        execute(world, result)
```

---

## 🚀 Recommended Next Steps

1. **Study** collaborator's code structure
2. **Clone** the repository using the setup script
3. **Run** example tasks to see it in action
4. **Identify** reusable components for your work
5. **Experiment** with adding your tasks to their framework
6. **Gradually adopt** design patterns in your code
7. **Maintain** your experimental scripts for research
8. **Use** clean architecture for production code

---

## 📝 Summary

**Your Code:**
- Great for rapid experimentation
- Good for research exploration
- Needs refactoring for production

**Collaborator's Code:**
- Production-ready architecture
- Easy to extend and maintain
- Best practices demonstrated
- Reference implementation for clean TAMP systems

**Best Approach:**
- Keep your experimental code
- Learn from collaborator's architecture
- Gradually adopt patterns
- Use collaborator's structure for new production code

---

**Document Purpose:** Help you understand the architectural differences and make informed decisions about integrating best practices into your work.
