# Action Plan: Integrating Collaborator's Repository

## 🎯 Executive Summary

This document provides a concrete, step-by-step action plan for cloning, understanding, and integrating your collaborator's RLBench long-horizon repository into your TAMP-PDDL workspace.

**Estimated Time:** 2-4 hours (depending on dependency installation)

**Prerequisites:**
- ✅ Linux system (you have this)
- ✅ Conda/Python environment
- ⚠️ CoppeliaSim (may need installation)
- ⚠️ COAST library (may need installation)

---

## 📅 Phase 1: Initial Setup (30-60 minutes)

### **Step 1.1: Clone the Repository**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL
bash setup_collaborator_repo.sh
```

This script will:
- Clone the repository
- Check your Python environment
- Create reference files
- Display installation instructions

**Expected Output:**
```
✅ Repository cloned successfully
✅ Created COLLABORATOR_REPO_REFERENCE.txt
✅ Created VIZ-COAST_ANALYSIS.md
```

### **Step 1.2: Verify Clone**
```bash
ls -la rlbench_long_horizon/
```

**Expected Structure:**
```
rlbench_long_horizon/
├── integration_dir/          ← Main focus
├── pddlstream execution/
├── examples/
├── RLBench/
└── README.md
```

### **Step 1.3: Navigate to Integration Directory**
```bash
cd rlbench_long_horizon/integration_dir
ls -la
```

**Expected Files:**
```
run.py
world.py
streams.py
actions.py
config.py
planners/
long_horizon_grill_pddl_files/
README.md
```

**✅ Checkpoint:** If you see these files, Phase 1 is complete!

---

## 📅 Phase 2: Environment Setup (30-90 minutes)

### **Step 2.1: Create/Activate Conda Environment**
```bash
# If you don't have rlbench environment:
conda create -n rlbench python=3.8 -y

# Activate
conda activate rlbench

# Verify
python --version  # Should show Python 3.8.x
which python      # Should point to conda env
```

### **Step 2.2: Install Core Dependencies**

#### **Install NumPy and SciPy:**
```bash
pip install numpy scipy
```

#### **Install RLBench (if available in repo):**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon

# Check if RLBench directory exists
if [ -d "RLBench" ]; then
    echo "RLBench found, installing..."
    pip install -e RLBench/
else
    echo "RLBench not in repo, install separately:"
    echo "pip install git+https://github.com/stepjam/RLBench.git"
fi
```

#### **Install PyRep:**

**⚠️ Important:** PyRep requires CoppeliaSim to be installed first!

**Option A: If CoppeliaSim Already Installed**
```bash
pip install pyrep
```

**Option B: If CoppeliaSim Not Installed**
1. Download CoppeliaSim:
   ```bash
   # Go to: https://www.coppeliarobotics.com/downloads
   # Download the appropriate version for Linux
   # Or use wget (check latest version):
   wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
   ```

2. Extract and setup:
   ```bash
   tar -xf CoppeliaSim_*.tar.xz
   mv CoppeliaSim_* ~/CoppeliaSim
   export COPPELIASIM_ROOT=~/CoppeliaSim
   echo 'export COPPELIASIM_ROOT=~/CoppeliaSim' >> ~/.bashrc
   ```

3. Install PyRep:
   ```bash
   pip install pyrep
   ```

**Option C: Skip for Now (Headless Testing)**
```bash
# You can test PDDL/planning logic without CoppeliaSim
# Just won't be able to execute in simulation
echo "Skipping PyRep for now, will test planning only"
```

### **Step 2.3: Install/Locate COAST**

**Check if COAST is in repo:**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon
find . -name "coast" -type d
```

**Option A: If Found in Repo**
```bash
cd coast  # or wherever it's located
pip install -e .
```

**Option B: If Not Found - Try PyPI**
```bash
pip install coast-tamp
# or
pip install coast
```

**Option C: If Not Available**
```bash
# May need to contact authors or find alternative
# For now, note this as a blocker and proceed with analysis
echo "COAST not available - will focus on code structure analysis"
```

### **Step 2.4: Install Fast Downward**

**Check if included in repo:**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon
find . -name "fast-downward.py" -type f
```

**If found:**
```bash
# Note the path, e.g.:
# ./pddlstream/downward/fast-downward.py
# Update config.py if needed
```

**If not found:**
```bash
# Clone Fast Downward
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL
git clone https://github.com/aibasel/downward.git fast-downward
cd fast-downward
./build.py
```

**✅ Checkpoint:** Run `pip list` and verify installed packages.

---

## 📅 Phase 3: Code Analysis (60-90 minutes)

### **Step 3.1: Read Documentation**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL

# Read the analysis document
less VIZ-COAST_ANALYSIS.md
# Press 'q' to quit

# Read the comparison document
less CODE_COMPARISON.md

# Read the quick reference
less COLLABORATOR_REPO_REFERENCE.txt
```

### **Step 3.2: Explore Code Structure**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/integration_dir

# View file structure
tree -L 2

# Or if tree not installed:
find . -maxdepth 2 -type f -name "*.py" | sort
```

### **Step 3.3: Examine Key Files**

#### **Read world.py**
```bash
less world.py
# Focus on:
# - RLBenchWorld class
# - from_task_name() method
# - plan_to_pose() method
# - get_coast_objects() method
```

#### **Read streams.py**
```bash
less streams.py
# Focus on:
# - Stream base class usage
# - SampleMotion class
# - SampleGrasp class
# - get_streams() factory function
```

#### **Read actions.py**
```bash
less actions.py
# Focus on:
# - Action base class usage
# - Pick class
# - Place class
# - get_actions() factory function
```

#### **Read config.py**
```bash
less config.py
# Focus on:
# - TASK_PDDL_FOLDERS mapping
# - CoastConfig dataclass
# - __post_init__ logic
```

#### **Read run.py**
```bash
less run.py
# Focus on:
# - main() function
# - run_coast_planning() function
# - execute_plan() function
# - CLI argument parsing
```

### **Step 3.4: Examine PDDL Files**
```bash
cd long_horizon_grill_pddl_files

# Domain definition
less domain.pddl
# Look for: predicates, actions

# Problem definition
less problem.pddl
# Look for: objects, init, goal

# Streams declaration
less streams.pddl
# Look for: stream declarations
```

**✅ Checkpoint:** You should understand:
- How RLBenchWorld provides simulation interface
- How Streams generate geometric parameters
- How Actions execute plans
- How Config manages settings
- How everything ties together in run.py

---

## 📅 Phase 4: Test Run (30-60 minutes)

### **Step 4.1: Dry Run (No Dependencies)**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/integration_dir

# Check if imports work
python -c "import sys; print(sys.version)"
python -c "import numpy; print('NumPy OK')"

# Try importing modules (may fail if dependencies missing)
python -c "from config import CoastConfig; print('Config OK')" || echo "Config import failed"
```

### **Step 4.2: Test Without Simulation (if COAST not available)**
```bash
# Create a minimal test script
cat > test_structure.py << 'EOF'
"""Test script to verify code structure without running COAST"""
import sys
from pathlib import Path

print("Testing imports...")

try:
    from config import CoastConfig, TASK_PDDL_FOLDERS
    print("✅ Config imported")
    print(f"   Available tasks: {list(TASK_PDDL_FOLDERS.keys())}")
except Exception as e:
    print(f"❌ Config import failed: {e}")

try:
    # Note: This will fail without RLBench/PyRep installed
    from world import RLBenchWorld
    print("✅ World imported")
except Exception as e:
    print(f"⚠️  World import failed (expected if RLBench not installed): {e}")

try:
    from streams import get_streams
    print("✅ Streams module imported")
except Exception as e:
    print(f"❌ Streams import failed: {e}")

try:
    from actions import get_actions
    print("✅ Actions module imported")
except Exception as e:
    print(f"❌ Actions import failed: {e}")

print("\nStructure test complete!")
EOF

python test_structure.py
```

### **Step 4.3: Full Test Run (if all dependencies available)**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/integration_dir

# Test with short timeout, headless mode
python run.py --task LongHorizonGrillTask --headless --timeout 60

# If successful, try with GUI
python run.py --task LongHorizonGrillTask --timeout 120

# If successful, try with execution
python run.py --task LongHorizonGrillTask --execute
```

**Expected Output (if successful):**
```
[Init] Loading RLBench environment...
[Init] Loaded task: LongHorizonGrillTask
[Init] Objects: ['chicken', 'plate', 'grill', ...]
[COAST] Starting planning...
[COAST] Planning completed in X.XXs
[COAST] Plan found with N actions:
  1. pick(chicken, grill_side, ...)
  2. place(chicken, grill_cooking_area, ...)
  ...
```

**✅ Checkpoint:** If you see planning output or clear error messages (missing dependencies), you're on track!

---

## 📅 Phase 5: Integration Planning (30 minutes)

### **Step 5.1: Compare with Your Code**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL

# Open both for comparison
code rlbench_long_horizon/integration_dir/streams.py &  # Or your editor
code rlbench_kitchen_streams.py &
```

### **Step 5.2: Identify Reusable Components**

**Create a checklist:**
```bash
cat > integration_checklist.txt << 'EOF'
Integration Checklist
=====================

Components to Adopt:
[ ] World interface pattern (world.py)
    - Replace global ENV variable
    - Use dependency injection
    
[ ] Stream classes (streams.py)
    - Convert functions to classes
    - Remove global state
    - Add type hints
    
[ ] Action classes (actions.py)
    - Encapsulate execution logic
    - Standardize interface
    
[ ] Configuration management (config.py)
    - Centralize settings
    - Task-to-PDDL mapping
    
[ ] Planner abstraction (planners/)
    - Abstract interface
    - Easy swapping
    
[ ] CLI interface (run.py)
    - Unified entry point
    - Argument parsing

Next Steps:
[ ] Decide: Full migration vs. gradual adoption vs. side-by-side
[ ] Create integration branch in your code
[ ] Port one task as proof-of-concept
[ ] Test and iterate

Notes:
- Start with world interface (biggest impact)
- Then streams (remove global state)
- Then actions (encapsulation)
- Config and CLI last (quality of life)
EOF

less integration_checklist.txt
```

### **Step 5.3: Document Findings**
```bash
# Create personal notes
cat > INTEGRATION_NOTES.md << 'EOF'
# Integration Notes

## What I Learned:
- [ Key insights from collaborator's code ]

## What I Want to Adopt:
- [ Specific patterns/components ]

## What I'll Keep from My Code:
- [ Experimental features, baselines, etc. ]

## Potential Challenges:
- [ Dependencies, refactoring effort, etc. ]

## Action Items:
1. [ First concrete step ]
2. [ Second concrete step ]
3. [ ... ]

## Questions to Resolve:
- [ Any unclear aspects ]
EOF

# Edit with your actual notes
nano INTEGRATION_NOTES.md  # or vim/code
```

**✅ Checkpoint:** You should have a clear plan for integration!

---

## 📅 Phase 6: Next Actions (Ongoing)

### **Option A: Gradual Adoption (Recommended)**

**Week 1: World Interface**
```bash
# In your code, create new world.py
cp rlbench_long_horizon/integration_dir/world.py ./world_refactored.py

# Adapt to your tasks
# Replace global ENV in one script
# Test thoroughly

# Commit progress
git add world_refactored.py
git commit -m "feat: add refactored world interface"
```

**Week 2: Streams Refactor**
```bash
# Convert one stream function to class
# Test with your existing orchestrator
# Gradually convert all streams
```

**Week 3: Actions Refactor**
```bash
# Encapsulate action execution
# Standardize interface
# Add error handling
```

**Week 4: Integration**
```bash
# Create unified entry point
# Add configuration management
# Full integration test
```

### **Option B: Side-by-Side (Quick Start)**

**Immediate:**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL

# Create production directory
mkdir -p production

# Copy collaborator's structure
cp -r rlbench_long_horizon/integration_dir/* production/

# Adapt for your tasks
cd production
# Edit config.py to add your tasks
# Create your task PDDL files
# Test

# Keep your experimental code separate
# Use production/ for clean implementations
```

### **Option C: Full Migration (Long-term Project)**

**Month 1: Planning**
- Detailed architecture design
- Migration roadmap
- Test strategy

**Month 2: Implementation**
- Refactor all components
- Migrate all tasks
- Comprehensive testing

**Month 3: Validation**
- Performance comparison
- Regression testing
- Documentation

---

## 🎓 Learning Resources

### **To Understand COAST:**
1. Read VIZ-COAST paper (in your directory)
2. Explore pddlstream examples in repo
3. Study streams.pddl syntax

### **To Understand RLBench:**
1. Official docs: https://github.com/stepjam/RLBench
2. Examples in rlbench_long_horizon/examples/
3. Your existing task implementations

### **To Understand Fast Downward:**
1. Official docs: https://www.fast-downward.org/
2. PDDL tutorials
3. Planner comparison papers

---

## 🚨 Troubleshooting Guide

### **Issue: Can't Clone Repository**
```bash
# Check network
ping github.com

# Try HTTPS vs SSH
git clone https://github.com/mechapuru/rlbench_long_horizon.git
# vs
git clone git@github.com:mechapuru/rlbench_long_horizon.git
```

### **Issue: Import Errors**
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Add integration_dir to path
export PYTHONPATH="/home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/integration_dir:$PYTHONPATH"

# Or in Python script:
import sys
sys.path.insert(0, '/path/to/integration_dir')
```

### **Issue: CoppeliaSim Not Found**
```bash
# Set environment variable
export COPPELIASIM_ROOT=/path/to/CoppeliaSim

# Add to bashrc for persistence
echo 'export COPPELIASIM_ROOT=/path/to/CoppeliaSim' >> ~/.bashrc
source ~/.bashrc
```

### **Issue: COAST Not Available**
```bash
# Try different installation methods
pip install coast-tamp
pip install coast
pip install git+https://github.com/coast-lab/coast.git  # if public

# If not available:
# 1. Focus on code structure analysis
# 2. Contact authors for access
# 3. Implement compatible interface yourself
```

### **Issue: Planning Timeouts**
```bash
# Increase timeout
python run.py --task MyTask --timeout 3600  # 1 hour

# Reduce problem complexity
# Edit problem.pddl to simplify

# Reduce max level
python run.py --task MyTask --max-level 3
```

---

## ✅ Success Criteria

By the end of this action plan, you should have:

- [x] ✅ Cloned repository successfully
- [x] ✅ Environment set up (conda/pip)
- [x] ✅ Dependencies installed (or blockers identified)
- [x] ✅ Code structure understood
- [x] ✅ PDDL files examined
- [x] ✅ Test run attempted
- [x] ✅ Integration plan created
- [x] ✅ Personal notes documented

**Optional (dependency-dependent):**
- [ ] ⚠️ Full system running with simulation
- [ ] ⚠️ New task added to framework
- [ ] ⚠️ Custom planner integrated

---

## 📞 Getting Help

**If you encounter issues:**

1. **Check Documentation:**
   - VIZ-COAST_ANALYSIS.md (in your directory)
   - CODE_COMPARISON.md (in your directory)
   - COLLABORATOR_REPO_REFERENCE.txt (in your directory)

2. **Repository Issues:**
   - https://github.com/mechapuru/rlbench_long_horizon/issues

3. **Dependencies:**
   - RLBench: https://github.com/stepjam/RLBench/issues
   - PyRep: https://github.com/stepjam/PyRep/issues
   - Fast Downward: https://www.fast-downward.org/

4. **Paper Authors:**
   - Contact info in paper (2510.25548v1.pdf)

---

## 🎯 Summary

**Time Investment:**
- Setup: 1-2 hours
- Analysis: 2-3 hours
- Integration: Ongoing (depending on approach)

**Expected Outcomes:**
- Understanding of production-grade TAMP architecture
- Reusable components for your work
- Best practices reference
- Cleaner, more maintainable code

**Next Immediate Step:**
```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL
bash setup_collaborator_repo.sh
```

**Good luck! 🚀**

---

**Document Created:** January 29, 2026
**Author:** GitHub Copilot
**Purpose:** Provide concrete, actionable steps for repository integration
