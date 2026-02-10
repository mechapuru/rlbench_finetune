#!/bin/bash

# Setup script for cloning and configuring collaborator's RLBench Long Horizon repository
# Location: /home/naren/iiith/Long_Horizon/TAMP-PDDL

set -e  # Exit on error

echo "========================================================"
echo "RLBench Long Horizon Setup Script"
echo "========================================================"
echo ""

# Define base directory
BASE_DIR="/home/naren/iiith/Long_Horizon/TAMP-PDDL"
REPO_NAME="rlbench_long_horizon"
REPO_URL="https://github.com/mechapuru/rlbench_long_horizon.git"

# Navigate to base directory
cd "$BASE_DIR"
echo "Working directory: $(pwd)"
echo ""

# Step 1: Clone repository
echo "Step 1: Cloning repository..."
if [ -d "$REPO_NAME" ]; then
    echo "  ⚠️  Directory $REPO_NAME already exists."
    read -p "  Do you want to remove it and re-clone? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$REPO_NAME"
        git clone "$REPO_URL"
        echo "  ✅ Repository cloned successfully"
    else
        echo "  ⏭️  Skipping clone, using existing directory"
    fi
else
    git clone "$REPO_URL"
    echo "  ✅ Repository cloned successfully"
fi
echo ""

# Step 2: Check Python environment
echo "Step 2: Checking Python environment..."
if command -v conda &> /dev/null; then
    echo "  ✅ Conda is installed"
    echo "  Current environment: $CONDA_DEFAULT_ENV"
    
    # Check if rlbench environment exists
    if conda env list | grep -q "rlbench"; then
        echo "  ✅ 'rlbench' conda environment exists"
        echo "  To activate: conda activate rlbench"
    else
        echo "  ⚠️  'rlbench' conda environment not found"
        read -p "  Do you want to create it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda create -n rlbench python=3.8 -y
            echo "  ✅ Created 'rlbench' environment"
            echo "  To activate: conda activate rlbench"
        fi
    fi
else
    echo "  ⚠️  Conda not found. Using system Python: $(which python3)"
fi
echo ""

# Step 3: Display installation instructions
echo "Step 3: Installation Instructions"
echo "========================================================"
echo ""
echo "To complete the setup, run the following commands:"
echo ""
echo "# 1. Activate the environment (if using conda)"
echo "conda activate rlbench"
echo ""
echo "# 2. Navigate to the repository"
echo "cd $BASE_DIR/$REPO_NAME"
echo ""
echo "# 3. Install RLBench (if available in repo)"
echo "if [ -d 'RLBench' ]; then"
echo "    pip install -e RLBench/"
echo "fi"
echo ""
echo "# 4. Install PyRep (requires CoppeliaSim)"
echo "# First download CoppeliaSim from: https://www.coppeliarobotics.com/downloads"
echo "# Then:"
echo "pip install pyrep"
echo ""
echo "# 5. Install COAST (if available)"
echo "# Check if coast directory exists or install from PyPI"
echo "pip install coast-tamp  # or pip install -e ../coast"
echo ""
echo "# 6. Install additional dependencies"
echo "pip install numpy scipy"
echo ""
echo "# 7. Test the installation"
echo "cd integration_dir"
echo "python -c 'from world import RLBenchWorld; print(\"✅ Import successful!\")'"
echo ""
echo "# 8. Run a test (headless mode)"
echo "python run.py --task LongHorizonGrillTask --headless --timeout 60"
echo ""
echo "========================================================"
echo ""

# Step 4: Create a quick reference file
echo "Step 4: Creating quick reference..."
cat > "$BASE_DIR/COLLABORATOR_REPO_REFERENCE.txt" << 'EOF'
===============================================================================
RLBench Long Horizon - Quick Reference
===============================================================================

Repository Location:
  /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/

Main Implementation:
  integration_dir/

Key Files:
  - run.py         : Main entry point
  - world.py       : RLBench/PyRep interface
  - streams.py     : Geometric samplers
  - actions.py     : Action implementations
  - config.py      : Configuration
  - planners/      : Pluggable task planners

===============================================================================
Usage Examples
===============================================================================

1. Planning Only (Headless):
   cd integration_dir
   python run.py --task LongHorizonGrillTask --headless

2. Planning + Execution (with GUI):
   python run.py --task LongHorizonGrillTask --execute

3. With Different Planner:
   python run.py --task LongHorizonGrillTask --planner fast_downward

4. Custom Timeout:
   python run.py --task LongHorizonGrillTask --timeout 600

===============================================================================
Adding a New Task
===============================================================================

1. Create PDDL folder: my_task_pddl_files/
2. Add: domain.pddl, problem.pddl, streams.pddl
3. Register in config.py:
   TASK_PDDL_FOLDERS["MyTask"] = "my_task_pddl_files"
4. Run: python run.py --task MyTask

===============================================================================
Integration with Your Code
===============================================================================

Option 1 - Standalone:
  cd /home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/integration_dir
  python run.py --task YourTask

Option 2 - Import Modules:
  import sys
  sys.path.append('/home/naren/iiith/Long_Horizon/TAMP-PDDL/rlbench_long_horizon/integration_dir')
  from world import RLBenchWorld
  from streams import get_streams
  from actions import get_actions

===============================================================================
Dependencies
===============================================================================

Core:
  - RLBench         : Robotics simulation
  - PyRep           : CoppeliaSim interface
  - COAST           : TAMP library
  - NumPy           : Numerical computing
  - Fast Downward   : Task planner

Optional:
  - OpenAI API      : For LLM planner
  - SciPy           : Additional math functions

===============================================================================
Troubleshooting
===============================================================================

1. Import Errors:
   - Check PYTHONPATH includes integration_dir
   - Verify all dependencies installed

2. CoppeliaSim Issues:
   - Ensure CoppeliaSim installed and in PATH
   - Check PyRep can find CoppeliaSim libraries

3. COAST Not Found:
   - May need to install separately
   - Check for coast/ directory in repo
   - Try: pip install coast-tamp

4. Planning Timeouts:
   - Increase --timeout parameter
   - Simplify problem or reduce --max-level

===============================================================================
Resources
===============================================================================

Paper: "Using VLM Reasoning to Constrain Task and Motion Planning"
  arXiv: 2510.25548
  Location: /home/naren/iiith/Long_Horizon/TAMP-PDDL/2510.25548v1.pdf

Analysis Document:
  /home/naren/iiith/Long_Horizon/TAMP-PDDL/VIZ-COAST_ANALYSIS.md

GitHub:
  https://github.com/mechapuru/rlbench_long_horizon

===============================================================================
EOF

echo "  ✅ Created COLLABORATOR_REPO_REFERENCE.txt"
echo ""

# Step 5: Summary
echo "========================================================"
echo "Setup Complete!"
echo "========================================================"
echo ""
echo "Next steps:"
echo "  1. Read: $BASE_DIR/VIZ-COAST_ANALYSIS.md"
echo "  2. Reference: $BASE_DIR/COLLABORATOR_REPO_REFERENCE.txt"
echo "  3. Follow installation instructions above"
echo "  4. Test the integration_dir code"
echo ""
echo "Repository cloned to:"
echo "  $BASE_DIR/$REPO_NAME/"
echo ""
echo "Happy coding! 🚀"
echo ""
