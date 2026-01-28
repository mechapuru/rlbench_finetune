import sys
import os
from pathlib import Path

# Mimic run.py path setup
sys.path.insert(0, str(Path(__file__).parent.parent))
# PROPOSED FIX: Add coast/symbolic to path
sys.path.insert(0, str(Path(__file__).parent.parent / "coast/symbolic"))

# Print sys.path to be sure
print("sys.path:", sys.path)

try:
    import symbolic
    print(f"Symbolic imported from: {getattr(symbolic, '__file__', 'no file')}")
    print(f"Dir(symbolic): {dir(symbolic)}")
    
    # Check for expected attributes (user mentioned 'attributes')
    # From __init__.py we saw 'Problem', '_P', etc.
    if 'Problem' in dir(symbolic):
        print("SUCCESS: 'Problem' found in symbolic.")
    else:
        print("FAILURE: 'Problem' NOT found in symbolic.")
        
except ImportError as e:
    print(f"ImportError importing symbolic: {e}")

# Also try importing via coast just in case
try:
    import coast.symbolic
    print(f"Coast.Symbolic imported from: {getattr(coast.symbolic, '__file__', 'no file')}")
except ImportError as e:
    print(f"ImportError importing coast.symbolic: {e}")
