"""
Test script for spam pick and place to cupboard_boundary
"""
import os
import sys

def _configure_qt():
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    candidate_dirs = [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
        os.path.join(coppelia_root, "qt", "plugins", "platforms"),
    ]
    for candidate in candidate_dirs:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break

_configure_qt()

sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

os.environ["HEADLESS"] = "False"

from generalize_pick_place_gui import run_task

if __name__ == "__main__":
    print("Testing: spam -> cupboard_boundary")
    run_task('spam', 'cupboard_boundary', close_on_finish=True)
