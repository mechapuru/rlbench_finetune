"""
Skeleton to PDDL Translator (Module 2.5)
========================================
Translates VLM action skeletons into PDDLStream problems
that can be solved and executed.
"""

import os
import sys
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import VLM planner types
from vlm_pipeline.vlm_planner import ActionSkeleton


@dataclass
class PDDLAtom:
    """Represents a PDDL atom (predicate with arguments)."""
    predicate: str
    args: Tuple[str, ...]
    
    def __str__(self):
        if not self.args:
            return f"({self.predicate})"
        return f"({self.predicate} {' '.join(self.args)})"
    
    def to_tuple(self):
        """Convert to PDDLStream tuple format."""
        return (self.predicate,) + self.args


class SkeletonToPDDL:
    """
    Translates VLM action skeletons into PDDL problem specifications
    compatible with PDDLStream.
    """
    
    # Map skeleton action names to PDDL action names
    ACTION_MAP = {
        'pick': 'pick',
        'place': 'place',
        'open-lid': 'open-lid',
        'open_lid': 'open-lid',
        'move': 'move',
        'retreat': 'retreat',
    }
    
    # Object name mappings (VLM name -> PDDL/Scene name)
    OBJECT_MAP = {
        'mug_box': 'mug_box',
        'mug_inside_box': 'mug_inside_box',
        'mug_table': 'mug_table',
        'mug_cupboard': 'mug_cupboard',
        'soup': 'soup',
        'mustard': 'mustard',
        'spam': 'spam',
        'sugar': 'sugar',
        'crackers': 'crackers',
        'box_lid': 'box_lid',
    }
    
    # Region name mappings
    REGION_MAP = {
        'placement_boundary': 'placement_boundary',
        'cupboard_boundary': 'cupboard_boundary',
        'groceries_boundary': 'groceries_boundary',
        'table': 'table',
        'box-top': 'box-top',
        'box-inside': 'box-inside',
    }
    
    def __init__(self, 
                 domain_path: str = None,
                 stream_path: str = None):
        """
        Initialize the translator.
        
        Args:
            domain_path: Path to PDDL domain file
            stream_path: Path to PDDL stream file
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.domain_path = domain_path or os.path.join(
            base_dir, 'pddl', 'rlbench_kitchen_domain_constrained.pddl'
        )
        self.stream_path = stream_path or os.path.join(
            base_dir, 'pddl', 'rlbench_kitchen_streams.pddl'
        )
        
    def skeleton_to_goal_sequence(self, 
                                   skeleton: List[ActionSkeleton]) -> List[PDDLAtom]:
        """
        Convert action skeleton to a sequence of goal states.
        
        Each action in the skeleton implies certain goal predicates:
        - pick(obj) -> (holding obj)
        - place(obj, region) -> (in-region obj region)
        - open-lid(lid) -> (opened lid)
        
        Args:
            skeleton: List of action skeletons from VLM
            
        Returns:
            List of goal atoms representing final state
        """
        goals = []
        
        for action in skeleton:
            if action.action_name == 'pick':
                # Pick doesn't add to final goal directly
                # (we care about where things end up)
                pass
                
            elif action.action_name == 'place':
                obj = self.OBJECT_MAP.get(action.args[0], action.args[0])
                region = self.REGION_MAP.get(action.args[1], action.args[1])
                goals.append(PDDLAtom('in-region', (obj, region)))
                
            elif action.action_name in ['open-lid', 'open_lid']:
                lid = self.OBJECT_MAP.get(action.args[0], action.args[0])
                goals.append(PDDLAtom('opened', (lid,)))
        
        # Always want hand empty at the end
        goals.append(PDDLAtom('hand-empty', ()))
        
        return goals
    
    def build_init_atoms(self, 
                         current_state: Optional[Dict] = None) -> List[Tuple]:
        """
        Build initial state atoms for PDDLStream.
        
        Args:
            current_state: Optional dictionary with current state info
            
        Returns:
            List of init atoms in tuple format
        """
        init = []
        
        # Robot types and predicates
        init.append(('robot', 'panda'))
        init.append(('hand-empty',))
        
        # Objects - all movable
        movable_objects = ['mug_box', 'mug_inside_box', 'soup', 'mug_table', 
                          'mug_cupboard', 'mustard', 'spam', 'sugar', 'crackers']
        for obj in movable_objects:
            init.append(('movable', obj))
        
        # Regions
        regions = ['placement_boundary', 'cupboard_boundary', 'groceries_boundary',
                   'table', 'box-top', 'box-inside']
        for r in regions:
            init.append(('region', r))
        
        # Lid
        init.append(('lid', 'box_lid'))
        init.append(('closed', 'box_lid'))
        
        # Initial locations
        # mug_box starts on box-top
        init.append(('in-region', 'mug_box', 'box-top'))
        
        # mug_inside_box is inside the box (uses 'inside' predicate)
        init.append(('inside', 'mug_inside_box', 'box_lid'))
        
        # Other objects on table
        init.append(('in-region', 'soup', 'table'))
        
        # Robot at home configuration
        # Note: The actual configuration will be sampled by streams
        # We use a symbolic placeholder
        init.append(('conf', 'q_home'))
        init.append(('at-conf', 'q_home'))
        init.append(('is-home', 'q_home'))
        
        return init
    
    def build_goal(self, skeleton: List[ActionSkeleton]) -> Tuple:
        """
        Build PDDLStream goal from skeleton.
        
        Args:
            skeleton: Action skeleton from VLM
            
        Returns:
            Goal tuple for PDDLStream
        """
        goal_atoms = self.skeleton_to_goal_sequence(skeleton)
        
        print(f"[Skeleton Translator] Converted {len(skeleton)} actions to {len(goal_atoms)} goal atoms:")
        for g in goal_atoms:
            print(f"  - {g}")
            
        # Convert to PDDLStream format
        if len(goal_atoms) == 1:
            return goal_atoms[0].to_tuple()
        else:
            # Conjunction of goals
            return ('and',) + tuple(a.to_tuple() for a in goal_atoms)
    
    def build_problem(self, 
                      skeleton: List[ActionSkeleton],
                      init_atoms: Optional[List[Tuple]] = None) -> Dict[str, Any]:
        """
        Build complete PDDLStream problem from skeleton.
        
        Args:
            skeleton: Action skeleton from VLM
            init_atoms: Optional pre-built init atoms (from environment)
            
        Returns:
            Dictionary with problem specification
        """
        # Read domain and stream files
        with open(self.domain_path, 'r') as f:
            domain_pddl = f.read()
        
        with open(self.stream_path, 'r') as f:
            stream_pddl = f.read()
        
        # Build init if not provided
        if init_atoms is None:
            init_atoms = self.build_init_atoms()
        
        # Build goal
        goal = self.build_goal(skeleton)
        
        return {
            'domain_pddl': domain_pddl,
            'stream_pddl': stream_pddl,
            'init': init_atoms,
            'goal': goal,
            'constant_map': {'box-top': 'box-top'},
            'skeleton': skeleton,
        }
    
    def skeleton_to_action_sequence(self, 
                                     skeleton: List[ActionSkeleton]) -> List[Dict]:
        """
        Convert skeleton to a structured action sequence for the executor.
        
        This creates a "staged" execution plan where each stage
        corresponds to one high-level action from the skeleton.
        
        Args:
            skeleton: Action skeleton from VLM
            
        Returns:
            List of action dictionaries with execution details
        """
        actions = []
        
        for i, action in enumerate(skeleton):
            action_dict = {
                'stage': i + 1,
                'action': action.action_name,
                'args': action.args,
                'pddl_action': self.ACTION_MAP.get(action.action_name, action.action_name),
            }
            
            # Add execution hints based on action type
            if action.action_name == 'pick':
                obj = self.OBJECT_MAP.get(action.args[0], action.args[0])
                action_dict['object'] = obj
                action_dict['requires'] = ['sample-pick-kin']
                action_dict['effects'] = [('holding', obj), ('not', ('hand-empty',))]
                
            elif action.action_name == 'place':
                obj = self.OBJECT_MAP.get(action.args[0], action.args[0])
                region = self.REGION_MAP.get(action.args[1], action.args[1])
                action_dict['object'] = obj
                action_dict['region'] = region
                action_dict['requires'] = ['sample-stable-pose', 'sample-place-kin']
                action_dict['effects'] = [('in-region', obj, region), ('hand-empty',)]
                
            elif action.action_name in ['open-lid', 'open_lid']:
                lid = self.OBJECT_MAP.get(action.args[0], action.args[0])
                action_dict['object'] = lid
                action_dict['requires'] = ['sample-open-lid']
                action_dict['effects'] = [('opened', lid), ('not', ('closed', lid))]
            
            actions.append(action_dict)
        
        return actions
    
    def validate_skeleton_against_domain(self, 
                                          skeleton: List[ActionSkeleton]) -> Tuple[bool, List[str]]:
        """
        Validate that skeleton uses only valid actions, objects, and regions.
        
        Args:
            skeleton: Action skeleton to validate
            
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        for i, action in enumerate(skeleton):
            # Check action is valid
            if action.action_name not in self.ACTION_MAP:
                errors.append(f"Step {i+1}: Unknown action '{action.action_name}'")
                continue
            
            # Check objects/regions
            if action.action_name == 'pick':
                if len(action.args) != 1:
                    errors.append(f"Step {i+1}: pick requires 1 argument, got {len(action.args)}")
                elif action.args[0] not in self.OBJECT_MAP:
                    errors.append(f"Step {i+1}: Unknown object '{action.args[0]}'")
                    
            elif action.action_name == 'place':
                if len(action.args) != 2:
                    errors.append(f"Step {i+1}: place requires 2 arguments, got {len(action.args)}")
                else:
                    if action.args[0] not in self.OBJECT_MAP:
                        errors.append(f"Step {i+1}: Unknown object '{action.args[0]}'")
                    if action.args[1] not in self.REGION_MAP:
                        errors.append(f"Step {i+1}: Unknown region '{action.args[1]}'")
                        
            elif action.action_name in ['open-lid', 'open_lid']:
                if len(action.args) != 1:
                    errors.append(f"Step {i+1}: open-lid requires 1 argument, got {len(action.args)}")
                elif action.args[0] not in ['box_lid']:
                    errors.append(f"Step {i+1}: Unknown lid '{action.args[0]}'")
        
        return len(errors) == 0, errors


# ============================================================================
# INTEGRATION WITH PDDLSTREAM
# ============================================================================

def create_pddlstream_problem(skeleton: List[ActionSkeleton],
                               stream_map: Dict = None,
                               env=None) -> Any:
    """
    Create a PDDLStream problem object from skeleton.
    
    This integrates with the existing solve_with_rules.py infrastructure.
    
    Args:
        skeleton: Action skeleton from VLM
        stream_map: Stream function map (from rlbench_kitchen_streams_constrained)
        env: RLBench environment (for init state)
        
    Returns:
        PDDLProblem object ready for solving
    """
    # Import PDDLStream
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pddlstream'))
    from pddlstream.language.constants import PDDLProblem
    from pddlstream.utils import read
    
    translator = SkeletonToPDDL()
    
    # Build problem spec
    problem_spec = translator.build_problem(skeleton)
    
    # Get stream map if not provided
    if stream_map is None:
        from rlbench_kitchen_streams_constrained import get_stream_map
        stream_map = get_stream_map()
    
    # Create PDDLProblem
    problem = PDDLProblem(
        domain_pddl=problem_spec['domain_pddl'],
        constant_map=problem_spec['constant_map'],
        stream_pddl=problem_spec['stream_pddl'],
        stream_map=stream_map,
        init=problem_spec['init'],
        goal=problem_spec['goal'],
    )
    
    return problem


# ============================================================================
# TESTING
# ============================================================================

def test_translation():
    """Test skeleton to PDDL translation."""
    print("Testing Skeleton to PDDL translation...")
    
    translator = SkeletonToPDDL()
    
    # Test skeleton
    skeleton = [
        ActionSkeleton('pick', ('mug_box',)),
        ActionSkeleton('place', ('mug_box', 'placement_boundary')),
        ActionSkeleton('open-lid', ('box_lid',)),
        ActionSkeleton('pick', ('mug_inside_box',)),
        ActionSkeleton('place', ('mug_inside_box', 'placement_boundary')),
        ActionSkeleton('pick', ('soup',)),
        ActionSkeleton('place', ('soup', 'cupboard_boundary')),
    ]
    
    # Test goal generation
    print("\n=== Goal Atoms ===")
    goals = translator.skeleton_to_goal_sequence(skeleton)
    for g in goals:
        print(f"  {g}")
    
    # Test action sequence
    print("\n=== Action Sequence ===")
    actions = translator.skeleton_to_action_sequence(skeleton)
    for a in actions:
        print(f"  Stage {a['stage']}: {a['action']}({', '.join(a['args'])})")
        print(f"    Requires: {a.get('requires', [])}")
        print(f"    Effects: {a.get('effects', [])}")
    
    # Test validation
    print("\n=== Validation ===")
    is_valid, errors = translator.validate_skeleton_against_domain(skeleton)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test full problem build
    print("\n=== Problem Spec ===")
    problem = translator.build_problem(skeleton)
    print(f"Goal: {problem['goal']}")
    print(f"Init atoms count: {len(problem['init'])}")
    

def test_invalid_skeleton():
    """Test validation with invalid skeleton."""
    print("\nTesting invalid skeleton validation...")
    
    translator = SkeletonToPDDL()
    
    invalid_skeleton = [
        ActionSkeleton('grab', ('mug_box',)),  # Invalid action
        ActionSkeleton('place', ('unknown_obj', 'table')),  # Unknown object
        ActionSkeleton('place', ('soup', 'moon')),  # Unknown region
    ]
    
    is_valid, errors = translator.validate_skeleton_against_domain(invalid_skeleton)
    print(f"Valid: {is_valid}")
    print("Errors:")
    for e in errors:
        print(f"  - {e}")


if __name__ == "__main__":
    test_translation()
    test_invalid_skeleton()
