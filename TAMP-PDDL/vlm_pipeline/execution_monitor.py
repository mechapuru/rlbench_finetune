"""
Execution Monitor
=================
Monitors action execution and validates success/failure.

Key Features:
1. Pre-execution checks (object exists, gripper state, etc.)
2. Post-execution validation (object in target region)
3. Clear failure classification for replanning decisions
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class FailureType(Enum):
    """Classification of execution failures for replanning."""
    NONE = "none"
    
    # Planning failures
    NO_IK_SOLUTION = "no_ik_solution"           # No inverse kinematics solution
    NO_MOTION_PLAN = "no_motion_plan"           # Motion planner failed
    NO_GRASP_FOUND = "no_grasp_found"           # No valid grasp configuration
    PDDL_NO_PLAN = "pddl_no_plan"               # PDDL planner found no solution
    
    # Execution failures  
    COLLISION_DETECTED = "collision_detected"   # Robot collided during motion
    GRASP_FAILED = "grasp_failed"               # Object not grasped properly
    OBJECT_DROPPED = "object_dropped"           # Object fell during transport
    PLACEMENT_FAILED = "placement_failed"       # Object not in target region
    
    # Precondition failures
    OBJECT_NOT_FOUND = "object_not_found"       # Object doesn't exist
    OBJECT_BLOCKED = "object_blocked"           # Object is blocked by another
    GRIPPER_NOT_EMPTY = "gripper_not_empty"     # Can't pick - already holding
    GRIPPER_EMPTY = "gripper_empty"             # Can't place - not holding
    LID_CLOSED = "lid_closed"                   # Can't access - lid is closed
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ExecutionFeedback:
    """Detailed feedback from action execution."""
    success: bool
    failure_type: FailureType
    message: str
    
    # For replanning context
    object_name: Optional[str] = None
    target_region: Optional[str] = None
    blocking_object: Optional[str] = None
    
    # Object positions (for debugging)
    object_position_before: Optional[List[float]] = None
    object_position_after: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "failure_type": self.failure_type.value,
            "message": self.message,
            "object_name": self.object_name,
            "target_region": self.target_region,
            "blocking_object": self.blocking_object,
        }


class ExecutionMonitor:
    """
    Monitors and validates action execution.
    """
    
    def __init__(self, env):
        self.env = env
        self.execution_log: List[ExecutionFeedback] = []
    
    def reset(self):
        """Reset the monitor."""
        self.execution_log = []
    
    # =========================================================================
    # PRE-EXECUTION CHECKS
    # =========================================================================
    
    def check_pick_preconditions(self, object_name: str) -> ExecutionFeedback:
        """Check if pick action can be attempted."""
        obj = self.env.get_object(object_name)
        
        if obj is None:
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.OBJECT_NOT_FOUND,
                message=f"Object '{object_name}' not found in environment",
                object_name=object_name
            )
        
        # Check if object is inside closed box
        if object_name in ['mug4', 'mug_inside_box']:
            lid = self.env.get_object('box_lid')
            if lid:
                lid_pos = lid.get_position()
                # Simple heuristic: if lid Z is low, it's closed
                if lid_pos[2] < 0.85:  # Adjust threshold as needed
                    return ExecutionFeedback(
                        success=False,
                        failure_type=FailureType.LID_CLOSED,
                        message=f"Cannot pick '{object_name}' - box lid is closed",
                        object_name=object_name,
                        blocking_object='box_lid'
                    )
        
        return ExecutionFeedback(
            success=True,
            failure_type=FailureType.NONE,
            message="Preconditions satisfied",
            object_name=object_name
        )
    
    def check_place_preconditions(self, object_name: str, target_region: str) -> ExecutionFeedback:
        """Check if place action can be attempted."""
        # For now, basic checks
        return ExecutionFeedback(
            success=True,
            failure_type=FailureType.NONE,
            message="Preconditions satisfied",
            object_name=object_name,
            target_region=target_region
        )
    
    def check_open_lid_preconditions(self) -> ExecutionFeedback:
        """Check if open_lid action can be attempted."""
        lid = self.env.get_object('box_lid')
        if lid is None:
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.OBJECT_NOT_FOUND,
                message="box_lid not found",
                object_name='box_lid'
            )
        
        # Check if mug is on top of box (blocking lid)
        mug2 = self.env.get_object('mug2')  # mug_box
        if mug2:
            mug_pos = mug2.get_position()
            lid_pos = lid.get_position()
            # If mug is above lid and close in XY
            if (mug_pos[2] > lid_pos[2] and 
                abs(mug_pos[0] - lid_pos[0]) < 0.15 and 
                abs(mug_pos[1] - lid_pos[1]) < 0.15):
                return ExecutionFeedback(
                    success=False,
                    failure_type=FailureType.OBJECT_BLOCKED,
                    message="Cannot open lid - mug_box is on top",
                    object_name='box_lid',
                    blocking_object='mug_box'
                )
        
        return ExecutionFeedback(
            success=True,
            failure_type=FailureType.NONE,
            message="Preconditions satisfied",
            object_name='box_lid'
        )
    
    # =========================================================================
    # POST-EXECUTION VALIDATION
    # =========================================================================
    
    def validate_pick_place(self, object_name: str, target_region: str,
                            pos_before: List[float]) -> ExecutionFeedback:
        """
        Validate that pick-place succeeded by checking object position.
        """
        obj = self.env.get_object(object_name)
        if obj is None:
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.OBJECT_NOT_FOUND,
                message=f"Object '{object_name}' not found after execution",
                object_name=object_name,
                target_region=target_region
            )
        
        pos_after = list(obj.get_position())
        
        # Check if object moved significantly
        displacement = np.linalg.norm(np.array(pos_after) - np.array(pos_before))
        
        if displacement < 0.05:  # Less than 5cm movement
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.GRASP_FAILED,
                message=f"Object '{object_name}' didn't move (displacement: {displacement:.3f}m)",
                object_name=object_name,
                target_region=target_region,
                object_position_before=pos_before,
                object_position_after=pos_after
            )
        
        # Check if object is in target region
        in_region = self._check_in_region(obj, target_region)
        
        if not in_region:
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.PLACEMENT_FAILED,
                message=f"Object '{object_name}' not in target region '{target_region}'",
                object_name=object_name,
                target_region=target_region,
                object_position_before=pos_before,
                object_position_after=pos_after
            )
        
        # Check if object fell (Z too low)
        if pos_after[2] < 0.7:  # Below table level
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.OBJECT_DROPPED,
                message=f"Object '{object_name}' appears to have fallen (z={pos_after[2]:.3f})",
                object_name=object_name,
                target_region=target_region,
                object_position_before=pos_before,
                object_position_after=pos_after
            )
        
        return ExecutionFeedback(
            success=True,
            failure_type=FailureType.NONE,
            message=f"Successfully placed '{object_name}' in '{target_region}'",
            object_name=object_name,
            target_region=target_region,
            object_position_before=pos_before,
            object_position_after=pos_after
        )
    
    def validate_open_lid(self, pos_before: List[float]) -> ExecutionFeedback:
        """Validate that lid was opened."""
        lid = self.env.get_object('box_lid')
        if lid is None:
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.OBJECT_NOT_FOUND,
                message="box_lid not found after execution",
                object_name='box_lid'
            )
        
        pos_after = list(lid.get_position())
        
        # Lid should have moved in X or Y direction
        displacement_xy = np.linalg.norm(
            np.array(pos_after[:2]) - np.array(pos_before[:2])
        )
        
        if displacement_xy < 0.1:  # Less than 10cm horizontal movement
            return ExecutionFeedback(
                success=False,
                failure_type=FailureType.GRASP_FAILED,
                message=f"Lid didn't slide open (displacement: {displacement_xy:.3f}m)",
                object_name='box_lid',
                object_position_before=pos_before,
                object_position_after=pos_after
            )
        
        return ExecutionFeedback(
            success=True,
            failure_type=FailureType.NONE,
            message="Successfully opened lid",
            object_name='box_lid',
            object_position_before=pos_before,
            object_position_after=pos_after
        )
    
    def _check_in_region(self, obj, region_name: str) -> bool:
        """Check if object is within a target region. Uses dynamic bounds from env."""
        pos = obj.get_position()
        
        # Special case: placement_boundary -> just check if on table
        if region_name == 'placement_boundary':
            table = self.env.regions.get('table') if hasattr(self.env, 'regions') else None
            if table is not None:
                t_min_x, t_max_x, t_min_y, t_max_y, t_min_z, t_max_z = table.get_bounding_box()
                tx, ty, tz = table.get_position()
                
                # Table world bounds with generous tolerance
                tolerance = 0.15
                in_x = (tx + t_min_x - tolerance) <= pos[0] <= (tx + t_max_x + tolerance)
                in_y = (ty + t_min_y - tolerance) <= pos[1] <= (ty + t_max_y + tolerance)
                in_z = pos[2] > 0.7  # Just check it's above table level
                
                return in_x and in_y and in_z
            else:
                return pos[2] > 0.7
        
        # Try to get region from environment dynamically
        region = self.env.regions.get(region_name) if hasattr(self.env, 'regions') else None
        
        if region is not None:
            # Get region bounds (local frame)
            r_min_x, r_max_x, r_min_y, r_max_y, r_min_z, r_max_z = region.get_bounding_box()
            
            # Get region position (world frame)
            rx, ry, rz = region.get_position()
            
            # Convert to world bounds
            world_min_x = rx + r_min_x
            world_max_x = rx + r_max_x
            world_min_y = ry + r_min_y
            world_max_y = ry + r_max_y
            world_min_z = rz + r_min_z
            world_max_z = rz + r_max_z
            
            # Add tolerance
            tolerance = 0.1
            
            in_x = (world_min_x - tolerance) <= pos[0] <= (world_max_x + tolerance)
            in_y = (world_min_y - tolerance) <= pos[1] <= (world_max_y + tolerance)
            in_z = (world_min_z - tolerance) <= pos[2] <= (world_max_z + tolerance)
            
            return in_x and in_y and in_z
        
        # Fallback: just check object didn't fall
        return pos[2] > 0.7
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    def log_feedback(self, feedback: ExecutionFeedback):
        """Log execution feedback."""
        self.execution_log.append(feedback)
        
        status = "✓" if feedback.success else "✗"
        print(f"  [{status}] {feedback.message}")
        
        if not feedback.success:
            print(f"      Failure Type: {feedback.failure_type.value}")
            if feedback.blocking_object:
                print(f"      Blocked by: {feedback.blocking_object}")
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of failures for replanning context."""
        failures = [f for f in self.execution_log if not f.success]
        
        if not failures:
            return {"has_failures": False}
        
        last_failure = failures[-1]
        return {
            "has_failures": True,
            "failure_count": len(failures),
            "last_failure": last_failure.to_dict(),
            "failure_types": [f.failure_type.value for f in failures]
        }
        
    def get_llm_failure_context(self) -> Optional[str]:
        """Translate the last failure into a causal semantic string for the LLM."""
        failures = [f for f in self.execution_log if not f.success]
        if not failures:
            return None
            
        f = failures[-1]
        context = f"FAILURE TYPE: {f.failure_type.name}\n"
        context += f"REASON: {f.message}\n"
        if f.blocking_object:
            context += f"CRITICAL: Action blocked by '{f.blocking_object}'. You MUST prioritize moving changing the state of the blocking object.\n"
            
        if f.failure_type in [FailureType.NO_IK_SOLUTION, FailureType.NO_MOTION_PLAN, FailureType.NO_GRASP_FOUND, FailureType.PDDL_NO_PLAN]:
            context += "SUGGESTION: The target object is physically unreachable. It might be blocked by a nearby object. Consider moving obstacles out of the way first."
            
        if f.failure_type in [FailureType.GRASP_FAILED, FailureType.PLACEMENT_FAILED, FailureType.OBJECT_DROPPED]:
            context += "SUGGESTION: The object slipped or was dropped during physical execution. Please re-evaluate your plan using the NEW physical state."
            
        return context


def classify_pddl_failure(error_message: str) -> FailureType:
    """Classify a PDDL/planning error into a failure type."""
    error_lower = error_message.lower()
    
    if "no plan" in error_lower or "no solution" in error_lower:
        return FailureType.PDDL_NO_PLAN
    if "ik" in error_lower or "inverse kinematics" in error_lower:
        return FailureType.NO_IK_SOLUTION
    if "collision" in error_lower:
        return FailureType.COLLISION_DETECTED
    if "grasp" in error_lower:
        return FailureType.NO_GRASP_FOUND
    if "motion" in error_lower or "path" in error_lower:
        return FailureType.NO_MOTION_PLAN
    if "empty trajectory" in error_lower:
        return FailureType.NO_MOTION_PLAN
    
    return FailureType.UNKNOWN
