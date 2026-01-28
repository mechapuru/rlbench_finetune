"""Stream implementations for COAST + RLBench integration.

This module implements the motion planning streams that COAST uses
to ground task plans with geometric information.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from coast.stream import Stream
from coast.object import Object
from coast.formula import _P, _and

# Import actions so they are discoverable by coast.main.load_geometric_actions
from .actions import Pick, Place, PlaceOnGrill, OpenLid, CloseLid, Cook


# ==================== Stream Objects ====================

@dataclass
class GraspPose:
    """Represents a grasp pose for an object."""
    position: np.ndarray
    orientation: np.ndarray  # Euler angles
    approach_direction: np.ndarray
    
    def __repr__(self):
        return f"GraspPose(pos={self.position[:2]}...)"


@dataclass
class PlacementPose:
    """Represents a placement pose for an object on a surface."""
    position: np.ndarray
    orientation: np.ndarray
    
    def __repr__(self):
        return f"PlacementPose(pos={self.position[:2]}...)"


@dataclass
class Trajectory:
    """Represents a robot trajectory."""
    waypoints: List[np.ndarray]  # List of joint configurations
    
    def __len__(self):
        return len(self.waypoints)
    
    def __repr__(self):
        return f"Trajectory({len(self.waypoints)} waypoints)"


# ==================== Stream Implementations ====================

class SampleGrasp(Stream):
    """Sample grasp poses for an object.
    
    Uses PyRep to sample valid grasp poses around the object.
    """
    
    name = "sample-grasp"
    inputs = [Object("?o", "object")]
    outputs = [Object("?g", "grasp")]
    fluents = []
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
        self.certified = _P("GraspSampled", "?o", "?g")
    
    def sample(
        self,
        inputs: Dict[str, Any],
        fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Sample a grasp pose for the object.
        
        Args:
            inputs: {"?o": object_handle}
            fluents: Current fluent values
            
        Returns:
            {"?g": GraspPose} if successful, None otherwise
        """
        obj = inputs["?o"]
        
        try:
            # Get object position
            obj_pos = np.array(obj.get_position())
            obj_orient = np.array(obj.get_orientation())
            
            # Sample grasp approach from above with some randomness
            approach_offset = np.array([
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02),
                0.1  # Approach from above
            ])
            
            grasp_pos = obj_pos + approach_offset
            
            # Orientation: pointing down with slight random rotation
            grasp_orient = np.array([np.pi, 0, np.random.uniform(-0.3, 0.3)])
            
            grasp = GraspPose(
                position=grasp_pos,
                orientation=grasp_orient,
                approach_direction=np.array([0, 0, -1])
            )
            
            return {"?g": grasp}
            
        except Exception as e:
            print(f"[SampleGrasp] Failed: {e}")
            return None


class SamplePose(Stream):
    """Sample placement poses on a surface.
    
    Uses PyRep to sample valid placement locations.
    """
    
    name = "sample-pose"
    inputs = [Object("?o", "object"), Object("?l", "location")]
    outputs = [Object("?p", "pose")]
    fluents = []
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
        self.certified = _P("PoseSampled", "?o", "?l", "?p")
    
    def sample(
        self,
        inputs: Dict[str, Any],
        fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Sample a placement pose on the location.
        
        Args:
            inputs: {"?o": object_handle, "?l": location_name}
            fluents: Current fluent values
            
        Returns:
            {"?p": PlacementPose} if successful, None otherwise
        """
        obj = inputs["?o"]
        location = inputs["?l"]
        
        try:
            # Get location surface position
            if hasattr(location, 'get_position'):
                loc_pos = np.array(location.get_position())
            else:
                # Location is a string name - look up in world
                loc_pos = self.world.get_object_position(str(location))
            
            # Sample position on surface with randomness
            place_pos = loc_pos + np.array([
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.05, 0.05),
                0.05  # Slightly above surface
            ])
            
            # Random orientation around z-axis
            place_orient = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
            
            pose = PlacementPose(
                position=place_pos,
                orientation=place_orient
            )
            
            return {"?p": pose}
            
        except Exception as e:
            print(f"[SamplePose] Failed: {e}")
            return None


class SampleIK(Stream):
    """Solve inverse kinematics for a target pose.
    
    Uses PyRep's IK solver to find joint configurations.
    """
    
    name = "sample-ik"
    inputs = [
        Object("?o", "object"),
        Object("?p", "pose"),
        Object("?g", "grasp")
    ]
    outputs = [Object("?q", "conf"), Object("?t", "traj")]
    fluents = ["AtConf"]
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
        self.certified = _P("IKSolved", "?o", "?p", "?g", "?q")
    
    def sample(
        self,
        inputs: Dict[str, Any],
        fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Solve IK for the target pose.
        
        Args:
            inputs: {"?o": object, "?p": PlacementPose, "?g": GraspPose}
            fluents: {"AtConf": [{"?q": current_config}]}
            
        Returns:
            {"?q": joint_config, "?t": Trajectory} if successful
        """
        grasp = inputs["?g"]
        
        try:
            # Solve IK for grasp position
            target_config = self.world.solve_ik(
                position=grasp.position,
                orientation=grasp.orientation,
                ignore_collisions=False
            )
            
            if target_config is None:
                return None
            
            # Plan motion from current config to target
            current_config = self.world.get_robot_config()
            
            path = self.world.plan_to_pose(
                position=grasp.position,
                orientation=grasp.orientation
            )
            
            if path is None:
                # Fall back to linear interpolation
                traj = Trajectory(waypoints=[current_config, target_config])
            else:
                # Extract waypoints from path
                waypoints = []
                while not path.step():
                    waypoints.append(np.array(path.get_executed_joint_position_action()))
                waypoints.append(target_config)
                traj = Trajectory(waypoints=waypoints)
            
            return {"?q": target_config, "?t": traj}
            
        except Exception as e:
            print(f"[SampleIK] Failed: {e}")
            return None


class SampleMotion(Stream):
    """Plan collision-free motion between configurations.
    
    Uses PyRep's motion planning to find valid trajectories.
    """
    
    name = "sample-motion"
    inputs = [Object("?q1", "conf"), Object("?q2", "conf")]
    outputs = [Object("?t", "traj")]
    fluents = []
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
        self.certified = _P("MotionPlanned", "?q1", "?q2", "?t")
    
    def sample(
        self,
        inputs: Dict[str, Any],
        fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Plan motion between configurations.
        
        Args:
            inputs: {"?q1": start_config, "?q2": end_config}
            
        Returns:
            {"?t": Trajectory} if successful
        """
        q1 = inputs["?q1"]
        q2 = inputs["?q2"]
        
        try:
            # Set start configuration
            self.world.set_robot_config(q1)
            
            # Get end-effector pose for target config
            self.world.set_robot_config(q2)
            target_pos = self.world.get_gripper_position()
            target_orient = np.array(self.world.robot.arm.get_tip().get_orientation())
            
            # Restore start config
            self.world.set_robot_config(q1)
            
            # Plan path
            path = self.world.plan_to_pose(
                position=target_pos,
                orientation=target_orient
            )
            
            if path is None:
                # Fall back to linear interpolation
                traj = Trajectory(waypoints=[q1, q2])
            else:
                waypoints = [q1]
                while not path.step():
                    waypoints.append(np.array(path.get_executed_joint_position_action()))
                waypoints.append(q2)
                traj = Trajectory(waypoints=waypoints)
            
            return {"?t": traj}
            
        except Exception as e:
            print(f"[SampleMotion] Failed: {e}")
            return None


class CheckCollision(Stream):
    """Check if a trajectory is collision-free.
    
    Validates trajectories against obstacles in the scene.
    """
    
    name = "check-collision"
    inputs = [Object("?t", "traj")]
    outputs = []
    fluents = ["AtPose"]
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
        self.certified = _P("CollisionFree", "?t")
    
    def sample(
        self,
        inputs: Dict[str, Any],
        fluents: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Check trajectory for collisions.
        
        Args:
            inputs: {"?t": Trajectory}
            fluents: {"AtPose": [{"?o": obj, "?p": pose}, ...]}
            
        Returns:
            {} if collision-free, None otherwise
        """
        traj = inputs["?t"]
        
        try:
            # Save current state
            original_config = self.world.get_robot_config()
            
            # Check each waypoint for collisions
            for waypoint in traj.waypoints:
                if self.world.check_config_collision(waypoint):
                    # Restore and return failure
                    self.world.set_robot_config(original_config)
                    return None
            
            # Restore original state
            self.world.set_robot_config(original_config)
            
            return {}  # Success - no collision
            
        except Exception as e:
            print(f"[CheckCollision] Failed: {e}")
            return None


# ==================== Geometric Predicates ====================

# Predicates used in geometric state tracking
GEOMETRIC_PREDICATES = [
    _P("AtConf", "?q"),
    _P("AtPose", "?o", "?p"),
    _P("InGrasp", "?o", "?g"),
    _P("GraspSampled", "?o", "?g"),
    _P("PoseSampled", "?o", "?l", "?p"),
    _P("IKSolved", "?o", "?p", "?g", "?q"),
    _P("MotionPlanned", "?q1", "?q2", "?t"),
    _P("CollisionFree", "?t"),
]


# ==================== Stream Registry ====================

def get_streams(world: Any) -> List[Stream]:
    """Get all stream instances for the world.
    
    Args:
        world: RLBenchWorld instance
        
    Returns:
        List of initialized Stream objects
    """
    return [
        SampleGrasp(world),
        SamplePose(world),
        SampleIK(world),
        SampleMotion(world),
        CheckCollision(world),
    ]
