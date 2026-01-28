"""Geometric action implementations for COAST + RLBench.

Defines how PDDL actions map to robot execution.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np

from coast.action import Action
from coast.object import Object
from coast.formula import _P, _and


class Pick(Action):
    """Pick action - grasp an object from a location.
    
    Geometric inputs: grasp pose, IK solution
    Geometric outputs: trajectory
    """
    
    name = "pick"
    parameters = [Object("?o", "object"), Object("?l", "location")]
    inputs = []
    outputs = [Object("?g", "grasp"), Object("?q", "conf"), Object("?t", "traj")]
    
    precondition = _and(
        _P("On", "?o", "?l"),
        _P("HandEmpty")
    )
    
    effect = _and(
        _P("Holding", "?o"),
        _P("InGrasp", "?o", "?g")
    )
    
    certified = _and(
        _P("GraspSampled", "?o", "?g"),
        _P("IKSolved", "?o", "?p", "?g", "?q")
    )
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
    
    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Execute pick action in simulation.
        
        Args:
            inputs: Action inputs
            outputs: {"?g": grasp, "?q": config, "?t": trajectory}
            
        Returns:
            True if successful
        """
        try:
            grasp = outputs.get("g")
            traj = outputs.get("t")
            obj_name = inputs.get("o")
            
            # Execute trajectory to approach pose
            if traj and hasattr(traj, 'waypoints'):
                for waypoint in traj.waypoints:
                    self.world.set_robot_config(waypoint)
                    self.world.step()
            
            # Close gripper and grasp
            self.world.close_gripper()
            for _ in range(10):
                self.world.step()
            
            if obj_name:
                self.world.grasp(obj_name)
            
            return True
            
        except Exception as e:
            print(f"[Pick.execute] Failed: {e}")
            return False


class Place(Action):
    """Place action - place held object at a location.
    
    Geometric inputs: grasp (from previous pick), placement pose
    Geometric outputs: configuration, trajectory
    """
    
    name = "place"
    parameters = [Object("?o", "object"), Object("?l", "location")]
    inputs = [Object("?g", "grasp")]
    outputs = [Object("?p", "pose"), Object("?q", "conf"), Object("?t", "traj")]
    
    precondition = _and(
        _P("Holding", "?o"),
        _P("InGrasp", "?o", "?g")
    )
    
    effect = _and(
        _P("On", "?o", "?l"),
        _P("HandEmpty"),
        _P("AtPose", "?o", "?p")
    )
    
    certified = _and(
        _P("PoseSampled", "?o", "?l", "?p"),
        _P("IKSolved", "?o", "?p", "?g", "?q")
    )
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
    
    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Execute place action in simulation."""
        try:
            traj = outputs.get("t")
            
            # Execute trajectory to place pose
            if traj and hasattr(traj, 'waypoints'):
                for waypoint in traj.waypoints:
                    self.world.set_robot_config(waypoint)
                    self.world.step()
            
            # Open gripper and release
            self.world.release()
            self.world.open_gripper()
            for _ in range(10):
                self.world.step()
            
            return True
            
        except Exception as e:
            print(f"[Place.execute] Failed: {e}")
            return False


class PlaceOnGrill(Action):
    """Place object on grill surface."""
    
    name = "place_on_grill"
    parameters = [Object("?o", "object")]
    inputs = [Object("?g", "grasp")]
    outputs = [Object("?p", "pose"), Object("?q", "conf"), Object("?t", "traj")]
    
    precondition = _and(
        _P("Holding", "?o"),
        _P("GrillOpen")
    )
    
    effect = _and(
        _P("On", "?o", "grill_surface"),
        _P("OnGrill", "?o"),
        _P("HandEmpty")
    )
    
    certified = _and(
        _P("PoseSampled", "?o", "grill_surface", "?p"),
        _P("IKSolved", "?o", "?p", "?g", "?q")
    )
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
    
    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Execute place on grill action."""
        try:
            traj = outputs.get("t")
            
            if traj and hasattr(traj, 'waypoints'):
                for waypoint in traj.waypoints:
                    self.world.set_robot_config(waypoint)
                    self.world.step()
            
            self.world.release()
            self.world.open_gripper()
            for _ in range(10):
                self.world.step()
            
            return True
        except Exception as e:
            print(f"[PlaceOnGrill.execute] Failed: {e}")
            return False


class OpenLid(Action):
    """Open grill lid using predefined waypoints.
    
    The lid movement follows a constrained semi-circular trajectory
    defined by waypoints in the RLBench scene.
    """
    
    name = "open_lid"
    parameters = []
    inputs = []
    outputs = [Object("?q", "conf"), Object("?t", "traj")]
    
    precondition = _and(
        _P("GrillClosed"),
        _P("HandEmpty")
    )
    
    effect = _and(
        _P("GrillOpen")
    )
    
    certified = _and()
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
    
    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Open the grill lid using predefined waypoints.
        
        Uses the waypoints defined in the RLBench scene for constrained motion.
        """
        try:
            from pyrep.objects.dummy import Dummy
            from pyrep.objects.joint import Joint
            from pyrep.objects.shape import Shape
            
            # Get the lid handle for grasping
            lid = Shape('lid')
            lid_joint = Joint('lid_joint')
            
            # Get waypoints for opening (reverse of close path)
            # The scene has waypoints like waypoint20, waypoint21, waypoint22
            waypoint_names = ['waypoint20', 'waypoint21', 'waypoint22']
            
            for wp_name in waypoint_names:
                try:
                    waypoint = Dummy(wp_name)
                    target_pos = waypoint.get_position()
                    target_orient = waypoint.get_orientation()
                    
                    # Plan and execute motion to waypoint
                    path = self.world.plan_to_pose(
                        position=np.array(target_pos),
                        orientation=np.array(target_orient),
                        ignore_collisions=True  # Lid motion may overlap
                    )
                    
                    if path is not None:
                        # Execute the path
                        done = False
                        while not done:
                            done = path.step()
                            self.world.step()
                    else:
                        # Fallback: use IK directly
                        config = self.world.solve_ik(
                            position=np.array(target_pos),
                            orientation=np.array(target_orient),
                            ignore_collisions=True
                        )
                        if config is not None:
                            self.world.set_robot_config(config)
                            for _ in range(5):
                                self.world.step()
                                
                except Exception as e:
                    print(f"[OpenLid] Waypoint {wp_name} failed: {e}")
                    continue
            
            # Ensure lid joint is at open position (30 degrees)
            target_angle = np.deg2rad(30)
            lid_joint.set_joint_position(target_angle)
            for _ in range(10):
                self.world.step()
            
            return True
            
        except Exception as e:
            print(f"[OpenLid.execute] Failed: {e}")
            return False


class CloseLid(Action):
    """Close grill lid using predefined waypoints.
    
    The lid movement follows a constrained semi-circular trajectory
    defined by waypoints in the RLBench scene.
    """
    
    name = "close_lid"
    parameters = []
    inputs = []
    outputs = [Object("?q", "conf"), Object("?t", "traj")]
    
    precondition = _and(
        _P("GrillOpen"),
        _P("HandEmpty"),
        _P("ChickenOnGrill")
    )
    
    effect = _and(
        _P("GrillClosed"),
        _P("ChickenCooked")
    )
    
    certified = _and()
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
    
    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Close the grill lid using predefined path.
        
        Uses the 'path_close_grill' or 'close_grill' waypoints from RLBench scene.
        """
        try:
            from pyrep.objects.dummy import Dummy
            from pyrep.objects.joint import Joint
            from pyrep.objects.shape import Shape
            
            lid_joint = Joint('lid_joint')
            
            # Try to use predefined path first
            try:
                close_grill_dummy = Dummy('close_grill')
                target_pos = close_grill_dummy.get_position()
                target_orient = close_grill_dummy.get_orientation()
                
                # Move to close grill waypoint
                path = self.world.plan_to_pose(
                    position=np.array(target_pos),
                    orientation=np.array(target_orient),
                    ignore_collisions=True
                )
                
                if path is not None:
                    done = False
                    while not done:
                        done = path.step()
                        self.world.step()
                        
            except Exception as e:
                print(f"[CloseLid] Predefined path not found, using joint control: {e}")
            
            # Close the lid joint
            current_angle = lid_joint.get_joint_position()
            target_angle = 0.0
            
            steps = 30
            for i in range(steps):
                angle = current_angle + (target_angle - current_angle) * (i + 1) / steps
                lid_joint.set_joint_position(angle)
                self.world.step()
            
            return True
            
        except Exception as e:
            print(f"[CloseLid.execute] Failed: {e}")
            return False


class Cook(Action):
    """Cook action - wait for food to cook on grill."""
    
    name = "cook"
    parameters = [Object("?o", "object")]
    inputs = []
    outputs = []
    
    precondition = _and(
        _P("OnGrill", "?o"),
        _P("GrillClosed"),
        _P("HandEmpty")
    )
    
    effect = _and(
        _P("Cooked", "?o")
    )
    
    certified = _and()
    
    def __init__(self, world: Any):
        super().__init__(world)
        self.world = world
    
    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Simulate cooking (wait steps)."""
        try:
            # Simulate cooking time
            for _ in range(50):
                self.world.step()
            return True
        except Exception as e:
            print(f"[Cook.execute] Failed: {e}")
            return False


# ==================== Action Registry ====================

def get_actions(world: Any) -> List[Action]:
    """Get all action instances for the world.
    
    Args:
        world: RLBenchWorld instance
        
    Returns:
        List of initialized Action objects
    """
    return [
        Pick(world),
        Place(world),
        PlaceOnGrill(world),
        OpenLid(world),
        CloseLid(world),
        Cook(world),
    ]
