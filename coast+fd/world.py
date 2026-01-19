"""RLBench World Interface for COAST.

Provides a bridge between COAST TAMP algorithm and RLBench/PyRep simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object as PyRepObject
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

from rlbench.environment import Environment
from rlbench.backend.task import Task
from rlbench.backend.robot import Robot
from rlbench.backend.observation import Observation


@dataclass
class ObjectInfo:
    """Information about an object in the scene."""
    name: str
    pyrep_object: PyRepObject
    object_type: str  # "graspable", "fixed", "joint"
    pose: Optional[np.ndarray] = None
    
    def get_position(self) -> np.ndarray:
        """Get object position."""
        return np.array(self.pyrep_object.get_position())
    
    def get_pose(self) -> np.ndarray:
        """Get object pose (position + quaternion)."""
        return np.array(self.pyrep_object.get_pose())


@dataclass 
class RLBenchWorld:
    """Interface between COAST and RLBench/PyRep simulation.
    
    Provides methods for:
    - Object pose queries
    - Robot configuration queries/commands
    - Motion planning (via PyRep)
    - Collision checking
    - Gripper control
    """
    
    env: Environment
    task: Task
    
    # Populated after initialization
    robot: Robot = field(init=False)
    pyrep: PyRep = field(init=False)
    objects: Dict[str, ObjectInfo] = field(default_factory=dict)
    graspable_objects: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize robot and PyRep references."""
        self.robot = self.env._robot
        self.pyrep = self.env._pyrep
    
    @classmethod
    def from_task_name(
        cls,
        task_name: str,
        headless: bool = False,
        variation: int = 0
    ) -> "RLBenchWorld":
        """Create world from task name.
        
        Args:
            task_name: Name of RLBench task (e.g., "LongHorizonGrillTask")
            headless: Run without GUI
            variation: Task variation index
            
        Returns:
            Initialized RLBenchWorld
        """
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig
        
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.joint_positions = True
        obs_config.gripper_pose = True
        
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointPosition(),
            gripper_action_mode=Discrete()
        )
        
        env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless
        )
        env.launch()
        
        # Load task
        task = env.get_task(cls._get_task_class(task_name))
        task.reset()
        
        world = cls(env=env, task=task._task)
        world._discover_objects()
        
        return world
    
    @staticmethod
    def _get_task_class(task_name: str):
        """Get task class from name."""
        import importlib
        module_name = ''.join(
            ['_' + c.lower() if c.isupper() else c for c in task_name]
        ).lstrip('_')
        module = importlib.import_module(f"rlbench.tasks.{module_name}")
        return getattr(module, task_name)
    
    def _discover_objects(self):
        """Discover objects in the scene from task."""
        # Get graspable objects from task
        for obj in self.task.get_graspable_objects():
            name = obj.get_name()
            self.objects[name] = ObjectInfo(
                name=name,
                pyrep_object=obj,
                object_type="graspable"
            )
            self.graspable_objects.append(name)
    
    def add_object(self, name: str, obj: PyRepObject, object_type: str = "fixed"):
        """Manually add an object to the world."""
        self.objects[name] = ObjectInfo(
            name=name,
            pyrep_object=obj,
            object_type=object_type
        )
        if object_type == "graspable":
            self.graspable_objects.append(name)
    
    # ==================== Robot State ====================
    
    def get_robot_config(self) -> np.ndarray:
        """Get current robot joint configuration."""
        return np.array(self.robot.arm.get_joint_positions())
    
    def set_robot_config(self, config: np.ndarray) -> None:
        """Set robot joint configuration."""
        self.robot.arm.set_joint_positions(list(config), disable_dynamics=True)
    
    def get_gripper_pose(self) -> np.ndarray:
        """Get gripper (end-effector) pose."""
        return np.array(self.robot.arm.get_tip().get_pose())
    
    def get_gripper_position(self) -> np.ndarray:
        """Get gripper position."""
        return np.array(self.robot.arm.get_tip().get_position())
    
    def is_gripper_open(self) -> bool:
        """Check if gripper is open."""
        return self.robot.gripper.get_open_amount()[0] > 0.9
    
    # ==================== Object State ====================
    
    def get_object_pose(self, name: str) -> np.ndarray:
        """Get pose of named object."""
        return self.objects[name].get_pose()
    
    def get_object_position(self, name: str) -> np.ndarray:
        """Get position of named object."""
        return self.objects[name].get_position()
    
    def get_object(self, name: str) -> PyRepObject:
        """Get PyRep object by name."""
        return self.objects[name].pyrep_object
    
    # ==================== Motion Planning ====================
    
    def plan_to_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ignore_collisions: bool = False
    ) -> Optional[Any]:
        """Plan path to target pose.
        
        Args:
            position: Target position [x, y, z]
            orientation: Target orientation as euler angles [rx, ry, rz]
            ignore_collisions: Whether to ignore collisions
            
        Returns:
            ArmConfigurationPath if successful, None otherwise
        """
        try:
            path = self.robot.arm.get_path(
                position=list(position),
                euler=list(orientation),
                ignore_collisions=ignore_collisions
            )
            return path
        except Exception:
            return None
    
    def plan_linear_to_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ignore_collisions: bool = False
    ) -> Optional[Any]:
        """Plan linear path to target pose."""
        try:
            path = self.robot.arm.get_linear_path(
                position=list(position),
                euler=list(orientation),
                ignore_collisions=ignore_collisions
            )
            return path
        except Exception:
            return None
    
    def solve_ik(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ignore_collisions: bool = False
    ) -> Optional[np.ndarray]:
        """Solve inverse kinematics for target pose.
        
        Returns:
            Joint configuration if successful, None otherwise
        """
        try:
            config = self.robot.arm.solve_ik_via_sampling(
                position=list(position),
                euler=list(orientation),
                ignore_collisions=ignore_collisions
            )
            return np.array(config)
        except Exception:
            return None
    
    # ==================== Collision Checking ====================
    
    def check_collision(self, obj: Optional[PyRepObject] = None) -> bool:
        """Check if robot arm is in collision."""
        return self.robot.arm.check_arm_collision(obj)
    
    def check_config_collision(self, config: np.ndarray) -> bool:
        """Check if configuration is in collision."""
        # Save current state
        current_config = self.get_robot_config()
        
        # Set to test config and check
        self.set_robot_config(config)
        in_collision = self.check_collision()
        
        # Restore
        self.set_robot_config(current_config)
        return in_collision
    
    # ==================== Gripper Control ====================
    
    def grasp(self, object_name: str) -> bool:
        """Grasp an object."""
        obj = self.get_object(object_name)
        return self.robot.gripper.grasp(obj)
    
    def release(self) -> None:
        """Release grasped object."""
        self.robot.gripper.release()
    
    def open_gripper(self) -> bool:
        """Open gripper."""
        return self.robot.gripper.actuate(1.0, 0.04)
    
    def close_gripper(self) -> bool:
        """Close gripper."""
        return self.robot.gripper.actuate(0.0, 0.04)
    
    def get_grasped_objects(self) -> List[PyRepObject]:
        """Get list of currently grasped objects."""
        return self.robot.gripper.get_grasped_objects()
    
    # ==================== Simulation Control ====================
    
    def step(self) -> None:
        """Step simulation."""
        self.pyrep.step()
    
    def get_observation(self) -> Observation:
        """Get current observation."""
        return self.env._scene.get_observation()
    
    def shutdown(self) -> None:
        """Shutdown environment."""
        self.env.shutdown()
    
    # ==================== COAST Integration ====================
    
    def get_stream_state(self) -> Set[str]:
        """Get initial stream state for COAST.
        
        Returns certified facts about initial configuration and object poses.
        """
        state = set()
        
        # Initial robot configuration
        state.add("AtConf(q0)")
        
        # Object poses
        for name in self.objects:
            state.add(f"AtPose({name}, p0_{name})")
        
        # Gripper state
        if self.is_gripper_open():
            state.add("HandEmpty()")
        
        return state
    
    def get_coast_objects(self) -> List[Dict[str, Any]]:
        """Get objects for COAST planning.
        
        Returns list of object dicts compatible with coast.Object.
        """
        objects = []
        
        # Initial configuration
        objects.append({
            "name": "q0",
            "type": "conf",
            "value": self.get_robot_config()
        })
        
        # Object poses
        for name, info in self.objects.items():
            objects.append({
                "name": f"p0_{name}",
                "type": "pose",
                "value": info.get_pose()
            })
            objects.append({
                "name": name,
                "type": "object",
                "value": info.pyrep_object
            })
        
        return objects
