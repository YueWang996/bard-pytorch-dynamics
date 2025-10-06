"""Joint representation for robot kinematic chains."""

from typing import Optional, Tuple
import torch


class Joint:
    """
    Robot joint connecting two links.
    
    Supports revolute, prismatic, and fixed joints.
    
    Attributes:
        name: Joint identifier
        offset: Transform from parent link frame to joint frame
        joint_type: Type of joint ('fixed', 'revolute', 'prismatic')
        axis: 3D axis of rotation/translation in joint frame
        limits: (lower, upper) position limits or None
        velocity_limits: (lower, upper) velocity limits or None
        effort_limits: (lower, upper) force/torque limits or None
    
    Conventions:
        - Revolute: Rotation about axis (radians)
        - Prismatic: Translation along axis (meters)
        - Fixed: No motion (axis ignored)
    """
    
    TYPES = ['fixed', 'revolute', 'prismatic']

    def __init__(
        self,
        name: Optional[str] = None,
        offset: Optional[object] = None,
        joint_type: str = 'fixed',
        axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        limits: Optional[Tuple[float, float]] = None,
        velocity_limits: Optional[Tuple[float, float]] = None,
        effort_limits: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize joint.
        
        Args:
            name: Joint name/identifier
            offset: Transform3d from parent link to joint frame
            joint_type: One of 'fixed', 'revolute', 'prismatic'
            axis: Joint axis as (x, y, z) tuple
            dtype: PyTorch dtype for axis tensor
            device: PyTorch device for axis tensor
            limits: (min, max) position limits in rad or meters
            velocity_limits: (min, max) velocity limits
            effort_limits: (min, max) torque/force limits
            
        Raises:
            RuntimeError: If joint_type is not in TYPES
        """
        self.name = name if name is not None else "unnamed_joint"
        self.offset = offset
        
        if joint_type not in self.TYPES:
            raise RuntimeError(
                f"Joint '{name}' has type '{joint_type}', "
                f"but only {self.TYPES} are supported"
            )
        self.joint_type = joint_type
        
        # Store axis as normalized tensor
        if axis is None:
            self.axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        else:
            if torch.is_tensor(axis):
                self.axis = axis.clone().detach().to(dtype=dtype, device=device)
            else:
                self.axis = torch.tensor(axis, dtype=dtype, device=device)
        
        # Normalize axis to unit length
        axis_norm = self.axis.norm()
        if axis_norm > 1e-12:
            self.axis = self.axis / axis_norm
        
        self.limits = limits
        self.velocity_limits = velocity_limits
        self.effort_limits = effort_limits

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> 'Joint':
        """
        Move joint data to specified dtype/device.
        
        Args:
            dtype: Target data type
            device: Target device
            
        Returns:
            Self for method chaining
        """
        if dtype is not None or device is not None:
            kwargs = {}
            if dtype is not None:
                kwargs['dtype'] = dtype
            if device is not None:
                kwargs['device'] = device
            self.axis = self.axis.to(**kwargs)
        
        if self.offset is not None:
            self.offset = self.offset.to(dtype=dtype, device=device)
        
        return self

    def clamp(self, joint_position: torch.Tensor) -> torch.Tensor:
        """
        Clamp joint position to limits.
        
        Args:
            joint_position: Current joint position(s)
            
        Returns:
            Clamped joint position(s)
        """
        if self.limits is None:
            return joint_position
        return torch.clamp(joint_position, self.limits[0], self.limits[1])

    def __repr__(self) -> str:
        has_limits = self.limits is not None
        return (f"Joint(name='{self.name}', "
                f"type='{self.joint_type}', "
                f"axis={self.axis.tolist()}, "
                f"has_limits={has_limits})")