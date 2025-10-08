"""Joint representation for robot kinematic chains."""

from typing import Optional, Tuple
import torch


class Joint:
    """Represents a robot joint, defining the motion between two links.

    This class stores the properties of a single joint, such as its type (revolute,
    prismatic, fixed), axis of motion, and physical limits.

    Attributes:
        name (str): The unique identifier for the joint.
        offset (Transform3d): The static transform from the parent link's frame
            to the joint's frame.
        joint_type (str): The type of joint, one of 'fixed', 'revolute', 'prismatic'.
        axis (torch.Tensor): A 3D unit vector representing the axis of motion in the
            joint's own frame.
        limits (Optional[Tuple[float, float]]): A tuple of (lower, upper)
            position limits in radians or meters. None if unbounded.
        velocity_limits (Optional[Tuple[float, float]]): A tuple of (lower, upper)
            velocity limits. None if unbounded.
        effort_limits (Optional[Tuple[float, float]]): A tuple of (lower, upper)
            force or torque limits. None if unbounded.
    """

    TYPES = ["fixed", "revolute", "prismatic"]

    def __init__(
        self,
        name: Optional[str] = None,
        offset: Optional[object] = None,
        joint_type: str = "fixed",
        axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        limits: Optional[Tuple[float, float]] = None,
        velocity_limits: Optional[Tuple[float, float]] = None,
        effort_limits: Optional[Tuple[float, float]] = None,
    ):
        """Initializes a Joint object.

        Args:
            name (Optional[str], optional): The name/identifier of the joint.
                Defaults to "unnamed_joint".
            offset (Optional[object], optional): A Transform3d object representing
                the static offset from the parent link frame. Defaults to None.
            joint_type (str, optional): The type of joint. Must be one of
                'fixed', 'revolute', or 'prismatic'. Defaults to 'fixed'.
            axis (Tuple[float, float, float], optional): The joint's axis of
                motion as an (x, y, z) tuple. Defaults to (0.0, 0.0, 1.0).
            dtype (torch.dtype, optional): The PyTorch data type for the axis tensor.
                Defaults to torch.float32.
            device (str, optional): The PyTorch device for the axis tensor.
                Defaults to "cpu".
            limits (Optional[Tuple[float, float]], optional): Position limits (min, max)
                in radians or meters. Defaults to None.
            velocity_limits (Optional[Tuple[float, float]], optional): Velocity limits.
                Defaults to None.
            effort_limits (Optional[Tuple[float, float]], optional): Effort (force/torque)
                limits. Defaults to None.

        Raises:
            RuntimeError: If `joint_type` is not one of the supported types.
        """
        self.name = name if name is not None else "unnamed_joint"
        self.offset = offset

        if joint_type not in self.TYPES:
            raise RuntimeError(
                f"Joint '{name}' has type '{joint_type}', " f"but only {self.TYPES} are supported"
            )
        self.joint_type = joint_type

        if axis is None:
            self.axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        else:
            if torch.is_tensor(axis):
                self.axis = axis.clone().detach().to(dtype=dtype, device=device)
            else:
                self.axis = torch.tensor(axis, dtype=dtype, device=device)

        axis_norm = self.axis.norm()
        if axis_norm > 1e-12:
            self.axis = self.axis / axis_norm

        self.limits = limits
        self.velocity_limits = velocity_limits
        self.effort_limits = effort_limits

    def to(
        self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ) -> "Joint":
        """Moves all tensor data in the joint to a specified device/dtype.

        Args:
            dtype (Optional[torch.dtype], optional): The target data type.
            device (Optional[torch.device], optional): The target device.

        Returns:
            Joint: Returns self for method chaining.
        """
        if dtype is not None or device is not None:
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            if device is not None:
                kwargs["device"] = device
            self.axis = self.axis.to(**kwargs)

        if self.offset is not None:
            self.offset = self.offset.to(dtype=dtype, device=device)

        return self

    def clamp(self, joint_position: torch.Tensor) -> torch.Tensor:
        """Clamps a given joint position to the joint's defined limits.

        If the joint has no limits, the input position is returned unchanged.

        Args:
            joint_position (torch.Tensor): The current joint position(s) to clamp.

        Returns:
            torch.Tensor: The clamped joint position(s).
        """
        if self.limits is None:
            return joint_position
        return torch.clamp(joint_position, self.limits[0], self.limits[1])

    def __repr__(self) -> str:
        """Returns a concise string representation of the Joint."""
        has_limits = self.limits is not None
        return (
            f"Joint(name='{self.name}', "
            f"type='{self.joint_type}', "
            f"axis={self.axis.tolist()}, "
            f"has_limits={has_limits})"
        )
