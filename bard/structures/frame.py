"""Frame representation for kinematic tree nodes."""

from typing import List, Optional
import torch
import pytorch_kinematics.transforms as tf
from pytorch_kinematics.transforms import axis_and_angle_to_matrix_33

from .joint import Joint
from .link import Link


# Tree drawing characters
SPACE = '    '
BRANCH = '│   '
TEE = '├── '
LAST = '└── '


class Frame:
    """A node in the robot kinematic tree, containing joint and link information.

    A frame represents a coordinate system attached to a part of the robot.
    The kinematic tree is built by connecting these frames in parent-child
    relationships, forming a chain from the root to the end-effectors.

    Attributes:
        name (str): The unique identifier for this frame.
        link (Link): The link object associated with this frame, containing
            physical properties like mass and inertia.
        joint (Joint): The joint that connects this frame to its parent,
            defining the motion between them.
        children (List[Frame]): A list of child frames attached to this frame.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        link: Optional[Link] = None,
        joint: Optional[Joint] = None,
        children: Optional[List['Frame']] = None
    ):
        """Initializes a kinematic frame.
        
        Args:
            name (Optional[str], optional): The name/identifier for the frame.
                Defaults to None.
            link (Optional[Link], optional): The link object for this frame.
                Defaults to an empty Link.
            joint (Optional[Joint], optional): The joint connecting this frame to
                its parent. Defaults to a fixed Joint.
            children (Optional[List[Frame]], optional): A list of child frames.
                Defaults to an empty list.
        """
        self.name = 'None' if name is None else name
        self.link = link if link is not None else Link()
        self.joint = joint if joint is not None else Joint()
        self.children = children if children is not None else []

    def __str__(self, prefix: str = '', root: bool = True) -> str:
        """Generates a string representation of the kinematic tree from this frame down."""
        pointers = [TEE] * (len(self.children) - 1) + [LAST]
        
        if root:
            result = prefix + self.name + "\n"
        else:
            result = ""
        
        for pointer, child in zip(pointers, self.children):
            result += prefix + pointer + child.name + "\n"
            if child.children:
                extension = BRANCH if pointer == TEE else SPACE
                result += child.__str__(prefix=prefix + extension, root=False)
        
        return result

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> 'Frame':
        """Moves all tensor data in the frame and its children to a specified device/dtype.

        Args:
            dtype (Optional[torch.dtype], optional): The target data type. Defaults to None.
            device (Optional[torch.device], optional): The target device. Defaults to None.

        Returns:
            Frame: Returns self for method chaining.
        """
        self.joint = self.joint.to(dtype=dtype, device=device)
        self.link = self.link.to(dtype=dtype, device=device)
        self.children = [c.to(dtype=dtype, device=device) for c in self.children]
        return self

    def add_child(self, child: 'Frame') -> None:
        """Adds a child frame to this frame's list of children.
        
        Args:
            child (Frame): The child frame to add.
        """
        self.children.append(child)

    def is_end_effector(self) -> bool:
        """Checks if this frame is a leaf node in the kinematic tree.
        
        Returns:
            bool: True if the frame has no children, otherwise False.
        """
        return len(self.children) == 0

    def get_transform(self, theta: torch.Tensor) -> tf.Transform3d:
        """Computes the transformation for the joint associated with this frame.

        This transform represents the motion of the joint given a joint angle `theta`.
        It is composed with the joint's static offset.
        
        Args:
            theta (torch.Tensor): The joint position(s). Can be a scalar or a
                batched tensor of shape (B,).
            
        Returns:
            tf.Transform3d: A Transform3d object representing the batched joint motion.
        """
        dtype = self.joint.axis.dtype
        device = self.joint.axis.device
        
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=dtype, device=device)
        
        if theta.ndim == 0:
            theta = theta.unsqueeze(0)
        
        batch_size = theta.shape[0]
        
        if self.joint.joint_type == 'revolute':
            rot = axis_and_angle_to_matrix_33(self.joint.axis, theta)
            transform = tf.Transform3d(rot=rot, dtype=dtype, device=device)
        elif self.joint.joint_type == 'prismatic':
            pos = theta.unsqueeze(1) * self.joint.axis
            transform = tf.Transform3d(pos=pos, dtype=dtype, device=device)
        elif self.joint.joint_type == 'fixed':
            transform = tf.Transform3d(default_batch_size=batch_size, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unsupported joint type: {self.joint.joint_type}")
        
        if self.joint.offset is None:
            return transform
        else:
            return self.joint.offset.compose(transform)

    def find_by_name(self, target_name: str) -> Optional['Frame']:
        """Finds a frame by name within this frame's subtree.
        
        Args:
            target_name (str): The name of the frame to find.
            
        Returns:
            Optional['Frame']: The Frame object if found, otherwise None.
        """
        if self.name == target_name:
            return self
        
        for child in self.children:
            result = child.find_by_name(target_name)
            if result is not None:
                return result
        
        return None

    def get_all_frames(self) -> List['Frame']:
        """Returns a list of all frames in the subtree starting from this frame.
        
        Returns:
            List[Frame]: A list of all frames in depth-first order.
        """
        frames = [self]
        for child in self.children:
            frames.extend(child.get_all_frames())
        return frames

    def count_joints(self, exclude_fixed: bool = True) -> int:
        """Counts the number of joints in the subtree starting from this frame.
        
        Args:
            exclude_fixed (bool, optional): If True, fixed joints are not
                counted. Defaults to True.
            
        Returns:
            int: The total number of joints.
        """
        count = 0
        if not (exclude_fixed and self.joint.joint_type == 'fixed'):
            count = 1
        
        for child in self.children:
            count += child.count_joints(exclude_fixed=exclude_fixed)
        
        return count

    def __repr__(self) -> str:
        """Returns a concise string representation of the Frame."""
        num_children = len(self.children)
        return (f"Frame(name='{self.name}', "
                f"joint_type='{self.joint.joint_type}', "
                f"num_children={num_children})")