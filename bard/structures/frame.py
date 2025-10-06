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
    """
    Node in robot kinematic tree containing joint and link information.
    
    A frame represents a coordinate system in the robot. The tree structure
    is built by connecting frames via parent-child relationships.
    
    Attributes:
        name: Frame identifier
        link: Link object with mass/inertia properties
        joint: Joint connecting this frame to parent
        children: List of child frames
    
    Tree Structure:
        root_frame
        ├── child1
        │   ├── grandchild1
        │   └── grandchild2
        └── child2
    """

    def __init__(
        self,
        name: Optional[str] = None,
        link: Optional[Link] = None,
        joint: Optional[Joint] = None,
        children: Optional[List['Frame']] = None
    ):
        """
        Initialize kinematic frame.
        
        Args:
            name: Frame name/identifier
            link: Link object for this frame
            joint: Joint connecting to parent frame
            children: List of child frames
        """
        self.name = 'None' if name is None else name
        self.link = link if link is not None else Link()
        self.joint = joint if joint is not None else Joint()
        self.children = children if children is not None else []

    def __str__(self, prefix: str = '', root: bool = True) -> str:
        """
        Pretty-print kinematic tree structure.
        
        Args:
            prefix: Indentation prefix for current level
            root: Whether this is the root frame
            
        Returns:
            String representation of tree
        """
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
        """
        Move frame data to specified dtype/device.
        
        Args:
            dtype: Target data type
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.joint = self.joint.to(dtype=dtype, device=device)
        self.link = self.link.to(dtype=dtype, device=device)
        self.children = [c.to(dtype=dtype, device=device) for c in self.children]
        return self

    def add_child(self, child: 'Frame') -> None:
        """
        Add child frame to this frame.
        
        Args:
            child: Child frame to add
        """
        self.children.append(child)

    def is_end_effector(self) -> bool:
        """
        Check if this frame is a leaf node (end-effector).
        
        Returns:
            True if frame has no children
        """
        return len(self.children) == 0

    def get_transform(self, theta: torch.Tensor) -> tf.Transform3d:
        """
        Compute joint transform for given joint position(s).
        
        Args:
            theta: Joint position(s), shape (B,) or scalar
            
        Returns:
            Transform3d representing joint motion
            
        Note:
            For batched theta, returns batched transforms.
        """
        dtype = self.joint.axis.dtype
        device = self.joint.axis.device
        
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=dtype, device=device)
        
        if theta.ndim == 0:
            theta = theta.unsqueeze(0)
        
        batch_size = theta.shape[0]
        
        if self.joint.joint_type == 'revolute':
            # Rotation about joint axis
            rot = axis_and_angle_to_matrix_33(self.joint.axis, theta)
            transform = tf.Transform3d(rot=rot, dtype=dtype, device=device)
            
        elif self.joint.joint_type == 'prismatic':
            # Translation along joint axis
            pos = theta.unsqueeze(1) * self.joint.axis
            transform = tf.Transform3d(pos=pos, dtype=dtype, device=device)
            
        elif self.joint.joint_type == 'fixed':
            # Identity transform
            transform = tf.Transform3d(
                default_batch_size=batch_size,
                dtype=dtype,
                device=device
            )
        else:
            raise ValueError(f"Unsupported joint type: {self.joint.joint_type}")
        
        # Compose with joint offset if present
        if self.joint.offset is None:
            return transform
        else:
            return self.joint.offset.compose(transform)

    def find_by_name(self, target_name: str) -> Optional['Frame']:
        """
        Find frame by name in subtree.
        
        Args:
            target_name: Name of frame to find
            
        Returns:
            Frame object if found, None otherwise
        """
        if self.name == target_name:
            return self
        
        for child in self.children:
            result = child.find_by_name(target_name)
            if result is not None:
                return result
        
        return None

    def get_all_frames(self) -> List['Frame']:
        """
        Get all frames in subtree (depth-first order).
        
        Returns:
            List of all frames including self
        """
        frames = [self]
        for child in self.children:
            frames.extend(child.get_all_frames())
        return frames

    def count_joints(self, exclude_fixed: bool = True) -> int:
        """
        Count joints in subtree.
        
        Args:
            exclude_fixed: Whether to exclude fixed joints
            
        Returns:
            Number of joints
        """
        count = 0
        if not (exclude_fixed and self.joint.joint_type == 'fixed'):
            count = 1
        
        for child in self.children:
            count += child.count_joints(exclude_fixed=exclude_fixed)
        
        return count

    def __repr__(self) -> str:
        num_children = len(self.children)
        return (f"Frame(name='{self.name}', "
                f"joint_type='{self.joint.joint_type}', "
                f"num_children={num_children})")