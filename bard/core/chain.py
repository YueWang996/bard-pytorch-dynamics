from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch

import bard.transforms as tf
from bard.structures import Frame, Joint


class Chain:
    """
    Robot kinematic chain supporting both fixed-base and floating-base robots.
    
    This class represents a robot's kinematic structure as a tree of frames,
    where each frame contains joint and link information. It supports vectorized
    batch operations for efficient parallel computation.
    
    Coordinate Conventions:
        Fixed-base robots:
            - q: [joint_angles...] (nq = n_joints)
            - v: [joint_velocities...] (nv = n_joints)
        
        Floating-base robots:
            - q: [tx, ty, tz, qw, qx, qy, qz, joint_angles...] (nq = 7 + n_joints)
            - v: [vx, vy, vz, wx, wy, wz, joint_velocities...] (nv = 6 + n_joints)
            
            where base pose uses unit quaternion [qw, qx, qy, qz] for orientation.
    
    Attributes:
        has_floating_base: Whether robot has a floating base
        nq: Total configuration space dimension
        nv: Total velocity space dimension
        n_joints: Number of actuated joints
        dtype: PyTorch data type
        device: PyTorch device
    
    Example:
        >>> chain = Chain(root_frame, floating_base=False)
        >>> q = torch.randn(batch_size, chain.nq)
        >>> pose = calc_forward_kinematics(chain, q, frame_id)
    """

    def __init__(
        self,
        root_frame: Frame,
        floating_base: bool = False,
        base_name: str = "base",
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Initialize kinematic chain from root frame.
        
        Args:
            root_frame: Root frame of the kinematic tree
            floating_base: If True, robot has 6-DOF floating base
            base_name: Name prefix for base coordinates
            dtype: PyTorch dtype for tensors
            device: PyTorch device for tensors
        """
        self._root = root_frame
        self.dtype = dtype
        self.device = device
        
        # Floating base configuration
        self.has_floating_base = bool(floating_base)
        self.base_name = base_name
        self.nq_base = 7 if self.has_floating_base else 0  # [xyz, qwxyz]
        self.nv_base = 6 if self.has_floating_base else 0  # [vxyz, wxyz]

        # Build tree structure
        self._build_tree_structure()
        
        # Joint limits
        low, high = self._get_joint_limits("limits")
        self.low = torch.tensor(low, device=self.device, dtype=self.dtype)
        self.high = torch.tensor(high, device=self.device, dtype=self.dtype)

    def _build_tree_structure(self) -> None:
        """Build flat representation of kinematic tree via depth-first traversal."""
        self.parents_indices: List[torch.Tensor] = []
        self.joint_indices: torch.Tensor = None
        self.joint_type_indices: torch.Tensor = None
        self.n_joints = len(self.get_joint_parameter_names())
        self.axes = torch.zeros([self.n_joints, 3], dtype=self.dtype, device=self.device)
        self.link_offsets: List[Optional[torch.Tensor]] = []
        self.joint_offsets: List[Optional[torch.Tensor]] = []
        self.frame_to_idx: Dict[str, int] = {}
        self.idx_to_frame: Dict[int, str] = {}
        
        # Temporary lists for building
        joint_indices_list = []
        joint_type_indices_list = []
        
        queue = [(self._root, -1, 0)]
        idx = 0
        
        while queue:
            frame, parent_idx, depth = queue.pop(0)
            name = frame.name.strip("\n")
            
            self.frame_to_idx[name] = idx
            self.idx_to_frame[idx] = name
            
            # Build path from root
            if parent_idx == -1:
                self.parents_indices.append([idx])
            else:
                self.parents_indices.append(self.parents_indices[parent_idx] + [idx])
            
            # Store offsets
            self.link_offsets.append(
                frame.link.offset.get_matrix() if frame.link.offset else None
            )
            self.joint_offsets.append(
                frame.joint.offset.get_matrix() if frame.joint.offset else None
            )
            
            # Joint info
            is_fixed = frame.joint.joint_type == 'fixed'
            if is_fixed:
                joint_indices_list.append(-1)
            else:
                jnt_idx = self.get_joint_parameter_names().index(frame.joint.name)
                self.axes[jnt_idx] = frame.joint.axis
                joint_indices_list.append(jnt_idx)
            
            joint_type_indices_list.append(Joint.TYPES.index(frame.joint.joint_type))
            
            # Queue children
            for child in frame.children:
                queue.append((child, idx, depth + 1))
            
            idx += 1
        
        # Convert to tensors
        self.joint_type_indices = torch.tensor(
            joint_type_indices_list, dtype=torch.long, device=self.device
        )
        self.joint_indices = torch.tensor(
            joint_indices_list, dtype=torch.long, device=self.device
        )
        self.parents_indices = [
            torch.tensor(p, dtype=torch.long, device=self.device) 
            for p in self.parents_indices
        ]

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None
    ) -> "Chain":
        """
        Move chain to different dtype and/or device.
        
        Args:
            dtype: Target PyTorch dtype
            device: Target PyTorch device
            
        Returns:
            Self for method chaining
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
            
        self._root = self._root.to(dtype=self.dtype, device=self.device)
        self.parents_indices = [p.to(device=self.device) for p in self.parents_indices]
        self.joint_type_indices = self.joint_type_indices.to(device=self.device)
        self.joint_indices = self.joint_indices.to(device=self.device)
        self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        self.link_offsets = [
            l.to(dtype=self.dtype, device=self.device) if l is not None else None 
            for l in self.link_offsets
        ]
        self.joint_offsets = [
            j.to(dtype=self.dtype, device=self.device) if j is not None else None 
            for j in self.joint_offsets
        ]
        self.low = self.low.to(dtype=self.dtype, device=self.device)
        self.high = self.high.to(dtype=self.dtype, device=self.device)
        return self

    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def nq(self) -> int:
        """Total configuration space dimension (base + joints)."""
        return self.nq_base + self.n_joints

    @property
    def nv(self) -> int:
        """Total velocity space dimension (base + joints)."""
        return self.nv_base + self.n_joints

    # ========================================================================
    # Coordinate naming
    # ========================================================================
    
    def get_generalized_coordinate_names(self, include_base: bool = True) -> List[str]:
        """
        Get names for all configuration variables.
        
        Args:
            include_base: Whether to include base coordinates
            
        Returns:
            List of coordinate names in order
        """
        base_names = []
        if include_base and self.has_floating_base:
            base_names = [
                f"{self.base_name}_tx", f"{self.base_name}_ty", f"{self.base_name}_tz",
                f"{self.base_name}_qw", f"{self.base_name}_qx", 
                f"{self.base_name}_qy", f"{self.base_name}_qz"
            ]
        return base_names + self.get_joint_parameter_names()

    def get_generalized_velocity_names(self, include_base: bool = True) -> List[str]:
        """
        Get names for all velocity variables.
        
        Args:
            include_base: Whether to include base velocities
            
        Returns:
            List of velocity names in order
        """
        base_names = []
        if include_base and self.has_floating_base:
            base_names = [
                f"{self.base_name}_vx", f"{self.base_name}_vy", f"{self.base_name}_vz",
                f"{self.base_name}_wx", f"{self.base_name}_wy", f"{self.base_name}_wz"
            ]
        
        joint_vel_names = [
            n.replace(":position", ":velocity") if ":position" in n else f"{n}_d"
            for n in self.get_joint_parameter_names()
        ]
        return base_names + joint_vel_names

    # ========================================================================
    # Pack/unpack helpers
    # ========================================================================
    
    def unpack_q(
        self, q: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Split generalized coordinates into base and joint components.
        
        Args:
            q: Generalized coordinates (..., nq)
            
        Returns:
            (q_base, q_joints) where q_base is None for fixed-base robots
        """
        q = self.ensure_tensor(q)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if not self.has_floating_base:
            return None, q
        return q[:, :7], q[:, 7:]

    def unpack_v(
        self, v: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Split generalized velocities into base and joint components.
        
        Args:
            v: Generalized velocities (..., nv)
            
        Returns:
            (v_base, v_joints) where v_base is None for fixed-base robots
        """
        v = self.ensure_tensor(v)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        if not self.has_floating_base:
            return None, v
        return v[:, :6], v[:, 6:]

    def pack_q(
        self,
        q_base: Optional[torch.Tensor],
        q_joints: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine base and joint positions into generalized coordinates.
        
        Args:
            q_base: Base position [tx, ty, tz, qw, qx, qy, qz] or None
            q_joints: Joint positions
            
        Returns:
            Generalized coordinates
        """
        if not self.has_floating_base:
            return q_joints
        if q_joints.ndim == 1:
            q_joints = q_joints.unsqueeze(0)
        if q_base.ndim == 1:
            q_base = q_base.unsqueeze(0)
        return torch.cat([q_base, q_joints], dim=-1)

    def pack_v(
        self,
        v_base: Optional[torch.Tensor],
        v_joints: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine base and joint velocities into generalized velocities.
        
        Args:
            v_base: Base velocity [vx, vy, vz, wx, wy, wz] or None
            v_joints: Joint velocities
            
        Returns:
            Generalized velocities
        """
        if not self.has_floating_base:
            return v_joints
        if v_joints.ndim == 1:
            v_joints = v_joints.unsqueeze(0)
        if v_base.ndim == 1:
            v_base = v_base.unsqueeze(0)
        return torch.cat([v_base, v_joints], dim=-1)

    # ========================================================================
    # Joint/frame queries
    # ========================================================================
    
    @staticmethod
    def _get_joints(frame: Frame, exclude_fixed: bool = True) -> List[Joint]:
        """Recursively collect all joints in tree."""
        joints = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joints.append(frame.joint)
        for child in frame.children:
            joints.extend(Chain._get_joints(child, exclude_fixed))
        return joints

    def get_joints(self, exclude_fixed: bool = True) -> List[Joint]:
        """Get all joints in the chain."""
        return self._get_joints(self._root, exclude_fixed=exclude_fixed)

    @lru_cache()
    def get_joint_parameter_names(self, exclude_fixed: bool = True) -> List[str]:
        """Get ordered list of joint parameter names."""
        return [j.name for j in self.get_joints(exclude_fixed=exclude_fixed)]

    def find_frame(self, name: str) -> Optional[Frame]:
        """
        Find frame by name in the kinematic tree.
        
        Args:
            name: Frame name to search for
            
        Returns:
            Frame object or None if not found
        """
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_frame_recursive(name: str, frame: Frame) -> Optional[Frame]:
        """Recursively search for frame by name."""
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if ret is not None:
                return ret
        return None

    def get_frame_names(self, exclude_fixed: bool = True) -> List[str]:
        """Get all frame names in the chain."""
        names = self._get_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_frame_names(frame: Frame, exclude_fixed: bool = True) -> List[str]:
        """Recursively collect frame names."""
        names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            names.append(frame.name)
        for child in frame.children:
            names.extend(Chain._get_frame_names(child, exclude_fixed))
        return names

    @lru_cache
    def get_frame_indices(self, *frame_names: str) -> torch.Tensor:
        """
        Get frame indices for given frame names.
        
        Args:
            *frame_names: Names of frames
            
        Returns:
            Tensor of frame indices
        """
        return torch.tensor(
            [self.frame_to_idx[n] for n in frame_names], 
            dtype=torch.long, device=self.device
        )

    # ========================================================================
    # Utilities
    # ========================================================================
    
    def ensure_tensor(
        self, 
        value: Union[torch.Tensor, np.ndarray, List, Dict]
    ) -> torch.Tensor:
        """
        Convert various input types to tensor in joint parameter order.
        
        Args:
            value: Input as tensor, array, list, or dict mapping joint names to values
            
        Returns:
            Tensor on correct device/dtype
            
        Raises:
            ValueError: If dict is missing required joints
            TypeError: If input type is not supported
        """
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        elif isinstance(value, np.ndarray):
            return torch.tensor(value, device=self.device, dtype=self.dtype)
        elif isinstance(value, list):
            return torch.tensor(value, device=self.device, dtype=self.dtype)
        elif isinstance(value, dict):
            return self._dict_to_tensor(value)
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def _dict_to_tensor(self, value_dict: Dict[str, float]) -> torch.Tensor:
        """Convert joint name dictionary to ordered tensor."""
        elem_shape = self._get_dict_elem_shape(value_dict)
        tensor = torch.full(
            [*elem_shape, self.n_joints], 
            torch.nan, 
            device=self.device, 
            dtype=self.dtype
        )
        joint_names = self.get_joint_parameter_names()
        for joint_name, val in value_dict.items():
            jnt_idx = joint_names.index(joint_name)
            tensor[..., jnt_idx] = val
        
        if torch.any(torch.isnan(tensor)):
            missing = [n for n, v in zip(joint_names, tensor[0]) if torch.isnan(v)]
            raise ValueError(f"Missing values for joints: {missing}")
        return tensor

    @staticmethod
    def _get_dict_elem_shape(value_dict: Dict) -> tuple:
        """Get shape of dictionary values."""
        elem = next(iter(value_dict.values()))
        if isinstance(elem, (np.ndarray, torch.Tensor)):
            return elem.shape
        return ()

    def clamp(self, joint_values: torch.Tensor) -> torch.Tensor:
        """
        Clamp joint values to joint limits.
        
        Args:
            joint_values: Joint values to clamp
            
        Returns:
            Clamped joint values
        """
        joint_values = self.ensure_tensor(joint_values)
        return torch.clamp(joint_values, self.low, self.high)

    def get_joint_limits(self) -> tuple[List[float], List[float]]:
        """Get position limits for all joints."""
        return self._get_joint_limits("limits")

    def get_joint_velocity_limits(self) -> tuple[List[float], List[float]]:
        """Get velocity limits for all joints."""
        return self._get_joint_limits("velocity_limits")

    def get_joint_effort_limits(self) -> tuple[List[float], List[float]]:
        """Get effort (torque/force) limits for all joints."""
        return self._get_joint_limits("effort_limits")

    def _get_joint_limits(
        self, param_name: str
    ) -> tuple[List[float], List[float]]:
        """Extract joint limit parameter from all joints."""
        low, high = [], []
        for joint in self.get_joints():
            val = getattr(joint, param_name)
            if val is None:
                low.append(-np.inf)
                high.append(np.inf)
            else:
                low.append(val[0])
                high.append(val[1])
        return low, high

    def __str__(self) -> str:
        """String representation showing kinematic tree."""
        return str(self._root)