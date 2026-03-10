"""
Defines the core kinematic chain data structure for representing robots.

This module contains the ``Chain`` class, which is the central object for all
kinematics and dynamics calculations in the ``bard`` library. It processes a
robot's structure, typically parsed from a URDF file, into an efficient,
indexed, and tensor-based representation. This class manages the robot's
topology, joint properties, and tensor attributes like data type and device,
providing a unified interface for all other algorithm classes.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch

import bard.transforms as tf
from bard.structures import Frame, Joint


class Chain:
    """Represents a robot's kinematic structure as a tree of frames.

    This class is the central object for all kinematics and dynamics calculations.
    It holds the robot's structure, parsed from a URDF, and provides methods
    for querying frames, joints, and managing data types and devices. It supports
    both fixed-base and floating-base robots for vectorized batch operations.

    **Coordinate Conventions:**

    * **Fixed-base robots:**
        * Configuration ``q``: ``[joint_angles...]`` (``nq = n_joints``)
        * Velocity ``v``: ``[joint_velocities...]`` (``nv = n_joints``)

    * **Floating-base robots:**
        * Configuration ``q``: ``[tx, ty, tz, qw, qx, qy, qz, joint_angles...]`` (``nq = 7 + n_joints``)
        * Velocity ``v``: ``[vx, vy, vz, wx, wy, wz, joint_velocities...]`` (``nv = 6 + n_joints``)

    The base orientation is represented by a unit quaternion ``[qw, qx, qy, qz]``.

    Attributes:
        has_floating_base (bool): ``True`` if the robot has a 6-DOF floating base.
        nq (int): The total dimension of the configuration space vector ``q``.
        nv (int): The total dimension of the velocity space vector ``v``.
        n_joints (int): The number of actuated (non-fixed) joints.
        n_nodes (int): The total number of frames (links) in the kinematic tree.
        dtype (torch.dtype): The PyTorch data type used for all tensors.
        device (torch.device): The PyTorch device (e.g., "cpu" or "cuda") for all tensors.
        low (torch.Tensor): A tensor of lower position limits for all actuated joints.
        high (torch.Tensor): A tensor of upper position limits for all actuated joints.
    """

    def __init__(
        self,
        root_frame: Frame,
        floating_base: bool = False,
        base_name: str = "base",
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initializes a kinematic chain from a root frame.

        Args:
            root_frame (Frame): The root frame of the kinematic tree structure,
                typically obtained from a URDF parser.
            floating_base (bool, optional): If ``True``, the robot is treated as having
                a 6-DOF floating base. Defaults to ``False``.
            base_name (str, optional): A name prefix used for base coordinates
                in methods like `get_generalized_coordinate_names`. Defaults to "base".
            dtype (torch.dtype, optional): The default PyTorch data type for
                all tensors in the chain. Defaults to ``torch.float32``.
            device (Union[str, torch.device], optional): The default PyTorch device
                for all tensors in the chain. Defaults to "cpu".
        """
        self._root = root_frame
        self.dtype = dtype
        self.device = device

        # Floating base configuration
        self.has_floating_base = bool(floating_base)
        self.base_name = base_name
        self.nq_base = 7 if self.has_floating_base else 0  # [xyz, qwxyz]
        self.nv_base = 6 if self.has_floating_base else 0  # [vxyz, wxyz]

        # Build tree structure and precompute properties
        self._build_tree_structure()

        # Precompute joint limits
        low, high = self._get_joint_limits("limits")
        self.low = torch.tensor(low, device=self.device, dtype=self.dtype)
        self.high = torch.tensor(high, device=self.device, dtype=self.dtype)

    def _build_tree_structure(self) -> None:
        """Builds a flat, indexed representation of the kinematic tree.

        This internal method traverses the kinematic tree starting from the root frame.
        It populates several pre-computed attributes, including parent-child
        relationships, topological order, joint axes, and spatial inertias. It also
        creates Python list versions of these attributes for high-speed, zero-overhead
        access within compiled dynamics and kinematics functions.
        """
        self.parents_indices: List[torch.Tensor] = []
        self.joint_indices: torch.Tensor = None
        self.joint_type_indices: torch.Tensor = None
        self.n_joints = len(self.get_joint_parameter_names())
        self.axes = torch.zeros([self.n_joints, 3], dtype=self.dtype, device=self.device)
        self.link_offsets: List[Optional[torch.Tensor]] = []
        self.joint_offsets: List[Optional[torch.Tensor]] = []
        self.frame_to_idx: Dict[str, int] = {}
        self.idx_to_frame: Dict[int, str] = {}

        joint_indices_list = []
        joint_type_indices_list = []

        queue = [(self._root, -1, 0)]
        idx = 0

        while queue:
            frame, parent_idx, depth = queue.pop(0)
            name = frame.name.strip("\n")

            self.frame_to_idx[name] = idx
            self.idx_to_frame[idx] = name

            if parent_idx == -1:
                self.parents_indices.append([idx])
            else:
                self.parents_indices.append(self.parents_indices[parent_idx] + [idx])

            self.link_offsets.append(frame.link.offset.get_matrix() if frame.link.offset else None)
            self.joint_offsets.append(
                frame.joint.offset.get_matrix() if frame.joint.offset else None
            )

            is_fixed = frame.joint.joint_type == "fixed"
            if is_fixed:
                joint_indices_list.append(-1)
            else:
                jnt_idx = self.get_joint_parameter_names().index(frame.joint.name)
                self.axes[jnt_idx] = frame.joint.axis
                joint_indices_list.append(jnt_idx)

            joint_type_indices_list.append(Joint.TYPES.index(frame.joint.joint_type))

            for child in frame.children:
                queue.append((child, idx, depth + 1))

            idx += 1

        n_nodes = idx
        self.n_nodes = n_nodes

        self.joint_type_indices = torch.tensor(
            joint_type_indices_list, dtype=torch.long, device=self.device
        )
        self.joint_indices = torch.tensor(joint_indices_list, dtype=torch.long, device=self.device)
        self.parents_indices = [
            torch.tensor(p, dtype=torch.long, device=self.device) for p in self.parents_indices
        ]

        # Pre-compute parent-child relationships as tensors
        self.parent_array = torch.full((n_nodes,), -1, dtype=torch.long, device=self.device)
        children_lists = [[] for _ in range(n_nodes)]

        for node in range(n_nodes):
            path = self.parents_indices[node]
            if len(path) > 1:
                p = int(path[-2].item())  # Do .item() ONCE during init
                self.parent_array[node] = p
                children_lists[p].append(node)

        # Calculate max_children AFTER populating children_lists
        max_children = max((len(children_lists[i]) for i in range(n_nodes)), default=0)

        # Handle edge case: if no node has children, still need at least size 1 to avoid empty tensor
        if max_children == 0:
            max_children = 1

        self.children_array = torch.full(
            (n_nodes, max_children), -1, dtype=torch.long, device=self.device
        )
        self.children_count = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

        for node in range(n_nodes):
            for child_idx, child in enumerate(children_lists[node]):
                self.children_array[node, child_idx] = child
                self.children_count[node] += 1

        # Pre-compute topological order
        self.topo_order = []
        stack = [i for i, p in enumerate(self.parent_array.tolist()) if p == -1]
        while stack:
            node_idx = stack.pop(0)
            self.topo_order.append(node_idx)
            child_count = int(self.children_count[node_idx].item())
            stack.extend(self.children_array[node_idx, :child_count].tolist())

        # Pre-compute spatial inertias for ALL links (batched)
        self.spatial_inertias = self._precompute_all_spatial_inertias(n_nodes)

        # NEW: Pre-compute Python lists for zero-overhead access in dynamics functions
        self.parent_list = self.parent_array.tolist()
        self.children_list = children_lists  # Already Python lists
        self.joint_indices_list = self.joint_indices.tolist()
        self.joint_type_indices_list = self.joint_type_indices.tolist()

        # NEW: Pre-compute parents_indices as Python lists for jacobian/kinematics
        self.parents_indices_list = [path.tolist() for path in self.parents_indices]

    def _precompute_all_spatial_inertias(self, n_nodes: int) -> torch.Tensor:
        """Pre-computes spatial inertia matrices for all links.

        This method iterates through all nodes (frames) in the chain and computes
        their 6x6 spatial inertia matrix in the local link frame. This is done
        once at initialization to avoid repeated calculations.

        Args:
            n_nodes (int): The total number of nodes in the chain.

        Returns:
            torch.Tensor: A tensor of shape ``(n_nodes, 6, 6)`` containing the
            spatial inertia matrix for each link.
        """

        def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
            x, y, z = v[..., 0], v[..., 1], v[..., 2]
            zeros = torch.zeros_like(x)
            return torch.stack(
                [
                    torch.stack([zeros, -z, y], dim=-1),
                    torch.stack([z, zeros, -x], dim=-1),
                    torch.stack([-y, x, zeros], dim=-1),
                ],
                dim=-2,
            )

        # Always compute spatial inertias in float64 to preserve URDF precision,
        # then cast to self.dtype at the end. This avoids float32 truncation of
        # parsed inertia parameters that would otherwise cause ~1e-6 errors in
        # dynamics algorithms (RNEA, ABA) when compared to Pinocchio.
        compute_dtype = torch.float64

        I_spatial = torch.zeros((n_nodes, 6, 6), dtype=compute_dtype, device=self.device)

        for node_idx in range(n_nodes):
            frame_name = self.idx_to_frame[node_idx]
            frame_obj = self.find_frame(frame_name)
            link = frame_obj.link
            inertial = getattr(link, "inertial", None)

            if inertial is None:
                continue

            offset_transform, mass, inertia_tensor = inertial

            # Extract COM pose
            if offset_transform is None:
                R = torch.eye(3, dtype=compute_dtype, device=self.device)
                com_pos = torch.zeros(3, dtype=compute_dtype, device=self.device)
            else:
                T = offset_transform.get_matrix()
                # Ensure T is 2D (4x4) by squeezing any batch dimensions
                if T.ndim > 2:
                    T = T.squeeze()
                T = T.to(dtype=compute_dtype, device=self.device)
                R = T[:3, :3].clone()  # Extract rotation, ensure contiguous
                com_pos = T[:3, 3].clone()  # Extract position

            # Convert mass to tensor ONCE
            if torch.is_tensor(mass):
                m = mass.to(dtype=compute_dtype, device=self.device)
            else:
                m = torch.tensor(mass, dtype=compute_dtype, device=self.device)

            # Rotational inertia
            if inertia_tensor is None:
                I_rot = torch.zeros((3, 3), dtype=compute_dtype, device=self.device)
            else:
                I_rot = inertia_tensor.clone()
                # Ensure I_rot is 2D (3x3)
                if I_rot.ndim > 2:
                    I_rot = I_rot.squeeze()
                I_rot = I_rot.to(dtype=compute_dtype, device=self.device)
                # Rotate to link frame
                I_rot = R @ I_rot @ R.transpose(-2, -1)

            # Build spatial inertia
            I3 = torch.eye(3, dtype=compute_dtype, device=self.device)
            com_skew = skew_symmetric(com_pos.unsqueeze(0)).squeeze(0)

            I_spatial[node_idx, :3, :3] = m * I3
            I_spatial[node_idx, :3, 3:] = -m * com_skew
            I_spatial[node_idx, 3:, :3] = m * com_skew
            I_spatial[node_idx, 3:, 3:] = I_rot - m * (com_skew @ com_skew)

        return I_spatial.to(dtype=self.dtype)

    def to(
        self, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None
    ) -> "Chain":
        """Moves all tensors in the chain to the specified dtype and/or device.

        This is an in-place operation that modifies the chain's internal tensors.
        It is useful for switching between CPU and GPU computation or changing
        floating-point precision.

        Args:
            dtype (torch.dtype, optional): The target data type. If ``None``, the
                current dtype is preserved.
            device (Union[str, torch.device], optional): The target device.
                If ``None``, the current device is preserved.

        Returns:
            Chain: The instance itself, allowing for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        self._root = self._root.to(dtype=self.dtype, device=self.device)
        self.parents_indices = [p.to(device=self.device) for p in self.parents_indices]
        self.parents_indices_list = [path.tolist() for path in self.parents_indices]
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
        self.parent_array = self.parent_array.to(device=self.device)
        self.children_array = self.children_array.to(device=self.device)
        self.children_count = self.children_count.to(device=self.device)
        self.spatial_inertias = self.spatial_inertias.to(dtype=self.dtype, device=self.device)

        # Regenerate Python lists from updated tensors
        self.parent_list = self.parent_array.tolist()
        self.children_list = [
            self.children_array[i, : self.children_count[i]].tolist() for i in range(self.n_nodes)
        ]
        self.joint_indices_list = self.joint_indices.tolist()
        self.joint_type_indices_list = self.joint_type_indices.tolist()
        self.parents_indices_list = [path.tolist() for path in self.parents_indices]

        return self

    @property
    def nq(self) -> int:
        """Returns the total dimension of the generalized joint position `q`."""
        return self.nq_base + self.n_joints

    @property
    def nv(self) -> int:
        """Returns the total dimension of the generalized joint velocity `v`."""
        return self.nv_base + self.n_joints

    def get_generalized_coordinate_names(self, include_base: bool = True) -> List[str]:
        """Returns the ordered names for all configuration variables `q`.

        Args:
            include_base (bool, optional): If ``True`` and the chain has a floating
                base, prepends the base coordinate names (e.g., "base_tx", "base_qy").
                Defaults to ``True``.

        Returns:
            List[str]: An ordered list of coordinate names.
        """
        base_names = []
        if include_base and self.has_floating_base:
            base_names = [
                f"{self.base_name}_tx",
                f"{self.base_name}_ty",
                f"{self.base_name}_tz",
                f"{self.base_name}_qw",
                f"{self.base_name}_qx",
                f"{self.base_name}_qy",
                f"{self.base_name}_qz",
            ]
        return base_names + self.get_joint_parameter_names()

    def get_generalized_velocity_names(self, include_base: bool = True) -> List[str]:
        """Returns the ordered names for all velocity variables `v`.

        Args:
            include_base (bool, optional): If ``True`` and the chain has a floating
                base, prepends the base velocity names (e.g., "base_vx", "base_wz").
                Defaults to ``True``.

        Returns:
            List[str]: An ordered list of velocity names.
        """
        base_names = []
        if include_base and self.has_floating_base:
            base_names = [
                f"{self.base_name}_vx",
                f"{self.base_name}_vy",
                f"{self.base_name}_vz",
                f"{self.base_name}_wx",
                f"{self.base_name}_wy",
                f"{self.base_name}_wz",
            ]

        joint_vel_names = [
            n.replace(":position", ":velocity") if ":position" in n else f"{n}_d"
            for n in self.get_joint_parameter_names()
        ]
        return base_names + joint_vel_names

    def unpack_q(self, q: torch.Tensor) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Splits generalized coordinates `q` into base and joint components.

        Args:
            q (torch.Tensor): Generalized coordinates of shape ``(..., nq)``.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: A tuple ``(q_base, q_joints)``.
            ``q_base`` has shape ``(..., 7)`` or is ``None`` for fixed-base robots.
            ``q_joints`` has shape ``(..., n_joints)``.
        """
        q = self.ensure_tensor(q)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if not self.has_floating_base:
            return None, q
        return q[:, :7], q[:, 7:]

    def unpack_v(self, v: torch.Tensor) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Splits generalized velocities `v` into base and joint components.

        Args:
            v (torch.Tensor): Generalized velocities of shape ``(..., nv)``.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: A tuple ``(v_base, v_joints)``.
            ``v_base`` has shape ``(..., 6)`` or is ``None`` for fixed-base robots.
            ``v_joints`` has shape ``(..., n_joints)``.
        """
        v = self.ensure_tensor(v)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        if not self.has_floating_base:
            return None, v
        return v[:, :6], v[:, 6:]

    def pack_q(self, q_base: Optional[torch.Tensor], q_joints: torch.Tensor) -> torch.Tensor:
        """Combines base and joint positions into a single `q` vector.

        Args:
            q_base (Optional[torch.Tensor]): Base position of shape ``(..., 7)``
                ``[tx,ty,tz,qw,qx,qy,qz]``, or ``None`` for fixed-base robots.
            q_joints (torch.Tensor): Joint positions of shape ``(..., n_joints)``.

        Returns:
            torch.Tensor: The full generalized coordinate vector `q` of shape ``(..., nq)``.
        """
        if not self.has_floating_base:
            return q_joints
        if q_joints.ndim == 1:
            q_joints = q_joints.unsqueeze(0)
        if q_base.ndim == 1:
            q_base = q_base.unsqueeze(0)
        return torch.cat([q_base, q_joints], dim=-1)

    def pack_v(self, v_base: Optional[torch.Tensor], v_joints: torch.Tensor) -> torch.Tensor:
        """Combines base and joint velocities into a single `v` vector.

        Args:
            v_base (Optional[torch.Tensor]): Base velocity of shape ``(..., 6)``
                ``[vx,vy,vz,wx,wy,wz]``, or ``None`` for fixed-base robots.
            v_joints (torch.Tensor): Joint velocities of shape ``(..., n_joints)``.

        Returns:
            torch.Tensor: The full generalized velocity vector `v` of shape ``(..., nv)``.
        """
        if not self.has_floating_base:
            return v_joints
        if v_joints.ndim == 1:
            v_joints = v_joints.unsqueeze(0)
        if v_base.ndim == 1:
            v_base = v_base.unsqueeze(0)
        return torch.cat([v_base, v_joints], dim=-1)

    @staticmethod
    def _get_joints(frame: Frame, exclude_fixed: bool = True) -> List[Joint]:
        """Recursively collects all joints in the tree."""
        joints = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joints.append(frame.joint)
        for child in frame.children:
            joints.extend(Chain._get_joints(child, exclude_fixed))
        return joints

    def get_joints(self, exclude_fixed: bool = True) -> List[Joint]:
        """Returns all Joint objects in the chain.

        Args:
            exclude_fixed (bool, optional): If True, fixed joints are omitted
                from the list. Defaults to True.

        Returns:
            List[Joint]: A list of Joint objects.
        """
        return self._get_joints(self._root, exclude_fixed=exclude_fixed)

    @lru_cache()
    def get_joint_parameter_names(self, exclude_fixed: bool = True) -> List[str]:
        """Returns the ordered list of actuated joint names.

        This order defines the canonical layout of the joint components in the `q` and `v`
        vectors and is used consistently across all calculations.

        Args:
            exclude_fixed (bool, optional): If ``True``, names of fixed joints are
                omitted. This is the standard behavior. Defaults to ``True``.

        Returns:
            List[str]: An ordered list of actuated joint names.
        """
        return [j.name for j in self.get_joints(exclude_fixed=exclude_fixed)]

    def find_frame(self, name: str) -> Optional[Frame]:
        """Finds a Frame object by its name in the kinematic tree.

        Args:
            name (str): The name of the frame to search for.

        Returns:
            Optional[Frame]: The Frame object if found, otherwise ``None``.
        """
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_frame_recursive(name: str, frame: Frame) -> Optional[Frame]:
        """Recursively searches for a frame by name."""
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if ret is not None:
                return ret
        return None

    def get_frame_names(self, exclude_fixed: bool = True) -> List[str]:
        """Returns all frame names in the chain in traversal order.

        Args:
            exclude_fixed (bool, optional): If ``True``, names of frames associated
                with fixed joints are omitted. Defaults to ``True``.

        Returns:
            List[str]: A list of all frame names.
        """
        names = self._get_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_frame_names(frame: Frame, exclude_fixed: bool = True) -> List[str]:
        """Recursively collects all frame names in the tree."""
        names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            names.append(frame.name)
        for child in frame.children:
            names.extend(Chain._get_frame_names(child, exclude_fixed))
        return names

    @lru_cache
    def get_frame_id(self, frame_name: str) -> int:
        """Gets the integer index for a frame name.

        This index corresponds to the row/column in pre-computed data structures
        like ``spatial_inertias``.

        Args:
            frame_name (str): The frame name.

        Returns:
            torch.Tensor: A tensor containing the integer index.
        """
        return self.frame_to_idx[frame_name]

    def ensure_tensor(self, value: Union[torch.Tensor, np.ndarray, List, Dict]) -> torch.Tensor:
        """Converts various input types to a tensor in the correct joint order.

        This utility handles conversion from numpy arrays, lists, or dictionaries
        (mapping joint names to values) into a ``torch.Tensor`` on the chain's
        device and dtype.

        Args:
            value (Union[torch.Tensor, np.ndarray, List, Dict]): The input value.
                If a dictionary is provided, it must contain all actuated joint names
                as keys.

        Returns:
            torch.Tensor: The converted tensor, on the correct device and dtype.

        Raises:
            ValueError: If a dictionary input is missing values for some joints.
            TypeError: If the input type is not supported.
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
        """Converts a joint name dictionary to an ordered tensor."""
        elem_shape = self._get_dict_elem_shape(value_dict)
        tensor = torch.full(
            [*elem_shape, self.n_joints], torch.nan, device=self.device, dtype=self.dtype
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
        """Gets the shape of dictionary values for batch handling."""
        elem = next(iter(value_dict.values()))
        if isinstance(elem, (np.ndarray, torch.Tensor)):
            return elem.shape
        return ()

    def clamp(self, joint_values: torch.Tensor) -> torch.Tensor:
        """Clamps joint values to the robot's defined joint position limits.

        Args:
            joint_values (torch.Tensor): A tensor of joint values, corresponding to
                the joint-space part of ``q``.

        Returns:
            torch.Tensor: The clamped joint values.
        """
        joint_values = self.ensure_tensor(joint_values)
        return torch.clamp(joint_values, self.low, self.high)

    def get_joint_limits(self) -> tuple[List[float], List[float]]:
        """Returns the position limits (low, high) for all actuated joints."""
        return self._get_joint_limits("limits")

    def get_joint_velocity_limits(self) -> tuple[List[float], List[float]]:
        """Returns the velocity limits (low, high) for all actuated joints."""
        return self._get_joint_limits("velocity_limits")

    def get_joint_effort_limits(self) -> tuple[List[float], List[float]]:
        """Returns the effort (torque/force) limits (low, high) for all actuated joints."""
        return self._get_joint_limits("effort_limits")

    def _get_joint_limits(self, param_name: str) -> tuple[List[float], List[float]]:
        """Extracts a joint limit parameter (e.g., 'limits') from all joints."""
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
        """Returns a string representation of the kinematic tree structure."""
        return str(self._root)
