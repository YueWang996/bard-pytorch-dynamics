"""
Class-based kinematic computations with pre-allocated memory.

This module provides classes for core kinematic calculations:
- ``ForwardKinematics``: Computes the world-frame pose of any link.
- ``SpatialAcceleration``: Computes the spatial acceleration of any link.

Both classes are designed for high performance by pre-allocating memory buffers,
making them suitable for use in computationally intensive loops, such as in
reinforcement learning or trajectory optimization.
"""

from typing import Dict, Optional, Any
import torch
import bard.transforms as tf
from bard.core import chain
from bard.structures import Joint
from bard.transforms import (
    axis_and_angle_to_matrix_44,
    axis_and_d_to_pris_matrix,
)

from .utils import (
    identity_transform,
    as_batched_transform,
    inv_homogeneous,
    motion_cross_product,
    spatial_adjoint,
)


class ForwardKinematics:
    """Forward kinematics computation with pre-allocated memory.

    This class computes the forward kinematics for a given frame (link) in the
    robot's kinematic chain. It determines the 4x4 homogeneous transformation
    matrix that represents the pose of the frame in the world coordinate system.

    To achieve high performance, this class avoids runtime memory allocation by
    pre-computing static data. An instance should be created once per robot chain
    and reused for all subsequent computations.

    Args:
        chain (chain.Chain): The robot's kinematic chain definition.
        max_batch_size (int, optional): The maximum batch size the instance will
            support. Defaults to 1024.
        compile_enabled (bool, optional): If ``True``, the core computation
            will be JIT-compiled with ``torch.compile``. Defaults to ``False``.
        compile_kwargs (Dict[str, Any], optional): A dictionary of keyword
            arguments to pass to ``torch.compile``. Defaults to ``None``.

    Attributes:
        chain (chain.Chain): The robot kinematic chain.
        max_batch_size (int): The maximum supported batch size.
        dtype (torch.dtype): The data type of the tensors.
        device (torch.device): The device where tensors are stored.

    Example:
        .. code-block:: python

            # Create an FK instance once
            fk = ForwardKinematics(robot_chain, max_batch_size=128)
            eef_frame_id = robot_chain.get_frame_id("end_effector_link")

            # Use in a loop for efficient computation
            for q in data_loader:
                T_world_eef = fk.calc(q, eef_frame_id)
                position = T_world_eef[:, :3, 3]
                # ... use the computed pose ...
    """
    
    def __init__(self, chain: chain.Chain, max_batch_size: int = 1024,
                 compile_enabled: Optional[bool] = False,
                 compile_kwargs: Optional[Dict[str, Any]] = None):
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device
        
        self.n_nodes = chain.n_nodes
        self.n_joints = chain.n_joints
        self.is_floating_base = chain.has_floating_base
        
        # Pre-compute normalized axes (same as jacobian_cached.py)
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Conditional compilation setup
        self._compile_enabled = compile_enabled
        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}
        self._setup_calc_callable()

    def enable_compilation(self, enabled: bool = True, **compile_kwargs):
        """Enable or disable ``torch.compile`` for the core computation.

        Args:
            enabled (bool, optional): If ``True``, compilation is enabled.
                Defaults to ``True``.
            **compile_kwargs: Additional keyword arguments to pass to
                ``torch.compile``.
        """
        self._compile_enabled = enabled
        if compile_kwargs:
            self._compile_kwargs.update(compile_kwargs)
        self._setup_calc_callable()
        
    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        """Move all internal buffers to a specified dtype and/or device.

        This is an in-place operation.

        Args:
            dtype (torch.dtype, optional): The target data type. Defaults to ``None``.
            device (torch.device, optional): The target device. Defaults to ``None``.

        Returns:
            ForwardKinematics: The instance itself for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        
        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)

        self._setup_calc_callable()
        
        return self
    
    def calc(
        self,
        q: torch.Tensor,
        frame_id: int,
    ) -> torch.Tensor:
        """Compute forward kinematics for a specific frame.

        Args:
            q (torch.Tensor): A batch of generalized positions.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 7 + n_joints)``,
                  where the first 7 elements are ``[tx, ty, tz, qw, qx, qy, qz]``.
            frame_id (int): The integer index of the target frame (link).

        Returns:
            torch.Tensor: A batch of ``(B, 4, 4)`` homogeneous transformation
            matrices representing the world-frame pose of the target frame.

        Raises:
            ValueError: If the input batch size ``B`` exceeds ``self.max_batch_size``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}..."
            )
        return self._calc_callable(q, frame_id)

    def _setup_calc_callable(self):
        fn = self._calc_impl
        if self._compile_enabled:
            fn = torch.compile(fn, **self._compile_kwargs)
        self._calc_callable = fn

    def _calc_impl(
        self,
        q: torch.Tensor,
        frame_id: int,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        
        # Get path from root to target frame as Python list
        path_nodes = self.chain.parents_indices_list[frame_id]
        
        # Split configuration
        if self.is_floating_base:
            q_base = q[:, :7]
            q_joints = q[:, 7:]
        else:
            q_base = None
            q_joints = q

        # Pre-compute per-joint motion transforms
        axes_expanded = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
        T_revolute = axis_and_angle_to_matrix_44(axes_expanded, q_joints)
        T_prismatic = axis_and_d_to_pris_matrix(axes_expanded, q_joints)

        I44 = identity_transform(batch_size, self.dtype, self.device)

        # Initialize world transform with base pose (if floating)
        if self.is_floating_base and q_base is not None:
            t = q_base[:, :3]
            qwqxqyqz = q_base[:, 3:]
            qwqxqyqz = qwqxqyqz / qwqxqyqz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            qw, qx, qy, qz = qwqxqyqz.unbind(-1)

            two = torch.tensor(2.0, dtype=self.dtype, device=self.device)
            x2, y2, z2 = two * qx * qx, two * qy * qy, two * qz * qz
            xy, xz, yz = two * qx * qy, two * qx * qz, two * qy * qz
            wz, wy, wx = two * qw * qz, two * qw * qy, two * qw * qx

            R_base_to_world = torch.empty(batch_size, 3, 3, dtype=self.dtype, device=self.device)
            R_base_to_world[:, 0, 0] = 1.0 - (y2 + z2)
            R_base_to_world[:, 0, 1] = xy - wz
            R_base_to_world[:, 0, 2] = xz + wy
            R_base_to_world[:, 1, 0] = xy + wz
            R_base_to_world[:, 1, 1] = 1.0 - (x2 + z2)
            R_base_to_world[:, 1, 2] = yz - wx
            R_base_to_world[:, 2, 0] = xz - wy
            R_base_to_world[:, 2, 1] = yz + wx
            R_base_to_world[:, 2, 2] = 1.0 - (x2 + y2)

            T_world_to_current = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_to_current[:, :3, :3] = R_base_to_world
            T_world_to_current[:, :3, 3] = t
            T_world_to_current[:, 3, 3] = 1.0
        else:
            T_world_to_current = I44.clone()

        # Forward pass: accumulate transforms along path
        for node_idx in path_nodes:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]

            T_joint_offset = as_batched_transform(
                self.chain.joint_offsets[node_idx], batch_size, self.dtype, self.device
            )

            # Select motion transform based on joint type
            is_revolute = (joint_type_idx == Joint.TYPES.index('revolute'))
            is_prismatic = (joint_type_idx == Joint.TYPES.index('prismatic'))
            
            if is_revolute:
                T_motion = T_revolute[:, joint_idx]
            elif is_prismatic:
                T_motion = T_prismatic[:, joint_idx]
            else:  # fixed
                T_motion = I44

            T_link_offset = as_batched_transform(
                self.chain.link_offsets[node_idx], batch_size, self.dtype, self.device
            )

            # Accumulate transform
            T_world_to_current = T_world_to_current @ T_joint_offset @ T_motion @ T_link_offset

        return T_world_to_current


class SpatialAcceleration:
    """End-effector spatial acceleration computation with pre-allocated memory.

    This class implements the forward pass of the RNEA algorithm to compute the
    6D spatial acceleration (linear and angular) of a specific frame. It is
    essential for tasks requiring acceleration-level analysis, such as operational
    space control.

    All necessary buffers are pre-allocated for maximum performance, making it
    efficient for use in control loops or other performance-sensitive code.

    Args:
        chain (chain.Chain): The robot's kinematic chain definition.
        max_batch_size (int, optional): The maximum batch size the instance will
            support. Defaults to 1024.
        compile_enabled (bool, optional): If ``True``, the core computation
            will be JIT-compiled with ``torch.compile``. Defaults to ``False``.
        compile_kwargs (Dict[str, Any], optional): A dictionary of keyword
            arguments to pass to ``torch.compile``. Defaults to ``{"mode": "reduce-overhead"}``.

    Attributes:
        chain (chain.Chain): The robot kinematic chain.
        max_batch_size (int): The maximum supported batch size.
        dtype (torch.dtype): The data type of the tensors.
        device (torch.device): The device where tensors are stored.

    Example:
        .. code-block:: python

            # Create an acceleration instance once
            accel = SpatialAcceleration(robot_chain, max_batch_size=128)
            eef_frame_id = robot_chain.get_frame_id("end_effector_link")

            # Use in a loop
            for q, qd, qdd in data_loader:
                a_world = accel.calc(q, qd, qdd, eef_frame_id, reference_frame="world")
                linear_accel = a_world[:, :3]
                # ... use the computed acceleration ...
    """
    def __init__(self, chain: chain.Chain, max_batch_size: int = 1024,
                 compile_enabled: Optional[bool] = False,
                 compile_kwargs: Optional[Dict[str, Any]] = {"mode": "reduce-overhead"}):
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device
        
        self.n_nodes = chain.n_nodes
        self.n_joints = chain.n_joints
        self.is_floating_base = chain.has_floating_base
        
        # Pre-allocate all buffers
        self.Xup = torch.zeros(self.n_nodes, max_batch_size, 6, 6, dtype=self.dtype, device=self.device)
        self.S = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        self.a = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        
        # Pre-allocate transform storage (avoid Python list mutation in compiled code)
        self.T_world_to_node = torch.zeros(
            self.n_nodes, max_batch_size, 4, 4, dtype=self.dtype, device=self.device
        )
        
        # Pre-compute normalized axes (same as jacobian_cached.py)
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Conditional compilation setup
        self._compile_enabled = compile_enabled
        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}
        self._setup_calc_callable()

    def enable_compilation(self, enabled: bool = True, **compile_kwargs):
        """Enable or disable ``torch.compile`` for the core computation.

        Args:
            enabled (bool, optional): If ``True``, compilation is enabled.
                Defaults to ``True``.
            **compile_kwargs: Additional keyword arguments to pass to
                ``torch.compile``.
        """
        self._compile_enabled = enabled
        if compile_kwargs:
            self._compile_kwargs.update(compile_kwargs)
        self._setup_calc_callable()

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        """Move all internal buffers to a specified dtype and/or device.

        This is an in-place operation.

        Args:
            dtype (torch.dtype, optional): The target data type. Defaults to ``None``.
            device (torch.device, optional): The target device. Defaults to ``None``.

        Returns:
            SpatialAcceleration: The instance itself for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        
        self.Xup = self.Xup.to(dtype=self.dtype, device=self.device)
        self.S = self.S.to(dtype=self.dtype, device=self.device)
        self.v = self.v.to(dtype=self.dtype, device=self.device)
        self.a = self.a.to(dtype=self.dtype, device=self.device)
        self.T_world_to_node = self.T_world_to_node.to(dtype=self.dtype, device=self.device)
        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)

        self._setup_calc_callable()
        
        return self
    
    def calc(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        frame_id: int,
        reference_frame: str = "local",
    ) -> torch.Tensor:
        """Compute the spatial acceleration of a frame.

        Args:
            q (torch.Tensor): A batch of generalized positions.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 7 + n_joints)``.
            qd (torch.Tensor): A batch of generalized velocities.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 6 + n_joints)``.
            qdd (torch.Tensor): A batch of generalized accelerations, with the
                same shape as ``qd``.
            frame_id (int): The integer index of the target frame.
            reference_frame (str, optional): The frame of reference for the
                output acceleration. Can be ``"world"`` or ``"local"`` (the frame's
                own coordinate system). Defaults to ``"local"``.

        Returns:
            torch.Tensor: The spatial acceleration ``[linear; angular]`` of the
            target frame, with shape ``(B, 6)``.

        Raises:
            ValueError: If the input batch size ``B`` exceeds ``self.max_batch_size``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}..."
            )
        return self._calc_callable(q, qd, qdd, frame_id, reference_frame)
    
    def _setup_calc_callable(self):
        fn = self._calc_impl
        if self._compile_enabled:
            fn = torch.compile(fn, **self._compile_kwargs)
        self._calc_callable = fn

    def _calc_impl(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        frame_id: int,
        reference_frame: str = "local",
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        
        # Get sliced views of pre-allocated buffers
        Xup = self.Xup[:, :batch_size, :, :]
        S = self.S[:, :batch_size, :]
        v = self.v[:, :batch_size, :]
        a = self.a[:, :batch_size, :]
        T_world_to_node = self.T_world_to_node[:, :batch_size, :, :]
        
        # Zero out buffers
        S.zero_()
        v.zero_()
        a.zero_()
        
        # Split base and joint components
        if self.is_floating_base:
            q_base = q[:, :7]
            q_joints = q[:, 7:]
            v_base = qd[:, :6]
            v_joints = qd[:, 6:]
            a_base = qdd[:, :6]
            a_joints = qdd[:, 6:]
        else:
            q_base = v_base = a_base = None
            q_joints = q
            v_joints = qd
            a_joints = qdd

        # World spatial acceleration (gravity disabled for pure kinematic acceleration)
        a_world0 = torch.zeros((batch_size, 6), dtype=self.dtype, device=self.device)

        # Pre-compute joint transforms
        axes_expanded = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
        T_revolute = axis_and_angle_to_matrix_44(axes_expanded, q_joints)
        T_prismatic = axis_and_d_to_pris_matrix(axes_expanded, q_joints)

        I44 = identity_transform(batch_size, self.dtype, self.device)

        # Build base transform if floating
        if self.is_floating_base and q_base is not None:
            t = q_base[:, :3]
            qwqxqyqz = q_base[:, 3:]
            qwqxqyqz = qwqxqyqz / qwqxqyqz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            qw, qx, qy, qz = qwqxqyqz.unbind(-1)

            two = torch.tensor(2.0, dtype=self.dtype, device=self.device)
            x2, y2, z2 = two * qx * qx, two * qy * qy, two * qz * qz
            xy, xz, yz = two * qx * qy, two * qx * qz, two * qy * qz
            wz, wy, wx = two * qw * qz, two * qw * qy, two * qw * qx

            R = torch.empty(batch_size, 3, 3, dtype=self.dtype, device=self.device)
            R[:, 0, 0] = 1.0 - (y2 + z2)
            R[:, 0, 1] = xy - wz
            R[:, 0, 2] = xz + wy
            R[:, 1, 0] = xy + wz
            R[:, 1, 1] = 1.0 - (x2 + z2)
            R[:, 1, 2] = yz - wx
            R[:, 2, 0] = xz - wy
            R[:, 2, 1] = yz + wx
            R[:, 2, 2] = 1.0 - (x2 + y2)

            T_world_to_base = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_to_base[:, :3, :3] = R
            T_world_to_base[:, :3, 3] = t
            T_world_to_base[:, 3, 3] = 1.0
        else:
            T_world_to_base = I44

        # Forward pass: propagate velocities and accelerations
        for node_idx in self.chain.topo_order:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]
            parent_idx = self.chain.parent_list[node_idx]

            T_joint_offset = as_batched_transform(
                self.chain.joint_offsets[node_idx], batch_size, self.dtype, self.device
            )
            T_link_offset = as_batched_transform(
                self.chain.link_offsets[node_idx], batch_size, self.dtype, self.device
            )

            # Select motion transform
            is_revolute = (joint_type_idx == Joint.TYPES.index('revolute'))
            is_prismatic = (joint_type_idx == Joint.TYPES.index('prismatic'))

            if is_revolute:
                T_motion = T_revolute[:, joint_idx]
            elif is_prismatic:
                T_motion = T_prismatic[:, joint_idx]
            else:  # fixed
                T_motion = I44

            # Parent -> child transform
            T_parent_to_child = T_joint_offset @ T_motion @ T_link_offset
            Xup_i = spatial_adjoint(inv_homogeneous(T_parent_to_child))
            Xup[node_idx] = Xup_i

            # Joint subspace
            is_actuated = is_revolute or is_prismatic
            if is_actuated:
                axis_local = self.axes_norm[joint_idx].expand(batch_size, -1)
                twist_joint = torch.zeros((batch_size, 6), dtype=self.dtype, device=self.device)
                if is_revolute:
                    twist_joint[:, 3:] = axis_local
                else:  # prismatic
                    twist_joint[:, :3] = axis_local
                S[node_idx] = (spatial_adjoint(T_link_offset) @ twist_joint.unsqueeze(-1)).squeeze(-1)
                v_joint = v_joints[:, joint_idx].unsqueeze(-1)
                a_joint = a_joints[:, joint_idx].unsqueeze(-1)
            else:
                v_joint = torch.zeros((batch_size, 1), dtype=self.dtype, device=self.device)
                a_joint = torch.zeros((batch_size, 1), dtype=self.dtype, device=self.device)

            # Parent state
            if parent_idx == -1:  # Root node
                if self.is_floating_base and v_base is not None:
                    v_parent = v_base
                    a_parent = a_base + a_world0
                else:
                    v_parent = torch.zeros((batch_size, 6), dtype=self.dtype, device=self.device)
                    a_parent = a_world0
                T_world_to_parent = T_world_to_base
            else:
                v_parent = v[parent_idx]
                a_parent = a[parent_idx]
                T_world_to_parent = T_world_to_node[parent_idx]

            # RNEA propagation
            vJ = S[node_idx] * v_joint
            v[node_idx] = (Xup_i @ v_parent.unsqueeze(-1)).squeeze(-1) + vJ

            coriolis = (motion_cross_product(v[node_idx]) @ vJ.unsqueeze(-1)).squeeze(-1)
            a[node_idx] = (
                (Xup_i @ a_parent.unsqueeze(-1)).squeeze(-1)
                + S[node_idx] * a_joint
                + coriolis
            )

            # Accumulate world pose
            T_world_to_node[node_idx] = T_world_to_parent @ T_parent_to_child

        # Extract acceleration at target frame
        a_local = a[frame_id]

        if reference_frame == "world":
            Ad_world_wrt_body = spatial_adjoint(T_world_to_node[frame_id])
            a_world = (Ad_world_wrt_body @ a_local.unsqueeze(-1)).squeeze(-1)
            return a_world
        else:  # local
            return a_local
