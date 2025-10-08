"""
Class-based robot dynamics algorithms with pre-allocated memory.

This module implements the core dynamics algorithms for the ``bard`` library.
It includes class-based, optimized implementations of the Recursive Newton-Euler
Algorithm (RNEA) for inverse dynamics and the Composite Rigid Body Algorithm (CRBA)
for calculating the mass matrix.

A key design feature of these classes is the pre-allocation of all necessary
memory buffers upon instantiation. This strategy minimizes or eliminates runtime
memory allocation, making these classes highly efficient for use in performance-critical
applications like reinforcement learning training loops where the same computations
are performed repeatedly.
"""

from typing import Any, Dict, Optional
import torch

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
    spatial_adjoint,
    motion_cross_product,
    force_cross_product,
)


class RNEA:
    """Recursive Newton-Euler Algorithm with pre-allocated memory.

    This class provides an efficient, batched implementation of the RNEA algorithm
    to compute inverse dynamics. It calculates the generalized forces (torques)
    required to produce a given state of joint positions, velocities, and accelerations.

    All temporary storage needed for the computation is pre-allocated upon
    instantiation, which eliminates runtime memory allocation overhead. This makes
    it ideal for high-performance scenarios. An instance should be created once
    per robot chain and reused for all subsequent computations.

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

            # Create an RNEA instance once
            rnea = RNEA(robot_chain, max_batch_size=128)

            # Use in a loop for efficient computation
            for q, qd, qdd in data_loader:
                tau = rnea.calc(q, qd, qdd)
                # ... use the computed torques ...
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
        
        # Pre-allocate all buffers
        self.Xup = torch.zeros(self.n_nodes, max_batch_size, 6, 6, dtype=self.dtype, device=self.device)
        self.S = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        self.a = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        self.f = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        
        # Pre-compute static data
        self.I_spatial = chain.spatial_inertias.unsqueeze(1)  # (n_nodes, 1, 6, 6)
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.gravity = torch.tensor([0.0, 0.0, -9.81], dtype=self.dtype, device=self.device)

        # Conditional compilation setup
        self._compile_enabled = compile_enabled
        self._compile_kwargs: Dict[str, Any] = dict(compile_kwargs) if compile_kwargs else {}
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
            RNEA: The instance itself for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        
        self.Xup = self.Xup.to(dtype=self.dtype, device=self.device)
        self.S = self.S.to(dtype=self.dtype, device=self.device)
        self.v = self.v.to(dtype=self.dtype, device=self.device)
        self.a = self.a.to(dtype=self.dtype, device=self.device)
        self.f = self.f.to(dtype=self.dtype, device=self.device)
        self.I_spatial = self.I_spatial.to(dtype=self.dtype, device=self.device)
        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)
        self.gravity = self.gravity.to(dtype=self.dtype, device=self.device)
        
        self._setup_calc_callable()

        return self
    
    def calc(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        gravity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute generalized forces (torques) via the RNEA algorithm.

        Args:
            q (torch.Tensor): A batch of generalized positions.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 7 + n_joints)``,
                  where the first 7 elements are ``[tx, ty, tz, qw, qx, qy, qz]``.
            qd (torch.Tensor): A batch of generalized velocities.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 6 + n_joints)``,
                  where the first 6 elements represent the base's spatial velocity
                  ``[vx, vy, vz, wx, wy, wz]``.
            qdd (torch.Tensor): A batch of generalized accelerations. Must have the
                same shape as ``qd``.
            gravity (torch.Tensor, optional): A 3-element gravity vector in the
                world frame. If ``None``, defaults to ``[0, 0, -9.81]``.

        Returns:
            torch.Tensor: A batch of generalized forces ``tau``.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 6 + n_joints)``.

        Raises:
            ValueError: If the input batch size ``B`` exceeds ``self.max_batch_size``.
        """
        
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}..."
            )
        return self._calc_callable(q, qd, qdd, gravity)
    
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
        gravity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}. "
                f"Create a new RNEA instance with larger max_batch_size."
            )
        
        # Get sliced views of pre-allocated buffers
        Xup = self.Xup[:, :batch_size, :, :]
        S = self.S[:, :batch_size, :]
        v = self.v[:, :batch_size, :]
        a = self.a[:, :batch_size, :]
        f = self.f[:, :batch_size, :]
        I_spatial = self.I_spatial.expand(-1, batch_size, -1, -1)
        
        # Zero out buffers
        S.zero_()
        v.zero_()
        a.zero_()
        f.zero_()
        
        # Split base and joint components
        if self.is_floating_base:
            q_base, q_joints = q[:, :7], q[:, 7:]
            v_base, v_joints = qd[:, :6], qd[:, 6:]
            a_base, a_joints = qdd[:, :6], qdd[:, 6:]
        else:
            q_joints, v_joints, a_joints = q, qd, qdd

        # Set up gravity
        g = self.gravity if gravity is None else gravity
        a_gravity_world = torch.zeros((batch_size, 6), dtype=self.dtype, device=self.device)
        a_gravity_world[:, :3] = -g.expand(batch_size, -1)

        I44 = identity_transform(batch_size, self.dtype, self.device)

        # Build base transform if floating
        if self.is_floating_base:
            t = q_base[:, :3]
            qwqxqyqz = q_base[:, 3:]
            qwqxqyqz = qwqxqyqz / qwqxqyqz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            qw, qx, qy, qz = qwqxqyqz.unbind(-1)
            
            two = torch.tensor(2.0, dtype=self.dtype, device=self.device)
            x2, y2, z2 = two * qx * qx, two * qy * qy, two * qz * qz
            xy, xz, yz = two * qx * qy, two * qx * qz, two * qy * qz
            wz, wy, wx = two * qw * qz, two * qw * qy, two * qw * qx
            
            R = torch.empty(batch_size, 3, 3, dtype=self.dtype, device=self.device)
            R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = 1.0 - (y2 + z2), xy - wz, xz + wy
            R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = xy + wz, 1.0 - (x2 + z2), yz - wx
            R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = xz - wy, yz + wx, 1.0 - (x2 + y2)
            
            T_world_to_base = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_to_base[:, :3, :3] = R
            T_world_to_base[:, :3, 3] = t
            T_world_to_base[:, 3, 3] = 1.0
            
            Ad_base_world = spatial_adjoint(inv_homogeneous(T_world_to_base))
            a_gravity_base = (Ad_base_world @ a_gravity_world.unsqueeze(-1)).squeeze(-1)
        else:
            a_gravity_base = a_gravity_world

        # ========================================================================
        # Forward pass: propagate velocities and accelerations
        # ========================================================================

        for node_idx in self.chain.topo_order:
            j_idx = self.chain.joint_indices_list[node_idx]
            j_type = self.chain.joint_type_indices_list[node_idx]
            p_idx = self.chain.parent_list[node_idx]

            T_joint_offset = as_batched_transform(
                self.chain.joint_offsets[node_idx], batch_size, self.dtype, self.device
            )
            T_link_offset = as_batched_transform(
                self.chain.link_offsets[node_idx], batch_size, self.dtype, self.device
            )
            
            is_revolute = (j_type == Joint.TYPES.index('revolute'))
            is_prismatic = (j_type == Joint.TYPES.index('prismatic'))
            is_actuated = is_revolute or is_prismatic
            
            if is_revolute:
                T_motion = axis_and_angle_to_matrix_44(
                    self.axes_norm[j_idx].expand(batch_size, -1), 
                    q_joints[:, j_idx]
                )
            elif is_prismatic:
                T_motion = axis_and_d_to_pris_matrix(
                    self.axes_norm[j_idx].expand(batch_size, -1), 
                    q_joints[:, j_idx]
                )
            else:
                T_motion = I44
            
            T_parent_child = T_joint_offset @ T_motion @ T_link_offset
            Xup[node_idx] = spatial_adjoint(inv_homogeneous(T_parent_child))

            # Joint subspace
            if is_actuated:
                axis_local = self.axes_norm[j_idx].expand(batch_size, -1)
                twist_joint = torch.zeros(batch_size, 6, dtype=self.dtype, device=self.device)
                
                if is_revolute:
                    twist_joint[:, 3:] = axis_local
                elif is_prismatic:
                    twist_joint[:, :3] = axis_local
                
                S[node_idx] = (spatial_adjoint(T_link_offset) @ twist_joint.unsqueeze(-1)).squeeze(-1)
                v_joint = v_joints[:, j_idx].unsqueeze(-1)
                a_joint = a_joints[:, j_idx].unsqueeze(-1)
            else:
                v_joint = torch.zeros((batch_size, 1), dtype=self.dtype, device=self.device)
                a_joint = torch.zeros((batch_size, 1), dtype=self.dtype, device=self.device)

            # Parent state
            if p_idx == -1:  # Root node
                if self.is_floating_base:
                    v_parent = v_base
                    a_parent = a_base + a_gravity_base
                else:
                    v_parent = torch.zeros((batch_size, 6), dtype=self.dtype, device=self.device)
                    a_parent = a_gravity_base
            else:
                v_parent = v[p_idx]
                a_parent = a[p_idx]
            
            vJ = S[node_idx] * v_joint
            v[node_idx] = (Xup[node_idx] @ v_parent.unsqueeze(-1)).squeeze(-1) + vJ
            
            coriolis = (motion_cross_product(v[node_idx]) @ vJ.unsqueeze(-1)).squeeze(-1)
            a[node_idx] = (
                (Xup[node_idx] @ a_parent.unsqueeze(-1)).squeeze(-1) 
                + S[node_idx] * a_joint 
                + coriolis
            )

        # ========================================================================
        # Backward pass: compute forces
        # ========================================================================

        for node_idx in reversed(self.chain.topo_order):
            Iv = (I_spatial[node_idx] @ v[node_idx].unsqueeze(-1)).squeeze(-1)
            f_node = (I_spatial[node_idx] @ a[node_idx].unsqueeze(-1)).squeeze(-1)
            f_node += (force_cross_product(v[node_idx]) @ Iv.unsqueeze(-1)).squeeze(-1)

            # Aggregate forces from children
            for child_idx in self.chain.children_list[node_idx]:
                f_node += (
                    Xup[child_idx].transpose(1, 2) @ f[child_idx].unsqueeze(-1)
                ).squeeze(-1)
            f[node_idx] = f_node
        
        # ========================================================================
        # Extract generalized forces
        # ========================================================================

        tau_all_nodes = torch.sum(S * f, dim=-1)  # (n_nodes, batch_size)

        if self.is_floating_base:
            tau = torch.zeros((batch_size, 6 + self.n_joints), dtype=self.dtype, device=self.device)
            urdf_root_idx = 1 if self.n_nodes > 1 else 0
            tau[:, :6] = f[urdf_root_idx]
            
            for node_idx in range(self.n_nodes):
                j_type = self.chain.joint_type_indices_list[node_idx]
                if j_type in [Joint.TYPES.index('revolute'), Joint.TYPES.index('prismatic')]:
                    j_col = self.chain.joint_indices_list[node_idx]
                    tau[:, 6 + j_col] = tau_all_nodes[node_idx]
        else:
            tau = torch.zeros((batch_size, self.n_joints), dtype=self.dtype, device=self.device)
            for node_idx in range(self.n_nodes):
                j_type = self.chain.joint_type_indices_list[node_idx]
                if j_type in [Joint.TYPES.index('revolute'), Joint.TYPES.index('prismatic')]:
                    j_col = self.chain.joint_indices_list[node_idx]
                    tau[:, j_col] = tau_all_nodes[node_idx]

        return tau


class CRBA:
    """Composite Rigid Body Algorithm with pre-allocated memory.

    This class provides an efficient, batched implementation of the CRBA algorithm
    to compute the joint-space inertia matrix (mass matrix) ``M``. The mass
    matrix relates joint accelerations to joint torques via the equation
    :math:`\\tau = M(q) \\ddot{q} + C(q, \\dot{q})`.

    All temporary storage needed for the computation is pre-allocated upon
    instantiation, which eliminates runtime memory allocation overhead. This makes
    it ideal for high-performance scenarios. An instance should be created once
    per robot chain and reused for all subsequent computations.

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

            # Create a CRBA instance once
            crba = CRBA(robot_chain, max_batch_size=128)

            # Use in a loop for efficient computation
            for q in data_loader:
                M = crba.calc(q)
                # ... use the computed mass matrix ...
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
        self.nv = 6 + self.n_joints if self.is_floating_base else self.n_joints
        
        # Pre-allocate all buffers
        self.Xup = torch.zeros(self.n_nodes, max_batch_size, 6, 6, dtype=self.dtype, device=self.device)
        self.S = torch.zeros(self.n_nodes, max_batch_size, 6, dtype=self.dtype, device=self.device)
        self.I_composite = torch.zeros(self.n_nodes, max_batch_size, 6, 6, dtype=self.dtype, device=self.device)
        self.M = torch.zeros(max_batch_size, self.nv, self.nv, dtype=self.dtype, device=self.device)
        
        # Pre-compute static data
        self.I_spatial = chain.spatial_inertias.unsqueeze(1)  # (n_nodes, 1, 6, 6)
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        
        # Pre-compute velocity index mapping as Python list (avoids .item() in compiled code)
        self.vel_indices_list = []
        for node_idx in range(self.n_nodes):
            joint_type_idx = chain.joint_type_indices_list[node_idx]
            if joint_type_idx != Joint.TYPES.index('fixed'):
                joint_col = chain.joint_indices_list[node_idx]
                vel_idx = (6 + joint_col) if self.is_floating_base else joint_col
                self.vel_indices_list.append(vel_idx)
            else:
                self.vel_indices_list.append(-1)
        
        self.urdf_root_idx = 1 if (self.is_floating_base and self.n_nodes > 1) else 0

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
            CRBA: The instance itself for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        
        self.Xup = self.Xup.to(dtype=self.dtype, device=self.device)
        self.S = self.S.to(dtype=self.dtype, device=self.device)
        self.I_composite = self.I_composite.to(dtype=self.dtype, device=self.device)
        self.M = self.M.to(dtype=self.dtype, device=self.device)
        self.I_spatial = self.I_spatial.to(dtype=self.dtype, device=self.device)
        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)
        # vel_indices_list is Python list, no need to move

        self._setup_calc_callable()
        
        return self
    

    def calc(self, q: torch.Tensor) -> torch.Tensor:
        """Compute the mass matrix ``M`` via the CRBA algorithm.

        Args:
            q (torch.Tensor): A batch of generalized positions.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 7 + n_joints)``,
                  where the first 7 elements are ``[tx, ty, tz, qw, qx, qy, qz]``.

        Returns:
            torch.Tensor: The batched mass matrix ``M``.
                - For fixed-base robots, shape is ``(B, n_joints, n_joints)``.
                - For floating-base robots, shape is ``(B, 6 + n_joints, 6 + n_joints)``.

        Raises:
            ValueError: If the input batch size ``B`` exceeds ``self.max_batch_size``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}..."
            )
        return self._calc_callable(q)
    
    def _setup_calc_callable(self):
        fn = self._calc_impl
        if self._compile_enabled:
            fn = torch.compile(fn, **self._compile_kwargs)
        self._calc_callable = fn
    
    def _calc_impl(self, q: torch.Tensor) -> torch.Tensor:
        batch_size = q.shape[0]
        
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}. "
                f"Create a new CRBA instance with larger max_batch_size."
            )
        
        # Get sliced views of pre-allocated buffers
        Xup = self.Xup[:, :batch_size, :, :]
        S = self.S[:, :batch_size, :]
        I_composite = self.I_composite[:, :batch_size, :, :]
        M = self.M[:batch_size, :self.nv, :self.nv]
        I_spatial = self.I_spatial.expand(-1, batch_size, -1, -1)
        
        # Zero out buffers
        S.zero_()
        M.zero_()
        I_composite.copy_(I_spatial)
        
        # Split configuration
        if self.is_floating_base:
            q_base, q_joints = q[:, :7], q[:, 7:]
        else:
            q_joints = q

        # Precompute transforms
        axes_batched = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
        T_revolute = axis_and_angle_to_matrix_44(axes_batched, q_joints)
        T_prismatic = axis_and_d_to_pris_matrix(axes_batched, q_joints)
        I44 = identity_transform(batch_size, self.dtype, self.device)

        # ========================================================================
        # Forward pass: compute transforms and joint subspaces
        # ========================================================================
        
        for node_idx in self.chain.topo_order:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]

            T_joint_offset = as_batched_transform(
                self.chain.joint_offsets[node_idx], batch_size, self.dtype, self.device
            )
            T_link_offset = as_batched_transform(
                self.chain.link_offsets[node_idx], batch_size, self.dtype, self.device
            )

            if joint_type_idx == Joint.TYPES.index('revolute'):
                T_motion = T_revolute[:, joint_idx]
            elif joint_type_idx == Joint.TYPES.index('prismatic'):
                T_motion = T_prismatic[:, joint_idx]
            else:
                T_motion = I44

            T_parent_child = T_joint_offset @ T_motion @ T_link_offset
            Xup[node_idx] = spatial_adjoint(inv_homogeneous(T_parent_child))

            # Joint subspace
            if joint_type_idx in [Joint.TYPES.index('revolute'), Joint.TYPES.index('prismatic')]:
                axis_local = self.axes_norm[joint_idx].view(1, 3).expand(batch_size, -1)
                twist_joint = torch.zeros((batch_size, 6), dtype=self.dtype, device=self.device)
                
                if joint_type_idx == Joint.TYPES.index('revolute'):
                    twist_joint[:, 3:] = axis_local
                else:
                    twist_joint[:, :3] = axis_local
                    
                S[node_idx] = (spatial_adjoint(T_link_offset) @ twist_joint.unsqueeze(-1)).squeeze(-1)

        # ========================================================================
        # Backward pass: compute composite inertias
        # ========================================================================
        
        for node_idx in reversed(self.chain.topo_order):
            p = self.chain.parent_list[node_idx]
            if p != -1:
                I_composite[p] = I_composite[p] + (
                    Xup[node_idx].transpose(1, 2) @ I_composite[node_idx] @ Xup[node_idx]
                )

        # ========================================================================
        # Assemble mass matrix
        # ========================================================================
        
        # Base inertia block (if floating)
        if self.is_floating_base:
            M[:, :6, :6] = I_composite[self.urdf_root_idx]

        # Joint columns
        for node_idx in self.chain.topo_order:
            col_idx = self.vel_indices_list[node_idx]  # Python list access - no .item()
            if col_idx == -1:
                continue
                
            S_i = S[node_idx].unsqueeze(-1)
            F_i = I_composite[node_idx] @ S_i

            # Diagonal element
            M[:, col_idx, col_idx] = (S_i.transpose(1, 2) @ F_i).squeeze(-1).squeeze(-1)

            # Base-joint coupling (if floating)
            if self.is_floating_base:
                f_at_base = F_i.clone()
                current_node = node_idx
                max_depth = 100
                depth = 0
                while current_node != self.urdf_root_idx and self.chain.parent_list[current_node] != -1 and depth < max_depth:
                    f_at_base = Xup[current_node].transpose(1, 2) @ f_at_base
                    current_node = self.chain.parent_list[current_node]
                    depth += 1
                
                M[:, :6, col_idx] = f_at_base.squeeze(-1)
                M[:, col_idx, :6] = f_at_base.squeeze(-1)

            # Joint-joint coupling
            f = F_i.clone()
            current_node = node_idx
            max_depth = 100
            depth = 0
            while self.chain.parent_list[current_node] != -1 and depth < max_depth:
                f = Xup[current_node].transpose(1, 2) @ f
                current_node = self.chain.parent_list[current_node]
                depth += 1
                
                parent_col = self.vel_indices_list[current_node]  # Python list access
                if parent_col != -1:
                    S_parent = S[current_node].unsqueeze(-1)
                    value = (S_parent.transpose(1, 2) @ f).squeeze(-1).squeeze(-1)
                    M[:, col_idx, parent_col] = value
                    M[:, parent_col, col_idx] = value

        return M