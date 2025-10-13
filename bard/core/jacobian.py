"""
Optimized Jacobian computation with pre-allocated memory.

This module provides efficient Jacobian matrix computation for robot kinematics,
optimized with JIT-compiled utility functions and pre-allocated memory buffers.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from bard.core import chain
from bard.structures import Joint
from bard.transforms.rotation_conversions import (
    axis_and_angle_to_matrix_44,
    axis_and_d_to_pris_matrix,
)

from .utils import (
    identity_transform,
    as_batched_transform,
    inv_homogeneous,
    spatial_adjoint,
    normalize_axis,
)


class Jacobian:
    """Optimized Jacobian matrix computation with pre-allocated memory.

    This class computes the geometric Jacobian matrix that relates joint velocities
    to end-effector spatial velocity. The Jacobian can be computed in either the
    world frame or the local (body) frame.

    Args:
        chain (chain.Chain): The robot's kinematic chain definition.
        max_batch_size (int, optional): The maximum batch size the instance will
            support. Defaults to 1024.
        compile_enabled (bool, optional): If ``True``, the core computation
            will be JIT-compiled with ``torch.compile``. Defaults to ``False``.
        compile_kwargs (Dict[str, Any], optional): A dictionary of keyword
            arguments to pass to ``torch.compile``.
            Defaults to ``{"mode": "reduce-overhead"}``.

    Attributes:
        chain (chain.Chain): The robot kinematic chain.
        max_batch_size (int): The maximum supported batch size.
        dtype (torch.dtype): The data type of the tensors.
        device (torch.device): The device where tensors are stored.
        nv (int): The dimension of velocity space.

    Example:
        .. code-block:: python

            # Create a Jacobian instance once
            jac = Jacobian(robot_chain, max_batch_size=128)
            eef_frame_id = robot_chain.get_frame_id("end_effector_link")

            # Use in a loop for efficient computation
            for q in data_loader:
                J_world = jac.calc(q, eef_frame_id, reference_frame="world")
                # ... use the Jacobian ...
    """

    def __init__(
        self,
        chain: chain.Chain,
        max_batch_size: int = 1024,
        compile_enabled: Optional[bool] = False,
        compile_kwargs: Optional[Dict[str, Any]] = {"mode": "reduce-overhead"},
    ):
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device

        self.n_nodes = chain.n_nodes
        self.n_joints = chain.n_joints
        self.is_floating_base = chain.has_floating_base
        self.nv = 6 + self.n_joints if self.is_floating_base else self.n_joints

        # Pre-allocate Jacobian matrices (the main memory consumers)
        self.J_local = torch.zeros(max_batch_size, 6, self.nv, dtype=self.dtype, device=self.device)
        self.J_world = torch.zeros(max_batch_size, 6, self.nv, dtype=self.dtype, device=self.device)

        # Pre-compute static data
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
            Jacobian: The instance itself for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        self.J_local = self.J_local.to(dtype=self.dtype, device=self.device)
        self.J_world = self.J_world.to(dtype=self.dtype, device=self.device)
        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)

        self._setup_calc_callable()

        return self

    def calc(
        self,
        q: torch.Tensor,
        frame_id: int,
        reference_frame: str,
        return_eef_pose: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Jacobian matrix for a specific frame.

        Args:
            q (torch.Tensor): A batch of generalized positions.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 7 + n_joints)``.
            frame_id (int): The integer index of the target frame.
            reference_frame (str): The frame of reference for the
                Jacobian. Can be ``"world"`` or ``"local"``.
            return_eef_pose (bool, optional): If ``True``, also returns the
                world-frame pose of the target frame. Defaults to ``False``.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If ``return_eef_pose=False``: The Jacobian matrix of shape ``(B, 6, nv)``.
                - If ``return_eef_pose=True``: A tuple ``(J, T_world_to_frame)`` where
                  ``J`` is the Jacobian and ``T_world_to_frame`` is the pose matrix
                  of shape ``(B, 4, 4)``.

        Raises:
            ValueError: If the input batch size ``B`` exceeds ``self.max_batch_size``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}. "
                f"Create a new Jacobian instance with larger max_batch_size."
            )
        if reference_frame not in ["world", "local"]:
            raise ValueError('reference_frame must be "world" or "local"')
        return self._calc_callable(q, frame_id, reference_frame, return_eef_pose)

    def _setup_calc_callable(self):
        fn = self._calc_impl
        if self._compile_enabled:
            fn = torch.compile(fn, **self._compile_kwargs)
        self._calc_callable = fn

    def _calc_impl(
        self,
        q: torch.Tensor,
        frame_id: int,
        reference_frame: str,
        return_eef_pose: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.shape[0]

        # Get path from root to target frame as Python list
        path_nodes = self.chain.parents_indices_list[frame_id]

        # Get sliced view of pre-allocated Jacobian
        J_local = self.J_local[:batch_size, :, :]
        J_local.zero_()

        # Split configuration
        if self.is_floating_base:
            q_base, q_joints = q[:, :7], q[:, 7:]
            nv_base = 6
        else:
            q_base = None
            q_joints = q
            nv_base = 0

        # Pre-compute per-joint motion transforms
        axes_expanded = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
        T_revolute = axis_and_angle_to_matrix_44(axes_expanded, q_joints)
        T_prismatic = axis_and_d_to_pris_matrix(axes_expanded, q_joints)

        I44 = identity_transform(batch_size, self.dtype, self.device)

        # ========================================================================
        # Build base transform for floating-base robots
        # ========================================================================

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

            T_world_to_base = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_to_base[:, :3, :3] = R_base_to_world
            T_world_to_base[:, :3, 3] = t
            T_world_to_base[:, 3, 3] = 1.0
            T_world_to_current = T_world_to_base
        else:
            T_world_to_base = I44.clone()
            T_world_to_current = I44.clone()

        # ========================================================================
        # Forward pass: accumulate transforms and collect actuated joints
        # ========================================================================

        # Build Python lists for active joints (same as original function-based code)
        world_T_joint_origin = []
        active_joint_types = []
        active_joint_cols = []
        active_joint_axes_local = []

        for node_idx in path_nodes:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]

            T_joint_offset = as_batched_transform(
                self.chain.joint_offsets[node_idx], batch_size, self.dtype, self.device
            )

            # Transform to joint origin in world frame
            T_world_to_joint_origin = T_world_to_current @ T_joint_offset

            # Store information for actuated joints
            is_revolute = joint_type_idx == Joint.TYPES.index("revolute")
            is_prismatic = joint_type_idx == Joint.TYPES.index("prismatic")

            if is_revolute or is_prismatic:
                world_T_joint_origin.append(T_world_to_joint_origin)
                active_joint_types.append(joint_type_idx)
                active_joint_cols.append(joint_idx)
                active_joint_axes_local.append(self.axes_norm[joint_idx])

            # Select motion transform based on joint type
            if is_revolute:
                T_motion = T_revolute[:, joint_idx]
            elif is_prismatic:
                T_motion = T_prismatic[:, joint_idx]
            else:  # fixed
                T_motion = I44

            T_link_offset = as_batched_transform(
                self.chain.link_offsets[node_idx], batch_size, self.dtype, self.device
            )

            # Update cumulative transform
            T_world_to_current = T_world_to_current @ T_joint_offset @ T_motion @ T_link_offset

        T_world_to_frame = T_world_to_current

        # ========================================================================
        # Compute Jacobian in local frame
        # ========================================================================

        T_frame_to_world = inv_homogeneous(T_world_to_frame)

        # Floating base Jacobian columns
        if self.is_floating_base and q_base is not None:
            T_frame_to_base = T_frame_to_world @ T_world_to_base
            Ad_frame_wrt_base = spatial_adjoint(T_frame_to_base)
            J_local[:, :, :nv_base] = Ad_frame_wrt_base

        # Articulated joint Jacobian columns (same as original function-based code)
        for jtype, jcol, axis_local, T_w_to_j in zip(
            active_joint_types, active_joint_cols, active_joint_axes_local, world_T_joint_origin
        ):
            # Transform from frame to joint origin
            T_frame_to_joint_origin = T_frame_to_world @ T_w_to_j
            Ad_frame_wrt_joint = spatial_adjoint(T_frame_to_joint_origin)

            # Compute joint twist in joint frame
            axis_local_batch = axis_local.view(1, 3).expand(batch_size, -1)
            axis_local_unit = normalize_axis(axis_local_batch)

            twist_joint = torch.zeros((batch_size, 6), dtype=self.dtype, device=self.device)
            if jtype == Joint.TYPES.index("revolute"):
                twist_joint[:, 3:] = axis_local_unit  # Angular velocity
            else:  # prismatic
                twist_joint[:, :3] = axis_local_unit  # Linear velocity

            # Transform twist to frame and store in Jacobian column
            col_vec = (Ad_frame_wrt_joint @ twist_joint.unsqueeze(-1)).squeeze(-1)
            J_local[:, :, nv_base + jcol] = col_vec

        # ========================================================================
        # Transform to requested reference frame
        # ========================================================================

        if reference_frame == "world":
            # Transform to world frame using adjoint
            Ad_world_wrt_frame = spatial_adjoint(T_world_to_frame)
            J = Ad_world_wrt_frame @ J_local
        else:  # local
            J = J_local

        if return_eef_pose:
            return J, T_world_to_frame
        return J
