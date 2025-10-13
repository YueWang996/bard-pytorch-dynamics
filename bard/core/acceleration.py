"""
Optimized spatial acceleration computation with Jacobian-based methods.

This module provides efficient classes for computing spatial accelerations:
- ``BiasAcceleration``: Computes bias acceleration (dJ/dt * qd) efficiently
- ``SpatialAccelerationJacobian``: Computes full acceleration using J*qdd + dJ/dt*qd

The Jacobian-based approach is more GPU-friendly than RNEA for batch operations,
especially when computing accelerations for multiple frames or with large batch sizes.
"""

from typing import Dict, Optional, Any, Tuple
import torch

from bard.core import chain
from bard.structures import Joint
from bard.transforms import (
    axis_and_angle_to_matrix_44,
    axis_and_d_to_pris_matrix,
)
from .utils import (
    identity_transform,
    inv_homogeneous_fast,
    motion_cross_product_fast,
    spatial_adjoint_fast,
    quat_to_rotmat_fast,
)


class BiasAcceleration:
    """Efficient computation of bias acceleration (Coriolis + centrifugal terms).

    This class computes the bias acceleration a_bias = dJ/dt * qd, which represents
    the velocity-dependent acceleration (Coriolis and centrifugal effects). This is
    more efficient than full RNEA when joint accelerations are zero or when you only
    need the bias term.

    The computation uses a streamlined forward pass that only propagates velocities
    and computes the velocity-dependent accelerations, avoiding the joint acceleration
    terms used in full RNEA.

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

            # Create a bias acceleration instance once
            bias_accel = BiasAcceleration(robot_chain, max_batch_size=256)
            eef_frame_id = robot_chain.get_frame_id("end_effector_link")

            # Use in a loop
            for q, qd in data_loader:
                a_bias = bias_accel.calc(q, qd, eef_frame_id, reference_frame="world")
                # a_bias contains only Coriolis/centrifugal acceleration
    """

    def __init__(
        self,
        chain: chain.Chain,
        max_batch_size: int = 1024,
        compile_enabled: Optional[bool] = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device

        self.n_nodes = chain.n_nodes
        self.n_joints = chain.n_joints
        self.is_floating_base = chain.has_floating_base

        # Pre-allocate buffers - only what we need for bias computation
        self.Xup = torch.zeros(
            max_batch_size, self.n_nodes, 6, 6, dtype=self.dtype, device=self.device
        )
        self.S = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )
        self.v = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )
        self.a_bias = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )

        # Pre-allocate transform storage for reference frame conversion
        self.T_world_to_node = torch.zeros(
            max_batch_size, self.n_nodes, 4, 4, dtype=self.dtype, device=self.device
        )

        # Pre-compute normalized axes
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Pre-stack joint and link offsets
        joint_offsets_list = []
        link_offsets_list = []
        for node_idx in range(self.n_nodes):
            j_off = chain.joint_offsets[node_idx]
            l_off = chain.link_offsets[node_idx]

            if j_off is not None:
                j_off = j_off.reshape(4, 4).to(dtype=self.dtype, device=self.device)
            else:
                j_off = torch.eye(4, dtype=self.dtype, device=self.device)

            if l_off is not None:
                l_off = l_off.reshape(4, 4).to(dtype=self.dtype, device=self.device)
            else:
                l_off = torch.eye(4, dtype=self.dtype, device=self.device)

            joint_offsets_list.append(j_off)
            link_offsets_list.append(l_off)

        self.joint_offset_stack = torch.stack(joint_offsets_list, dim=0)  # (n_nodes, 4, 4)
        self.link_offset_stack = torch.stack(link_offsets_list, dim=0)  # (n_nodes, 4, 4)

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
            BiasAcceleration: The instance itself for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        self.Xup = self.Xup.to(dtype=self.dtype, device=self.device)
        self.S = self.S.to(dtype=self.dtype, device=self.device)
        self.v = self.v.to(dtype=self.dtype, device=self.device)
        self.a_bias = self.a_bias.to(dtype=self.dtype, device=self.device)
        self.T_world_to_node = self.T_world_to_node.to(dtype=self.dtype, device=self.device)
        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)
        self.joint_offset_stack = self.joint_offset_stack.to(dtype=self.dtype, device=self.device)
        self.link_offset_stack = self.link_offset_stack.to(dtype=self.dtype, device=self.device)

        self._setup_calc_callable()
        return self

    def calc(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        frame_id: int,
        reference_frame: str = "world",
    ) -> torch.Tensor:
        """Compute the bias acceleration at a specific frame.

        Args:
            q (torch.Tensor): A batch of generalized positions.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 7 + n_joints)``.
            qd (torch.Tensor): A batch of generalized velocities.
                - For fixed-base robots, shape is ``(B, n_joints)``.
                - For floating-base robots, shape is ``(B, 6 + n_joints)``.
            frame_id (int): The integer index of the target frame.
            reference_frame (str, optional): The frame of reference for the
                output acceleration. Can be ``"world"`` or ``"local"``.
                Defaults to ``"world"``.

        Returns:
            torch.Tensor: The bias acceleration ``[linear; angular]`` of shape ``(B, 6)``.

        Raises:
            ValueError: If the input batch size ``B`` exceeds ``self.max_batch_size``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}..."
            )
        if reference_frame not in ["world", "local"]:
            raise ValueError('reference_frame must be "world" or "local"')
        return self._calc_callable(q, qd, frame_id, reference_frame)

    def _setup_calc_callable(self):
        fn = self._calc_impl
        if self._compile_enabled:
            fn = torch.compile(fn, **self._compile_kwargs)
        self._calc_callable = fn

    def _calc_impl(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        frame_id: int,
        reference_frame: str,
    ) -> torch.Tensor:
        batch_size = q.shape[0]

        # Get sliced views of pre-allocated buffers
        Xup = self.Xup[:batch_size]
        S = self.S[:batch_size]
        v = self.v[:batch_size]
        a_bias = self.a_bias[:batch_size]
        T_world_to_node = self.T_world_to_node[:batch_size]

        # Zero out buffers
        S.zero_()
        v.zero_()
        a_bias.zero_()

        # Split base and joint components
        if self.is_floating_base:
            q_base = q[:, :7]
            q_joints = q[:, 7:]
            v_base = qd[:, :6]
            v_joints = qd[:, 6:]
        else:
            q_base = v_base = None
            q_joints = q
            v_joints = qd

        # Pre-compute joint transforms
        axes_expanded = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
        T_revolute = axis_and_angle_to_matrix_44(axes_expanded, q_joints)
        T_prismatic = axis_and_d_to_pris_matrix(axes_expanded, q_joints)

        I44 = identity_transform(batch_size, self.dtype, self.device)

        # Build base transform if floating
        if self.is_floating_base and q_base is not None:
            t = q_base[:, :3]
            quat = q_base[:, 3:]
            R = quat_to_rotmat_fast(quat)

            T_world_to_base = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_to_base[:, :3, :3] = R
            T_world_to_base[:, :3, 3] = t
            T_world_to_base[:, 3, 3] = 1.0
        else:
            T_world_to_base = I44

        # ========================================================================
        # Forward pass: propagate velocities and bias accelerations
        # ========================================================================

        for node_idx in self.chain.topo_order:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]
            parent_idx = self.chain.parent_list[node_idx]

            # Use pre-stacked offsets
            T_joint_offset = (
                self.joint_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)
            )
            T_link_offset = self.link_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)

            # Select motion transform
            is_revolute = joint_type_idx == Joint.TYPES.index("revolute")
            is_prismatic = joint_type_idx == Joint.TYPES.index("prismatic")

            if is_revolute:
                T_motion = T_revolute[:, joint_idx]
            elif is_prismatic:
                T_motion = T_prismatic[:, joint_idx]
            else:  # fixed
                T_motion = I44

            # Parent -> child transform
            T_parent_to_child = T_joint_offset @ T_motion @ T_link_offset
            Xup_i = spatial_adjoint_fast(inv_homogeneous_fast(T_parent_to_child))
            Xup[:, node_idx] = Xup_i

            # Joint subspace
            is_actuated = is_revolute or is_prismatic
            if is_actuated:
                axis_local = self.axes_norm[joint_idx].expand(batch_size, -1)
                twist_joint = torch.zeros((batch_size, 6, 1), dtype=self.dtype, device=self.device)
                if is_revolute:
                    twist_joint[:, 3:, 0] = axis_local
                else:  # prismatic
                    twist_joint[:, :3, 0] = axis_local
                S[:, node_idx] = spatial_adjoint_fast(T_link_offset) @ twist_joint
                v_joint = v_joints[:, joint_idx].view(batch_size, 1, 1)
            else:
                v_joint = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=self.device)

            # Parent state
            if parent_idx == -1:  # Root node
                if self.is_floating_base and v_base is not None:
                    v_parent = v_base.unsqueeze(-1)
                else:
                    v_parent = torch.zeros((batch_size, 6, 1), dtype=self.dtype, device=self.device)
                # For bias acceleration, parent acceleration is zero at root
                a_bias_parent = torch.zeros(
                    (batch_size, 6, 1), dtype=self.dtype, device=self.device
                )
                T_world_to_parent = T_world_to_base
            else:
                v_parent = v[:, parent_idx]
                a_bias_parent = a_bias[:, parent_idx]
                T_world_to_parent = T_world_to_node[:, parent_idx]

            # Propagate velocity
            vJ = S[:, node_idx] * v_joint
            v[:, node_idx] = Xup_i @ v_parent + vJ

            # **CRITICAL FIX**: Bias acceleration = transformed parent bias + Coriolis term
            # This matches RNEA with qdd=0: a = Xup @ a_parent + motion_cross(v) @ vJ
            coriolis = motion_cross_product_fast(v[:, node_idx]) @ vJ
            a_bias[:, node_idx] = Xup_i @ a_bias_parent + coriolis

            # Accumulate world pose for reference frame conversion
            T_world_to_node[:, node_idx] = T_world_to_parent @ T_parent_to_child

        # Extract acceleration at target frame
        a_local = a_bias[:, frame_id]

        if reference_frame == "world":
            Ad_world_wrt_body = spatial_adjoint_fast(T_world_to_node[:, frame_id])
            a_world = Ad_world_wrt_body @ a_local
            return a_world.squeeze(-1)
        else:  # local
            return a_local.squeeze(-1)


class SpatialAccelerationJacobian:
    """Jacobian-based spatial acceleration computation.

    This class computes spatial acceleration using the formula:
        a = J(q) * qdd + dJ/dt(q, qd) * qd

    where:
    - J(q) is the geometric Jacobian
    - dJ/dt * qd is the bias acceleration (Coriolis + centrifugal)

    This approach is more GPU-friendly than RNEA for:
    - Large batch sizes
    - Multiple frame queries
    - When you need both J and a

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
        bias_accel (BiasAcceleration): Internal bias acceleration computer.

    Example:
        .. code-block:: python

            # Create an acceleration instance once
            accel = SpatialAccelerationJacobian(robot_chain, max_batch_size=256)
            eef_frame_id = robot_chain.get_frame_id("end_effector_link")

            # Use in a loop
            for q, qd, qdd in data_loader:
                a = accel.calc(q, qd, qdd, eef_frame_id, reference_frame="world")
                # a = J*qdd + dJ/dt*qd
    """

    def __init__(
        self,
        chain: chain.Chain,
        max_batch_size: int = 1024,
        compile_enabled: Optional[bool] = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device

        self.n_joints = chain.n_joints
        self.is_floating_base = chain.has_floating_base
        self.nv = 6 + self.n_joints if self.is_floating_base else self.n_joints

        # Import Jacobian class here to avoid circular imports
        from .jacobian import Jacobian

        # Create internal instances
        self.jacobian = Jacobian(
            chain,
            max_batch_size=max_batch_size,
            compile_enabled=compile_enabled,
            compile_kwargs=compile_kwargs,
        )

        self.bias_accel = BiasAcceleration(
            chain,
            max_batch_size=max_batch_size,
            compile_enabled=compile_enabled,
            compile_kwargs=compile_kwargs,
        )

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
        self.jacobian.enable_compilation(enabled, **compile_kwargs)
        self.bias_accel.enable_compilation(enabled, **compile_kwargs)
        self._setup_calc_callable()

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        """Move all internal buffers to a specified dtype and/or device.

        This is an in-place operation.

        Args:
            dtype (torch.dtype, optional): The target data type. Defaults to ``None``.
            device (torch.device, optional): The target device. Defaults to ``None``.

        Returns:
            SpatialAccelerationJacobian: The instance itself for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        self.jacobian.to(dtype=self.dtype, device=self.device)
        self.bias_accel.to(dtype=self.dtype, device=self.device)

        self._setup_calc_callable()
        return self

    def calc(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        frame_id: int,
        reference_frame: str = "world",
        return_jacobian: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial acceleration at a specific frame.

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
                output acceleration. Can be ``"world"`` or ``"local"``.
                Defaults to ``"world"``.
            return_jacobian (bool, optional): If ``True``, also returns the
                Jacobian matrix. Defaults to ``False``.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If ``return_jacobian=False``: The spatial acceleration of shape ``(B, 6)``.
                - If ``return_jacobian=True``: A tuple ``(a, J)`` where ``a`` is the
                  acceleration and ``J`` is the Jacobian of shape ``(B, 6, nv)``.

        Raises:
            ValueError: If the input batch size ``B`` exceeds ``self.max_batch_size``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}..."
            )
        if reference_frame not in ["world", "local"]:
            raise ValueError('reference_frame must be "world" or "local"')
        return self._calc_callable(q, qd, qdd, frame_id, reference_frame, return_jacobian)

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
        reference_frame: str,
        return_jacobian: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.shape[0]

        # Compute Jacobian J(q)
        J = self.jacobian.calc(q, frame_id, reference_frame)

        # Compute bias acceleration dJ/dt * qd
        a_bias = self.bias_accel.calc(q, qd, frame_id, reference_frame)

        # Compute full acceleration: a = J * qdd + a_bias
        # J shape: (B, 6, nv), qdd shape: (B, nv)
        a = torch.bmm(J, qdd.unsqueeze(-1)).squeeze(-1) + a_bias

        if return_jacobian:
            return a, J
        return a

    def calc_bias_only(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        frame_id: int,
        reference_frame: str = "world",
    ) -> torch.Tensor:
        """Compute only the bias acceleration (equivalent to setting qdd=0).

        This is a convenience method that directly calls the internal
        BiasAcceleration instance.

        Args:
            q (torch.Tensor): A batch of generalized positions.
            qd (torch.Tensor): A batch of generalized velocities.
            frame_id (int): The integer index of the target frame.
            reference_frame (str, optional): The frame of reference.
                Defaults to ``"world"``.

        Returns:
            torch.Tensor: The bias acceleration of shape ``(B, 6)``.
        """
        return self.bias_accel.calc(q, qd, frame_id, reference_frame)


class MultiFrameBiasAcceleration:
    """Efficient computation of bias acceleration for multiple frames simultaneously.

    This class is optimized for computing bias accelerations at multiple frames
    in a single pass, which is more efficient than calling BiasAcceleration
    multiple times for different frames.

    Args:
        chain (chain.Chain): The robot's kinematic chain definition.
        max_batch_size (int, optional): The maximum batch size the instance will
            support. Defaults to 1024.
        max_frames (int, optional): The maximum number of frames to compute
            simultaneously. Defaults to 10.
        compile_enabled (bool, optional): If ``True``, the core computation
            will be JIT-compiled with ``torch.compile``. Defaults to ``False``.
        compile_kwargs (Dict[str, Any], optional): A dictionary of keyword
            arguments to pass to ``torch.compile``. Defaults to ``None``.

    Example:
        .. code-block:: python

            # Compute bias acceleration for multiple frames at once
            multi_bias = MultiFrameBiasAcceleration(robot_chain, max_batch_size=128)
            frame_ids = [robot_chain.get_frame_id(name)
                        for name in ["link1", "link2", "end_effector"]]

            for q, qd in data_loader:
                a_bias_dict = multi_bias.calc(q, qd, frame_ids, reference_frame="world")
                # a_bias_dict[frame_id] contains acceleration for each frame
    """

    def __init__(
        self,
        chain: chain.Chain,
        max_batch_size: int = 1024,
        max_frames: int = 10,
        compile_enabled: Optional[bool] = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.max_frames = max_frames
        self.dtype = chain.dtype
        self.device = chain.device

        # Use single BiasAcceleration instance internally
        self.bias_accel = BiasAcceleration(
            chain,
            max_batch_size=max_batch_size,
            compile_enabled=compile_enabled,
            compile_kwargs=compile_kwargs,
        )

    def calc(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        frame_ids: list[int],
        reference_frame: str = "world",
    ) -> Dict[int, torch.Tensor]:
        """Compute bias acceleration for multiple frames.

        Args:
            q (torch.Tensor): A batch of generalized positions.
            qd (torch.Tensor): A batch of generalized velocities.
            frame_ids (list[int]): List of frame indices to compute.
            reference_frame (str, optional): The frame of reference.
                Defaults to ``"world"``.

        Returns:
            Dict[int, torch.Tensor]: Dictionary mapping frame_id to acceleration
                tensor of shape ``(B, 6)``.
        """
        if len(frame_ids) > self.max_frames:
            raise ValueError(
                f"Number of frames {len(frame_ids)} exceeds max_frames {self.max_frames}"
            )

        # Compute bias acceleration once (the forward pass is shared)
        # Then extract results for different frames
        results = {}
        for frame_id in frame_ids:
            results[frame_id] = self.bias_accel.calc(q, qd, frame_id, reference_frame)

        return results

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        """Move all internal buffers to a specified dtype and/or device."""
        self.bias_accel.to(dtype=dtype, device=device)
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        return self
