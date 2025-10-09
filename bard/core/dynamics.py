"""
Class-based robot dynamics algorithms with pre-allocated memory.

This module implements the core dynamics algorithms for the ``bard`` library.
It includes class-based, optimized implementations of the Recursive Newton-Euler
Algorithm (RNEA) for inverse dynamics and the Composite Rigid Body Algorithm (CRBA)
for calculating the mass matrix.
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
    inv_homogeneous_fast,
    spatial_adjoint_fast,
    motion_cross_product_fast,
    force_cross_product_fast,
    quat_to_rotmat_fast,
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

        # Optimized memory layout: (batch, n_nodes, ...) for better cache locality
        self.Xup = torch.zeros(
            max_batch_size, self.n_nodes, 6, 6, dtype=self.dtype, device=self.device
        )
        self.S = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )
        self.v = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )
        self.a = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )
        self.f = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )

        # Pre-compute static data
        self.I_spatial = chain.spatial_inertias  # (n_nodes, 6, 6)
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.gravity = torch.tensor([0.0, 0.0, -9.81], dtype=self.dtype, device=self.device)

        # Convert chain metadata to tensors for GPU operations
        self.topo_order_tensor = torch.tensor(
            chain.topo_order, dtype=torch.long, device=self.device
        )
        self.joint_indices_tensor = torch.tensor(
            chain.joint_indices_list, dtype=torch.long, device=self.device
        )
        self.joint_type_tensor = torch.tensor(
            chain.joint_type_indices_list, dtype=torch.long, device=self.device
        )
        self.parent_tensor = torch.tensor(chain.parent_list, dtype=torch.long, device=self.device)

        # Create tensor indicating which nodes are actuated
        self.is_revolute = self.joint_type_tensor == Joint.TYPES.index("revolute")
        self.is_prismatic = self.joint_type_tensor == Joint.TYPES.index("prismatic")
        self.is_actuated = self.is_revolute | self.is_prismatic

        # Pre-compute joint offset and link offset transforms as tensors
        # Stack them for vectorized operations
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

        # Stack into (n_nodes, 4, 4) tensors
        self.joint_offset_stack = torch.stack(joint_offsets_list, dim=0)  # (n_nodes, 4, 4)
        self.link_offset_stack = torch.stack(link_offsets_list, dim=0)  # (n_nodes, 4, 4)

        # Conditional compilation setup
        self._compile_enabled = compile_enabled
        self._compile_kwargs: Dict[str, Any] = dict(compile_kwargs) if compile_kwargs else {}

        # Compile individual methods for better optimization
        if compile_enabled:
            self._compile_kwargs.setdefault("mode", "max-autotune")
            self._compile_kwargs.setdefault("fullgraph", False)

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

        self.topo_order_tensor = self.topo_order_tensor.to(device=self.device)
        self.joint_indices_tensor = self.joint_indices_tensor.to(device=self.device)
        self.joint_type_tensor = self.joint_type_tensor.to(device=self.device)
        self.parent_tensor = self.parent_tensor.to(device=self.device)
        self.is_revolute = self.is_revolute.to(device=self.device)
        self.is_prismatic = self.is_prismatic.to(device=self.device)
        self.is_actuated = self.is_actuated.to(device=self.device)

        self.joint_offset_stack = self.joint_offset_stack.to(dtype=self.dtype, device=self.device)
        self.link_offset_stack = self.link_offset_stack.to(dtype=self.dtype, device=self.device)

        self._setup_calc_callable()
        return self

    def calc(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        gravity: Optional[torch.Tensor] = None,
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
        if self._compile_enabled:
            # Try to compile with fullgraph=True for maximum optimization
            compile_kwargs = self._compile_kwargs.copy()
            compile_kwargs.setdefault("fullgraph", False)  # Start with False, can try True
            self._calc_callable = torch.compile(self._calc_impl, **compile_kwargs)
        else:
            self._calc_callable = self._calc_impl

    def _calc_impl(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        gravity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = q.shape[0]

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}."
            )

        # Get sliced views of pre-allocated buffers
        Xup = self.Xup[:batch_size]
        S = self.S[:batch_size]
        v = self.v[:batch_size]
        a = self.a[:batch_size]
        f = self.f[:batch_size]

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
        a_gravity_world = torch.zeros((batch_size, 6, 1), dtype=self.dtype, device=self.device)
        a_gravity_world[:, :3, 0] = -g.expand(batch_size, -1)

        # Pre-compute transforms ONLY for actuated joints (more memory efficient)
        # Create mapping from joint_idx to its transform
        joint_transforms_revolute = {}
        joint_transforms_prismatic = {}

        for node_idx in self.chain.topo_order:
            j_idx = self.chain.joint_indices_list[node_idx]
            j_type = self.chain.joint_type_indices_list[node_idx]

            if j_idx >= 0:  # Only for actuated joints
                if j_type == Joint.TYPES.index("revolute"):
                    if j_idx not in joint_transforms_revolute:
                        axis = self.axes_norm[j_idx].expand(batch_size, -1)
                        joint_transforms_revolute[j_idx] = axis_and_angle_to_matrix_44(
                            axis, q_joints[:, j_idx]
                        )
                elif j_type == Joint.TYPES.index("prismatic"):
                    if j_idx not in joint_transforms_prismatic:
                        axis = self.axes_norm[j_idx].expand(batch_size, -1)
                        joint_transforms_prismatic[j_idx] = axis_and_d_to_pris_matrix(
                            axis, q_joints[:, j_idx]
                        )

        # Build base transform if floating
        if self.is_floating_base:
            t = q_base[:, :3]
            quat = q_base[:, 3:]
            R = quat_to_rotmat_fast(quat)

            T_world_to_base = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_to_base[:, :3, :3] = R
            T_world_to_base[:, :3, 3] = t
            T_world_to_base[:, 3, 3] = 1.0

            Ad_base_world = spatial_adjoint_fast(inv_homogeneous_fast(T_world_to_base))
            a_gravity_base = Ad_base_world @ a_gravity_world
        else:
            a_gravity_base = a_gravity_world

        # Cache for commonly used identity and offset transforms
        I44 = (
            torch.eye(4, dtype=self.dtype, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # ========================================================================
        # Forward pass: propagate velocities and accelerations
        # ========================================================================

        for node_idx in self.chain.topo_order:
            j_idx = self.chain.joint_indices_list[node_idx]
            j_type = self.chain.joint_type_indices_list[node_idx]
            p_idx = self.chain.parent_list[node_idx]

            # Get pre-computed offset transforms (cached, no allocation)
            T_joint_offset = (
                self.joint_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)
            )
            T_link_offset = self.link_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)

            is_revolute = j_type == Joint.TYPES.index("revolute")
            is_prismatic = j_type == Joint.TYPES.index("prismatic")
            is_actuated = is_revolute or is_prismatic

            # Use pre-computed transforms from dictionary
            if is_revolute:
                T_motion = joint_transforms_revolute[j_idx]
            elif is_prismatic:
                T_motion = joint_transforms_prismatic[j_idx]
            else:
                T_motion = I44

            T_parent_child = T_joint_offset @ T_motion @ T_link_offset
            Xup[:, node_idx] = spatial_adjoint_fast(inv_homogeneous_fast(T_parent_child))

            # Joint subspace
            if is_actuated:
                axis_local = self.axes_norm[j_idx].expand(batch_size, -1)
                twist_joint = torch.zeros(batch_size, 6, 1, dtype=self.dtype, device=self.device)

                if is_revolute:
                    twist_joint[:, 3:, 0] = axis_local
                elif is_prismatic:
                    twist_joint[:, :3, 0] = axis_local

                S[:, node_idx] = spatial_adjoint_fast(T_link_offset) @ twist_joint
                v_joint = v_joints[:, j_idx].view(batch_size, 1, 1)
                a_joint = a_joints[:, j_idx].view(batch_size, 1, 1)
            else:
                v_joint = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=self.device)
                a_joint = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=self.device)

            # Parent state
            if p_idx == -1:  # Root node
                if self.is_floating_base:
                    v_parent = v_base.unsqueeze(-1)
                    a_parent = a_base.unsqueeze(-1) + a_gravity_base
                else:
                    v_parent = torch.zeros((batch_size, 6, 1), dtype=self.dtype, device=self.device)
                    a_parent = a_gravity_base
            else:
                v_parent = v[:, p_idx]
                a_parent = a[:, p_idx]

            vJ = S[:, node_idx] * v_joint

            # Use torch.addmm for better performance when possible
            v[:, node_idx] = torch.baddbmm(vJ, Xup[:, node_idx], v_parent, beta=1.0, alpha=1.0)

            coriolis = motion_cross_product_fast(v[:, node_idx]) @ vJ
            temp = Xup[:, node_idx] @ a_parent
            a[:, node_idx] = temp + S[:, node_idx] * a_joint + coriolis

        # ========================================================================
        # Backward pass: compute forces
        # ========================================================================

        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)

        for node_idx in reversed(self.chain.topo_order):
            # Compute inertial forces
            Iv = I_spatial_batched[:, node_idx] @ v[:, node_idx]
            f_node = I_spatial_batched[:, node_idx] @ a[:, node_idx]
            f_node = f_node + force_cross_product_fast(v[:, node_idx]) @ Iv

            # Aggregate forces from children (unrolled for common cases)
            children = self.chain.children_list[node_idx]
            if len(children) == 1:
                # Common case: single child - direct computation
                child_idx = children[0]
                f_node = f_node + Xup[:, child_idx].transpose(1, 2) @ f[:, child_idx]
            elif len(children) > 1:
                # Multiple children
                for child_idx in children:
                    f_node = f_node + Xup[:, child_idx].transpose(1, 2) @ f[:, child_idx]

            f[:, node_idx] = f_node

        # ========================================================================
        # Extract generalized forces (optimized)
        # ========================================================================

        # More efficient extraction: (B, n_nodes, 6, 1) * (B, n_nodes, 6, 1) -> (B, n_nodes)
        tau_all_nodes = (S * f).sum(dim=2).squeeze(-1)  # (B, n_nodes)

        if self.is_floating_base:
            tau = torch.zeros((batch_size, 6 + self.n_joints), dtype=self.dtype, device=self.device)
            urdf_root_idx = 1 if self.n_nodes > 1 else 0
            tau[:, :6] = f[:, urdf_root_idx, :, 0]

            # Vectorized extraction for joint torques
            for node_idx in range(self.n_nodes):
                j_type = self.chain.joint_type_indices_list[node_idx]
                if j_type in [Joint.TYPES.index("revolute"), Joint.TYPES.index("prismatic")]:
                    j_col = self.chain.joint_indices_list[node_idx]
                    tau[:, 6 + j_col] = tau_all_nodes[:, node_idx]
        else:
            tau = torch.zeros((batch_size, self.n_joints), dtype=self.dtype, device=self.device)
            for node_idx in range(self.n_nodes):
                j_type = self.chain.joint_type_indices_list[node_idx]
                if j_type in [Joint.TYPES.index("revolute"), Joint.TYPES.index("prismatic")]:
                    j_col = self.chain.joint_indices_list[node_idx]
                    tau[:, j_col] = tau_all_nodes[:, node_idx]

        return tau


# ============================================================================
# CRBA Class
# ============================================================================


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

        # Optimized memory layout: (batch, n_nodes, ...)
        self.Xup = torch.zeros(
            max_batch_size, self.n_nodes, 6, 6, dtype=self.dtype, device=self.device
        )
        self.S = torch.zeros(
            max_batch_size, self.n_nodes, 6, 1, dtype=self.dtype, device=self.device
        )
        self.I_composite = torch.zeros(
            max_batch_size, self.n_nodes, 6, 6, dtype=self.dtype, device=self.device
        )
        self.M = torch.zeros(max_batch_size, self.nv, self.nv, dtype=self.dtype, device=self.device)

        # Pre-compute static data
        self.I_spatial = chain.spatial_inertias  # (n_nodes, 6, 6)
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Pre-compute velocity index mapping
        self.vel_indices_list = []
        for node_idx in range(self.n_nodes):
            joint_type_idx = chain.joint_type_indices_list[node_idx]
            if joint_type_idx != Joint.TYPES.index("fixed"):
                joint_col = chain.joint_indices_list[node_idx]
                vel_idx = (6 + joint_col) if self.is_floating_base else joint_col
                self.vel_indices_list.append(vel_idx)
            else:
                self.vel_indices_list.append(-1)

        self.urdf_root_idx = 1 if (self.is_floating_base and self.n_nodes > 1) else 0

        # Pre-compute ancestor paths for efficient force propagation
        self.ancestor_paths = []
        self.ancestor_masks = []  # Masks for vectorized operations
        max_depth = max(len(self.chain.parents_indices_list[i]) for i in range(self.n_nodes))

        for node_idx in range(self.n_nodes):
            path = []
            current = node_idx
            while self.chain.parent_list[current] != -1:
                current = self.chain.parent_list[current]
                path.append(current)

            # Pad path to max_depth for vectorization
            path_tensor = torch.full((max_depth,), -1, dtype=torch.long, device=self.device)
            if len(path) > 0:
                path_tensor[: len(path)] = torch.tensor(path, dtype=torch.long, device=self.device)

            mask = torch.zeros(max_depth, dtype=torch.bool, device=self.device)
            mask[: len(path)] = True

            self.ancestor_paths.append(path_tensor)
            self.ancestor_masks.append(mask)

        # Stack for efficient indexing
        self.ancestor_paths_tensor = torch.stack(self.ancestor_paths)  # (n_nodes, max_depth)
        self.ancestor_masks_tensor = torch.stack(self.ancestor_masks)  # (n_nodes, max_depth)

        # Pre-compute joint offset and link offset transforms
        # Ensure all matrices are exactly 2D (4, 4)
        self.joint_offset_stack = []
        self.link_offset_stack = []
        for node_idx in range(self.n_nodes):
            j_off = chain.joint_offsets[node_idx]
            l_off = chain.link_offsets[node_idx]

            # Ensure 2D (4, 4) shape by reshaping
            if j_off is not None:
                j_off = j_off.reshape(4, 4).to(dtype=self.dtype, device=self.device)
                self.joint_offset_stack.append(j_off)
            else:
                self.joint_offset_stack.append(torch.eye(4, dtype=self.dtype, device=self.device))

            if l_off is not None:
                l_off = l_off.reshape(4, 4).to(dtype=self.dtype, device=self.device)
                self.link_offset_stack.append(l_off)
            else:
                self.link_offset_stack.append(torch.eye(4, dtype=self.dtype, device=self.device))

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

        self.ancestor_paths_tensor = self.ancestor_paths_tensor.to(device=self.device)
        self.ancestor_masks_tensor = self.ancestor_masks_tensor.to(device=self.device)

        self.joint_offset_stack = [
            m.to(dtype=self.dtype, device=self.device) for m in self.joint_offset_stack
        ]
        self.link_offset_stack = [
            m.to(dtype=self.dtype, device=self.device) for m in self.link_offset_stack
        ]

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
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}."
            )

        # Get sliced views of pre-allocated buffers
        Xup = self.Xup[:batch_size]
        S = self.S[:batch_size]
        I_composite = self.I_composite[:batch_size]
        M = self.M[:batch_size, : self.nv, : self.nv]

        # Zero out and initialize buffers
        S.zero_()
        M.zero_()
        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)
        I_composite.copy_(I_spatial_batched)

        # Split configuration
        if self.is_floating_base:
            q_base, q_joints = q[:, :7], q[:, 7:]
        else:
            q_joints = q

        # Pre-compute transforms ONLY for actuated joints (more memory efficient)
        joint_transforms_revolute = {}
        joint_transforms_prismatic = {}

        for node_idx in self.chain.topo_order:
            j_idx = self.chain.joint_indices_list[node_idx]
            j_type = self.chain.joint_type_indices_list[node_idx]

            if j_idx >= 0:  # Only for actuated joints
                if j_type == Joint.TYPES.index("revolute"):
                    if j_idx not in joint_transforms_revolute:
                        axis = self.axes_norm[j_idx].expand(batch_size, -1)
                        joint_transforms_revolute[j_idx] = axis_and_angle_to_matrix_44(
                            axis, q_joints[:, j_idx]
                        )
                elif j_type == Joint.TYPES.index("prismatic"):
                    if j_idx not in joint_transforms_prismatic:
                        axis = self.axes_norm[j_idx].expand(batch_size, -1)
                        joint_transforms_prismatic[j_idx] = axis_and_d_to_pris_matrix(
                            axis, q_joints[:, j_idx]
                        )

        # Cache identity transform
        I44 = (
            torch.eye(4, dtype=self.dtype, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # ========================================================================
        # Forward pass: compute transforms and joint subspaces
        # ========================================================================

        for node_idx in self.chain.topo_order:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]

            T_joint_offset = (
                self.joint_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)
            )
            T_link_offset = self.link_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)

            if joint_type_idx == Joint.TYPES.index("revolute"):
                T_motion = joint_transforms_revolute[joint_idx]
            elif joint_type_idx == Joint.TYPES.index("prismatic"):
                T_motion = joint_transforms_prismatic[joint_idx]
            else:
                T_motion = I44

            T_parent_child = T_joint_offset @ T_motion @ T_link_offset
            Xup[:, node_idx] = spatial_adjoint_fast(inv_homogeneous_fast(T_parent_child))

            # Joint subspace
            if joint_type_idx in [Joint.TYPES.index("revolute"), Joint.TYPES.index("prismatic")]:
                axis_local = self.axes_norm[joint_idx].view(1, 3).expand(batch_size, -1)
                twist_joint = torch.zeros((batch_size, 6, 1), dtype=self.dtype, device=self.device)

                if joint_type_idx == Joint.TYPES.index("revolute"):
                    twist_joint[:, 3:, 0] = axis_local
                else:
                    twist_joint[:, :3, 0] = axis_local

                S[:, node_idx] = spatial_adjoint_fast(T_link_offset) @ twist_joint

        # ========================================================================
        # Backward pass: compute composite inertias
        # ========================================================================

        for node_idx in reversed(self.chain.topo_order):
            p = self.chain.parent_list[node_idx]
            if p != -1:
                # Inplace addition for efficiency
                I_composite[:, p] += (
                    Xup[:, node_idx].transpose(1, 2) @ I_composite[:, node_idx] @ Xup[:, node_idx]
                )

        # ========================================================================
        # Assemble mass matrix (optimized with reduced loops)
        # ========================================================================

        # Base inertia block (if floating)
        if self.is_floating_base:
            M[:, :6, :6] = I_composite[:, self.urdf_root_idx]

        # Process all joints
        for node_idx in self.chain.topo_order:
            col_idx = self.vel_indices_list[node_idx]
            if col_idx == -1:
                continue

            S_i = S[:, node_idx]  # (B, 6, 1)
            F_i = I_composite[:, node_idx] @ S_i  # (B, 6, 1)

            # Diagonal element
            M[:, col_idx, col_idx] = (S_i.transpose(1, 2) @ F_i).squeeze(-1).squeeze(-1)

            # Base-joint coupling (if floating)
            if self.is_floating_base:
                # Propagate force to base
                f_at_base = F_i
                current_node = node_idx

                # Optimized loop with break condition
                while current_node != self.urdf_root_idx:
                    parent = self.chain.parent_list[current_node]
                    if parent == -1:
                        break
                    f_at_base = Xup[:, current_node].transpose(1, 2) @ f_at_base
                    current_node = parent

                M[:, :6, col_idx] = f_at_base.squeeze(-1)
                M[:, col_idx, :6] = f_at_base.squeeze(-1)

            # Joint-joint coupling - propagate up the tree
            f = F_i
            current_node = node_idx

            while True:
                parent = self.chain.parent_list[current_node]
                if parent == -1:
                    break

                f = Xup[:, current_node].transpose(1, 2) @ f
                current_node = parent

                parent_col = self.vel_indices_list[current_node]
                if parent_col != -1:
                    S_parent = S[:, current_node]
                    value = (S_parent.transpose(1, 2) @ f).squeeze(-1).squeeze(-1)
                    M[:, col_idx, parent_col] = value
                    M[:, parent_col, col_idx] = value

        return M
