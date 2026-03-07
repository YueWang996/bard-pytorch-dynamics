"""
Unified robot dynamics interface with kinematics state caching.

This module provides the ``RobotDynamics`` class, the primary v2 API for all
kinematics and dynamics computations. It eliminates redundant kinematic tree
traversals by computing shared quantities once via ``update_kinematics`` and
reusing the cached ``KinematicsState`` across all subsequent algorithm calls.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from bard.core.chain import Chain
from bard.core.state import KinematicsState
from bard.structures import Joint
from bard.transforms import (
    axis_and_angle_to_matrix_44,
    axis_and_d_to_pris_matrix,
)
from .utils import (
    identity_transform,
    as_batched_transform,
    inv_homogeneous_fast,
    motion_cross_product_fast,
    force_cross_product_fast,
    spatial_adjoint_fast,
    quat_to_rotmat_fast,
    spatial_adjoint,
    inv_homogeneous,
    normalize_axis,
)


class RobotDynamics:
    """Unified interface for batched robot kinematics and dynamics.

    This class stores static robot data (topology, inertias, pre-stacked offsets)
    once and provides a kinematics state cache to eliminate redundant computation
    when multiple algorithms are called for the same ``(q, qd)`` pair.

    **Typical usage (cached workflow for RL):**

    .. code-block:: python

        rd = RobotDynamics(chain, max_batch_size=4096)
        eef_id = chain.get_frame_id("end_effector")

        # In the control loop:
        state = rd.update_kinematics(q, qd)  # ONE tree traversal
        T = rd.forward_kinematics(eef_id, state)
        J = rd.jacobian(eef_id, state, reference_frame="world")
        tau = rd.rnea(qdd, state, gravity=gravity)
        M = rd.crba(state)

    **Standalone usage (single algorithm, no explicit cache):**

    .. code-block:: python

        T = rd.fk(q, frame_id)  # path-only traversal, no cache

    Args:
        chain: The robot's kinematic chain.
        max_batch_size: Maximum supported batch size for pre-allocated buffers.
        compile_enabled: If True, JIT-compile internal methods with ``torch.compile``.
        compile_kwargs: Additional kwargs for ``torch.compile``.
    """

    def __init__(
        self,
        chain: Chain,
        max_batch_size: int = 1024,
        compile_enabled: bool = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device

        self.n_nodes = chain.n_nodes
        self.n_joints = chain.n_joints
        self.is_floating_base = chain.has_floating_base
        self.nv = 6 + self.n_joints if self.is_floating_base else self.n_joints

        # --- Static data (ONE copy, shared across all algorithms) ---
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.I_spatial = chain.spatial_inertias  # (n_nodes, 6, 6)
        self.gravity = torch.tensor([0.0, 0.0, -9.81], dtype=self.dtype, device=self.device)

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

        # Velocity index mapping (for CRBA mass matrix assembly)
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

        # --- Pre-allocate state cache buffers ---
        B = max_batch_size
        N = self.n_nodes
        self._buf_T_pc = torch.zeros(B, N, 4, 4, dtype=self.dtype, device=self.device)
        self._buf_Xup = torch.zeros(B, N, 6, 6, dtype=self.dtype, device=self.device)
        self._buf_S = torch.zeros(B, N, 6, 1, dtype=self.dtype, device=self.device)
        self._buf_T_world = torch.zeros(B, N, 4, 4, dtype=self.dtype, device=self.device)
        self._buf_v = torch.zeros(B, N, 6, 1, dtype=self.dtype, device=self.device)

        # Pre-allocate algorithm-specific buffers
        self._buf_a = torch.zeros(B, N, 6, 1, dtype=self.dtype, device=self.device)
        self._buf_f = torch.zeros(B, N, 6, 1, dtype=self.dtype, device=self.device)
        self._buf_I_composite = torch.zeros(B, N, 6, 6, dtype=self.dtype, device=self.device)
        self._buf_M = torch.zeros(B, self.nv, self.nv, dtype=self.dtype, device=self.device)
        self._buf_J_local = torch.zeros(B, 6, self.nv, dtype=self.dtype, device=self.device)

        # --- Compilation ---
        self._compile_enabled = compile_enabled
        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}
        self._setup_callables()

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "RobotDynamics":
        """Move all internal buffers to a specified dtype and/or device.

        Args:
            dtype: Target data type.
            device: Target device.

        Returns:
            Self for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)
        self.I_spatial = self.I_spatial.to(dtype=self.dtype, device=self.device)
        self.gravity = self.gravity.to(dtype=self.dtype, device=self.device)
        self.joint_offset_stack = self.joint_offset_stack.to(dtype=self.dtype, device=self.device)
        self.link_offset_stack = self.link_offset_stack.to(dtype=self.dtype, device=self.device)

        for name in [
            "_buf_T_pc", "_buf_Xup", "_buf_S", "_buf_T_world", "_buf_v",
            "_buf_a", "_buf_f", "_buf_I_composite", "_buf_M", "_buf_J_local",
        ]:
            setattr(self, name, getattr(self, name).to(dtype=self.dtype, device=self.device))

        self._setup_callables()
        return self

    def enable_compilation(self, enabled: bool = True, **compile_kwargs):
        """Enable or disable ``torch.compile`` for internal methods.

        Args:
            enabled: Whether to enable compilation.
            **compile_kwargs: Additional kwargs for ``torch.compile``.
        """
        self._compile_enabled = enabled
        if compile_kwargs:
            self._compile_kwargs.update(compile_kwargs)
        self._setup_callables()

    def _setup_callables(self):
        self._update_kinematics_fn = self._update_kinematics_impl
        self._fk_fn = self._fk_impl
        self._rnea_fn = self._rnea_impl
        self._crba_fn = self._crba_impl
        self._jacobian_fn = self._jacobian_impl
        self._spatial_acceleration_fn = self._spatial_acceleration_impl
        if self._compile_enabled:
            kwargs = self._compile_kwargs.copy()
            self._update_kinematics_fn = torch.compile(self._update_kinematics_impl, **kwargs)
            self._fk_fn = torch.compile(self._fk_impl, **kwargs)
            self._rnea_fn = torch.compile(self._rnea_impl, **kwargs)
            self._crba_fn = torch.compile(self._crba_impl, **kwargs)
            self._jacobian_fn = torch.compile(self._jacobian_impl, **kwargs)
            self._spatial_acceleration_fn = torch.compile(
                self._spatial_acceleration_impl, **kwargs
            )

    # ========================================================================
    # Core: update_kinematics
    # ========================================================================

    def update_kinematics(
        self,
        q: torch.Tensor,
        qd: Optional[torch.Tensor] = None,
    ) -> KinematicsState:
        """Compute and cache all shared kinematic quantities in a single tree traversal.

        This is the core method of the v2 API. Call it once per control step,
        then pass the returned ``KinematicsState`` to all algorithm methods.

        If ``qd`` is provided, velocity-level quantities (spatial velocities)
        are also computed. If ``qd`` is ``None``, only position-level quantities
        are computed (sufficient for FK, Jacobian, CRBA).

        Args:
            q: Generalized positions ``(B, nq)``.
            qd: Generalized velocities ``(B, nv)``, or ``None``.

        Returns:
            A ``KinematicsState`` containing cached quantities.

        Raises:
            ValueError: If batch size exceeds ``max_batch_size``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}."
            )
        return self._update_kinematics_fn(q, qd)

    def _update_kinematics_impl(
        self,
        q: torch.Tensor,
        qd: Optional[torch.Tensor],
    ) -> KinematicsState:
        batch_size = q.shape[0]
        compute_velocity = qd is not None

        # Slice pre-allocated buffers
        T_pc = self._buf_T_pc[:batch_size]
        Xup = self._buf_Xup[:batch_size]
        S = self._buf_S[:batch_size]
        T_world = self._buf_T_world[:batch_size]
        v = self._buf_v[:batch_size]

        S.zero_()
        v.zero_()

        # Split configuration
        if self.is_floating_base:
            q_base = q[:, :7]
            q_joints = q[:, 7:]
            if compute_velocity:
                v_base = qd[:, :6]
                v_joints = qd[:, 6:]
            else:
                v_base = v_joints = None
        else:
            q_base = None
            q_joints = q
            v_base = None
            v_joints = qd if compute_velocity else None

        # Pre-compute ALL per-joint motion transforms at once
        axes_expanded = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
        T_revolute = axis_and_angle_to_matrix_44(axes_expanded, q_joints)
        T_prismatic = axis_and_d_to_pris_matrix(axes_expanded, q_joints)

        I44 = torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Base transform
        if self.is_floating_base and q_base is not None:
            t = q_base[:, :3]
            quat = q_base[:, 3:]
            R = quat_to_rotmat_fast(quat)
            T_world_base = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_base[:, :3, :3] = R
            T_world_base[:, :3, 3] = t
            T_world_base[:, 3, 3] = 1.0
        else:
            T_world_base = I44

        # Forward pass through topological order
        for node_idx in self.chain.topo_order:
            j_idx = self.chain.joint_indices_list[node_idx]
            j_type = self.chain.joint_type_indices_list[node_idx]
            p_idx = self.chain.parent_list[node_idx]

            T_joint_offset = self.joint_offset_stack[node_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            T_link_offset = self.link_offset_stack[node_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )

            is_revolute = j_type == Joint.TYPES.index("revolute")
            is_prismatic = j_type == Joint.TYPES.index("prismatic")

            if is_revolute:
                T_motion = T_revolute[:, j_idx]
            elif is_prismatic:
                T_motion = T_prismatic[:, j_idx]
            else:
                T_motion = I44

            # 1. T_parent_to_child
            T_pc_i = T_joint_offset @ T_motion @ T_link_offset
            T_pc[:, node_idx] = T_pc_i

            # 2. Xup = Ad(inv(T_pc))
            Xup[:, node_idx] = spatial_adjoint_fast(inv_homogeneous_fast(T_pc_i))

            # 3. Joint subspace S
            is_actuated = is_revolute or is_prismatic
            if is_actuated:
                axis_local = self.axes_norm[j_idx].expand(batch_size, -1)
                twist_joint = torch.zeros(batch_size, 6, 1, dtype=self.dtype, device=self.device)
                if is_revolute:
                    twist_joint[:, 3:, 0] = axis_local
                else:
                    twist_joint[:, :3, 0] = axis_local
                S[:, node_idx] = spatial_adjoint_fast(T_link_offset) @ twist_joint

            # 4. T_world
            if p_idx == -1:
                T_world[:, node_idx] = T_world_base @ T_pc_i
            else:
                T_world[:, node_idx] = T_world[:, p_idx] @ T_pc_i

            # 5. Velocity (if requested)
            if compute_velocity:
                if is_actuated and v_joints is not None:
                    v_joint = v_joints[:, j_idx].view(batch_size, 1, 1)
                else:
                    v_joint = torch.zeros(batch_size, 1, 1, dtype=self.dtype, device=self.device)

                if p_idx == -1:
                    if self.is_floating_base and v_base is not None:
                        v_parent = v_base.unsqueeze(-1)
                    else:
                        v_parent = torch.zeros(
                            batch_size, 6, 1, dtype=self.dtype, device=self.device
                        )
                else:
                    v_parent = v[:, p_idx]

                vJ = S[:, node_idx] * v_joint
                v[:, node_idx] = Xup[:, node_idx] @ v_parent + vJ

        return KinematicsState(
            T_parent_to_child=T_pc,
            Xup=Xup,
            S=S,
            T_world=T_world,
            v=v,
            batch_size=batch_size,
            has_velocity=compute_velocity,
        )

    # ========================================================================
    # Cached algorithms
    # ========================================================================

    def forward_kinematics(
        self,
        frame_id: int,
        state: KinematicsState,
    ) -> torch.Tensor:
        """World-frame pose of a frame (O(1) lookup from cached state).

        Args:
            frame_id: Target frame index.
            state: Cached kinematics state from ``update_kinematics``.

        Returns:
            Homogeneous transform ``(B, 4, 4)``.
        """
        return state.T_world[:state.batch_size, frame_id]

    def jacobian(
        self,
        frame_id: int,
        state: KinematicsState,
        reference_frame: str = "world",
        return_eef_pose: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Geometric Jacobian using cached state.

        Args:
            frame_id: Target frame index.
            state: Cached kinematics state.
            reference_frame: ``"world"`` or ``"local"``.
            return_eef_pose: If True, also returns the world-frame pose.

        Returns:
            Jacobian ``(B, 6, nv)``, optionally with pose ``(B, 4, 4)``.
        """
        if reference_frame not in ["world", "local"]:
            raise ValueError('reference_frame must be "world" or "local"')
        return self._jacobian_fn(frame_id, state, reference_frame, return_eef_pose)

    def rnea(
        self,
        qdd: torch.Tensor,
        state: KinematicsState,
        gravity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inverse dynamics (RNEA) using cached state.

        Requires ``state.has_velocity == True``.

        Args:
            qdd: Generalized accelerations ``(B, nv)``.
            state: Cached kinematics state (must include velocities).
            gravity: 3-element gravity vector. Defaults to ``[0, 0, -9.81]``.

        Returns:
            Generalized forces ``(B, nv)``.
        """
        if not state.has_velocity:
            raise ValueError(
                "RNEA requires velocity data. "
                "Call update_kinematics(q, qd) with qd provided."
            )
        return self._rnea_fn(qdd, state, gravity)

    def crba(
        self,
        state: KinematicsState,
    ) -> torch.Tensor:
        """Mass matrix (CRBA) using cached state.

        Args:
            state: Cached kinematics state.

        Returns:
            Mass matrix ``(B, nv, nv)``.
        """
        return self._crba_fn(state)

    def spatial_acceleration(
        self,
        qdd: torch.Tensor,
        frame_id: int,
        state: KinematicsState,
        reference_frame: str = "world",
    ) -> torch.Tensor:
        """Spatial acceleration using cached state.

        Requires ``state.has_velocity == True``.

        Args:
            qdd: Generalized accelerations ``(B, nv)``.
            frame_id: Target frame index.
            state: Cached kinematics state (must include velocities).
            reference_frame: ``"world"`` or ``"local"``.

        Returns:
            Spatial acceleration ``(B, 6)`` as ``[linear; angular]``.
        """
        if not state.has_velocity:
            raise ValueError(
                "Spatial acceleration requires velocity data. "
                "Call update_kinematics(q, qd) with qd provided."
            )
        if reference_frame not in ["world", "local"]:
            raise ValueError('reference_frame must be "world" or "local"')
        return self._spatial_acceleration_fn(qdd, frame_id, state, reference_frame)

    # ========================================================================
    # Standalone FK (path-only, no cache)
    # ========================================================================

    def fk(
        self,
        q: torch.Tensor,
        frame_id: int,
    ) -> torch.Tensor:
        """Standalone forward kinematics via path-only traversal.

        This does NOT populate or use the cache. It only traverses the path
        from root to the target frame, making it efficient for single-frame
        queries without needing a full tree traversal.

        Args:
            q: Generalized positions ``(B, nq)``.
            frame_id: Target frame index.

        Returns:
            Homogeneous transform ``(B, 4, 4)``.
        """
        batch_size = q.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}."
            )
        return self._fk_fn(q, frame_id)

    # ========================================================================
    # Implementation methods
    # ========================================================================

    def _fk_impl(self, q: torch.Tensor, frame_id: int) -> torch.Tensor:
        batch_size = q.shape[0]
        path_nodes = self.chain.parents_indices_list[frame_id]

        if self.is_floating_base:
            q_base = q[:, :7]
            q_joints = q[:, 7:]
        else:
            q_base = None
            q_joints = q

        axes_expanded = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
        T_revolute = axis_and_angle_to_matrix_44(axes_expanded, q_joints)
        T_prismatic = axis_and_d_to_pris_matrix(axes_expanded, q_joints)
        I44 = identity_transform(batch_size, self.dtype, self.device)

        if self.is_floating_base and q_base is not None:
            t = q_base[:, :3]
            quat = q_base[:, 3:]
            R = quat_to_rotmat_fast(quat)
            T_world_to_current = torch.zeros(
                batch_size, 4, 4, dtype=self.dtype, device=self.device
            )
            T_world_to_current[:, :3, :3] = R
            T_world_to_current[:, :3, 3] = t
            T_world_to_current[:, 3, 3] = 1.0
        else:
            T_world_to_current = I44.clone()

        for node_idx in path_nodes:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]

            T_joint_offset = self.joint_offset_stack[node_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )

            is_revolute = joint_type_idx == Joint.TYPES.index("revolute")
            is_prismatic = joint_type_idx == Joint.TYPES.index("prismatic")

            if is_revolute:
                T_motion = T_revolute[:, joint_idx]
            elif is_prismatic:
                T_motion = T_prismatic[:, joint_idx]
            else:
                T_motion = I44

            T_link_offset = self.link_offset_stack[node_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            T_world_to_current = T_world_to_current @ T_joint_offset @ T_motion @ T_link_offset

        return T_world_to_current

    def _jacobian_impl(
        self,
        frame_id: int,
        state: KinematicsState,
        reference_frame: str,
        return_eef_pose: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = state.batch_size
        path_nodes = self.chain.parents_indices_list[frame_id]

        J_local = self._buf_J_local[:batch_size, :, :]
        J_local.zero_()

        nv_base = 6 if self.is_floating_base else 0

        T_world_to_frame = state.T_world[:batch_size, frame_id]
        T_frame_to_world = inv_homogeneous_fast(T_world_to_frame)

        # Floating base Jacobian columns
        if self.is_floating_base:
            T_world_to_base = state.T_world[:batch_size, self.chain.topo_order[0]]
            T_frame_to_base = T_frame_to_world @ T_world_to_base
            Ad_frame_wrt_base = spatial_adjoint_fast(T_frame_to_base)
            J_local[:, :, :nv_base] = Ad_frame_wrt_base

        # Joint Jacobian columns
        for node_idx in path_nodes:
            joint_idx = self.chain.joint_indices_list[node_idx]
            joint_type_idx = self.chain.joint_type_indices_list[node_idx]

            is_revolute = joint_type_idx == Joint.TYPES.index("revolute")
            is_prismatic = joint_type_idx == Joint.TYPES.index("prismatic")

            if not (is_revolute or is_prismatic):
                continue

            # World transform of this node's joint origin
            # T_world_to_joint_origin = T_world[:, parent] @ T_joint_offset @ ...
            # But we can approximate using T_world[:, node] and back-transform by link_offset
            # Actually, we need the transform at the joint origin, not at the link frame.
            # The joint origin is: T_world_parent @ T_joint_offset
            # We have T_world[:, node] = T_world_parent @ T_joint_offset @ T_motion @ T_link_offset
            # So T_world_joint_origin = T_world[:, node] @ inv(T_link_offset) @ inv(T_motion)
            # Alternatively, reconstruct from parent:
            p_idx = self.chain.parent_list[node_idx]
            if p_idx == -1:
                if self.is_floating_base:
                    T_world_parent = state.T_world[:batch_size, self.chain.topo_order[0]]
                else:
                    T_world_parent = torch.eye(
                        4, dtype=self.dtype, device=self.device
                    ).unsqueeze(0).expand(batch_size, -1, -1)
            else:
                T_world_parent = state.T_world[:batch_size, p_idx]

            T_joint_offset = self.joint_offset_stack[node_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            T_world_to_joint_origin = T_world_parent @ T_joint_offset

            T_frame_to_joint_origin = T_frame_to_world @ T_world_to_joint_origin
            Ad_frame_wrt_joint = spatial_adjoint_fast(T_frame_to_joint_origin)

            axis_local = self.axes_norm[joint_idx]
            axis_local_batch = axis_local.view(1, 3).expand(batch_size, -1)

            twist_joint = torch.zeros(batch_size, 6, dtype=self.dtype, device=self.device)
            if is_revolute:
                twist_joint[:, 3:] = axis_local_batch
            else:
                twist_joint[:, :3] = axis_local_batch

            col_vec = (Ad_frame_wrt_joint @ twist_joint.unsqueeze(-1)).squeeze(-1)
            J_local[:, :, nv_base + joint_idx] = col_vec

        # Transform to requested reference frame
        if reference_frame == "world":
            Ad_world_wrt_frame = spatial_adjoint_fast(T_world_to_frame)
            J = Ad_world_wrt_frame @ J_local
        else:
            J = J_local.clone()

        if return_eef_pose:
            return J, T_world_to_frame
        return J

    def _rnea_impl(
        self,
        qdd: torch.Tensor,
        state: KinematicsState,
        gravity: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = state.batch_size
        Xup = state.Xup
        S = state.S
        v = state.v

        # Slice algorithm-specific buffers
        a = self._buf_a[:batch_size]
        f = self._buf_f[:batch_size]
        a.zero_()
        f.zero_()

        # Split qdd
        if self.is_floating_base:
            a_base = qdd[:, :6]
            a_joints = qdd[:, 6:]
        else:
            a_base = None
            a_joints = qdd

        # Gravity
        g = self.gravity if gravity is None else gravity
        a_gravity_world = torch.zeros(batch_size, 6, 1, dtype=self.dtype, device=self.device)
        a_gravity_world[:, :3, 0] = -g.expand(batch_size, -1)

        if self.is_floating_base:
            # Transform gravity to base frame
            T_world_base = state.T_world[:batch_size, self.chain.topo_order[0]]
            Ad_base_world = spatial_adjoint_fast(inv_homogeneous_fast(T_world_base))
            a_gravity_base = Ad_base_world @ a_gravity_world
        else:
            a_gravity_base = a_gravity_world

        # Forward pass: compute accelerations using cached Xup, S, v
        for node_idx in self.chain.topo_order:
            j_idx = self.chain.joint_indices_list[node_idx]
            j_type = self.chain.joint_type_indices_list[node_idx]
            p_idx = self.chain.parent_list[node_idx]

            is_revolute = j_type == Joint.TYPES.index("revolute")
            is_prismatic = j_type == Joint.TYPES.index("prismatic")
            is_actuated = is_revolute or is_prismatic

            if is_actuated:
                a_joint = a_joints[:, j_idx].view(batch_size, 1, 1)
            else:
                a_joint = torch.zeros(batch_size, 1, 1, dtype=self.dtype, device=self.device)

            # Parent acceleration
            if p_idx == -1:
                if self.is_floating_base:
                    a_parent = a_base.unsqueeze(-1) + a_gravity_base
                else:
                    a_parent = a_gravity_base
            else:
                a_parent = a[:, p_idx]

            # Compute vJ from cached v: vJ = S * qd (already encoded in v)
            # v[i] = Xup[i] @ v[parent] + S[i] * qd_i
            # So vJ = v[i] - Xup[i] @ v[parent]
            if p_idx == -1:
                if self.is_floating_base:
                    # Need the base velocity to compute vJ
                    # v_parent for root was set during update_kinematics
                    # We can reconstruct: vJ = v[node] - Xup[node] @ v_base
                    # But we don't have v_base stored... We need to reconstruct it.
                    # Actually, the velocity was stored correctly in update_kinematics.
                    # For the root node: v[root] = Xup[root] @ v_base + S[root] * qd_joint
                    # We need vJ = S[root] * qd_joint = v[root] - Xup[root] @ v_base
                    # Simpler: just compute vJ from S and the joint velocity
                    pass
                else:
                    pass

            # Use the same Coriolis computation pattern as original RNEA
            # We know S is cached, and we can extract qd_joint from the velocity
            if is_actuated:
                # Reconstruct v_joint from the cached data
                # In update_kinematics: vJ = S[:, node] * v_joint_scalar
                # And v[:, node] = Xup[:, node] @ v_parent + vJ
                # We can get vJ = v[:, node] - Xup[:, node] @ v_parent
                if p_idx == -1:
                    if self.is_floating_base:
                        # We need to find v_base. For the first node in topo_order,
                        # v_parent = v_base which is qd[:, :6]
                        # But we don't have qd stored. So compute from cached v.
                        v_parent_node = torch.zeros(
                            batch_size, 6, 1, dtype=self.dtype, device=self.device
                        )
                    else:
                        v_parent_node = torch.zeros(
                            batch_size, 6, 1, dtype=self.dtype, device=self.device
                        )
                else:
                    v_parent_node = v[:batch_size, p_idx]

                vJ = v[:batch_size, node_idx] - Xup[:batch_size, node_idx] @ v_parent_node
            else:
                vJ = torch.zeros(batch_size, 6, 1, dtype=self.dtype, device=self.device)

            coriolis = motion_cross_product_fast(v[:batch_size, node_idx]) @ vJ
            temp = Xup[:batch_size, node_idx] @ a_parent
            a[:, node_idx] = temp + S[:batch_size, node_idx] * a_joint + coriolis

        # Backward pass: force propagation
        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)

        for node_idx in reversed(self.chain.topo_order):
            Iv = I_spatial_batched[:, node_idx] @ v[:batch_size, node_idx]
            f_node = I_spatial_batched[:, node_idx] @ a[:, node_idx]
            f_node = f_node + force_cross_product_fast(v[:batch_size, node_idx]) @ Iv

            children = self.chain.children_list[node_idx]
            if len(children) == 1:
                child_idx = children[0]
                f_node = (
                    f_node
                    + Xup[:batch_size, child_idx].transpose(1, 2) @ f[:, child_idx]
                )
            elif len(children) > 1:
                for child_idx in children:
                    f_node = (
                        f_node
                        + Xup[:batch_size, child_idx].transpose(1, 2) @ f[:, child_idx]
                    )

            f[:, node_idx] = f_node

        # Extract generalized forces
        tau_all_nodes = (S[:batch_size] * f).sum(dim=2).squeeze(-1)

        if self.is_floating_base:
            tau = torch.zeros(
                batch_size, 6 + self.n_joints, dtype=self.dtype, device=self.device
            )
            tau[:, :6] = f[:, self.urdf_root_idx, :, 0]
            for node_idx in range(self.n_nodes):
                j_type = self.chain.joint_type_indices_list[node_idx]
                if j_type in [Joint.TYPES.index("revolute"), Joint.TYPES.index("prismatic")]:
                    j_col = self.chain.joint_indices_list[node_idx]
                    tau[:, 6 + j_col] = tau_all_nodes[:, node_idx]
        else:
            tau = torch.zeros(batch_size, self.n_joints, dtype=self.dtype, device=self.device)
            for node_idx in range(self.n_nodes):
                j_type = self.chain.joint_type_indices_list[node_idx]
                if j_type in [Joint.TYPES.index("revolute"), Joint.TYPES.index("prismatic")]:
                    j_col = self.chain.joint_indices_list[node_idx]
                    tau[:, j_col] = tau_all_nodes[:, node_idx]

        return tau

    def _crba_impl(self, state: KinematicsState) -> torch.Tensor:
        batch_size = state.batch_size
        Xup = state.Xup
        S = state.S

        I_composite = self._buf_I_composite[:batch_size]
        M = self._buf_M[:batch_size, :self.nv, :self.nv]

        M.zero_()
        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)
        I_composite.copy_(I_spatial_batched)

        # Backward pass: composite inertia accumulation
        for node_idx in reversed(self.chain.topo_order):
            p = self.chain.parent_list[node_idx]
            if p != -1:
                I_composite[:, p] += (
                    Xup[:batch_size, node_idx].transpose(1, 2)
                    @ I_composite[:, node_idx]
                    @ Xup[:batch_size, node_idx]
                )

        # Assemble mass matrix
        if self.is_floating_base:
            M[:, :6, :6] = I_composite[:, self.urdf_root_idx]

        for node_idx in self.chain.topo_order:
            col_idx = self.vel_indices_list[node_idx]
            if col_idx == -1:
                continue

            S_i = S[:batch_size, node_idx]
            F_i = I_composite[:, node_idx] @ S_i

            # Diagonal element
            M[:, col_idx, col_idx] = (S_i.transpose(1, 2) @ F_i).squeeze(-1).squeeze(-1)

            # Base-joint coupling
            if self.is_floating_base:
                f_at_base = F_i
                current_node = node_idx
                while current_node != self.urdf_root_idx:
                    parent = self.chain.parent_list[current_node]
                    if parent == -1:
                        break
                    f_at_base = Xup[:batch_size, current_node].transpose(1, 2) @ f_at_base
                    current_node = parent
                M[:, :6, col_idx] = f_at_base.squeeze(-1)
                M[:, col_idx, :6] = f_at_base.squeeze(-1)

            # Joint-joint coupling
            f_prop = F_i
            current_node = node_idx
            while True:
                parent = self.chain.parent_list[current_node]
                if parent == -1:
                    break
                f_prop = Xup[:batch_size, current_node].transpose(1, 2) @ f_prop
                current_node = parent
                parent_col = self.vel_indices_list[current_node]
                if parent_col != -1:
                    S_parent = S[:batch_size, current_node]
                    value = (S_parent.transpose(1, 2) @ f_prop).squeeze(-1).squeeze(-1)
                    M[:, col_idx, parent_col] = value
                    M[:, parent_col, col_idx] = value

        return M

    def _spatial_acceleration_impl(
        self,
        qdd: torch.Tensor,
        frame_id: int,
        state: KinematicsState,
        reference_frame: str,
    ) -> torch.Tensor:
        batch_size = state.batch_size
        Xup = state.Xup
        S = state.S
        v = state.v

        a = self._buf_a[:batch_size]
        a.zero_()

        if self.is_floating_base:
            a_base = qdd[:, :6]
            a_joints = qdd[:, 6:]
        else:
            a_base = None
            a_joints = qdd

        # No gravity for pure kinematic acceleration
        a_world0 = torch.zeros(batch_size, 6, 1, dtype=self.dtype, device=self.device)

        # Forward pass using cached Xup, S, v
        for node_idx in self.chain.topo_order:
            j_idx = self.chain.joint_indices_list[node_idx]
            j_type = self.chain.joint_type_indices_list[node_idx]
            p_idx = self.chain.parent_list[node_idx]

            is_revolute = j_type == Joint.TYPES.index("revolute")
            is_prismatic = j_type == Joint.TYPES.index("prismatic")
            is_actuated = is_revolute or is_prismatic

            if is_actuated:
                a_joint = a_joints[:, j_idx].view(batch_size, 1, 1)
            else:
                a_joint = torch.zeros(batch_size, 1, 1, dtype=self.dtype, device=self.device)

            if p_idx == -1:
                if self.is_floating_base:
                    a_parent = a_base.unsqueeze(-1) + a_world0
                else:
                    a_parent = a_world0
            else:
                a_parent = a[:, p_idx]

            # Reconstruct vJ from cached data
            if is_actuated:
                if p_idx == -1:
                    v_parent_node = torch.zeros(
                        batch_size, 6, 1, dtype=self.dtype, device=self.device
                    )
                else:
                    v_parent_node = v[:batch_size, p_idx]
                vJ = v[:batch_size, node_idx] - Xup[:batch_size, node_idx] @ v_parent_node
            else:
                vJ = torch.zeros(batch_size, 6, 1, dtype=self.dtype, device=self.device)

            coriolis = motion_cross_product_fast(v[:batch_size, node_idx]) @ vJ
            a[:, node_idx] = (
                Xup[:batch_size, node_idx] @ a_parent
                + S[:batch_size, node_idx] * a_joint
                + coriolis
            )

        # Extract acceleration at target frame
        a_local = a[:, frame_id]

        if reference_frame == "world":
            Ad_world_wrt_body = spatial_adjoint_fast(state.T_world[:batch_size, frame_id])
            a_world = Ad_world_wrt_body @ a_local
            return a_world.squeeze(-1)
        else:
            return a_local.squeeze(-1)
