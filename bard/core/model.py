"""
Immutable robot model containing structure, topology, and algorithm kernels.

The ``Model`` class is the primary object in the v0.3 API. It absorbs the
``Chain`` (kinematic tree topology) and the static precomputed data that was
previously spread across ``RobotDynamics.__init__``. All algorithm
implementations live here as private methods that accept a ``Data`` workspace.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from bard.core.chain import Chain
from bard.core.data import Data
from bard.structures import Joint
from bard.transforms import (
    axis_and_angle_to_matrix_44,
    axis_and_d_to_pris_matrix,
)
from .utils import (
    as_batched_transform,
    inv_homogeneous_fast,
    spatial_adjoint_fast,
    quat_to_rotmat_fast,
    spatial_adjoint,
    inv_homogeneous,
    normalize_axis,
)


class Model:
    """Immutable robot model for batched kinematics and dynamics.

    A ``Model`` holds the robot's kinematic tree topology, pre-computed inertias,
    joint axes, offset matrices, and compilation settings. It is constructed once
    (typically via :func:`bard.build_model_from_urdf`) and then used with one or
    more :class:`~bard.core.data.Data` workspaces to perform computations.

    Attributes:
        nq (int): Configuration space dimension.
        nv (int): Velocity space dimension.
        n_joints (int): Number of actuated joints.
        n_frames (int): Total number of frames (links) in the kinematic tree.
        has_floating_base (bool): Whether the robot has a 6-DOF floating base.
    """

    def __init__(
        self,
        chain: Chain,
        compile_enabled: bool = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._chain = chain
        self.dtype = chain.dtype
        self.device = chain.device

        # --- Topology (proxy from chain) ---
        self.n_frames = chain.n_nodes
        self.n_joints = chain.n_joints
        self.has_floating_base = chain.has_floating_base

        # --- Static precomputed data (from RobotDynamics) ---
        self.axes_norm = chain.axes / chain.axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.I_spatial = chain.spatial_inertias
        self.gravity = torch.tensor([0.0, 0.0, -9.81], dtype=self.dtype, device=self.device)

        # Pre-stack joint and link offsets
        joint_offsets_list = []
        link_offsets_list = []
        for node_idx in range(self.n_frames):
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

        self.joint_offset_stack = torch.stack(joint_offsets_list, dim=0)
        self.link_offset_stack = torch.stack(link_offsets_list, dim=0)

        # --- Cached joint type constants (avoid repeated list.index lookups) ---
        self._FIXED = Joint.TYPES.index("fixed")
        self._REVOLUTE = Joint.TYPES.index("revolute")
        self._PRISMATIC = Joint.TYPES.index("prismatic")

        # Velocity index mapping (for CRBA mass matrix assembly)
        self.vel_indices_list = []
        for node_idx in range(self.n_frames):
            joint_type_idx = chain.joint_type_indices_list[node_idx]
            if joint_type_idx != self._FIXED:
                joint_col = chain.joint_indices_list[node_idx]
                vel_idx = (6 + joint_col) if self.has_floating_base else joint_col
                self.vel_indices_list.append(vel_idx)
            else:
                self.vel_indices_list.append(-1)

        self.urdf_root_idx = 1 if (self.has_floating_base and self.n_frames > 1) else 0

        # Pre-computed per-node flags
        self._is_actuated = [
            chain.joint_type_indices_list[i] in (self._REVOLUTE, self._PRISMATIC)
            for i in range(self.n_frames)
        ]

        # --- Pre-compute static joint subspace S (batch-independent) ---
        self.S_static = torch.zeros(self.n_frames, 6, 1, dtype=self.dtype, device=self.device)
        for node_idx in range(self.n_frames):
            j_idx = chain.joint_indices_list[node_idx]
            j_type = chain.joint_type_indices_list[node_idx]
            if j_type == self._REVOLUTE or j_type == self._PRISMATIC:
                twist = torch.zeros(6, 1, dtype=self.dtype, device=self.device)
                axis = self.axes_norm[j_idx]
                if j_type == self._REVOLUTE:
                    twist[3:, 0] = axis
                else:
                    twist[:3, 0] = axis
                Ad_link = spatial_adjoint_fast(
                    self.link_offset_stack[node_idx].unsqueeze(0)
                ).squeeze(0)
                self.S_static[node_idx] = Ad_link @ twist

        # --- Pre-compute raw twist axes for Jacobian (batch-independent) ---
        self._twist_axes = torch.zeros(self.n_frames, 6, dtype=self.dtype, device=self.device)
        for node_idx in range(self.n_frames):
            j_idx = chain.joint_indices_list[node_idx]
            j_type = chain.joint_type_indices_list[node_idx]
            if j_type == self._REVOLUTE:
                self._twist_axes[node_idx, 3:] = self.axes_norm[j_idx]
            elif j_type == self._PRISMATIC:
                self._twist_axes[node_idx, :3] = self.axes_norm[j_idx]

        # --- Zero constants (expand is free — returns a view, no allocation) ---
        self._zero_scalar = torch.zeros(1, 1, 1, dtype=self.dtype, device=self.device)
        self._zero_6x1 = torch.zeros(1, 6, 1, dtype=self.dtype, device=self.device)
        self._I44 = torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0)

        # --- Compilation ---
        self._compile_enabled = compile_enabled
        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}
        self._setup_callables()

    # ========================================================================
    # Properties (proxy from Chain)
    # ========================================================================

    @property
    def nq(self) -> int:
        """Configuration space dimension."""
        return self._chain.nq

    @property
    def nv(self) -> int:
        """Velocity space dimension."""
        return self._chain.nv

    @property
    def joint_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lower and upper position limits for all actuated joints."""
        return self._chain.low, self._chain.high

    # ========================================================================
    # Query methods (proxy from Chain)
    # ========================================================================

    def get_frame_id(self, frame_name: str) -> int:
        """Gets the integer index for a frame name."""
        return self._chain.get_frame_id(frame_name)

    def get_frame_names(self, exclude_fixed: bool = True) -> List[str]:
        """Returns all frame names in traversal order."""
        return self._chain.get_frame_names(exclude_fixed=exclude_fixed)

    def get_joint_names(self) -> List[str]:
        """Returns the ordered list of actuated joint names."""
        return self._chain.get_joint_parameter_names()

    def get_joint_parameter_names(self, exclude_fixed: bool = True) -> List[str]:
        """Returns the ordered list of actuated joint names."""
        return self._chain.get_joint_parameter_names(exclude_fixed=exclude_fixed)

    def ensure_tensor(self, value) -> torch.Tensor:
        """Converts various input types to a tensor on the model's device/dtype."""
        return self._chain.ensure_tensor(value)

    def unpack_q(self, q: torch.Tensor):
        """Splits q into (q_base, q_joints)."""
        return self._chain.unpack_q(q)

    def unpack_v(self, v: torch.Tensor):
        """Splits v into (v_base, v_joints)."""
        return self._chain.unpack_v(v)

    def pack_q(self, q_base, q_joints: torch.Tensor) -> torch.Tensor:
        """Combines base and joint positions into q."""
        return self._chain.pack_q(q_base, q_joints)

    def pack_v(self, v_base, v_joints: torch.Tensor) -> torch.Tensor:
        """Combines base and joint velocities into v."""
        return self._chain.pack_v(v_base, v_joints)

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def create_data(self, max_batch_size: int = 1024) -> Data:
        """Creates a new Data workspace for this model.

        Args:
            max_batch_size: Maximum supported batch size for pre-allocated buffers.

        Returns:
            A new :class:`Data` instance.
        """
        return Data(
            n_nodes=self.n_frames,
            nv=self.nv,
            max_batch_size=max_batch_size,
            dtype=self.dtype,
            device=self.device,
        )

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Model":
        """Move all internal tensors to a specified dtype and/or device.

        Returns:
            Self for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        self._chain.to(dtype=self.dtype, device=self.device)

        self.axes_norm = self.axes_norm.to(dtype=self.dtype, device=self.device)
        self.I_spatial = self.I_spatial.to(dtype=self.dtype, device=self.device)
        self.gravity = self.gravity.to(dtype=self.dtype, device=self.device)
        self.joint_offset_stack = self.joint_offset_stack.to(dtype=self.dtype, device=self.device)
        self.link_offset_stack = self.link_offset_stack.to(dtype=self.dtype, device=self.device)
        self.S_static = self.S_static.to(dtype=self.dtype, device=self.device)
        self._twist_axes = self._twist_axes.to(dtype=self.dtype, device=self.device)
        self._zero_scalar = self._zero_scalar.to(dtype=self.dtype, device=self.device)
        self._zero_6x1 = self._zero_6x1.to(dtype=self.dtype, device=self.device)
        self._I44 = self._I44.to(dtype=self.dtype, device=self.device)

        self._setup_callables()
        return self

    def enable_compilation(self, enabled: bool = True, **compile_kwargs):
        """Enable or disable ``torch.compile`` for internal methods."""
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
        self._aba_fn = self._aba_impl
        if self._compile_enabled:
            kwargs = self._compile_kwargs.copy()
            self._update_kinematics_fn = torch.compile(self._update_kinematics_impl, **kwargs)
            self._fk_fn = torch.compile(self._fk_impl, **kwargs)
            self._rnea_fn = torch.compile(self._rnea_impl, **kwargs)
            self._crba_fn = torch.compile(self._crba_impl, **kwargs)
            self._jacobian_fn = torch.compile(self._jacobian_impl, **kwargs)
            self._spatial_acceleration_fn = torch.compile(self._spatial_acceleration_impl, **kwargs)
            self._aba_fn = torch.compile(self._aba_impl, **kwargs)

    # ========================================================================
    # Core: update_kinematics
    # ========================================================================

    def _update_kinematics_impl(
        self,
        data: Data,
        q: torch.Tensor,
        qd: Optional[torch.Tensor],
    ) -> None:
        batch_size = q.shape[0]
        compute_velocity = qd is not None

        # Slice pre-allocated buffers
        T_pc = data.T_pc[:batch_size]
        Xup = data.Xup[:batch_size]
        S = data.S[:batch_size]
        T_world = data.T_world[:batch_size]
        v = data.v[:batch_size]

        vJ = data.vJ[:batch_size]

        S.zero_()
        v.zero_()
        vJ.zero_()

        # Split configuration
        if self.has_floating_base:
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

        I44 = self._I44.expand(batch_size, -1, -1)

        # Base transform
        if self.has_floating_base and q_base is not None:
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
        for node_idx in self._chain.topo_order:
            j_idx = self._chain.joint_indices_list[node_idx]
            j_type = self._chain.joint_type_indices_list[node_idx]
            p_idx = self._chain.parent_list[node_idx]

            T_joint_offset = (
                self.joint_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)
            )
            T_link_offset = self.link_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)

            is_revolute = j_type == self._REVOLUTE
            is_prismatic = j_type == self._PRISMATIC

            if is_revolute:
                T_motion = T_revolute[:, j_idx]
            elif is_prismatic:
                T_motion = T_prismatic[:, j_idx]
            else:
                T_motion = I44

            # 1. T_parent_to_child
            T_pc_i = T_joint_offset @ T_motion @ T_link_offset
            T_pc[:, node_idx] = T_pc_i

            # 2. Xup = Ad(inv(T_pc)) — inlined to avoid per-node allocations
            R_pc = T_pc_i[:, :3, :3]
            Rt = R_pc.transpose(1, 2)
            p_inv = -(Rt @ T_pc_i[:, :3, 3].unsqueeze(-1)).squeeze(-1)
            px, py, pz = p_inv.unbind(-1)
            xup_i = Xup[:, node_idx]
            xup_i[:, :3, :3] = Rt
            xup_i[:, 3:, 3:] = Rt
            xup_i[:, 0, 3:] = -pz.unsqueeze(-1) * Rt[:, 1] + py.unsqueeze(-1) * Rt[:, 2]
            xup_i[:, 1, 3:] = pz.unsqueeze(-1) * Rt[:, 0] - px.unsqueeze(-1) * Rt[:, 2]
            xup_i[:, 2, 3:] = -py.unsqueeze(-1) * Rt[:, 0] + px.unsqueeze(-1) * Rt[:, 1]

            # 3. Joint subspace S (pre-computed, batch-independent)
            is_actuated = is_revolute or is_prismatic
            if is_actuated:
                S[:, node_idx] = self.S_static[node_idx]

            # 4. T_world
            if p_idx == -1:
                T_world[:, node_idx] = T_world_base @ T_pc_i
            else:
                T_world[:, node_idx] = T_world[:, p_idx] @ T_pc_i

            # 5. Velocity (if requested) — also cache vJ for RNEA/ABA
            if compute_velocity:
                if is_actuated and v_joints is not None:
                    v_joint = v_joints[:, j_idx].view(batch_size, 1, 1)
                else:
                    v_joint = self._zero_scalar.expand(batch_size, -1, -1)

                if p_idx == -1:
                    if self.has_floating_base and v_base is not None:
                        v_parent = v_base.unsqueeze(-1)
                    else:
                        v_parent = self._zero_6x1.expand(batch_size, -1, -1)
                else:
                    v_parent = v[:, p_idx]

                vJ_i = S[:, node_idx] * v_joint
                vJ[:, node_idx] = vJ_i
                v[:, node_idx] = Xup[:, node_idx] @ v_parent + vJ_i

        data.batch_size = batch_size
        data.has_velocity = compute_velocity

    # ========================================================================
    # Standalone FK (path-only, no cache)
    # ========================================================================

    def _fk_impl(self, data: Data, q: torch.Tensor, frame_id: int) -> torch.Tensor:
        batch_size = q.shape[0]
        path_nodes = self._chain.parents_indices_list[frame_id]

        if self.has_floating_base:
            q_base = q[:, :7]
            q_joints = q[:, 7:]
        else:
            q_base = None
            q_joints = q

        I44 = self._I44.expand(batch_size, -1, -1)

        if self.has_floating_base and q_base is not None:
            t = q_base[:, :3]
            quat = q_base[:, 3:]
            R = quat_to_rotmat_fast(quat)
            T_world_to_current = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=self.device)
            T_world_to_current[:, :3, :3] = R
            T_world_to_current[:, :3, 3] = t
            T_world_to_current[:, 3, 3] = 1.0
        else:
            T_world_to_current = I44.clone()

        # Only compute motion transforms for joints on the path (not all joints)
        axes_expanded = self.axes_norm.unsqueeze(0).expand(batch_size, -1, -1)

        for node_idx in path_nodes:
            joint_idx = self._chain.joint_indices_list[node_idx]
            joint_type_idx = self._chain.joint_type_indices_list[node_idx]

            T_joint_offset = (
                self.joint_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)
            )

            is_revolute = joint_type_idx == self._REVOLUTE
            is_prismatic = joint_type_idx == self._PRISMATIC

            if is_revolute:
                axis = axes_expanded[:, joint_idx : joint_idx + 1]
                angle = q_joints[:, joint_idx : joint_idx + 1]
                T_motion = axis_and_angle_to_matrix_44(axis, angle)[:, 0]
            elif is_prismatic:
                axis = axes_expanded[:, joint_idx : joint_idx + 1]
                d_val = q_joints[:, joint_idx : joint_idx + 1]
                T_motion = axis_and_d_to_pris_matrix(axis, d_val)[:, 0]
            else:
                T_motion = I44

            T_link_offset = self.link_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)
            T_world_to_current = T_world_to_current @ T_joint_offset @ T_motion @ T_link_offset

        return T_world_to_current

    # ========================================================================
    # Jacobian
    # ========================================================================

    def _jacobian_impl(
        self,
        data: Data,
        frame_id: int,
        reference_frame: str,
        return_eef_pose: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = data.batch_size
        path_nodes = self._chain.parents_indices_list[frame_id]

        J_local = data.J_local[:batch_size, :, :]
        J_local.zero_()

        nv_base = 6 if self.has_floating_base else 0

        T_world_to_frame = data.T_world[:batch_size, frame_id]
        T_frame_to_world = inv_homogeneous_fast(T_world_to_frame)

        # Floating base Jacobian columns
        if self.has_floating_base:
            T_world_to_base = data.T_world[:batch_size, self._chain.topo_order[0]]
            T_frame_to_base = T_frame_to_world @ T_world_to_base
            Ad_frame_wrt_base = spatial_adjoint_fast(T_frame_to_base)
            J_local[:, :, :nv_base] = Ad_frame_wrt_base

        # Joint Jacobian columns
        for node_idx in path_nodes:
            joint_idx = self._chain.joint_indices_list[node_idx]
            joint_type_idx = self._chain.joint_type_indices_list[node_idx]

            is_revolute = joint_type_idx == self._REVOLUTE
            is_prismatic = joint_type_idx == self._PRISMATIC

            if not (is_revolute or is_prismatic):
                continue

            p_idx = self._chain.parent_list[node_idx]
            if p_idx == -1:
                if self.has_floating_base:
                    T_world_parent = data.T_world[:batch_size, self._chain.topo_order[0]]
                else:
                    T_world_parent = self._I44.expand(batch_size, -1, -1)
            else:
                T_world_parent = data.T_world[:batch_size, p_idx]

            T_joint_offset = (
                self.joint_offset_stack[node_idx].unsqueeze(0).expand(batch_size, -1, -1)
            )
            T_world_to_joint_origin = T_world_parent @ T_joint_offset

            T_frame_to_joint_origin = T_frame_to_world @ T_world_to_joint_origin
            Ad_frame_wrt_joint = spatial_adjoint_fast(T_frame_to_joint_origin)

            twist_joint = self._twist_axes[node_idx].unsqueeze(0).expand(batch_size, -1)

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

    # ========================================================================
    # RNEA (Inverse Dynamics)
    # ========================================================================

    def _rnea_impl(
        self,
        data: Data,
        qdd: torch.Tensor,
        gravity: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = data.batch_size
        Xup = data.Xup
        S = data.S
        v = data.v
        vJ = data.vJ

        a = data.a[:batch_size]
        f = data.f[:batch_size]
        a.zero_()
        f.zero_()

        if self.has_floating_base:
            a_base = qdd[:, :6]
            a_joints = qdd[:, 6:]
        else:
            a_base = None
            a_joints = qdd

        g = self.gravity if gravity is None else gravity
        a_gravity_world = data.a_gravity_scratch[:batch_size]
        a_gravity_world.zero_()
        a_gravity_world[:, :3, 0] = -g.expand(batch_size, -1)

        if self.has_floating_base:
            T_world_base = data.T_world[:batch_size, self._chain.topo_order[0]]
            Ad_base_world = spatial_adjoint_fast(inv_homogeneous_fast(T_world_base))
            a_gravity_base = Ad_base_world @ a_gravity_world
        else:
            a_gravity_base = a_gravity_world

        # Forward pass: compute accelerations (using cached vJ from update_kinematics)
        for node_idx in self._chain.topo_order:
            j_idx = self._chain.joint_indices_list[node_idx]
            p_idx = self._chain.parent_list[node_idx]
            is_actuated = self._is_actuated[node_idx]

            if is_actuated:
                a_joint = a_joints[:, j_idx].view(batch_size, 1, 1)
            else:
                a_joint = self._zero_scalar.expand(batch_size, -1, -1)

            if p_idx == -1:
                if self.has_floating_base:
                    a_parent = a_base.unsqueeze(-1) + a_gravity_base
                else:
                    a_parent = a_gravity_base
            else:
                a_parent = a[:, p_idx]

            # Inline motion cross product: ad_v @ vJ (avoids (B,6,6) allocation)
            v_i = v[:batch_size, node_idx]
            vJ_i = vJ[:batch_size, node_idx]
            crx = data._cross_scratch[:batch_size]
            crx[:, :3] = torch.linalg.cross(v_i[:, 3:], vJ_i[:, :3], dim=1) + torch.linalg.cross(
                v_i[:, :3], vJ_i[:, 3:], dim=1
            )
            crx[:, 3:] = torch.linalg.cross(v_i[:, 3:], vJ_i[:, 3:], dim=1)
            a[:, node_idx] = (
                Xup[:batch_size, node_idx] @ a_parent + S[:batch_size, node_idx] * a_joint + crx
            )

        # Backward pass: force propagation
        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)

        for node_idx in reversed(self._chain.topo_order):
            v_i = v[:batch_size, node_idx]
            Iv = I_spatial_batched[:, node_idx] @ v_i
            f_node = I_spatial_batched[:, node_idx] @ a[:, node_idx]
            # Inline force cross product: ad*_v @ Iv (avoids (B,6,6) allocation)
            # ad*_v = [w×, 0; v×, w×] where w=ang, v=lin
            fcx = data._cross_scratch[:batch_size]
            fcx[:, :3] = torch.linalg.cross(v_i[:, 3:], Iv[:, :3], dim=1)
            fcx[:, 3:] = torch.linalg.cross(v_i[:, :3], Iv[:, :3], dim=1) + torch.linalg.cross(
                v_i[:, 3:], Iv[:, 3:], dim=1
            )
            f_node = f_node + fcx

            children = self._chain.children_list[node_idx]
            if len(children) == 1:
                child_idx = children[0]
                f_node = f_node + Xup[:batch_size, child_idx].transpose(1, 2) @ f[:, child_idx]
            elif len(children) > 1:
                for child_idx in children:
                    f_node = f_node + Xup[:batch_size, child_idx].transpose(1, 2) @ f[:, child_idx]

            f[:, node_idx] = f_node

        # Extract generalized forces (pre-allocated output)
        tau = data.tau_out[:batch_size]
        tau.zero_()
        tau_all_nodes = (S[:batch_size] * f).sum(dim=2).squeeze(-1)

        if self.has_floating_base:
            tau[:, :6] = f[:, self.urdf_root_idx, :, 0]
            for node_idx in range(self.n_frames):
                if self._is_actuated[node_idx]:
                    j_col = self._chain.joint_indices_list[node_idx]
                    tau[:, 6 + j_col] = tau_all_nodes[:, node_idx]
        else:
            for node_idx in range(self.n_frames):
                if self._is_actuated[node_idx]:
                    j_col = self._chain.joint_indices_list[node_idx]
                    tau[:, j_col] = tau_all_nodes[:, node_idx]

        return tau

    # ========================================================================
    # CRBA (Mass Matrix)
    # ========================================================================

    def _crba_impl(self, data: Data) -> torch.Tensor:
        batch_size = data.batch_size
        Xup = data.Xup
        S = data.S

        I_composite = data.I_composite[:batch_size]
        M = data.M[:batch_size, : self.nv, : self.nv]

        M.zero_()
        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)
        I_composite.copy_(I_spatial_batched)

        # Backward pass: composite inertia accumulation
        for node_idx in reversed(self._chain.topo_order):
            p = self._chain.parent_list[node_idx]
            if p != -1:
                I_composite[:, p] += (
                    Xup[:batch_size, node_idx].transpose(1, 2)
                    @ I_composite[:, node_idx]
                    @ Xup[:batch_size, node_idx]
                )

        # Assemble mass matrix
        if self.has_floating_base:
            M[:, :6, :6] = I_composite[:, self.urdf_root_idx]

        for node_idx in self._chain.topo_order:
            col_idx = self.vel_indices_list[node_idx]
            if col_idx == -1:
                continue

            S_i = S[:batch_size, node_idx]
            F_i = I_composite[:, node_idx] @ S_i

            # Diagonal element
            M[:, col_idx, col_idx] = (S_i.transpose(1, 2) @ F_i).squeeze(-1).squeeze(-1)

            # Base-joint coupling
            if self.has_floating_base:
                f_at_base = F_i
                current_node = node_idx
                while current_node != self.urdf_root_idx:
                    parent = self._chain.parent_list[current_node]
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
                parent = self._chain.parent_list[current_node]
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

    # ========================================================================
    # Spatial Acceleration
    # ========================================================================

    def _spatial_acceleration_impl(
        self,
        data: Data,
        qdd: torch.Tensor,
        frame_id: int,
        reference_frame: str,
    ) -> torch.Tensor:
        batch_size = data.batch_size
        Xup = data.Xup
        S = data.S
        v = data.v

        a = data.a[:batch_size]
        a.zero_()

        if self.has_floating_base:
            a_base = qdd[:, :6]
            a_joints = qdd[:, 6:]
        else:
            a_base = None
            a_joints = qdd

        a_world0 = self._zero_6x1.expand(batch_size, -1, -1)

        for node_idx in self._chain.topo_order:
            j_idx = self._chain.joint_indices_list[node_idx]
            j_type = self._chain.joint_type_indices_list[node_idx]
            p_idx = self._chain.parent_list[node_idx]

            is_revolute = j_type == self._REVOLUTE
            is_prismatic = j_type == self._PRISMATIC
            is_actuated = is_revolute or is_prismatic

            if is_actuated:
                a_joint = a_joints[:, j_idx].view(batch_size, 1, 1)
            else:
                a_joint = self._zero_scalar.expand(batch_size, -1, -1)

            if p_idx == -1:
                if self.has_floating_base:
                    a_parent = a_base.unsqueeze(-1) + a_world0
                else:
                    a_parent = a_world0
            else:
                a_parent = a[:, p_idx]

            if is_actuated:
                if p_idx == -1:
                    v_parent_node = self._zero_6x1.expand(batch_size, -1, -1)
                else:
                    v_parent_node = v[:batch_size, p_idx]
                vJ = v[:batch_size, node_idx] - Xup[:batch_size, node_idx] @ v_parent_node
            else:
                vJ = self._zero_6x1.expand(batch_size, -1, -1)

            # Inline motion cross product: ad_v @ vJ
            v_i = v[:batch_size, node_idx]
            crx = data._cross_scratch[:batch_size]
            crx[:, :3] = torch.linalg.cross(v_i[:, 3:], vJ[:, :3], dim=1) + torch.linalg.cross(
                v_i[:, :3], vJ[:, 3:], dim=1
            )
            crx[:, 3:] = torch.linalg.cross(v_i[:, 3:], vJ[:, 3:], dim=1)
            a[:, node_idx] = (
                Xup[:batch_size, node_idx] @ a_parent + S[:batch_size, node_idx] * a_joint + crx
            )

        a_local = a[:, frame_id]

        if reference_frame == "world":
            Ad_world_wrt_body = spatial_adjoint_fast(data.T_world[:batch_size, frame_id])
            a_world = Ad_world_wrt_body @ a_local
            return a_world.squeeze(-1)
        else:
            return a_local.squeeze(-1)

    # ========================================================================
    # ABA (Forward Dynamics)
    # ========================================================================

    def _aba_impl(
        self,
        data: Data,
        tau: torch.Tensor,
        gravity: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Articulated Body Algorithm (Featherstone RBDA Table 9.5)."""
        batch_size = data.batch_size
        Xup = data.Xup
        S = data.S
        v = data.v
        cached_vJ = data.vJ

        IA = data.IA[:batch_size]
        pA = data.pA[:batch_size]
        U = data.U[:batch_size]
        d = data.d[:batch_size]
        u = data.u[:batch_size]
        a = data.a[:batch_size]
        c = data.f[:batch_size]  # Reuse RNEA force buffer for coriolis term

        a.zero_()
        c.zero_()

        if self.has_floating_base:
            tau_base = tau[:, :6]
            tau_joints = tau[:, 6:]
        else:
            tau_base = None
            tau_joints = tau

        g = self.gravity if gravity is None else gravity

        # Gravity pseudo-acceleration: a_0 = -g (Featherstone convention)
        a_gravity_world = data.a_gravity_scratch[:batch_size]
        a_gravity_world.zero_()
        a_gravity_world[:, :3, 0] = -g.expand(batch_size, -1)

        if self.has_floating_base:
            T_world_base = data.T_world[:batch_size, self._chain.topo_order[0]]
            Ad_base_world = spatial_adjoint_fast(inv_homogeneous_fast(T_world_base))
            a_gravity_base = Ad_base_world @ a_gravity_world
        else:
            a_gravity_base = a_gravity_world

        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # ---- Pass 1: Initialize IA=I, compute pA (bias force), c (coriolis) ----
        IA.copy_(I_spatial_batched)

        for node_idx in self._chain.topo_order:
            is_actuated = self._is_actuated[node_idx]

            v_i = v[:batch_size, node_idx]
            Iv = I_spatial_batched[:, node_idx] @ v_i
            # Inline force cross product: ad*_v @ Iv
            fcx = data._cross_scratch[:batch_size]
            fcx[:, :3] = torch.linalg.cross(v_i[:, 3:], Iv[:, :3], dim=1)
            fcx[:, 3:] = torch.linalg.cross(v_i[:, :3], Iv[:, :3], dim=1) + torch.linalg.cross(
                v_i[:, 3:], Iv[:, 3:], dim=1
            )
            pA[:, node_idx] = fcx

            # c_i = v_i x vJ — inline motion cross product
            if is_actuated:
                vJ_i = cached_vJ[:batch_size, node_idx]
                crx = data._cross_scratch[:batch_size]
                crx[:, :3] = torch.linalg.cross(
                    v_i[:, 3:], vJ_i[:, :3], dim=1
                ) + torch.linalg.cross(v_i[:, :3], vJ_i[:, 3:], dim=1)
                crx[:, 3:] = torch.linalg.cross(v_i[:, 3:], vJ_i[:, 3:], dim=1)
                c[:, node_idx] = crx

        # ---- Pass 2: Backward — accumulate articulated body inertia ----
        for node_idx in reversed(self._chain.topo_order):
            j_idx = self._chain.joint_indices_list[node_idx]
            p_idx = self._chain.parent_list[node_idx]
            is_actuated = self._is_actuated[node_idx]

            if is_actuated:
                S_i = S[:batch_size, node_idx]
                IA_i = IA[:, node_idx]

                U_i = IA_i @ S_i
                U[:, node_idx] = U_i
                d_i = (S_i.transpose(1, 2) @ U_i).squeeze(-1).squeeze(-1)
                d[:, node_idx] = d_i

                u_i = tau_joints[:, j_idx] - (S_i.transpose(1, 2) @ pA[:, node_idx]).squeeze(
                    -1
                ).squeeze(-1)
                u[:, node_idx] = u_i

                d_safe = d_i.clamp(min=1e-12)
                d_inv = 1.0 / d_safe.unsqueeze(-1).unsqueeze(-1)
                Ia = IA_i - U_i @ U_i.transpose(1, 2) * d_inv
                pa = (
                    pA[:, node_idx]
                    + Ia @ c[:, node_idx]
                    + U_i * (u_i / d_safe).unsqueeze(-1).unsqueeze(-1)
                )

                if p_idx != -1:
                    Xup_i_T = Xup[:batch_size, node_idx].transpose(1, 2)
                    IA[:, p_idx] = IA[:, p_idx] + Xup_i_T @ Ia @ Xup[:batch_size, node_idx]
                    pA[:, p_idx] = pA[:, p_idx] + Xup_i_T @ pa
            else:
                # Fixed joint: propagate IA and pA upward
                if p_idx != -1:
                    Xup_i_T = Xup[:batch_size, node_idx].transpose(1, 2)
                    IA[:, p_idx] = (
                        IA[:, p_idx] + Xup_i_T @ IA[:, node_idx] @ Xup[:batch_size, node_idx]
                    )
                    pA[:, p_idx] = pA[:, p_idx] + Xup_i_T @ pA[:, node_idx]

        # ---- Pass 3: Forward — compute accelerations (pre-allocated output) ----
        root_idx = self.urdf_root_idx
        qdd_out = data.qdd_out[:batch_size]
        qdd_out.zero_()

        if self.has_floating_base:
            # Solve for base acceleration: IA * a = tau_base - pA
            assert tau_base is not None
            a[:, root_idx] = torch.linalg.solve(
                IA[:, root_idx],
                tau_base.unsqueeze(-1) - pA[:, root_idx],
            )

            # Extract qdd_base: transform a back to node 0 frame, subtract gravity
            inv_Xup_root = spatial_adjoint_fast(data.T_pc[:batch_size, root_idx])
            a_in_node0 = inv_Xup_root @ a[:, root_idx]
            qdd_out[:, :6] = (a_in_node0 - a_gravity_base).squeeze(-1)

        for node_idx in self._chain.topo_order:
            p_idx = self._chain.parent_list[node_idx]
            is_actuated = self._is_actuated[node_idx]

            # Skip root nodes for floating base (already computed)
            if self.has_floating_base and (node_idx == 0 or node_idx == root_idx):
                continue

            if p_idx == -1:
                a_parent = a_gravity_base
            else:
                a_parent = a[:, p_idx]

            # a'_i = Xup_i * a_parent + c_i
            a_prime = Xup[:batch_size, node_idx] @ a_parent + c[:, node_idx]

            if is_actuated:
                qdd_i = (
                    u[:, node_idx]
                    - (U[:, node_idx].transpose(1, 2) @ a_prime).squeeze(-1).squeeze(-1)
                ) / d[:, node_idx].clamp(min=1e-12)
                a[:, node_idx] = a_prime + S[:batch_size, node_idx] * qdd_i.unsqueeze(-1).unsqueeze(
                    -1
                )

                vel_idx = self.vel_indices_list[node_idx]
                if vel_idx != -1:
                    qdd_out[:, vel_idx] = qdd_i
            else:
                a[:, node_idx] = a_prime

        return qdd_out

    def __str__(self) -> str:
        return str(self._chain)
