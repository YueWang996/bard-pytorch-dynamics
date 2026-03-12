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
from bard.transforms import (  # noqa: F401 — kept for potential external use
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

try:
    from .triton_kernels import fused_xtmx_add, HAS_TRITON
except ImportError:
    HAS_TRITON = False
    fused_xtmx_add = None


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

        # --- Pre-compute per-node decomposed transforms for fast runtime ---
        # For fixed joints: T_static_pc = T_joint_offset @ T_link_offset (no runtime matmul)
        # For revolute joints: pre-compute Rodrigues constants so runtime has NO matmuls:
        #   R_combined = R_static_rot + sin(q) * A_rot + (1-cos(q)) * B_rot
        #   t_combined = t_static_rot + sin(q) * a_trans + (1-cos(q)) * b_trans
        self.T_static_pc = torch.zeros(self.n_frames, 4, 4, dtype=self.dtype, device=self.device)
        self.R_pre = torch.zeros(self.n_frames, 3, 3, dtype=self.dtype, device=self.device)
        self.t_pre = torch.zeros(self.n_frames, 3, dtype=self.dtype, device=self.device)
        self.R_post = torch.zeros(self.n_frames, 3, 3, dtype=self.dtype, device=self.device)
        self.t_post = torch.zeros(self.n_frames, 3, dtype=self.dtype, device=self.device)

        # Rodrigues decomposition constants for revolute joints (matmul-free runtime)
        self.R_static_rot = torch.zeros(self.n_frames, 3, 3, dtype=self.dtype, device=self.device)
        self.A_rot = torch.zeros(self.n_frames, 3, 3, dtype=self.dtype, device=self.device)
        self.B_rot = torch.zeros(self.n_frames, 3, 3, dtype=self.dtype, device=self.device)
        self.t_static_rot = torch.zeros(self.n_frames, 3, dtype=self.dtype, device=self.device)
        self.a_trans = torch.zeros(self.n_frames, 3, dtype=self.dtype, device=self.device)
        self.b_trans = torch.zeros(self.n_frames, 3, dtype=self.dtype, device=self.device)

        for i in range(self.n_frames):
            j_off = self.joint_offset_stack[i]
            l_off = self.link_offset_stack[i]
            self.T_static_pc[i] = j_off @ l_off
            R_pre = j_off[:3, :3]
            t_pre = j_off[:3, 3]
            R_post = l_off[:3, :3]
            t_post = l_off[:3, 3]
            self.R_pre[i] = R_pre
            self.t_pre[i] = t_pre
            self.R_post[i] = R_post
            self.t_post[i] = t_post

            # Pre-compute Rodrigues constants for revolute joints
            j_idx = chain.joint_indices_list[i]
            j_type = chain.joint_type_indices_list[i]
            if j_type == Joint.TYPES.index("revolute"):
                axis = self.axes_norm[j_idx]
                # K = skew-symmetric matrix of axis
                K = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
                K[0, 1] = -axis[2]
                K[0, 2] = axis[1]
                K[1, 0] = axis[2]
                K[1, 2] = -axis[0]
                K[2, 0] = -axis[1]
                K[2, 1] = axis[0]
                K2 = K @ K
                # R_combined = R_pre @ (I + sin*K + (1-c)*K²) @ R_post
                #            = R_pre@R_post + sin*(R_pre@K@R_post) + (1-c)*(R_pre@K²@R_post)
                self.R_static_rot[i] = R_pre @ R_post
                self.A_rot[i] = R_pre @ K @ R_post
                self.B_rot[i] = R_pre @ K2 @ R_post
                # t_combined = R_pre @ R_mot @ t_post + t_pre
                #            = (R_pre@t_post + t_pre) + sin*(R_pre@K@t_post) + (1-c)*(R_pre@K²@t_post)
                self.t_static_rot[i] = R_pre @ t_post + t_pre
                self.a_trans[i] = R_pre @ K @ t_post
                self.b_trans[i] = R_pre @ K2 @ t_post

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

        # --- Pre-compute Jacobian column constants (avoids per-joint 6x6 adjoint) ---
        # For each joint: rotated_axis = R_joint_offset @ axis (in parent frame)
        #                  p_offset = joint origin position in parent frame
        self._jac_rotated_axis = torch.zeros(self.n_frames, 3, dtype=self.dtype, device=self.device)
        self._jac_p_offset = torch.zeros(self.n_frames, 3, dtype=self.dtype, device=self.device)
        for node_idx in range(self.n_frames):
            j_idx = chain.joint_indices_list[node_idx]
            j_type = chain.joint_type_indices_list[node_idx]
            R_offset = self.joint_offset_stack[node_idx, :3, :3]
            self._jac_p_offset[node_idx] = self.joint_offset_stack[node_idx, :3, 3]
            if j_type == self._REVOLUTE or j_type == self._PRISMATIC:
                self._jac_rotated_axis[node_idx] = R_offset @ self.axes_norm[j_idx]

        # --- Zero constants (expand is free — returns a view, no allocation) ---
        self._zero_scalar = torch.zeros(1, 1, 1, dtype=self.dtype, device=self.device)
        self._zero_6x1 = torch.zeros(1, 6, 1, dtype=self.dtype, device=self.device)
        self._I44 = torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0)

        # --- Pre-compute vectorized update_kinematics data ---
        self._rev_nodes_list = []
        self._rev_joint_indices_list = []
        self._pris_nodes_list = []
        self._pris_joint_indices_list = []
        for node_idx in range(self.n_frames):
            j_type = chain.joint_type_indices_list[node_idx]
            j_idx = chain.joint_indices_list[node_idx]
            if j_type == self._REVOLUTE:
                self._rev_nodes_list.append(node_idx)
                self._rev_joint_indices_list.append(j_idx)
            elif j_type == self._PRISMATIC:
                self._pris_nodes_list.append(node_idx)
                self._pris_joint_indices_list.append(j_idx)
        self._n_rev = len(self._rev_nodes_list)
        self._n_pris = len(self._pris_nodes_list)
        self._init_vectorized_constants()
        self._tree_levels = self._compute_tree_levels()

        # --- Pre-compute RNEA tau extraction indices (vectorized gather) ---
        self._actuated_nodes_list = []
        self._actuated_vel_indices_list = []
        for node_idx in range(self.n_frames):
            if self._is_actuated[node_idx]:
                self._actuated_nodes_list.append(node_idx)
                j_col = chain.joint_indices_list[node_idx]
                vel_idx = (6 + j_col) if self.has_floating_base else j_col
                self._actuated_vel_indices_list.append(vel_idx)
        self._actuated_nodes_t = torch.tensor(
            self._actuated_nodes_list, dtype=torch.long, device=self.device
        )
        self._actuated_vel_indices_t = torch.tensor(
            self._actuated_vel_indices_list, dtype=torch.long, device=self.device
        )

        # --- Pre-compute collapsed FK paths (merge consecutive fixed joints) ---
        self._fk_collapsed_paths = self._build_fk_collapsed_paths()

        # --- Compilation ---
        self._compile_enabled = compile_enabled
        self._use_triton_kernels = HAS_TRITON  # Use fused Triton kernels when available
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
        self.T_static_pc = self.T_static_pc.to(dtype=self.dtype, device=self.device)
        self.R_pre = self.R_pre.to(dtype=self.dtype, device=self.device)
        self.t_pre = self.t_pre.to(dtype=self.dtype, device=self.device)
        self.R_post = self.R_post.to(dtype=self.dtype, device=self.device)
        self.t_post = self.t_post.to(dtype=self.dtype, device=self.device)
        self.R_static_rot = self.R_static_rot.to(dtype=self.dtype, device=self.device)
        self.A_rot = self.A_rot.to(dtype=self.dtype, device=self.device)
        self.B_rot = self.B_rot.to(dtype=self.dtype, device=self.device)
        self.t_static_rot = self.t_static_rot.to(dtype=self.dtype, device=self.device)
        self.a_trans = self.a_trans.to(dtype=self.dtype, device=self.device)
        self.b_trans = self.b_trans.to(dtype=self.dtype, device=self.device)
        self.S_static = self.S_static.to(dtype=self.dtype, device=self.device)
        self._twist_axes = self._twist_axes.to(dtype=self.dtype, device=self.device)
        self._jac_rotated_axis = self._jac_rotated_axis.to(dtype=self.dtype, device=self.device)
        self._jac_p_offset = self._jac_p_offset.to(dtype=self.dtype, device=self.device)
        self._zero_scalar = self._zero_scalar.to(dtype=self.dtype, device=self.device)
        self._zero_6x1 = self._zero_6x1.to(dtype=self.dtype, device=self.device)
        self._I44 = self._I44.to(dtype=self.dtype, device=self.device)

        # Rebuild vectorized constants and collapsed FK paths on new device/dtype
        self._init_vectorized_constants()
        self._tree_levels = self._compute_tree_levels()
        self._fk_collapsed_paths = self._build_fk_collapsed_paths()
        self._actuated_nodes_t = self._actuated_nodes_t.to(device=self.device)
        self._actuated_vel_indices_t = self._actuated_vel_indices_t.to(device=self.device)

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
        needs_grad = q.requires_grad or (qd is not None and qd.requires_grad)
        if needs_grad:
            return self._update_kinematics_functional(data, q, qd)

        batch_size = q.shape[0]
        compute_velocity = qd is not None
        N = self.n_frames

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

        # Set batch_size early (needed by _ensure_xup if called during velocity)
        data.batch_size = batch_size

        # Store q for lazy T_pc, T_world, and FK computation
        data._q = q
        data._t_pc_valid = False
        data._t_world_valid = False
        data._xup_valid = False

        # ---- Velocity propagation (needs Xup + S → triggers T_pc) ----
        v = data.v[:batch_size]
        vJ = data.vJ[:batch_size]

        if compute_velocity:
            # Velocity requires Xup and S — compute them now
            self._ensure_xup(data)

            S = data.S[:batch_size]
            for nodes_t, parents_t, act_pos_t, act_jnt_t in self._tree_levels:
                n_level = nodes_t.shape[0]

                # Parent velocity
                if parents_t[0] == -1:
                    if self.has_floating_base and v_base is not None:
                        v_parent = v_base.unsqueeze(-1).unsqueeze(1).expand(-1, n_level, -1, -1)
                    else:
                        v_parent = self._zero_6x1.expand(batch_size, n_level, -1, -1)
                else:
                    v_parent = v[:, parents_t]  # (B, n_level, 6, 1)

                # Joint velocities for this level
                v_joint = q.new_zeros(batch_size, n_level, 1, 1)
                if act_pos_t is not None and v_joints is not None:
                    v_joint[:, act_pos_t, 0, 0] = v_joints[:, act_jnt_t]

                Xup = data.Xup[:batch_size]
                S_level = S[:, nodes_t]
                vJ_level = S_level * v_joint
                v[:, nodes_t] = Xup[:, nodes_t] @ v_parent + vJ_level
                vJ[:, nodes_t] = vJ_level
        else:
            v.zero_()
            vJ.zero_()

        data.has_velocity = compute_velocity

    def _update_kinematics_functional(
        self,
        data: Data,
        q: torch.Tensor,
        qd: Optional[torch.Tensor],
    ) -> None:
        """Autograd-compatible update_kinematics using only functional ops.

        Slower than the in-place version but supports gradient computation
        through q and qd. Builds all tensors without in-place indexed writes.
        """
        batch_size = q.shape[0]
        compute_velocity = qd is not None
        N = self.n_frames

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

        I44 = self._I44.expand(batch_size, -1, -1)
        bottom_row = q.new_zeros(batch_size, 1, 4)
        bottom_row = bottom_row.clone()
        bottom_row[:, 0, 3] = 1.0

        # Base transform (functional)
        if self.has_floating_base and q_base is not None:
            t = q_base[:, :3]
            quat = q_base[:, 3:]
            R = quat_to_rotmat_fast(quat)
            top = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (B, 3, 4)
            T_world_base = torch.cat([top, bottom_row], dim=-2)  # (B, 4, 4)
        else:
            T_world_base = I44

        # Lists to collect per-node results (functional, no in-place writes)
        T_pc_list = [None] * N
        Xup_list = [None] * N
        T_world_list = [None] * N
        v_list = [None] * N
        vJ_list = [None] * N
        S_list = [None] * N

        zeros_33 = q.new_zeros(batch_size, 3, 3)

        for node_idx in self._chain.topo_order:
            j_idx = self._chain.joint_indices_list[node_idx]
            j_type = self._chain.joint_type_indices_list[node_idx]
            p_idx = self._chain.parent_list[node_idx]

            is_revolute = j_type == self._REVOLUTE
            is_prismatic = j_type == self._PRISMATIC

            # 1. Build T_pc functionally
            if is_revolute:
                angle = q_joints[:, j_idx]
                s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
                one_c = (1.0 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1)
                R_combined = (
                    self.R_static_rot[node_idx]
                    + s * self.A_rot[node_idx]
                    + one_c * self.B_rot[node_idx]
                )
                s3 = s.squeeze(-1)
                one_c3 = one_c.squeeze(-1)
                t_combined = (
                    self.t_static_rot[node_idx]
                    + s3 * self.a_trans[node_idx]
                    + one_c3 * self.b_trans[node_idx]
                )
                top = torch.cat([R_combined, t_combined.unsqueeze(-1)], dim=-1)
                T_pc_i = torch.cat([top, bottom_row], dim=-2)
            elif is_prismatic:
                d_val = q_joints[:, j_idx]
                axis = self.axes_norm[j_idx]
                t_motion = axis * d_val.unsqueeze(-1)
                t_combined = (
                    self.R_pre[node_idx] @ (t_motion + self.t_post[node_idx]).unsqueeze(-1)
                ).squeeze(-1) + self.t_pre[node_idx]
                R_static = (
                    self.T_static_pc[node_idx, :3, :3].unsqueeze(0).expand(batch_size, -1, -1)
                )
                top = torch.cat([R_static, t_combined.unsqueeze(-1)], dim=-1)
                T_pc_i = torch.cat([top, bottom_row], dim=-2)
            else:
                T_pc_i = self.T_static_pc[node_idx].unsqueeze(0).expand(batch_size, -1, -1)

            T_pc_list[node_idx] = T_pc_i

            # 2. Xup = Ad(inv(T_pc)) — functional construction
            R_pc = T_pc_i[:, :3, :3]
            Rt = R_pc.transpose(1, 2)
            p_inv = -(Rt @ T_pc_i[:, :3, 3].unsqueeze(-1)).squeeze(-1)
            px, py, pz = p_inv.unbind(-1)

            pxRt_row0 = -pz.unsqueeze(-1) * Rt[:, 1] + py.unsqueeze(-1) * Rt[:, 2]
            pxRt_row1 = pz.unsqueeze(-1) * Rt[:, 0] - px.unsqueeze(-1) * Rt[:, 2]
            pxRt_row2 = -py.unsqueeze(-1) * Rt[:, 0] + px.unsqueeze(-1) * Rt[:, 1]
            pxRt = torch.stack([pxRt_row0, pxRt_row1, pxRt_row2], dim=1)  # (B, 3, 3)

            top_xup = torch.cat([Rt, pxRt], dim=-1)  # (B, 3, 6)
            bot_xup = torch.cat([zeros_33, Rt], dim=-1)  # (B, 3, 6)
            xup_i = torch.cat([top_xup, bot_xup], dim=-2)  # (B, 6, 6)
            Xup_list[node_idx] = xup_i

            # 3. S (batch-independent)
            is_actuated = is_revolute or is_prismatic
            S_list[node_idx] = self.S_static[node_idx] if is_actuated else self.S_static[node_idx]

            # 4. T_world
            if p_idx == -1:
                T_world_list[node_idx] = T_world_base @ T_pc_i
            else:
                T_world_list[node_idx] = T_world_list[p_idx] @ T_pc_i

            # 5. Velocity
            if compute_velocity:
                if is_actuated and v_joints is not None:
                    v_joint = v_joints[:, j_idx].view(batch_size, 1, 1)
                else:
                    v_joint = self._zero_scalar.expand(batch_size, -1, -1)

                vJ_i = S_list[node_idx] * v_joint
                vJ_list[node_idx] = vJ_i

                if p_idx == -1:
                    if self.has_floating_base and v_base is not None:
                        v_parent = v_base.unsqueeze(-1)
                    else:
                        v_parent = self._zero_6x1.expand(batch_size, -1, -1)
                else:
                    v_parent = v_list[p_idx]

                v_list[node_idx] = xup_i @ v_parent + vJ_i
            else:
                vJ_list[node_idx] = self._zero_6x1.expand(batch_size, -1, -1)
                v_list[node_idx] = self._zero_6x1.expand(batch_size, -1, -1)

        # Stack into tensors and assign to data (replacing pre-allocated buffers)
        data.Xup = torch.stack(Xup_list, dim=1)  # (B, N, 6, 6)
        data.Xup_T = data.Xup.transpose(2, 3).contiguous()  # (B, N, 6, 6)
        data.T_world = torch.stack(T_world_list, dim=1)  # (B, N, 4, 4)
        data.T_pc = torch.stack(T_pc_list, dim=1)  # (B, N, 4, 4)
        data.S = self.S_static.unsqueeze(0).expand(batch_size, -1, -1, -1)
        data.v = torch.stack(v_list, dim=1)  # (B, N, 6, 1)
        data.vJ = torch.stack(vJ_list, dim=1)  # (B, N, 6, 1)
        data.batch_size = batch_size
        data.has_velocity = compute_velocity
        data._t_pc_valid = True  # Functional path computes T_pc inline
        data._xup_valid = True  # Functional path computes Xup inline
        data._t_world_valid = True  # Functional path computes T_world inline
        data._q = q

    # ========================================================================
    # Vectorized update_kinematics helpers
    # ========================================================================

    def _init_vectorized_constants(self):
        """Initialize stacked constants for vectorized update_kinematics."""
        if self._n_rev > 0:
            self._rev_nodes_t = torch.tensor(
                self._rev_nodes_list, dtype=torch.long, device=self.device
            )
            self._rev_joint_indices_t = torch.tensor(
                self._rev_joint_indices_list, dtype=torch.long, device=self.device
            )
            self._R_static_rev = self.R_static_rot[self._rev_nodes_t]
            self._A_rev = self.A_rot[self._rev_nodes_t]
            self._B_rev = self.B_rot[self._rev_nodes_t]
            self._t_static_rev = self.t_static_rot[self._rev_nodes_t]
            self._a_rev = self.a_trans[self._rev_nodes_t]
            self._b_rev = self.b_trans[self._rev_nodes_t]

        if self._n_pris > 0:
            self._pris_nodes_t = torch.tensor(
                self._pris_nodes_list, dtype=torch.long, device=self.device
            )
            self._pris_joint_indices_t = torch.tensor(
                self._pris_joint_indices_list, dtype=torch.long, device=self.device
            )

    def _compute_tree_levels(self):
        """Pre-compute level-order grouping for batched tree propagation.

        Returns a list of (nodes_t, parents_t, act_pos_t, act_jnt_t) tuples,
        one per tree level. Each level's nodes can be processed in parallel
        because their parents are all from previous levels.
        """
        parent_list = self._chain.parent_list
        children_list = self._chain.children_list
        levels = []
        current_level_nodes = [n for n in range(self.n_frames) if parent_list[n] == -1]
        while current_level_nodes:
            parent_indices = [parent_list[n] for n in current_level_nodes]
            nodes_t = torch.tensor(current_level_nodes, dtype=torch.long, device=self.device)
            parents_t = torch.tensor(parent_indices, dtype=torch.long, device=self.device)
            # Pre-compute velocity gather data for this level
            act_positions = []
            act_joint_indices = []
            for i, node_idx in enumerate(current_level_nodes):
                if self._is_actuated[node_idx]:
                    act_positions.append(i)
                    act_joint_indices.append(self._chain.joint_indices_list[node_idx])
            act_pos_t = (
                torch.tensor(act_positions, dtype=torch.long, device=self.device)
                if act_positions
                else None
            )
            act_jnt_t = (
                torch.tensor(act_joint_indices, dtype=torch.long, device=self.device)
                if act_joint_indices
                else None
            )
            levels.append((nodes_t, parents_t, act_pos_t, act_jnt_t))
            next_level = []
            for n in current_level_nodes:
                next_level.extend(children_list[n])
            current_level_nodes = next_level
        return levels

    def _ensure_t_pc(self, data: Data) -> None:
        """Compute T_pc from stored q if not already valid.

        Called lazily by _ensure_xup and _ensure_t_world. FK uses path-only
        computation and doesn't need T_pc.
        """
        if data._t_pc_valid:
            return
        batch_size = data.batch_size
        q = data._q

        if self.has_floating_base:
            q_joints = q[:, 7:]
        else:
            q_joints = q

        T_pc = data.T_pc[:batch_size]
        T_pc.copy_(self.T_static_pc.unsqueeze(0).expand(batch_size, -1, -1, -1))

        if self._n_rev > 0:
            angles = q_joints[:, self._rev_joint_indices_t]
            sin_a = torch.sin(angles)
            cos_a = torch.cos(angles)
            s = sin_a.unsqueeze(-1).unsqueeze(-1)
            one_c = (1.0 - cos_a).unsqueeze(-1).unsqueeze(-1)
            R_all = self._R_static_rev + s * self._A_rev + one_c * self._B_rev
            s3 = sin_a.unsqueeze(-1)
            one_c3 = (1.0 - cos_a).unsqueeze(-1)
            t_all = self._t_static_rev + s3 * self._a_rev + one_c3 * self._b_rev
            T_pc[:, self._rev_nodes_t, :3, :3] = R_all
            T_pc[:, self._rev_nodes_t, :3, 3] = t_all

        if self._n_pris > 0:
            d_vals = q_joints[:, self._pris_joint_indices_t]
            axes_pris = self.axes_norm[self._pris_joint_indices_t]
            t_motion = axes_pris * d_vals.unsqueeze(-1)
            R_pre_pris = self.R_pre[self._pris_nodes_t]
            t_post_pris = self.t_post[self._pris_nodes_t]
            t_pre_pris = self.t_pre[self._pris_nodes_t]
            t_combined = (R_pre_pris @ (t_motion + t_post_pris).unsqueeze(-1)).squeeze(
                -1
            ) + t_pre_pris
            T_pc[:, self._pris_nodes_t, :3, 3] = t_combined

        data._t_pc_valid = True

    def _ensure_xup(self, data: Data) -> None:
        """Compute Xup, Xup_T, and S from cached T_pc if not already valid.

        Called lazily by dynamics algorithms (CRBA, RNEA, ABA) that need
        spatial transforms. FK and Jacobian skip this for performance.
        """
        if data._xup_valid:
            return
        self._ensure_t_pc(data)
        if data._xup_valid:
            return
        batch_size = data.batch_size
        T_pc = data.T_pc[:batch_size]

        # Vectorized Xup: Ad(inv(T_pc)) for all N nodes at once
        R_pc = T_pc[:, :, :3, :3]  # (B, N, 3, 3) view
        Rt = R_pc.transpose(-2, -1).contiguous()  # (B, N, 3, 3)
        p_pc = T_pc[:, :, :3, 3]  # (B, N, 3) view
        p_inv = -(Rt @ p_pc.unsqueeze(-1)).squeeze(-1)  # (B, N, 3)
        px = p_inv[..., 0]  # (B, N)
        py = p_inv[..., 1]
        pz = p_inv[..., 2]

        Xup = data.Xup[:batch_size]
        Xup[:, :, :3, :3] = Rt
        Xup[:, :, 3:, 3:] = Rt
        Xup[:, :, 0, 3:] = -pz.unsqueeze(-1) * Rt[:, :, 1] + py.unsqueeze(-1) * Rt[:, :, 2]
        Xup[:, :, 1, 3:] = pz.unsqueeze(-1) * Rt[:, :, 0] - px.unsqueeze(-1) * Rt[:, :, 2]
        Xup[:, :, 2, 3:] = -py.unsqueeze(-1) * Rt[:, :, 0] + px.unsqueeze(-1) * Rt[:, :, 1]

        # Contiguous transpose for efficient downstream matmuls
        data.Xup_T[:batch_size].copy_(Xup.transpose(2, 3))

        # S (static — single broadcast copy)
        data.S[:batch_size].copy_(self.S_static.unsqueeze(0).expand(batch_size, -1, -1, -1))

        data._xup_valid = True

    def _ensure_t_world(self, data: Data) -> None:
        """Compute T_world for all nodes from cached T_pc if not already valid.

        Called lazily by Jacobian and other algorithms that need full T_world.
        FK uses path-only computation instead for better performance.
        """
        if data._t_world_valid:
            return
        self._ensure_t_pc(data)
        batch_size = data.batch_size
        T_pc = data.T_pc[:batch_size]
        R_pc_all = T_pc[:, :, :3, :3].contiguous()
        t_pc_all = T_pc[:, :, :3, 3].contiguous()

        q = data._q
        if self.has_floating_base and q is not None:
            q_base = q[:, :7]
            R_base = quat_to_rotmat_fast(q_base[:, 3:])
            t_base = q_base[:, :3]
        else:
            R_base = self._I44[:, :3, :3].expand(batch_size, -1, -1)
            t_base = T_pc.new_zeros(batch_size, 3)

        R_world_nodes = [None] * self.n_frames
        t_world_nodes = [None] * self.n_frames

        for node_idx in self._chain.topo_order:
            p_idx = self._chain.parent_list[node_idx]
            R_pc_i = R_pc_all[:, node_idx]
            t_pc_i = t_pc_all[:, node_idx]

            if p_idx == -1:
                R_p, t_p = R_base, t_base
            else:
                R_p, t_p = R_world_nodes[p_idx], t_world_nodes[p_idx]

            R_world_nodes[node_idx] = R_p @ R_pc_i
            t_world_nodes[node_idx] = (R_p @ t_pc_i.unsqueeze(-1)).squeeze(-1) + t_p

        T_world = data.T_world[:batch_size]
        T_world[:, :, :3, :3] = torch.stack(R_world_nodes, dim=1)
        T_world[:, :, :3, 3] = torch.stack(t_world_nodes, dim=1)
        T_world[:, :, 3, :3] = 0
        T_world[:, :, 3, 3] = 1.0

        # Also set T_world_base scratch for gravity computation
        if self.has_floating_base and q is not None:
            T_world_base = data._T_scratch[:batch_size]
            T_world_base[:, :3, :3] = R_base
            T_world_base[:, :3, 3] = t_base
            T_world_base[:, 3, :3] = 0
            T_world_base[:, 3, 3] = 1.0

        data._t_world_valid = True

    def _get_root_t_world(self, data: Data) -> torch.Tensor:
        """Get T_world for root node only (for gravity in RNEA/ABA).

        Computes only the root transform without full T_world propagation.
        """
        if data._t_world_valid:
            return data.T_world[: data.batch_size, self._chain.topo_order[0]]

        self._ensure_t_pc(data)
        batch_size = data.batch_size
        q = data._q
        T_pc_root = data.T_pc[:batch_size, self._chain.topo_order[0]]

        if self.has_floating_base and q is not None:
            q_base = q[:, :7]
            R_base = quat_to_rotmat_fast(q_base[:, 3:])
            t_base = q_base[:, :3]
            # T_world_root = T_base @ T_pc_root
            T_base = data._T_scratch[:batch_size]
            T_base[:, :3, :3] = R_base
            T_base[:, :3, 3] = t_base
            T_base[:, 3, :3] = 0
            T_base[:, 3, 3] = 1.0
            return T_base @ T_pc_root
        else:
            return T_pc_root

    # ========================================================================
    # Standalone FK (path-only, no cache)
    # ========================================================================

    def _build_fk_collapsed_paths(self):
        """Pre-compute collapsed FK paths for all frames.

        Merges consecutive fixed joints into a single (R, t) pair to reduce
        the number of chain matmuls at runtime.
        Each entry is a tuple: (type, ...) where type is:
          0 = merged fixed: (0, R_3x3, t_3)
          1 = revolute: (1, node_idx, joint_idx)
          2 = prismatic: (2, node_idx, joint_idx)
        """
        collapsed = []
        for fid in range(self.n_frames):
            path_nodes = self._chain.parents_indices_list[fid]
            entries = []
            # Accumulate consecutive fixed joints
            R_acc = None
            t_acc = None
            for ni in path_nodes:
                jt = self._chain.joint_type_indices_list[ni]
                if jt == self._FIXED:
                    R_i = self.T_static_pc[ni, :3, :3]
                    t_i = self.T_static_pc[ni, :3, 3]
                    if R_acc is None:
                        R_acc = R_i.clone()
                        t_acc = t_i.clone()
                    else:
                        t_acc = R_acc @ t_i + t_acc
                        R_acc = R_acc @ R_i
                else:
                    # Flush accumulated fixed joints
                    if R_acc is not None:
                        entries.append((0, R_acc, t_acc))
                        R_acc = None
                        t_acc = None
                    ji = self._chain.joint_indices_list[ni]
                    if jt == self._REVOLUTE:
                        entries.append((1, ni, ji))
                    else:
                        entries.append((2, ni, ji))
            # Flush trailing fixed joints
            if R_acc is not None:
                entries.append((0, R_acc, t_acc))
            collapsed.append(entries)
        return collapsed

    def _fk_impl(self, data: Data, q: torch.Tensor, frame_id: int) -> torch.Tensor:
        batch_size = q.shape[0]
        collapsed_path = self._fk_collapsed_paths[frame_id]

        if self.has_floating_base:
            q_base = q[:, :7]
            q_joints = q[:, 7:]
        else:
            q_base = None
            q_joints = q

        # Use decomposed R (B,3,3) + t (B,3) propagation instead of 4x4 matmul
        if self.has_floating_base and q_base is not None:
            t_world = q_base[:, :3]
            R_world = quat_to_rotmat_fast(q_base[:, 3:])
        else:
            R_world = self._I44[0, :3, :3].unsqueeze(0).expand(batch_size, 3, 3).contiguous()
            t_world = q.new_zeros(batch_size, 3)

        for entry in collapsed_path:
            entry_type = entry[0]
            if entry_type == 0:  # merged fixed joints
                R_pc = entry[1]  # (3,3)
                t_pc = entry[2]  # (3,)
            elif entry_type == 1:  # revolute
                node_idx = entry[1]
                joint_idx = entry[2]
                angle = q_joints[:, joint_idx]
                s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
                one_c = (1.0 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1)
                R_pc = (
                    self.R_static_rot[node_idx]
                    + s * self.A_rot[node_idx]
                    + one_c * self.B_rot[node_idx]
                )
                s3 = s.squeeze(-1)
                one_c3 = one_c.squeeze(-1)
                t_pc = (
                    self.t_static_rot[node_idx]
                    + s3 * self.a_trans[node_idx]
                    + one_c3 * self.b_trans[node_idx]
                )
            else:  # prismatic
                node_idx = entry[1]
                joint_idx = entry[2]
                d_val = q_joints[:, joint_idx]
                axis = self.axes_norm[joint_idx]
                t_motion = axis * d_val.unsqueeze(-1)
                t_pc = (
                    self.R_pre[node_idx] @ (t_motion + self.t_post[node_idx]).unsqueeze(-1)
                ).squeeze(-1) + self.t_pre[node_idx]
                R_pc = self.T_static_pc[node_idx, :3, :3]

            # Propagate: t_new = R @ t_pc + t, R_new = R @ R_pc
            t_world = (R_world @ t_pc.unsqueeze(-1)).squeeze(-1) + t_world
            R_world = R_world @ R_pc

        # Assemble final (B,4,4) result — functional to enable cudagraphs
        top = torch.cat([R_world, t_world.unsqueeze(-1)], dim=-1)  # (B,3,4)
        bottom = R_world.new_zeros(batch_size, 1, 4)
        bottom[:, 0, 3] = 1.0
        return torch.cat([top, bottom], dim=-2)  # (B,4,4)

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
        self._ensure_t_world(data)
        batch_size = data.batch_size
        path_nodes = self._chain.parents_indices_list[frame_id]
        nv_base = 6 if self.has_floating_base else 0
        use_world = reference_frame == "world"

        T_world_to_frame = data.T_world[:batch_size, frame_id]
        R_frame = T_world_to_frame[:, :3, :3]  # (B, 3, 3)
        p_frame = T_world_to_frame[:, :3, 3]  # (B, 3)

        # Output Jacobian — computed directly in target frame, no final transform
        J = data.Xup.new_zeros(batch_size, 6, self.nv)

        # Floating base columns (6x6 adjoint — one-time, not per-joint)
        if self.has_floating_base:
            T_world_to_base = data.T_world[:batch_size, self._chain.topo_order[0]]
            if use_world:
                Ad_base = spatial_adjoint_fast(T_world_to_base)
            else:
                T_frame_to_base = inv_homogeneous_fast(T_world_to_frame) @ T_world_to_base
                Ad_base = spatial_adjoint_fast(T_frame_to_base)
            J[:, :, :nv_base] = Ad_base

        # Pre-compute frame rotation transpose for local-frame case
        if not use_world:
            Rt_frame = R_frame.transpose(1, 2)  # (B, 3, 3)

        # Joint columns — direct computation without 6x6 adjoint construction
        # For revolute joint with axis a:
        #   Ad(T) @ [0; a] = [p × (R@a); R@a]
        # For prismatic joint with axis a:
        #   Ad(T) @ [a; 0] = [R@a; 0]
        # Pre-computed: _jac_rotated_axis[i] = R_joint_offset @ axis
        #               _jac_p_offset[i] = joint offset translation
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
                    T_wp = data.T_world[:batch_size, self._chain.topo_order[0]]
                else:
                    T_wp = self._I44.expand(batch_size, -1, -1)
            else:
                T_wp = data.T_world[:batch_size, p_idx]

            R_parent = T_wp[:, :3, :3]  # (B, 3, 3)
            p_parent = T_wp[:, :3, 3]  # (B, 3)

            # Ra_world = R_parent @ (R_offset @ axis) — pre-computed inner product
            rotated_axis = self._jac_rotated_axis[node_idx]  # (3,)
            Ra_world = R_parent @ rotated_axis  # (B, 3)

            col = nv_base + joint_idx

            if is_revolute:
                # p_joint_world = R_parent @ p_offset + p_parent
                p_offset = self._jac_p_offset[node_idx]  # (3,)
                p_wj = R_parent @ p_offset + p_parent  # (B, 3)

                if use_world:
                    J[:, :3, col] = torch.linalg.cross(p_wj, Ra_world, dim=1)
                    J[:, 3:, col] = Ra_world
                else:
                    Ra_local = (Rt_frame @ Ra_world.unsqueeze(-1)).squeeze(-1)
                    p_rel = (Rt_frame @ (p_wj - p_frame).unsqueeze(-1)).squeeze(-1)
                    J[:, :3, col] = torch.linalg.cross(p_rel, Ra_local, dim=1)
                    J[:, 3:, col] = Ra_local
            else:
                # Prismatic: J_col = [R@a; 0]
                if use_world:
                    J[:, :3, col] = Ra_world
                else:
                    J[:, :3, col] = (Rt_frame @ Ra_world.unsqueeze(-1)).squeeze(-1)

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
        self._ensure_xup(data)
        batch_size = data.batch_size
        Xup = data.Xup[:batch_size]
        S = data.S[:batch_size]
        v = data.v[:batch_size]
        vJ = data.vJ[:batch_size]

        needs_grad = qdd.requires_grad
        if needs_grad:
            a = qdd.new_zeros(batch_size, self.n_frames, 6, 1)
            f = qdd.new_zeros(batch_size, self.n_frames, 6, 1)
        else:
            a = data.a[:batch_size]
            a.zero_()
            f = data.f[:batch_size]
            f.zero_()

        if self.has_floating_base:
            a_base = qdd[:, :6]
            a_joints = qdd[:, 6:]
        else:
            a_base = None
            a_joints = qdd

        # Gravity construction
        g = self.gravity if gravity is None else gravity
        neg_g = (-g).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        a_gravity_world = torch.cat([neg_g, neg_g.new_zeros(batch_size, 3, 1)], dim=1)

        if self.has_floating_base:
            T_world_root = self._get_root_t_world(data)
            Ad_inv = spatial_adjoint_fast(inv_homogeneous_fast(T_world_root))
            a_gravity_base = Ad_inv @ a_gravity_world
        else:
            a_gravity_base = a_gravity_world

        # Forward pass: level-order batched acceleration computation
        for nodes_t, parents_t, act_pos_t, act_jnt_t in self._tree_levels:
            n_level = nodes_t.shape[0]

            # Gather parent accelerations for this level
            if parents_t[0] == -1:
                if self.has_floating_base:
                    a_parent = (
                        (a_base.unsqueeze(-1) + a_gravity_base)
                        .unsqueeze(1)
                        .expand(-1, n_level, -1, -1)
                    )
                else:
                    a_parent = a_gravity_base.unsqueeze(1).expand(-1, n_level, -1, -1)
            else:
                a_parent = a[:, parents_t]  # (B, n_level, 6, 1)

            # Batched Xup @ a_parent for all nodes at this level
            a_level = Xup[:, nodes_t] @ a_parent  # (B, n_level, 6, 1)

            # Add joint acceleration and coriolis for actuated joints
            if act_pos_t is not None:
                a_joint = a_joints[:, act_jnt_t].unsqueeze(-1).unsqueeze(-1)  # (B, n_act, 1, 1)
                a_level[:, act_pos_t] = a_level[:, act_pos_t] + S[:, nodes_t[act_pos_t]] * a_joint

                # Motion cross product: ad_v @ vJ for actuated joints only
                v_act = v[:, nodes_t[act_pos_t]]  # (B, n_act, 6, 1)
                vJ_act = vJ[:, nodes_t[act_pos_t]]  # (B, n_act, 6, 1)
                crx = torch.cat(
                    [
                        torch.linalg.cross(v_act[:, :, 3:], vJ_act[:, :, :3], dim=2)
                        + torch.linalg.cross(v_act[:, :, :3], vJ_act[:, :, 3:], dim=2),
                        torch.linalg.cross(v_act[:, :, 3:], vJ_act[:, :, 3:], dim=2),
                    ],
                    dim=2,
                )
                a_level[:, act_pos_t] = a_level[:, act_pos_t] + crx

            a[:, nodes_t] = a_level

        # Backward pass: vectorized force initialization + per-node propagation
        Xup_T = data.Xup_T[:batch_size]

        # Pre-compute I @ v and I @ a for all nodes at once (single batched matmul)
        I_spatial_expanded = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)
        Iv_all = I_spatial_expanded @ v  # (B, N, 6, 1)

        # Force = I*a + force_cross_product(v, I*v) for all nodes (vectorized)
        v_ang = v[:, :, 3:]  # (B, N, 3, 1)
        v_lin = v[:, :, :3]  # (B, N, 3, 1)
        Iv_lin = Iv_all[:, :, :3]
        Iv_ang = Iv_all[:, :, 3:]
        fcx = torch.cat(
            [
                torch.linalg.cross(v_ang, Iv_lin, dim=2),
                torch.linalg.cross(v_lin, Iv_lin, dim=2) + torch.linalg.cross(v_ang, Iv_ang, dim=2),
            ],
            dim=2,
        )
        f[:] = I_spatial_expanded @ a + fcx

        # Propagate child forces to parents (per-node, reverse topological order)
        for node_idx in reversed(self._chain.topo_order):
            children = self._chain.children_list[node_idx]
            if len(children) == 1:
                f[:, node_idx] = f[:, node_idx] + Xup_T[:, children[0]] @ f[:, children[0]]
            elif len(children) > 1:
                for child_idx in children:
                    f[:, node_idx] = f[:, node_idx] + Xup_T[:, child_idx] @ f[:, child_idx]

        # Extract generalized forces
        tau = qdd.new_zeros(batch_size, self.nv)
        tau_all_nodes = (S * f).sum(dim=2).squeeze(-1)  # (B, N)

        if self.has_floating_base:
            tau[:, :6] = f[:, self.urdf_root_idx, :, 0]
        tau[:, self._actuated_vel_indices_t] = tau_all_nodes[:, self._actuated_nodes_t]

        return tau

    # ========================================================================
    # CRBA (Mass Matrix)
    # ========================================================================

    def _crba_impl(self, data: Data) -> torch.Tensor:
        self._ensure_xup(data)
        batch_size = data.batch_size
        Xup = data.Xup
        S = data.S

        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)
        Ic = [I_spatial_batched[:, i].clone() for i in range(self.n_frames)]
        M = Xup.new_zeros(batch_size, self.nv, self.nv)
        Xup_T = data.Xup_T[:batch_size]

        # Backward pass: composite inertia accumulation
        use_triton = Xup.is_cuda and self._use_triton_kernels and not Xup.requires_grad
        if use_triton:
            for node_idx in reversed(self._chain.topo_order):
                p = self._chain.parent_list[node_idx]
                if p != -1:
                    fused_xtmx_add(
                        Xup_T[:, node_idx],
                        Ic[node_idx],
                        Xup[:batch_size, node_idx],
                        Ic[p],
                    )
        else:
            for node_idx in reversed(self._chain.topo_order):
                p = self._chain.parent_list[node_idx]
                if p != -1:
                    Ic[p] = Ic[p] + (Xup_T[:, node_idx] @ Ic[node_idx] @ Xup[:batch_size, node_idx])

        # Assemble mass matrix
        root_idx = self.urdf_root_idx
        if self.has_floating_base:
            M[:, :6, :6] = Ic[root_idx]

        for node_idx in self._chain.topo_order:
            col_idx = self.vel_indices_list[node_idx]
            if col_idx == -1:
                continue

            S_i = S[:batch_size, node_idx]
            F_i = Ic[node_idx] @ S_i

            # Diagonal element
            M[:, col_idx, col_idx] = (S_i.transpose(1, 2) @ F_i).squeeze(-1).squeeze(-1)

            # Single walk-up: joint-joint coupling + base-joint coupling merged
            f_prop = F_i
            current_node = node_idx
            f_at_base = F_i if (self.has_floating_base and node_idx == root_idx) else None

            while True:
                parent = self._chain.parent_list[current_node]
                if parent == -1:
                    break
                f_prop = Xup_T[:, current_node] @ f_prop
                current_node = parent

                # Capture f when reaching urdf_root for base-joint coupling
                if self.has_floating_base and current_node == root_idx and f_at_base is None:
                    f_at_base = f_prop

                parent_col = self.vel_indices_list[current_node]
                if parent_col != -1:
                    S_parent = S[:batch_size, current_node]
                    value = (S_parent.transpose(1, 2) @ f_prop).squeeze(-1).squeeze(-1)
                    M[:, col_idx, parent_col] = value
                    M[:, parent_col, col_idx] = value

            if self.has_floating_base and f_at_base is not None:
                M[:, :6, col_idx] = f_at_base.squeeze(-1)
                M[:, col_idx, :6] = f_at_base.squeeze(-1)

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
        self._ensure_xup(data)
        batch_size = data.batch_size
        Xup = data.Xup
        S = data.S
        v = data.v
        vJ = data.vJ  # Cached from update_kinematics

        if self.has_floating_base:
            a_base = qdd[:, :6]
            a_joints = qdd[:, 6:]
        else:
            a_base = None
            a_joints = qdd

        # Only traverse path to target frame (not all nodes)
        path_nodes = self._chain.parents_indices_list[frame_id]

        # Dict for path accelerations (sparse — only nodes on the path)
        a_path = {}

        for node_idx in path_nodes:
            j_idx = self._chain.joint_indices_list[node_idx]
            is_actuated = self._is_actuated[node_idx]

            if is_actuated:
                a_joint = a_joints[:, j_idx].view(batch_size, 1, 1)
            else:
                a_joint = self._zero_scalar.expand(batch_size, -1, -1)

            p_idx = self._chain.parent_list[node_idx]
            if p_idx == -1:
                if self.has_floating_base:
                    a_parent = a_base.unsqueeze(-1)
                else:
                    a_parent = self._zero_6x1.expand(batch_size, -1, -1)
            else:
                a_parent = a_path[p_idx]

            # Use cached vJ from update_kinematics
            vJ_i = vJ[:batch_size, node_idx]

            # Functional motion cross product: ad_v × vJ
            v_i = v[:batch_size, node_idx]
            crx = torch.cat(
                [
                    torch.linalg.cross(v_i[:, 3:], vJ_i[:, :3], dim=1)
                    + torch.linalg.cross(v_i[:, :3], vJ_i[:, 3:], dim=1),
                    torch.linalg.cross(v_i[:, 3:], vJ_i[:, 3:], dim=1),
                ],
                dim=1,
            )
            a_path[node_idx] = (
                Xup[:batch_size, node_idx] @ a_parent + S[:batch_size, node_idx] * a_joint + crx
            )

        a_local = a_path[frame_id]

        if reference_frame == "world":
            self._ensure_t_world(data)
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
        self._ensure_xup(data)
        batch_size = data.batch_size
        Xup = data.Xup
        S = data.S
        v = data.v
        cached_vJ = data.vJ

        needs_grad = tau.requires_grad
        I_spatial_batched = self.I_spatial.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if needs_grad:
            # Autograd path: fresh allocations to avoid in-place mutation issues
            IA = I_spatial_batched.clone()
            pA = tau.new_zeros(batch_size, self.n_frames, 6, 1)
            U = tau.new_zeros(batch_size, self.n_frames, 6, 1)
            d = tau.new_zeros(batch_size, self.n_frames)
            u = tau.new_zeros(batch_size, self.n_frames)
            a = tau.new_zeros(batch_size, self.n_frames, 6, 1)
            c = tau.new_zeros(batch_size, self.n_frames, 6, 1)
        else:
            # Fast path: use pre-allocated workspace buffers (avoids GPU malloc)
            IA = data.IA[:batch_size]
            IA.copy_(I_spatial_batched)
            pA = data.pA[:batch_size]
            pA.zero_()
            U = data.U[:batch_size]
            d = data.d[:batch_size]
            u = data.u[:batch_size]
            a = data.a[:batch_size]
            a.zero_()
            c = data.f[:batch_size]  # Reuse f buffer for coriolis
            c.zero_()

        if self.has_floating_base:
            tau_base = tau[:, :6]
            tau_joints = tau[:, 6:]
        else:
            tau_base = None
            tau_joints = tau

        g = self.gravity if gravity is None else gravity

        # Functional gravity construction
        neg_g = (-g).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (B, 3, 1)
        a_gravity_world = torch.cat([neg_g, neg_g.new_zeros(batch_size, 3, 1)], dim=1)

        if self.has_floating_base:
            T_world_root = self._get_root_t_world(data)
            Ad_inv = spatial_adjoint_fast(inv_homogeneous_fast(T_world_root))
            a_gravity_base = Ad_inv @ a_gravity_world
        else:
            a_gravity_base = a_gravity_world

        # ---- Pass 1: Initialize IA=I, compute pA and c — fully vectorized ----
        # pA = ad*_v @ (I * v) for all nodes at once
        v_b = v[:batch_size]  # ensure correct batch slice
        Iv_all = I_spatial_batched @ v_b  # (B, N, 6, 1)
        v_ang = v_b[:, :, 3:]  # (B, N, 3, 1)
        v_lin = v_b[:, :, :3]  # (B, N, 3, 1)
        Iv_lin = Iv_all[:, :, :3]
        Iv_ang = Iv_all[:, :, 3:]
        pA[:] = torch.cat(
            [
                torch.linalg.cross(v_ang, Iv_lin, dim=2),
                torch.linalg.cross(v_lin, Iv_lin, dim=2) + torch.linalg.cross(v_ang, Iv_ang, dim=2),
            ],
            dim=2,
        )

        # c = ad_v @ vJ for actuated joints only (vectorized)
        if len(self._actuated_nodes_t) > 0:
            act_t = self._actuated_nodes_t
            v_act = v_b[:, act_t]  # (B, n_act, 6, 1)
            vJ_act = cached_vJ[:batch_size, act_t]  # (B, n_act, 6, 1)
            c[:, act_t] = torch.cat(
                [
                    torch.linalg.cross(v_act[:, :, 3:], vJ_act[:, :, :3], dim=2)
                    + torch.linalg.cross(v_act[:, :, :3], vJ_act[:, :, 3:], dim=2),
                    torch.linalg.cross(v_act[:, :, 3:], vJ_act[:, :, 3:], dim=2),
                ],
                dim=2,
            )

        # ---- Pass 2: Backward — accumulate articulated body inertia ----
        # Use fused Triton kernel for Xup_T @ Ia @ Xup when on CUDA and no grads needed
        use_triton = Xup.is_cuda and self._use_triton_kernels and not tau.requires_grad
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
                    Xup_i_T = data.Xup_T[:batch_size, node_idx]
                    if use_triton:
                        fused_xtmx_add(Xup_i_T, Ia, Xup[:batch_size, node_idx], IA[:, p_idx])
                    else:
                        IA[:, p_idx] = IA[:, p_idx] + Xup_i_T @ Ia @ Xup[:batch_size, node_idx]
                    pA[:, p_idx] = pA[:, p_idx] + Xup_i_T @ pa
            else:
                # Fixed joint: propagate IA and pA upward
                if p_idx != -1:
                    Xup_i_T = data.Xup_T[:batch_size, node_idx]
                    if use_triton:
                        fused_xtmx_add(
                            Xup_i_T,
                            IA[:, node_idx],
                            Xup[:batch_size, node_idx],
                            IA[:, p_idx],
                        )
                    else:
                        IA[:, p_idx] = (
                            IA[:, p_idx] + Xup_i_T @ IA[:, node_idx] @ Xup[:batch_size, node_idx]
                        )
                    pA[:, p_idx] = pA[:, p_idx] + Xup_i_T @ pA[:, node_idx]

        # ---- Pass 3: Forward — compute accelerations ----
        root_idx = self.urdf_root_idx
        qdd_out = tau.new_zeros(batch_size, self.nv)

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

            if is_actuated:
                # a'_i = Xup_i * a_parent + c_i
                a_prime = Xup[:batch_size, node_idx] @ a_parent + c[:, node_idx]
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
                # Fixed joint: c=0, so a_i = Xup @ a_parent
                a[:, node_idx] = Xup[:batch_size, node_idx] @ a_parent

        return qdd_out

    def __str__(self) -> str:
        return str(self._chain)
