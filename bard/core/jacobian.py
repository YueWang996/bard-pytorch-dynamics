from typing import Optional, Tuple, List, Union
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
    to_matrix44,
    inv_homogeneous,
    spatial_adjoint,
    normalize_axis,
)


# ---------- robust POE Jacobian for Chain (full-dim, returns 6 x n_dof) ----------
def calc_jacobian(
    chain: chain.Chain,
    q: torch.Tensor,
    frame_id: Optional[Union[int, str]],
    reference_frame: str = "world",  # "world" (default) or "local"
    return_eef_pose: bool = False,
):
    """
    POE/Adjoint Jacobian (geometric) for a *tree-like* Chain.

    This now supports both fixed-base and floating-base robots.
    For floating-base, q = [tx, ty, tz, qw, qx, qy, qz, joint_angles...].
    The resulting Jacobian will have 6 (base) + n_joints (articulated) columns.

    Convention: spatial vectors are [v; ω] so that [v; ω] = J * v_dot.

    Args:
        chain: Chain (full robot, tree allowed).
        q: (D,) or (B, D) tensor/array/list/dict.
            For fixed-base, D = n_joints.
            For floating-base, D = 7 + n_joints.
            Order for joints per chain.get_joint_parameter_names().
        frame_id: int or str. Name or index of the target frame. You can obtain the index
            via `chain.get_frame_indices(name).item()`.
        reference_frame: "world" (default) returns J expressed in WORLD.
                         "local" returns J expressed in the target frame's LOCAL (body) frame.
        return_eef_pose: If True, also return WORLD pose of the frame (B,4,4).

    Returns:
        J: (B, 6, D_vel) in the requested `reference_frame`, where D_vel is nv_base + n_joints.
        (optionally) T_world_to_frame: (B, 4, 4) — WORLD pose of the frame.
    """
    # normalize inputs
    if hasattr(chain, "ensure_tensor"):
        q_in = chain.ensure_tensor(q)
    else:
        q_in = torch.as_tensor(q, dtype=chain.dtype, device=chain.device)
    if q_in.ndim == 1:
        q_in = q_in.unsqueeze(0)

    B, D = q_in.shape
    dtype, device = chain.dtype, chain.device

    # ---- split floating-base vs joint coordinates ----
    has_fb = bool(getattr(chain, "has_floating_base", False))
    nq_base = 7 if has_fb else 0
    nv_base = int(getattr(chain, "nv_base", 0))
    n_joints = chain.n_joints
    expect_with_base = n_joints + nq_base

    if has_fb and D == expect_with_base:
        q_base = q_in[:, :7]  # [tx,ty,tz, qw,qx,qy,qz]
        q_j = q_in[:, 7:]
    elif D == n_joints:
        # Assume fixed-base if dimensions match
        q_base = None
        q_j = q_in
    else:
        raise ValueError(
            f"Input q has dimension {D}, but expected {n_joints} (fixed-base) or {expect_with_base} (floating-base)"
        )

    # Precompute per-joint motion transforms using articulated joint values
    axes = chain.axes.to(dtype=dtype, device=device)
    axes_expanded = axes.unsqueeze(0).repeat(B, 1, 1)
    T_rev = axis_and_angle_to_matrix_44(axes_expanded, q_j)
    T_pri = axis_and_d_to_pris_matrix(axes_expanded, q_j)

    # Identify node indices along the path root->target frame
    if isinstance(frame_id, str):
        tgt_idx = chain.get_frame_indices(frame_id).item()
    else:
        tgt_idx = int(frame_id)
    path_nodes = chain.parents_indices[tgt_idx]

    # PASS 1: accumulate WORLD transforms and store WORLD->joint_origin for joints on the path
    I44 = identity_transform(B, dtype, device)

    # Seed WORLD transform with base pose if present
    if q_base is not None:
        t = q_base[:, :3]
        qwqxqyqz = q_base[:, 3:]
        qwqxqyqz = qwqxqyqz / qwqxqyqz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        qw, qx, qy, qz = qwqxqyqz.unbind(-1)
        
        two = torch.tensor(2.0, dtype=dtype, device=device)
        x2, y2, z2 = two * qx * qx, two * qy * qy, two * qz * qz
        xy, xz, yz = two * qx * qy, two * qx * qz, two * qy * qz
        wz, wy, wx = two * qw * qz, two * qw * qy, two * qw * qx

        R_base_to_world = torch.empty(B, 3, 3, dtype=dtype, device=device)
        R_base_to_world[:, 0, 0] = 1.0 - (y2 + z2)
        R_base_to_world[:, 0, 1] = xy - wz
        R_base_to_world[:, 0, 2] = xz + wy
        R_base_to_world[:, 1, 0] = xy + wz
        R_base_to_world[:, 1, 1] = 1.0 - (x2 + z2)
        R_base_to_world[:, 1, 2] = yz - wx
        R_base_to_world[:, 2, 0] = xz - wy
        R_base_to_world[:, 2, 1] = yz + wx
        R_base_to_world[:, 2, 2] = 1.0 - (x2 + y2)

        T_world_to_base = torch.zeros(B, 4, 4, dtype=dtype, device=device)
        T_world_to_base[:, :3, :3] = R_base_to_world
        T_world_to_base[:, :3, 3] = t
        T_world_to_base[:, 3, 3] = 1.0
        T_world_to_current = T_world_to_base
    else:
        T_world_to_base = I44.clone()
        T_world_to_current = I44.clone()

    world_T_joint_origin = []
    active_joint_types = []
    active_joint_cols = []
    active_joint_axes_local = []

    for node_idx_t in path_nodes:
        i = int(node_idx_t)
        T_joint_off = to_matrix44(chain.joint_offsets[i])
        T_joint_origin = as_batched_transform(T_joint_off, B, dtype, device)
        j_idx, j_type = int(chain.joint_indices[i]), int(chain.joint_type_indices[i])

        T_world_to_joint_origin = T_world_to_current @ T_joint_origin

        if j_type in [Joint.TYPES.index("revolute"), Joint.TYPES.index("prismatic")]:
            world_T_joint_origin.append(T_world_to_joint_origin)
            active_joint_types.append(j_type)
            active_joint_cols.append(j_idx)
            active_joint_axes_local.append(axes[j_idx])

        if j_type == Joint.TYPES.index("revolute"):
            T_motion = T_rev[:, j_idx]
        elif j_type == Joint.TYPES.index("prismatic"):
            T_motion = T_pri[:, j_idx]
        else:  # fixed
            T_motion = I44

        T_link_off = to_matrix44(chain.link_offsets[i])
        T_link = as_batched_transform(T_link_off, B, dtype, device)
        T_world_to_current = T_world_to_current @ T_joint_origin @ T_motion @ T_link

    T_world_to_frame = T_world_to_current

    # PASS 2: LOCAL Jacobian at 'frame'
    n_dof = nv_base + n_joints
    J_local = torch.zeros((B, 6, n_dof), dtype=dtype, device=device)
    T_frame_to_world = inv_homogeneous(T_world_to_frame)

    # Floating base Jacobian columns
    if has_fb:
        T_frame_to_base = T_frame_to_world @ T_world_to_base
        Ad_frame_wrt_base = spatial_adjoint(T_frame_to_base)
        J_local[:, :, :nv_base] = Ad_frame_wrt_base

    # Articulated joint Jacobian columns
    for jtype, jcol, axis_local, T_w_to_j in zip(
        active_joint_types, active_joint_cols, active_joint_axes_local, world_T_joint_origin
    ):
        T_frame_to_joint_origin = T_frame_to_world @ T_w_to_j
        Ad_frame_wrt_joint = spatial_adjoint(T_frame_to_joint_origin)

        axis_local_batch = axis_local.view(1, 3).expand(B, -1)
        axis_local_unit = normalize_axis(axis_local_batch)

        twist_joint = torch.zeros((B, 6), dtype=dtype, device=device)
        if jtype == Joint.TYPES.index("revolute"):
            twist_joint[:, 3:] = axis_local_unit  # ω
        else:  # prismatic
            twist_joint[:, :3] = axis_local_unit  # v

        col_vec = (Ad_frame_wrt_joint @ twist_joint.unsqueeze(-1)).squeeze(-1)
        J_local[:, :, nv_base + jcol] = col_vec

    # WORLD-frame using adjoint
    Ad_world_wrt_frame = spatial_adjoint(T_world_to_frame)
    J_world = Ad_world_wrt_frame @ J_local

    # Select requested frame
    ref = reference_frame.lower()
    if ref in ("world", "global"):
        J = J_world
    elif ref in ("local", "body", "frame"):
        J = J_local
    else:
        raise ValueError(f"reference_frame must be 'world' or 'local', got: {reference_frame!r}")

    if return_eef_pose:
        return J, T_world_to_frame
    return J
