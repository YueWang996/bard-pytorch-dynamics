import torch
import bard.transforms as tf
from bard.core import chain
from bard.structures import Joint
from bard.transforms import (
    axis_and_angle_to_matrix_44,
    axis_and_d_to_pris_matrix,
)

from .utils import (
    build_parent_children,
    identity_transform,
    as_batched_transform,
    inv_homogeneous,
    motion_cross_product,
    spatial_adjoint,
    to_matrix44,
    normalize_joint_positions,
)


@torch.compile
def calc_forward_kinematics(
    chain: chain.Chain,
    q: torch.Tensor,
    frame_id: int,
) -> tf.Transform3d:
    """Calculates the forward kinematics for a specific frame.

    This function computes the world pose of a single target frame given the
    robot's generalized coordinates. It supports batching for parallel computation.

    Args:
        chain (bard.core.chain.Chain): The robot chain object.
        q (torch.Tensor): The generalized coordinates of the robot.
            Shape (B, nq) or (nq,), where B is the batch size and nq is the
            number of configuration variables.
        frame_id (int): The integer index of the target frame.

    Returns:
        bard.transforms.Transform3d: A Transform3d object containing the
            batched (B, 4, 4) homogeneous transformation matrices representing
            the world pose of the frame.
    """
    # ---- normalize inputs ----
    q_in = normalize_joint_positions(chain, q)  # (B, D)
    B, D = q_in.shape
    dtype, device = chain.dtype, chain.device

    # ---- split floating-base vs joint coordinates (robust to legacy chains) ----
    has_fb = bool(getattr(chain, "has_floating_base", False))
    nq_base = 7 if has_fb else 0
    expect_with_base = (chain.n_joints + nq_base)

    if has_fb and D == expect_with_base:
        q_base = q_in[:, :7]          # [tx,ty,tz, qw,qx,qy,qz]
        q_j    = q_in[:, 7:]
    else:
        # Treat as fixed-base inputs (or legacy caller that only passed joints)
        q_base = None
        q_j    = q_in

    # ---- resolve frame index (accept name for convenience, API prefers index) ----
    tgt_idx = chain.get_frame_indices(frame_id).item() if isinstance(frame_id, str) else int(frame_id)

    # ---- precompute joint motion transforms (vectorized over batch) ----
    axes = chain.axes.to(dtype=dtype, device=device)                         # (nJ, 3)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)           # ensure unit
    axes_exp = axes.unsqueeze(0).expand(B, -1, -1)                            # (B, nJ, 3)
    q_cast  = q_j.to(dtype=dtype)                                             # (B, nJ)

    T_rev = axis_and_angle_to_matrix_44(axes_exp, q_cast)  # (B, nJ, 4, 4)
    T_pri = axis_and_d_to_pris_matrix(axes_exp, q_cast)    # (B, nJ, 4, 4)

    # ---- seed WORLD transform with base pose (if present) ----
    if q_base is not None:
        # Build T_world<-base from [tx,ty,tz, qw,qx,qy,qz]
        t = q_base[:, :3]
        qwqxqyqz = q_base[:, 3:]
        qwqxqyqz = qwqxqyqz / qwqxqyqz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        qw, qx, qy, qz = qwqxqyqz.unbind(-1)

        two = torch.tensor(2.0, dtype=dtype, device=device)
        x2 = two * qx * qx; y2 = two * qy * qy; z2 = two * qz * qz
        xy = two * qx * qy; xz = two * qx * qz; yz = two * qy * qz
        wz = two * qw * qz; wy = two * qw * qy; wx = two * qw * qx

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

        T = torch.zeros(B, 4, 4, dtype=dtype, device=device)
        T[:, :3, :3] = R_base_to_world  # No transpose!
        T[:, :3,  3] = t
        T[:,  3,  3] = 1.0
    else:
        T = identity_transform(B, dtype, device).clone()

    # ---- accumulate WORLD transform along the path root->target ----
    for node_idx in chain.parents_indices[tgt_idx]:
        i = int(node_idx)

        # parent -> joint origin
        T_joint = as_batched_transform(to_matrix44(chain.joint_offsets[i]), B, dtype, device)

        # joint motion in JOINT frame
        j_idx  = int(chain.joint_indices[i])
        j_type = int(chain.joint_type_indices[i])
        if j_type == Joint.TYPES.index('revolute'):
            T_motion = T_rev[:, j_idx]
        elif j_type == Joint.TYPES.index('prismatic'):
            T_motion = T_pri[:, j_idx]
        elif j_type == Joint.TYPES.index('fixed'):
            T_motion = identity_transform(B, dtype, device)
        else:
            raise NotImplementedError(f"Unsupported joint type index: {j_type}")

        # joint -> child link
        T_link = as_batched_transform(to_matrix44(chain.link_offsets[i]), B, dtype, device)

        # compose this node (WORLD <- parent) ∘ joint ∘ motion ∘ link
        T = T @ T_joint @ T_motion @ T_link

        # (optional) small numeric guard: reproject SO(3)
        R = T[:, :3, :3]
        U, _, Vh = torch.linalg.svd(R)
        M = U @ Vh
        det = torch.det(M)
        D = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        D[:, 2, 2] = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
        T[:, :3, :3] = U @ D @ Vh

    # ---- return WORLD pose of the requested frame ----
    return tf.Transform3d(matrix=T)

@torch.compile
def end_effector_acceleration(
    chain,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    frame_name: str,
    reference_frame: str = "local",
):
    """Computes the classical spatial acceleration of a frame.

    This function implements the forward pass of the Recursive Newton-Euler
    Algorithm (RNEA) to determine the acceleration of a body, including
    Coriolis and centrifugal effects.

    Args:
        chain (bard.core.chain.Chain): The robot chain object.
        q (torch.Tensor): Generalized coordinates (position). Shape (B, nq) or (nq,).
        qd (torch.Tensor): Generalized velocities. Shape (B, nv) or (nv,).
        qdd (torch.Tensor): Generalized accelerations. Shape (B, nv) or (nv,).
        frame_name (Union[str, int]): The name or integer index of the target frame.
        reference_frame (str, optional): The frame of reference for the output
            acceleration. Can be "world" or "local". Defaults to "local".

    Returns:
        torch.Tensor: The spatial acceleration vector [linear; angular] of
            shape (B, 6).
    """
    dtype, device = chain.dtype, chain.device

    # Normalize inputs
    if hasattr(chain, "ensure_tensor"):
        q_in = chain.ensure_tensor(q)
        qd_in = chain.ensure_tensor(qd)
        qdd_in = chain.ensure_tensor(qdd)
    else:
        q_in = torch.as_tensor(q, dtype=dtype, device=device)
        qd_in = torch.as_tensor(qd, dtype=dtype, device=device)
        qdd_in = torch.as_tensor(qdd, dtype=dtype, device=device)
    
    if q_in.ndim == 1: q_in = q_in.unsqueeze(0)
    if qd_in.ndim == 1: qd_in = qd_in.unsqueeze(0)
    if qdd_in.ndim == 1: qdd_in = qdd_in.unsqueeze(0)

    B = q_in.shape[0]
    has_fb = bool(getattr(chain, "has_floating_base", False))

    # Split base and joint components
    if has_fb:
        q_base, q_j = q_in[:, :7], q_in[:, 7:]
        v_base, v_j = qd_in[:, :6], qd_in[:, 6:]
        a_base, a_j = qdd_in[:, :6], qdd_in[:, 6:]
    else:
        q_base = v_base = a_base = None
        q_j, v_j, a_j = q_in, qd_in, qdd_in

    # World spatial acceleration (gravity disabled)
    a_world0 = torch.zeros((B, 6), dtype=dtype, device=device)

    # Tree topology
    parent, children = build_parent_children(chain)
    n_nodes = len(parent)
    base_nodes = [i for i, p in enumerate(parent) if p == -1]
    topo = []
    stack = base_nodes[:]
    while stack:
        i = stack.pop(0)
        topo.append(i)
        stack.extend(children[i])

    # Precompute joint transforms
    axes_raw = chain.axes.to(dtype=dtype, device=device)
    axes = axes_raw / axes_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axes_exp = axes.unsqueeze(0).expand(B, -1, -1)
    T_rev = axis_and_angle_to_matrix_44(axes_exp, q_j)
    T_pri = axis_and_d_to_pris_matrix(axes_exp, q_j)

    # Storage
    Xup = [None] * n_nodes
    S = [torch.zeros((B, 6), dtype=dtype, device=device) for _ in range(n_nodes)]
    v = [torch.zeros((B, 6), dtype=dtype, device=device) for _ in range(n_nodes)]
    a = [torch.zeros((B, 6), dtype=dtype, device=device) for _ in range(n_nodes)]
    T_world_to_node = [None] * n_nodes
    I44 = identity_transform(B, dtype, device)

    # Build base transform if floating
    if has_fb and q_base is not None:
        t = q_base[:, :3]
        qwqxqyqz = q_base[:, 3:]
        qwqxqyqz = qwqxqyqz / qwqxqyqz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        qw, qx, qy, qz = qwqxqyqz.unbind(-1)
        
        two = torch.tensor(2.0, dtype=dtype, device=device)
        x2, y2, z2 = two * qx * qx, two * qy * qy, two * qz * qz
        xy, xz, yz = two * qx * qy, two * qx * qz, two * qy * qz
        wz, wy, wx = two * qw * qz, two * qw * qy, two * qw * qx
        
        R = torch.empty(B, 3, 3, dtype=dtype, device=device)
        R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = 1.0 - (y2 + z2), xy - wz, xz + wy
        R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = xy + wz, 1.0 - (x2 + z2), yz - wx
        R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = xz - wy, yz + wx, 1.0 - (x2 + y2)
        
        T_world_to_base = torch.zeros(B, 4, 4, dtype=dtype, device=device)
        T_world_to_base[:, :3, :3] = R
        T_world_to_base[:, :3, 3] = t
        T_world_to_base[:, 3, 3] = 1.0
    else:
        T_world_to_base = I44

    # Forward pass
    for i in topo:
        j_idx = int(chain.joint_indices[i])
        j_type = int(chain.joint_type_indices[i])

        # Joint and link offsets
        T_joint = as_batched_transform(to_matrix44(chain.joint_offsets[i]), B, dtype, device)
        T_link = as_batched_transform(to_matrix44(chain.link_offsets[i]), B, dtype, device)

        # Joint motion
        if j_type == Joint.TYPES.index('revolute'):
            T_motion = T_rev[:, j_idx]
        elif j_type == Joint.TYPES.index('prismatic'):
            T_motion = T_pri[:, j_idx]
        elif j_type == Joint.TYPES.index('fixed'):
            T_motion = I44
        else:
            raise NotImplementedError(f"Unsupported joint type: {j_type}")

        # Parent -> child transform
        T_parent_to_child = T_joint @ T_motion @ T_link
        Xup_i = spatial_adjoint(inv_homogeneous(T_parent_to_child))
        Xup[i] = Xup_i

        # Joint subspace in child frame
        if j_type in [Joint.TYPES.index('revolute'), Joint.TYPES.index('prismatic')]:
            axis_local = axes[j_idx].view(1, 3).expand(B, -1)
            twist_joint = torch.zeros((B, 6), dtype=dtype, device=device)
            if j_type == Joint.TYPES.index('revolute'):
                twist_joint[:, 3:] = axis_local
            else:
                twist_joint[:, :3] = axis_local
            S[i] = (spatial_adjoint(T_link) @ twist_joint.unsqueeze(-1)).squeeze(-1)
            v_j_i = v_j[:, j_idx].unsqueeze(-1)
            a_j_i = a_j[:, j_idx].unsqueeze(-1)
        else:
            v_j_i = torch.zeros((B, 1), dtype=dtype, device=device)
            a_j_i = torch.zeros((B, 1), dtype=dtype, device=device)

        # Parent state
        if parent[i] == -1:
            # Root node - use base velocity/acceleration if floating
            v_parent = v_base if (has_fb and v_base is not None) else torch.zeros((B, 6), dtype=dtype, device=device)
            a_parent = (a_base if (has_fb and a_base is not None) else torch.zeros((B, 6), dtype=dtype, device=device)) + a_world0
            T_world_to_parent = T_world_to_base
        else:
            v_parent = v[parent[i]]
            a_parent = a[parent[i]]
            T_world_to_parent = T_world_to_node[parent[i]]

        # RNEA propagation
        vJ = S[i] * v_j_i
        v[i] = (Xup_i @ v_parent.unsqueeze(-1)).squeeze(-1) + vJ
        a[i] = (Xup_i @ a_parent.unsqueeze(-1)).squeeze(-1) + S[i] * a_j_i \
               + (motion_cross_product(v[i]) @ vJ.unsqueeze(-1)).squeeze(-1)

        # Accumulate world pose
        T_world_to_node[i] = T_world_to_parent @ T_parent_to_child

    # Select frame
    node_idx = chain.get_frame_indices(frame_name).item() if isinstance(frame_name, str) else int(frame_name)
    a_local = a[node_idx]

    ref = reference_frame.lower()
    if ref in ("local", "body", "frame"):
        return a_local.squeeze(0)
    
    if ref in ("world", "global"):
        Ad_world_wrt_body = spatial_adjoint(T_world_to_node[node_idx])
        a_world = (Ad_world_wrt_body @ a_local.unsqueeze(-1)).squeeze(-1)
        return a_world.squeeze(0)
    
    raise ValueError(f"reference_frame must be 'local' or 'world', got: {reference_frame!r}")
