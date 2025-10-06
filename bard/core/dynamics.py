"""
Robot dynamics algorithms for inverse dynamics and mass matrix computation.

This module implements:
- Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics
- Composite Rigid Body Algorithm (CRBA) for mass matrix
Both algorithms support tree-structured robots with optional floating bases.
"""

from typing import Optional, List
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
    to_matrix44,
    inv_homogeneous,
    spatial_adjoint,
    motion_cross_product,
    force_cross_product,
)


def calc_inverse_dynamics(
    chain: chain.Chain,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    gravity: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute generalized forces via Recursive Newton-Euler Algorithm (RNEA).
    
    Solves the inverse dynamics problem: given configuration, velocities, and
    accelerations, compute the required forces/torques. Supports both fixed-base
    and floating-base robots with tree structures.
    
    Algorithm Overview:
        1. Forward pass: Propagate velocities and accelerations from base to leaves
        2. Backward pass: Propagate forces from leaves to base
    
    Args:
        chain: Robot kinematic chain
        q: Generalized positions
            - Fixed-base: (B, n_joints) or (n_joints,)
            - Floating-base: (B, 7+n_joints) or (7+n_joints,)
        qd: Generalized velocities
            - Fixed-base: (B, n_joints) or (n_joints,)
            - Floating-base: (B, 6+n_joints) or (6+n_joints,)
        qdd: Generalized accelerations (same shape as qd)
        gravity: Gravity vector (3,) in world frame, default [0, 0, -9.81]
    
    Returns:
        Generalized forces tau:
            - Fixed-base: (B, n_joints) joint torques/forces
            - Floating-base: (B, 6+n_joints) where first 6 are base wrench
    
    Example:
        >>> tau = calc_inverse_dynamics(chain, q, qd, qdd)
        >>> # For robot with 7 joints: tau.shape == (batch_size, 7)
    
    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
    """
    # Extract tree structure ONCE outside compiled region
    parent = chain.parent_array.tolist()
    n_nodes = len(parent)
    children = [
        chain.children_array[i, :chain.children_count[i]].tolist() 
        for i in range(n_nodes)
    ]
    
    # Pre-extract other chain data that needs .item() calls
    joint_indices = chain.joint_indices
    joint_type_indices = chain.joint_type_indices
    axes = chain.axes
    spatial_inertias = chain.spatial_inertias
    
    return _calc_inverse_dynamics_compiled(
        chain, q, qd, qdd, gravity, parent, children,
        joint_indices, joint_type_indices, axes, spatial_inertias
    )


@torch.compile
def _calc_inverse_dynamics_compiled(
    chain,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    gravity: Optional[torch.Tensor],
    parent: List[int],
    children: List[List[int]],
    joint_indices: torch.Tensor,
    joint_type_indices: torch.Tensor,
    axes: torch.Tensor,
    spatial_inertias: torch.Tensor,
) -> torch.Tensor:
    """Compiled RNEA implementation."""
    dtype, device = chain.dtype, chain.device
    
    # --- Batching and Input Setup ---
    def _to_batch(x):
        t = torch.as_tensor(x, dtype=dtype, device=device)
        return torch.atleast_2d(t).to(dtype=dtype, device=device)

    q_in, qd_in, qdd_in = _to_batch(q), _to_batch(qd), _to_batch(qdd)
    batch_size = q_in.shape[0]

    has_floating_base = bool(getattr(chain, "has_floating_base", False))
    n_joints = chain.n_joints

    if has_floating_base:
        q_base, q_joints = q_in[:, :7], q_in[:, 7:]
        v_base, v_joints = qd_in[:, :6], qd_in[:, 6:]
        a_base, a_joints = qdd_in[:, :6], qdd_in[:, 6:]
    else:
        q_joints, v_joints, a_joints = q_in, qd_in, qdd_in

    if gravity is None:
        g = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
    else:
        g = torch.as_tensor(gravity, dtype=dtype, device=device)
    
    a_gravity_world = torch.zeros((batch_size, 6), dtype=dtype, device=device)
    a_gravity_world[:, :3] = -g.expand(batch_size, -1)

    # --- Tree Structure ---
    n_nodes = len(parent)
    
    # Convert to tensors for efficient indexing
    parent_tensor = torch.tensor(parent, device=device, dtype=torch.long)
    joint_indices = joint_indices.to(device=device, dtype=torch.long)
    joint_type_indices = joint_type_indices.to(device=device, dtype=torch.long)

    # Topological sort
    topo_order = []
    stack = [i for i, p in enumerate(parent) if p == -1]
    while stack:
        node_idx = stack.pop(0)
        topo_order.append(node_idx)
        stack.extend(children[node_idx])

    axes = axes.to(dtype=dtype, device=device)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    
    # Pre-gather q, qd, qdd for joints based on node order
    q_nodes = q_joints[:, joint_indices]
    v_nodes = v_joints[:, joint_indices]
    a_nodes = a_joints[:, joint_indices]

    # --- Storage ---
    Xup = torch.zeros(n_nodes, batch_size, 6, 6, dtype=dtype, device=device)
    S = torch.zeros(n_nodes, batch_size, 6, dtype=dtype, device=device)
    v = torch.zeros(n_nodes, batch_size, 6, dtype=dtype, device=device)
    a = torch.zeros(n_nodes, batch_size, 6, dtype=dtype, device=device)
    f = torch.zeros(n_nodes, batch_size, 6, dtype=dtype, device=device)
    I_spatial = spatial_inertias.unsqueeze(1).expand(-1, batch_size, -1, -1).to(dtype=dtype, device=device)

    I44 = identity_transform(batch_size, dtype, device)

    if has_floating_base:
        # Build base transform from quaternion
        t = q_base[:, :3]
        qwqxqyqz = q_base[:, 3:]
        qwqxqyqz = qwqxqyqz / qwqxqyqz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        qw, qx, qy, qz = qwqxqyqz.unbind(-1)
        
        two = torch.tensor(2.0, dtype=dtype, device=device)
        x2, y2, z2 = two * qx * qx, two * qy * qy, two * qz * qz
        xy, xz, yz = two * qx * qy, two * qx * qz, two * qy * qz
        wz, wy, wx = two * qw * qz, two * qw * qy, two * qw * qx
        
        R = torch.empty(batch_size, 3, 3, dtype=dtype, device=device)
        R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = 1.0 - (y2 + z2), xy - wz, xz + wy
        R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = xy + wz, 1.0 - (x2 + z2), yz - wx
        R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = xz - wy, yz + wx, 1.0 - (x2 + y2)
        
        T_world_to_base = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)
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
    
    for node_idx in topo_order:
        j_idx = joint_indices[node_idx]
        j_type = joint_type_indices[node_idx]
        p_idx = parent_tensor[node_idx]

        T_joint_offset = as_batched_transform(
            to_matrix44(chain.joint_offsets[node_idx]), batch_size, dtype, device
        )
        T_link_offset = as_batched_transform(
            to_matrix44(chain.link_offsets[node_idx]), batch_size, dtype, device
        )
        
        # Calculate all motion transforms, then select with where
        is_revolute = (j_type == Joint.TYPES.index('revolute'))
        is_prismatic = (j_type == Joint.TYPES.index('prismatic'))
        is_actuated = is_revolute | is_prismatic
        
        T_rev = axis_and_angle_to_matrix_44(
            axes[j_idx].expand(batch_size, -1), q_nodes[:, node_idx]
        )
        T_pris = axis_and_d_to_pris_matrix(
            axes[j_idx].expand(batch_size, -1), q_nodes[:, node_idx]
        )
        
        T_motion = torch.where(
            is_revolute.view(1, 1, 1).expand(batch_size, 4, 4),
            T_rev,
            I44
        )
        T_motion = torch.where(
            is_prismatic.view(1, 1, 1).expand(batch_size, 4, 4),
            T_pris,
            T_motion
        )
        
        T_parent_child = T_joint_offset @ T_motion @ T_link_offset
        Xup[node_idx] = spatial_adjoint(inv_homogeneous(T_parent_child))

        # --- Subspace, Velocity, and Acceleration ---
        axis_local = axes[j_idx].expand(batch_size, -1)
        twist_joint = torch.zeros(batch_size, 6, dtype=dtype, device=device)
        twist_joint[:, 3:] = torch.where(
            is_revolute.view(1, 1).expand(batch_size, 3),
            axis_local,
            twist_joint[:, 3:]
        )
        twist_joint[:, :3] = torch.where(
            is_prismatic.view(1, 1).expand(batch_size, 3),
            axis_local,
            twist_joint[:, :3]
        )
        
        S_node = (spatial_adjoint(T_link_offset) @ twist_joint.unsqueeze(-1)).squeeze(-1)
        S[node_idx] = S_node * is_actuated.float().view(1, 1)

        v_joint = v_nodes[:, node_idx].unsqueeze(-1) * is_actuated.float()
        a_joint = a_nodes[:, node_idx].unsqueeze(-1) * is_actuated.float()

        # Parent state using tensor indexing
        is_root = (p_idx == -1)
        if has_floating_base:
            v_parent = torch.where(
                is_root.view(1, 1).expand(batch_size, 6),
                v_base,
                v[p_idx] if not is_root else v[0]  # Fallback
            )
            a_parent_base = a_base + a_gravity_base
            a_parent = torch.where(
                is_root.view(1, 1).expand(batch_size, 6),
                a_parent_base,
                a[p_idx] if not is_root else a[0]  # Fallback
            )
        else:
            zeros_6 = torch.zeros(batch_size, 6, dtype=dtype, device=device)
            v_parent = torch.where(
                is_root.view(1, 1).expand(batch_size, 6),
                zeros_6,
                v[p_idx] if not is_root else zeros_6
            )
            a_parent = torch.where(
                is_root.view(1, 1).expand(batch_size, 6),
                a_gravity_base,
                a[p_idx] if not is_root else a_gravity_base
            )
        
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
    
    for node_idx in reversed(topo_order):
        Iv = (I_spatial[node_idx] @ v[node_idx].unsqueeze(-1)).squeeze(-1)
        f_node = (I_spatial[node_idx] @ a[node_idx].unsqueeze(-1)).squeeze(-1)
        f_node += (force_cross_product(v[node_idx]) @ Iv.unsqueeze(-1)).squeeze(-1)

        # Aggregate forces from children
        for child_idx in children[node_idx]:
            f_node += (
                Xup[child_idx].transpose(1, 2) @ f[child_idx].unsqueeze(-1)
            ).squeeze(-1)
        f[node_idx] = f_node
    
    # ========================================================================
    # Extract generalized forces
    # ========================================================================
    
    is_actuated_mask = (
        (joint_type_indices == Joint.TYPES.index('revolute')) | 
        (joint_type_indices == Joint.TYPES.index('prismatic'))
    )
    
    # Project forces onto joint subspaces
    tau_all_nodes = torch.sum(S * f, dim=-1)  # (n_nodes, batch_size)
    
    if has_floating_base:
        tau = torch.zeros((batch_size, 6 + n_joints), dtype=dtype, device=device)
        urdf_root_idx = 1 if n_nodes > 1 else 0
        tau[:, :6] = f[urdf_root_idx]
        
        # Map actuated joints to their columns
        actuated_indices = torch.nonzero(is_actuated_mask, as_tuple=False).squeeze(-1)
        for idx in actuated_indices:
            j_col = joint_indices[idx]
            tau[:, 6 + j_col] = tau_all_nodes[idx]
    else:
        tau = torch.zeros((batch_size, n_joints), dtype=dtype, device=device)
        actuated_indices = torch.nonzero(is_actuated_mask, as_tuple=False).squeeze(-1)
        for idx in actuated_indices:
            j_col = joint_indices[idx]
            tau[:, j_col] = tau_all_nodes[idx]

    return tau


def crba_inertia_matrix(
    chain: chain.Chain,
    q: torch.Tensor
) -> torch.Tensor:
    """
    Compute mass matrix via Composite Rigid Body Algorithm (CRBA).
    
    Computes the symmetric positive-definite mass matrix M(q) where:
        M(q) * qdd = tau - C(q, qd) - G(q)
    
    The mass matrix relates generalized accelerations to forces/torques.
    
    Args:
        chain: Robot kinematic chain
        q: Generalized positions
            - Fixed-base: (B, n_joints) or (n_joints,)
            - Floating-base: (B, 7+n_joints) or (7+n_joints,)
    
    Returns:
        Mass matrix M:
            - Fixed-base: (B, n_joints, n_joints)
            - Floating-base: (B, 6+n_joints, 6+n_joints)
    
    Properties:
        - Symmetric: M = M^T
        - Positive definite: x^T M x > 0 for all x ≠ 0
        - Configuration dependent: M = M(q)
    
    Example:
        >>> M = crba_inertia_matrix(chain, q)
        >>> # Compute required torque for desired acceleration
        >>> tau = M @ qdd_desired
    
    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
    """
    # Extract tree structure ONCE outside compiled region
    parent = chain.parent_array.tolist()
    n_nodes = len(parent)
    children = [
        chain.children_array[i, :chain.children_count[i]].tolist() 
        for i in range(n_nodes)
    ]
    
    # Pre-extract chain data
    joint_indices = chain.joint_indices
    joint_type_indices = chain.joint_type_indices
    axes = chain.axes
    spatial_inertias = chain.spatial_inertias
    
    return _crba_inertia_matrix_compiled(
        chain, q, parent, children,
        joint_indices, joint_type_indices, axes, spatial_inertias
    )


@torch.compile
def _crba_inertia_matrix_compiled(
    chain,
    q: torch.Tensor,
    parent: List[int],
    children: List[List[int]],
    joint_indices: torch.Tensor,
    joint_type_indices: torch.Tensor,
    axes: torch.Tensor,
    spatial_inertias: torch.Tensor,
) -> torch.Tensor:
    """Compiled CRBA implementation."""
    dtype, device = chain.dtype, chain.device

    def _to_batch(x):
        t = chain.ensure_tensor(x) if hasattr(chain, "ensure_tensor") else torch.as_tensor(x, dtype=dtype, device=device)
        return torch.atleast_2d(t).to(dtype=dtype, device=device)

    q_in = _to_batch(q)
    batch_size = q_in.shape[0]

    has_floating_base = bool(getattr(chain, "has_floating_base", False))
    n_joints = chain.n_joints

    if has_floating_base:
        q_base, q_joints = q_in[:, :7], q_in[:, 7:]
        nv = 6 + n_joints
    else:
        q_base = None
        q_joints = q_in
        nv = n_joints

    # Build tree structure
    n_nodes = len(parent)
    base_nodes = [i for i, p in enumerate(parent) if p == -1]
    
    topo_order = []
    stack = base_nodes[:]
    while stack:
        node_idx = stack.pop(0)
        topo_order.append(node_idx)
        stack.extend(children[node_idx])

    # Precompute transforms
    axes_raw = axes.to(dtype=dtype, device=device)
    axes_norm = axes_raw / axes_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axes_batched = axes_norm.unsqueeze(0).expand(batch_size, -1, -1)
    T_revolute = axis_and_angle_to_matrix_44(axes_batched, q_joints)
    T_prismatic = axis_and_d_to_pris_matrix(axes_batched, q_joints)

    I44 = identity_transform(batch_size, dtype, device)
    Xup = [None] * n_nodes
    S = [torch.zeros((batch_size, 6), dtype=dtype, device=device) for _ in range(n_nodes)]
    I_spatial = [spatial_inertias[i].unsqueeze(0).expand(batch_size, -1, -1) for i in range(n_nodes)]

    # ========================================================================
    # Forward pass: compute transforms and joint subspaces
    # ========================================================================
    
    for node_idx in topo_order:
        joint_idx = joint_indices[node_idx]
        joint_type_idx = joint_type_indices[node_idx]

        T_joint_offset = as_batched_transform(
            to_matrix44(chain.joint_offsets[node_idx]), batch_size, dtype, device
        )
        T_link_offset = as_batched_transform(
            to_matrix44(chain.link_offsets[node_idx]), batch_size, dtype, device
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
            axis_local = axes_norm[joint_idx].view(1, 3).expand(batch_size, -1)
            twist_joint = torch.zeros((batch_size, 6), dtype=dtype, device=device)
            
            if joint_type_idx == Joint.TYPES.index('revolute'):
                twist_joint[:, 3:] = axis_local
            else:
                twist_joint[:, :3] = axis_local
                
            S[node_idx] = (spatial_adjoint(T_link_offset) @ twist_joint.unsqueeze(-1)).squeeze(-1)

    # ========================================================================
    # Backward pass: compute composite inertias
    # ========================================================================
    
    I_composite = [I_spatial[i].clone() for i in range(n_nodes)]
    
    for node_idx in reversed(topo_order):
        p = parent[node_idx]
        if p != -1:
            # Add child's composite inertia to parent
            I_composite[p] = I_composite[p] + (
                Xup[node_idx].transpose(1, 2) @ I_composite[node_idx] @ Xup[node_idx]
            )

    # ========================================================================
    # Assemble mass matrix
    # ========================================================================
    
    M = torch.zeros((batch_size, nv, nv), dtype=dtype, device=device)

    # Map nodes to velocity indices
    node_to_vel_idx = {}
    for node_idx in range(n_nodes):
        joint_type_idx = joint_type_indices[node_idx]
        if joint_type_idx != Joint.TYPES.index('fixed'):
            joint_col = joint_indices[node_idx]
            node_to_vel_idx[node_idx] = (6 + joint_col) if has_floating_base else joint_col

    # Base inertia block (if floating)
    urdf_root_idx = 1 if (has_floating_base and n_nodes > 1) else 0
    if has_floating_base:
        M[:, :6, :6] = I_composite[urdf_root_idx]

    # Joint columns
    for node_idx in topo_order:
        if node_idx not in node_to_vel_idx:
            continue
            
        col_idx = node_to_vel_idx[node_idx]
        S_i = S[node_idx].unsqueeze(-1)
        F_i = I_composite[node_idx] @ S_i

        # Diagonal element
        M[:, col_idx, col_idx] = (S_i.transpose(1, 2) @ F_i).squeeze(-1).squeeze(-1)

        # Base-joint coupling (if floating)
        if has_floating_base:
            # Propagate force to base
            f_at_base = F_i.clone()
            current_node = node_idx
            while current_node != urdf_root_idx and parent[current_node] != -1:
                f_at_base = Xup[current_node].transpose(1, 2) @ f_at_base
                current_node = parent[current_node]
            
            M[:, :6, col_idx] = f_at_base.squeeze(-1)
            M[:, col_idx, :6] = f_at_base.squeeze(-1)

        # Joint-joint coupling
        f = F_i.clone()
        current_node = node_idx
        while parent[current_node] != -1:
            f = Xup[current_node].transpose(1, 2) @ f
            current_node = parent[current_node]
            
            if current_node in node_to_vel_idx:
                parent_col = node_to_vel_idx[current_node]
                S_parent = S[current_node].unsqueeze(-1)
                value = (S_parent.transpose(1, 2) @ f).squeeze(-1).squeeze(-1)
                M[:, col_idx, parent_col] = value
                M[:, parent_col, col_idx] = value

    return M