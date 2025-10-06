"""
Robot dynamics algorithms for inverse dynamics and mass matrix computation.

This module implements:
- Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics
- Composite Rigid Body Algorithm (CRBA) for mass matrix
Both algorithms support tree-structured robots with optional floating bases.
"""

from typing import Optional
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
    build_parent_children,
    motion_cross_product,
    force_cross_product,
    compute_spatial_inertia,
    base_pose_to_transform,
)

@torch.compile
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
    dtype, device = chain.dtype, chain.device

    # Normalize inputs to batched form
    def _to_batch(x):
        if hasattr(chain, "ensure_tensor"):
            t = chain.ensure_tensor(x)
        else:
            t = torch.as_tensor(x, dtype=dtype, device=device)
        return torch.atleast_2d(t).to(dtype=dtype, device=device)

    q_in = _to_batch(q)
    qd_in = _to_batch(qd)
    qdd_in = _to_batch(qdd)
    batch_size = q_in.shape[0]

    # Parse floating base
    has_floating_base = bool(getattr(chain, "has_floating_base", False))
    n_joints = chain.n_joints

    if has_floating_base:
        q_base, q_joints = q_in[:, :7], q_in[:, 7:]
        v_base, v_joints = qd_in[:, :6], qd_in[:, 6:]
        a_base, a_joints = qdd_in[:, :6], qdd_in[:, 6:]
    else:
        q_base = v_base = a_base = None
        q_joints, v_joints, a_joints = q_in, qd_in, qdd_in

    # Gravity as spatial acceleration in world frame
    if gravity is None:
        g = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
    else:
        g = torch.as_tensor(gravity, dtype=dtype, device=device)
    g = g.reshape(1, 3).expand(batch_size, -1)
    
    a_gravity_world = torch.zeros((batch_size, 6), dtype=dtype, device=device)
    a_gravity_world[:, :3] = -g

    # Build tree structure
    parent, children = build_parent_children(chain)
    n_nodes = len(parent)
    base_nodes = [i for i, p in enumerate(parent) if p == -1]
    
    # Topological ordering for forward pass
    topo_order = []
    stack = base_nodes[:]
    while stack:
        node_idx = stack.pop(0)
        topo_order.append(node_idx)
        stack.extend(children[node_idx])

    # Precompute joint motion transforms
    axes_raw = chain.axes.to(dtype=dtype, device=device)
    axes = axes_raw / axes_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axes_batched = axes.unsqueeze(0).expand(batch_size, -1, -1)
    T_revolute = axis_and_angle_to_matrix_44(axes_batched, q_joints)
    T_prismatic = axis_and_d_to_pris_matrix(axes_batched, q_joints)

    # Storage for forward pass
    Xup = [None] * n_nodes  # Parent-to-child spatial transforms
    S = [torch.zeros((batch_size, 6), dtype=dtype, device=device) for _ in range(n_nodes)]  # Joint subspaces
    v = [torch.zeros((batch_size, 6), dtype=dtype, device=device) for _ in range(n_nodes)]  # Link velocities
    a = [torch.zeros((batch_size, 6), dtype=dtype, device=device) for _ in range(n_nodes)]  # Link accelerations
    I_spatial = [None] * n_nodes  # Spatial inertias
    f = [torch.zeros((batch_size, 6), dtype=dtype, device=device) for _ in range(n_nodes)]  # Link forces

    I44 = identity_transform(batch_size, dtype, device)

    # Build base transform and transform gravity if floating
    if has_floating_base and q_base is not None:
        T_world_to_base = base_pose_to_transform(q_base)
        # Transform gravity from world to base frame
        Ad_base_world = spatial_adjoint(inv_homogeneous(T_world_to_base))
        a_gravity_base = (Ad_base_world @ a_gravity_world.unsqueeze(-1)).squeeze(-1)
    else:
        a_gravity_base = a_gravity_world

    # ========================================================================
    # Forward pass: propagate velocities and accelerations
    # ========================================================================
    
    for node_idx in topo_order:
        joint_idx = int(chain.joint_indices[node_idx])
        joint_type_idx = int(chain.joint_type_indices[node_idx])

        # Get transforms for this node
        T_joint_offset = as_batched_transform(
            to_matrix44(chain.joint_offsets[node_idx]), batch_size, dtype, device
        )
        T_link_offset = as_batched_transform(
            to_matrix44(chain.link_offsets[node_idx]), batch_size, dtype, device
        )

        # Joint motion transform
        if joint_type_idx == Joint.TYPES.index('revolute'):
            T_motion = T_revolute[:, joint_idx]
        elif joint_type_idx == Joint.TYPES.index('prismatic'):
            T_motion = T_prismatic[:, joint_idx]
        elif joint_type_idx == Joint.TYPES.index('fixed'):
            T_motion = I44
        else:
            raise NotImplementedError(f"Unsupported joint type: {joint_type_idx}")

        # Complete parent-to-child transform
        T_parent_child = T_joint_offset @ T_motion @ T_link_offset
        Xup[node_idx] = spatial_adjoint(inv_homogeneous(T_parent_child))

        # Joint subspace (motion direction)
        if joint_type_idx in [Joint.TYPES.index('revolute'), Joint.TYPES.index('prismatic')]:
            axis_local = axes[joint_idx].view(1, 3).expand(batch_size, -1)
            twist_joint = torch.zeros((batch_size, 6), dtype=dtype, device=device)
            
            if joint_type_idx == Joint.TYPES.index('revolute'):
                twist_joint[:, 3:] = axis_local  # Angular velocity
            else:  # prismatic
                twist_joint[:, :3] = axis_local  # Linear velocity
            
            # Transform to child frame
            S[node_idx] = (spatial_adjoint(T_link_offset) @ twist_joint.unsqueeze(-1)).squeeze(-1)
            v_joint = v_joints[:, joint_idx].unsqueeze(-1)
            a_joint = a_joints[:, joint_idx].unsqueeze(-1)
        else:
            v_joint = torch.zeros((batch_size, 1), dtype=dtype, device=device)
            a_joint = torch.zeros((batch_size, 1), dtype=dtype, device=device)

        # Spatial inertia
        frame_name = chain.idx_to_frame[node_idx]
        frame_obj = chain.find_frame(frame_name)
        I_spatial[node_idx] = compute_spatial_inertia(
            frame_obj.link, batch_size, dtype, device
        )

        # Get parent state
        if parent[node_idx] == -1:
            # Root node - use base velocity/acceleration if floating
            v_parent = v_base if (has_floating_base and v_base is not None) else torch.zeros((batch_size, 6), dtype=dtype, device=device)
            a_parent = (a_base if (has_floating_base and a_base is not None) else torch.zeros((batch_size, 6), dtype=dtype, device=device)) + a_gravity_base
        else:
            v_parent = v[parent[node_idx]]
            a_parent = a[parent[node_idx]]

        # Propagate velocity and acceleration
        vJ = S[node_idx] * v_joint
        v[node_idx] = (Xup[node_idx] @ v_parent.unsqueeze(-1)).squeeze(-1) + vJ
        
        # Acceleration with Coriolis term
        coriolis_term = (motion_cross_product(v[node_idx]) @ vJ.unsqueeze(-1)).squeeze(-1)
        a[node_idx] = (Xup[node_idx] @ a_parent.unsqueeze(-1)).squeeze(-1) + S[node_idx] * a_joint + coriolis_term

    # ========================================================================
    # Backward pass: compute forces
    # ========================================================================
    
    for node_idx in reversed(topo_order):
        # Newton-Euler equation: f = I*a + v×(I*v)
        Iv = (I_spatial[node_idx] @ v[node_idx].unsqueeze(-1)).squeeze(-1)
        f[node_idx] = (I_spatial[node_idx] @ a[node_idx].unsqueeze(-1)).squeeze(-1)
        f[node_idx] = f[node_idx] + (force_cross_product(v[node_idx]) @ Iv.unsqueeze(-1)).squeeze(-1)

        # Add forces from children
        for child_idx in children[node_idx]:
            f[node_idx] = f[node_idx] + (Xup[child_idx].transpose(1, 2) @ f[child_idx].unsqueeze(-1)).squeeze(-1)

    # ========================================================================
    # Extract generalized forces
    # ========================================================================
    
    if has_floating_base:
        tau = torch.zeros((batch_size, 6 + n_joints), dtype=dtype, device=device)
        
        # Base wrench from URDF root body (node 1, not synthetic wrapper)
        urdf_root_idx = 1 if n_nodes > 1 else 0
        tau[:, :6] = f[urdf_root_idx]
        
        # Joint torques/forces
        for node_idx in range(n_nodes):
            joint_type_idx = int(chain.joint_type_indices[node_idx])
            if joint_type_idx in [Joint.TYPES.index('revolute'), Joint.TYPES.index('prismatic')]:
                joint_col = int(chain.joint_indices[node_idx])
                tau[:, 6 + joint_col] = torch.sum(S[node_idx] * f[node_idx], dim=-1)
    else:
        tau = torch.zeros((batch_size, n_joints), dtype=dtype, device=device)
        
        for node_idx in range(n_nodes):
            joint_type_idx = int(chain.joint_type_indices[node_idx])
            if joint_type_idx in [Joint.TYPES.index('revolute'), Joint.TYPES.index('prismatic')]:
                joint_col = int(chain.joint_indices[node_idx])
                tau[:, joint_col] = torch.sum(S[node_idx] * f[node_idx], dim=-1)

    return tau


@torch.compile
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
    parent, children = build_parent_children(chain)
    n_nodes = len(parent)
    base_nodes = [i for i, p in enumerate(parent) if p == -1]
    
    topo_order = []
    stack = base_nodes[:]
    while stack:
        node_idx = stack.pop(0)
        topo_order.append(node_idx)
        stack.extend(children[node_idx])

    # Precompute transforms
    axes_raw = chain.axes.to(dtype=dtype, device=device)
    axes = axes_raw / axes_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axes_batched = axes.unsqueeze(0).expand(batch_size, -1, -1)
    T_revolute = axis_and_angle_to_matrix_44(axes_batched, q_joints)
    T_prismatic = axis_and_d_to_pris_matrix(axes_batched, q_joints)

    I44 = identity_transform(batch_size, dtype, device)
    Xup = [None] * n_nodes
    S = [torch.zeros((batch_size, 6), dtype=dtype, device=device) for _ in range(n_nodes)]
    I_spatial = [None] * n_nodes

    # ========================================================================
    # Forward pass: compute transforms and joint subspaces
    # ========================================================================
    
    for node_idx in topo_order:
        joint_idx = int(chain.joint_indices[node_idx])
        joint_type_idx = int(chain.joint_type_indices[node_idx])

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
            axis_local = axes[joint_idx].view(1, 3).expand(batch_size, -1)
            twist_joint = torch.zeros((batch_size, 6), dtype=dtype, device=device)
            
            if joint_type_idx == Joint.TYPES.index('revolute'):
                twist_joint[:, 3:] = axis_local
            else:
                twist_joint[:, :3] = axis_local
                
            S[node_idx] = (spatial_adjoint(T_link_offset) @ twist_joint.unsqueeze(-1)).squeeze(-1)

        # Spatial inertia
        frame_name = chain.idx_to_frame[node_idx]
        frame_obj = chain.find_frame(frame_name)
        I_spatial[node_idx] = compute_spatial_inertia(
            frame_obj.link, batch_size, dtype, device
        )

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
        joint_type_idx = int(chain.joint_type_indices[node_idx])
        if joint_type_idx != Joint.TYPES.index('fixed'):
            joint_col = int(chain.joint_indices[node_idx])
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