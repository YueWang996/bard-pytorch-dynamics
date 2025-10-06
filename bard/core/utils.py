"""
Utility functions for spatial algebra and robot dynamics computations.

This module provides low-level utilities for:
- Spatial vector operations (adjoints, cross products)
- Homogeneous transformation utilities
- Spatial inertia computations
- Tree structure helpers
"""

from typing import List, Optional, Tuple, Union
import torch

TensorLike = Union[torch.Tensor, List[float]]


# ============================================================================
# Spatial algebra utilities
# ============================================================================

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Compute skew-symmetric matrix from 3D vector.
    
    For vector v = [x, y, z], returns:
        [[ 0, -z,  y],
         [ z,  0, -x],
         [-y,  x,  0]]
    
    Args:
        v: Vector of shape (..., 3)
        
    Returns:
        Skew-symmetric matrix of shape (..., 3, 3)
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = torch.zeros_like(x)
    return torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1),
    ], dim=-2)


def spatial_adjoint(T: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial adjoint matrix from homogeneous transform.
    
    The adjoint maps spatial velocities between coordinate frames.
    For twist convention [v; ω], the adjoint is:
        [[R,    [p]×R],
         [0,    R    ]]
    
    Args:
        T: Homogeneous transformation matrix (B, 4, 4)
        
    Returns:
        Adjoint matrix (B, 6, 6)
    """
    R = T[:, :3, :3]
    p = T[:, :3, 3]
    batch = T.shape[0]
    
    Ad = torch.empty((batch, 6, 6), dtype=T.dtype, device=T.device)
    Ad[:, :3, :3] = R
    Ad[:, 3:, :3] = 0
    Ad[:, :3, 3:] = skew_symmetric(p) @ R
    Ad[:, 3:, 3:] = R
    return Ad


def inv_homogeneous(T: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse of homogeneous transformation matrix.
    
    For T = [R, p; 0, 1], returns T^{-1} = [R^T, -R^T p; 0, 1]
    
    Args:
        T: Homogeneous transformation (B, 4, 4)
        
    Returns:
        Inverse transformation (B, 4, 4)
    """
    R = T[:, :3, :3]
    Rt = R.transpose(1, 2)
    p = T[:, :3, 3]
    p_inv = -(Rt @ p.unsqueeze(-1)).squeeze(-1)
    
    T_inv = torch.empty_like(T)
    T_inv[:, :3, :3] = Rt
    T_inv[:, :3, 3] = p_inv
    T_inv[:, 3, :3] = 0
    T_inv[:, 3, 3] = 1
    return T_inv


def identity_transform(
    batch: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Create batch of identity transformation matrices.
    
    Args:
        batch: Batch size
        dtype: Data type
        device: Device
        
    Returns:
        Identity matrices (B, 4, 4)
    """
    return torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)


def motion_cross_product(twist: torch.Tensor) -> torch.Tensor:
    """
    Compute motion cross product ad_{twist} for spatial velocities.
    
    For twist = [v; ω] where v is linear and ω is angular velocity,
    returns the 6×6 matrix representing the Lie bracket operation.
    
    Args:
        twist: Spatial velocity (B, 6) as [v; ω]
        
    Returns:
        Cross product matrix (B, 6, 6)
    """
    v = twist[..., :3]
    w = twist[..., 3:]
    batch = twist.shape[0]
    
    zeros = torch.zeros((batch, 3, 3), dtype=twist.dtype, device=twist.device)
    w_skew = skew_symmetric(w)
    v_skew = skew_symmetric(v)
    
    top = torch.cat([w_skew, v_skew], dim=-1)
    bottom = torch.cat([zeros, w_skew], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def force_cross_product(twist: torch.Tensor) -> torch.Tensor:
    """
    Compute force cross product (dual of motion cross product).
    
    This is the adjoint operator for spatial forces: ad*_{twist} = -ad_{twist}^T
    
    Args:
        twist: Spatial velocity (B, 6)
        
    Returns:
        Force cross product matrix (B, 6, 6)
    """
    return -motion_cross_product(twist).transpose(1, 2)


# ============================================================================
# Quaternion utilities
# ============================================================================

def quaternion_to_rotation_matrix(
    q: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion (B, 4) as [qw, qx, qy, qz]
        normalize: Whether to normalize quaternion first
        
    Returns:
        Rotation matrix (B, 3, 3)
    """
    if normalize:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    
    qw, qx, qy, qz = q.unbind(-1)
    dtype, device = q.dtype, q.device
    
    two = torch.tensor(2.0, dtype=dtype, device=device)
    x2, y2, z2 = two * qx * qx, two * qy * qy, two * qz * qz
    xy, xz, yz = two * qx * qy, two * qx * qz, two * qy * qz
    wx, wy, wz = two * qw * qx, two * qw * qy, two * qw * qz
    
    batch = q.shape[0]
    R = torch.empty(batch, 3, 3, dtype=dtype, device=device)
    R[:, 0, 0] = 1.0 - (y2 + z2)
    R[:, 0, 1] = xy - wz
    R[:, 0, 2] = xz + wy
    R[:, 1, 0] = xy + wz
    R[:, 1, 1] = 1.0 - (x2 + z2)
    R[:, 1, 2] = yz - wx
    R[:, 2, 0] = xz - wy
    R[:, 2, 1] = yz + wx
    R[:, 2, 2] = 1.0 - (x2 + y2)
    
    return R


def base_pose_to_transform(
    q_base: torch.Tensor,
    normalize_quat: bool = True
) -> torch.Tensor:
    """
    Convert base pose to homogeneous transformation matrix.
    
    Args:
        q_base: Base pose (B, 7) as [tx, ty, tz, qw, qx, qy, qz]
        normalize_quat: Whether to normalize quaternion
        
    Returns:
        Transformation matrix (B, 4, 4)
    """
    batch = q_base.shape[0]
    dtype, device = q_base.dtype, q_base.device
    
    t = q_base[:, :3]
    quat = q_base[:, 3:]
    
    R = quaternion_to_rotation_matrix(quat, normalize=normalize_quat)
    
    T = torch.zeros(batch, 4, 4, dtype=dtype, device=device)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    
    return T


# ============================================================================
# Spatial inertia
# ============================================================================

def compute_spatial_inertia(
    link,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    node_idx: Optional[int] = None,
    chain = None,
) -> torch.Tensor:
    """
    Compute 6×6 spatial inertia matrix from link properties.
    
    The spatial inertia for twist convention [v; ω] is:
        [[m*I,     -m*[c]×],
         [m*[c]×,  I_c - m*[c]×[c]×]]
    
    where m is mass, c is COM position, and I_c is rotational inertia about COM.
    
    Args:
        link: Link object with inertial property
        batch: Batch size
        dtype: Data type
        device: Device
        
    Returns:
        Spatial inertia matrix (B, 6, 6)
    """
    if chain is not None and node_idx is not None:
        # Use pre-computed spatial inertia from chain
        I_base = chain.spatial_inertias[node_idx]
        return I_base.unsqueeze(0).expand(batch, -1, -1)
    
    I_spatial = torch.zeros((batch, 6, 6), dtype=dtype, device=device)
    
    inertial = getattr(link, 'inertial', None)
    if inertial is None:
        return I_spatial

    offset_transform, mass, inertia_tensor = inertial

    # Extract COM pose
    if offset_transform is None:
        R = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)
        com_pos = torch.zeros((batch, 3), dtype=dtype, device=device)
    else:
        T = as_batched_transform(offset_transform.get_matrix(), batch, dtype, device)
        R = T[:, :3, :3]
        com_pos = T[:, :3, 3]

    # Mass scalar
    if not torch.is_tensor(mass):
        mass = torch.tensor(mass, dtype=dtype, device=device)
    m = mass.view(1).to(dtype=dtype, device=device).expand(batch)

    # Rotational inertia
    if inertia_tensor is None:
        I_rotational = torch.zeros((batch, 3, 3), dtype=dtype, device=device)
    else:
        if inertia_tensor.ndim == 2:
            I_rotational = inertia_tensor.to(dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)
        else:
            I_rotational = inertia_tensor.to(dtype=dtype, device=device)
        # Rotate to link frame
        I_rotational = R @ I_rotational @ R.transpose(1, 2)

    # Build spatial inertia matrix
    I3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)
    com_skew = skew_symmetric(com_pos)
    m_com_skew = m.view(batch, 1, 1) * com_skew
    
    # Upper-left: m*I
    I_spatial[:, :3, :3] = m.view(batch, 1, 1) * I3
    # Upper-right: -m*[c]×
    I_spatial[:, :3, 3:] = -m_com_skew
    # Lower-left: m*[c]×
    I_spatial[:, 3:, :3] = m_com_skew
    # Lower-right: I_c - m*[c]×[c]×
    I_spatial[:, 3:, 3:] = I_rotational - (m.view(batch, 1, 1) * (com_skew @ com_skew))
    
    return I_spatial


# ============================================================================
# Transform utilities
# ============================================================================

def as_batched_transform(
    T: Optional[torch.Tensor],
    batch: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Convert transform to batched form.
    
    Args:
        T: Transformation matrix (4, 4) or (B, 4, 4) or None
        batch: Target batch size
        dtype: Data type
        device: Device
        
    Returns:
        Batched transformation (B, 4, 4)
    """
    if T is None:
        return identity_transform(batch, dtype, device)
    if T.ndim == 2:
        return T.to(dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)
    return T.to(dtype=dtype, device=device)


def to_matrix44(transform) -> Optional[torch.Tensor]:
    """
    Extract 4×4 matrix from Transform3d or return tensor as-is.
    
    Args:
        transform: Transform3d object or tensor or None
        
    Returns:
        4×4 matrix or None
    """
    if transform is None:
        return None
    if hasattr(transform, "get_matrix"):
        return transform.get_matrix()
    return transform


def normalize_axis(
    axis: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Normalize axis vector to unit length.
    
    Args:
        axis: Axis vector (..., 3)
        eps: Small value to prevent division by zero
        
    Returns:
        Unit axis vector
    """
    if axis.ndim == 1:
        axis = axis.unsqueeze(0)
    norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    return axis / norm


def reproject_rotation(R: torch.Tensor) -> torch.Tensor:
    """
    Project matrix onto SO(3) using SVD.
    
    Ensures rotation matrix remains orthonormal despite numerical errors.
    
    Args:
        R: Rotation matrix (B, 3, 3)
        
    Returns:
        Projected rotation matrix (B, 3, 3)
    """
    batch = R.shape[0]
    dtype, device = R.dtype, R.device
    
    U, _, Vh = torch.linalg.svd(R)
    M = U @ Vh
    det = torch.det(M)
    
    # Ensure right-handed coordinate system (det = +1)
    D = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1).clone()
    D[:, 2, 2] = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    
    return U @ D @ Vh


# ============================================================================
# Tree structure utilities
# ============================================================================

def build_parent_children(chain) -> Tuple[List[int], List[List[int]]]:
    """
    Extract parent-child relationships from chain structure.
    
    Args:
        chain: Robot chain
        
    Returns:
        (parent_list, children_list) where:
            - parent_list[i] is the parent index of node i (-1 for root)
            - children_list[i] is list of child indices of node i
    """
    parent = chain.parent_array.cpu().tolist()
    
    children = []
    for i in range(len(parent)):
        count = int(chain.children_count[i].item())
        child_list = chain.children_array[i, :count].cpu().tolist()
        children.append(child_list)
            
    return parent, children


def normalize_joint_positions(
    chain,
    q: TensorLike
) -> torch.Tensor:
    """
    Normalize joint positions to batched tensor form.
    
    Args:
        chain: Robot chain
        q: Joint positions as tensor, list, array, or dict
        
    Returns:
        Batched tensor (B, n_joints)
    """
    if hasattr(chain, "ensure_tensor"):
        q_tensor = chain.ensure_tensor(q)
    else:
        q_tensor = torch.as_tensor(q, dtype=chain.dtype, device=chain.device)
    return torch.atleast_2d(q_tensor)


# ============================================================================
# Validation utilities
# ============================================================================

def validate_configuration_size(
    chain,
    q: torch.Tensor,
    expected_size: Optional[int] = None
) -> None:
    """
    Validate that configuration has correct size.
    
    Args:
        chain: Robot chain
        q: Configuration tensor
        expected_size: Expected size (defaults to chain.nq)
        
    Raises:
        ValueError: If size is incorrect
    """
    if expected_size is None:
        expected_size = chain.nq
    
    if q.shape[-1] != expected_size:
        has_fb = getattr(chain, "has_floating_base", False)
        if has_fb:
            msg = (f"Expected configuration size {expected_size} "
                   f"(7 base + {chain.n_joints} joints), got {q.shape[-1]}")
        else:
            msg = (f"Expected configuration size {expected_size} "
                   f"({chain.n_joints} joints), got {q.shape[-1]}")
        raise ValueError(msg)


def validate_frame_id(
    chain,
    frame_id: Union[str, int]
) -> int:
    """
    Validate and convert frame identifier to index.
    
    Args:
        chain: Robot chain
        frame_id: Frame name or index
        
    Returns:
        Frame index
        
    Raises:
        ValueError: If frame not found
    """
    if isinstance(frame_id, str):
        if frame_id not in chain.frame_to_idx:
            raise ValueError(f"Frame '{frame_id}' not found in chain")
        return chain.frame_to_idx[frame_id]
    
    idx = int(frame_id)
    if idx < 0 or idx >= len(chain.idx_to_frame):
        raise ValueError(f"Frame index {idx} out of range [0, {len(chain.idx_to_frame)})")
    return idx