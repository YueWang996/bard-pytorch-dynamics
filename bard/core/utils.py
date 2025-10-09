"""
Utility functions for spatial algebra and robot dynamics computations.

This module provides low-level utilities for:
- Spatial vector operations (adjoints, cross products)
- Homogeneous transformation utilities
- Spatial inertia computations
- Tree structure helpers

Many functions are JIT-compiled for performance.
"""

from typing import List, Optional, Tuple, Union
import torch

TensorLike = Union[torch.Tensor, List[float]]


# ============================================================================
# JIT-compiled optimized spatial algebra utilities
# ============================================================================


@torch.jit.script
def quat_to_rotmat_fast(quat: torch.Tensor) -> torch.Tensor:
    """Optimized quaternion to rotation matrix conversion.

    Args:
        quat: Quaternion (B, 4) as [qw, qx, qy, qz]

    Returns:
        Rotation matrix (B, 3, 3)
    """
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    qw, qx, qy, qz = quat.unbind(-1)

    # Precompute all products once
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    # Build matrix efficiently
    R = torch.stack(
        [
            torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )

    return R


@torch.jit.script
def spatial_adjoint_fast(T: torch.Tensor) -> torch.Tensor:
    """Optimized spatial adjoint computation without intermediate allocations.

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

    # Compute skew(p) @ R directly without creating skew matrix
    px, py, pz = p.unbind(-1)
    pxR = torch.stack(
        [
            -pz.unsqueeze(-1) * R[:, 1] + py.unsqueeze(-1) * R[:, 2],
            pz.unsqueeze(-1) * R[:, 0] - px.unsqueeze(-1) * R[:, 2],
            -py.unsqueeze(-1) * R[:, 0] + px.unsqueeze(-1) * R[:, 1],
        ],
        dim=1,
    )

    # Build result efficiently
    batch = T.shape[0]
    Ad = torch.zeros((batch, 6, 6), dtype=T.dtype, device=T.device)
    Ad[:, :3, :3] = R
    Ad[:, :3, 3:] = pxR
    Ad[:, 3:, 3:] = R
    return Ad


@torch.jit.script
def inv_homogeneous_fast(T: torch.Tensor) -> torch.Tensor:
    """Optimized inverse of homogeneous transformation matrix.

    For T = [R, p; 0, 1], returns T^{-1} = [R^T, -R^T p; 0, 1]

    Args:
        T: Homogeneous transformation (B, 4, 4)

    Returns:
        Inverse transformation (B, 4, 4)
    """
    R = T[:, :3, :3]
    Rt = R.transpose(1, 2)
    p = T[:, :3, 3]

    # Compute -R^T @ p efficiently
    p_inv = -(Rt @ p.unsqueeze(-1)).squeeze(-1)

    # Build result without intermediate allocation
    batch = T.shape[0]
    T_inv = torch.zeros_like(T)
    T_inv[:, :3, :3] = Rt
    T_inv[:, :3, 3] = p_inv
    T_inv[:, 3, 3] = 1.0

    return T_inv


@torch.jit.script
def motion_cross_product_fast(twist: torch.Tensor) -> torch.Tensor:
    """Optimized motion cross product ad_{twist} for spatial velocities.

    For twist = [v; ω] where v is linear and ω is angular velocity,
    returns the 6×6 matrix representing the Lie bracket operation.

    Args:
        twist: Spatial velocity (B, 6, 1) as [v; ω]

    Returns:
        Cross product matrix (B, 6, 6)
    """
    v = twist[:, :3, 0]
    w = twist[:, 3:, 0]
    batch = twist.shape[0]

    # Compute skew matrices directly
    wx, wy, wz = w.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    zeros = torch.zeros_like(wx)

    w_skew = torch.stack(
        [
            torch.stack([zeros, -wz, wy], dim=-1),
            torch.stack([wz, zeros, -wx], dim=-1),
            torch.stack([-wy, wx, zeros], dim=-1),
        ],
        dim=-2,
    )

    v_skew = torch.stack(
        [
            torch.stack([zeros, -vz, vy], dim=-1),
            torch.stack([vz, zeros, -vx], dim=-1),
            torch.stack([-vy, vx, zeros], dim=-1),
        ],
        dim=-2,
    )

    result = torch.zeros((batch, 6, 6), dtype=twist.dtype, device=twist.device)
    result[:, :3, :3] = w_skew
    result[:, :3, 3:] = v_skew
    result[:, 3:, 3:] = w_skew

    return result


@torch.jit.script
def force_cross_product_fast(twist: torch.Tensor) -> torch.Tensor:
    """Optimized force cross product (dual of motion cross product).

    This is the adjoint operator for spatial forces: ad*_{twist} = -ad_{twist}^T

    Args:
        twist: Spatial velocity (B, 6, 1)

    Returns:
        Force cross product matrix (B, 6, 6)
    """
    return -motion_cross_product_fast(twist).transpose(1, 2)


@torch.jit.script
def skew_symmetric_fast(v: torch.Tensor) -> torch.Tensor:
    """Optimized skew-symmetric matrix computation.

    For vector v = [x, y, z], returns:
        [[ 0, -z,  y],
         [ z,  0, -x],
         [-y,  x,  0]]

    Args:
        v: Vector of shape (B, 3)

    Returns:
        Skew-symmetric matrix of shape (B, 3, 3)
    """
    x, y, z = v.unbind(-1)
    zeros = torch.zeros_like(x)

    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


# ============================================================================
# Original (non-JIT) spatial algebra utilities for backward compatibility
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
    # Add batch dimension if needed
    if v.ndim == 1:
        v = v.unsqueeze(0)
        result = skew_symmetric_fast(v)
        return result.squeeze(0)
    return skew_symmetric_fast(v)


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
    return spatial_adjoint_fast(T)


def inv_homogeneous(T: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse of homogeneous transformation matrix.

    For T = [R, p; 0, 1], returns T^{-1} = [R^T, -R^T p; 0, 1]

    Args:
        T: Homogeneous transformation (B, 4, 4)

    Returns:
        Inverse transformation (B, 4, 4)
    """
    return inv_homogeneous_fast(T)


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
    # Ensure correct shape for fast version
    if twist.ndim == 2 and twist.shape[-1] == 6:
        twist = twist.unsqueeze(-1)
    return motion_cross_product_fast(twist)


def force_cross_product(twist: torch.Tensor) -> torch.Tensor:
    """
    Compute force cross product (dual of motion cross product).

    This is the adjoint operator for spatial forces: ad*_{twist} = -ad_{twist}^T

    Args:
        twist: Spatial velocity (B, 6)

    Returns:
        Force cross product matrix (B, 6, 6)
    """
    # Ensure correct shape for fast version
    if twist.ndim == 2 and twist.shape[-1] == 6:
        twist = twist.unsqueeze(-1)
    return force_cross_product_fast(twist)


def identity_transform(batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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


# ============================================================================
# Quaternion utilities
# ============================================================================


def quaternion_to_rotation_matrix(q: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion (B, 4) as [qw, qx, qy, qz]
        normalize: Whether to normalize quaternion first

    Returns:
        Rotation matrix (B, 3, 3)
    """
    if not normalize:
        return quat_to_rotmat_fast(q)

    # Fast version already normalizes
    return quat_to_rotmat_fast(q)


def base_pose_to_transform(q_base: torch.Tensor, normalize_quat: bool = True) -> torch.Tensor:
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

    R = quat_to_rotmat_fast(quat) if normalize_quat else quat_to_rotmat_fast(quat)

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
    chain=None,
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

    inertial = getattr(link, "inertial", None)
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
            I_rotational = (
                inertia_tensor.to(dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)
            )
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
    T: Optional[torch.Tensor], batch: int, dtype: torch.dtype, device: torch.device
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
    if transform is None or isinstance(transform, torch.Tensor):
        return transform
    if hasattr(transform, "get_matrix"):
        return transform.get_matrix()
    return transform


def normalize_axis(axis: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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


def normalize_joint_positions(chain, q: TensorLike) -> torch.Tensor:
    """
    Normalize joint positions to batched tensor form.

    Args:
        chain: Robot chain
        q: Joint positions as tensor, list, array, or dict

    Returns:
        Batched tensor (B, n_joints)
    """
    if isinstance(q, torch.Tensor):
        return torch.atleast_2d(q.to(dtype=chain.dtype, device=chain.device))

    if hasattr(chain, "ensure_tensor"):
        q_tensor = chain.ensure_tensor(q)
    else:
        q_tensor = torch.as_tensor(q, dtype=chain.dtype, device=chain.device)
    return torch.atleast_2d(q_tensor)


# ============================================================================
# Validation utilities
# ============================================================================


def validate_configuration_size(
    chain, q: torch.Tensor, expected_size: Optional[int] = None
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
            msg = (
                f"Expected configuration size {expected_size} "
                f"(7 base + {chain.n_joints} joints), got {q.shape[-1]}"
            )
        else:
            msg = (
                f"Expected configuration size {expected_size} "
                f"({chain.n_joints} joints), got {q.shape[-1]}"
            )
        raise ValueError(msg)


def validate_frame_id(chain, frame_id: Union[str, int]) -> int:
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
