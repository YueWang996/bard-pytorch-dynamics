"""
Top-level free functions for the bard v0.3 unified API.

All computation goes through ``bard.*`` module-level functions that accept
a :class:`~bard.core.model.Model` and a :class:`~bard.core.data.Data`.
"""

from typing import Optional, Tuple, Union
import torch

from bard.core.model import Model
from bard.core.data import Data


def create_data(model: Model, max_batch_size: int = 1024) -> Data:
    """Creates a new computation workspace for a model.

    Args:
        model: The robot model.
        max_batch_size: Maximum supported batch size for pre-allocated buffers.

    Returns:
        A new :class:`Data` instance.
    """
    return model.create_data(max_batch_size)


def update_kinematics(
    model: Model,
    data: Data,
    q: torch.Tensor,
    qd: Optional[torch.Tensor] = None,
) -> Data:
    """Compute and cache all shared kinematic quantities in a single tree traversal.

    Call this once per control step, then use the cached ``data`` for all
    subsequent algorithm calls.

    If ``qd`` is provided, velocity-level quantities (spatial velocities) are
    also computed. If ``qd`` is ``None``, only position-level quantities are
    computed (sufficient for FK, Jacobian, CRBA).

    Args:
        model: The robot model.
        data: The computation workspace (mutated in-place).
        q: Generalized positions ``(B, nq)``.
        qd: Generalized velocities ``(B, nv)``, or ``None``.

    Returns:
        The same ``data`` object (for convenience chaining).

    Raises:
        ValueError: If batch size exceeds ``data.max_batch_size``.
    """
    batch_size = q.shape[0]
    if batch_size > data.max_batch_size:
        raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {data.max_batch_size}.")
    model._update_kinematics_fn(data, q, qd)
    return data


def forward_kinematics(
    model: Model,
    data: Data,
    frame_id: int,
    *,
    q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """World-frame pose of a frame.

    **Cached mode** (default): O(1) lookup from a prior ``update_kinematics`` call.

    **Standalone mode**: Pass ``q`` to perform a path-only traversal without
    needing ``update_kinematics``. This is efficient for single-frame queries.

    Args:
        model: The robot model.
        data: The computation workspace.
        frame_id: Target frame index (from ``model.get_frame_id``).
        q: If provided, performs standalone FK (path-only traversal).

    Returns:
        Homogeneous transform ``(B, 4, 4)``.
    """
    if q is not None:
        batch_size = q.shape[0]
        if batch_size > data.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {data.max_batch_size}."
            )
        return model._fk_fn(data, q, frame_id)
    return data.T_world[: data.batch_size, frame_id]


def jacobian(
    model: Model,
    data: Data,
    frame_id: int,
    *,
    reference_frame: str = "world",
    return_pose: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Geometric Jacobian using cached state.

    Requires a prior ``update_kinematics`` call.

    Args:
        model: The robot model.
        data: The computation workspace.
        frame_id: Target frame index.
        reference_frame: ``"world"`` or ``"local"``.
        return_pose: If True, also returns the world-frame pose.

    Returns:
        Jacobian ``(B, 6, nv)``, optionally with pose ``(B, 4, 4)``.
    """
    if reference_frame not in ("world", "local"):
        raise ValueError('reference_frame must be "world" or "local"')
    return model._jacobian_fn(data, frame_id, reference_frame, return_pose)


def rnea(
    model: Model,
    data: Data,
    qdd: torch.Tensor,
    *,
    gravity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse dynamics (Recursive Newton-Euler Algorithm) using cached state.

    Requires ``update_kinematics`` with ``qd`` provided.

    Args:
        model: The robot model.
        data: The computation workspace (must include velocities).
        qdd: Generalized accelerations ``(B, nv)``.
        gravity: 3-element gravity vector. Defaults to ``[0, 0, -9.81]``.

    Returns:
        Generalized forces ``(B, nv)``.
    """
    if not data.has_velocity:
        raise ValueError(
            "RNEA requires velocity data. "
            "Call update_kinematics(model, data, q, qd) with qd provided."
        )
    return model._rnea_fn(data, qdd, gravity)


def crba(model: Model, data: Data) -> torch.Tensor:
    """Mass matrix (Composite Rigid Body Algorithm) using cached state.

    Requires a prior ``update_kinematics`` call.

    Args:
        model: The robot model.
        data: The computation workspace.

    Returns:
        Mass matrix ``(B, nv, nv)``.
    """
    return model._crba_fn(data)


def spatial_acceleration(
    model: Model,
    data: Data,
    qdd: torch.Tensor,
    frame_id: int,
    *,
    reference_frame: str = "world",
) -> torch.Tensor:
    """Spatial acceleration using cached state.

    Requires ``update_kinematics`` with ``qd`` provided.

    Args:
        model: The robot model.
        data: The computation workspace (must include velocities).
        qdd: Generalized accelerations ``(B, nv)``.
        frame_id: Target frame index.
        reference_frame: ``"world"`` or ``"local"``.

    Returns:
        Spatial acceleration ``(B, 6)`` as ``[linear; angular]``.
    """
    if not data.has_velocity:
        raise ValueError(
            "Spatial acceleration requires velocity data. "
            "Call update_kinematics(model, data, q, qd) with qd provided."
        )
    if reference_frame not in ("world", "local"):
        raise ValueError('reference_frame must be "world" or "local"')
    return model._spatial_acceleration_fn(data, qdd, frame_id, reference_frame)
