"""
Kinematics state cache for eliminating redundant computations.

This module defines the ``KinematicsState`` dataclass that stores shared
kinematic quantities computed once per (q, qd) pair and reused by all
downstream dynamics/kinematics algorithms.
"""

from dataclasses import dataclass
import torch


@dataclass
class KinematicsState:
    """Cached kinematics quantities computed by ``RobotDynamics.update_kinematics``.

    All tensors have a leading batch dimension ``B``. Position-level quantities
    (``T_parent_to_child``, ``Xup``, ``S``, ``T_world``) are always populated.
    Velocity-level quantities (``v``) are only populated when ``qd`` is provided
    to ``update_kinematics``.

    Attributes:
        T_parent_to_child: Per-node parent-to-child transforms ``(B, n_nodes, 4, 4)``.
        Xup: Spatial adjoint of inverse parent-to-child transform ``(B, n_nodes, 6, 6)``.
        S: Joint subspace vectors ``(B, n_nodes, 6, 1)``.
        T_world: World-frame pose of each node ``(B, n_nodes, 4, 4)``.
        v: Spatial velocities ``(B, n_nodes, 6, 1)``. Zero if ``has_velocity`` is False.
        batch_size: The actual batch size (may be less than buffer size).
        has_velocity: Whether velocity-level quantities were computed.
    """

    T_parent_to_child: torch.Tensor
    Xup: torch.Tensor
    S: torch.Tensor
    T_world: torch.Tensor
    v: torch.Tensor
    batch_size: int
    has_velocity: bool
