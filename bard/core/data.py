"""
Mutable computation workspace for robot dynamics algorithms.

This module defines the ``Data`` class, which holds all pre-allocated tensor
buffers used by kinematics and dynamics algorithms. It replaces the previous
``KinematicsState`` dataclass by also owning the algorithm-specific buffers
(accelerations, forces, mass matrix, Jacobian), enabling a clean separation
between the immutable robot model and the mutable computation state.
"""

from typing import Optional, Union
import torch


class Data:
    """Pre-allocated workspace for batched kinematics and dynamics computations.

    A ``Data`` object is created from a :class:`~bard.core.model.Model` via
    :func:`bard.create_data` and holds all mutable tensor buffers. Multiple
    ``Data`` instances can coexist for the same ``Model`` (e.g., for parallel
    simulation or planning vs. control).

    Attributes:
        max_batch_size: Maximum supported batch size.
        batch_size: Actual batch size of the last ``update_kinematics`` call.
        has_velocity: Whether velocity-level quantities were computed.
    """

    def __init__(
        self,
        n_nodes: int,
        nv: int,
        max_batch_size: int = 1024,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        self.max_batch_size = max_batch_size
        self.batch_size: int = 0
        self.has_velocity: bool = False

        B, N = max_batch_size, n_nodes

        # Kinematics cache buffers
        self.T_pc = torch.zeros(B, N, 4, 4, dtype=dtype, device=device)
        self.Xup = torch.zeros(B, N, 6, 6, dtype=dtype, device=device)
        self.S = torch.zeros(B, N, 6, 1, dtype=dtype, device=device)
        self.T_world = torch.zeros(B, N, 4, 4, dtype=dtype, device=device)
        self.v = torch.zeros(B, N, 6, 1, dtype=dtype, device=device)

        # Algorithm-specific buffers
        self.a = torch.zeros(B, N, 6, 1, dtype=dtype, device=device)
        self.f = torch.zeros(B, N, 6, 1, dtype=dtype, device=device)
        self.I_composite = torch.zeros(B, N, 6, 6, dtype=dtype, device=device)
        self.M = torch.zeros(B, nv, nv, dtype=dtype, device=device)
        self.J_local = torch.zeros(B, 6, nv, dtype=dtype, device=device)

        # Cached joint velocity (filled by update_kinematics, reused by RNEA/ABA)
        self.vJ = torch.zeros(B, N, 6, 1, dtype=dtype, device=device)

        # ABA-specific buffers
        self.IA = torch.zeros(B, N, 6, 6, dtype=dtype, device=device)
        self.pA = torch.zeros(B, N, 6, 1, dtype=dtype, device=device)
        self.U = torch.zeros(B, N, 6, 1, dtype=dtype, device=device)
        self.d = torch.zeros(B, N, dtype=dtype, device=device)
        self.u = torch.zeros(B, N, dtype=dtype, device=device)

        # Pre-allocated output buffers
        self.qdd_out = torch.zeros(B, nv, dtype=dtype, device=device)
        self.tau_out = torch.zeros(B, nv, dtype=dtype, device=device)
