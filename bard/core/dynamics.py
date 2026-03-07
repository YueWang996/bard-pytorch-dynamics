"""
Dynamics algorithm classes (deprecated, use ``RobotDynamics`` instead).

This module provides backward-compatible wrappers around ``RobotDynamics``.
New code should use ``RobotDynamics`` directly for better performance.
"""

import warnings
from typing import Any, Dict, Optional
import torch

from bard.core.chain import Chain
from bard.core.robot_dynamics import RobotDynamics


class RNEA:
    """Recursive Newton-Euler Algorithm (deprecated).

    .. deprecated::
        Use :class:`~bard.core.robot_dynamics.RobotDynamics` instead.
        This class is a thin wrapper that delegates to
        ``RobotDynamics.update_kinematics()`` + ``RobotDynamics.rnea()``.

    Args:
        chain: The robot's kinematic chain.
        max_batch_size: Maximum supported batch size.
        compile_enabled: If True, enable ``torch.compile``.
        compile_kwargs: Additional kwargs for ``torch.compile``.
    """

    def __init__(
        self,
        chain: Chain,
        max_batch_size: int = 1024,
        compile_enabled: bool = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        warnings.warn(
            "RNEA is deprecated, use RobotDynamics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._rd = RobotDynamics(
            chain,
            max_batch_size=max_batch_size,
            compile_enabled=compile_enabled,
            compile_kwargs=compile_kwargs,
        )
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device

    def enable_compilation(self, enabled: bool = True, **compile_kwargs):
        self._rd.enable_compilation(enabled, **compile_kwargs)

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        self._rd.to(dtype=dtype, device=device)
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        return self

    def calc(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        gravity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute generalized forces via RNEA.

        Args:
            q: Generalized positions ``(B, nq)``.
            qd: Generalized velocities ``(B, nv)``.
            qdd: Generalized accelerations ``(B, nv)``.
            gravity: 3-element gravity vector. Defaults to ``[0, 0, -9.81]``.

        Returns:
            Generalized forces ``(B, nv)``.
        """
        state = self._rd.update_kinematics(q, qd)
        return self._rd.rnea(qdd, state, gravity)


class CRBA:
    """Composite Rigid Body Algorithm (deprecated).

    .. deprecated::
        Use :class:`~bard.core.robot_dynamics.RobotDynamics` instead.
        This class is a thin wrapper that delegates to
        ``RobotDynamics.update_kinematics()`` + ``RobotDynamics.crba()``.

    Args:
        chain: The robot's kinematic chain.
        max_batch_size: Maximum supported batch size.
        compile_enabled: If True, enable ``torch.compile``.
        compile_kwargs: Additional kwargs for ``torch.compile``.
    """

    def __init__(
        self,
        chain: Chain,
        max_batch_size: int = 1024,
        compile_enabled: bool = False,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        warnings.warn(
            "CRBA is deprecated, use RobotDynamics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._rd = RobotDynamics(
            chain,
            max_batch_size=max_batch_size,
            compile_enabled=compile_enabled,
            compile_kwargs=compile_kwargs,
        )
        self.chain = chain
        self.max_batch_size = max_batch_size
        self.dtype = chain.dtype
        self.device = chain.device

    def enable_compilation(self, enabled: bool = True, **compile_kwargs):
        self._rd.enable_compilation(enabled, **compile_kwargs)

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        self._rd.to(dtype=dtype, device=device)
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        return self

    def calc(self, q: torch.Tensor) -> torch.Tensor:
        """Compute mass matrix via CRBA.

        Args:
            q: Generalized positions ``(B, nq)``.

        Returns:
            Mass matrix ``(B, nv, nv)``.
        """
        state = self._rd.update_kinematics(q)
        return self._rd.crba(state)
