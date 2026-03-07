"""
Kinematic computation classes (deprecated, use ``RobotDynamics`` instead).

This module provides backward-compatible wrappers around ``RobotDynamics``.
New code should use ``RobotDynamics`` directly for better performance.
"""

import warnings
from typing import Dict, Optional, Any
import torch
from bard.core.chain import Chain
from bard.core.robot_dynamics import RobotDynamics


class ForwardKinematics:
    """Forward kinematics computation (deprecated).

    .. deprecated::
        Use :class:`~bard.core.robot_dynamics.RobotDynamics` instead.
        This class is a thin wrapper that delegates to ``RobotDynamics.fk()``.

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
            "ForwardKinematics is deprecated, use RobotDynamics instead.",
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

    def calc(self, q: torch.Tensor, frame_id: int) -> torch.Tensor:
        """Compute forward kinematics for a specific frame.

        Args:
            q: Generalized positions ``(B, nq)``.
            frame_id: Target frame index.

        Returns:
            Homogeneous transform ``(B, 4, 4)``.
        """
        return self._rd.fk(q, frame_id)


class SpatialAcceleration:
    """Spatial acceleration computation (deprecated).

    .. deprecated::
        Use :class:`~bard.core.robot_dynamics.RobotDynamics` instead.
        This class is a thin wrapper that delegates to
        ``RobotDynamics.update_kinematics()`` + ``RobotDynamics.spatial_acceleration()``.

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
            "SpatialAcceleration is deprecated, use RobotDynamics instead.",
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
        frame_id: int,
        reference_frame: str,
    ) -> torch.Tensor:
        """Compute spatial acceleration of a frame.

        Args:
            q: Generalized positions ``(B, nq)``.
            qd: Generalized velocities ``(B, nv)``.
            qdd: Generalized accelerations ``(B, nv)``.
            frame_id: Target frame index.
            reference_frame: ``"world"`` or ``"local"``.

        Returns:
            Spatial acceleration ``(B, 6)``.
        """
        state = self._rd.update_kinematics(q, qd)
        return self._rd.spatial_acceleration(qdd, frame_id, state, reference_frame)
