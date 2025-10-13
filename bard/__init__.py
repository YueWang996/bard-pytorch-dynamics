"""
BARD - Batched Articulated Robot Dynamics.

This library provides efficient, GPU-accelerated implementations of robot
kinematics and dynamics algorithms with support for batch processing.
"""

try:
    from importlib.metadata import version

    __version__ = version("bard")
except Exception:
    __version__ = "unknown"

__author__ = "Yue Wang"

from .core.chain import Chain
from .core.kinematics import ForwardKinematics, SpatialAcceleration
from .core.jacobian import Jacobian
from .core.dynamics import RNEA, CRBA
from .parsers.urdf import build_chain_from_urdf

__all__ = [
    "Chain",
    "ForwardKinematics",
    "SpatialAcceleration",
    "Jacobian",
    "RNEA",
    "CRBA",
    "build_chain_from_urdf",
]
