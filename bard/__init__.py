"""
BARD - Batched Articulated Robot Dynamics.

This library provides efficient, GPU-accelerated implementations of robot
kinematics and dynamics algorithms with support for batch processing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core.chain import Chain
from .core.kinematics import calc_forward_kinematics, end_effector_acceleration
from .core.jacobian import calc_jacobian
from .core.dynamics import calc_inverse_dynamics, crba_inertia_matrix
from .parsers.urdf import build_chain_from_urdf

__all__ = [
    "Chain",
    "calc_forward_kinematics",
    "calc_jacobian",
    "end_effector_acceleration",
    "calc_inverse_dynamics",
    "crba_inertia_matrix",
    "build_chain_from_urdf",
    "build_serial_chain_from_urdf",
]