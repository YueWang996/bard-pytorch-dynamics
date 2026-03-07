"""
BARD - Batched Articulated Robot Dynamics.

A lightweight, PyTorch-native library for rigid-body dynamics that leverages
tensor operations to perform efficient, batched computations on CPU or GPU.

Example usage::

    import bard

    model = bard.build_model_from_urdf("robot.urdf", dtype=torch.float32)
    data = bard.create_data(model, max_batch_size=4096)

    bard.update_kinematics(model, data, q, qd)
    T = bard.forward_kinematics(model, data, frame_id)
    J = bard.jacobian(model, data, frame_id)
    tau = bard.rnea(model, data, qdd)
    M = bard.crba(model, data)
"""

try:
    from importlib.metadata import version

    __version__ = version("bard")
except Exception:
    __version__ = "unknown"

__author__ = "Yue Wang"

from .core.model import Model
from .core.data import Data
from .parsers.urdf import build_model_from_urdf
from .api import (
    create_data,
    update_kinematics,
    forward_kinematics,
    jacobian,
    rnea,
    crba,
    spatial_acceleration,
)

__all__ = [
    "Model",
    "Data",
    "build_model_from_urdf",
    "create_data",
    "update_kinematics",
    "forward_kinematics",
    "jacobian",
    "rnea",
    "crba",
    "spatial_acceleration",
]
