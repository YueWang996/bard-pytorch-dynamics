import os
from pathlib import Path
import torch
import numpy as np
import pinocchio as pin

# ------------------------------
# Config resolution (no env needed)
# If 'benchlocal.py' exists alongside this file, its constants override defaults.
# ------------------------------

script_dir = Path(__file__).parent

# Defaults (used if benchlocal.py is absent or missing a field)
_DEFAULT_URDF_PATH = script_dir / "../examples/example_robots/go2_description/urdf/go2.urdf"
_DEFAULT_BATCH_SIZES = [10, 100, 1000, 10000]
_DEFAULT_NUM_REPEATS = 100
_DEFAULT_WARMUP_ITERS = 50
_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEFAULT_DTYPE = torch.float64

# Try to import local overrides
URDF_PATH = _DEFAULT_URDF_PATH
BATCH_SIZES = _DEFAULT_BATCH_SIZES
NUM_REPEATS = _DEFAULT_NUM_REPEATS
WARMUP_ITERS = _DEFAULT_WARMUP_ITERS
DEVICE = _DEFAULT_DEVICE
DTYPE = _DEFAULT_DTYPE

try:
    # Import benchlocal if present
    from . import benchlocal as _bl  # type: ignore

    URDF_PATH = Path(getattr(_bl, "URDF_PATH", URDF_PATH))
    BATCH_SIZES = list(getattr(_bl, "BATCH_SIZES", BATCH_SIZES))
    NUM_REPEATS = int(getattr(_bl, "NUM_REPEATS", NUM_REPEATS))
    WARMUP_ITERS = int(getattr(_bl, "WARMUP_ITERS", WARMUP_ITERS))
    DEVICE = str(getattr(_bl, "DEVICE", DEVICE))
    _dtype = getattr(_bl, "DTYPE", DTYPE)
    if isinstance(_dtype, str):
        DTYPE = torch.float64 if _dtype.lower() in ("float64", "double") else torch.float32
    else:
        DTYPE = _dtype
except Exception:
    # No local overrides; keep defaults
    pass

# ------------------------------
# Pinocchio helpers and wrappers
# ------------------------------


def build_pin_model(urdf_path: Path):
    """Create Pinocchio model+data for a free-flyer base."""
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()
    return model, data


def ref_frame_to_pin(reference_frame: str):
    """Map string ('world' or 'local') to Pinocchio enum."""
    if reference_frame == "world":
        return pin.ReferenceFrame.WORLD
    elif reference_frame == "local":
        return pin.ReferenceFrame.LOCAL
    else:
        raise ValueError("reference_frame must be 'world' or 'local'")


class PinocchioTorchWrapper:
    """Accept torch tensors, include conversion overhead, return torch tensors.

    Methods:
      - calc_frame_accel(q, qd, qdd, frame_id, reference_frame) -> (B,6)
      - calc_bias_accel(q, qd, frame_id, reference_frame) -> (B,6)  [NEW: qdd=0]
      - calc_mass_matrix(q) -> (B,nv,nv)  [CRBA]
      - calc_frame_jacobian(q, frame_id, reference_frame) -> (B,6,nv)
      - calc_frame_pose(q, frame_id) -> (B,4,4)  [Forward Kinematics]
      - calc_inverse_dynamics(q, qd, qdd) -> (B,nv)  [RNEA]
    """

    def __init__(self, model: pin.Model, device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.model = model
        self.data = model.createData()
        self.device = device
        self.dtype = dtype

    # ---- Utilities ----
    def _to_numpy_q(self, q_t):
        """Convert Bard-layout q (tx,ty,tz, qw,qx,qy,qz, joints...) to Pinocchio layout per-sample."""
        B = q_t.shape[0]
        out = []
        for i in range(B):
            q_i = q_t[i].detach().cpu().numpy()
            # Bard quaternion order assumed [qw, qx, qy, qz] in q[3:7]; Pin wants [qx,qy,qz,qw]
            out.append(np.concatenate([q_i[:3], q_i[4:7], q_i[3:4], q_i[7:]]))
        return out

    @torch.no_grad()
    def calc_frame_accel(self, q, qd, qdd, frame_id, reference_frame="world"):
        """Compute full spatial acceleration.
        
        Args:
            q: Batch of positions (B, nq)
            qd: Batch of velocities (B, nv)
            qdd: Batch of accelerations (B, nv)
            frame_id: Frame index
            reference_frame: 'world' or 'local'
        
        Returns:
            Spatial acceleration (B, 6)
        """
        B = q.shape[0]
        out = np.zeros((B, 6), dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        pin_ref = ref_frame_to_pin(reference_frame)
        q_list = self._to_numpy_q(q)
        for i in range(B):
            qd_i = qd[i].detach().cpu().numpy()
            qdd_i = qdd[i].detach().cpu().numpy()
            pin.forwardKinematics(self.model, self.data, q_list[i], qd_i, qdd_i)
            a = pin.getFrameAcceleration(self.model, self.data, frame_id, pin_ref).vector
            out[i, :] = a
        return torch.tensor(out, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def calc_bias_accel(self, q, qd, frame_id, reference_frame="world"):
        """Compute bias acceleration (Coriolis + centrifugal terms only).
        
        This is equivalent to calling calc_frame_accel with qdd=0.
        
        Args:
            q: Batch of positions (B, nq)
            qd: Batch of velocities (B, nv)
            frame_id: Frame index
            reference_frame: 'world' or 'local'
        
        Returns:
            Bias acceleration (B, 6)
        """
        B = q.shape[0]
        out = np.zeros((B, 6), dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        pin_ref = ref_frame_to_pin(reference_frame)
        q_list = self._to_numpy_q(q)
        for i in range(B):
            qd_i = qd[i].detach().cpu().numpy()
            qdd_zero = np.zeros_like(qd_i)  # Zero acceleration for bias computation
            pin.forwardKinematics(self.model, self.data, q_list[i], qd_i, qdd_zero)
            a = pin.getFrameAcceleration(self.model, self.data, frame_id, pin_ref).vector
            out[i, :] = a
        return torch.tensor(out, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def calc_mass_matrix(self, q):
        q_list = self._to_numpy_q(q)
        B = len(q_list)
        nv = self.model.nv
        out = np.zeros((B, nv, nv), dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        for i in range(B):
            M = pin.crba(self.model, self.data, q_list[i])
            M = np.triu(M) + np.triu(M, 1).T
            out[i, :, :] = M
        return torch.tensor(out, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def calc_frame_jacobian(self, q, frame_id, reference_frame="world"):
        q_list = self._to_numpy_q(q)
        B = len(q_list)
        nv = self.model.nv
        out = np.zeros((B, 6, nv), dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        pin_ref = ref_frame_to_pin(reference_frame)
        for i in range(B):
            pin.framesForwardKinematics(self.model, self.data, q_list[i])
            J = pin.computeFrameJacobian(self.model, self.data, q_list[i], frame_id, pin_ref)
            out[i, :, :] = J
        return torch.tensor(out, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def calc_frame_pose(self, q, frame_id):
        q_list = self._to_numpy_q(q)
        B = len(q_list)
        out = np.zeros((B, 4, 4), dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        for i in range(B):
            pin.framesForwardKinematics(self.model, self.data, q_list[i])
            T = self.data.oMf[frame_id].homogeneous
            out[i, :, :] = T
        return torch.tensor(out, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def calc_inverse_dynamics(self, q, qd, qdd):
        q_list = self._to_numpy_q(q)
        B = len(q_list)
        nv = self.model.nv
        out = np.zeros((B, nv), dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        for i in range(B):
            qd_i = qd[i].detach().cpu().numpy()
            qdd_i = qdd[i].detach().cpu().numpy()
            tau = pin.rnea(self.model, self.data, q_list[i], qd_i, qdd_i)
            out[i, :] = tau
        return torch.tensor(out, device=self.device, dtype=self.dtype)
    