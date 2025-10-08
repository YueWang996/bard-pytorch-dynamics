from pathlib import Path
import sys
import os
import pytest
import torch
import numpy as np

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

# Skip all tests if Pinocchio is not available
pytestmark = pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio library not found")


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def urdf_string():
    """Provides the URDF content as a string from the fixtures directory."""
    script_dir = Path(__file__).parent
    urdf_path = script_dir / "go2_description/urdf/go2.urdf"
    if not os.path.exists(urdf_path):
        pytest.skip(f"Required test asset not found: {urdf_path}")
    with open(urdf_path, "rb") as f:
        return f.read()

@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request):
    """Parameterizes tests for both float and double precision."""
    return request.param

@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA not available"
        )),
        # pytest.param("mps", marks=pytest.mark.skipif(
        #     not torch.backends.mps.is_available(),
        #     reason="Apple MPS not available"
        # ))
    ]
)
def device(request):
    """Parameterizes tests for CPU, CUDA, and Apple MPS if available."""
    return request.param


# ============================================================================
# Helper Functions
# ============================================================================

def compare_transforms(T_ours, T_pin, dtype):
    """
    Compares two 4x4 homogeneous transformation matrices with precision-aware tolerance.
    """
    # Set tolerance based on the data type being tested
    pos_tol = 1e-3 if dtype == torch.float32 else 1e-7
    rot_tol = 1e-3 if dtype == torch.float32 else 1e-7

    R_ours, p_ours = T_ours[:3, :3], T_ours[:3, 3]
    R_pin, p_pin = T_pin.rotation, T_pin.translation
    
    pos_err = float(np.linalg.norm(p_ours - p_pin))
    
    # Geodesic distance on SO(3)
    R_err = R_ours @ R_pin.T
    trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.arccos(trace_val))
    
    assert pos_err < pos_tol, f"Position error {pos_err:.3e} exceeds tolerance {pos_tol:.1e}"
    assert rot_err < rot_tol, f"Rotation error {rot_err:.3e} exceeds tolerance {rot_tol:.1e}"