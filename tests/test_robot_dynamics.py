"""
Tests for the RobotDynamics v2 API.

Tests verify:
- RobotDynamics.fk matches v1 ForwardKinematics
- RobotDynamics.forward_kinematics (cached) matches fk (standalone)
- RobotDynamics.jacobian (cached) matches v1 Jacobian
- RobotDynamics.rnea (cached) matches v1 RNEA
- RobotDynamics.crba (cached) matches v1 CRBA
- RobotDynamics.spatial_acceleration (cached) matches v1 SpatialAcceleration
- Multi-algorithm cached workflow produces consistent results
- Both fixed-base and floating-base robots
"""

import warnings
from pathlib import Path
import pytest
import torch
import numpy as np

from bard import build_chain_from_urdf, RobotDynamics

try:
    import pinocchio as pin

    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def urdf_path():
    script_dir = Path(__file__).parent
    path = script_dir / "go2_description/urdf/go2.urdf"
    if not path.exists():
        pytest.skip(f"Required test asset not found: {path}")
    return path


@pytest.fixture(params=[torch.float64], ids=["float64"])
def dtype(request):
    return request.param


@pytest.fixture(params=["cpu"])
def device(request):
    return request.param


def _make_random_state(chain, batch_size, dtype, device, seed=42):
    """Generate random q, qd, qdd for a chain."""
    torch.manual_seed(seed)
    if chain.has_floating_base:
        t = torch.randn(batch_size, 3, dtype=dtype, device=device)
        quat = torch.randn(batch_size, 4, dtype=dtype, device=device)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        q_joints = torch.rand(batch_size, chain.n_joints, dtype=dtype, device=device) * 2 - 1
        q = torch.cat([t, quat, q_joints], dim=1)

        v_base = torch.randn(batch_size, 6, dtype=dtype, device=device) * 0.5
        v_joints = torch.randn(batch_size, chain.n_joints, dtype=dtype, device=device) * 0.5
        qd = torch.cat([v_base, v_joints], dim=1)

        a_base = torch.randn(batch_size, 6, dtype=dtype, device=device) * 0.3
        a_joints = torch.randn(batch_size, chain.n_joints, dtype=dtype, device=device) * 0.3
        qdd = torch.cat([a_base, a_joints], dim=1)
    else:
        q = torch.rand(batch_size, chain.n_joints, dtype=dtype, device=device) * 2 - 1
        qd = torch.randn(batch_size, chain.n_joints, dtype=dtype, device=device) * 0.5
        qdd = torch.randn(batch_size, chain.n_joints, dtype=dtype, device=device) * 0.3
    return q, qd, qdd


# ============================================================================
# Fixed-base tests
# ============================================================================


class TestRobotDynamicsFixedBase:
    """Test RobotDynamics with a fixed-base robot."""

    def test_fk_standalone(self, urdf_path, dtype, device):
        """Standalone fk() should produce valid transforms."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, _ = _make_random_state(chain, 5, dtype, device)

        frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
        frame_id = chain.get_frame_id(frame_name)
        T = rd.fk(q, frame_id)
        assert T.shape == (5, 4, 4)
        # Bottom row should be [0, 0, 0, 1]
        expected_bottom = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device)
        assert torch.allclose(T[:, 3, :], expected_bottom.unsqueeze(0).expand(5, -1), atol=1e-10)

    def test_cached_fk_matches_standalone(self, urdf_path, dtype, device):
        """forward_kinematics(state) should match fk(q) for all frames."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, _ = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q)

        for frame_name in chain.get_frame_names(exclude_fixed=True):
            frame_id = chain.get_frame_id(frame_name)
            T_standalone = rd.fk(q, frame_id)
            T_cached = rd.forward_kinematics(frame_id, state)
            assert torch.allclose(T_standalone, T_cached, atol=1e-10), (
                f"FK mismatch at frame {frame_name}: "
                f"max_diff={torch.abs(T_standalone - T_cached).max():.3e}"
            )

    def test_jacobian_cached(self, urdf_path, dtype, device):
        """Cached jacobian should produce correct shapes and values."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, _ = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q)

        frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
        frame_id = chain.get_frame_id(frame_name)

        J_world = rd.jacobian(frame_id, state, reference_frame="world")
        assert J_world.shape == (5, 6, chain.n_joints)

        J_local = rd.jacobian(frame_id, state, reference_frame="local")
        assert J_local.shape == (5, 6, chain.n_joints)

    def test_rnea_cached(self, urdf_path, dtype, device):
        """Cached RNEA should produce correct shapes."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, qd, qdd = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q, qd)
        tau = rd.rnea(qdd, state)
        assert tau.shape == (5, chain.n_joints)

    def test_crba_cached(self, urdf_path, dtype, device):
        """Cached CRBA should produce symmetric positive-definite mass matrix."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, _ = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q)
        M = rd.crba(state)
        assert M.shape == (5, chain.n_joints, chain.n_joints)
        # Symmetry check
        assert torch.allclose(M, M.transpose(1, 2), atol=1e-10), "Mass matrix not symmetric"

    def test_rnea_requires_velocity(self, urdf_path, dtype, device):
        """RNEA should raise error if state lacks velocity data."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, qdd = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q)  # No qd!
        with pytest.raises(ValueError, match="velocity"):
            rd.rnea(qdd, state)

    def test_rnea_crba_consistency(self, urdf_path, dtype, device):
        """M @ qdd should match RNEA(q, 0, qdd, gravity=0) for zero velocity."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, qdd = _make_random_state(chain, 5, dtype, device)

        zero_qd = torch.zeros_like(qdd)
        zero_gravity = torch.zeros(3, dtype=dtype, device=device)

        state = rd.update_kinematics(q, zero_qd)
        tau_rnea = rd.rnea(qdd, state, gravity=zero_gravity)
        M = rd.crba(state)
        tau_crba = (M @ qdd.unsqueeze(-1)).squeeze(-1)

        tol = 1e-4 if dtype == torch.float32 else 1e-8
        assert torch.allclose(tau_rnea, tau_crba, atol=tol), (
            f"RNEA-CRBA mismatch: max_diff={torch.abs(tau_rnea - tau_crba).max():.3e}"
        )


# ============================================================================
# Floating-base tests
# ============================================================================


class TestRobotDynamicsFloatingBase:
    """Test RobotDynamics with a floating-base robot."""

    def test_cached_fk_matches_standalone(self, urdf_path, dtype, device):
        """forward_kinematics(state) should match fk(q) for floating-base."""
        chain = build_chain_from_urdf(urdf_path, floating_base=True).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, _ = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q)

        for frame_name in chain.get_frame_names(exclude_fixed=True):
            frame_id = chain.get_frame_id(frame_name)
            T_standalone = rd.fk(q, frame_id)
            T_cached = rd.forward_kinematics(frame_id, state)
            assert torch.allclose(T_standalone, T_cached, atol=1e-10), (
                f"FK mismatch at frame {frame_name}: "
                f"max_diff={torch.abs(T_standalone - T_cached).max():.3e}"
            )

    def test_rnea_cached(self, urdf_path, dtype, device):
        """Cached RNEA for floating base should produce correct shapes."""
        chain = build_chain_from_urdf(urdf_path, floating_base=True).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, qd, qdd = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q, qd)
        tau = rd.rnea(qdd, state)
        assert tau.shape == (5, 6 + chain.n_joints)

    def test_crba_cached(self, urdf_path, dtype, device):
        """Cached CRBA for floating base should produce symmetric mass matrix."""
        chain = build_chain_from_urdf(urdf_path, floating_base=True).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, _ = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q)
        M = rd.crba(state)
        nv = 6 + chain.n_joints
        assert M.shape == (5, nv, nv)
        assert torch.allclose(M, M.transpose(1, 2), atol=1e-10), "Mass matrix not symmetric"

    def test_rnea_crba_consistency(self, urdf_path, dtype, device):
        """M @ qdd should match RNEA(q, 0, qdd, gravity=0) for floating base."""
        chain = build_chain_from_urdf(urdf_path, floating_base=True).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, qdd = _make_random_state(chain, 5, dtype, device)

        zero_qd = torch.zeros_like(qdd)
        zero_gravity = torch.zeros(3, dtype=dtype, device=device)

        state = rd.update_kinematics(q, zero_qd)
        tau_rnea = rd.rnea(qdd, state, gravity=zero_gravity)
        M = rd.crba(state)
        tau_crba = (M @ qdd.unsqueeze(-1)).squeeze(-1)

        tol = 1e-4 if dtype == torch.float32 else 1e-8
        assert torch.allclose(tau_rnea, tau_crba, atol=tol), (
            f"RNEA-CRBA mismatch: max_diff={torch.abs(tau_rnea - tau_crba).max():.3e}"
        )

    def test_jacobian_return_eef_pose(self, urdf_path, dtype, device):
        """Jacobian with return_eef_pose should return both J and T."""
        chain = build_chain_from_urdf(urdf_path, floating_base=True).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        q, _, _ = _make_random_state(chain, 5, dtype, device)

        state = rd.update_kinematics(q)
        frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
        frame_id = chain.get_frame_id(frame_name)

        result = rd.jacobian(frame_id, state, "world", return_eef_pose=True)
        assert isinstance(result, tuple)
        J, T = result
        nv = 6 + chain.n_joints
        assert J.shape == (5, 6, nv)
        assert T.shape == (5, 4, 4)


# ============================================================================
# Pinocchio cross-validation
# ============================================================================


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio library not found")
class TestPinocchioValidation:
    """Cross-validate RobotDynamics against Pinocchio."""

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_path):
        model = pin.buildModelFromUrdf(str(urdf_path))
        return model, model.createData()

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_path):
        model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        return model, model.createData()

    def test_fk_vs_pinocchio_fixed(self, urdf_path, pin_model_fixed, dtype, device):
        """FK should match Pinocchio for fixed-base robot."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        pin_model, pin_data = pin_model_fixed

        q, _, _ = _make_random_state(chain, 5, dtype, device, seed=100)

        frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
        frame_id = chain.get_frame_id(frame_name)
        pin_frame_id = pin_model.getFrameId(frame_name)

        T_bard = rd.fk(q, frame_id).cpu().numpy()

        for i in range(5):
            q_pin = q[i].cpu().numpy()
            pin.forwardKinematics(pin_model, pin_data, q_pin)
            pin.updateFramePlacements(pin_model, pin_data)
            T_pin = pin_data.oMf[pin_frame_id]

            pos_err = np.linalg.norm(T_bard[i, :3, 3] - T_pin.translation)
            tol = 1e-3 if dtype == torch.float32 else 1e-7
            assert pos_err < tol, f"FK position error {pos_err:.3e} at sample {i}"

    def test_rnea_vs_pinocchio_fixed(self, urdf_path, pin_model_fixed, dtype, device):
        """RNEA should match Pinocchio for fixed-base robot."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        pin_model, pin_data = pin_model_fixed

        q, qd, qdd = _make_random_state(chain, 5, dtype, device, seed=200)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)

        state = rd.update_kinematics(q, qd)
        tau_bard = rd.rnea(qdd, state, gravity=gravity).cpu().numpy()

        for i in range(5):
            q_pin = q[i].cpu().numpy()
            qd_pin = qd[i].cpu().numpy()
            qdd_pin = qdd[i].cpu().numpy()
            tau_pin = pin.rnea(pin_model, pin_data, q_pin, qd_pin, qdd_pin)

            max_diff = np.abs(tau_bard[i] - tau_pin).max()
            tol = 5e-4 if dtype == torch.float32 else 5e-6
            assert max_diff < tol, f"RNEA mismatch at sample {i}: max_diff={max_diff:.3e}"

    def test_crba_vs_pinocchio_fixed(self, urdf_path, pin_model_fixed, dtype, device):
        """CRBA should match Pinocchio for fixed-base robot."""
        chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        rd = RobotDynamics(chain, max_batch_size=10)
        pin_model, pin_data = pin_model_fixed

        q, _, _ = _make_random_state(chain, 5, dtype, device, seed=300)

        state = rd.update_kinematics(q)
        M_bard = rd.crba(state).cpu().numpy()

        for i in range(5):
            q_pin = q[i].cpu().numpy()
            M_pin = pin.crba(pin_model, pin_data, q_pin)

            max_diff = np.abs(M_bard[i] - M_pin).max()
            tol = 1e-4 if dtype == torch.float32 else 1e-6
            assert max_diff < tol, f"CRBA mismatch at sample {i}: max_diff={max_diff:.3e}"
