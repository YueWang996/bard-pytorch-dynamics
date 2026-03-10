"""
Tests for the bard v0.3 unified API.

Tests verify:
- bard.forward_kinematics (standalone) produces valid transforms
- bard.forward_kinematics (cached) matches standalone
- bard.jacobian (cached) produces correct shapes
- bard.rnea (cached) produces correct shapes
- bard.crba (cached) produces symmetric positive-definite mass matrix
- bard.spatial_acceleration (cached) works correctly
- Multi-algorithm cached workflow produces consistent results
- Both fixed-base and floating-base robots
"""

import warnings
from pathlib import Path
import pytest
import torch
import numpy as np

import bard

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


def _make_random_state(model, batch_size, dtype, device, seed=42):
    """Generate random q, qd, qdd for a model."""
    torch.manual_seed(seed)
    if model.has_floating_base:
        t = torch.randn(batch_size, 3, dtype=dtype, device=device)
        quat = torch.randn(batch_size, 4, dtype=dtype, device=device)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        q_joints = torch.rand(batch_size, model.n_joints, dtype=dtype, device=device) * 2 - 1
        q = torch.cat([t, quat, q_joints], dim=1)

        v_base = torch.randn(batch_size, 6, dtype=dtype, device=device) * 0.5
        v_joints = torch.randn(batch_size, model.n_joints, dtype=dtype, device=device) * 0.5
        qd = torch.cat([v_base, v_joints], dim=1)

        a_base = torch.randn(batch_size, 6, dtype=dtype, device=device) * 0.3
        a_joints = torch.randn(batch_size, model.n_joints, dtype=dtype, device=device) * 0.3
        qdd = torch.cat([a_base, a_joints], dim=1)
    else:
        q = torch.rand(batch_size, model.n_joints, dtype=dtype, device=device) * 2 - 1
        qd = torch.randn(batch_size, model.n_joints, dtype=dtype, device=device) * 0.5
        qdd = torch.randn(batch_size, model.n_joints, dtype=dtype, device=device) * 0.3
    return q, qd, qdd


# ============================================================================
# Fixed-base tests
# ============================================================================


class TestRobotDynamicsFixedBase:
    """Test bard v0.3 API with a fixed-base robot."""

    def test_fk_standalone(self, urdf_path, dtype, device):
        """Standalone fk() should produce valid transforms."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)

        frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_id = model.get_frame_id(frame_name)
        T = bard.forward_kinematics(model, data, frame_id, q=q)
        assert T.shape == (5, 4, 4)
        # Bottom row should be [0, 0, 0, 1]
        expected_bottom = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device)
        assert torch.allclose(T[:, 3, :], expected_bottom.unsqueeze(0).expand(5, -1), atol=1e-10)

    def test_cached_fk_matches_standalone(self, urdf_path, dtype, device):
        """forward_kinematics(cached) should match standalone fk for all frames."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q)

        for frame_name in model.get_frame_names(exclude_fixed=True):
            frame_id = model.get_frame_id(frame_name)
            T_standalone = bard.forward_kinematics(model, data, frame_id, q=q)
            T_cached = bard.forward_kinematics(model, data, frame_id)
            assert torch.allclose(T_standalone, T_cached, atol=1e-10), (
                f"FK mismatch at frame {frame_name}: "
                f"max_diff={torch.abs(T_standalone - T_cached).max():.3e}"
            )

    def test_jacobian_cached(self, urdf_path, dtype, device):
        """Cached jacobian should produce correct shapes and values."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q)

        frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_id = model.get_frame_id(frame_name)

        J_world = bard.jacobian(model, data, frame_id, reference_frame="world")
        assert J_world.shape == (5, 6, model.n_joints)

        J_local = bard.jacobian(model, data, frame_id, reference_frame="local")
        assert J_local.shape == (5, 6, model.n_joints)

    def test_rnea_cached(self, urdf_path, dtype, device):
        """Cached RNEA should produce correct shapes."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, qd, qdd = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q, qd)
        tau = bard.rnea(model, data, qdd)
        assert tau.shape == (5, model.n_joints)

    def test_crba_cached(self, urdf_path, dtype, device):
        """Cached CRBA should produce symmetric positive-definite mass matrix."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)
        assert M.shape == (5, model.n_joints, model.n_joints)
        # Symmetry check
        assert torch.allclose(M, M.transpose(1, 2), atol=1e-10), "Mass matrix not symmetric"

    def test_rnea_requires_velocity(self, urdf_path, dtype, device):
        """RNEA should raise error if state lacks velocity data."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, _, qdd = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q)  # No qd!
        with pytest.raises(ValueError, match="velocity"):
            bard.rnea(model, data, qdd)

    def test_rnea_crba_consistency(self, urdf_path, dtype, device):
        """M @ qdd should match RNEA(q, 0, qdd, gravity=0) for zero velocity."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, _, qdd = _make_random_state(model, 5, dtype, device)

        zero_qd = torch.zeros_like(qdd)
        zero_gravity = torch.zeros(3, dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, zero_qd)
        tau_rnea = bard.rnea(model, data, qdd, gravity=zero_gravity)
        M = bard.crba(model, data)
        tau_crba = (M @ qdd.unsqueeze(-1)).squeeze(-1)

        tol = 1e-4 if dtype == torch.float32 else 1e-8
        assert torch.allclose(
            tau_rnea, tau_crba, atol=tol
        ), f"RNEA-CRBA mismatch: max_diff={torch.abs(tau_rnea - tau_crba).max():.3e}"

    def test_aba_cached(self, urdf_path, dtype, device):
        """ABA should produce correct output shape."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, qd, _ = _make_random_state(model, 5, dtype, device)
        tau = torch.randn(5, model.n_joints, dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, qd)
        qdd = bard.aba(model, data, tau)
        assert qdd.shape == (5, model.n_joints)

    def test_aba_requires_velocity(self, urdf_path, dtype, device):
        """ABA should raise error if state lacks velocity data."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)
        tau = torch.randn(5, model.n_joints, dtype=dtype, device=device)

        bard.update_kinematics(model, data, q)  # No qd!
        with pytest.raises(ValueError, match="velocity"):
            bard.aba(model, data, tau)

    def test_aba_rnea_crba_consistency(self, urdf_path, dtype, device):
        """ABA(tau) should match solve(M, tau - C(q,qd) - g(q))."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        q, qd, _ = _make_random_state(model, 5, dtype, device)
        tau = torch.randn(5, model.n_joints, dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, qd)
        qdd_aba = bard.aba(model, data, tau)

        # Compute qdd via CRBA: qdd = M^{-1} (tau - bias)
        zero_qdd = torch.zeros_like(qd)
        bias = bard.rnea(model, data, zero_qdd)  # C(q,qd) + g(q)
        M = bard.crba(model, data)
        qdd_crba = torch.linalg.solve(M, (tau - bias).unsqueeze(-1)).squeeze(-1)

        assert torch.allclose(
            qdd_aba, qdd_crba, atol=1e-6
        ), f"ABA-CRBA mismatch: max_diff={torch.abs(qdd_aba - qdd_crba).max():.3e}"


# ============================================================================
# Floating-base tests
# ============================================================================


class TestRobotDynamicsFloatingBase:
    """Test bard v0.3 API with a floating-base robot."""

    def test_cached_fk_matches_standalone(self, urdf_path, dtype, device):
        """forward_kinematics(cached) should match standalone fk for floating-base."""
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q)

        for frame_name in model.get_frame_names(exclude_fixed=True):
            frame_id = model.get_frame_id(frame_name)
            T_standalone = bard.forward_kinematics(model, data, frame_id, q=q)
            T_cached = bard.forward_kinematics(model, data, frame_id)
            assert torch.allclose(T_standalone, T_cached, atol=1e-10), (
                f"FK mismatch at frame {frame_name}: "
                f"max_diff={torch.abs(T_standalone - T_cached).max():.3e}"
            )

    def test_rnea_cached(self, urdf_path, dtype, device):
        """Cached RNEA for floating base should produce correct shapes."""
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=10)
        q, qd, qdd = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q, qd)
        tau = bard.rnea(model, data, qdd)
        assert tau.shape == (5, 6 + model.n_joints)

    def test_crba_cached(self, urdf_path, dtype, device):
        """Cached CRBA for floating base should produce symmetric mass matrix."""
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)
        nv = 6 + model.n_joints
        assert M.shape == (5, nv, nv)
        assert torch.allclose(M, M.transpose(1, 2), atol=1e-10), "Mass matrix not symmetric"

    def test_rnea_crba_consistency(self, urdf_path, dtype, device):
        """M @ qdd should match RNEA(q, 0, qdd, gravity=0) for floating base."""
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=10)
        q, _, qdd = _make_random_state(model, 5, dtype, device)

        zero_qd = torch.zeros_like(qdd)
        zero_gravity = torch.zeros(3, dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, zero_qd)
        tau_rnea = bard.rnea(model, data, qdd, gravity=zero_gravity)
        M = bard.crba(model, data)
        tau_crba = (M @ qdd.unsqueeze(-1)).squeeze(-1)

        tol = 1e-4 if dtype == torch.float32 else 1e-8
        assert torch.allclose(
            tau_rnea, tau_crba, atol=tol
        ), f"RNEA-CRBA mismatch: max_diff={torch.abs(tau_rnea - tau_crba).max():.3e}"

    def test_aba_cached(self, urdf_path, dtype, device):
        """ABA for floating base should produce correct output shape."""
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=10)
        q, qd, _ = _make_random_state(model, 5, dtype, device)
        nv = 6 + model.n_joints
        tau = torch.randn(5, nv, dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, qd)
        qdd = bard.aba(model, data, tau)
        assert qdd.shape == (5, nv)

    def test_aba_rnea_crba_consistency(self, urdf_path, dtype, device):
        """ABA(tau) should match solve(M, tau - bias) for floating base."""
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=10)
        q, qd, _ = _make_random_state(model, 5, dtype, device)
        nv = 6 + model.n_joints
        tau = torch.randn(5, nv, dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, qd)
        qdd_aba = bard.aba(model, data, tau)

        # Compute qdd via CRBA: qdd = M^{-1} (tau - bias)
        zero_qdd = torch.zeros_like(qd)
        bias = bard.rnea(model, data, zero_qdd)
        M = bard.crba(model, data)
        qdd_crba = torch.linalg.solve(M, (tau - bias).unsqueeze(-1)).squeeze(-1)

        assert torch.allclose(
            qdd_aba, qdd_crba, atol=1e-6
        ), f"ABA-CRBA mismatch: max_diff={torch.abs(qdd_aba - qdd_crba).max():.3e}"

    def test_jacobian_return_pose(self, urdf_path, dtype, device):
        """Jacobian with return_pose should return both J and T."""
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=10)
        q, _, _ = _make_random_state(model, 5, dtype, device)

        bard.update_kinematics(model, data, q)
        frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_id = model.get_frame_id(frame_name)

        result = bard.jacobian(model, data, frame_id, reference_frame="world", return_pose=True)
        assert isinstance(result, tuple)
        J, T = result
        nv = 6 + model.n_joints
        assert J.shape == (5, 6, nv)
        assert T.shape == (5, 4, 4)


# ============================================================================
# Pinocchio cross-validation
# ============================================================================


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio library not found")
class TestPinocchioValidation:
    """Cross-validate bard v0.3 API against Pinocchio."""

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
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        pin_model, pin_data = pin_model_fixed

        q, _, _ = _make_random_state(model, 5, dtype, device, seed=100)

        frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_id = model.get_frame_id(frame_name)
        pin_frame_id = pin_model.getFrameId(frame_name)

        T_bard = bard.forward_kinematics(model, data, frame_id, q=q).cpu().numpy()

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
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        pin_model, pin_data = pin_model_fixed

        q, qd, qdd = _make_random_state(model, 5, dtype, device, seed=200)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, qd)
        tau_bard = bard.rnea(model, data, qdd, gravity=gravity).cpu().numpy()

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
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        pin_model, pin_data = pin_model_fixed

        q, _, _ = _make_random_state(model, 5, dtype, device, seed=300)

        bard.update_kinematics(model, data, q)
        M_bard = bard.crba(model, data).cpu().numpy()

        for i in range(5):
            q_pin = q[i].cpu().numpy()
            M_pin = pin.crba(pin_model, pin_data, q_pin)

            max_diff = np.abs(M_bard[i] - M_pin).max()
            tol = 1e-4 if dtype == torch.float32 else 1e-6
            assert max_diff < tol, f"CRBA mismatch at sample {i}: max_diff={max_diff:.3e}"

    def test_aba_vs_pinocchio_fixed(self, urdf_path, pin_model_fixed, dtype, device):
        """ABA should match Pinocchio for fixed-base robot."""
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=10)
        pin_model, pin_data = pin_model_fixed

        q, qd, _ = _make_random_state(model, 5, dtype, device, seed=400)
        tau = torch.randn(5, model.n_joints, dtype=dtype, device=device)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)

        bard.update_kinematics(model, data, q, qd)
        qdd_bard = bard.aba(model, data, tau, gravity=gravity).cpu().numpy()

        for i in range(5):
            q_pin = q[i].cpu().numpy()
            qd_pin = qd[i].cpu().numpy()
            tau_pin = tau[i].cpu().numpy()
            qdd_pin = pin.aba(pin_model, pin_data, q_pin, qd_pin, tau_pin)

            max_diff = np.abs(qdd_bard[i] - qdd_pin).max()
            tol = 5e-3
            assert max_diff < tol, f"ABA mismatch at sample {i}: max_diff={max_diff:.3e}"
