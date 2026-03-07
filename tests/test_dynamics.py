"""
Comprehensive tests for Dynamics algorithms (RNEA and CRBA for both fixed-base and floating-base).

Tests cover:
- RNEA: Inverse dynamics (tau = RNEA(q, qd, qdd))
- CRBA: Mass matrix computation (M = CRBA(q))
- Component separation (gravity, coriolis, inertia terms)
- RNEA-CRBA consistency (M*qdd == RNEA with zero velocity and gravity)
- Batched operations
- Compilation compatibility
"""

import warnings
import pytest
import torch
import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None

import bard


def compare_vectors(tau_bard, tau_pin, dtype, name="vector"):
    warn_tol = 5e-4 if dtype == torch.float32 else 5e-6
    fail_tol = warn_tol * 1.2
    max_diff = np.abs(tau_bard - tau_pin).max()
    if max_diff >= fail_tol:
        assert False, f"{name} mismatch: max_diff={max_diff:.3e} > fail_tol={fail_tol:.1e}"
    elif max_diff >= warn_tol:
        warnings.warn(
            f"{name} close to tolerance: max_diff={max_diff:.3e}, warn_tol={warn_tol:.1e}"
        )


def compare_matrices(M_bard, M_pin, dtype, name="matrix"):
    warn_tol = 1e-4 if dtype == torch.float32 else 1e-6
    fail_tol = warn_tol * 1.2
    if M_pin.shape != M_bard.shape:
        raise AssertionError(f"Shape mismatch: Bard {M_bard.shape} vs Pinocchio {M_pin.shape}")
    max_diff = np.abs(M_bard - M_pin).max()
    if max_diff >= fail_tol:
        assert False, f"{name} mismatch: max_diff={max_diff:.3e} > fail_tol={fail_tol:.1e}"
    elif max_diff >= warn_tol:
        warnings.warn(
            f"{name} close to tolerance: max_diff={max_diff:.3e}, warn_tol={warn_tol:.1e}"
        )


@pytest.mark.skipif(not hasattr(pin, "buildModelFromUrdf"), reason="Pinocchio not available")
class TestDynamics:
    """Test suite for dynamics algorithms (RNEA and CRBA) with both fixed-base and floating-base robots."""

    # ========================================================================
    # Fixed-Base RNEA Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_path):
        model = pin.buildModelFromUrdf(str(urdf_path))
        return model, model.createData()

    def test_fixed_base_rnea_full(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_fixed

        torch.manual_seed(1000)
        q = torch.rand(1, model.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, model.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, model.n_joints, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q, qd)
        tau_bard = bard.rnea(model, data, qdd)[0].cpu().numpy()

        tau_pin = pin.rnea(
            pin_model_obj, pin_data, q[0].cpu().numpy(), qd[0].cpu().numpy(), qdd[0].cpu().numpy()
        )
        compare_vectors(tau_bard, tau_pin, dtype, name="Full RNEA")

    def test_fixed_base_rnea_components(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_fixed

        torch.manual_seed(1001)
        q = torch.rand(1, model.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, model.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, model.n_joints, device=device, dtype=dtype)

        q_pin = q[0].cpu().numpy()
        qd_pin = qd[0].cpu().numpy()

        # Gravity term: RNEA(q, 0, 0)
        zeros = torch.zeros_like(q)
        bard.update_kinematics(model, data, q, zeros)
        g_bard = bard.rnea(model, data, zeros)[0].cpu().numpy()
        g_pin = pin.computeGeneralizedGravity(pin_model_obj, pin_data, q_pin)
        compare_vectors(g_bard, g_pin, dtype, name="Gravity term")

        # Coriolis term: RNEA(q, qd, 0, g=0)
        zero_gravity = torch.zeros(3, device=device, dtype=dtype)
        bard.update_kinematics(model, data, q, qd)
        c_bard = bard.rnea(model, data, zeros, gravity=zero_gravity)[0].cpu().numpy()
        c_pin = pin.computeCoriolisMatrix(pin_model_obj, pin_data, q_pin, qd_pin) @ qd_pin
        compare_vectors(c_bard, c_pin, dtype, name="Coriolis term")

        # Full RNEA
        bard.update_kinematics(model, data, q, qd)
        tau_bard = bard.rnea(model, data, qdd)[0].cpu().numpy()
        tau_pin = pin.rnea(pin_model_obj, pin_data, q_pin, qd_pin, qdd[0].cpu().numpy())
        compare_vectors(tau_bard, tau_pin, dtype, name="Full RNEA")

    def test_fixed_base_rnea_batched(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20
        data = bard.create_data(model, max_batch_size=batch_size)

        torch.manual_seed(1002)
        q_batch = torch.rand(batch_size, model.n_joints, device=device, dtype=dtype) * np.pi
        qd_batch = torch.randn(batch_size, model.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.randn(batch_size, model.n_joints, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q_batch, qd_batch)
        tau_bard_batch = bard.rnea(model, data, qdd_batch).cpu().numpy()

        for i in range(batch_size):
            tau_pin = pin.rnea(
                pin_model_obj,
                pin_data,
                q_batch[i].cpu().numpy(),
                qd_batch[i].cpu().numpy(),
                qdd_batch[i].cpu().numpy(),
            )
            compare_vectors(tau_bard_batch[i], tau_pin, dtype, name=f"RNEA batch[{i}]")

    # ========================================================================
    # Fixed-Base CRBA Tests
    # ========================================================================

    def test_fixed_base_crba(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_fixed

        torch.manual_seed(1010)
        q = torch.rand(1, model.n_joints, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q)
        M_bard = bard.crba(model, data)[0].cpu().numpy()
        M_pin = pin.crba(pin_model_obj, pin_data, q[0].cpu().numpy())
        compare_matrices(M_bard, M_pin, dtype, name="CRBA mass matrix")

    def test_fixed_base_crba_batched(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20
        data = bard.create_data(model, max_batch_size=batch_size)

        torch.manual_seed(1011)
        q_batch = torch.rand(batch_size, model.n_joints, device=device, dtype=dtype) * np.pi

        bard.update_kinematics(model, data, q_batch)
        M_bard_batch = bard.crba(model, data).cpu().numpy()

        for i in range(batch_size):
            M_pin = pin.crba(pin_model_obj, pin_data, q_batch[i].cpu().numpy())
            compare_matrices(M_bard_batch[i], M_pin, dtype, name=f"CRBA batch[{i}]")

    def test_fixed_base_rnea_crba_consistency(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)

        torch.manual_seed(1020)
        q = torch.rand(1, model.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, model.n_joints, device=device, dtype=dtype)

        zeros = torch.zeros_like(q)
        zero_gravity = torch.zeros(3, device=device, dtype=dtype)
        bard.update_kinematics(model, data, q, zeros)
        tau_rnea = bard.rnea(model, data, qdd, gravity=zero_gravity)[0]

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)[0]
        tau_crba = M @ qdd[0]

        tol = 1e-5 if dtype == torch.float32 else 1e-6
        assert torch.allclose(
            tau_rnea, tau_crba, atol=tol
        ), f"RNEA-CRBA consistency check failed: max diff = {(tau_rnea - tau_crba).abs().max():.3e}"

    # ========================================================================
    # Floating-Base RNEA Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_path):
        model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        return model, model.createData()

    def test_floating_base_rnea_full(self, urdf_path, pin_model_floating, dtype, device):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_floating

        torch.manual_seed(2000)
        translations = torch.randn(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(1, model.n_joints, device=device, dtype=dtype)
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qd = torch.randn(1, 6 + model.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, 6 + model.n_joints, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q, qd)
        tau_bard = bard.rnea(model, data, qdd)[0].cpu().numpy()

        q_pin = np.concatenate(
            [
                translations[0].cpu().numpy(),
                quats_wxyz[0, 1:].cpu().numpy(),
                quats_wxyz[0, 0:1].cpu().numpy(),
                q_joints[0].cpu().numpy(),
            ]
        )
        tau_pin = pin.rnea(
            pin_model_obj, pin_data, q_pin, qd[0].cpu().numpy(), qdd[0].cpu().numpy()
        )
        compare_vectors(tau_bard, tau_pin, dtype, name="Floating-base full RNEA")

    def test_floating_base_rnea_batched(self, urdf_path, pin_model_floating, dtype, device):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20
        data = bard.create_data(model, max_batch_size=batch_size)

        torch.manual_seed(2001)
        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, model.n_joints, device=device, dtype=dtype) * np.pi
        q_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qd_batch = torch.randn(batch_size, 6 + model.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.randn(batch_size, 6 + model.n_joints, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q_batch, qd_batch)
        tau_bard_batch = bard.rnea(model, data, qdd_batch).cpu().numpy()

        for i in range(batch_size):
            q_pin = np.concatenate(
                [
                    translations[i].cpu().numpy(),
                    quats_wxyz[i, 1:].cpu().numpy(),
                    quats_wxyz[i, 0:1].cpu().numpy(),
                    q_joints[i].cpu().numpy(),
                ]
            )
            tau_pin = pin.rnea(
                pin_model_obj,
                pin_data,
                q_pin,
                qd_batch[i].cpu().numpy(),
                qdd_batch[i].cpu().numpy(),
            )
            compare_vectors(tau_bard_batch[i], tau_pin, dtype, name=f"Floating RNEA batch[{i}]")

    # ========================================================================
    # Floating-Base CRBA Tests
    # ========================================================================

    def test_floating_base_crba(self, urdf_path, pin_model_floating, dtype, device):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_floating

        torch.manual_seed(2010)
        translations = torch.randn(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(1, model.n_joints, device=device, dtype=dtype)
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        bard.update_kinematics(model, data, q)
        M_bard = bard.crba(model, data)[0].cpu().numpy()

        q_pin = np.concatenate(
            [
                translations[0].cpu().numpy(),
                quats_wxyz[0, 1:].cpu().numpy(),
                quats_wxyz[0, 0:1].cpu().numpy(),
                q_joints[0].cpu().numpy(),
            ]
        )
        M_pin = pin.crba(pin_model_obj, pin_data, q_pin)
        compare_matrices(M_bard, M_pin, dtype, name="Floating-base CRBA")

    def test_floating_base_crba_batched(self, urdf_path, pin_model_floating, dtype, device):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20
        data = bard.create_data(model, max_batch_size=batch_size)

        torch.manual_seed(2011)
        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, model.n_joints, device=device, dtype=dtype) * np.pi
        q_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        bard.update_kinematics(model, data, q_batch)
        M_bard_batch = bard.crba(model, data).cpu().numpy()

        for i in range(batch_size):
            q_pin = np.concatenate(
                [
                    translations[i].cpu().numpy(),
                    quats_wxyz[i, 1:].cpu().numpy(),
                    quats_wxyz[i, 0:1].cpu().numpy(),
                    q_joints[i].cpu().numpy(),
                ]
            )
            M_pin = pin.crba(pin_model_obj, pin_data, q_pin)
            compare_matrices(M_bard_batch[i], M_pin, dtype, name=f"Floating CRBA batch[{i}]")

    def test_floating_base_rnea_crba_consistency(
        self, urdf_path, pin_model_floating, dtype, device
    ):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=1)

        torch.manual_seed(2020)
        translations = torch.randn(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(1, model.n_joints, device=device, dtype=dtype)
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qdd = torch.randn(1, 6 + model.n_joints, device=device, dtype=dtype)

        zeros = torch.zeros(1, 6 + model.n_joints, device=device, dtype=dtype)
        zero_gravity = torch.zeros(3, device=device, dtype=dtype)
        bard.update_kinematics(model, data, q, zeros)
        tau_rnea = bard.rnea(model, data, qdd, gravity=zero_gravity)[0]

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)[0]
        tau_crba = M @ qdd[0]

        tol = 1e-5 if dtype == torch.float32 else 1e-6
        assert torch.allclose(
            tau_rnea, tau_crba, atol=tol
        ), f"Floating-base RNEA-CRBA consistency failed: max diff = {(tau_rnea - tau_crba).abs().max():.3e}"

    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================

    def test_rnea_batch_size_validation(self, urdf_path, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=5)

        q_ok = torch.rand(5, model.n_joints, device=device, dtype=dtype)
        qd_ok = torch.randn(5, model.n_joints, device=device, dtype=dtype)
        qdd_ok = torch.randn(5, model.n_joints, device=device, dtype=dtype)
        bard.update_kinematics(model, data, q_ok, qd_ok)
        result = bard.rnea(model, data, qdd_ok)
        assert result.shape[0] == 5

        q_large = torch.rand(10, model.n_joints, device=device, dtype=dtype)
        qd_large = torch.randn(10, model.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            bard.update_kinematics(model, data, q_large, qd_large)

    def test_crba_batch_size_validation(self, urdf_path, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=5)

        q_ok = torch.rand(5, model.n_joints, device=device, dtype=dtype)
        bard.update_kinematics(model, data, q_ok)
        result = bard.crba(model, data)
        assert result.shape[0] == 5

        q_large = torch.rand(10, model.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            bard.update_kinematics(model, data, q_large)

    def test_crba_symmetry(self, urdf_path, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)

        torch.manual_seed(3000)
        q = torch.rand(1, model.n_joints, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)[0]

        tol = 1e-8 if dtype == torch.float64 else 1e-6
        assert torch.allclose(
            M, M.T, atol=tol
        ), f"Mass matrix is not symmetric: max asymmetry = {(M - M.T).abs().max():.3e}"

    def test_crba_positive_definite(self, urdf_path, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)

        torch.manual_seed(3001)
        q = torch.rand(1, model.n_joints, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)[0]

        eigenvalues = torch.linalg.eigvalsh(M)
        min_eigenvalue = eigenvalues.min().item()
        assert (
            min_eigenvalue > 0
        ), f"Mass matrix is not positive definite: min eigenvalue = {min_eigenvalue:.3e}"

    def test_rnea_with_compilation(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that RNEA works correctly with torch.compile enabled."""
        if device == "cpu":
            pytest.skip("torch.compile inductor backend requires CUDA on Windows")
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=5)

        torch.manual_seed(55)
        q = torch.rand(5, model.n_joints, device=device, dtype=dtype) * np.pi - np.pi / 2
        qd = torch.randn(5, model.nv, device=device, dtype=dtype)
        qdd = torch.randn(5, model.nv, device=device, dtype=dtype)

        bard.update_kinematics(model, data, q, qd)
        tau_ref = bard.rnea(model, data, qdd).clone()

        model.enable_compilation(True)
        data_compiled = bard.create_data(model, max_batch_size=5)
        bard.update_kinematics(model, data_compiled, q, qd)
        tau_compiled = bard.rnea(model, data_compiled, qdd)

        tol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(tau_ref, tau_compiled, atol=tol), (
            f"Compiled RNEA differs: max diff = " f"{(tau_ref - tau_compiled).abs().max():.3e}"
        )

    def test_crba_with_compilation(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that CRBA works correctly with torch.compile enabled."""
        if device == "cpu":
            pytest.skip("torch.compile inductor backend requires CUDA on Windows")
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=5)

        torch.manual_seed(66)
        q = torch.rand(5, model.n_joints, device=device, dtype=dtype) * np.pi - np.pi / 2

        bard.update_kinematics(model, data, q)
        M_ref = bard.crba(model, data).clone()

        model.enable_compilation(True)
        data_compiled = bard.create_data(model, max_batch_size=5)
        bard.update_kinematics(model, data_compiled, q)
        M_compiled = bard.crba(model, data_compiled)

        tol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(M_ref, M_compiled, atol=tol), (
            f"Compiled CRBA differs: max diff = " f"{(M_ref - M_compiled).abs().max():.3e}"
        )
