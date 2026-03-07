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
import pinocchio as pin

from bard import build_chain_from_urdf, RobotDynamics


def compare_vectors(tau_bard, tau_pin, dtype, name="vector"):
    """
    Compare force/torque vectors with appropriate tolerances.

    Args:
        tau_bard: Bard vector
        tau_pin: Pinocchio vector
        dtype: torch dtype for tolerance selection
        name: Name for error messages
    """
    warn_tol = 5e-4 if dtype == torch.float32 else 5e-6
    # Fail tolerance (20% buffer above warning)
    fail_tol = warn_tol * 1.2

    max_diff = np.abs(tau_bard - tau_pin).max()

    if max_diff >= fail_tol:
        # Hard fail
        assert False, f"{name} mismatch: max_diff={max_diff:.3e} > fail_tol={fail_tol:.1e}"
    elif max_diff >= warn_tol:
        # Warning but pass
        warnings.warn(
            f"{name} close to tolerance: max_diff={max_diff:.3e}, warn_tol={warn_tol:.1e}"
        )


def compare_matrices(M_bard, M_pin, dtype, name="matrix"):
    """
    Compare mass matrices with appropriate tolerances.

    Args:
        M_bard: Bard matrix
        M_pin: Pinocchio matrix
        dtype: torch dtype for tolerance selection
        name: Name for error messages
    """
    warn_tol = 1e-4 if dtype == torch.float32 else 1e-6
    fail_tol = warn_tol * 1.2

    if M_pin.shape != M_bard.shape:
        raise AssertionError(f"Shape mismatch: Bard {M_bard.shape} vs Pinocchio {M_pin.shape}")

    max_diff = np.abs(M_bard - M_pin).max()
    mean_diff = np.abs(M_bard - M_pin).mean()

    if max_diff >= fail_tol:
        # Hard fail
        assert False, f"{name} mismatch: max_diff={max_diff:.3e} > fail_tol={fail_tol:.1e}"
    elif max_diff >= warn_tol:
        # Warning but pass
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
        """Builds Pinocchio model for fixed-base robot."""
        model = pin.buildModelFromUrdf(urdf_path)
        return model, model.createData()

    def test_fixed_base_rnea_full(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies full RNEA (gravity + coriolis + inertia) for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(1000)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

        # Bard RNEA
        state = rd.update_kinematics(q, qd)
        tau_bard = rd.rnea(qdd, state)[0].cpu().numpy()

        # Pinocchio RNEA
        q_pin = q[0].cpu().numpy()
        qd_pin = qd[0].cpu().numpy()
        qdd_pin = qdd[0].cpu().numpy()
        tau_pin = pin.rnea(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)

        compare_vectors(tau_bard, tau_pin, dtype, name="Full RNEA")

    def test_fixed_base_rnea_components(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies RNEA component separation (gravity, coriolis, full) for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(1001)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

        q_pin = q[0].cpu().numpy()
        qd_pin = qd[0].cpu().numpy()
        qdd_pin = qdd[0].cpu().numpy()

        # Gravity term: RNEA(q, 0, 0)
        zeros = torch.zeros_like(q)
        state_g = rd.update_kinematics(q, zeros)
        g_bard = rd.rnea(zeros, state_g)[0].cpu().numpy()
        g_pin = pin.computeGeneralizedGravity(pin_model_obj, pin_data, q_pin)
        compare_vectors(g_bard, g_pin, dtype, name="Gravity term")

        # Coriolis term: RNEA(q, qd, 0, g=0)
        zero_gravity = torch.zeros(3, device=device, dtype=dtype)
        state_c = rd.update_kinematics(q, qd)
        c_bard = rd.rnea(zeros, state_c, gravity=zero_gravity)[0].cpu().numpy()
        c_pin = pin.computeCoriolisMatrix(pin_model_obj, pin_data, q_pin, qd_pin) @ qd_pin
        compare_vectors(c_bard, c_pin, dtype, name="Coriolis term")

        # Full RNEA
        state_full = rd.update_kinematics(q, qd)
        tau_bard = rd.rnea(qdd, state_full)[0].cpu().numpy()
        tau_pin = pin.rnea(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
        compare_vectors(tau_bard, tau_pin, dtype, name="Full RNEA")

    def test_fixed_base_rnea_batched(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies batched RNEA computation for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20

        rd = RobotDynamics(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(1002)
        q_batch = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
        qd_batch = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)

        # Batched computation
        state = rd.update_kinematics(q_batch, qd_batch)
        tau_bard_batch = rd.rnea(qdd_batch, state).cpu().numpy()

        # Verify each sample
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
        """Verifies CRBA mass matrix computation for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(1010)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)

        # Bard CRBA
        state = rd.update_kinematics(q)
        M_bard = rd.crba(state)[0].cpu().numpy()

        # Pinocchio CRBA
        M_pin = pin.crba(pin_model_obj, pin_data, q[0].cpu().numpy())

        compare_matrices(M_bard, M_pin, dtype, name="CRBA mass matrix")

    def test_fixed_base_crba_batched(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies batched CRBA computation for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20

        rd = RobotDynamics(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(1011)
        q_batch = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi

        # Batched computation
        state = rd.update_kinematics(q_batch)
        M_bard_batch = rd.crba(state).cpu().numpy()

        # Verify each sample
        for i in range(batch_size):
            M_pin = pin.crba(pin_model_obj, pin_data, q_batch[i].cpu().numpy())
            compare_matrices(M_bard_batch[i], M_pin, dtype, name=f"CRBA batch[{i}]")

    def test_fixed_base_rnea_crba_consistency(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies RNEA-CRBA consistency: M*qdd == RNEA(q, 0, qdd, g=0)."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(1020)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

        # RNEA with zero velocity and gravity
        zeros = torch.zeros_like(q)
        zero_gravity = torch.zeros(3, device=device, dtype=dtype)
        state_rnea = rd.update_kinematics(q, zeros)
        tau_rnea = rd.rnea(qdd, state_rnea, gravity=zero_gravity)[0]

        # CRBA
        state_crba = rd.update_kinematics(q)
        M = rd.crba(state_crba)[0]
        tau_crba = M @ qdd[0]

        # Should match
        tol = 1e-5 if dtype == torch.float32 else 1e-6
        assert torch.allclose(
            tau_rnea, tau_crba, atol=tol
        ), f"RNEA-CRBA consistency check failed: max diff = {(tau_rnea - tau_crba).abs().max():.3e}"

    # ========================================================================
    # Floating-Base RNEA Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_path):
        """Builds Pinocchio model for floating-base robot."""
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        return model, model.createData()

    def test_floating_base_rnea_full(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies full RNEA for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(2000)

        # Generate random state
        translations = torch.randn(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qd = torch.randn(1, 6 + bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, 6 + bard_chain.n_joints, device=device, dtype=dtype)

        # Bard RNEA
        state = rd.update_kinematics(q, qd)
        tau_bard = rd.rnea(qdd, state)[0].cpu().numpy()

        # Convert to Pinocchio format
        q_pin = np.concatenate(
            [
                translations[0].cpu().numpy(),
                quats_wxyz[0, 1:].cpu().numpy(),
                quats_wxyz[0, 0:1].cpu().numpy(),
                q_joints[0].cpu().numpy(),
            ]
        )
        qd_pin = qd[0].cpu().numpy()
        qdd_pin = qdd[0].cpu().numpy()

        # Pinocchio RNEA
        tau_pin = pin.rnea(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)

        compare_vectors(tau_bard, tau_pin, dtype, name="Floating-base full RNEA")

    def test_floating_base_rnea_batched(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies batched RNEA for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20

        rd = RobotDynamics(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(2001)

        # Generate batched states
        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
        q_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qd_batch = torch.randn(batch_size, 6 + bard_chain.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.randn(batch_size, 6 + bard_chain.n_joints, device=device, dtype=dtype)

        # Batched computation
        state = rd.update_kinematics(q_batch, qd_batch)
        tau_bard_batch = rd.rnea(qdd_batch, state).cpu().numpy()

        # Verify each sample
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
        """Verifies CRBA for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(2010)

        translations = torch.randn(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        # Bard CRBA
        state = rd.update_kinematics(q)
        M_bard = rd.crba(state)[0].cpu().numpy()

        # Pinocchio CRBA
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
        """Verifies batched CRBA for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20

        rd = RobotDynamics(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(2011)

        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
        q_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        # Batched computation
        state = rd.update_kinematics(q_batch)
        M_bard_batch = rd.crba(state).cpu().numpy()

        # Verify each sample
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
        """Verifies RNEA-CRBA consistency for floating-base: M*qdd == RNEA(q, 0, qdd, g=0)."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(2020)

        translations = torch.randn(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qdd = torch.randn(1, 6 + bard_chain.n_joints, device=device, dtype=dtype)

        # RNEA with zero velocity and gravity
        zeros = torch.zeros(1, 6 + bard_chain.n_joints, device=device, dtype=dtype)
        zero_gravity = torch.zeros(3, device=device, dtype=dtype)
        state_rnea = rd.update_kinematics(q, zeros)
        tau_rnea = rd.rnea(qdd, state_rnea, gravity=zero_gravity)[0]

        # CRBA
        state_crba = rd.update_kinematics(q)
        M = rd.crba(state_crba)[0]
        tau_crba = M @ qdd[0]

        # Should match
        tol = 1e-5 if dtype == torch.float32 else 1e-6
        assert torch.allclose(
            tau_rnea, tau_crba, atol=tol
        ), f"Floating-base RNEA-CRBA consistency failed: max diff = {(tau_rnea - tau_crba).abs().max():.3e}"

    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================

    def test_rnea_batch_size_validation(self, urdf_path, dtype, device):
        """Verifies that RNEA raises error when exceeding max_batch_size."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        rd = RobotDynamics(bard_chain, max_batch_size=5, compile_enabled=False)

        # Should work
        q_ok = torch.rand(5, bard_chain.n_joints, device=device, dtype=dtype)
        qd_ok = torch.randn(5, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_ok = torch.randn(5, bard_chain.n_joints, device=device, dtype=dtype)
        state = rd.update_kinematics(q_ok, qd_ok)
        result = rd.rnea(qdd_ok, state)
        assert result.shape[0] == 5, "Should process 5 samples"

        # Should raise
        q_large = torch.rand(10, bard_chain.n_joints, device=device, dtype=dtype)
        qd_large = torch.randn(10, bard_chain.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            _ = rd.update_kinematics(q_large, qd_large)

    def test_crba_batch_size_validation(self, urdf_path, dtype, device):
        """Verifies that CRBA raises error when exceeding max_batch_size."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        rd = RobotDynamics(bard_chain, max_batch_size=5, compile_enabled=False)

        # Should work
        q_ok = torch.rand(5, bard_chain.n_joints, device=device, dtype=dtype)
        state = rd.update_kinematics(q_ok)
        result = rd.crba(state)
        assert result.shape[0] == 5, "Should process 5 samples"

        # Should raise
        q_large = torch.rand(10, bard_chain.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            _ = rd.update_kinematics(q_large)

    def test_crba_symmetry(self, urdf_path, dtype, device):
        """Verifies that mass matrix is symmetric."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(3000)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)

        state = rd.update_kinematics(q)
        M = rd.crba(state)[0]

        # Check symmetry
        tol = 1e-8 if dtype == torch.float64 else 1e-6
        assert torch.allclose(
            M, M.T, atol=tol
        ), f"Mass matrix is not symmetric: max asymmetry = {(M - M.T).abs().max():.3e}"

    def test_crba_positive_definite(self, urdf_path, dtype, device):
        """Verifies that mass matrix is positive definite."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        rd = RobotDynamics(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(3001)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)

        state = rd.update_kinematics(q)
        M = rd.crba(state)[0]

        # Check positive definiteness via eigenvalues
        eigenvalues = torch.linalg.eigvalsh(M)
        min_eigenvalue = eigenvalues.min().item()

        assert (
            min_eigenvalue > 0
        ), f"Mass matrix is not positive definite: min eigenvalue = {min_eigenvalue:.3e}"

    def test_rnea_with_compilation(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that RNEA works with torch.compile enabled."""
        # Skip compilation tests for now
        pytest.skip("Placeholder test - compilation tests is to be implemented")

    def test_crba_with_compilation(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that CRBA works with torch.compile enabled."""
        # Skip compilation tests for now
        pytest.skip("Placeholder test - compilation tests is to be implemented")
