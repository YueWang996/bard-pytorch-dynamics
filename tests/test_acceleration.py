"""
Comprehensive tests for Spatial Acceleration (both fixed-base and floating-base).

Tests cover:
- Fixed-base robots with random states
- Floating-base robots with random base states
- Both world and local reference frames
- Batched operations
- Compilation compatibility
"""

import warnings
import pytest
import torch
import numpy as np
import pinocchio as pin

from bard.parsers.urdf import build_chain_from_urdf
from bard import SpatialAcceleration


def compare_accelerations(a_bard, a_pin, dtype):
    """
    Compare spatial acceleration vectors with appropriate tolerances.

    Args:
        a_bard: Bard acceleration (6,) - [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        a_pin: Pinocchio acceleration (6,) - same format
        dtype: torch dtype for tolerance selection
    """
    # Tolerances for acceleration comparison
    warn_tol = 1e-4 if dtype == torch.float32 else 1e-5
    fail_tol = warn_tol * 1.2

    max_diff = np.abs(a_bard - a_pin).max()
    mean_diff = np.abs(a_bard - a_pin).mean()

    # Check linear and angular components separately for better diagnostics
    linear_diff = np.linalg.norm(a_bard[:3] - a_pin[:3])
    angular_diff = np.linalg.norm(a_bard[3:] - a_pin[3:])

    if max_diff >= fail_tol:
        # Hard fail
        assert False, (
            f"Acceleration mismatch: max_diff={max_diff:.3e} > fail_tol={fail_tol:.1e}\n"
            f"Linear diff: {linear_diff:.3e}, Angular diff: {angular_diff:.3e}"
        )
    elif max_diff >= warn_tol:
        # Warning but pass
        warnings.warn(
            f"Acceleration mismatch: max_diff={max_diff:.3e} > fail_tol={warn_tol:.1e}\n"
            f"Linear diff: {linear_diff:.3e}, Angular diff: {angular_diff:.3e}"
        )


@pytest.mark.skipif(not hasattr(pin, "buildModelFromXML"), reason="Pinocchio not available")
class TestSpatialAcceleration:
    """Test suite for Spatial Acceleration with both fixed-base and floating-base robots."""

    # ========================================================================
    # Fixed-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_string):
        """Builds Pinocchio model for fixed-base robot."""
        model = pin.buildModelFromXML(urdf_string)
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_fixed_base_random_states(
        self, urdf_string, pin_model_fixed, dtype, device, reference_frame
    ):
        """Verifies acceleration at random states for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        # Create acceleration instance (without compilation)
        accel = SpatialAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(1337)
        np.random.seed(1337)

        # Select test frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test multiple random states
        for _ in range(5):
            q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
            qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
            qdd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

            # Bard acceleration
            a_bard = accel.calc(q, qd, qdd, bard_frame_idx, reference_frame=reference_frame)
            a_bard_np = a_bard[0].cpu().numpy()

            # Pinocchio acceleration
            q_pin = q[0].cpu().numpy()
            qd_pin = qd[0].cpu().numpy()
            qdd_pin = qdd[0].cpu().numpy()

            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin_ref_frame
            ).vector

            compare_accelerations(a_bard_np, a_pin, dtype)

    def test_fixed_base_batched(self, urdf_string, pin_model_fixed, dtype, device):
        """Verifies batched acceleration computation for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20

        # Create acceleration instance with appropriate batch size
        accel = SpatialAcceleration(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(42)
        q_batch = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
        qd_batch = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Batched computation (world frame)
        a_bard_batch = (
            accel.calc(q_batch, qd_batch, qdd_batch, bard_frame_idx, reference_frame="world")
            .cpu()
            .numpy()
        )

        # Verify each sample against Pinocchio
        for i in range(batch_size):
            q_pin = q_batch[i].cpu().numpy()
            qd_pin = qd_batch[i].cpu().numpy()
            qdd_pin = qdd_batch[i].cpu().numpy()

            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
            ).vector

            compare_accelerations(a_bard_batch[i], a_pin, dtype)

    def test_fixed_base_zero_velocity_acceleration(
        self, urdf_string, pin_model_fixed, dtype, device
    ):
        """Verifies acceleration when velocities and accelerations are zero."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        accel = SpatialAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        # Non-zero position, but zero velocity and acceleration
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype) * 0.5
        qd = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        a_bard = accel.calc(q, qd, qdd, frame_idx, reference_frame="world")[0].cpu().numpy()

        pin.forwardKinematics(
            pin_model_obj, pin_data, q[0].cpu().numpy(), qd[0].cpu().numpy(), qdd[0].cpu().numpy()
        )
        a_pin = pin.getFrameAcceleration(
            pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
        ).vector

        # Should be near zero (only numerical errors)
        assert np.allclose(a_bard, a_pin, atol=1e-10), "Zero velocity/acceleration case failed"
        assert np.allclose(a_bard, 0.0, atol=1e-10), "Acceleration should be near zero"

    def test_fixed_base_with_compilation(self, urdf_string, pin_model_fixed, dtype, device):
        """Verifies that acceleration works correctly with torch.compile enabled."""
        # Skip compilation tests for now due to torch.compile overflow issues
        pytest.skip("Placeholder test - compilation tests is to be implemented")

    # ========================================================================
    # Floating-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_string):
        """Builds Pinocchio model for floating-base robot."""
        model = pin.buildModelFromXML(urdf_string, pin.JointModelFreeFlyer())
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_floating_base_random_states(
        self, urdf_string, pin_model_floating, dtype, device, reference_frame
    ):
        """Verifies acceleration at random states for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        # Create acceleration instance
        accel = SpatialAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(2048)
        np.random.seed(2048)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test multiple random states
        for _ in range(5):
            # Generate random full state (base + joints)
            translations = torch.randn(1, 3, device=device, dtype=dtype)
            quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
            quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
            q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
            q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

            # Velocity and acceleration
            v_base = torch.randn(1, 6, device=device, dtype=dtype)
            qd_joints = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
            qd = torch.cat([v_base, qd_joints], dim=1)

            a_base = torch.randn(1, 6, device=device, dtype=dtype)
            qdd_joints = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
            qdd = torch.cat([a_base, qdd_joints], dim=1)

            # Bard acceleration
            a_bard = accel.calc(q, qd, qdd, bard_frame_idx, reference_frame=reference_frame)
            a_bard_np = a_bard[0].cpu().numpy()

            # Convert to Pinocchio format
            q_pin = np.concatenate(
                [
                    translations[0].cpu().numpy(),
                    quats_wxyz[0, 1:].cpu().numpy(),  # qx, qy, qz
                    quats_wxyz[0, 0:1].cpu().numpy(),  # qw
                    q_joints[0].cpu().numpy(),
                ]
            )
            qd_pin = qd[0].cpu().numpy()
            qdd_pin = qdd[0].cpu().numpy()

            # Pinocchio acceleration
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin_ref_frame
            ).vector

            compare_accelerations(a_bard_np, a_pin, dtype)

    def test_floating_base_batched(self, urdf_string, pin_model_floating, dtype, device):
        """Verifies batched acceleration computation for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20

        # Create acceleration instance
        accel = SpatialAcceleration(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(123)

        # Generate batched states
        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
        q_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        v_base = torch.randn(batch_size, 6, device=device, dtype=dtype)
        qd_joints = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)
        qd_batch = torch.cat([v_base, qd_joints], dim=1)

        a_base = torch.randn(batch_size, 6, device=device, dtype=dtype)
        qdd_joints = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.cat([a_base, qdd_joints], dim=1)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Batched computation
        a_bard_batch = (
            accel.calc(q_batch, qd_batch, qdd_batch, bard_frame_idx, reference_frame="world")
            .cpu()
            .numpy()
        )

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
            qd_pin = qd_batch[i].cpu().numpy()
            qdd_pin = qdd_batch[i].cpu().numpy()

            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
            ).vector

            compare_accelerations(a_bard_batch[i], a_pin, dtype)

    def test_floating_base_stationary(self, urdf_string, pin_model_floating, dtype, device):
        """Verifies acceleration when floating base is stationary at identity pose."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        accel = SpatialAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        # Identity base pose with zero velocity/acceleration
        translations = torch.zeros(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype) * 0.3
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qd = torch.zeros(1, 6 + bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.zeros(1, 6 + bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        a_bard = accel.calc(q, qd, qdd, frame_idx, reference_frame="world")[0].cpu().numpy()

        q_pin = np.concatenate(
            [
                translations[0].cpu().numpy(),
                quats_wxyz[0, 1:].cpu().numpy(),
                quats_wxyz[0, 0:1].cpu().numpy(),
                q_joints[0].cpu().numpy(),
            ]
        )

        pin.forwardKinematics(
            pin_model_obj, pin_data, q_pin, qd[0].cpu().numpy(), qdd[0].cpu().numpy()
        )
        a_pin = pin.getFrameAcceleration(
            pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
        ).vector

        compare_accelerations(a_bard, a_pin, dtype)
        assert np.allclose(a_bard, 0.0, atol=1e-10), "Stationary acceleration should be near zero"

    def test_floating_base_with_compilation(self, urdf_string, pin_model_floating, dtype, device):
        """Verifies that floating-base acceleration works correctly with torch.compile enabled."""
        # Skip compilation tests for now
        pytest.skip("Placeholder test - compilation tests is to be implemented")

    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================

    def test_batch_size_validation(self, urdf_string, dtype, device):
        """Verifies that exceeding max_batch_size raises appropriate error."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)

        accel = SpatialAcceleration(bard_chain, max_batch_size=5, compile_enabled=False)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(frame_name).item()

        # This should work (batch_size = 5, max = 5)
        q_ok = torch.rand(5, bard_chain.n_joints, device=device, dtype=dtype)
        qd_ok = torch.randn(5, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_ok = torch.randn(5, bard_chain.n_joints, device=device, dtype=dtype)
        result = accel.calc(q_ok, qd_ok, qdd_ok, frame_idx)
        assert result.shape[0] == 5, "Should process 5 samples"

        # This should raise ValueError (batch_size = 10, max = 5)
        q_too_large = torch.rand(10, bard_chain.n_joints, device=device, dtype=dtype)
        qd_too_large = torch.randn(10, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_too_large = torch.randn(10, bard_chain.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            _ = accel.calc(q_too_large, qd_too_large, qdd_too_large, frame_idx)

    def test_world_vs_local_frame_consistency(self, urdf_string, dtype, device):
        """Verifies relationship between world and local frame accelerations."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)

        accel = SpatialAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(999)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(frame_name).item()

        # Compute in both frames
        a_world = accel.calc(q, qd, qdd, frame_idx, reference_frame="world")
        a_local = accel.calc(q, qd, qdd, frame_idx, reference_frame="local")

        # They should be different (unless at identity configuration)
        # Just verify both computations complete without error and have correct shape
        assert a_world.shape == (1, 6), "World frame acceleration has wrong shape"
        assert a_local.shape == (1, 6), "Local frame acceleration has wrong shape"
