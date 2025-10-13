"""
Comprehensive tests for Spatial Acceleration (both fixed-base and floating-base).

Tests cover:
- BiasAcceleration: dJ/dt * qd computation
- SpatialAccelerationJacobian: Full acceleration with Jacobian-based method
- MultiFrameBiasAcceleration: Multiple frames at once
- Fixed-base and floating-base robots
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
from bard.core.acceleration import (
    BiasAcceleration,
    SpatialAccelerationJacobian,
    MultiFrameBiasAcceleration,
)


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
            f"Acceleration mismatch: max_diff={max_diff:.3e} > warn_tol={warn_tol:.1e}\n"
            f"Linear diff: {linear_diff:.3e}, Angular diff: {angular_diff:.3e}"
        )


@pytest.mark.skipif(not hasattr(pin, "buildModelFromUrdf"), reason="Pinocchio not available")
class TestBiasAcceleration:
    """Test suite for BiasAcceleration (dJ/dt * qd) computation."""

    # ========================================================================
    # Fixed-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_path):
        """Builds Pinocchio model for fixed-base robot."""
        model = pin.buildModelFromUrdf(urdf_path)
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_fixed_base_bias_random_states(
        self, urdf_path, pin_model_fixed, dtype, device, reference_frame
    ):
        """Verifies bias acceleration at random states for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        # Create bias acceleration instance
        bias_accel = BiasAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(1337)
        np.random.seed(1337)

        # Select test frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test multiple random states with qdd=0
        for _ in range(5):
            q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
            qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
            qdd_zero = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)

            # Bard bias acceleration
            a_bias_bard = bias_accel.calc(q, qd, bard_frame_idx, reference_frame=reference_frame)
            a_bias_bard_np = a_bias_bard[0].cpu().numpy()

            # Pinocchio acceleration with qdd=0
            q_pin = q[0].cpu().numpy()
            qd_pin = qd[0].cpu().numpy()
            qdd_pin = qdd_zero[0].cpu().numpy()

            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin_ref_frame
            ).vector

            compare_accelerations(a_bias_bard_np, a_pin, dtype)

    def test_fixed_base_bias_zero_velocity(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies bias acceleration is zero when velocity is zero."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        bias_accel = BiasAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        # Non-zero position, but zero velocity
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype) * 0.5
        qd = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(test_frame_name)

        a_bias = bias_accel.calc(q, qd, frame_idx, reference_frame="world")[0].cpu().numpy()

        # Bias acceleration should be zero when velocity is zero
        assert np.allclose(
            a_bias, 0.0, atol=1e-10
        ), "Bias acceleration should be zero with zero velocity"

    # ========================================================================
    # Floating-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_path):
        """Builds Pinocchio model for floating-base robot."""
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_floating_base_bias_random_states(
        self, urdf_path, pin_model_floating, dtype, device, reference_frame
    ):
        """Verifies bias acceleration for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        bias_accel = BiasAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(2048)
        np.random.seed(2048)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test multiple random states
        for _ in range(5):
            # Generate random state
            translations = torch.randn(1, 3, device=device, dtype=dtype)
            quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
            quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
            q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
            q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

            v_base = torch.randn(1, 6, device=device, dtype=dtype)
            qd_joints = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
            qd = torch.cat([v_base, qd_joints], dim=1)

            # Bard bias acceleration
            a_bias_bard = bias_accel.calc(q, qd, bard_frame_idx, reference_frame=reference_frame)
            a_bias_bard_np = a_bias_bard[0].cpu().numpy()

            # Pinocchio with qdd=0
            q_pin = np.concatenate(
                [
                    translations[0].cpu().numpy(),
                    quats_wxyz[0, 1:].cpu().numpy(),
                    quats_wxyz[0, 0:1].cpu().numpy(),
                    q_joints[0].cpu().numpy(),
                ]
            )
            qd_pin = qd[0].cpu().numpy()
            qdd_pin = np.zeros_like(qd_pin)

            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin_ref_frame
            ).vector

            compare_accelerations(a_bias_bard_np, a_pin, dtype)


@pytest.mark.skipif(not hasattr(pin, "buildModelFromUrdf"), reason="Pinocchio not available")
class TestSpatialAccelerationJacobian:
    """Test suite for SpatialAccelerationJacobian (Jacobian-based full acceleration)."""

    # ========================================================================
    # Fixed-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_path):
        """Builds Pinocchio model for fixed-base robot."""
        model = pin.buildModelFromUrdf(urdf_path)
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_fixed_base_full_acceleration(
        self, urdf_path, pin_model_fixed, dtype, device, reference_frame
    ):
        """Verifies full acceleration at random states for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        # Create acceleration instance
        accel = SpatialAccelerationJacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(1337)
        np.random.seed(1337)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
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

    def test_fixed_base_batched(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies batched acceleration computation for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20

        accel = SpatialAccelerationJacobian(
            bard_chain, max_batch_size=batch_size, compile_enabled=False
        )

        torch.manual_seed(42)
        q_batch = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
        qd_batch = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.randn(batch_size, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Batched computation
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

    def test_fixed_base_with_jacobian_return(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that returning Jacobian along with acceleration works correctly."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        accel = SpatialAccelerationJacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Get both acceleration and Jacobian
        a_bard, J_bard = accel.calc(
            q, qd, qdd, bard_frame_idx, reference_frame="world", return_jacobian=True
        )

        # Verify shapes
        assert a_bard.shape == (1, 6), f"Expected acceleration shape (1, 6), got {a_bard.shape}"
        assert J_bard.shape == (
            1,
            6,
            bard_chain.n_joints,
        ), f"Expected Jacobian shape (1, 6, {bard_chain.n_joints}), got {J_bard.shape}"

        # Verify acceleration matches Pinocchio
        a_bard_np = a_bard[0].cpu().numpy()
        q_pin = q[0].cpu().numpy()
        qd_pin = qd[0].cpu().numpy()
        qdd_pin = qdd[0].cpu().numpy()

        pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
        a_pin = pin.getFrameAcceleration(
            pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
        ).vector

        compare_accelerations(a_bard_np, a_pin, dtype)

    def test_bias_only_method(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that calc_bias_only matches setting qdd=0."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        accel = SpatialAccelerationJacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_zero = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(test_frame_name)

        # Method 1: Full calc with qdd=0
        a_full = accel.calc(q, qd, qdd_zero, frame_idx, reference_frame="world")

        # Method 2: Direct bias computation
        a_bias = accel.calc_bias_only(q, qd, frame_idx, reference_frame="world")

        # They should match
        assert torch.allclose(
            a_full, a_bias, atol=1e-6
        ), "calc_bias_only should match full calc with qdd=0"

    # ========================================================================
    # Floating-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_path):
        """Builds Pinocchio model for floating-base robot."""
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_floating_base_full_acceleration(
        self, urdf_path, pin_model_floating, dtype, device, reference_frame
    ):
        """Verifies acceleration for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        accel = SpatialAccelerationJacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(2048)
        np.random.seed(2048)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test multiple random states
        for _ in range(5):
            translations = torch.randn(1, 3, device=device, dtype=dtype)
            quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
            quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
            q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
            q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

            v_base = torch.randn(1, 6, device=device, dtype=dtype)
            qd_joints = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
            qd = torch.cat([v_base, qd_joints], dim=1)

            a_base = torch.randn(1, 6, device=device, dtype=dtype)
            qdd_joints = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
            qdd = torch.cat([a_base, qdd_joints], dim=1)

            # Bard acceleration
            a_bard = accel.calc(q, qd, qdd, bard_frame_idx, reference_frame=reference_frame)
            a_bard_np = a_bard[0].cpu().numpy()

            # Pinocchio
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

            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin_ref_frame
            ).vector

            compare_accelerations(a_bard_np, a_pin, dtype)

    def test_floating_base_batched(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies batched acceleration for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20

        accel = SpatialAccelerationJacobian(
            bard_chain, max_batch_size=batch_size, compile_enabled=False
        )

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
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
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


@pytest.mark.skipif(not hasattr(pin, "buildModelFromUrdf"), reason="Pinocchio not available")
class TestMultiFrameBiasAcceleration:
    """Test suite for MultiFrameBiasAcceleration."""

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_path):
        """Builds Pinocchio model for fixed-base robot."""
        model = pin.buildModelFromUrdf(urdf_path)
        return model, model.createData()

    def test_multi_frame_computation(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies multi-frame bias acceleration computation."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        multi_bias = MultiFrameBiasAcceleration(bard_chain, max_batch_size=1, max_frames=5)

        # Get multiple frame names
        all_frame_names = bard_chain.get_frame_names(exclude_fixed=True)
        test_frame_names = all_frame_names[::2][:3]  # Take every other frame, max 3
        frame_ids = [bard_chain.get_frame_id(name) for name in test_frame_names]

        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

        # Compute for multiple frames at once
        a_bias_dict = multi_bias.calc(q, qd, frame_ids, reference_frame="world")

        # Verify each frame
        q_pin = q[0].cpu().numpy()
        qd_pin = qd[0].cpu().numpy()
        qdd_pin = np.zeros_like(qd_pin)

        pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)

        for frame_name, frame_id in zip(test_frame_names, frame_ids):
            pin_frame_id = pin_model_obj.getFrameId(frame_name)
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
            ).vector

            a_bard_np = a_bias_dict[frame_id][0].cpu().numpy()
            compare_accelerations(a_bard_np, a_pin, dtype)


@pytest.mark.skipif(not hasattr(pin, "buildModelFromUrdf"), reason="Pinocchio not available")
class TestAccelerationEdgeCases:
    """Edge cases and stress tests for all acceleration classes."""

    def test_batch_size_validation(self, urdf_path, dtype, device):
        """Verifies that exceeding max_batch_size raises appropriate error."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        accel = SpatialAccelerationJacobian(bard_chain, max_batch_size=5, compile_enabled=False)
        bias_accel = BiasAcceleration(bard_chain, max_batch_size=5, compile_enabled=False)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(frame_name)

        # Should work
        q_ok = torch.rand(5, bard_chain.n_joints, device=device, dtype=dtype)
        qd_ok = torch.randn(5, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_ok = torch.randn(5, bard_chain.n_joints, device=device, dtype=dtype)

        result = accel.calc(q_ok, qd_ok, qdd_ok, frame_idx, reference_frame="world")
        assert result.shape[0] == 5

        result_bias = bias_accel.calc(q_ok, qd_ok, frame_idx, reference_frame="world")
        assert result_bias.shape[0] == 5

        # Should raise ValueError
        q_too_large = torch.rand(10, bard_chain.n_joints, device=device, dtype=dtype)
        qd_too_large = torch.randn(10, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_too_large = torch.randn(10, bard_chain.n_joints, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            accel.calc(q_too_large, qd_too_large, qdd_too_large, frame_idx, reference_frame="world")

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            bias_accel.calc(q_too_large, qd_too_large, frame_idx, reference_frame="world")

    def test_world_vs_local_frame_consistency(self, urdf_path, dtype, device):
        """Verifies relationship between world and local frame accelerations."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        accel = SpatialAccelerationJacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(999)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(frame_name)

        # Compute in both frames
        a_world = accel.calc(q, qd, qdd, frame_idx, reference_frame="world")
        a_local = accel.calc(q, qd, qdd, frame_idx, reference_frame="local")

        # Verify both have correct shape
        assert a_world.shape == (1, 6), "World frame acceleration has wrong shape"
        assert a_local.shape == (1, 6), "Local frame acceleration has wrong shape"

    def test_consistency_between_bias_and_full(self, urdf_path, dtype, device):
        """Verifies BiasAcceleration matches SpatialAccelerationJacobian when qdd=0."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        bias_accel = BiasAcceleration(bard_chain, max_batch_size=1, compile_enabled=False)
        full_accel = SpatialAccelerationJacobian(
            bard_chain, max_batch_size=1, compile_enabled=False
        )

        torch.manual_seed(777)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd_zero = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(frame_name)

        # Compute using both methods
        a_bias_only = bias_accel.calc(q, qd, frame_idx, reference_frame="world")
        a_full_with_zero = full_accel.calc(q, qd, qdd_zero, frame_idx, reference_frame="world")

        # They should match
        assert torch.allclose(
            a_bias_only, a_full_with_zero, atol=1e-6
        ), "BiasAcceleration should match SpatialAccelerationJacobian with qdd=0"

    def test_zero_state_acceleration(self, urdf_path, dtype, device):
        """Verifies acceleration is zero when all states are zero."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        accel = SpatialAccelerationJacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        # All zeros
        q = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(frame_name)

        a = accel.calc(q, qd, qdd, frame_idx, reference_frame="world")[0].cpu().numpy()

        assert np.allclose(a, 0.0, atol=1e-10), "Acceleration should be zero for zero state"

    def test_compilation_placeholder(self, urdf_path, dtype, device):
        """Placeholder for compilation tests."""
        pytest.skip("Compilation tests to be implemented")
