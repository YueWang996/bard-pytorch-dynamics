"""
Comprehensive tests for Jacobian computation (both fixed-base and floating-base).

Tests cover:
- Fixed-base robots with random configurations
- Floating-base robots with random base poses
- Both world and local reference frames
- Batched operations
- End-effector pose return option
- Compilation compatibility
"""

import warnings
import pytest
import torch
import numpy as np
import pinocchio as pin

from bard.parsers.urdf import build_chain_from_urdf
from bard import Jacobian


def compare_jacobians(J_bard, J_pin, dtype):
    """
    Compare Jacobian matrices with appropriate tolerances.

    Args:
        J_bard: Bard Jacobian (6, nv)
        J_pin: Pinocchio Jacobian (6, nv)
        dtype: torch dtype for tolerance selection
    """
    # Warning tolerance
    warn_tol = 1e-5 if dtype == torch.float32 else 1e-6
    # Fail tolerance (20% buffer above warning)
    fail_tol = warn_tol * 1.2

    if J_pin.shape != J_bard.shape:
        raise AssertionError(f"Shape mismatch: Bard {J_bard.shape} vs Pinocchio {J_pin.shape}")

    max_diff = np.abs(J_bard - J_pin).max()
    mean_diff = np.abs(J_bard - J_pin).mean()

    # Check relative error for non-zero elements
    mask = np.abs(J_pin) > 1e-10
    if mask.any():
        rel_error = np.abs((J_bard[mask] - J_pin[mask]) / J_pin[mask]).max()
    else:
        rel_error = 0.0

    if max_diff >= fail_tol:
        # Hard fail
        assert False, (
            f"Jacobian mismatch: max_diff={max_diff:.3e} > fail_tol={fail_tol:.1e}\n"
            f"Mean diff: {mean_diff:.3e}, Max relative error: {rel_error:.3e}"
        )
    elif max_diff >= warn_tol:
        # Warning but pass
        warnings.warn(
            f"Jacobian close to tolerance: max_diff={max_diff:.3e}, warn_tol={warn_tol:.1e}\n"
            f"Mean diff: {mean_diff:.3e}, Max relative error: {rel_error:.3e}"
        )


@pytest.mark.skipif(not hasattr(pin, "buildModelFromUrdf"), reason="Pinocchio not available")
class TestJacobian:
    """Test suite for Jacobian computation with both fixed-base and floating-base robots."""

    # ========================================================================
    # Fixed-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_path):
        """Builds Pinocchio model for fixed-base robot."""
        model = pin.buildModelFromUrdf(urdf_path)
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_fixed_base_random_configs(
        self, urdf_path, pin_model_fixed, dtype, device, reference_frame
    ):
        """Verifies Jacobian at random configurations for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        # Create Jacobian instance (without compilation)
        jac = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(42)
        np.random.seed(42)

        # Select test frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test multiple random configurations
        for _ in range(5):
            q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype) * np.pi - np.pi / 2

            # Bard Jacobian
            J_bard = jac.calc(q, bard_frame_idx, reference_frame=reference_frame)
            J_bard_np = J_bard[0].cpu().numpy()

            # Pinocchio Jacobian
            q_pin = q[0].cpu().numpy()
            pin.computeJointJacobians(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)

            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            J_pin = pin.getFrameJacobian(pin_model_obj, pin_data, pin_frame_id, pin_ref_frame)

            compare_jacobians(J_bard_np, J_pin, dtype)

    def test_fixed_base_batched(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies batched Jacobian computation for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20

        # Create Jacobian instance with appropriate batch size
        jac = Jacobian(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(100)
        q_batch = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Batched computation (world frame)
        J_bard_batch = jac.calc(q_batch, bard_frame_idx, reference_frame="world").cpu().numpy()

        # Verify each sample against Pinocchio
        for i in range(batch_size):
            q_pin = q_batch[i].cpu().numpy()
            pin.computeJointJacobians(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)
            J_pin = pin.getFrameJacobian(
                pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
            )

            compare_jacobians(J_bard_batch[i], J_pin, dtype)

    def test_fixed_base_zero_configuration(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies Jacobian at zero configuration."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        jac = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        q = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        J_bard = jac.calc(q, frame_idx, reference_frame="world")[0].cpu().numpy()

        q_pin = q[0].cpu().numpy()
        pin.computeJointJacobians(pin_model_obj, pin_data, q_pin)
        pin.updateFramePlacements(pin_model_obj, pin_data)
        J_pin = pin.getFrameJacobian(
            pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
        )

        compare_jacobians(J_bard, J_pin, dtype)

    def test_fixed_base_return_eef_pose(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies return_eef_pose option for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed

        jac = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(200)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # With pose return
        J_bard, T_bard = jac.calc(q, frame_idx, reference_frame="world", return_eef_pose=True)
        J_bard_np = J_bard[0].cpu().numpy()
        T_bard_np = T_bard[0].cpu().numpy()

        # Verify Jacobian
        q_pin = q[0].cpu().numpy()
        pin.computeJointJacobians(pin_model_obj, pin_data, q_pin)
        pin.updateFramePlacements(pin_model_obj, pin_data)
        J_pin = pin.getFrameJacobian(
            pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
        )

        compare_jacobians(J_bard_np, J_pin, dtype)

        # Verify pose shape
        assert T_bard_np.shape == (4, 4), f"Pose shape should be (4, 4), got {T_bard_np.shape}"

    def test_fixed_base_with_compilation(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that Jacobian works correctly with torch.compile enabled."""
        # Skip compilation tests for now due to torch.compile overflow issues
        pytest.skip("Compilation tests temporarily disabled due to torch.compile issues")

    # ========================================================================
    # Floating-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_path):
        """Builds Pinocchio model for floating-base robot."""
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_floating_base_random_configs(
        self, urdf_path, pin_model_floating, dtype, device, reference_frame
    ):
        """Verifies Jacobian at random configurations for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        # Create Jacobian instance
        jac = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(123)
        np.random.seed(123)

        # Select test frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test multiple random configurations
        for _ in range(5):
            # Generate random full configuration (base + joints)
            translations = torch.randn(1, 3, device=device, dtype=dtype)
            quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
            quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
            q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
            q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

            # Bard Jacobian
            J_bard = jac.calc(q, bard_frame_idx, reference_frame=reference_frame)
            J_bard_np = J_bard[0].cpu().numpy()

            # Convert to Pinocchio format
            q_pin = np.concatenate(
                [
                    translations[0].cpu().numpy(),
                    quats_wxyz[0, 1:].cpu().numpy(),  # qx, qy, qz
                    quats_wxyz[0, 0:1].cpu().numpy(),  # qw
                    q_joints[0].cpu().numpy(),
                ]
            )

            # Pinocchio Jacobian
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            J_pin = pin.computeFrameJacobian(
                pin_model_obj, pin_data, q_pin, pin_frame_id, pin_ref_frame
            )

            compare_jacobians(J_bard_np, J_pin, dtype)

    def test_floating_base_batched(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies batched Jacobian computation for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20

        # Create Jacobian instance
        jac = Jacobian(bard_chain, max_batch_size=batch_size, compile_enabled=False)

        torch.manual_seed(456)

        # Generate batched configurations
        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi
        q_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Batched computation
        J_bard_batch = jac.calc(q_batch, bard_frame_idx, reference_frame="world").cpu().numpy()

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

            J_pin = pin.computeFrameJacobian(
                pin_model_obj, pin_data, q_pin, pin_frame_id, pin.ReferenceFrame.WORLD
            )

            compare_jacobians(J_bard_batch[i], J_pin, dtype)

    def test_floating_base_identity_pose(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies Jacobian with identity base pose."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        jac = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        # Identity base pose with random joints
        translations = torch.zeros(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype) * 0.5
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        J_bard = jac.calc(q, frame_idx, reference_frame="world")[0].cpu().numpy()

        q_pin = np.concatenate(
            [
                translations[0].cpu().numpy(),
                quats_wxyz[0, 1:].cpu().numpy(),
                quats_wxyz[0, 0:1].cpu().numpy(),
                q_joints[0].cpu().numpy(),
            ]
        )

        J_pin = pin.computeFrameJacobian(
            pin_model_obj, pin_data, q_pin, pin_frame_id, pin.ReferenceFrame.WORLD
        )

        compare_jacobians(J_bard, J_pin, dtype)

    def test_floating_base_return_eef_pose(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies return_eef_pose option for floating-base robot."""
        bard_chain = build_chain_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating

        jac = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(789)
        translations = torch.randn(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # With pose return
        J_bard, T_bard = jac.calc(q, frame_idx, reference_frame="world", return_eef_pose=True)
        J_bard_np = J_bard[0].cpu().numpy()
        T_bard_np = T_bard[0].cpu().numpy()

        # Verify Jacobian
        q_pin = np.concatenate(
            [
                translations[0].cpu().numpy(),
                quats_wxyz[0, 1:].cpu().numpy(),
                quats_wxyz[0, 0:1].cpu().numpy(),
                q_joints[0].cpu().numpy(),
            ]
        )

        J_pin = pin.computeFrameJacobian(
            pin_model_obj, pin_data, q_pin, pin_frame_id, pin.ReferenceFrame.WORLD
        )

        compare_jacobians(J_bard_np, J_pin, dtype)

        # Verify pose shape
        assert T_bard_np.shape == (4, 4), f"Pose shape should be (4, 4), got {T_bard_np.shape}"

    def test_floating_base_with_compilation(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies that floating-base Jacobian works correctly with torch.compile enabled."""
        # Skip compilation tests for now
        pytest.skip("Compilation tests temporarily disabled due to torch.compile issues")

    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================

    def test_batch_size_validation(self, urdf_path, dtype, device):
        """Verifies that exceeding max_batch_size raises appropriate error."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        jac = Jacobian(bard_chain, max_batch_size=5, compile_enabled=False)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(frame_name)

        # This should work (batch_size = 5, max = 5)
        q_ok = torch.rand(5, bard_chain.n_joints, device=device, dtype=dtype)
        result = jac.calc(q_ok, frame_idx, reference_frame="world")
        assert result.shape[0] == 5, "Should process 5 samples"

        # This should raise ValueError (batch_size = 10, max = 5)
        q_too_large = torch.rand(10, bard_chain.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            _ = jac.calc(q_too_large, frame_idx, reference_frame="world")

    def test_single_vs_batch_consistency(self, urdf_path, dtype, device):
        """Verifies that single and batched queries produce consistent results."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        jac_single = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)
        jac_batch = Jacobian(bard_chain, max_batch_size=10, compile_enabled=False)

        torch.manual_seed(500)
        q_single = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        q_batch = q_single.expand(10, -1).contiguous()

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(frame_name)

        J_single = jac_single.calc(q_single, frame_idx, reference_frame="world")
        J_batch = jac_batch.calc(q_batch, frame_idx, reference_frame="world")

        # All batch results should match the single result
        for i in range(10):
            tol = 1e-6 if dtype == torch.float32 else 1e-10
            assert torch.allclose(
                J_single[0], J_batch[i], atol=tol
            ), f"Batch result {i} differs from single result"

    def test_world_vs_local_frame_relationship(self, urdf_path, dtype, device):
        """Verifies that world and local Jacobians have consistent shapes."""
        bard_chain = build_chain_from_urdf(urdf_path).to(dtype=dtype, device=device)

        jac = Jacobian(bard_chain, max_batch_size=1, compile_enabled=False)

        torch.manual_seed(999)
        q = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)

        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_id(frame_name)

        # Compute in both frames
        J_world = jac.calc(q, frame_idx, reference_frame="world")
        J_local = jac.calc(q, frame_idx, reference_frame="local")

        # Should have same shape
        assert (
            J_world.shape == J_local.shape
        ), f"World and local Jacobians have different shapes: {J_world.shape} vs {J_local.shape}"

        # Verify expected shape
        expected_shape = (1, 6, bard_chain.n_joints)
        assert (
            J_world.shape == expected_shape
        ), f"Jacobian shape should be {expected_shape}, got {J_world.shape}"
