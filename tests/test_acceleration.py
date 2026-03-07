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

try:
    import pinocchio as pin
except ImportError:
    pin = None

import bard


def compare_accelerations(a_bard, a_pin, dtype):
    warn_tol = 1e-4 if dtype == torch.float32 else 1e-5
    fail_tol = warn_tol * 1.2
    max_diff = np.abs(a_bard - a_pin).max()
    linear_diff = np.linalg.norm(a_bard[:3] - a_pin[:3])
    angular_diff = np.linalg.norm(a_bard[3:] - a_pin[3:])

    if max_diff >= fail_tol:
        assert False, (
            f"Acceleration mismatch: max_diff={max_diff:.3e} > fail_tol={fail_tol:.1e}\n"
            f"Linear diff: {linear_diff:.3e}, Angular diff: {angular_diff:.3e}"
        )
    elif max_diff >= warn_tol:
        warnings.warn(
            f"Acceleration mismatch: max_diff={max_diff:.3e} > fail_tol={warn_tol:.1e}\n"
            f"Linear diff: {linear_diff:.3e}, Angular diff: {angular_diff:.3e}"
        )


@pytest.mark.skipif(not hasattr(pin, "buildModelFromUrdf"), reason="Pinocchio not available")
class TestSpatialAcceleration:
    """Test suite for Spatial Acceleration with both fixed-base and floating-base robots."""

    # ========================================================================
    # Fixed-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_path):
        model = pin.buildModelFromUrdf(str(urdf_path))
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_fixed_base_random_states(
        self, urdf_path, pin_model_fixed, dtype, device, reference_frame
    ):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_fixed

        torch.manual_seed(1337)
        np.random.seed(1337)

        test_frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = model.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        for _ in range(5):
            q = torch.rand(1, model.n_joints, device=device, dtype=dtype)
            qd = torch.randn(1, model.n_joints, device=device, dtype=dtype)
            qdd = torch.randn(1, model.n_joints, device=device, dtype=dtype)

            bard.update_kinematics(model, data, q, qd)
            a_bard = bard.spatial_acceleration(
                model, data, qdd, bard_frame_idx, reference_frame=reference_frame
            )
            a_bard_np = a_bard[0].cpu().numpy()

            pin.forwardKinematics(
                pin_model_obj,
                pin_data,
                q[0].cpu().numpy(),
                qd[0].cpu().numpy(),
                qdd[0].cpu().numpy(),
            )
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin_ref_frame
            ).vector

            compare_accelerations(a_bard_np, a_pin, dtype)

    def test_fixed_base_batched(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20
        data = bard.create_data(model, max_batch_size=batch_size)

        torch.manual_seed(42)
        q_batch = torch.rand(batch_size, model.n_joints, device=device, dtype=dtype) * np.pi
        qd_batch = torch.randn(batch_size, model.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.randn(batch_size, model.n_joints, device=device, dtype=dtype)

        test_frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = model.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        bard.update_kinematics(model, data, q_batch, qd_batch)
        a_bard_batch = (
            bard.spatial_acceleration(
                model, data, qdd_batch, bard_frame_idx, reference_frame="world"
            )
            .cpu()
            .numpy()
        )

        for i in range(batch_size):
            pin.forwardKinematics(
                pin_model_obj,
                pin_data,
                q_batch[i].cpu().numpy(),
                qd_batch[i].cpu().numpy(),
                qdd_batch[i].cpu().numpy(),
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
            ).vector
            compare_accelerations(a_bard_batch[i], a_pin, dtype)

    def test_fixed_base_zero_velocity_acceleration(self, urdf_path, pin_model_fixed, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_fixed

        q = torch.rand(1, model.n_joints, device=device, dtype=dtype) * 0.5
        qd = torch.zeros(1, model.n_joints, device=device, dtype=dtype)
        qdd = torch.zeros(1, model.n_joints, device=device, dtype=dtype)

        test_frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = model.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        bard.update_kinematics(model, data, q, qd)
        a_bard = (
            bard.spatial_acceleration(model, data, qdd, frame_idx, reference_frame="world")[0]
            .cpu()
            .numpy()
        )

        pin.forwardKinematics(
            pin_model_obj, pin_data, q[0].cpu().numpy(), qd[0].cpu().numpy(), qdd[0].cpu().numpy()
        )
        a_pin = pin.getFrameAcceleration(
            pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
        ).vector

        assert np.allclose(a_bard, a_pin, atol=1e-10), "Zero velocity/acceleration case failed"
        assert np.allclose(a_bard, 0.0, atol=1e-10), "Acceleration should be near zero"

    def test_fixed_base_with_compilation(self, urdf_path, pin_model_fixed, dtype, device):
        """Verifies that spatial acceleration works with torch.compile enabled."""
        if device == "cpu":
            pytest.skip("torch.compile inductor backend requires CUDA on Windows")
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=5)

        torch.manual_seed(44)
        q = torch.rand(5, model.n_joints, device=device, dtype=dtype) * np.pi - np.pi / 2
        qd = torch.randn(5, model.nv, device=device, dtype=dtype)
        qdd = torch.randn(5, model.nv, device=device, dtype=dtype)
        frame_idx = model.get_frame_id(model.get_frame_names(exclude_fixed=True)[-1])

        bard.update_kinematics(model, data, q, qd)
        a_ref = bard.spatial_acceleration(
            model, data, qdd, frame_idx, reference_frame="world"
        ).clone()

        model.enable_compilation(True)
        data_compiled = bard.create_data(model, max_batch_size=5)
        bard.update_kinematics(model, data_compiled, q, qd)
        a_compiled = bard.spatial_acceleration(
            model, data_compiled, qdd, frame_idx, reference_frame="world"
        )

        tol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(a_ref, a_compiled, atol=tol), (
            f"Compiled spatial acceleration differs: max diff = "
            f"{(a_ref - a_compiled).abs().max():.3e}"
        )

    # ========================================================================
    # Floating-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_path):
        model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_floating_base_random_states(
        self, urdf_path, pin_model_floating, dtype, device, reference_frame
    ):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_floating

        torch.manual_seed(2048)
        np.random.seed(2048)

        test_frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = model.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        for _ in range(5):
            translations = torch.randn(1, 3, device=device, dtype=dtype)
            quats_wxyz = torch.randn(1, 4, device=device, dtype=dtype)
            quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
            q_joints = torch.rand(1, model.n_joints, device=device, dtype=dtype)
            q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

            v_base = torch.randn(1, 6, device=device, dtype=dtype)
            qd_joints = torch.randn(1, model.n_joints, device=device, dtype=dtype)
            qd = torch.cat([v_base, qd_joints], dim=1)

            a_base = torch.randn(1, 6, device=device, dtype=dtype)
            qdd_joints = torch.randn(1, model.n_joints, device=device, dtype=dtype)
            qdd = torch.cat([a_base, qdd_joints], dim=1)

            bard.update_kinematics(model, data, q, qd)
            a_bard = bard.spatial_acceleration(
                model, data, qdd, bard_frame_idx, reference_frame=reference_frame
            )
            a_bard_np = a_bard[0].cpu().numpy()

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
            pin_ref_frame = (
                pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin_ref_frame
            ).vector

            compare_accelerations(a_bard_np, a_pin, dtype)

    def test_floating_base_batched(self, urdf_path, pin_model_floating, dtype, device):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20
        data = bard.create_data(model, max_batch_size=batch_size)

        torch.manual_seed(123)
        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, model.n_joints, device=device, dtype=dtype) * np.pi
        q_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        v_base = torch.randn(batch_size, 6, device=device, dtype=dtype)
        qd_joints = torch.randn(batch_size, model.n_joints, device=device, dtype=dtype)
        qd_batch = torch.cat([v_base, qd_joints], dim=1)

        a_base = torch.randn(batch_size, 6, device=device, dtype=dtype)
        qdd_joints = torch.randn(batch_size, model.n_joints, device=device, dtype=dtype)
        qdd_batch = torch.cat([a_base, qdd_joints], dim=1)

        test_frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = model.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        bard.update_kinematics(model, data, q_batch, qd_batch)
        a_bard_batch = (
            bard.spatial_acceleration(
                model, data, qdd_batch, bard_frame_idx, reference_frame="world"
            )
            .cpu()
            .numpy()
        )

        for i in range(batch_size):
            q_pin = np.concatenate(
                [
                    translations[i].cpu().numpy(),
                    quats_wxyz[i, 1:].cpu().numpy(),
                    quats_wxyz[i, 0:1].cpu().numpy(),
                    q_joints[i].cpu().numpy(),
                ]
            )
            pin.forwardKinematics(
                pin_model_obj,
                pin_data,
                q_pin,
                qd_batch[i].cpu().numpy(),
                qdd_batch[i].cpu().numpy(),
            )
            a_pin = pin.getFrameAcceleration(
                pin_model_obj, pin_data, pin_frame_id, pin.ReferenceFrame.WORLD
            ).vector
            compare_accelerations(a_bard_batch[i], a_pin, dtype)

    def test_floating_base_stationary(self, urdf_path, pin_model_floating, dtype, device):
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=1)
        pin_model_obj, pin_data = pin_model_floating

        translations = torch.zeros(1, 3, device=device, dtype=dtype)
        quats_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        q_joints = torch.rand(1, model.n_joints, device=device, dtype=dtype) * 0.3
        q = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        qd = torch.zeros(1, 6 + model.n_joints, device=device, dtype=dtype)
        qdd = torch.zeros(1, 6 + model.n_joints, device=device, dtype=dtype)

        test_frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = model.get_frame_id(test_frame_name)
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        bard.update_kinematics(model, data, q, qd)
        a_bard = (
            bard.spatial_acceleration(model, data, qdd, frame_idx, reference_frame="world")[0]
            .cpu()
            .numpy()
        )

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

    def test_floating_base_with_compilation(self, urdf_path, pin_model_floating, dtype, device):
        """Verifies that floating-base spatial acceleration works with torch.compile enabled."""
        if device == "cpu":
            pytest.skip("torch.compile inductor backend requires CUDA on Windows")
        model = bard.build_model_from_urdf(urdf_path, floating_base=True).to(
            dtype=dtype, device=device
        )
        data = bard.create_data(model, max_batch_size=5)

        torch.manual_seed(45)
        translations = torch.randn(5, 3, device=device, dtype=dtype)
        quats = torch.randn(5, 4, device=device, dtype=dtype)
        quats = quats / torch.linalg.norm(quats, dim=1, keepdim=True)
        q_joints = torch.rand(5, model.n_joints, device=device, dtype=dtype) * np.pi
        q = torch.cat([translations, quats, q_joints], dim=1)
        qd = torch.randn(5, model.nv, device=device, dtype=dtype)
        qdd = torch.randn(5, model.nv, device=device, dtype=dtype)
        frame_idx = model.get_frame_id(model.get_frame_names(exclude_fixed=True)[-1])

        bard.update_kinematics(model, data, q, qd)
        a_ref = bard.spatial_acceleration(
            model, data, qdd, frame_idx, reference_frame="world"
        ).clone()

        model.enable_compilation(True)
        data_compiled = bard.create_data(model, max_batch_size=5)
        bard.update_kinematics(model, data_compiled, q, qd)
        a_compiled = bard.spatial_acceleration(
            model, data_compiled, qdd, frame_idx, reference_frame="world"
        )

        tol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(a_ref, a_compiled, atol=tol), (
            f"Compiled floating-base spatial acceleration differs: max diff = "
            f"{(a_ref - a_compiled).abs().max():.3e}"
        )

    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================

    def test_batch_size_validation(self, urdf_path, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=5)

        frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = model.get_frame_id(frame_name)

        q_ok = torch.rand(5, model.n_joints, device=device, dtype=dtype)
        qd_ok = torch.randn(5, model.n_joints, device=device, dtype=dtype)
        qdd_ok = torch.randn(5, model.n_joints, device=device, dtype=dtype)
        bard.update_kinematics(model, data, q_ok, qd_ok)
        result = bard.spatial_acceleration(model, data, qdd_ok, frame_idx, reference_frame="world")
        assert result.shape[0] == 5

        q_too_large = torch.rand(10, model.n_joints, device=device, dtype=dtype)
        qd_too_large = torch.randn(10, model.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            bard.update_kinematics(model, data, q_too_large, qd_too_large)

    def test_world_vs_local_frame_consistency(self, urdf_path, dtype, device):
        model = bard.build_model_from_urdf(urdf_path).to(dtype=dtype, device=device)
        data = bard.create_data(model, max_batch_size=1)

        torch.manual_seed(999)
        q = torch.rand(1, model.n_joints, device=device, dtype=dtype)
        qd = torch.randn(1, model.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(1, model.n_joints, device=device, dtype=dtype)

        frame_name = model.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = model.get_frame_id(frame_name)

        bard.update_kinematics(model, data, q, qd)
        a_world = bard.spatial_acceleration(model, data, qdd, frame_idx, reference_frame="world")
        a_local = bard.spatial_acceleration(model, data, qdd, frame_idx, reference_frame="local")

        assert a_world.shape == (1, 6), "World frame acceleration has wrong shape"
        assert a_local.shape == (1, 6), "Local frame acceleration has wrong shape"
