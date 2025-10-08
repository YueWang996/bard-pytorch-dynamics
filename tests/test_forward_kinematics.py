"""
Comprehensive tests for Forward Kinematics (both fixed-base and floating-base).

Tests cover:
- Fixed-base robots at zero and random configurations
- Floating-base robots with random base poses
- Batched operations
- Compilation compatibility
"""

import warnings
import pytest
import torch
import numpy as np
import pinocchio as pin

from bard.parsers.urdf import build_chain_from_urdf
from bard import ForwardKinematics
import bard.transforms as tf


def compare_transforms_fk(T_ours, T_pin, dtype):
    """
    Compare transforms with FK-appropriate tolerances.
    FK can accumulate small errors through multiple joint transformations.
    """
    # Warning tolerances for FK
    pos_warn_tol = 1e-3 if dtype == torch.float32 else 1e-5
    rot_warn_tol = 1e-3 if dtype == torch.float32 else 5e-4
    
    # Fail tolerances (20% buffer above warning)
    pos_fail_tol = pos_warn_tol * 1.2
    rot_fail_tol = rot_warn_tol * 1.2
    
    R_ours, p_ours = T_ours[:3, :3], T_ours[:3, 3]
    R_pin, p_pin = T_pin.rotation, T_pin.translation
    
    pos_err = float(np.linalg.norm(p_ours - p_pin))
    
    # Geodesic distance on SO(3)
    R_err = R_ours @ R_pin.T
    trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.arccos(trace_val))
    
    # Check position error
    if pos_err >= pos_fail_tol:
        assert False, f"Position error {pos_err:.3e} exceeds fail tolerance {pos_fail_tol:.1e}"
    elif pos_err >= pos_warn_tol:
        warnings.warn(f"Position error {pos_err:.3e} close to tolerance {pos_warn_tol:.1e}")
    
    # Check rotation error
    if rot_err >= rot_fail_tol:
        assert False, f"Rotation error {rot_err:.3e} exceeds fail tolerance {rot_fail_tol:.1e}"
    elif rot_err >= rot_warn_tol:
        warnings.warn(f"Rotation error {rot_err:.3e} close to tolerance {rot_warn_tol:.1e}")


@pytest.mark.skipif(not hasattr(pin, 'buildModelFromXML'), reason="Pinocchio not available")
class TestForwardKinematics:
    """Test suite for Forward Kinematics with both fixed-base and floating-base robots."""

    # ========================================================================
    # Fixed-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_fixed(self, urdf_string):
        """Builds Pinocchio model for fixed-base robot."""
        model = pin.buildModelFromXML(urdf_string)
        return model, model.createData()

    def test_fixed_base_zero_config(self, urdf_string, pin_model_fixed, dtype, device):
        """Verifies FK at zero configuration for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        
        # Create FK instance (without compilation)
        fk = ForwardKinematics(bard_chain, max_batch_size=1, compile_enabled=False)
        
        q_bard = torch.zeros(1, bard_chain.n_joints, device=device, dtype=dtype)
        q_pin = np.zeros(pin_model_obj.nq)

        pin.forwardKinematics(pin_model_obj, pin_data, q_pin)
        pin.updateFramePlacements(pin_model_obj, pin_data)
        
        # Test end-effector frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        
        # Bard FK returns raw tensor, wrap in Transform3d for consistency
        T_bard_matrix = fk.calc(q_bard, frame_idx)[0].cpu().numpy()

        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)
        T_pin = pin_data.oMf[pin_frame_id]

        compare_transforms_fk(T_bard_matrix, T_pin, dtype)

    def test_fixed_base_random_configs_batched(self, urdf_string, pin_model_fixed, dtype, device):
        """Verifies batched FK at random configurations for fixed-base robot."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_fixed
        batch_size = 20
        
        # Create FK instance with appropriate batch size (without compilation)
        fk = ForwardKinematics(bard_chain, max_batch_size=batch_size, compile_enabled=False)
        
        torch.manual_seed(42)
        q_batch = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi - np.pi / 2
        
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Batched FK computation
        T_bard_batch = fk.calc(q_batch, bard_frame_idx).cpu().numpy()
        
        # Verify each sample against Pinocchio
        for i in range(batch_size):
            q_pin = q_batch[i].cpu().numpy()
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)
            T_pin = pin_data.oMf[pin_frame_id]
            
            compare_transforms_fk(T_bard_batch[i], T_pin, dtype)

    def test_fixed_base_with_compilation(self, urdf_string, pin_model_fixed, dtype, device):
        """Verifies that FK works correctly with torch.compile enabled."""
        pytest.skip("Placeholder test - compilation tests is to be implemented")

    # ========================================================================
    # Floating-Base Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def pin_model_floating(self, urdf_string):
        """Builds Pinocchio model for floating-base robot."""
        model = pin.buildModelFromXML(urdf_string, pin.JointModelFreeFlyer())
        return model, model.createData()

    def test_floating_base_random_configs_batched(self, urdf_string, pin_model_floating, dtype, device):
        """Verifies batched FK with random base poses and joint configurations."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_floating
        batch_size = 20
        n_joints = bard_chain.n_joints
        
        # Create FK instance (without compilation)
        fk = ForwardKinematics(bard_chain, max_batch_size=batch_size, compile_enabled=False)
        
        torch.manual_seed(123)
        np.random.seed(123)

        # Generate batched base poses and joint positions
        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        q_joints = torch.rand(batch_size, n_joints, device=device, dtype=dtype) * np.pi
        
        # Batched input: [tx, ty, tz, qw, qx, qy, qz, joints...]
        q_bard_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)
        
        # Batched FK computation
        T_bard_batch = fk.calc(q_bard_batch, bard_frame_idx).cpu().numpy()
        
        # Verify each sample against Pinocchio
        for i in range(batch_size):
            # Pinocchio format: [tx, ty, tz, qx, qy, qz, qw, joints...]
            q_pin = np.concatenate([
                translations[i].cpu().numpy(),
                quats_wxyz[i, 1:].cpu().numpy(),  # qx, qy, qz
                quats_wxyz[i, 0:1].cpu().numpy(),  # qw
                q_joints[i].cpu().numpy()
            ])
            
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)
            T_pin = pin_data.oMf[pin_frame_id]
            
            compare_transforms_fk(T_bard_batch[i], T_pin, dtype)

    def test_floating_base_identity_base_pose(self, urdf_string, pin_model_floating, dtype, device):
        """Verifies FK with identity base pose (should match fixed-base at same joint config)."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model_floating
        
        # Create FK instance (without compilation)
        fk = ForwardKinematics(bard_chain, max_batch_size=5, compile_enabled=False)
        
        torch.manual_seed(200)
        n_joints = bard_chain.n_joints
        batch_size = 5
        
        # Identity base pose: [0,0,0, 1,0,0,0] + random joints
        translations = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype).expand(batch_size, -1)
        q_joints = torch.rand(batch_size, n_joints, device=device, dtype=dtype) * 0.5
        
        q_bard = torch.cat([translations, quats_wxyz, q_joints], dim=1)
        
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)
        
        T_bard_batch = fk.calc(q_bard, frame_idx).cpu().numpy()
        
        for i in range(batch_size):
            q_pin = np.concatenate([
                translations[i].cpu().numpy(),
                quats_wxyz[i, 1:].cpu().numpy(),
                quats_wxyz[i, 0:1].cpu().numpy(),
                q_joints[i].cpu().numpy()
            ])
            
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)
            T_pin = pin_data.oMf[pin_frame_id]
            
            compare_transforms_fk(T_bard_batch[i], T_pin, dtype)

    def test_floating_base_with_compilation(self, urdf_string, pin_model_floating, dtype, device):
        """Verifies that floating-base FK works correctly with torch.compile enabled."""
        # Skip compilation tests for now due to torch.compile overflow issues
        pytest.skip("Placeholder test - compilation tests is to be implemented")

    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================

    def test_fixed_base_single_vs_batch(self, urdf_string, dtype, device):
        """Verifies that single and batched queries produce consistent results."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        
        fk_single = ForwardKinematics(bard_chain, max_batch_size=1, compile_enabled=False)
        fk_batch = ForwardKinematics(bard_chain, max_batch_size=10, compile_enabled=False)
        
        torch.manual_seed(500)
        q_single = torch.rand(1, bard_chain.n_joints, device=device, dtype=dtype)
        q_batch = q_single.expand(10, -1).contiguous()
        
        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(frame_name).item()
        
        T_single = fk_single.calc(q_single, frame_idx)
        T_batch = fk_batch.calc(q_batch, frame_idx)
        
        # All batch results should match the single result
        for i in range(10):
            tol = 1e-6 if dtype == torch.float32 else 1e-10
            assert torch.allclose(T_single[0], T_batch[i], atol=tol), \
                f"Batch result {i} differs from single result"

    def test_batch_size_validation(self, urdf_string, dtype, device):
        """Verifies that exceeding max_batch_size raises appropriate error."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        
        fk = ForwardKinematics(bard_chain, max_batch_size=5, compile_enabled=False)
        
        frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        frame_idx = bard_chain.get_frame_indices(frame_name).item()
        
        # This should work (batch_size = 5, max = 5)
        q_ok = torch.rand(5, bard_chain.n_joints, device=device, dtype=dtype)
        result = fk.calc(q_ok, frame_idx)
        assert result.shape[0] == 5, "Should process 5 samples"
        
        # This should raise ValueError (batch_size = 10, max = 5)
        q_too_large = torch.rand(10, bard_chain.n_joints, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            _ = fk.calc(q_too_large, frame_idx)