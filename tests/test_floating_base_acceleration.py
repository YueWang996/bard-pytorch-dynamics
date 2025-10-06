# tests/test_floating_base_acceleration.py

import pytest
import torch
import numpy as np
import pinocchio as pin

# Imports from your library
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.kinematics import end_effector_acceleration

# Imports from conftest
from conftest import compare_transforms


@pytest.mark.skipif(not hasattr(pin, 'buildModelFromXML'), reason="Pinocchio fixtures not fully available")
class TestAccelerationFloatingBase:
    """Test suite for floating-base robot end-effector acceleration."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a floating-base robot."""
        model = pin.buildModelFromXML(urdf_string, pin.JointModelFreeFlyer())
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_acceleration_random_states(self, urdf_string, pin_model, dtype, device, reference_frame):
        """Verifies floating-base acceleration at random states."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        
        torch.manual_seed(2048)
        np.random.seed(2048)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        for _ in range(5):
            # Generate random full state (base + joints)
            # Position q
            translations = torch.randn(3, device=device, dtype=dtype)
            quats_wxyz = torch.randn(4, device=device, dtype=dtype)
            quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz)
            q_joints = torch.rand(bard_chain.n_joints, device=device, dtype=dtype)
            q_bard = torch.cat([translations, quats_wxyz, q_joints])
            
            # Velocity qd and acceleration qdd
            v_base = torch.randn(6, device=device, dtype=dtype)
            qd_joints = torch.randn(bard_chain.n_joints, device=device, dtype=dtype)
            qd_bard = torch.cat([v_base, qd_joints])
            
            a_base = torch.randn(6, device=device, dtype=dtype)
            qdd_joints = torch.randn(bard_chain.n_joints, device=device, dtype=dtype)
            qdd_bard = torch.cat([a_base, qdd_joints])

            # Bard acceleration
            a_bard = end_effector_acceleration(bard_chain, q_bard, qd_bard, qdd_bard, 
                                               bard_frame_idx, reference_frame=reference_frame)
            a_bard_np = a_bard.cpu().numpy()

            # Convert to Pinocchio format
            q_pin = np.concatenate([translations.cpu().numpy(), quats_wxyz[1:].cpu().numpy(), 
                                      quats_wxyz[0:1].cpu().numpy(), q_joints.cpu().numpy()])
            qd_pin = qd_bard.cpu().numpy()
            qdd_pin = qdd_bard.cpu().numpy()

            # Pinocchio acceleration
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            pin_ref_frame = pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            a_pin = pin.getFrameAcceleration(pin_model_obj, pin_data, pin_frame_id, pin_ref_frame).vector
            
            # Compare results
            tol = 1e-4 if dtype == torch.float32 else 1e-7
            assert np.allclose(a_bard_np, a_pin, atol=tol), f"Acceleration mismatch in '{reference_frame}' frame"