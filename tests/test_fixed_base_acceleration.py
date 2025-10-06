# tests/test_fixed_base_acceleration.py

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
class TestAccelerationFixedBase:
    """Test suite for fixed-base robot end-effector acceleration."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a fixed-base robot."""
        model = pin.buildModelFromXML(urdf_string)
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_acceleration_random_states(self, urdf_string, pin_model, dtype, device, reference_frame):
        """Verifies acceleration at random states against Pinocchio."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        
        torch.manual_seed(1337)
        np.random.seed(1337)

        # Select a test frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        for _ in range(5):
            # Generate random state
            q_bard = torch.rand(bard_chain.n_joints, device=device, dtype=dtype)
            qd_bard = torch.randn(bard_chain.n_joints, device=device, dtype=dtype)
            qdd_bard = torch.randn(bard_chain.n_joints, device=device, dtype=dtype)
            
            q_pin = q_bard.cpu().numpy()
            qd_pin = qd_bard.cpu().numpy()
            qdd_pin = qdd_bard.cpu().numpy()

            # Bard acceleration
            a_bard = end_effector_acceleration(bard_chain, q_bard, qd_bard, qdd_bard, 
                                               bard_frame_idx, reference_frame=reference_frame)
            a_bard_np = a_bard.cpu().numpy()

            # Pinocchio acceleration
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)
            pin_ref_frame = pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            a_pin = pin.getFrameAcceleration(pin_model_obj, pin_data, pin_frame_id, pin_ref_frame).vector

            # Compare results
            tol = 1e-4 if dtype == torch.float32 else 1e-7
            assert np.allclose(a_bard_np, a_pin, atol=tol), f"Acceleration mismatch in '{reference_frame}' frame"