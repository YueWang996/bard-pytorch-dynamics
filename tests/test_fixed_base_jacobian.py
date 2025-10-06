# tests/test_fixed_base_jacobian.py

import pytest
import torch
import numpy as np
import pinocchio as pin

# Imports from your library
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.jacobian import calc_jacobian

@pytest.mark.skipif(not hasattr(pin, 'buildModelFromXML'), reason="Pinocchio fixtures not fully available")
class TestJacobianFixedBase:
    """Test suite for fixed-base robot Jacobian."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a fixed-base robot."""
        model = pin.buildModelFromXML(urdf_string)
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_jacobian_random_configs(self, urdf_string, pin_model, dtype, device, reference_frame):
        """Verifies the Jacobian at random configurations against Pinocchio."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        
        torch.manual_seed(42)
        np.random.seed(42)

        # Select a test frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        # Test several random configurations
        for _ in range(5):
            q_bard = torch.rand(bard_chain.n_joints, device=device, dtype=dtype) * np.pi - np.pi / 2
            q_pin = q_bard.cpu().numpy()

            # Bard Jacobian
            J_bard = calc_jacobian(bard_chain, q_bard, bard_frame_idx, reference_frame=reference_frame)
            J_bard_np = J_bard[0].cpu().numpy()

            # Pinocchio Jacobian
            pin.computeJointJacobians(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)
            
            pin_ref_frame = pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            J_pin = pin.getFrameJacobian(pin_model_obj, pin_data, pin_frame_id, pin_ref_frame)

            # Compare results
            tol = 1e-5 if dtype == torch.float32 else 1e-7
            assert np.allclose(J_bard_np, J_pin, atol=tol), f"Jacobian mismatch in '{reference_frame}' frame"