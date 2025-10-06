import pytest
import torch
import numpy as np
import pinocchio as pin

# Imports from your library
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.jacobian import calc_jacobian

@pytest.mark.skipif(not hasattr(pin, 'buildModelFromXML'), reason="Pinocchio fixtures not fully available")
class TestJacobianFloatingBase:
    """Test suite for floating-base robot Jacobian."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a floating-base robot."""
        model = pin.buildModelFromXML(urdf_string, pin.JointModelFreeFlyer())
        return model, model.createData()

    @pytest.mark.parametrize("reference_frame", ["world", "local"])
    def test_jacobian_random_configs(self, urdf_string, pin_model, dtype, device, reference_frame):
        """Verifies the floating-base Jacobian at random configurations."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        
        torch.manual_seed(123)
        np.random.seed(123)

        # Select a test frame
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)
        
        for _ in range(5):
            # Generate a random full configuration (base + joints)
            translations = torch.randn(3, device=device, dtype=dtype)
            quats_wxyz = torch.randn(4, device=device, dtype=dtype)
            quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz)
            q_joints = torch.rand(bard_chain.n_joints, device=device, dtype=dtype) * np.pi
            q_bard = torch.cat([translations, quats_wxyz, q_joints])
            
            # Convert to Pinocchio's configuration format [tx,ty,tz, qx,qy,qz,qw, joints...]
            q_pin = np.concatenate([
                translations.cpu().numpy(),
                quats_wxyz[1:].cpu().numpy(),
                quats_wxyz[0:1].cpu().numpy(),
                q_joints.cpu().numpy()
            ])

            # Bard Jacobian
            J_bard = calc_jacobian(bard_chain, q_bard, bard_frame_idx, reference_frame=reference_frame)
            J_bard_np = J_bard[0].cpu().numpy()

            # Pinocchio Jacobian
            # For a floating base, computeFrameJacobian gives the full Jacobian (base + joints)
            pin_ref_frame = pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
            J_pin = pin.computeFrameJacobian(pin_model_obj, pin_data, q_pin, pin_frame_id, pin_ref_frame)

            # Compare results
            tol = 1e-5 if dtype == torch.float32 else 1e-7
            assert np.allclose(J_bard_np, J_pin, atol=tol), f"Jacobian mismatch in '{reference_frame}' frame"