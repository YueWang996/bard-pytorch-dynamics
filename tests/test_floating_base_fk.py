import pytest
import torch
import numpy as np
import pinocchio as pin

# Imports from your library
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.kinematics import calc_forward_kinematics

# Imports from conftest
from conftest import compare_transforms


@pytest.mark.skipif(not hasattr(pin, 'buildModelFromXML'), reason="Pinocchio fixtures not fully available")
class TestForwardKinematicsFloatingBase:
    """Test suite for floating-base robot forward kinematics."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a floating-base robot."""
        model = pin.buildModelFromXML(urdf_string, pin.JointModelFreeFlyer())
        return model, model.createData()

    def test_fk_at_random_configs_batched(self, urdf_string, pin_model, dtype, device):
        """Verifies batched FK with random base poses and joint configurations."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        batch_size = 20
        n_joints = bard_chain.n_joints
        
        torch.manual_seed(123)
        np.random.seed(123)

        translations = torch.randn(batch_size, 3, device=device, dtype=dtype)
        quats_wxyz = torch.randn(batch_size, 4, device=device, dtype=dtype)
        quats_wxyz = quats_wxyz / torch.linalg.norm(quats_wxyz, dim=1, keepdim=True)
        
        q_joints = torch.rand(batch_size, n_joints, device=device, dtype=dtype) * np.pi
        q_bard_batch = torch.cat([translations, quats_wxyz, q_joints], dim=1)

        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)
        
        T_bard_batch = calc_forward_kinematics(bard_chain, q_bard_batch, bard_frame_idx).get_matrix().cpu().numpy()
        
        for i in range(batch_size):
            # Pinocchio uses [tx,ty,tz, qx,qy,qz,qw]
            q_pin = np.concatenate([
                translations[i].cpu().numpy(),
                quats_wxyz[i, 1:].cpu().numpy(),
                quats_wxyz[i, 0:1].cpu().numpy(),
                q_joints[i].cpu().numpy()
            ])
            
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)
            T_pin = pin_data.oMf[pin_frame_id]
            
            compare_transforms(T_bard_batch[i], T_pin, dtype)