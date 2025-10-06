# tests/test_fixed_base_fk.py

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
class TestForwardKinematicsFixedBase:
    """Test suite for fixed-base robot forward kinematics."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a fixed-base robot."""
        model = pin.buildModelFromXML(urdf_string)
        return model, model.createData()

    def test_fk_at_zero_config(self, urdf_string, pin_model, dtype, device):
        """Verifies FK at zero configuration for all frames."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        
        q_bard = torch.zeros(bard_chain.n_joints, device=device, dtype=dtype)
        q_pin = np.zeros(pin_model_obj.nq)

        pin.forwardKinematics(pin_model_obj, pin_data, q_pin)
        pin.updateFramePlacements(pin_model_obj, pin_data)
        
        for frame_name in bard_chain.get_frame_names(exclude_fixed=False):
            if not pin_model_obj.existFrame(frame_name):
                continue

            frame_idx = bard_chain.get_frame_indices(frame_name).item()
            T_bard_matrix = calc_forward_kinematics(bard_chain, q_bard, frame_idx).get_matrix()[0].cpu().numpy()
            
            pin_frame_id = pin_model_obj.getFrameId(frame_name)
            T_pin = pin_data.oMf[pin_frame_id]

            compare_transforms(T_bard_matrix, T_pin, dtype)

    def test_fk_at_random_configs_batched(self, urdf_string, pin_model, dtype, device):
        """Verifies batched FK at random configurations."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        batch_size = 20
        
        torch.manual_seed(42)
        q_batch = torch.rand(batch_size, bard_chain.n_joints, device=device, dtype=dtype) * np.pi - np.pi / 2
        
        test_frame_name = bard_chain.get_frame_names(exclude_fixed=True)[-1]
        bard_frame_idx = bard_chain.get_frame_indices(test_frame_name).item()
        pin_frame_id = pin_model_obj.getFrameId(test_frame_name)

        T_bard_batch = calc_forward_kinematics(bard_chain, q_batch, bard_frame_idx).get_matrix().cpu().numpy()
        
        for i in range(batch_size):
            q_pin = q_batch[i].cpu().numpy()
            pin.forwardKinematics(pin_model_obj, pin_data, q_pin)
            pin.updateFramePlacements(pin_model_obj, pin_data)
            T_pin = pin_data.oMf[pin_frame_id]
            
            compare_transforms(T_bard_batch[i], T_pin, dtype)