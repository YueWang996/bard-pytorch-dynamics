# tests/test_fixed_base_dynamics.py

import pytest
import torch
import numpy as np
import pinocchio as pin

from bard.parsers.urdf import build_chain_from_urdf
from bard.core.dynamics import calc_inverse_dynamics, crba_inertia_matrix

@pytest.mark.skipif(not hasattr(pin, 'buildModelFromXML'), reason="Pinocchio fixtures not fully available")
class TestDynamicsFixedBase:
    """Test suite for fixed-base robot inverse dynamics (RNEA) and CRBA."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a fixed-base robot."""
        model = pin.buildModelFromXML(urdf_string)
        return model, model.createData()

    def test_rnea_components(self, urdf_string, pin_model, dtype, device):
        """Verifies RNEA components (gravity, coriolis, inertia) against Pinocchio."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        
        torch.manual_seed(1000)
        q = torch.rand(bard_chain.n_joints, device=device, dtype=dtype)
        qd = torch.randn(bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(bard_chain.n_joints, device=device, dtype=dtype)
        
        q_pin, qd_pin, qdd_pin = q.cpu().numpy(), qd.cpu().numpy(), qdd.cpu().numpy()
        
        # Pinocchio calculations for components
        g_pin = pin.computeGeneralizedGravity(pin_model_obj, pin_data, q_pin)
        c_pin = pin.computeCoriolisMatrix(pin_model_obj, pin_data, q_pin, qd_pin) @ qd_pin
        tau_full_pin = pin.rnea(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)

        # Bard calculations
        zeros = torch.zeros_like(q)
        g_bard = calc_inverse_dynamics(bard_chain, q, zeros, zeros).cpu().numpy()[0]
        c_bard = calc_inverse_dynamics(bard_chain, q, qd, zeros, gravity=torch.zeros(3)).cpu().numpy()[0]
        tau_full_bard = calc_inverse_dynamics(bard_chain, q, qd, qdd).cpu().numpy()[0]
        
        tol = 1e-5 if dtype == torch.float32 else 1e-7
        assert np.allclose(g_bard, g_pin, atol=tol), "Gravity term mismatch"
        assert np.allclose(c_bard, c_pin, atol=tol), "Coriolis/centrifugal term mismatch"
        assert np.allclose(tau_full_bard, tau_full_pin, atol=tol), "Full RNEA mismatch"
        
    def test_crba_and_consistency(self, urdf_string, pin_model, dtype, device):
        """Verifies CRBA mass matrix and consistency with RNEA."""
        bard_chain = build_chain_from_urdf(urdf_string).to(dtype=dtype, device=device)
        pin_model_obj, pin_data = pin_model
        
        torch.manual_seed(1001)
        q = torch.rand(bard_chain.n_joints, device=device, dtype=dtype)
        qdd = torch.randn(bard_chain.n_joints, device=device, dtype=dtype)
        
        # CRBA check
        M_bard = crba_inertia_matrix(bard_chain, q).cpu().numpy()[0]
        M_pin = pin.crba(pin_model_obj, pin_data, q.cpu().numpy())
        
        tol_crba = 1e-4 if dtype == torch.float32 else 1e-7
        assert np.allclose(M_bard, M_pin, atol=tol_crba), "CRBA mass matrix mismatch"

        # Consistency check: M*qdd = RNEA(q, qd=0, qdd, g=0)
        tau_rnea_inertia = calc_inverse_dynamics(bard_chain, q, torch.zeros_like(q), qdd, gravity=torch.zeros(3))
        tau_from_crba = torch.from_numpy(M_bard).to(device, dtype) @ qdd

        tol_consistency = 1e-5 if dtype == torch.float32 else 1e-7
        assert torch.allclose(tau_rnea_inertia[0], tau_from_crba, atol=tol_consistency), "RNEA and CRBA are not consistent"