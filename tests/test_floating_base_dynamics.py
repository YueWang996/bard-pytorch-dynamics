# tests/test_floating_base_dynamics.py

import pytest
import torch
import numpy as np
import pinocchio as pin

from bard.parsers.urdf import build_chain_from_urdf
from bard.core.dynamics import calc_inverse_dynamics, crba_inertia_matrix


@pytest.mark.skipif(
    not hasattr(pin, "buildModelFromXML"), reason="Pinocchio fixtures not fully available"
)
class TestDynamicsFloatingBase:
    """Test suite for floating-base robot inverse dynamics (RNEA) and CRBA."""

    @pytest.fixture(scope="class")
    def pin_model(self, urdf_string):
        """Builds the Pinocchio model for a floating-base robot."""
        model = pin.buildModelFromXML(urdf_string, pin.JointModelFreeFlyer())
        return model, model.createData()

    def test_rnea_full(self, urdf_string, pin_model, dtype, device):
        """Verifies full RNEA (gravity, coriolis, inertia) against Pinocchio."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model

        torch.manual_seed(2000)

        # Bard state vectors
        q_bard = torch.randn(bard_chain.nq, device=device, dtype=dtype)
        q_bard[3:7] = q_bard[3:7] / torch.linalg.norm(q_bard[3:7])  # Normalize quaternion
        qd_bard = torch.randn(bard_chain.nv, device=device, dtype=dtype)
        qdd_bard = torch.randn(bard_chain.nv, device=device, dtype=dtype)

        # Pinocchio state vectors
        q_pin = np.concatenate(
            [
                q_bard[:3].cpu().numpy(),
                q_bard[4:7].cpu().numpy(),
                q_bard[3:4].cpu().numpy(),
                q_bard[7:].cpu().numpy(),
            ]
        )
        qd_pin, qdd_pin = qd_bard.cpu().numpy(), qdd_bard.cpu().numpy()

        # Compare full RNEA
        tau_bard = calc_inverse_dynamics(bard_chain, q_bard, qd_bard, qdd_bard).cpu().numpy()[0]
        tau_pin = pin.rnea(pin_model_obj, pin_data, q_pin, qd_pin, qdd_pin)

        tol = 1e-5 if dtype == torch.float32 else 1e-7
        assert np.allclose(tau_bard, tau_pin, atol=tol), "Full RNEA mismatch"

    def test_crba_and_consistency(self, urdf_string, pin_model, dtype, device):
        """Verifies CRBA mass matrix and consistency with RNEA for a floating base."""
        bard_chain = build_chain_from_urdf(urdf_string, floating_base=True).to(
            dtype=dtype, device=device
        )
        pin_model_obj, pin_data = pin_model

        torch.manual_seed(2001)
        q_bard = torch.randn(bard_chain.nq, device=device, dtype=dtype)
        q_bard[3:7] = q_bard[3:7] / torch.linalg.norm(q_bard[3:7])
        qdd_bard = torch.randn(bard_chain.nv, device=device, dtype=dtype)

        # Convert q for Pinocchio
        q_pin = np.concatenate(
            [
                q_bard[:3].cpu().numpy(),
                q_bard[4:7].cpu().numpy(),
                q_bard[3:4].cpu().numpy(),
                q_bard[7:].cpu().numpy(),
            ]
        )

        # CRBA check
        M_bard = crba_inertia_matrix(bard_chain, q_bard).cpu().numpy()[0]
        M_pin = pin.crba(pin_model_obj, pin_data, q_pin)

        tol_crba = 1e-4 if dtype == torch.float32 else 1e-7
        assert np.allclose(M_bard, M_pin, atol=tol_crba), "CRBA mass matrix mismatch"

        # Consistency check: M*qdd = RNEA(q, qd=0, qdd, g=0)
        tau_rnea_inertia = calc_inverse_dynamics(
            bard_chain, q_bard, torch.zeros_like(qdd_bard), qdd_bard, gravity=torch.zeros(3)
        )
        tau_from_crba = torch.from_numpy(M_bard).to(device, dtype) @ qdd_bard

        tol_consistency = 1e-5 if dtype == torch.float32 else 1e-7
        assert torch.allclose(
            tau_rnea_inertia[0], tau_from_crba, atol=tol_consistency
        ), "RNEA and CRBA are not consistent"
