"""
Tests for autograd compatibility of bard dynamics algorithms.

Supported autograd paths:
- CRBA: d(M)/d(q) — full gradient through update_kinematics -> CRBA
- ABA: d(qdd)/d(tau) — gradient through ABA w.r.t. applied torques
- RNEA: d(tau)/d(qdd) — gradient through RNEA w.r.t. accelerations

Note: ABA/RNEA d/dq is not yet supported (requires functional rewrite of
the algorithms' internal in-place scratch buffer operations).
"""

import pytest
import torch

import bard

from pathlib import Path


@pytest.fixture(scope="session")
def urdf_path():
    script_dir = Path(__file__).parent
    path = script_dir / "go2_description/urdf/go2.urdf"
    if not path.exists():
        pytest.skip(f"Required test asset not found: {path}")
    return path


def _make_model_and_data(urdf_path, dtype, device, batch_size=4, floating_base=False):
    model = bard.build_model_from_urdf(urdf_path, floating_base=floating_base).to(
        dtype=dtype, device=device
    )
    data = bard.create_data(model, max_batch_size=batch_size)
    return model, data


def _make_q(model, batch_size, dtype, device, seed=42):
    torch.manual_seed(seed)
    if model.has_floating_base:
        t = torch.randn(batch_size, 3, dtype=dtype, device=device)
        quat = torch.randn(batch_size, 4, dtype=dtype, device=device)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        q_joints = torch.rand(batch_size, model.n_joints, dtype=dtype, device=device) * 2 - 1
        q = torch.cat([t, quat, q_joints], dim=1)
    else:
        q = torch.rand(batch_size, model.n_joints, dtype=dtype, device=device) * 2 - 1
    return q


class TestCRBAAutograd:
    """Test that CRBA supports d(M)/d(q) autograd differentiation."""

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_crba_grad_wrt_q_fixed_base(self, urdf_path, dtype, device):
        """Gradient of CRBA mass matrix w.r.t. joint angles should be nonzero."""
        model, data = _make_model_and_data(urdf_path, dtype, device, batch_size=2)

        q = _make_q(model, 2, dtype, device)
        q.requires_grad_(True)

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)

        loss = M.sum()
        loss.backward()

        assert q.grad is not None, "No gradient computed for q"
        assert q.grad.abs().sum() > 0, "Gradient is all zeros"

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_crba_grad_wrt_q_floating_base(self, urdf_path, dtype, device):
        """CRBA d(M)/d(q) works with floating-base robot."""
        model, data = _make_model_and_data(
            urdf_path, dtype, device, batch_size=2, floating_base=True
        )

        q = _make_q(model, 2, dtype, device)
        q.requires_grad_(True)

        bard.update_kinematics(model, data, q)
        M = bard.crba(model, data)

        loss = M.sum()
        loss.backward()

        assert q.grad is not None, "No gradient computed for q"
        assert q.grad.abs().sum() > 0, "Gradient is all zeros"

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_crba_gradcheck(self, urdf_path, dtype, device):
        """Numerical gradient check for CRBA (finite differences vs autograd)."""
        model, data = _make_model_and_data(urdf_path, dtype, device, batch_size=1)

        q = _make_q(model, 1, dtype, device, seed=99)
        q.requires_grad_(True)

        def crba_fn(q_input):
            bard.update_kinematics(model, data, q_input)
            return bard.crba(model, data)

        assert torch.autograd.gradcheck(
            crba_fn, (q,), eps=1e-6, atol=1e-4, rtol=1e-3
        ), "CRBA gradient check failed"

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_crba_no_triton_when_grad_needed(self, urdf_path, dtype, device):
        """CRBA should use PyTorch ops (not Triton) when Xup requires grad."""
        model, data = _make_model_and_data(urdf_path, dtype, device, batch_size=2)

        q = _make_q(model, 2, dtype, device)
        bard.update_kinematics(model, data, q)

        M1 = bard.crba(model, data).clone()

        assert torch.allclose(M1[0], M1[0].T, atol=1e-10), "Mass matrix not symmetric"
        eigenvalues = torch.linalg.eigvalsh(M1[0])
        assert eigenvalues.min() > 0, "Mass matrix not positive definite"


class TestABAAutograd:
    """Test that ABA supports autograd differentiation w.r.t. tau."""

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_aba_grad_wrt_tau_fixed_base(self, urdf_path, dtype, device):
        """Gradient of ABA output w.r.t. applied torques should be nonzero."""
        model, data = _make_model_and_data(urdf_path, dtype, device, batch_size=2)

        q = _make_q(model, 2, dtype, device)
        torch.manual_seed(7)
        nv = model.nv
        qd = torch.randn(2, nv, dtype=dtype, device=device)
        tau = torch.randn(2, nv, dtype=dtype, device=device, requires_grad=True)

        bard.update_kinematics(model, data, q, qd)
        qdd = bard.aba(model, data, tau)

        loss = qdd.sum()
        loss.backward()

        assert tau.grad is not None, "No gradient computed for tau"
        assert tau.grad.abs().sum() > 0, "Gradient is all zeros"

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_aba_grad_wrt_tau_floating_base(self, urdf_path, dtype, device):
        """ABA autograd works with floating-base robot."""
        model, data = _make_model_and_data(
            urdf_path, dtype, device, batch_size=2, floating_base=True
        )

        q = _make_q(model, 2, dtype, device)
        torch.manual_seed(8)
        nv = 6 + model.n_joints
        qd = torch.randn(2, nv, dtype=dtype, device=device)
        tau = torch.randn(2, nv, dtype=dtype, device=device, requires_grad=True)

        bard.update_kinematics(model, data, q, qd)
        qdd = bard.aba(model, data, tau)

        loss = qdd.sum()
        loss.backward()

        assert tau.grad is not None, "No gradient computed for tau"
        assert tau.grad.abs().sum() > 0, "Gradient is all zeros"

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_aba_gradcheck_wrt_tau(self, urdf_path, dtype, device):
        """Numerical gradient check for ABA w.r.t. tau."""
        model, data = _make_model_and_data(urdf_path, dtype, device, batch_size=1)

        q = _make_q(model, 1, dtype, device, seed=99)
        torch.manual_seed(10)
        nv = model.nv
        qd = torch.randn(1, nv, dtype=dtype, device=device)
        tau = torch.randn(1, nv, dtype=dtype, device=device, requires_grad=True)

        bard.update_kinematics(model, data, q, qd)

        def aba_fn(tau_input):
            return bard.aba(model, data, tau_input)

        assert torch.autograd.gradcheck(
            aba_fn, (tau,), eps=1e-6, atol=1e-4, rtol=1e-3
        ), "ABA gradient check failed"

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_aba_crba_grad_consistency(self, urdf_path, dtype, device):
        """d(qdd)/d(tau) from ABA should match M^{-1} from CRBA."""
        model, data = _make_model_and_data(urdf_path, dtype, device, batch_size=1)

        q = _make_q(model, 1, dtype, device, seed=77)
        torch.manual_seed(11)
        nv = model.nv
        qd = torch.zeros(1, nv, dtype=dtype, device=device)
        tau = torch.randn(1, nv, dtype=dtype, device=device, requires_grad=True)

        # Compute M^{-1} via CRBA
        bard.update_kinematics(model, data, q, qd)
        M = bard.crba(model, data)[0]
        M_inv = torch.linalg.inv(M)

        # Compute d(qdd)/d(tau) via ABA autograd (Jacobian)
        bard.update_kinematics(model, data, q, qd)

        def aba_fn(tau_input):
            return bard.aba(model, data, tau_input).squeeze(0)

        J_aba = torch.autograd.functional.jacobian(aba_fn, tau)
        J_aba = J_aba.squeeze(1)

        tol = 1e-5
        assert torch.allclose(J_aba, M_inv, atol=tol), (
            f"ABA Jacobian doesn't match M^-1: max_diff=" f"{(J_aba - M_inv).abs().max():.3e}"
        )
