from pathlib import Path
import torch
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.dynamics import RNEA

script_dir = Path(__file__).parent
urdf_path = script_dir / "example_robots/go2_description/urdf/go2.urdf"


def main():
    """
    An example of computing inverse dynamics (RNEA) for a floating-base robot.
    """

    # 1. Build the chain with floating_base=True
    chain = build_chain_from_urdf(urdf_path, floating_base=True).to(dtype=torch.float64)
    print(f"Floating-base robot loaded. nq={chain.nq}, nv={chain.nv}")

    # Instantiate the RNEA class once
    rnea = RNEA(chain, max_batch_size=1)

    # 2. Define a complete state for the robot
    q_base = torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q_joints = torch.zeros(chain.n_joints, dtype=torch.float64)
    q = torch.cat([q_base, q_joints])
    qd = torch.randn(chain.nv, dtype=torch.float64)
    qdd = torch.randn(chain.nv, dtype=torch.float64)

    # Add a batch dimension to all inputs
    q_batch, qd_batch, qdd_batch = q.unsqueeze(0), qd.unsqueeze(0), qdd.unsqueeze(0)

    # 3. Define world gravity
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)

    # 4. Compute inverse dynamics with RNEA
    tau_batch = rnea.calc(q_batch, qd_batch, qdd_batch, gravity=gravity)

    # 5. Interpret the results by extracting the single entry from the batch
    tau = tau_batch[0]
    base_wrench = tau[:6]
    joint_torques = tau[6:]

    print("\n--- Inverse Dynamics Results ---")
    print(f"Computed Base Wrench (Force, Torque): \n{base_wrench.numpy()}")
    print(f"\nComputed Joint Torques: \n{joint_torques.numpy()}")


if __name__ == "__main__":
    main()
