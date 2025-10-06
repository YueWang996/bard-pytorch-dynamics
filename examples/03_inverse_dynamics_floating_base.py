import torch
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.dynamics import calc_inverse_dynamics

def main():
    """
    An example of computing inverse dynamics (RNEA) for a floating-base robot.
    """
    try:
        with open("examples/simple_arm.urdf", "r") as f:
            urdf_string = f.read()
    except FileNotFoundError:
        print("Error: test_robot.urdf not found. Please provide a valid path.")
        return

    # 1. Build the chain with floating_base=True
    chain = build_chain_from_urdf(urdf_string, floating_base=True).to(dtype=torch.float64)
    print(f"Floating-base robot loaded. nq={chain.nq}, nv={chain.nv}")

    # 2. Define a complete state for the robot
    # Configuration `q`: [tx, ty, tz, qw, qx, qy, qz, joint_angles...]
    q_base = torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64) # Near-identity pose
    q_joints = torch.zeros(chain.n_joints, dtype=torch.float64)
    q = torch.cat([q_base, q_joints])
    
    # Velocity `qd`: [vx, vy, vz, wx, wy, wz, joint_velocities...]
    qd = torch.randn(chain.nv, dtype=torch.float64)
    
    # Acceleration `qdd`: [ax, ay, az, alphax, alphay, alphaz, joint_accelerations...]
    qdd = torch.randn(chain.nv, dtype=torch.float64)

    # 3. Define world gravity
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)

    # 4. Compute inverse dynamics with RNEA
    # This calculates the forces/torques required to achieve the given accelerations
    tau = calc_inverse_dynamics(chain, q, qd, qdd, gravity=gravity)[0]

    # 5. Interpret the results
    base_wrench = tau[:6]  # First 6 elements are the force and torque on the base
    joint_torques = tau[6:]

    print("\n--- Inverse Dynamics Results ---")
    print(f"Computed Base Wrench (Force, Torque): \n{base_wrench.numpy()}")
    print(f"\nComputed Joint Torques: \n{joint_torques.numpy()}")

if __name__ == "__main__":
    main()