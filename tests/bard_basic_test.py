"""
Basic test script for spine.urdf using the `bard` library.
This mirrors the C library and Pinocchio test cases for verification.

Tests:
1. Zero configuration - FK, Jacobian, and Spatial Acceleration
2. Non-zero joint configuration
3. With base translation
4. With velocities and accelerations
"""

import os
import torch
import numpy as np

from bard.parsers.urdf import build_chain_from_urdf
from bard import ForwardKinematics, Jacobian, SpatialAcceleration
import bard.transforms as tf

# Path to URDF
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "spine.urdf")

# Global settings
DTYPE = torch.float64
DEVICE = "cpu"


def print_separator(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def comprehensive_test():
    """
    Main test function mirroring the C library test cases using the `bard` library.
    """
    print_separator("Test Suite for Bard Library Validation")

    # Load model with a floating base
    chain = build_chain_from_urdf(URDF_PATH, floating_base=True).to(dtype=DTYPE, device=DEVICE)

    # Instantiate bard modules
    fk_module = ForwardKinematics(chain, max_batch_size=1, compile_enabled=False)
    jac_module = Jacobian(chain, max_batch_size=1, compile_enabled=False)
    acc_module = SpatialAcceleration(chain, max_batch_size=1, compile_enabled=False)

    # Get frame IDs for testing
    frame_names = ["hind_spine", "front_spine", "front_body"]
    frame_ids = [chain.get_frame_id(name) for name in frame_names]

    # --- Test Case 1: Zero Configuration ---
    print_subsection("Test Case 1: Zero Configuration")
    # q = [tx, ty, tz, qw, qx, qy, qz, joint1, joint2, joint3]
    q1 = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=DTYPE, device=DEVICE)
    qd1 = torch.zeros(1, 6 + chain.n_joints, dtype=DTYPE, device=DEVICE)
    qdd1 = torch.zeros(1, 6 + chain.n_joints, dtype=DTYPE, device=DEVICE)
    
    print_state(q1, qd1, qdd1)
    run_and_print_kinematics(fk_module, jac_module, acc_module, q1, qd1, qdd1, frame_ids, frame_names)

    # --- Test Case 2: Non-zero Configuration ---
    print_subsection("Test Case 2: Non-zero Configuration")
    q2 = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0.1, 0.2, 0.3]], dtype=DTYPE, device=DEVICE)
    qd2 = torch.zeros(1, 6 + chain.n_joints, dtype=DTYPE, device=DEVICE)
    qdd2 = torch.zeros(1, 6 + chain.n_joints, dtype=DTYPE, device=DEVICE)

    print_state(q2, qd2, qdd2)
    run_and_print_kinematics(fk_module, jac_module, acc_module, q2, qd2, qdd2, frame_ids, frame_names)

    # --- Test Case 3: With Base Translation ---
    print_subsection("Test Case 3: With Base Translation")
    q3 = torch.tensor([[0.5, 0.3, 0.2, 1, 0, 0, 0, 0.15, -0.1, 0.25]], dtype=DTYPE, device=DEVICE)
    qd3 = torch.zeros(1, 6 + chain.n_joints, dtype=DTYPE, device=DEVICE)
    qdd3 = torch.zeros(1, 6 + chain.n_joints, dtype=DTYPE, device=DEVICE)

    print_state(q3, qd3, qdd3)
    run_and_print_kinematics(fk_module, jac_module, acc_module, q3, qd3, qdd3, [frame_ids[-1]], [frame_names[-1]])

    # --- Test Case 4: With Velocities and Accelerations ---
    print_subsection("Test Case 4: With Velocities and Accelerations")
    q4 = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0.2, 0.1, -0.15]], dtype=DTYPE, device=DEVICE)
    
    # qd = [base_linear, base_angular, joint_vels]
    qd4 = torch.tensor([[0.1, 0.05, 0, 0, 0, 0.1, 0.5, 0.3, 0.2]], dtype=DTYPE, device=DEVICE)
    
    # qdd = [base_linear, base_angular, joint_accels]
    qdd4 = torch.tensor([[0.01, 0, 0, 0, 0, 0.05, 0.1, 0.05, 0.08]], dtype=DTYPE, device=DEVICE)

    print_state(q4, qd4, qdd4)
    run_and_print_kinematics(fk_module, jac_module, acc_module, q4, qd4, qdd4, [frame_ids[-1]], [frame_names[-1]])


def run_and_print_kinematics(fk, jac, acc, q, qd, qdd, frame_ids, frame_names):
    """Helper to run all bard kinematics calculations and print results."""
    for i, frame_id in enumerate(frame_ids):
        frame_name = frame_names[i]
        print(f"\n  Frame {frame_id} ({frame_name}):")

        # 1. Forward Kinematics
        T_matrix = fk.calc(q, frame_id)[0].cpu().numpy()
        print("    FK (4x4 row-major):")
        print_matrix(T_matrix)

        # 2. Jacobian
        J = jac.calc(q, frame_id, reference_frame="world")[0].cpu().numpy()
        print("    Jacobian (6xNV, world):")
        print_matrix(J)

        # 3. Spatial Acceleration
        accel = acc.calc(q, qd, qdd, frame_id, reference_frame="world")[0].cpu().numpy()
        print(f"    Spatial Accel (world): [{accel[0]:.6f}, {accel[1]:.6f}, {accel[2]:.6f}, "
              f"{accel[3]:.6f}, {accel[4]:.6f}, {accel[5]:.6f}]")


def print_state(q, qd, qdd):
    """Prints the full robot state (q, v, a)."""
    q_np = q[0].cpu().numpy()
    v = qd[0].cpu().numpy()
    a = qdd[0].cpu().numpy()
    
    # q = [tx, ty, tz, qw, qx, qy, qz, joint1, joint2, joint3]
    print(f"q_base:   [{q_np[0]:.4f}, {q_np[1]:.4f}, {q_np[2]:.4f}, {q_np[3]:.4f}, {q_np[4]:.4f}, {q_np[5]:.4f}, {q_np[6]:.4f}]")
    print(f"q_joints: [{q_np[7]:.4f}, {q_np[8]:.4f}, {q_np[9]:.4f}]")
    
    print(f"qd_base:  [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}, {v[4]:.4f}, {v[5]:.4f}]")
    print(f"qd_joints:[{v[6]:.4f}, {v[7]:.4f}, {v[8]:.4f}]")
    
    print(f"qdd_base: [{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}, {a[3]:.4f}, {a[4]:.4f}, {a[5]:.4f}]")
    print(f"qdd_joints:[{a[6]:.4f}, {a[7]:.4f}, {a[8]:.4f}]")


def print_matrix(mat):
    """Prints a matrix with consistent formatting."""
    for r in range(mat.shape[0]):
        row_str = "      "
        for c in range(mat.shape[1]):
            row_str += f"{mat[r, c]:10.6f} "
        print(row_str)


if __name__ == "__main__":
    comprehensive_test()
