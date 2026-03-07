"""
Basic test script for spine.urdf using Pinocchio only.
This mirrors the C library test cases for verification.

Tests:
1. Zero configuration - FK and gravity compensation
2. Base pitched 90 degrees - FK and torques
3. Bent joints (30 deg each) - FK and gravity compensation
4. CRBA Mass Matrix computation
5. Forward Dynamics (free fall)
"""

import os
import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None

# Path to URDF
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "spine.urdf")


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
    Main test function mirroring the updated C library test cases.
    Tests FK, Jacobian, RNEA, CRBA, gravity compensation, and spatial acceleration.
    """
    print_separator("Test Suite for Pinocchio Validation")

    # Load model with a floating base
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()

    print(f"Robot: {model.name}")
    print(f"Links: {model.nframes}, Joints: {model.njoints}, DOF: {model.nv}\n")

    # Print frames
    print("Frames:")
    for i in range(model.nframes):
        frame = model.frames[i]
        print(f"  [{i}] {frame.name} (parent={frame.parent}, jidx={frame.parentJoint})")
    print("")

    # End-effector frame
    frame_name = "front_body"
    frame_id = model.getFrameId(frame_name)

    # Test configuration
    # C-style quaternion: [qw, qx, qy, qz]
    # Pinocchio: [x, y, z, qx, qy, qz, qw, joints...]
    q = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.1, -0.1, 0.05])
    v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 0.2])
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print_state(q, v, a, model)

    # ===== Forward Kinematics =====
    print_subsection("Forward Kinematics")
    pin.forwardKinematics(model, data, q, v, a)
    pin.updateFramePlacements(model, data)

    for i in range(model.nframes):
        T = data.oMf[i].homogeneous
        print(
            f"Frame {i} ({model.frames[i].name}) position: "
            f"[{T[0, 3]:.4f}, {T[1, 3]:.4f}, {T[2, 3]:.4f}]"
        )

    # ===== Jacobian =====
    print_subsection("Jacobian (end frame, world)")
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.WORLD)
    print_matrix("J", J, 6, model.nv)

    # ===== Inverse Dynamics (RNEA) =====
    print_subsection("Inverse Dynamics (RNEA)")
    tau = pin.rnea(model, data, q, v, a)
    print_vec("tau (base)", tau[:6])
    print_vec("tau (joints)", tau[6:])

    # ===== Mass Matrix (CRBA) =====
    print_subsection("Mass Matrix (CRBA)")
    M = pin.crba(model, data, q)
    M = np.triu(M) + np.triu(M, 1).T
    print_matrix("M", M, model.nv, model.nv)

    # ===== Gravity Compensation =====
    print_subsection("Gravity Compensation")
    tau_g = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
    print_vec("tau_g (joints)", tau_g[6:])

    # ===== Spatial Acceleration =====
    print_subsection("Spatial Acceleration")
    accel = pin.getFrameAcceleration(model, data, frame_id, pin.ReferenceFrame.WORLD)
    print_vec("a_world", accel.vector)


def run_and_print_kinematics(model, data, q, v, a, frame_id, frame_name):
    """Deprecated helper (kept for compatibility)."""
    pin.forwardKinematics(model, data, q, v, a)
    pin.updateFramePlacements(model, data)
    T = data.oMf[frame_id].homogeneous
    print(f"\n  Frame {frame_id} ({frame_name}):")
    print("    FK (4x4 column-major, OpenGL style):")
    for r in range(4):
        row_str = "      "
        for c in range(4):
            row_str += f"{T[r, c]:10.6f} "
        print(row_str)


def print_state(q, v, a, model):
    """Prints the full robot state (q, v, a)."""
    # C-style: [x, y, z, qw, qx, qy, qz]
    # Pinocchio: [x, y, z, qx, qy, qz, qw]
    q_base_c_order = np.array([q[0], q[1], q[2], q[6], q[3], q[4], q[5]])

    print(
        f"q_base:   [{q_base_c_order[0]:.4f}, {q_base_c_order[1]:.4f}, {q_base_c_order[2]:.4f}, "
        f"{q_base_c_order[3]:.4f}, {q_base_c_order[4]:.4f}, {q_base_c_order[5]:.4f}, {q_base_c_order[6]:.4f}]"
    )
    print(f"q_joints: [{q[7]:.4f}, {q[8]:.4f}, {q[9]:.4f}]")

    print(f"qd_base:  [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}, {v[4]:.4f}, {v[5]:.4f}]")
    print(f"qd_joints:[{v[6]:.4f}, {v[7]:.4f}, {v[8]:.4f}]")

    print(f"qdd_base: [{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}, {a[3]:.4f}, {a[4]:.4f}, {a[5]:.4f}]")
    print(f"qdd_joints:[{a[6]:.4f}, {a[7]:.4f}, {a[8]:.4f}]")


def print_matrix(name, mat, rows, cols):
    """Prints a matrix with consistent formatting."""
    print(f"{name}:")
    for r in range(rows):
        row_str = "  ["
        for c in range(cols):
            row_str += f"{mat[r, c]:8.4f}"
            if c < cols - 1:
                row_str += ", "
        row_str += "]"
        print(row_str)


def print_vec(name, vec):
    """Prints a vector with consistent formatting."""
    row_str = f"{name}: ["
    for i in range(len(vec)):
        row_str += f"{vec[i]:8.4f}"
        if i < len(vec) - 1:
            row_str += ", "
    row_str += "]"
    print(row_str)


def print_model_info():
    """Print detailed model information for debugging."""
    print_separator("Model Information")

    # Fixed base model
    model = pin.buildModelFromUrdf(URDF_PATH)
    print(f"Fixed-base model:")
    print(f"  Name: {model.name}")
    print(f"  nq: {model.nq}, nv: {model.nv}")
    print(f"  njoints: {model.njoints}")
    print(f"  gravity: {model.gravity}")

    print("\nJoint details:")
    for i in range(model.njoints):
        print(f"  Joint {i}: {model.names[i]}")
        if i > 0:
            placement = model.jointPlacements[i]
            print(
                f"    Placement: trans=[{placement.translation[0]:.4f}, {placement.translation[1]:.4f}, {placement.translation[2]:.4f}]"
            )

    print("\nFrame details:")
    for i in range(model.nframes):
        frame = model.frames[i]
        print(f"  Frame {i}: {frame.name} (type: {frame.type})")

    # Floating base model
    print("\n" + "=" * 50)
    model_float = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    print(f"Floating-base model:")
    print(f"  Name: {model_float.name}")
    print(f"  nq: {model_float.nq}, nv: {model_float.nv}")
    print(f"  njoints: {model_float.njoints}")


if __name__ == "__main__":
    if pin is None:
        raise SystemExit(
            "This script requires pinocchio. "
            "Install it with: conda install -c conda-forge pinocchio"
        )
    print("\n" + "=" * 60)
    print("  Pinocchio Verification Test for spine.urdf")
    print("  (Mirrors C Library Test Cases)")
    print("=" * 60)

    # Run comprehensive tests
    comprehensive_test()
