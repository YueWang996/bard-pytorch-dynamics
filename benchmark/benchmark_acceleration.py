"""
Simplified benchmark script for class-based Spatial Acceleration API.

Compares Bard's batched acceleration computation against Pinocchio's sequential implementation.
"""

from pathlib import Path
import torch
import numpy as np
import pinocchio as pin
import time

from bard.parsers.urdf import build_chain_from_urdf
from bard import SpatialAcceleration

# ============================================================================
# Configuration
# ============================================================================
script_dir = Path(__file__).parent
URDF_PATH = script_dir / "../tests/spined_dog_asset/spined_dog_no_foot.urdf"

BATCH_SIZES = [10, 100, 1000, 10000]
NUM_REPEATS = 100
WARMUP_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

print(f"Device: {DEVICE}, Dtype: {DTYPE}")

# ============================================================================
# Setup
# ============================================================================


def load_robot():
    """Load robot in both Bard and Pinocchio."""
    with open(URDF_PATH, "r") as f:
        urdf_string = f.read()

    # Bard
    chain = build_chain_from_urdf(urdf_string, floating_base=True)
    chain = chain.to(dtype=DTYPE, device=DEVICE)

    # Pinocchio
    pin_model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    pin_data = pin_model.createData()

    return chain, pin_model, pin_data


def generate_random_configuration(chain, batch_size):
    """Generate random configuration, velocity, and acceleration vectors."""
    # Bard format
    # Position: [tx, ty, tz, qw, qx, qy, qz, joint_angles...]
    q = torch.randn(batch_size, chain.nq, device=DEVICE, dtype=DTYPE)
    q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)

    # Velocity: [vx, vy, vz, wx, wy, wz, joint_velocities...]
    qd = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)

    # Acceleration: same shape as velocity
    qdd = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)

    # Pinocchio format (convert quaternion from [qw,qx,qy,qz] to [qx,qy,qz,qw])
    q_pin = []
    qd_pin = []
    qdd_pin = []

    for i in range(batch_size):
        q_i = q[i].cpu().numpy()
        qd_i = qd[i].cpu().numpy()
        qdd_i = qdd[i].cpu().numpy()

        # Position: reorder quaternion
        q_pin.append(
            np.concatenate(
                [q_i[:3], q_i[4:7], q_i[3:4], q_i[7:]]  # tx, ty, tz  # qx, qy, qz  # qw  # joints
            )
        )

        # Velocity and acceleration: no reordering needed (spatial velocity format)
        qd_pin.append(qd_i)
        qdd_pin.append(qdd_i)

    return q, qd, qdd, q_pin, qd_pin, qdd_pin


# ============================================================================
# Benchmarking
# ============================================================================


def benchmark_bard(accel, q, qd, qdd, frame_id, reference_frame, num_repeats, warmup_iters):
    """Benchmark Bard Spatial Acceleration."""
    # Warmup
    for _ in range(warmup_iters):
        _ = accel.calc(q, qd, qdd, frame_id, reference_frame=reference_frame)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_repeats):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = accel.calc(q, qd, qdd, frame_id, reference_frame=reference_frame)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return np.array(times)


def benchmark_pinocchio(
    model, data, q_list, qd_list, qdd_list, frame_id, reference_frame, num_repeats, warmup_iters
):
    """Benchmark Pinocchio Spatial Acceleration."""
    batch_size = len(q_list)

    # Pinocchio reference frame
    pin_ref_frame = (
        pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
    )

    # Warmup
    for _ in range(warmup_iters):
        for i in range(batch_size):
            pin.forwardKinematics(model, data, q_list[i], qd_list[i], qdd_list[i])
            _ = pin.getFrameAcceleration(model, data, frame_id, pin_ref_frame).vector

    # Benchmark
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        for i in range(batch_size):
            pin.forwardKinematics(model, data, q_list[i], qd_list[i], qdd_list[i])
            _ = pin.getFrameAcceleration(model, data, frame_id, pin_ref_frame).vector
        times.append(time.perf_counter() - start)

    return np.array(times)


def verify_correctness(
    accel,
    q,
    qd,
    qdd,
    bard_frame_id,
    pin_frame_id,
    reference_frame,
    pin_model,
    pin_data,
    q_pin,
    qd_pin,
    qdd_pin,
):
    """Verify that Bard and Pinocchio produce similar results."""
    # Compute acceleration with Bard
    a_bard = (
        accel.calc(q[:1], qd[:1], qdd[:1], bard_frame_id, reference_frame=reference_frame)[0]
        .cpu()
        .numpy()
    )

    # Compute acceleration with Pinocchio
    pin.forwardKinematics(pin_model, pin_data, q_pin[0], qd_pin[0], qdd_pin[0])
    pin_ref_frame = (
        pin.ReferenceFrame.WORLD if reference_frame == "world" else pin.ReferenceFrame.LOCAL
    )
    a_pin = pin.getFrameAcceleration(pin_model, pin_data, pin_frame_id, pin_ref_frame).vector

    # Compare acceleration vectors
    if a_pin.shape != a_bard.shape:
        print(f"Shape mismatch: Bard {a_bard.shape} vs Pinocchio {a_pin.shape}")
        return False

    max_diff = np.abs(a_bard - a_pin).max()
    mean_diff = np.abs(a_bard - a_pin).mean()

    print(f"Max difference:  {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Check linear and angular components separately
    linear_bard = a_bard[:3]
    linear_pin = a_pin[:3]
    angular_bard = a_bard[3:]
    angular_pin = a_pin[3:]

    linear_error = np.linalg.norm(linear_bard - linear_pin)
    angular_error = np.linalg.norm(angular_bard - angular_pin)

    print(f"Linear acceleration error:  {linear_error:.2e}")
    print(f"Angular acceleration error: {angular_error:.2e}")

    # Check relative error for non-zero elements
    mask = np.abs(a_pin) > 1e-10
    if mask.any():
        rel_error = np.abs((a_bard[mask] - a_pin[mask]) / a_pin[mask]).max()
        print(f"Max relative error: {rel_error:.2e}")

    # More lenient tolerance for floating point
    tolerance = 1e-5 if DTYPE == torch.float64 else 1e-4
    if max_diff > tolerance:
        print(f"WARNING: Results differ by more than {tolerance:.2e}")
        print(f"\nBard acceleration:\n{a_bard}")
        print(f"\nPinocchio acceleration:\n{a_pin}")
        return False

    print("✓ Correctness verified")
    return True


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("Spatial Acceleration Benchmark: Bard (Class-based) vs Pinocchio")
    print("=" * 70)

    # Load robot
    chain, pin_model, pin_data = load_robot()

    # Create SpatialAcceleration object once with max batch size
    max_batch = max(BATCH_SIZES)
    accel = SpatialAcceleration(chain, max_batch_size=max_batch)

    # Select test frame (end-effector)
    test_frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
    bard_frame_id = chain.get_frame_indices(test_frame_name).item()
    pin_frame_id = pin_model.getFrameId(test_frame_name)

    print(f"\nSpatialAcceleration object created with max_batch_size={max_batch}")
    print(f"Robot: {chain.n_joints} joints, {chain.n_nodes} nodes")
    print(f"Test frame: {test_frame_name} (id={bard_frame_id})")
    print(
        f"Acceleration size: (6,) - [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]"
    )

    # Test both reference frames
    for reference_frame in ["world", "local"]:
        print(f"\n{'=' * 70}")
        print(f"Reference Frame: {reference_frame.upper()}")
        print(f"{'=' * 70}")

        # Results storage
        results = []

        # Benchmark each batch size
        for batch_size in BATCH_SIZES:
            print(f"\n{'─' * 70}")
            print(f"Batch Size: {batch_size}")
            print(f"{'─' * 70}")

            # Generate data
            q, qd, qdd, q_pin, qd_pin, qdd_pin = generate_random_configuration(chain, batch_size)

            # Verify correctness on first batch
            if batch_size == BATCH_SIZES[0] and reference_frame == "world":
                print("\nVerifying correctness...")
                if not verify_correctness(
                    accel,
                    q,
                    qd,
                    qdd,
                    bard_frame_id,
                    pin_frame_id,
                    reference_frame,
                    pin_model,
                    pin_data,
                    q_pin,
                    qd_pin,
                    qdd_pin,
                ):
                    print("ERROR: Correctness check failed!")
                    return

            # Benchmark Bard
            print("\nBenchmarking Bard...")
            bard_times = benchmark_bard(
                accel, q, qd, qdd, bard_frame_id, reference_frame, NUM_REPEATS, WARMUP_ITERS
            )
            bard_mean = np.mean(bard_times) * 1000  # Convert to ms
            bard_std = np.std(bard_times) * 1000

            # Benchmark Pinocchio
            print("Benchmarking Pinocchio...")
            pin_times = benchmark_pinocchio(
                pin_model,
                pin_data,
                q_pin,
                qd_pin,
                qdd_pin,
                pin_frame_id,
                reference_frame,
                NUM_REPEATS,
                WARMUP_ITERS,
            )
            pin_mean = np.mean(pin_times) * 1000  # Convert to ms
            pin_std = np.std(pin_times) * 1000

            # Calculate speedup
            speedup = pin_mean / bard_mean

            # Store results
            results.append(
                {
                    "batch": batch_size,
                    "bard_mean": bard_mean,
                    "bard_std": bard_std,
                    "pin_mean": pin_mean,
                    "pin_std": pin_std,
                    "speedup": speedup,
                }
            )

            # Print results
            print(f"\nResults:")
            print(f"  Bard:      {bard_mean:7.2f} ± {bard_std:5.2f} ms")
            print(f"  Pinocchio: {pin_mean:7.2f} ± {pin_std:5.2f} ms")
            print(f"  Speedup:   {speedup:7.2f}x")

        # Print summary table for this reference frame
        print(f"\n{'=' * 70}")
        print(f"SUMMARY - {reference_frame.upper()} FRAME")
        print(f"{'=' * 70}")
        print(f"{'Batch':<8} {'Bard (ms)':<15} {'Pinocchio (ms)':<15} {'Speedup'}")
        print("─" * 70)
        for r in results:
            print(
                f"{r['batch']:<8} "
                f"{r['bard_mean']:6.2f} ± {r['bard_std']:5.2f}  "
                f"{r['pin_mean']:7.2f} ± {r['pin_std']:5.2f}  "
                f"{r['speedup']:7.2f}x"
            )
        print("=" * 70)


if __name__ == "__main__":
    main()
