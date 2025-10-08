"""
Simplified benchmark script for class-based Jacobian API.

Compares Bard's batched Jacobian against Pinocchio's sequential implementation.
"""

from pathlib import Path
import torch
import numpy as np
import pinocchio as pin
import time

from bard.parsers.urdf import build_chain_from_urdf
from bard import Jacobian
from benchconf import URDF_PATH, BATCH_SIZES, NUM_REPEATS, WARMUP_ITERS, DEVICE, DTYPE


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
    """Generate random configuration vectors."""
    # Bard format
    q = torch.randn(batch_size, chain.nq, device=DEVICE, dtype=DTYPE)
    # Normalize quaternions (indices 3:7 for floating base)
    q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)

    # Pinocchio format (convert quaternion from [qw,qx,qy,qz] to [qx,qy,qz,qw])
    q_pin = []
    for i in range(batch_size):
        q_i = q[i].cpu().numpy()
        q_pin.append(
            np.concatenate(
                [q_i[:3], q_i[4:7], q_i[3:4], q_i[7:]]  # tx, ty, tz  # qx, qy, qz  # qw  # joints
            )
        )

    return q, q_pin


# ============================================================================
# Benchmarking
# ============================================================================


def benchmark_bard(jac, q, frame_id, reference_frame, num_repeats, warmup_iters):
    """Benchmark Bard Jacobian."""
    # Warmup
    for _ in range(warmup_iters):
        _ = jac.calc(q, frame_id, reference_frame=reference_frame)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_repeats):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = jac.calc(q, frame_id, reference_frame=reference_frame)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return np.array(times)


def benchmark_pinocchio(model, data, q_list, frame_id, reference_frame, num_repeats, warmup_iters):
    """Benchmark Pinocchio Jacobian."""
    batch_size = len(q_list)

    # Pinocchio reference frame
    if reference_frame == "world":
        pin_ref = pin.ReferenceFrame.WORLD
    else:
        pin_ref = pin.ReferenceFrame.LOCAL

    # Warmup
    for _ in range(warmup_iters):
        for i in range(batch_size):
            pin.framesForwardKinematics(model, data, q_list[i])
            _ = pin.computeFrameJacobian(model, data, q_list[i], frame_id, pin_ref)

    # Benchmark
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        for i in range(batch_size):
            pin.framesForwardKinematics(model, data, q_list[i])
            _ = pin.computeFrameJacobian(model, data, q_list[i], frame_id, pin_ref)
        times.append(time.perf_counter() - start)

    return np.array(times)


def verify_correctness(
    jac, q, bard_frame_id, pin_frame_id, reference_frame, pin_model, pin_data, q_pin
):
    """Verify that Bard and Pinocchio produce similar results."""
    # Compute Jacobian with Bard
    J_bard = jac.calc(q[:1], bard_frame_id, reference_frame=reference_frame)[0].cpu().numpy()

    # Compute Jacobian with Pinocchio (IMPORTANT: must call framesForwardKinematics first!)
    pin.framesForwardKinematics(pin_model, pin_data, q_pin[0])

    # Get Jacobian in correct reference frame
    if reference_frame == "world":
        # World frame Jacobian
        J_pin = pin.computeFrameJacobian(
            pin_model, pin_data, q_pin[0], pin_frame_id, pin.ReferenceFrame.WORLD
        )
    else:
        # Local frame Jacobian
        J_pin = pin.computeFrameJacobian(
            pin_model, pin_data, q_pin[0], pin_frame_id, pin.ReferenceFrame.LOCAL
        )

    # Pinocchio returns a 6xN matrix, need to ensure same shape
    if J_pin.shape != J_bard.shape:
        print(f"Shape mismatch: Bard {J_bard.shape} vs Pinocchio {J_pin.shape}")
        return False

    max_diff = np.abs(J_bard - J_pin).max()
    mean_diff = np.abs(J_bard - J_pin).mean()

    print(f"Max difference:  {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Check relative error for non-zero elements
    mask = np.abs(J_pin) > 1e-10
    if mask.any():
        rel_error = np.abs((J_bard[mask] - J_pin[mask]) / J_pin[mask]).max()
        print(f"Max relative error: {rel_error:.2e}")

    # More lenient tolerance for floating point
    tolerance = 1e-5 if DTYPE == torch.float64 else 1e-4
    if max_diff > tolerance:
        print(f"WARNING: Results differ by more than {tolerance:.2e}")
        print(f"\nBard Jacobian sample:\n{J_bard[:, 6:]}")
        print(f"\nPinocchio Jacobian sample:\n{J_pin[:, 6:]}")
        return False

    print("✓ Correctness verified")
    return True


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("Jacobian Benchmark: Bard (Class-based) vs Pinocchio")
    print("=" * 70)

    # Load robot
    chain, pin_model, pin_data = load_robot()

    # Create Jacobian object once with max batch size
    max_batch = max(BATCH_SIZES)
    jac = Jacobian(chain, max_batch_size=max_batch)

    # Select test frame (end-effector)
    test_frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
    bard_frame_id = chain.get_frame_indices(test_frame_name).item()
    pin_frame_id = pin_model.getFrameId(test_frame_name)

    print(f"\nJacobian object created with max_batch_size={max_batch}")
    print(f"Robot: {chain.n_joints} joints, {chain.n_nodes} nodes")
    print(f"Test frame: {test_frame_name} (id={bard_frame_id})")
    print(f"Jacobian size: 6 x {chain.nv}")

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
            q, q_pin = generate_random_configuration(chain, batch_size)

            # Verify correctness on first batch
            if batch_size == BATCH_SIZES[0] and reference_frame == "world":
                print("\nVerifying correctness...")
                if not verify_correctness(
                    jac, q, bard_frame_id, pin_frame_id, reference_frame, pin_model, pin_data, q_pin
                ):
                    print("ERROR: Correctness check failed!")
                    return

            # Benchmark Bard
            print("\nBenchmarking Bard...")
            bard_times = benchmark_bard(
                jac, q, bard_frame_id, reference_frame, NUM_REPEATS, WARMUP_ITERS
            )
            bard_mean = np.mean(bard_times) * 1000  # Convert to ms
            bard_std = np.std(bard_times) * 1000

            # Benchmark Pinocchio
            print("Benchmarking Pinocchio...")
            pin_times = benchmark_pinocchio(
                pin_model, pin_data, q_pin, pin_frame_id, reference_frame, NUM_REPEATS, WARMUP_ITERS
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
