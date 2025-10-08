"""
Simplified benchmark script for class-based CRBA API.

Compares Bard's batched CRBA against Pinocchio's sequential implementation.
"""

from pathlib import Path
import torch
import numpy as np
import pinocchio as pin
import time

from bard.parsers.urdf import build_chain_from_urdf
from bard import CRBA
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


def benchmark_bard(crba, q, num_repeats, warmup_iters):
    """Benchmark Bard CRBA."""
    # Warmup
    for _ in range(warmup_iters):
        _ = crba.calc(q)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_repeats):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = crba.calc(q)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return np.array(times)


def benchmark_pinocchio(model, data, q_list, num_repeats, warmup_iters):
    """Benchmark Pinocchio CRBA."""
    batch_size = len(q_list)

    # Warmup
    for _ in range(warmup_iters):
        for i in range(batch_size):
            _ = pin.crba(model, data, q_list[i])

    # Benchmark
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        for i in range(batch_size):
            _ = pin.crba(model, data, q_list[i])
        times.append(time.perf_counter() - start)

    return np.array(times)


def verify_correctness(crba, q, pin_model, pin_data, q_pin):
    """Verify that Bard and Pinocchio produce similar results."""
    # Compute mass matrix with both
    M_bard = crba.calc(q[:1])[0].cpu().numpy()
    M_pin = pin.crba(pin_model, pin_data, q_pin[0])

    # CRBA in Pinocchio only fills upper triangle, need to symmetrize
    M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T

    max_diff = np.abs(M_bard - M_pin).max()
    mean_diff = np.abs(M_bard - M_pin).mean()

    print(f"Max difference:  {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Check relative error for non-zero elements
    mask = np.abs(M_pin) > 1e-10
    if mask.any():
        rel_error = np.abs((M_bard[mask] - M_pin[mask]) / M_pin[mask]).max()
        print(f"Max relative error: {rel_error:.2e}")

    tolerance = 1e-6
    if max_diff > tolerance:
        print(f"WARNING: Results differ by more than {tolerance:.2e}")
        return False

    print("✓ Correctness verified")
    return True


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("CRBA Benchmark: Bard (Class-based) vs Pinocchio")
    print("=" * 70)

    # Load robot
    chain, pin_model, pin_data = load_robot()

    # Create CRBA object once with max batch size
    max_batch = max(BATCH_SIZES)
    crba = CRBA(chain, max_batch_size=max_batch)

    print(f"\nCRBA object created with max_batch_size={max_batch}")
    print(f"Robot: {chain.n_joints} joints, {chain.n_nodes} nodes")
    print(f"Mass matrix size: {chain.nv} x {chain.nv}")

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
        if batch_size == BATCH_SIZES[0]:
            print("\nVerifying correctness...")
            if not verify_correctness(crba, q, pin_model, pin_data, q_pin):
                print("ERROR: Correctness check failed!")
                return

        # Benchmark Bard
        print("\nBenchmarking Bard...")
        bard_times = benchmark_bard(crba, q, NUM_REPEATS, WARMUP_ITERS)
        bard_mean = np.mean(bard_times) * 1000  # Convert to ms
        bard_std = np.std(bard_times) * 1000

        # Benchmark Pinocchio
        print("Benchmarking Pinocchio...")
        pin_times = benchmark_pinocchio(pin_model, pin_data, q_pin, NUM_REPEATS, WARMUP_ITERS)
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

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
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
