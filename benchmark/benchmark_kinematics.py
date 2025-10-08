"""
Simplified benchmark script for class-based Forward Kinematics API.

Compares Bard's batched FK against Pinocchio's sequential implementation.
"""

from pathlib import Path
import torch
import numpy as np
import pinocchio as pin
import time

from bard.parsers.urdf import build_chain_from_urdf
from bard import ForwardKinematics

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
    """Generate random configuration vectors."""
    # Bard format: [tx, ty, tz, qw, qx, qy, qz, joint_angles...]
    q = torch.randn(batch_size, chain.nq, device=DEVICE, dtype=DTYPE)
    # Normalize quaternions (indices 3:7 for floating base)
    q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)
    
    # Pinocchio format: [tx, ty, tz, qx, qy, qz, qw, joint_angles...]
    q_pin = []
    for i in range(batch_size):
        q_i = q[i].cpu().numpy()
        q_pin.append(np.concatenate([
            q_i[:3],           # tx, ty, tz
            q_i[4:7],          # qx, qy, qz
            q_i[3:4],          # qw
            q_i[7:]            # joints
        ]))
    
    return q, q_pin


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_bard(fk, q, frame_id, num_repeats, warmup_iters):
    """Benchmark Bard Forward Kinematics."""
    # Warmup
    for _ in range(warmup_iters):
        _ = fk.calc(q, frame_id)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_repeats):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = fk.calc(q, frame_id)
        
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        
        times.append(time.perf_counter() - start)
    
    return np.array(times)


def benchmark_pinocchio(model, data, q_list, frame_id, num_repeats, warmup_iters):
    """Benchmark Pinocchio Forward Kinematics."""
    batch_size = len(q_list)
    
    # Warmup
    for _ in range(warmup_iters):
        for i in range(batch_size):
            pin.framesForwardKinematics(model, data, q_list[i])
            _ = data.oMf[frame_id]
    
    # Benchmark
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        for i in range(batch_size):
            pin.framesForwardKinematics(model, data, q_list[i])
            _ = data.oMf[frame_id]
        times.append(time.perf_counter() - start)
    
    return np.array(times)


def verify_correctness(fk, q, bard_frame_id, pin_frame_id, pin_model, pin_data, q_pin):
    """Verify that Bard and Pinocchio produce similar results."""
    # Compute FK with Bard (now returns raw tensor)
    T_bard = fk.calc(q[:1], bard_frame_id)[0].cpu().numpy()

    # Compute FK with Pinocchio
    pin.framesForwardKinematics(pin_model, pin_data, q_pin[0])
    T_pin = pin_data.oMf[pin_frame_id].homogeneous
    
    # Compare transformation matrices
    if T_pin.shape != T_bard.shape:
        print(f"Shape mismatch: Bard {T_bard.shape} vs Pinocchio {T_pin.shape}")
        return False
    
    max_diff = np.abs(T_bard - T_pin).max()
    mean_diff = np.abs(T_bard - T_pin).mean()
    
    print(f"Max difference:  {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Check position error
    pos_bard = T_bard[:3, 3]
    pos_pin = T_pin[:3, 3]
    pos_error = np.linalg.norm(pos_bard - pos_pin)
    print(f"Position error:  {pos_error:.2e}")
    
    # Check rotation error (Frobenius norm of difference)
    rot_bard = T_bard[:3, :3]
    rot_pin = T_pin[:3, :3]
    rot_error = np.linalg.norm(rot_bard - rot_pin, 'fro')
    print(f"Rotation error:  {rot_error:.2e}")
    
    # More lenient tolerance for floating point
    tolerance = 1e-5 if DTYPE == torch.float64 else 1e-4
    if max_diff > tolerance:
        print(f"WARNING: Results differ by more than {tolerance:.2e}")
        print(f"\nBard Transform:\n{T_bard}")
        print(f"\nPinocchio Transform:\n{T_pin}")
        return False
    
    print("✓ Correctness verified")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Forward Kinematics Benchmark: Bard (Class-based) vs Pinocchio")
    print("=" * 70)
    
    # Load robot
    chain, pin_model, pin_data = load_robot()
    
    # Create ForwardKinematics object once with max batch size
    max_batch = max(BATCH_SIZES)
    fk = ForwardKinematics(chain, max_batch_size=max_batch)
    
    # Select test frame (end-effector)
    test_frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
    bard_frame_id = chain.get_frame_indices(test_frame_name).item()
    pin_frame_id = pin_model.getFrameId(test_frame_name)
    
    print(f"\nForwardKinematics object created with max_batch_size={max_batch}")
    print(f"Robot: {chain.n_joints} joints, {chain.n_nodes} nodes")
    print(f"Test frame: {test_frame_name} (id={bard_frame_id})")
    
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
            if not verify_correctness(fk, q, bard_frame_id, pin_frame_id, 
                                     pin_model, pin_data, q_pin):
                print("ERROR: Correctness check failed!")
                return
        
        # Benchmark Bard
        print("\nBenchmarking Bard...")
        bard_times = benchmark_bard(fk, q, bard_frame_id,
                                   NUM_REPEATS, WARMUP_ITERS)
        bard_mean = np.mean(bard_times) * 1000  # Convert to ms
        bard_std = np.std(bard_times) * 1000
        
        # Benchmark Pinocchio
        print("Benchmarking Pinocchio...")
        pin_times = benchmark_pinocchio(pin_model, pin_data, q_pin, 
                                       pin_frame_id,
                                       NUM_REPEATS, WARMUP_ITERS)
        pin_mean = np.mean(pin_times) * 1000  # Convert to ms
        pin_std = np.std(pin_times) * 1000
        
        # Calculate speedup
        speedup = pin_mean / bard_mean
        
        # Store results
        results.append({
            'batch': batch_size,
            'bard_mean': bard_mean,
            'bard_std': bard_std,
            'pin_mean': pin_mean,
            'pin_std': pin_std,
            'speedup': speedup
        })
        
        # Print results
        print(f"\nResults:")
        print(f"  Bard:      {bard_mean:7.2f} ± {bard_std:5.2f} ms")
        print(f"  Pinocchio: {pin_mean:7.2f} ± {pin_std:5.2f} ms")
        print(f"  Speedup:   {speedup:7.2f}x")
    
    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY - FORWARD KINEMATICS")
    print(f"{'=' * 70}")
    print(f"{'Batch':<8} {'Bard (ms)':<15} {'Pinocchio (ms)':<15} {'Speedup'}")
    print("─" * 70)
    for r in results:
        print(f"{r['batch']:<8} "
              f"{r['bard_mean']:6.2f} ± {r['bard_std']:5.2f}  "
              f"{r['pin_mean']:7.2f} ± {r['pin_std']:5.2f}  "
              f"{r['speedup']:7.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()