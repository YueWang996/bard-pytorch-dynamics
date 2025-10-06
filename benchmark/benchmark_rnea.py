import torch._dynamo
import logging

# Configure to only show warnings and errors
torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = False

# Set logging level to WARNING to filter out INFO logs
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


import torch
import numpy as np
import pinocchio as pin
import time
import matplotlib.pyplot as plt

# --- Imports from your library ---
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.dynamics import calc_inverse_dynamics

# ============================================================================
# Benchmark Configuration
# ============================================================================
URDF_PATH = "/Users/justin/PycharmProjects/bard/tests/spined_dog_asset/spined_dog_no_foot.urdf"

# A list of batch sizes to test
BATCH_SIZES = [10, 100, 1000, 10000, 50000, 100000]

# Number of times to repeat each test to get an average
NUM_REPEATS = 20  # Increased for better statistics

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# DEVICE = "cpu"  # Uncomment to force CPU
if DEVICE == "mps":
    DTYPE = torch.float32  # MPS has limited float64 support
    print("WARNING: Using float32 on MPS. Pinocchio uses float64, so comparison is not exact.")
else:
    DTYPE = torch.float64  # Use float64 for fair comparison

# ============================================================================

def setup_bard_chain():
    """Loads the URDF and builds the bard Chain object."""
    try:
        with open(URDF_PATH, "r") as f:
            urdf_string = f.read()
    except FileNotFoundError:
        print(f"Error: URDF not found at {URDF_PATH}")
        exit()
    chain = build_chain_from_urdf(urdf_string, floating_base=True)
    return chain.to(dtype=DTYPE, device=DEVICE)

def setup_pinocchio_model():
    """Loads the URDF and builds the Pinocchio Model and Data objects."""
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    return model, model.createData()

def generate_batch_data(chain, batch_size):
    """Generates random state vectors for a given batch size."""
    # Generate torch tensors for bard
    q_bard = torch.randn(batch_size, chain.nq, device=DEVICE, dtype=DTYPE)
    q_bard[:, 3:7] = q_bard[:, 3:7] / torch.linalg.norm(q_bard[:, 3:7], dim=1, keepdim=True)
    qd_bard = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)
    qdd_bard = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)

    # Pre-convert to numpy (float64 for Pinocchio)
    # IMPORTANT: Move to CPU first if on MPS, since MPS doesn't support float64
    q_pin_list = []
    for i in range(batch_size):
        q_bard_i = q_bard[i].cpu()  # Move to CPU first
        if DTYPE == torch.float32:
            q_bard_i = q_bard_i.to(dtype=torch.float64)  # Then convert to float64
        
        # Pinocchio expects [x,y,z, qx,qy,qz,qw, joints...]
        q_pin = np.concatenate([
            q_bard_i[:3].numpy(),      # translation
            q_bard_i[4:7].numpy(),     # qx,qy,qz
            q_bard_i[3:4].numpy(),     # qw
            q_bard_i[7:].numpy()       # joints
        ])
        q_pin_list.append(q_pin)
    
    # Same for velocities and accelerations
    qd_bard_cpu = qd_bard.cpu()
    qdd_bard_cpu = qdd_bard.cpu()
    if DTYPE == torch.float32:
        qd_bard_cpu = qd_bard_cpu.to(dtype=torch.float64)
        qdd_bard_cpu = qdd_bard_cpu.to(dtype=torch.float64)
    
    qd_pin_np = qd_bard_cpu.numpy()
    qdd_pin_np = qdd_bard_cpu.numpy()

    return q_bard, qd_bard, qdd_bard, q_pin_list, qd_pin_np, qdd_pin_np

def verify_correctness(bard_chain, q_b, qd_b, qdd_b, pin_model, pin_data, q_p, qd_p, qdd_p):
    """Verify that Bard and Pinocchio produce similar results."""
    print("\nVerifying correctness...")
    
    # Compute with both libraries
    tau_bard = calc_inverse_dynamics(bard_chain, q_b[0:1], qd_b[0:1], qdd_b[0:1])
    tau_pin = pin.rnea(pin_model, pin_data, q_p[0], qd_p[0], qdd_p[0])
    
    # Convert to comparable format
    tau_bard_cpu = tau_bard[0].cpu().to(dtype=torch.float64).numpy()
    
    # Compare base wrench (first 6 elements)
    base_diff = np.abs(tau_bard_cpu[:6] - tau_pin[:6]).max()
    print(f"  Base wrench max difference: {base_diff:.2e}")
    
    # Compare joint torques
    joint_diff = np.abs(tau_bard_cpu[6:] - tau_pin[6:]).max()
    print(f"  Joint torques max difference: {joint_diff:.2e}")
    
    # Tolerance depends on dtype
    tolerance = 1e-3 if DTYPE == torch.float32 else 1e-6
    
    if base_diff > tolerance or joint_diff > tolerance:
        print(f"  WARNING: Results differ by more than tolerance ({tolerance:.2e})")
        return False
    else:
        print(f"  ✓ Results match within tolerance")
        return True

def benchmark_bard(chain, q, qd, qdd, num_repeats):
    """Benchmark Bard with explicit compilation and warm-up."""
    
    # Step 1: Explicit compilation (first call triggers torch.compile)
    print("  Compiling for this batch size...")
    _ = calc_inverse_dynamics(chain, q, qd, qdd)
    if DEVICE == "cuda": 
        torch.cuda.synchronize()
    elif DEVICE == "mps": 
        torch.mps.synchronize()
    
    # Step 2: Warm-up runs (cache warming, consistent GPU state)
    print("  Warming up...")
    for _ in range(5):  # More warm-up runs
        _ = calc_inverse_dynamics(chain, q, qd, qdd)
        if DEVICE == "cuda": 
            torch.cuda.synchronize()
        elif DEVICE == "mps": 
            torch.mps.synchronize()
    
    # Step 3: Actual benchmark
    print("  Benchmarking...")
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        _ = calc_inverse_dynamics(chain, q, qd, qdd)
        
        if DEVICE == "cuda": 
            torch.cuda.synchronize()
        elif DEVICE == "mps": 
            torch.mps.synchronize()
            
        end = time.perf_counter()
        times.append(end - start)
    
    return np.array(times)

def benchmark_pinocchio(model, data, q_list, qd_np, qdd_np, num_repeats):
    """Benchmark Pinocchio with proper warm-up."""
    batch_size = len(q_list)
    
    # Warm-up
    for i in range(min(10, batch_size)):
        _ = pin.rnea(model, data, q_list[i], qd_np[i], qdd_np[i])
    
    # Actual benchmark
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        for i in range(batch_size):
            _ = pin.rnea(model, data, q_list[i], qd_np[i], qdd_np[i])
        end = time.perf_counter()
        times.append(end - start)
    
    return np.array(times)

def main():
    print("=" * 70)
    print(f"RNEA Benchmark: Bard (Batched) vs Pinocchio (Sequential)")
    print("=" * 70)
    print(f"Device: {DEVICE.upper()}")
    print(f"Dtype: {DTYPE}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Repeats per batch: {NUM_REPEATS}")
    print("=" * 70)

    # --- Setup ---
    bard_chain = setup_bard_chain()
    pin_model, pin_data = setup_pinocchio_model()

    # Storage for results
    results = {
        'batch_sizes': [],
        'bard_mean': [],
        'bard_std': [],
        'pin_mean': [],
        'pin_std': [],
        'speedup': []
    }

    # --- Main Benchmark Loop ---
    for B in BATCH_SIZES:
        print(f"\n{'─' * 70}")
        print(f"Batch Size: {B}")
        print(f"{'─' * 70}")
        
        # Generate data
        q_b, qd_b, qdd_b, q_p, qd_p, qdd_p = generate_batch_data(bard_chain, B)
        
        # Verify correctness on first batch
        if B == BATCH_SIZES[0]:
            if not verify_correctness(bard_chain, q_b, qd_b, qdd_b, 
                                     pin_model, pin_data, q_p, qd_p, qdd_p):
                print("ERROR: Correctness check failed. Stopping benchmark.")
                return

        # Benchmark Bard
        print("\nBenchmarking Bard...")
        bard_times = benchmark_bard(bard_chain, q_b, qd_b, qdd_b, NUM_REPEATS)
        bard_mean = np.mean(bard_times)
        bard_std = np.std(bard_times)
        
        # Benchmark Pinocchio
        print("Benchmarking Pinocchio...")
        pin_times = benchmark_pinocchio(pin_model, pin_data, q_p, qd_p, qdd_p, NUM_REPEATS)
        pin_mean = np.mean(pin_times)
        pin_std = np.std(pin_times)
        
        # Calculate speedup
        speedup = pin_mean / bard_mean
        
        # Store results
        results['batch_sizes'].append(B)
        results['bard_mean'].append(bard_mean)
        results['bard_std'].append(bard_std)
        results['pin_mean'].append(pin_mean)
        results['pin_std'].append(pin_std)
        results['speedup'].append(speedup)
        
        # Print results
        print(f"\nResults:")
        print(f"  Bard:      {bard_mean*1000:8.2f} ± {bard_std*1000:6.2f} ms")
        print(f"  Pinocchio: {pin_mean*1000:8.2f} ± {pin_std*1000:6.2f} ms")
        print(f"  Speedup:   {speedup:8.2f}x")

    # --- Print Summary Table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Batch':<8} {'Bard (ms)':<15} {'Pinocchio (ms)':<15} {'Speedup':<10}")
    print("─" * 70)
    for i, B in enumerate(results['batch_sizes']):
        print(f"{B:<8} {results['bard_mean'][i]*1000:6.2f} ± {results['bard_std'][i]*1000:5.2f}  "
              f"{results['pin_mean'][i]*1000:7.2f} ± {results['pin_std'][i]*1000:5.2f}  "
              f"{results['speedup'][i]:7.2f}x")
    print("=" * 70)

    # --- Plotting Results ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Execution time
    ax1.errorbar(results['batch_sizes'], results['bard_mean'], 
                 yerr=results['bard_std'], fmt='o-', label='Bard (Batched)', capsize=5)
    ax1.errorbar(results['batch_sizes'], results['pin_mean'], 
                 yerr=results['pin_std'], fmt='o-', label='Pinocchio (Sequential)', capsize=5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time per Batch (seconds)')
    ax1.set_title(f'RNEA Performance Comparison ({DEVICE.upper()}, {DTYPE})')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 2: Speedup
    ax2.plot(results['batch_sizes'], results['speedup'], 'o-', color='green', linewidth=2)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xscale('log')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (Pinocchio time / Bard time)')
    ax2.set_title('Speedup vs Batch Size')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    
    output_filename = f"rnea_benchmark_{DEVICE}.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\nBenchmark plot saved to {output_filename}")


if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib not found. Please install it:")
        print("pip install matplotlib")
        exit()
    main()