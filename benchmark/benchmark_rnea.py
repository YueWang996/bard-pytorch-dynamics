# examples/benchmark_rnea.py

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
# Use your robot's URDF file
URDF_PATH = "/Users/justin/PycharmProjects/bard/tests/spined_dog_asset/spined_dog_no_foot.urdf"

# A list of batch sizes to test
BATCH_SIZES = [10, 100, 1000, 10000, 50000, 100000]

# Number of times to repeat each test to get an average
NUM_REPEATS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

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
    # Pinocchio works in float64, so we keep it that way for its part of the test
    return model, model.createData()

def generate_batch_data(chain, batch_size):
    """Generates random state vectors for a given batch size."""
    # Generate torch tensors for bard
    q_bard = torch.randn(batch_size, chain.nq, device=DEVICE, dtype=DTYPE)
    q_bard[:, 3:7] = q_bard[:, 3:7] / torch.linalg.norm(q_bard[:, 3:7], dim=1, keepdim=True)
    qd_bard = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)
    qdd_bard = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)

    # Pre-convert to numpy (float64 for Pinocchio)
    q_pin_list = []
    q_bard_64 = q_bard.to(dtype=DTYPE)
    for i in range(batch_size):
        q_bard_i = q_bard_64[i]
        q_pin = np.concatenate([
            q_bard_i[:3].cpu().numpy(), q_bard_i[4:7].cpu().numpy(),
            q_bard_i[3:4].cpu().numpy(), q_bard_i[7:].cpu().numpy()
        ])
        q_pin_list.append(q_pin)
    
    qd_pin_np = qd_bard.to(dtype=DTYPE).cpu().numpy()
    qdd_pin_np = qdd_bard.to(dtype=DTYPE).cpu().numpy()

    return q_bard, qd_bard, qdd_bard, q_pin_list, qd_pin_np, qdd_pin_np

def main():
    if DEVICE == "mps":
        print("Apple MPS backend detected. Using float32 for benchmark.")
    print(f"Starting RNEA benchmark on device: {DEVICE}")
    print(f"Comparing batch sizes: {BATCH_SIZES}\n")

    # --- Setup ---
    bard_chain = setup_bard_chain()
    pin_model, pin_data = setup_pinocchio_model()

    bard_times = []
    pinocchio_times = []

    # --- Main Benchmark Loop ---
    for B in BATCH_SIZES:
        q_b, qd_b, qdd_b, q_p, qd_p, qdd_p = generate_batch_data(bard_chain, B)

        # --- Benchmark bard (batched) ---
        # GPU/MPS warm-up: run once before timing to handle setup overhead
        if DEVICE in ["cuda", "mps"]:
            calc_inverse_dynamics(bard_chain, q_b, qd_b, qdd_b)
            if DEVICE == "cuda": torch.cuda.synchronize()
            if DEVICE == "mps": torch.mps.synchronize()

        start_time = time.perf_counter()
        for _ in range(NUM_REPEATS):
            calc_inverse_dynamics(bard_chain, q_b, qd_b, qdd_b)
        
        # Ensure all async operations are finished before stopping the timer
        if DEVICE == "cuda": torch.cuda.synchronize()
        if DEVICE == "mps": torch.mps.synchronize()
        end_time = time.perf_counter()
        bard_time = (end_time - start_time) / NUM_REPEATS
        bard_times.append(bard_time)

        # --- Benchmark Pinocchio (looped) ---
        start_time = time.perf_counter()
        for _ in range(NUM_REPEATS):
            for i in range(B):
                pin.rnea(pin_model, pin_data, q_p[i], qd_p[i], qdd_p[i])
        end_time = time.perf_counter()
        pinocchio_time = (end_time - start_time) / NUM_REPEATS
        pinocchio_times.append(pinocchio_time)

        speedup = pinocchio_time / bard_time
        print(f"Batch Size: {B:<5} | Bard: {bard_time*1000:.3f} ms | Pinocchio: {pinocchio_time*1000:.3f} ms | Speedup: {speedup:.2f}x")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(BATCH_SIZES, bard_times, 'o-', label='Bard (Batched)')
    plt.plot(BATCH_SIZES, pinocchio_times, 'o-', label='Pinocchio (Looped)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'RNEA Performance Comparison (Device: {DEVICE.upper()})')
    plt.xlabel('Batch Size')
    plt.ylabel('Time per Batch (seconds)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    output_filename = "rnea_benchmark_mps.png"
    plt.savefig(output_filename)
    print(f"\nBenchmark plot saved to {output_filename}")


if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib not found. Please install it to generate the plot:")
        print("pip install matplotlib")
        exit()
    main()