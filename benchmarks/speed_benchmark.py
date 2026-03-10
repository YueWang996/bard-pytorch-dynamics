"""
Experiment 4B: Speed benchmark — bard (GPU/CPU) vs Pinocchio (CPU).

Measures throughput (samples/sec) for FK, Jacobian, RNEA, CRBA, ABA, and
a Combined workflow (FK+Jacobian+RNEA with cached kinematics).

Usage:
    python benchmarks/speed_benchmark.py --device cuda
    python benchmarks/speed_benchmark.py --device cpu --robots go2 xarm7
    python benchmarks/speed_benchmark.py --device cuda --batch-sizes 1 64 256 1024 4096
"""

import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import bard

try:
    import pinocchio as pin

    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

ASSETS_DIR = Path(__file__).parent / "assets"

ROBOT_REGISTRY = {
    "go2": {
        "urdf": ASSETS_DIR / "go2_description/urdf/go2.urdf",
        "floating_base": True,
        "label": "Go2 (12-DOF)",
    },
    "h1": {
        "urdf": ASSETS_DIR / "h1_description/urdf/h1.urdf",
        "floating_base": True,
        "label": "H1 (19-DOF)",
    },
    "g1": {
        "urdf": ASSETS_DIR / "g1_description/urdf/g1_23dof.urdf",
        "floating_base": True,
        "label": "G1 (23-DOF)",
    },
    "xarm7": {
        "urdf": ASSETS_DIR / "xarm_description/urdf/xarm7.urdf",
        "floating_base": False,
        "label": "xArm7 (7-DOF)",
    },
    "dog11": {
        "urdf": ASSETS_DIR / "dog_11dof/urdf/spined_dog.urdf",
        "floating_base": True,
        "label": "Dog (11-DOF)",
    },
}

DEFAULT_BATCH_SIZES = [1, 16, 64, 256, 1024, 4096, 16384]
NUM_REPEATS = 100
WARMUP_ITERS = 20


def generate_random_q(model, B, device, dtype, floating_base):
    """Generate random joint configurations."""
    q = torch.randn(B, model.nq, device=device, dtype=dtype)
    if floating_base:
        q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)
    return q


def bard_q_to_pin_list(q_tensor, floating_base):
    """Convert a batch of bard q tensors to list of Pinocchio q arrays."""
    B = q_tensor.shape[0]
    q_np = q_tensor.detach().cpu().numpy()
    out = []
    for i in range(B):
        if floating_base:
            qi = q_np[i]
            out.append(np.concatenate([qi[:3], qi[4:7], qi[3:4], qi[7:]]))
        else:
            out.append(q_np[i].copy())
    return out


def sync_if_cuda(device):
    if "cuda" in str(device):
        torch.cuda.synchronize()


def time_fn(fn, n_repeats, device):
    """Time a function, returning array of elapsed times."""
    sync_if_cuda(device)
    times = []
    for _ in range(n_repeats):
        sync_if_cuda(device)
        t0 = time.perf_counter()
        fn()
        sync_if_cuda(device)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def bench_robot(robot_name, robot_info, device, dtype, batch_sizes, n_repeats):
    """Run speed benchmarks for a single robot, all algorithms, all batch sizes."""
    urdf_path = robot_info["urdf"]
    floating_base = robot_info["floating_base"]

    if not urdf_path.exists():
        print(f"  [SKIP] URDF not found: {urdf_path}")
        return None

    # Load bard model
    bard_model = bard.build_model_from_urdf(str(urdf_path), floating_base=floating_base).to(
        dtype=dtype, device=device
    )

    # Load Pinocchio model
    if floating_base:
        pin_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    else:
        pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_data = pin_model.createData()

    nv = bard_model.nv
    frame_names = bard_model.get_frame_names(exclude_fixed=True)
    test_frame = frame_names[-1] if frame_names else bard_model.get_frame_names()[0]
    bard_fid = bard_model.get_frame_id(test_frame)
    pin_fid = pin_model.getFrameId(test_frame)

    algorithms = ["FK", "Jacobian", "RNEA", "CRBA", "ABA", "Combined"]
    results = {algo: {} for algo in algorithms}

    max_batch = max(batch_sizes)
    bard_data = bard.create_data(bard_model, max_batch_size=max_batch)

    for B in batch_sizes:
        print(f"    batch_size={B}...", end="", flush=True)

        q = generate_random_q(bard_model, B, device, dtype, floating_base)
        qd = torch.randn(B, nv, device=device, dtype=dtype)
        qdd = torch.randn(B, nv, device=device, dtype=dtype)
        tau = torch.randn(B, nv, device=device, dtype=dtype)
        q_pin_list = bard_q_to_pin_list(q, floating_base)
        qd_np = qd.detach().cpu().numpy()
        qdd_np = qdd.detach().cpu().numpy()
        tau_np = tau.detach().cpu().numpy()

        # Pre-compute kinematics for cached algorithms
        bard.update_kinematics(bard_model, bard_data, q, qd)

        # --- FK ---
        # Warmup
        for _ in range(WARMUP_ITERS):
            bard.forward_kinematics(bard_model, bard_data, bard_fid, q=q)

        t_bard = time_fn(
            lambda: bard.forward_kinematics(bard_model, bard_data, bard_fid, q=q),
            n_repeats,
            device,
        )

        def pin_fk():
            for i in range(B):
                pin.framesForwardKinematics(pin_model, pin_data, q_pin_list[i])

        t_pin = time_fn(pin_fk, n_repeats, "cpu")
        results["FK"][B] = {"bard": t_bard, "pinocchio": t_pin}

        # --- Jacobian (cached) ---
        for _ in range(WARMUP_ITERS):
            bard.jacobian(bard_model, bard_data, bard_fid)

        t_bard = time_fn(
            lambda: bard.jacobian(bard_model, bard_data, bard_fid),
            n_repeats,
            device,
        )

        def pin_jac():
            for i in range(B):
                pin.computeJointJacobians(pin_model, pin_data, q_pin_list[i])
                pin.framesForwardKinematics(pin_model, pin_data, q_pin_list[i])
                pin.getFrameJacobian(pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD)

        t_pin = time_fn(pin_jac, n_repeats, "cpu")
        results["Jacobian"][B] = {"bard": t_bard, "pinocchio": t_pin}

        # --- RNEA (cached) ---
        for _ in range(WARMUP_ITERS):
            bard.rnea(bard_model, bard_data, qdd)

        t_bard = time_fn(
            lambda: bard.rnea(bard_model, bard_data, qdd),
            n_repeats,
            device,
        )

        def pin_rnea():
            for i in range(B):
                pin.rnea(pin_model, pin_data, q_pin_list[i], qd_np[i], qdd_np[i])

        t_pin = time_fn(pin_rnea, n_repeats, "cpu")
        results["RNEA"][B] = {"bard": t_bard, "pinocchio": t_pin}

        # --- CRBA (cached) ---
        for _ in range(WARMUP_ITERS):
            bard.crba(bard_model, bard_data)

        t_bard = time_fn(
            lambda: bard.crba(bard_model, bard_data),
            n_repeats,
            device,
        )

        def pin_crba():
            for i in range(B):
                pin.crba(pin_model, pin_data, q_pin_list[i])

        t_pin = time_fn(pin_crba, n_repeats, "cpu")
        results["CRBA"][B] = {"bard": t_bard, "pinocchio": t_pin}

        # --- ABA (cached) ---
        for _ in range(WARMUP_ITERS):
            bard.aba(bard_model, bard_data, tau)

        t_bard = time_fn(
            lambda: bard.aba(bard_model, bard_data, tau),
            n_repeats,
            device,
        )

        def pin_aba():
            for i in range(B):
                pin.aba(pin_model, pin_data, q_pin_list[i], qd_np[i], tau_np[i])

        t_pin = time_fn(pin_aba, n_repeats, "cpu")
        results["ABA"][B] = {"bard": t_bard, "pinocchio": t_pin}

        # --- Combined: update_kinematics + FK + Jacobian + RNEA ---
        def bard_combined():
            bard.update_kinematics(bard_model, bard_data, q, qd)
            bard.forward_kinematics(bard_model, bard_data, bard_fid)
            bard.jacobian(bard_model, bard_data, bard_fid)
            bard.rnea(bard_model, bard_data, qdd)

        for _ in range(WARMUP_ITERS):
            bard_combined()

        t_bard = time_fn(bard_combined, n_repeats, device)

        def pin_combined():
            for i in range(B):
                pin.framesForwardKinematics(pin_model, pin_data, q_pin_list[i])
                pin.computeJointJacobians(pin_model, pin_data, q_pin_list[i])
                pin.getFrameJacobian(pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD)
                pin.rnea(pin_model, pin_data, q_pin_list[i], qd_np[i], qdd_np[i])

        t_pin = time_fn(pin_combined, n_repeats, "cpu")
        results["Combined"][B] = {"bard": t_bard, "pinocchio": t_pin}

        print(" done")

    return results


def print_summary_table(robot_name, robot_info, results, batch_sizes, device):
    """Print throughput and speedup table."""
    from tabulate import tabulate

    print(f"\n{'=' * 90}")
    print(f"Speed Benchmark: {robot_info['label']} | device={device}")
    print(f"{'=' * 90}")

    algorithms = ["FK", "Jacobian", "RNEA", "CRBA", "ABA", "Combined"]

    for algo in algorithms:
        print(f"\n  {algo}:")
        headers = ["batch", "bard (ms)", "pin (ms)", "throughput bard", "throughput pin", "speedup"]
        rows = []
        for B in batch_sizes:
            if B not in results[algo]:
                continue
            t_bard_mean = float(np.mean(results[algo][B]["bard"])) * 1000
            t_pin_mean = float(np.mean(results[algo][B]["pinocchio"])) * 1000

            tp_bard = B / (t_bard_mean / 1000) if t_bard_mean > 0 else float("inf")
            tp_pin = B / (t_pin_mean / 1000) if t_pin_mean > 0 else float("inf")
            speedup = t_pin_mean / t_bard_mean if t_bard_mean > 0 else float("inf")

            rows.append(
                [
                    B,
                    f"{t_bard_mean:.3f}",
                    f"{t_pin_mean:.3f}",
                    f"{tp_bard:.0f} s/s",
                    f"{tp_pin:.0f} s/s",
                    f"{speedup:.1f}x",
                ]
            )
        print(tabulate(rows, headers=headers, tablefmt="grid"))


def save_results_npz(all_results, device, output_dir):
    """Save raw timing data as .npz for reproducibility."""
    data = {}
    for robot_name, results in all_results.items():
        if results is None:
            continue
        for algo in results:
            for B in results[algo]:
                key_bard = f"{robot_name}_{algo}_B{B}_bard"
                key_pin = f"{robot_name}_{algo}_B{B}_pinocchio"
                data[key_bard] = results[algo][B]["bard"]
                data[key_pin] = results[algo][B]["pinocchio"]

    output_path = output_dir / f"speed_{device}.npz"
    np.savez(output_path, **data)
    print(f"\nRaw timing data saved to: {output_path}")


def main():
    if not HAS_PINOCCHIO:
        print("ERROR: Pinocchio is required for speed benchmarks.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Speed benchmark: bard vs Pinocchio")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=list(ROBOT_REGISTRY.keys()),
        choices=list(ROBOT_REGISTRY.keys()),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument("--n-repeats", type=int, default=NUM_REPEATS)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--save", action="store_true", help="Save raw timing data as .npz")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    print(f"Device: {device} | Dtype: {args.dtype} | Repeats: {args.n_repeats}")
    print(f"Batch sizes: {args.batch_sizes}")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}
    for robot_name in args.robots:
        robot_info = ROBOT_REGISTRY[robot_name]
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {robot_info['label']}...")
        print(f"{'=' * 60}")

        results = bench_robot(
            robot_name, robot_info, device, dtype, args.batch_sizes, args.n_repeats
        )
        all_results[robot_name] = results

        if results is not None:
            print_summary_table(robot_name, robot_info, results, args.batch_sizes, device)

    if args.save:
        save_results_npz(all_results, device, output_dir)


if __name__ == "__main__":
    main()
