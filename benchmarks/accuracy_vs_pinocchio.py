"""
Experiment 4A: Accuracy verification of bard vs Pinocchio.

Tests FK, Jacobian, RNEA, CRBA, ABA across multiple robot models.
Outputs a formatted table suitable for the ICANN 2026 paper.

Usage:
    python benchmarks/accuracy_vs_pinocchio.py
    python benchmarks/accuracy_vs_pinocchio.py --robots go2 xarm7
    python benchmarks/accuracy_vs_pinocchio.py --n-samples 200
"""

import argparse
import sys
import os
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

# Registry of available robot models
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


def bard_q_to_pin_q(q_bard_np, floating_base):
    """Convert bard quaternion layout [tx,ty,tz, qw,qx,qy,qz, ...] to Pinocchio [tx,ty,tz, qx,qy,qz,qw, ...]."""
    if not floating_base:
        return q_bard_np.copy()
    return np.concatenate([q_bard_np[:3], q_bard_np[4:7], q_bard_np[3:4], q_bard_np[7:]])


def generate_random_configs(bard_model, pin_model, n_samples, floating_base):
    """Generate random (q, qd, qdd, tau) samples compatible with both bard and Pinocchio."""
    nv = bard_model.nv

    configs = []
    for _ in range(n_samples):
        if floating_base:
            # pin.randomConfiguration gives inf for unbounded free-flyer base
            # Manually generate base position + unit quaternion + joint angles
            base_pos = np.random.randn(3) * 0.5
            quat_raw = np.random.randn(4)
            quat_raw /= np.linalg.norm(quat_raw)  # [qx, qy, qz, qw] for Pinocchio

            # Get random joint angles within limits
            lower = pin_model.lowerPositionLimit[7:]
            upper = pin_model.upperPositionLimit[7:]
            joints = lower + np.random.rand(len(lower)) * (upper - lower)

            # Pinocchio layout: [tx, ty, tz, qx, qy, qz, qw, joints...]
            q_pin = np.concatenate([base_pos, quat_raw[:3], quat_raw[3:4], joints])
            # Bard layout: [tx, ty, tz, qw, qx, qy, qz, joints...]
            q_bard = np.concatenate([base_pos, quat_raw[3:4], quat_raw[:3], joints])
        else:
            q_pin = pin.randomConfiguration(pin_model)
            q_bard = q_pin.copy()

        qd = np.random.randn(nv) * 2.0
        qdd = np.random.randn(nv) * 2.0
        tau = np.random.randn(nv) * 10.0

        configs.append(
            {
                "q_bard": q_bard,
                "q_pin": q_pin,
                "qd": qd,
                "qdd": qdd,
                "tau": tau,
            }
        )
    return configs


def run_accuracy_test(robot_name, robot_info, n_samples, dtype):
    """Run accuracy comparison for a single robot model."""
    urdf_path = robot_info["urdf"]
    floating_base = robot_info["floating_base"]

    if not urdf_path.exists():
        print(f"  [SKIP] URDF not found: {urdf_path}")
        return None

    # Load models
    bard_model = bard.build_model_from_urdf(str(urdf_path), floating_base=floating_base).to(
        dtype=dtype, device="cpu"
    )
    bard_data = bard.create_data(bard_model, max_batch_size=1)

    if floating_base:
        pin_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    else:
        pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_data = pin_model.createData()

    nv = bard_model.nv

    # Generate random configs
    configs = generate_random_configs(bard_model, pin_model, n_samples, floating_base)

    # Pick a non-fixed frame for FK/Jacobian comparison
    frame_names = bard_model.get_frame_names(exclude_fixed=True)
    test_frame = frame_names[-1] if frame_names else bard_model.get_frame_names()[0]
    bard_fid = bard_model.get_frame_id(test_frame)
    pin_fid = pin_model.getFrameId(test_frame)

    # Collect errors
    errors = {"FK": [], "Jacobian": [], "RNEA": [], "CRBA": [], "ABA": []}

    for cfg in configs:
        q_t = torch.tensor(cfg["q_bard"], dtype=dtype).unsqueeze(0)
        qd_t = torch.tensor(cfg["qd"], dtype=dtype).unsqueeze(0)
        qdd_t = torch.tensor(cfg["qdd"], dtype=dtype).unsqueeze(0)
        tau_t = torch.tensor(cfg["tau"], dtype=dtype).unsqueeze(0)

        q_pin = cfg["q_pin"]
        qd_np = cfg["qd"]
        qdd_np = cfg["qdd"]
        tau_np = cfg["tau"]

        # --- FK ---
        T_bard = bard.forward_kinematics(bard_model, bard_data, bard_fid, q=q_t)[0].numpy()
        pin.framesForwardKinematics(pin_model, pin_data, q_pin)
        T_pin = pin_data.oMf[pin_fid].homogeneous
        errors["FK"].append(np.max(np.abs(T_bard - T_pin)))

        # --- Update kinematics for cached algorithms ---
        bard.update_kinematics(bard_model, bard_data, q_t, qd_t)

        # --- Jacobian ---
        J_bard = bard.jacobian(bard_model, bard_data, bard_fid, reference_frame="world")[0].numpy()
        pin.computeJointJacobians(pin_model, pin_data, q_pin)
        pin.framesForwardKinematics(pin_model, pin_data, q_pin)
        J_pin = pin.getFrameJacobian(pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD)
        errors["Jacobian"].append(np.max(np.abs(J_bard - J_pin)))

        # --- RNEA ---
        tau_bard = bard.rnea(bard_model, bard_data, qdd_t)[0].numpy()
        tau_pin = pin.rnea(pin_model, pin_data, q_pin, qd_np, qdd_np)
        errors["RNEA"].append(np.max(np.abs(tau_bard - tau_pin)))

        # --- CRBA ---
        M_bard = bard.crba(bard_model, bard_data)[0].numpy()
        M_pin = pin.crba(pin_model, pin_data, q_pin)
        M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T  # Symmetrize
        errors["CRBA"].append(np.max(np.abs(M_bard - M_pin)))

        # --- ABA ---
        qdd_bard = bard.aba(bard_model, bard_data, tau_t)[0].numpy()
        qdd_pin = pin.aba(pin_model, pin_data, q_pin, qd_np, tau_np)
        errors["ABA"].append(np.max(np.abs(qdd_bard - qdd_pin)))

    # Summarize
    results = {}
    for algo, errs in errors.items():
        errs = np.array(errs)
        results[algo] = {
            "mean": float(np.mean(errs)),
            "max": float(np.max(errs)),
            "std": float(np.std(errs)),
        }
    return results


def print_results_table(all_results, dtype):
    """Print a formatted table of results."""
    from tabulate import tabulate

    precision = "float64" if dtype == torch.float64 else "float32"
    print(f"\n{'=' * 80}")
    print(f"Accuracy Verification: bard vs Pinocchio ({precision})")
    print(f"{'=' * 80}")

    algorithms = ["FK", "Jacobian", "RNEA", "CRBA", "ABA"]
    headers = ["Algorithm"] + [
        info["label"] for name, info in ROBOT_REGISTRY.items() if name in all_results
    ]
    rows = []

    for algo in algorithms:
        row = [algo]
        for robot_name in all_results:
            res = all_results[robot_name]
            if res is None:
                row.append("N/A")
            else:
                row.append(f"{res[algo]['max']:.2e}")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Also print mean/std detail
    print(f"\nDetailed statistics (mean +/- std):")
    for robot_name, res in all_results.items():
        if res is None:
            continue
        label = ROBOT_REGISTRY[robot_name]["label"]
        print(f"\n  {label}:")
        for algo in algorithms:
            r = res[algo]
            print(f"    {algo:10s}: max={r['max']:.2e}  mean={r['mean']:.2e}  std={r['std']:.2e}")


def save_results(all_results, dtype, output_dir):
    """Save results as CSV for paper table generation."""
    precision = "float64" if dtype == torch.float64 else "float32"
    output_path = output_dir / f"accuracy_{precision}.csv"

    with open(output_path, "w") as f:
        robot_names = [name for name in all_results if all_results[name] is not None]
        f.write("algorithm," + ",".join(robot_names) + "\n")
        for algo in ["FK", "Jacobian", "RNEA", "CRBA", "ABA"]:
            row = [algo]
            for name in robot_names:
                row.append(f"{all_results[name][algo]['max']:.2e}")
            f.write(",".join(row) + "\n")

    print(f"\nResults saved to: {output_path}")


def main():
    if not HAS_PINOCCHIO:
        print("ERROR: Pinocchio is required for accuracy benchmarks.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Accuracy verification: bard vs Pinocchio")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=list(ROBOT_REGISTRY.keys()),
        choices=list(ROBOT_REGISTRY.keys()),
        help="Robot models to test",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of random samples per robot"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64", "both"],
        default="both",
        help="Precision to test",
    )
    parser.add_argument("--save", action="store_true", help="Save results as CSV")
    args = parser.parse_args()

    dtypes = []
    if args.dtype in ("float64", "both"):
        dtypes.append(torch.float64)
    if args.dtype in ("float32", "both"):
        dtypes.append(torch.float32)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    for dtype in dtypes:
        precision = "float64" if dtype == torch.float64 else "float32"
        print(f"\nRunning accuracy tests with {precision}, {args.n_samples} samples per robot...")

        all_results = {}
        for robot_name in args.robots:
            robot_info = ROBOT_REGISTRY[robot_name]
            print(f"\n  Testing {robot_info['label']}...")
            all_results[robot_name] = run_accuracy_test(
                robot_name, robot_info, args.n_samples, dtype
            )

        print_results_table(all_results, dtype)

        if args.save:
            save_results(all_results, dtype, output_dir)


if __name__ == "__main__":
    main()
