"""
Speed benchmark: bard vs Pinocchio vs ADAM.

Compares five methods across FK, Jacobian, RNEA, CRBA, ABA, and a Combined
workflow for multiple robot models at various batch sizes.

Methods:
  1. Pinocchio (C++)       — Raw C++ calls via Python bindings, numpy I/O
  2. Pinocchio (PyTorch)   — Same C++ calls, but input/output as PyTorch tensors
  3. ADAM                   — adam-robotics PyTorch backend
  4. bard                  — No torch.compile
  5. bard (compiled)       — With torch.compile

Usage:
    python benchmarks/speed_benchmark.py --device cpu
    python benchmarks/speed_benchmark.py --device cuda --robots go2 xarm7
    python benchmarks/speed_benchmark.py --device cuda --batch-sizes 1 64 256 1024 4096
"""

import argparse
import sys
import os
import time
import xml.etree.ElementTree as ET
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

try:
    from adam.pytorch import KinDynComputations as AdamKinDyn

    HAS_ADAM = True
except ImportError:
    HAS_ADAM = False

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

# Method names (display order)
METHOD_PIN_CPP = "Pinocchio (C++)"
METHOD_PIN_TORCH = "Pinocchio (PyTorch)"
METHOD_ADAM = "ADAM"
METHOD_BARD = "bard"
METHOD_BARD_COMPILED = "bard (compiled)"

ALL_METHODS = [METHOD_PIN_CPP, METHOD_PIN_TORCH, METHOD_ADAM, METHOD_BARD, METHOD_BARD_COMPILED]
ALGORITHMS = ["FK", "Jacobian", "RNEA", "CRBA", "ABA", "Combined"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_urdf_joint_names(urdf_path):
    """Extract actuated joint names from a URDF file."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    return [
        j.get("name")
        for j in root.findall("joint")
        if j.get("type") in ("revolute", "continuous", "prismatic")
    ]


def generate_random_q(model, B, device, dtype, floating_base):
    """Generate random joint configurations for bard."""
    q = torch.randn(B, model.nq, device=device, dtype=dtype)
    if floating_base:
        q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)
    return q


def bard_q_to_pin_list(q_tensor, floating_base):
    """Convert bard q batch to list of Pinocchio q arrays (numpy)."""
    q_np = q_tensor.detach().cpu().numpy()
    out = []
    for i in range(q_np.shape[0]):
        if floating_base:
            qi = q_np[i]
            out.append(np.concatenate([qi[:3], qi[4:7], qi[3:4], qi[7:]]))
        else:
            out.append(q_np[i].copy())
    return out


def bard_q_to_adam_inputs(q_tensor, qd_tensor, floating_base, dtype):
    """Convert bard (q, qd) to ADAM inputs (w_H_b, s, base_vel, joint_vel)."""
    B = q_tensor.shape[0]
    if floating_base:
        # Extract base position and quaternion [qw, qx, qy, qz]
        pos = q_tensor[:, :3]  # (B, 3)
        quat = q_tensor[:, 3:7]  # (B, 4) — [qw, qx, qy, qz]
        s = q_tensor[:, 7:]  # (B, n_joints)

        # Build 4x4 homogeneous transform from pos + quat
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        w_H_b = torch.zeros(B, 4, 4, dtype=dtype, device=q_tensor.device)
        w_H_b[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        w_H_b[:, 0, 1] = 2 * (qx * qy - qz * qw)
        w_H_b[:, 0, 2] = 2 * (qx * qz + qy * qw)
        w_H_b[:, 1, 0] = 2 * (qx * qy + qz * qw)
        w_H_b[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        w_H_b[:, 1, 2] = 2 * (qy * qz - qx * qw)
        w_H_b[:, 2, 0] = 2 * (qx * qz - qy * qw)
        w_H_b[:, 2, 1] = 2 * (qy * qz + qx * qw)
        w_H_b[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        w_H_b[:, 0, 3] = pos[:, 0]
        w_H_b[:, 1, 3] = pos[:, 1]
        w_H_b[:, 2, 3] = pos[:, 2]
        w_H_b[:, 3, 3] = 1.0

        base_vel = qd_tensor[:, :6]  # (B, 6)
        joint_vel = qd_tensor[:, 6:]  # (B, n_joints)
    else:
        w_H_b = (
            torch.eye(4, dtype=dtype, device=q_tensor.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .contiguous()
        )
        s = q_tensor
        base_vel = torch.zeros(B, 6, dtype=dtype, device=q_tensor.device)
        joint_vel = qd_tensor

    return w_H_b, s, base_vel, joint_vel


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


# ---------------------------------------------------------------------------
# Per-method benchmark closures
# ---------------------------------------------------------------------------


def make_pin_cpp_fns(pin_model, pin_data, pin_fid, q_pin_list, qd_np, qdd_np, tau_np, B):
    """Return benchmark closures for Pinocchio C++ (raw numpy)."""

    def fk():
        for i in range(B):
            pin.framesForwardKinematics(pin_model, pin_data, q_pin_list[i])

    def jacobian():
        for i in range(B):
            pin.computeJointJacobians(pin_model, pin_data, q_pin_list[i])
            pin.framesForwardKinematics(pin_model, pin_data, q_pin_list[i])
            pin.getFrameJacobian(pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD)

    def rnea():
        for i in range(B):
            pin.rnea(pin_model, pin_data, q_pin_list[i], qd_np[i], qdd_np[i])

    def crba():
        for i in range(B):
            pin.crba(pin_model, pin_data, q_pin_list[i])

    def aba():
        for i in range(B):
            pin.aba(pin_model, pin_data, q_pin_list[i], qd_np[i], tau_np[i])

    def combined():
        for i in range(B):
            pin.framesForwardKinematics(pin_model, pin_data, q_pin_list[i])
            pin.computeJointJacobians(pin_model, pin_data, q_pin_list[i])
            pin.getFrameJacobian(pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD)
            pin.rnea(pin_model, pin_data, q_pin_list[i], qd_np[i], qdd_np[i])

    return {
        "FK": fk,
        "Jacobian": jacobian,
        "RNEA": rnea,
        "CRBA": crba,
        "ABA": aba,
        "Combined": combined,
    }


def make_pin_torch_fns(
    pin_model, pin_data, pin_fid, q_torch, qd_torch, qdd_torch, tau_torch, floating_base, B, nv
):
    """Return benchmark closures for Pinocchio with PyTorch tensor I/O.

    Simulates the cost of using Pinocchio inside a PyTorch pipeline:
    convert tensors -> call C++ in a loop -> stack results back to tensors.
    """

    def _q_to_pin(q_batch):
        """Convert a PyTorch batch to per-sample Pinocchio numpy arrays."""
        q_np = q_batch.detach().cpu().numpy()
        out = []
        for i in range(q_np.shape[0]):
            if floating_base:
                qi = q_np[i]
                out.append(np.concatenate([qi[:3], qi[4:7], qi[3:4], qi[7:]]))
            else:
                out.append(q_np[i].copy())
        return out

    def fk():
        q_list = _q_to_pin(q_torch)
        results = []
        for i in range(B):
            pin.framesForwardKinematics(pin_model, pin_data, q_list[i])
            results.append(pin_data.oMf[pin_fid].homogeneous.copy())
        torch.from_numpy(np.stack(results))

    def jacobian():
        q_list = _q_to_pin(q_torch)
        results = []
        for i in range(B):
            pin.computeJointJacobians(pin_model, pin_data, q_list[i])
            pin.framesForwardKinematics(pin_model, pin_data, q_list[i])
            J = pin.getFrameJacobian(pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD)
            results.append(J.copy())
        torch.from_numpy(np.stack(results))

    def rnea():
        q_list = _q_to_pin(q_torch)
        qd_np = qd_torch.detach().cpu().numpy()
        qdd_np = qdd_torch.detach().cpu().numpy()
        results = []
        for i in range(B):
            tau = pin.rnea(pin_model, pin_data, q_list[i], qd_np[i], qdd_np[i])
            results.append(tau.copy())
        torch.from_numpy(np.stack(results))

    def crba():
        q_list = _q_to_pin(q_torch)
        results = []
        for i in range(B):
            M = pin.crba(pin_model, pin_data, q_list[i])
            results.append(M.copy())
        torch.from_numpy(np.stack(results))

    def aba():
        q_list = _q_to_pin(q_torch)
        qd_np = qd_torch.detach().cpu().numpy()
        tau_np = tau_torch.detach().cpu().numpy()
        results = []
        for i in range(B):
            qdd = pin.aba(pin_model, pin_data, q_list[i], qd_np[i], tau_np[i])
            results.append(qdd.copy())
        torch.from_numpy(np.stack(results))

    def combined():
        q_list = _q_to_pin(q_torch)
        qd_np = qd_torch.detach().cpu().numpy()
        qdd_np = qdd_torch.detach().cpu().numpy()
        fk_results, jac_results, rnea_results = [], [], []
        for i in range(B):
            pin.framesForwardKinematics(pin_model, pin_data, q_list[i])
            fk_results.append(pin_data.oMf[pin_fid].homogeneous.copy())
            pin.computeJointJacobians(pin_model, pin_data, q_list[i])
            J = pin.getFrameJacobian(pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD)
            jac_results.append(J.copy())
            tau = pin.rnea(pin_model, pin_data, q_list[i], qd_np[i], qdd_np[i])
            rnea_results.append(tau.copy())
        torch.from_numpy(np.stack(fk_results))
        torch.from_numpy(np.stack(jac_results))
        torch.from_numpy(np.stack(rnea_results))

    return {
        "FK": fk,
        "Jacobian": jacobian,
        "RNEA": rnea,
        "CRBA": crba,
        "ABA": aba,
        "Combined": combined,
    }


def make_adam_fns(adam_model, w_H_b, s, base_vel, joint_vel, tau_joints, test_frame):
    """Return benchmark closures for ADAM PyTorch backend."""

    def fk():
        adam_model.forward_kinematics(test_frame, w_H_b, s)

    def jacobian():
        adam_model.jacobian(test_frame, w_H_b, s)

    def rnea():
        adam_model.bias_force(w_H_b, s, base_vel, joint_vel)

    def crba():
        adam_model.mass_matrix(w_H_b, s)

    def aba():
        adam_model.aba(w_H_b, s, base_vel, joint_vel, tau_joints)

    def combined():
        adam_model.forward_kinematics(test_frame, w_H_b, s)
        adam_model.jacobian(test_frame, w_H_b, s)
        adam_model.bias_force(w_H_b, s, base_vel, joint_vel)

    return {
        "FK": fk,
        "Jacobian": jacobian,
        "RNEA": rnea,
        "CRBA": crba,
        "ABA": aba,
        "Combined": combined,
    }


def make_bard_fns(bard_model, bard_data, bard_fid, q, qd, qdd, tau):
    """Return benchmark closures for bard (kinematics cached)."""

    def fk():
        bard.forward_kinematics(bard_model, bard_data, bard_fid, q=q)

    def jacobian():
        bard.jacobian(bard_model, bard_data, bard_fid)

    def rnea():
        bard.rnea(bard_model, bard_data, qdd)

    def crba():
        bard.crba(bard_model, bard_data)

    def aba():
        bard.aba(bard_model, bard_data, tau)

    def combined():
        bard.update_kinematics(bard_model, bard_data, q, qd)
        bard.forward_kinematics(bard_model, bard_data, bard_fid)
        bard.jacobian(bard_model, bard_data, bard_fid)
        bard.rnea(bard_model, bard_data, qdd)

    return {
        "FK": fk,
        "Jacobian": jacobian,
        "RNEA": rnea,
        "CRBA": crba,
        "ABA": aba,
        "Combined": combined,
    }


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def bench_robot(robot_name, robot_info, device, dtype, batch_sizes, n_repeats):
    """Run speed benchmarks for a single robot, all methods, all algorithms."""
    urdf_path = robot_info["urdf"]
    floating_base = robot_info["floating_base"]

    if not urdf_path.exists():
        print(f"  [SKIP] URDF not found: {urdf_path}")
        return None

    # --- Load models ---
    # bard (no compile)
    bard_model = bard.build_model_from_urdf(str(urdf_path), floating_base=floating_base).to(
        dtype=dtype, device=device
    )
    # bard (compiled)
    bard_model_compiled = bard.build_model_from_urdf(
        str(urdf_path), floating_base=floating_base
    ).to(dtype=dtype, device=device)
    bard_model_compiled.enable_compilation(True)
    print(f"  bard models loaded (nq={bard_model.nq}, nv={bard_model.nv})")

    # Pinocchio
    if HAS_PINOCCHIO:
        if floating_base:
            pin_model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        else:
            pin_model = pin.buildModelFromUrdf(str(urdf_path))
        pin_data = pin_model.createData()
        print(f"  Pinocchio model loaded")
    else:
        pin_model = pin_data = None
        print(f"  Pinocchio not available, skipping")

    # ADAM
    adam_model = None
    if HAS_ADAM:
        try:
            joint_names = get_urdf_joint_names(urdf_path)
            adam_model = AdamKinDyn(str(urdf_path), joint_names)
            print(f"  ADAM model loaded ({len(joint_names)} joints)")
        except Exception as e:
            print(f"  ADAM failed to load: {e}")
    else:
        print(f"  ADAM not available, skipping")

    nv = bard_model.nv
    frame_names = bard_model.get_frame_names(exclude_fixed=True)
    test_frame = frame_names[-1] if frame_names else bard_model.get_frame_names()[0]
    bard_fid = bard_model.get_frame_id(test_frame)
    pin_fid = pin_model.getFrameId(test_frame) if pin_model else None

    max_batch = max(batch_sizes)
    bard_data = bard.create_data(bard_model, max_batch_size=max_batch)
    bard_data_compiled = bard.create_data(bard_model_compiled, max_batch_size=max_batch)

    # results[algo][B][method] = timing_array
    results = {algo: {} for algo in ALGORITHMS}

    for B in batch_sizes:
        print(f"    B={B}...", end="", flush=True)

        # Generate shared inputs
        q = generate_random_q(bard_model, B, device, dtype, floating_base)
        qd = torch.randn(B, nv, device=device, dtype=dtype)
        qdd = torch.randn(B, nv, device=device, dtype=dtype)
        tau = torch.randn(B, nv, device=device, dtype=dtype)

        # Pre-compute kinematics for cached bard algorithms
        bard.update_kinematics(bard_model, bard_data, q, qd)
        bard.update_kinematics(bard_model_compiled, bard_data_compiled, q, qd)

        # Prepare per-method inputs
        q_pin_list = bard_q_to_pin_list(q, floating_base) if pin_model else None
        qd_np = qd.detach().cpu().numpy()
        qdd_np = qdd.detach().cpu().numpy()
        tau_np = tau.detach().cpu().numpy()

        # ADAM inputs
        if adam_model is not None:
            w_H_b, s_adam, base_vel, joint_vel = bard_q_to_adam_inputs(q, qd, floating_base, dtype)
            n_joints = s_adam.shape[1]
            if floating_base:
                tau_joints = tau[:, 6:]  # ADAM ABA takes joint torques only
            else:
                tau_joints = tau
        else:
            w_H_b = s_adam = base_vel = joint_vel = tau_joints = None

        # Build closures for each method
        method_fns = {}

        if pin_model is not None:
            method_fns[METHOD_PIN_CPP] = make_pin_cpp_fns(
                pin_model, pin_data, pin_fid, q_pin_list, qd_np, qdd_np, tau_np, B
            )
            method_fns[METHOD_PIN_TORCH] = make_pin_torch_fns(
                pin_model, pin_data, pin_fid, q, qd, qdd, tau, floating_base, B, nv
            )

        if adam_model is not None:
            method_fns[METHOD_ADAM] = make_adam_fns(
                adam_model, w_H_b, s_adam, base_vel, joint_vel, tau_joints, test_frame
            )

        method_fns[METHOD_BARD] = make_bard_fns(bard_model, bard_data, bard_fid, q, qd, qdd, tau)
        method_fns[METHOD_BARD_COMPILED] = make_bard_fns(
            bard_model_compiled, bard_data_compiled, bard_fid, q, qd, qdd, tau
        )

        # Run benchmarks
        for algo in ALGORITHMS:
            if B not in results[algo]:
                results[algo][B] = {}
            for method_name, fns in method_fns.items():
                fn = fns[algo]
                # Warmup
                for _ in range(WARMUP_ITERS):
                    fn()
                # Time
                method_device = (
                    "cpu" if method_name in (METHOD_PIN_CPP, METHOD_PIN_TORCH) else device
                )
                t = time_fn(fn, n_repeats, method_device)
                results[algo][B][method_name] = t

        print(" done")

    return results


def print_summary_table(robot_name, robot_info, results, batch_sizes, device):
    """Print throughput and speedup table."""
    from tabulate import tabulate

    print(f"\n{'=' * 120}")
    print(f"Speed Benchmark: {robot_info['label']} | device={device}")
    print(f"{'=' * 120}")

    for algo in ALGORITHMS:
        print(f"\n  {algo}:")
        # Determine which methods are present
        sample_B = batch_sizes[0]
        if sample_B not in results[algo]:
            continue
        methods = [m for m in ALL_METHODS if m in results[algo][sample_B]]

        headers = ["batch"] + [f"{m} (ms)" for m in methods]
        rows = []
        for B in batch_sizes:
            if B not in results[algo]:
                continue
            row = [B]
            for m in methods:
                if m in results[algo][B]:
                    t_ms = float(np.median(results[algo][B][m])) * 1000
                    row.append(f"{t_ms:.3f}")
                else:
                    row.append("N/A")
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Print speedup relative to Pinocchio (C++)
        if METHOD_PIN_CPP in methods:
            print(f"\n  {algo} — Speedup vs {METHOD_PIN_CPP}:")
            speedup_methods = [m for m in methods if m != METHOD_PIN_CPP]
            headers = ["batch"] + speedup_methods
            rows = []
            for B in batch_sizes:
                if B not in results[algo]:
                    continue
                t_pin = float(np.median(results[algo][B][METHOD_PIN_CPP]))
                row = [B]
                for m in speedup_methods:
                    if m in results[algo][B]:
                        t_m = float(np.median(results[algo][B][m]))
                        speedup = t_pin / t_m if t_m > 0 else float("inf")
                        row.append(f"{speedup:.2f}x")
                    else:
                        row.append("N/A")
                rows.append(row)
            print(tabulate(rows, headers=headers, tablefmt="grid"))


def save_results_npz(all_results, device, output_dir):
    """Save raw timing data as .npz for reproducibility."""
    data = {}
    for robot_name, results in all_results.items():
        if results is None:
            continue
        for algo in results:
            for B in results[algo]:
                for method_name, timing in results[algo][B].items():
                    # Sanitize method name for npz key
                    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
                    key = f"{robot_name}_{algo}_B{B}_{safe_name}"
                    data[key] = timing

    if "cuda" in str(device) and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        output_path = output_dir / f"speed_{gpu_name}.npz"
    else:
        output_path = output_dir / f"speed_{device}.npz"
    np.savez(output_path, **data)
    print(f"\nRaw timing data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Speed benchmark: bard vs Pinocchio vs ADAM")
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

    print(f"Device: {device} | Dtype: {args.dtype} | Repeats: {args.n_repeats}")
    print(f"Batch sizes: {args.batch_sizes}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pinocchio: {'available' if HAS_PINOCCHIO else 'NOT available'}")
    print(f"ADAM: {'available' if HAS_ADAM else 'NOT available'}")

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
