"""
Benchmark: New Acceleration API (BiasAcceleration + SpatialAccelerationJacobian)

Compares:
  a) BiasAcceleration (Bard) - dJ/dt * qd only
  b) SpatialAccelerationJacobian (Bard) - J*qdd + dJ/dt*qd
  c) Pinocchio (Python API, no torch<->numpy overhead)
  d) PinocchioTorchWrapper (includes torch<->numpy overhead)

Tests both bias-only (qdd=0) and full acceleration scenarios.
"""

import time
import numpy as np
import torch
import pinocchio as pin
from tabulate import tabulate

from bard.parsers.urdf import build_chain_from_urdf
from bard.core.acceleration import (
    BiasAcceleration,
    SpatialAccelerationJacobian,
)
from benchconf import (
    URDF_PATH,
    BATCH_SIZES,
    NUM_REPEATS,
    WARMUP_ITERS,
    DEVICE,
    DTYPE,
    build_pin_model,
    PinocchioTorchWrapper,
    ref_frame_to_pin,
)

# ------------------------------
# Setup
# ------------------------------


def load_robot():
    # Bard
    chain = build_chain_from_urdf(URDF_PATH, floating_base=True).to(dtype=DTYPE, device=DEVICE)

    # Pinocchio
    pin_model, pin_data = build_pin_model(URDF_PATH)
    return chain, pin_model, pin_data


def generate_random_configuration(chain, batch_size):
    # Bard / torch
    q = torch.randn(batch_size, chain.nq, device=DEVICE, dtype=DTYPE)
    # normalize base quaternion (assumes q = [xyz, qw qx qy qz, joints...])
    q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)
    qd = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)
    qdd = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)

    # Pinocchio expects quaternion as [qx, qy, qz, qw]; convert per sample
    q_pin, qd_pin, qdd_pin = [], [], []
    for i in range(batch_size):
        q_i = q[i].detach().cpu().numpy()
        qd_i = qd[i].detach().cpu().numpy()
        qdd_i = qdd[i].detach().cpu().numpy()
        # Bard quaternion order [qw, qx, qy, qz] -> Pinocchio [qx, qy, qz, qw]
        q_pin.append(np.concatenate([q_i[:3], q_i[4:7], q_i[3:4], q_i[7:]]))
        qd_pin.append(qd_i)
        qdd_pin.append(qdd_i)
    return q, qd, qdd, q_pin, qd_pin, qdd_pin


# ------------------------------
# Benchmark functions - Bias Acceleration (qdd=0)
# ------------------------------


def bench_bias_bard(bias_accel, q, qd, frame_id, ref_frame, nrep, nwarm):
    """Benchmark BiasAcceleration class."""
    for _ in range(nwarm):
        _ = bias_accel.calc(q, qd, frame_id, reference_frame=ref_frame)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = bias_accel.calc(q, qd, frame_id, reference_frame=ref_frame)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


def bench_bias_pin(model, data, q_list, qd_list, frame_id, ref_frame, nrep, nwarm):
    """Benchmark Pinocchio with qdd=0."""
    pin_ref = ref_frame_to_pin(ref_frame)
    B = len(q_list)
    qdd_zero = [np.zeros_like(qd_list[i]) for i in range(B)]
    
    for _ in range(nwarm):
        for i in range(B):
            pin.forwardKinematics(model, data, q_list[i], qd_list[i], qdd_zero[i])
            _ = pin.getFrameAcceleration(model, data, frame_id, pin_ref).vector

    times = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        for i in range(B):
            pin.forwardKinematics(model, data, q_list[i], qd_list[i], qdd_zero[i])
            _ = pin.getFrameAcceleration(model, data, frame_id, pin_ref).vector
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


def bench_bias_pin_torch(wrapper, q, qd, frame_id, ref_frame, nrep, nwarm):
    """Benchmark PinocchioTorchWrapper.calc_bias_accel."""
    for _ in range(nwarm):
        _ = wrapper.calc_bias_accel(q, qd, frame_id, reference_frame=ref_frame)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = wrapper.calc_bias_accel(q, qd, frame_id, reference_frame=ref_frame)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


# ------------------------------
# Benchmark functions - Full Acceleration
# ------------------------------


def bench_full_bard(full_accel, q, qd, qdd, frame_id, ref_frame, nrep, nwarm):
    """Benchmark SpatialAccelerationJacobian class."""
    for _ in range(nwarm):
        _ = full_accel.calc(q, qd, qdd, frame_id, reference_frame=ref_frame)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = full_accel.calc(q, qd, qdd, frame_id, reference_frame=ref_frame)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


def bench_full_pin(model, data, q_list, qd_list, qdd_list, frame_id, ref_frame, nrep, nwarm):
    """Benchmark Pinocchio with full qdd."""
    pin_ref = ref_frame_to_pin(ref_frame)
    B = len(q_list)
    
    for _ in range(nwarm):
        for i in range(B):
            pin.forwardKinematics(model, data, q_list[i], qd_list[i], qdd_list[i])
            _ = pin.getFrameAcceleration(model, data, frame_id, pin_ref).vector

    times = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        for i in range(B):
            pin.forwardKinematics(model, data, q_list[i], qd_list[i], qdd_list[i])
            _ = pin.getFrameAcceleration(model, data, frame_id, pin_ref).vector
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


def bench_full_pin_torch(wrapper, q, qd, qdd, frame_id, ref_frame, nrep, nwarm):
    """Benchmark PinocchioTorchWrapper.calc_frame_accel with full qdd."""
    for _ in range(nwarm):
        _ = wrapper.calc_frame_accel(q, qd, qdd, frame_id, reference_frame=ref_frame)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = wrapper.calc_frame_accel(q, qd, qdd, frame_id, reference_frame=ref_frame)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


# ------------------------------
# Correctness verification
# ------------------------------


def verify_bias_acceleration(
    bias_accel,
    wrapper,
    q,
    qd,
    bard_frame_id,
    pin_frame_id,
    ref_frame,
    pin_model,
    pin_data,
    q_pin,
    qd_pin,
):
    """Verify BiasAcceleration matches both Pinocchio and PinocchioTorchWrapper."""
    # Bard BiasAcceleration
    a_bard = (
        bias_accel.calc(q[:1], qd[:1], bard_frame_id, reference_frame=ref_frame)[0]
        .detach()
        .cpu()
        .numpy()
    )
    
    # Pinocchio (native)
    qdd_zero = np.zeros_like(qd_pin[0])
    pin.forwardKinematics(pin_model, pin_data, q_pin[0], qd_pin[0], qdd_zero)
    a_pin = pin.getFrameAcceleration(
        pin_model, pin_data, pin_frame_id, ref_frame_to_pin(ref_frame)
    ).vector
    
    # PinocchioTorchWrapper
    a_wrapper = (
        wrapper.calc_bias_accel(q[:1], qd[:1], pin_frame_id, reference_frame=ref_frame)[0]
        .detach()
        .cpu()
        .numpy()
    )
    
    tol = 1e-5 if DTYPE == torch.float64 else 1e-4
    
    diff_pin = float(np.max(np.abs(a_bard - a_pin)))
    diff_wrapper = float(np.max(np.abs(a_bard - a_wrapper)))
    
    ok_pin = diff_pin <= tol
    ok_wrapper = diff_wrapper <= tol
    
    print(f"  BiasAcceleration vs Pinocchio:        max_diff={diff_pin:.2e} tol={tol:.2e} {'PASS' if ok_pin else 'FAIL'}")
    print(f"  BiasAcceleration vs PinTorchWrapper:  max_diff={diff_wrapper:.2e} tol={tol:.2e} {'PASS' if ok_wrapper else 'FAIL'}")
    
    return ok_pin and ok_wrapper


def verify_full_acceleration(
    full_accel,
    wrapper,
    q,
    qd,
    qdd,
    bard_frame_id,
    pin_frame_id,
    ref_frame,
    pin_model,
    pin_data,
    q_pin,
    qd_pin,
    qdd_pin,
):
    """Verify SpatialAccelerationJacobian matches both Pinocchio and PinocchioTorchWrapper."""
    # Bard SpatialAccelerationJacobian
    a_bard = (
        full_accel.calc(q[:1], qd[:1], qdd[:1], bard_frame_id, reference_frame=ref_frame)[0]
        .detach()
        .cpu()
        .numpy()
    )
    
    # Pinocchio (native)
    pin.forwardKinematics(pin_model, pin_data, q_pin[0], qd_pin[0], qdd_pin[0])
    a_pin = pin.getFrameAcceleration(
        pin_model, pin_data, pin_frame_id, ref_frame_to_pin(ref_frame)
    ).vector
    
    # PinocchioTorchWrapper
    a_wrapper = (
        wrapper.calc_frame_accel(q[:1], qd[:1], qdd[:1], pin_frame_id, reference_frame=ref_frame)[0]
        .detach()
        .cpu()
        .numpy()
    )
    
    tol = 1e-5 if DTYPE == torch.float64 else 1e-4
    
    diff_pin = float(np.max(np.abs(a_bard - a_pin)))
    diff_wrapper = float(np.max(np.abs(a_bard - a_wrapper)))
    
    ok_pin = diff_pin <= tol
    ok_wrapper = diff_wrapper <= tol
    
    print(f"  SpatialAccelJacobian vs Pinocchio:        max_diff={diff_pin:.2e} tol={tol:.2e} {'PASS' if ok_pin else 'FAIL'}")
    print(f"  SpatialAccelJacobian vs PinTorchWrapper:  max_diff={diff_wrapper:.2e} tol={tol:.2e} {'PASS' if ok_wrapper else 'FAIL'}")
    
    return ok_pin and ok_wrapper


# ------------------------------
# Main
# ------------------------------


def main():
    print("=" * 80)
    print("Benchmarking Acceleration API")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"URDF: {URDF_PATH.name}")
    print()

    chain, pin_model, pin_data = load_robot()
    max_batch = max(BATCH_SIZES)
    
    # Create acceleration instances
    bias_accel = BiasAcceleration(
        chain, max_batch_size=max_batch, compile_enabled=(DEVICE == "cuda")
    ).to(dtype=DTYPE, device=DEVICE)
    
    full_accel = SpatialAccelerationJacobian(
        chain, max_batch_size=max_batch, compile_enabled=(DEVICE == "cuda")
    ).to(dtype=DTYPE, device=DEVICE)

    # Get end-effector frame
    test_frame = chain.get_frame_names(exclude_fixed=True)[-1]
    bard_fid = chain.get_frame_id(test_frame)
    pin_fid = pin_model.getFrameId(test_frame)

    print(f"Test frame: {test_frame} (id={bard_fid})")
    print(f"DOF: nq={chain.nq}, nv={chain.nv}")
    print()

    for ref in ["world", "local"]:
        print("=" * 80)
        print(f"REFERENCE FRAME: {ref.upper()}")
        print("=" * 80)
        print()
        
        # ====================================================================
        # Part 1: Bias Acceleration (qdd=0)
        # ====================================================================
        print("-" * 80)
        print("PART 1: Bias Acceleration (dJ/dt * qd, with qdd=0)")
        print("-" * 80)
        print()
        
        bias_rows = []
        for B in BATCH_SIZES:
            q, qd, qdd, q_pin, qd_pin, qdd_pin = generate_random_configuration(chain, B)

            # Wrapper
            pin_wrapper = PinocchioTorchWrapper(pin_model, device=DEVICE, dtype=DTYPE)

            # Correctness check (only for smallest batch in world frame)
            if B == BATCH_SIZES[0] and ref == "world":
                print("Correctness verification:")
                verify_bias_acceleration(
                    bias_accel, pin_wrapper, q, qd, bard_fid, pin_fid, ref, 
                    pin_model, pin_data, q_pin, qd_pin
                )
                print()

            # Benchmark
            print(f"Running batch_size={B}...", end="", flush=True)
            
            t_bard = bench_bias_bard(bias_accel, q, qd, bard_fid, ref, NUM_REPEATS, WARMUP_ITERS)
            print(f" Bard={np.mean(t_bard)*1000:.2f}ms", end="", flush=True)
            
            t_pin = bench_bias_pin(
                pin_model, pin_data, q_pin, qd_pin, pin_fid, ref, NUM_REPEATS, WARMUP_ITERS
            )
            print(f" Pin={np.mean(t_pin)*1000:.2f}ms", end="", flush=True)
            
            t_pyt = bench_bias_pin_torch(
                pin_wrapper, q, qd, pin_fid, ref, NUM_REPEATS, WARMUP_ITERS
            )
            print(f" PinTorch={np.mean(t_pyt)*1000:.2f}ms")

            bard_ms = float(np.mean(t_bard) * 1000.0)
            pin_ms = float(np.mean(t_pin) * 1000.0)
            pyt_ms = float(np.mean(t_pyt) * 1000.0)

            baseline = pyt_ms
            bard_speedup = (baseline / bard_ms) if bard_ms > 0.0 else float("inf")
            pin_speedup = (baseline / pin_ms) if pin_ms > 0.0 else float("inf")

            bias_rows.append(
                [
                    B,
                    f"{baseline:.2f}",
                    f"{pin_ms:.2f}",
                    f"{bard_ms:.2f}",
                    f"{pin_speedup:.2f}x",
                    f"{bard_speedup:.2f}x",
                ]
            )

        print()
        headers = [
            "Batch",
            "PinTorch(ms)",
            "Pin(ms)",
            "Bard(ms)",
            "Pin Speedup",
            "Bard Speedup",
        ]
        print(tabulate(bias_rows, headers=headers, tablefmt="grid"))
        print()
        
        # ====================================================================
        # Part 2: Full Acceleration
        # ====================================================================
        print("-" * 80)
        print("PART 2: Full Acceleration (J*qdd + dJ/dt*qd)")
        print("-" * 80)
        print()
        
        full_rows = []
        for B in BATCH_SIZES:
            q, qd, qdd, q_pin, qd_pin, qdd_pin = generate_random_configuration(chain, B)

            # Wrapper
            pin_wrapper = PinocchioTorchWrapper(pin_model, device=DEVICE, dtype=DTYPE)

            # Correctness check (only for smallest batch in world frame)
            if B == BATCH_SIZES[0] and ref == "world":
                print("Correctness verification:")
                verify_full_acceleration(
                    full_accel, pin_wrapper, q, qd, qdd, bard_fid, pin_fid, ref, 
                    pin_model, pin_data, q_pin, qd_pin, qdd_pin
                )
                print()

            # Benchmark
            print(f"Running batch_size={B}...", end="", flush=True)
            
            t_bard = bench_full_bard(
                full_accel, q, qd, qdd, bard_fid, ref, NUM_REPEATS, WARMUP_ITERS
            )
            print(f" Bard={np.mean(t_bard)*1000:.2f}ms", end="", flush=True)
            
            t_pin = bench_full_pin(
                pin_model, pin_data, q_pin, qd_pin, qdd_pin, pin_fid, ref, 
                NUM_REPEATS, WARMUP_ITERS
            )
            print(f" Pin={np.mean(t_pin)*1000:.2f}ms", end="", flush=True)
            
            t_pyt = bench_full_pin_torch(
                pin_wrapper, q, qd, qdd, pin_fid, ref, NUM_REPEATS, WARMUP_ITERS
            )
            print(f" PinTorch={np.mean(t_pyt)*1000:.2f}ms")

            bard_ms = float(np.mean(t_bard) * 1000.0)
            pin_ms = float(np.mean(t_pin) * 1000.0)
            pyt_ms = float(np.mean(t_pyt) * 1000.0)

            baseline = pyt_ms
            bard_speedup = (baseline / bard_ms) if bard_ms > 0.0 else float("inf")
            pin_speedup = (baseline / pin_ms) if pin_ms > 0.0 else float("inf")

            full_rows.append(
                [
                    B,
                    f"{baseline:.2f}",
                    f"{pin_ms:.2f}",
                    f"{bard_ms:.2f}",
                    f"{pin_speedup:.2f}x",
                    f"{bard_speedup:.2f}x",
                ]
            )

        print()
        headers = [
            "Batch",
            "PinTorch(ms)",
            "Pin(ms)",
            "Bard(ms)",
            "Pin Speedup",
            "Bard Speedup",
        ]
        print(tabulate(full_rows, headers=headers, tablefmt="grid"))
        print()
        
        # ====================================================================
        # Part 3: Comparison - BiasAcceleration vs SpatialAccelerationJacobian
        # ====================================================================
        print("-" * 80)
        print("PART 3: Method Comparison (both with qdd=0)")
        print("-" * 80)
        print()
        
        comp_rows = []
        for B in BATCH_SIZES:
            q, qd, qdd, q_pin, qd_pin, qdd_pin = generate_random_configuration(chain, B)
            qdd_zero = torch.zeros_like(qd)

            print(f"Running batch_size={B}...", end="", flush=True)
            
            t_bias = bench_bias_bard(bias_accel, q, qd, bard_fid, ref, NUM_REPEATS, WARMUP_ITERS)
            print(f" BiasAccel={np.mean(t_bias)*1000:.2f}ms", end="", flush=True)
            
            t_full = bench_full_bard(
                full_accel, q, qd, qdd_zero, bard_fid, ref, NUM_REPEATS, WARMUP_ITERS
            )
            print(f" FullAccel(qdd=0)={np.mean(t_full)*1000:.2f}ms")

            bias_ms = float(np.mean(t_bias) * 1000.0)
            full_ms = float(np.mean(t_full) * 1000.0)
            
            speedup = (full_ms / bias_ms) if bias_ms > 0.0 else float("inf")

            comp_rows.append(
                [
                    B,
                    f"{bias_ms:.2f}",
                    f"{full_ms:.2f}",
                    f"{speedup:.2f}x",
                ]
            )

        print()
        headers = [
            "Batch",
            "BiasAccel(ms)",
            "FullAccel(ms)",
            "BiasAccel Speedup",
        ]
        print(tabulate(comp_rows, headers=headers, tablefmt="grid"))
        print()
        print("Note: BiasAcceleration is optimized for qdd=0, while SpatialAccelerationJacobian")
        print("      is more general. BiasAcceleration should be faster for bias-only computation.")
        print()


if __name__ == "__main__":
    main()
