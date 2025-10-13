"""
Simplified benchmark: Bard vs Pinocchio vs PinocchioTorchWrapper

- ASCII-only output
- Minimal logging
- Measures three paths:
  a) Bard (PyTorch-native solver)
  b) Pinocchio (Python API, no torch<->numpy overhead; pure numpy lists)
  c) PinocchioTorchWrapper (includes torch<->numpy overhead inside the wrapper)
"""

import time
import numpy as np
import torch
import pinocchio as pin
from tabulate import tabulate

from bard.parsers.urdf import build_chain_from_urdf
from bard import SpatialAcceleration
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
    # normalize base quaternion (assumes q = [xyz, qw qx qy qz, joints...] or similar Bard layout)
    q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)
    qd = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)
    qdd = torch.randn(batch_size, chain.nv, device=DEVICE, dtype=DTYPE)

    # Pinocchio expects quaternion as [qx, qy, qz, qw]; convert per sample
    q_pin, qd_pin, qdd_pin = [], [], []
    for i in range(batch_size):
        q_i = q[i].detach().cpu().numpy()
        qd_i = qd[i].detach().cpu().numpy()
        qdd_i = qdd[i].detach().cpu().numpy()
        # Bard quaternion order assumed [qw, qx, qy, qz] in q[3:7]; rotate to [qx, qy, qz, qw]
        q_pin.append(np.concatenate([q_i[:3], q_i[4:7], q_i[3:4], q_i[7:]]))
        qd_pin.append(qd_i)
        qdd_pin.append(qdd_i)
    return q, qd, qdd, q_pin, qd_pin, qdd_pin


# ------------------------------
# Bench functions
# ------------------------------


def bench_bard(accel, q, qd, qdd, frame_id, ref_frame, nrep, nwarm):
    for _ in range(nwarm):
        _ = accel.calc(q, qd, qdd, frame_id, reference_frame=ref_frame)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = accel.calc(q, qd, qdd, frame_id, reference_frame=ref_frame)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


def bench_pin(model, data, q_list, qd_list, qdd_list, frame_id, ref_frame, nrep, nwarm):
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


def bench_pin_torch(wrapper, q, qd, qdd, frame_id, ref_frame, nrep, nwarm):
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
# Correctness (short, ASCII-only)
# ------------------------------


def verify_short(
    accel,
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
    a_bard = (
        accel.calc(q[:1], qd[:1], qdd[:1], bard_frame_id, reference_frame=ref_frame)[0]
        .detach()
        .cpu()
        .numpy()
    )
    pin.forwardKinematics(pin_model, pin_data, q_pin[0], qd_pin[0], qdd_pin[0])
    a_pin = pin.getFrameAcceleration(
        pin_model, pin_data, pin_frame_id, ref_frame_to_pin(ref_frame)
    ).vector
    max_diff = float(np.max(np.abs(a_bard - a_pin)))
    tol = 1e-5 if DTYPE == torch.float64 else 1e-4
    ok = max_diff <= tol
    print(
        "check max_abs_diff={:.2e} tol={:.2e} status={}".format(
            max_diff, tol, "ok" if ok else "mismatch"
        )
    )
    return ok


# ------------------------------
# Main
# ------------------------------


def main():
    print("Benchmarking SpatialAcceleration.calc()")
    print("device={} dtype={}".format(DEVICE, DTYPE))

    chain, pin_model, pin_data = load_robot()
    max_batch = max(BATCH_SIZES)
    accel = SpatialAcceleration(
        chain, max_batch_size=max_batch, compile_enabled=(DEVICE == "cuda")
    ).to(dtype=DTYPE, device=DEVICE)

    # end-effector frame
    test_frame = chain.get_frame_names(exclude_fixed=True)[-1]
    bard_fid = chain.get_frame_id(test_frame)
    pin_fid = pin_model.getFrameId(test_frame)

    # tiny one-liner about the model
    print("frame={} id={} nq={} nv={}".format(test_frame, bard_fid, chain.nq, chain.nv))

    for ref in ["world", "local"]:
        rows = []
        for B in BATCH_SIZES:
            q, qd, qdd, q_pin, qd_pin, qdd_pin = generate_random_configuration(chain, B)

            # short correctness only once for smallest batch in world frame
            if B == BATCH_SIZES[0] and ref == "world":
                verify_short(
                    accel,
                    q,
                    qd,
                    qdd,
                    bard_fid,
                    pin_fid,
                    ref,
                    pin_model,
                    pin_data,
                    q_pin,
                    qd_pin,
                    qdd_pin,
                )

            # wrappers
            pin_wrapper = PinocchioTorchWrapper(pin_model, device=DEVICE, dtype=DTYPE)

            # run
            print("running batch_size={}...".format(B), end="", flush=True)
            t_bard = bench_bard(accel, q, qd, qdd, bard_fid, ref, NUM_REPEATS, WARMUP_ITERS)
            print(" bard finished({:.2f}ms). ".format(np.mean(t_bard) * 1000.0), end="", flush=True)
            t_pin = bench_pin(
                pin_model, pin_data, q_pin, qd_pin, qdd_pin, pin_fid, ref, NUM_REPEATS, WARMUP_ITERS
            )
            print(
                " pinocchio finished({:.2f}ms). ".format(np.mean(t_pin) * 1000.0),
                end="",
                flush=True,
            )
            t_pyt = bench_pin_torch(
                pin_wrapper, q, qd, qdd, pin_fid, ref, NUM_REPEATS, WARMUP_ITERS
            )
            print(" pinocchio-torch finished({:.2f}ms).".format(np.mean(t_pyt) * 1000.0))

            bard_ms = float(np.mean(t_bard) * 1000.0)
            pin_ms = float(np.mean(t_pin) * 1000.0)
            pyt_ms = float(np.mean(t_pyt) * 1000.0)

            baseline = pyt_ms  # PinocchioTorchWrapper
            pin_speed = (baseline / pin_ms) if pin_ms > 0.0 else float("inf")
            bard_speed = (baseline / bard_ms) if bard_ms > 0.0 else float("inf")

            rows.append(
                [
                    B,
                    f"{baseline:.2f}",
                    f"{pin_ms:.2f}",
                    f"{bard_ms:.2f}",
                    f"{pin_speed:.2f}x",
                    f"{bard_speed:.2f}x",
                ]
            )

        print(f"Reference frame: {ref}")
        headers = [
            "batch",
            "baseline(ms)",
            "pin_time(ms)",
            "bard_time(ms)",
            "pin_speedup",
            "bard_speedup",
        ]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print("")


if __name__ == "__main__":
    main()
