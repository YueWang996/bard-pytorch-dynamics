"""
Jacobian benchmark: Bard vs Pinocchio vs PinocchioTorchWrapper
ASCII-only, minimal logging. Baseline = PinocchioTorchWrapper.
"""

import time
import numpy as np
import torch
import pinocchio as pin
from tabulate import tabulate

from bard.parsers.urdf import build_chain_from_urdf
from bard import Jacobian
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
    with open(URDF_PATH, "rb") as f:
        urdf_string = f.read()
    chain = build_chain_from_urdf(urdf_string, floating_base=True).to(dtype=DTYPE, device=DEVICE)
    pin_model, pin_data = build_pin_model(URDF_PATH)
    return chain, pin_model, pin_data


def generate_random_q(chain, B):
    q = torch.randn(B, chain.nq, device=DEVICE, dtype=DTYPE)
    q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)
    q_pin = []
    for i in range(B):
        qi = q[i].detach().cpu().numpy()
        q_pin.append(np.concatenate([qi[:3], qi[4:7], qi[3:4], qi[7:]]))
    return q, q_pin


# ------------------------------
# Bench funcs
# ------------------------------


def bench_bard(op, q, frame_id, ref, nrep, nwarm):
    for _ in range(nwarm):
        _ = op.calc(q, frame_id, reference_frame=ref)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = op.calc(q, frame_id, reference_frame=ref)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def bench_pin(model, data, q_list, frame_id, ref, nrep, nwarm):
    B = len(q_list)
    pin_ref = ref_frame_to_pin(ref)
    for _ in range(nwarm):
        for i in range(B):
            pin.framesForwardKinematics(model, data, q_list[i])
            _ = pin.computeFrameJacobian(model, data, q_list[i], frame_id, pin_ref)
    ts = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        for i in range(B):
            pin.framesForwardKinematics(model, data, q_list[i])
            _ = pin.computeFrameJacobian(model, data, q_list[i], frame_id, pin_ref)
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def bench_pin_torch(wrapper, q, frame_id, ref, nrep, nwarm):
    for _ in range(nwarm):
        _ = wrapper.calc_frame_jacobian(q, frame_id, ref)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = wrapper.calc_frame_jacobian(q, frame_id, ref)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


# ------------------------------
# Correctness (short)
# ------------------------------


def verify_short(op, q, bard_fid, pin_fid, ref, pin_model, pin_data, q_pin):
    J_bard = op.calc(q[:1], bard_fid, reference_frame=ref)[0].detach().cpu().numpy()
    pin.framesForwardKinematics(pin_model, pin_data, q_pin[0])
    J_pin = pin.computeFrameJacobian(pin_model, pin_data, q_pin[0], pin_fid, ref_frame_to_pin(ref))
    max_diff = float(np.max(np.abs(J_bard - J_pin)))
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
    print("Benchmarking Jacobian.calc()")
    print("device={} dtype={}".format(DEVICE, DTYPE))
    chain, pin_model, pin_data = load_robot()
    max_batch = max(BATCH_SIZES)
    op = Jacobian(chain, max_batch_size=max_batch, compile_enabled=(DEVICE == "cuda")).to(
        dtype=DTYPE, device=DEVICE
    )
    test_frame = chain.get_frame_names(exclude_fixed=True)[-1]
    bard_fid = chain.get_frame_indices(test_frame).item()
    pin_fid = pin_model.getFrameId(test_frame)
    print("frame={} id={} nv={}".format(test_frame, bard_fid, chain.nv))

    for ref in ["world", "local"]:
        rows = []
        for B in BATCH_SIZES:
            q, q_pin = generate_random_q(chain, B)
            if B == BATCH_SIZES[0] and ref == "world":
                verify_short(op, q, bard_fid, pin_fid, ref, pin_model, pin_data, q_pin)

            wrapper = PinocchioTorchWrapper(pin_model, device=DEVICE, dtype=DTYPE)

            print("running batch_size={}...".format(B), end="", flush=True)
            t_bard = bench_bard(op, q, bard_fid, ref, NUM_REPEATS, WARMUP_ITERS)
            print(" bard finished({:.2f}ms). ".format(np.mean(t_bard) * 1000.0), end="", flush=True)
            t_pin = bench_pin(pin_model, pin_data, q_pin, pin_fid, ref, NUM_REPEATS, WARMUP_ITERS)
            print(
                " pinocchio finished({:.2f}ms). ".format(np.mean(t_pin) * 1000.0),
                end="",
                flush=True,
            )
            t_pyt = bench_pin_torch(wrapper, q, pin_fid, ref, NUM_REPEATS, WARMUP_ITERS)
            print(" pinocchio-torch finished({:.2f}ms).".format(np.mean(t_pyt) * 1000.0))

            bard_ms = float(np.mean(t_bard) * 1000.0)
            pin_ms = float(np.mean(t_pin) * 1000.0)
            pyt_ms = float(np.mean(t_pyt) * 1000.0)

            baseline = pyt_ms
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

        print("Reference frame: {}".format(ref))
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
