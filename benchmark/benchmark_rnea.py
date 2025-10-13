"""
RNEA benchmark: Bard vs Pinocchio vs PinocchioTorchWrapper
ASCII-only, minimal logging. Baseline = PinocchioTorchWrapper.
"""

import time
import numpy as np
import torch
import pinocchio as pin
from tabulate import tabulate

from bard.parsers.urdf import build_chain_from_urdf
from bard import RNEA
from benchconf import (
    URDF_PATH,
    BATCH_SIZES,
    NUM_REPEATS,
    WARMUP_ITERS,
    DEVICE,
    DTYPE,
    build_pin_model,
    PinocchioTorchWrapper,
)

# ------------------------------
# Setup
# ------------------------------


def load_robot():
    chain = build_chain_from_urdf(URDF_PATH, floating_base=True).to(dtype=DTYPE, device=DEVICE)
    pin_model, pin_data = build_pin_model(URDF_PATH)
    return chain, pin_model, pin_data


def generate_random_state(chain, B):
    q = torch.randn(B, chain.nq, device=DEVICE, dtype=DTYPE)
    q[:, 3:7] = q[:, 3:7] / torch.linalg.norm(q[:, 3:7], dim=1, keepdim=True)
    qd = torch.randn(B, chain.nv, device=DEVICE, dtype=DTYPE)
    qdd = torch.randn(B, chain.nv, device=DEVICE, dtype=DTYPE)
    q_pin = []
    for i in range(B):
        qi = q[i].detach().cpu().numpy()
        q_pin.append(np.concatenate([qi[:3], qi[4:7], qi[3:4], qi[7:]]))
    return q, qd, qdd, q_pin


# ------------------------------
# Bench funcs
# ------------------------------


def bench_bard(op, q, qd, qdd, nrep, nwarm):
    for _ in range(nwarm):
        _ = op.calc(q, qd, qdd)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = op.calc(q, qd, qdd)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def bench_pin(model, data, q_list, qd, qdd, nrep, nwarm):
    B = len(q_list)
    qd_np = qd.detach().cpu().numpy()
    qdd_np = qdd.detach().cpu().numpy()
    for _ in range(nwarm):
        for i in range(B):
            _ = pin.rnea(model, data, q_list[i], qd_np[i], qdd_np[i])
    ts = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        for i in range(B):
            _ = pin.rnea(model, data, q_list[i], qd_np[i], qdd_np[i])
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def bench_pin_torch(wrapper, q, qd, qdd, nrep, nwarm):
    for _ in range(nwarm):
        _ = wrapper.calc_inverse_dynamics(q, qd, qdd)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(nrep):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = wrapper.calc_inverse_dynamics(q, qd, qdd)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


# ------------------------------
# Correctness (short)
# ------------------------------


def verify_short(op, q, qd, qdd, pin_model, pin_data, q_pin):
    tau_bard = op.calc(q[:1], qd[:1], qdd[:1])[0].detach().cpu().numpy()
    qd0 = qd[:1].detach().cpu().numpy()[0]
    qdd0 = qdd[:1].detach().cpu().numpy()[0]
    tau_pin = pin.rnea(pin_model, pin_data, q_pin[0], qd0, qdd0)
    max_diff = float(np.max(np.abs(tau_bard - tau_pin)))
    tol = 5e-6 if DTYPE == torch.float64 else 5e-5
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
    print("Benchmarking RNEA.calc()")
    print("device={} dtype={}".format(DEVICE, DTYPE))
    chain, pin_model, pin_data = load_robot()
    max_batch = max(BATCH_SIZES)
    op = RNEA(chain, max_batch_size=max_batch, compile_enabled=(DEVICE == "cuda")).to(
        dtype=DTYPE, device=DEVICE
    )
    print("rnea nv={}".format(chain.nv))

    rows = []
    for B in BATCH_SIZES:
        q, qd, qdd, q_pin = generate_random_state(chain, B)
        if B == BATCH_SIZES[0]:
            verify_short(op, q, qd, qdd, pin_model, pin_data, q_pin)

        wrapper = PinocchioTorchWrapper(pin_model, device=DEVICE, dtype=DTYPE)

        print("running batch_size={}...".format(B), end="", flush=True)
        t_bard = bench_bard(op, q, qd, qdd, NUM_REPEATS, WARMUP_ITERS)
        print(" bard finished({:.2f}ms). ".format(np.mean(t_bard) * 1000.0), end="", flush=True)
        t_pin = (
            bench_pin(pin_model, pin_data, q_pin, q, qd)
            if False
            else bench_pin(pin_model, pin_data, q_pin, qd, qdd, NUM_REPEATS, WARMUP_ITERS)
        )
        print(" pinocchio finished({:.2f}ms). ".format(np.mean(t_pin) * 1000.0), end="", flush=True)
        t_pyt = bench_pin_torch(wrapper, q, qd, qdd, NUM_REPEATS, WARMUP_ITERS)
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
