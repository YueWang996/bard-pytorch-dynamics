"""
Generate publication-quality figures from benchmark results.

Usage:
    python benchmarks/plot_results.py
    python benchmarks/plot_results.py --input results/speed_cuda.npz --robots go2 xarm7
"""

import argparse
import re
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

ROBOT_LABELS = {
    "go2": "Go2 (12-DOF)",
    "h1": "H1 (19-DOF)",
    "g1": "G1 (23-DOF)",
    "xarm7": "xArm7 (7-DOF)",
    "dog11": "Dog (11-DOF)",
}

ALGORITHMS = ["FK", "Jacobian", "RNEA", "CRBA", "ABA", "Combined"]

# Known backend safe names (as produced by save_results_npz)
# Order: Pinocchio (PyTorch) is baseline, then reference, then competitors
BACKEND_SAFE_NAMES = [
    "Pinocchio_PyTorch",
    "Pinocchio_C++",
    "ADAM",
    "bard",
    "bard_compiled",
]

BASELINE_BACKEND = "Pinocchio_PyTorch"

# Display labels
BACKEND_LABELS = {
    "Pinocchio_PyTorch": "Pinocchio (PyTorch)",
    "Pinocchio_C++": "Pinocchio (C++)",
    "ADAM": "ADAM",
    "bard": "bard",
    "bard_compiled": "bard (compiled)",
}

# Color scheme for 5 methods
COLORS = {
    "Pinocchio_PyTorch": "#FF5722",
    "Pinocchio_C++": "#FF9800",
    "ADAM": "#9C27B0",
    "bard": "#2196F3",
    "bard_compiled": "#4CAF50",
}

LINESTYLES = {
    "Pinocchio_PyTorch": "--",
    "Pinocchio_C++": ":",
    "ADAM": "-.",
    "bard": "-",
    "bard_compiled": "-",
}

MARKERS = {
    "Pinocchio_PyTorch": "s",
    "Pinocchio_C++": "D",
    "ADAM": "^",
    "bard": "o",
    "bard_compiled": "v",
}


def load_data(npz_path):
    """Load .npz and parse into structured dict.

    Key format from save_results_npz:
        {robot}_{algo}_B{batch}_{safe_method_name}

    safe_method_name can be multi-word (e.g. Pinocchio_PyTorch, bard_compiled),
    so we match known backend suffixes instead of naive splitting.
    """
    raw = np.load(npz_path)
    data = {}

    for key in raw.files:
        # Try to match each known backend suffix
        backend = None
        prefix = None
        for bname in sorted(BACKEND_SAFE_NAMES, key=len, reverse=True):
            if key.endswith(f"_{bname}"):
                backend = bname
                prefix = key[: -len(bname) - 1]
                break
        if backend is None:
            continue

        # Parse batch size: prefix ends with _B{number}
        m = re.search(r"_B(\d+)$", prefix)
        if not m:
            continue
        batch_size = int(m.group(1))
        robot_algo = prefix[: m.start()]

        # Split robot and algo
        robot = None
        algo = None
        for a in ALGORITHMS:
            if robot_algo.endswith(f"_{a}"):
                robot = robot_algo[: -len(a) - 1]
                algo = a
                break
        if robot is None:
            continue

        if robot not in data:
            data[robot] = {}
        if algo not in data[robot]:
            data[robot][algo] = {}
        if batch_size not in data[robot][algo]:
            data[robot][algo][batch_size] = {}

        data[robot][algo][batch_size][backend] = raw[key]

    return data


def plot_throughput_per_robot(data, robot_name, output_dir):
    """Create a figure with subplots for each algorithm, one robot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    label = ROBOT_LABELS.get(robot_name, robot_name)
    fig.suptitle(f"Throughput — {label}", fontsize=14, fontweight="bold")

    for idx, algo in enumerate(ALGORITHMS):
        ax = axes[idx]
        if algo not in data.get(robot_name, {}):
            ax.set_title(algo)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        algo_data = data[robot_name][algo]
        batch_sizes = sorted(algo_data.keys())

        for backend in BACKEND_SAFE_NAMES:
            medians = []
            bs_list = []
            for B in batch_sizes:
                if backend in algo_data[B]:
                    med = np.median(algo_data[B][backend])
                    if med > 0:
                        medians.append(B / med)
                        bs_list.append(B)

            if medians:
                ax.plot(
                    bs_list,
                    medians,
                    color=COLORS[backend],
                    linestyle=LINESTYLES[backend],
                    marker=MARKERS[backend],
                    markersize=4,
                    label=BACKEND_LABELS[backend],
                    linewidth=2,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(algo, fontsize=12)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"throughput_{robot_name}.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_speedup_table(data, output_dir, target_batch=4096):
    """Generate a speedup comparison table (vs Pinocchio PyTorch) across all robots."""
    # Methods to show speedup for (exclude baseline itself)
    speedup_backends = [b for b in BACKEND_SAFE_NAMES if b != BASELINE_BACKEND]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    headers = ["Algorithm"] + [ROBOT_LABELS.get(r, r) for r in data.keys()]
    cell_data = []

    for algo in ALGORITHMS:
        row = [algo]
        for robot in data.keys():
            if algo in data[robot] and target_batch in data[robot][algo]:
                bd = data[robot][algo][target_batch]
                t_base = np.median(bd.get(BASELINE_BACKEND, [1]))
                # Show bard (compiled) speedup as the primary number
                t_bard = np.median(bd.get("bard_compiled", bd.get("bard", [1])))
                speedup = t_base / t_bard if t_bard > 0 else float("inf")
                row.append(f"{speedup:.1f}x")
            else:
                row.append("N/A")
        cell_data.append(row)

    table = ax.table(cellText=cell_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4CAF50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Speedup vs {BACKEND_LABELS[BASELINE_BACKEND]} (B={target_batch})",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    output_path = output_dir / f"speedup_table_B{target_batch}.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_all_robots_single_algo(data, algo, output_dir):
    """One plot per algorithm showing speedup vs Pinocchio (PyTorch) for all robots.

    Shows bard and bard (compiled) speedup lines per robot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"{algo} — Speedup vs {BACKEND_LABELS[BASELINE_BACKEND]}",
        fontsize=13,
        fontweight="bold",
    )

    robot_colors = plt.cm.Set2(np.linspace(0, 1, len(data)))

    for idx, (robot, robot_data) in enumerate(data.items()):
        if algo not in robot_data:
            continue

        algo_data = robot_data[algo]
        batch_sizes = sorted(algo_data.keys())

        for backend, ls in [("bard", "--"), ("bard_compiled", "-")]:
            speedups = []
            bs_list = []
            for B in batch_sizes:
                t_base = np.median(algo_data[B].get(BASELINE_BACKEND, [1]))
                t_m = np.median(algo_data[B].get(backend, [0]))
                if t_m > 0:
                    speedups.append(t_base / t_m)
                    bs_list.append(B)

            if speedups:
                suffix = "" if backend == "bard_compiled" else " (eager)"
                label = f"{ROBOT_LABELS.get(robot, robot)}{suffix}"
                ax.plot(
                    bs_list,
                    speedups,
                    marker="o",
                    markersize=5,
                    label=label,
                    color=robot_colors[idx],
                    linewidth=2,
                    linestyle=ls,
                )

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Break-even")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(f"Speedup vs {BACKEND_LABELS[BASELINE_BACKEND]}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / f"speedup_{algo}.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", type=str, default=None, help="Path to .npz file")
    parser.add_argument("--robots", nargs="+", default=None, help="Filter robots")
    parser.add_argument(
        "--target-batch", type=int, default=4096, help="Batch size for speedup table"
    )
    args = parser.parse_args()

    # Find .npz file
    if args.input:
        npz_path = Path(args.input)
    else:
        npz_files = list(RESULTS_DIR.glob("speed_*.npz"))
        if not npz_files:
            print("No .npz result files found in results/. Run speed_benchmark.py --save first.")
            sys.exit(1)
        npz_path = npz_files[0]
        print(f"Using: {npz_path}")

    data = load_data(npz_path)

    if args.robots:
        data = {k: v for k, v in data.items() if k in args.robots}

    if not data:
        print("No data found for specified robots.")
        sys.exit(1)

    print(f"Robots found: {list(data.keys())}")

    output_dir = RESULTS_DIR
    output_dir.mkdir(exist_ok=True)

    # Per-robot throughput plots
    for robot in data:
        plot_throughput_per_robot(data, robot, output_dir)

    # Speedup table
    plot_speedup_table(data, output_dir, target_batch=args.target_batch)

    # Per-algorithm speedup across robots
    for algo in ALGORITHMS:
        plot_all_robots_single_algo(data, algo, output_dir)

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
