"""
Generate publication-quality figures from benchmark results.

Usage:
    python benchmarks/plot_results.py
    python benchmarks/plot_results.py --input results/speed_cuda.npz --robots go2 xarm7
"""

import argparse
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

# Color scheme
COLORS = {
    "bard": "#2196F3",
    "pinocchio": "#FF5722",
}
LINESTYLES = {
    "bard": "-",
    "pinocchio": "--",
}


def load_data(npz_path):
    """Load .npz and parse into structured dict."""
    raw = np.load(npz_path)
    data = {}

    for key in raw.files:
        # key format: robotname_algo_B{batchsize}_{backend}
        parts = key.rsplit("_", 1)
        backend = parts[1]  # "bard" or "pinocchio"
        prefix = parts[0]

        # Find batch size
        b_idx = prefix.rfind("_B")
        batch_size = int(prefix[b_idx + 2 :])
        robot_algo = prefix[:b_idx]

        # Split robot and algo
        for algo in ALGORITHMS:
            if robot_algo.endswith(f"_{algo}"):
                robot = robot_algo[: -len(algo) - 1]
                break
        else:
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
    n_algos = len(ALGORITHMS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    label = ROBOT_LABELS.get(robot_name, robot_name)
    fig.suptitle(f"Throughput: bard vs Pinocchio — {label}", fontsize=14, fontweight="bold")

    for idx, algo in enumerate(ALGORITHMS):
        ax = axes[idx]
        if algo not in data.get(robot_name, {}):
            ax.set_title(algo)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        algo_data = data[robot_name][algo]
        batch_sizes = sorted(algo_data.keys())

        for backend in ["bard", "pinocchio"]:
            throughputs = []
            bs_list = []
            for B in batch_sizes:
                if backend in algo_data[B]:
                    mean_time = np.mean(algo_data[B][backend])
                    if mean_time > 0:
                        throughputs.append(B / mean_time)
                        bs_list.append(B)

            if throughputs:
                ax.plot(
                    bs_list,
                    throughputs,
                    color=COLORS[backend],
                    linestyle=LINESTYLES[backend],
                    marker="o",
                    markersize=4,
                    label=backend,
                    linewidth=2,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(algo, fontsize=12)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"throughput_{robot_name}.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_speedup_table(data, output_dir, target_batch=4096):
    """Generate a speedup comparison across all robots at a given batch size."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    headers = ["Algorithm"] + [ROBOT_LABELS.get(r, r) for r in data.keys()]
    cell_data = []

    for algo in ALGORITHMS:
        row = [algo]
        for robot in data.keys():
            if algo in data[robot] and target_batch in data[robot][algo]:
                t_bard = np.mean(data[robot][algo][target_batch].get("bard", [1]))
                t_pin = np.mean(data[robot][algo][target_batch].get("pinocchio", [1]))
                speedup = t_pin / t_bard if t_bard > 0 else float("inf")
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

    output_path = output_dir / f"speedup_table_B{target_batch}.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_all_robots_single_algo(data, algo, output_dir):
    """One plot per algorithm showing all robots."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"{algo} — Speedup vs Batch Size", fontsize=13, fontweight="bold")

    robot_colors = plt.cm.Set2(np.linspace(0, 1, len(data)))

    for idx, (robot, robot_data) in enumerate(data.items()):
        if algo not in robot_data:
            continue

        algo_data = robot_data[algo]
        batch_sizes = sorted(algo_data.keys())
        speedups = []
        bs_list = []

        for B in batch_sizes:
            t_bard = np.mean(algo_data[B].get("bard", [1]))
            t_pin = np.mean(algo_data[B].get("pinocchio", [1]))
            if t_bard > 0:
                speedups.append(t_pin / t_bard)
                bs_list.append(B)

        label = ROBOT_LABELS.get(robot, robot)
        ax.plot(bs_list, speedups, marker="o", markersize=5, label=label, color=robot_colors[idx], linewidth=2)

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Break-even")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (bard / Pinocchio)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / f"speedup_{algo}.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", type=str, default=None, help="Path to .npz file")
    parser.add_argument("--robots", nargs="+", default=None, help="Filter robots")
    parser.add_argument("--target-batch", type=int, default=4096, help="Batch size for speedup table")
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
