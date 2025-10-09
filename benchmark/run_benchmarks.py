"""
Unified benchmark runner (no env vars, all hard-coded here).
- Iterates over a list of URDFs.
- For each URDF, writes benchlocal.py to hard-code parameters for all benchmarks.
- Runs all benchmark_* scripts, capturing ASCII logs with labels.
"""

import sys
import subprocess
from pathlib import Path

import torch

# ------------------------------
# User-configurable parameters
# ------------------------------

script_dir = Path(__file__).parent

# List of URDFs to run (fill this list)
URDF_LIST = [
    script_dir / "../examples/example_robots/go2_description/urdf/go2.urdf",
    # Add more URDFs here
]

# Benchmark knobs
BATCH_SIZES = [10, 100, 1000, 10000]
NUM_REPEATS = 100
WARMUP_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float64"  # or "float32"

# Output logs
OUT_DIR = script_dir / "bench_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark scripts to run (in order)
SCRIPTS = [
    "benchmark_acceleration.py",
    "benchmark_crba.py",
    "benchmark_jacobian.py",
    "benchmark_kinematics.py",
    "benchmark_rnea.py",
]

# ------------------------------
# Helpers
# ------------------------------


def write_benchlocal(bench_dir: Path, urdf_path: Path):
    """Create benchlocal.py beside benchconf.py to hard-code parameters."""
    code = [
        "from pathlib import Path",
        f"URDF_PATH = Path(r'{urdf_path.resolve()}')",
        f"BATCH_SIZES = {BATCH_SIZES}",
        f"NUM_REPEATS = {NUM_REPEATS}",
        f"WARMUP_ITERS = {WARMUP_ITERS}",
        f"DEVICE = '{DEVICE}'",
        f"DTYPE = '{DTYPE}'",
        "",
    ]
    (bench_dir / "benchlocal.py").write_text("\n".join(code))


def run_one_script(py_exe, script_path: Path, urdf_path: Path, log_file: Path):
    header = "=== SCRIPT={} ===\n".format(script_path.name)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(header)
        f.flush()
        cmd = [py_exe, "-u", str(script_path)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(script_path.parent),
        )
        for line in proc.stdout:
            # ensure ASCII; replace non-ascii with '?'
            try:
                text = line.decode("utf-8", errors="replace")
            except Exception:
                text = line.decode("latin-1", errors="replace")
            text_ascii = text.encode("ascii", errors="replace").decode("ascii")
            f.write(text_ascii)
        proc.wait()
        f.write("\n")


# ------------------------------
# Main
# ------------------------------


def main():
    py_exe = sys.executable
    scripts_dir = Path(__file__).parent
    benchconf_dir = scripts_dir  # benchconf.py is colocated

    for urdf in URDF_LIST:
        # 1) write benchlocal.py to hard-code parameters for this URDF
        write_benchlocal(benchconf_dir, urdf)

        # 2) run all scripts into a single per-URDF log
        tag = urdf.stem
        log_file = OUT_DIR / f"bench_{tag}.txt"
        with open(log_file, "w") as z:
            z.write("Benchmark log for URDF: {}\n\n".format(urdf))

        for s in SCRIPTS:
            sp = scripts_dir / s
            if not sp.exists():
                print("warning: script {} not found".format(sp))
                continue
            run_one_script(py_exe, sp, urdf, log_file)

        print("wrote {}".format(log_file))

        try:
            (benchconf_dir / "benchlocal.py").unlink(missing_ok=True)
            (benchconf_dir / "__pycache__" / "benchlocal.cpython-*.pyc").unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
