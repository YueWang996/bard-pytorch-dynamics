"""
Unified benchmark runner (no env vars, all hard-coded here).
- Iterates over a list of URDFs.
- For each URDF, writes benchlocal.py to hard-code parameters for all benchmarks.
- Runs all benchmark_* scripts, capturing ASCII logs with labels.
- Includes platform information at the start of each log.
"""

import sys
import subprocess
import platform
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
# Platform information collection
# ------------------------------


def get_platform_info():
    """Collect essential platform information."""
    info = []
    info.append("=" * 80)
    info.append("PLATFORM INFORMATION")
    info.append("=" * 80)
    info.append("")
    
    # Basic system info
    info.append(f"OS: {platform.system()} {platform.release()}")
    info.append(f"Architecture: {platform.machine()}")
    info.append(f"Python: {sys.version.split()[0]}")
    info.append(f"PyTorch: {torch.__version__}")
    info.append("")
    
    # Device and GPU info
    info.append(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        info.append(f"CUDA Version: {torch.version.cuda}")
        info.append("")
        info.append("GPU(s):")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            info.append(f"  [{i}] {props.name} - {mem_gb:.1f} GB - Compute {props.major}.{props.minor}")
    elif platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            info.append("Apple Metal (MPS): Available")
        else:
            info.append("Apple Metal (MPS): Not Available")
    else:
        info.append("Running on CPU (no GPU detected)")
    
    info.append("")
    
    # Benchmark config
    info.append(f"Batch Sizes: {BATCH_SIZES}")
    info.append(f"Repeats: {NUM_REPEATS}, Warmup: {WARMUP_ITERS}")
    info.append(f"Data Type: {DTYPE}")
    info.append("")
    
    info.append("=" * 80)
    info.append("")
    
    return "\n".join(info)


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
    header = "=" * 80 + "\n"
    header += f"SCRIPT: {script_path.name}\n"
    header += "=" * 80 + "\n"
    
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
        f.write("\n\n")


# ------------------------------
# Main
# ------------------------------


def main():
    py_exe = sys.executable
    scripts_dir = Path(__file__).parent
    benchconf_dir = scripts_dir  # benchconf.py is colocated
    
    # Collect platform info once
    platform_info = get_platform_info()
    
    # Print platform info to console
    print(platform_info)

    for urdf in URDF_LIST:
        # 1) write benchlocal.py to hard-code parameters for this URDF
        write_benchlocal(benchconf_dir, urdf)

        # 2) run all scripts into a single per-URDF log
        tag = urdf.stem
        log_file = OUT_DIR / f"bench_{tag}.txt"
        
        # Write header with platform info
        with open(log_file, "w") as f:
            f.write(platform_info)
            f.write(f"Benchmark for: {urdf.name}\n\n")

        for s in SCRIPTS:
            sp = scripts_dir / s
            if not sp.exists():
                print(f"Warning: script {sp} not found")
                continue
            
            print(f"Running {s}...")
            run_one_script(py_exe, sp, urdf, log_file)

        print(f"Wrote: {log_file}")

        # Cleanup
        try:
            (benchconf_dir / "benchlocal.py").unlink(missing_ok=True)
        except Exception:
            pass
        
        try:
            import glob
            for pyc in glob.glob(str(benchconf_dir / "__pycache__" / "benchlocal.cpython-*.pyc")):
                Path(pyc).unlink(missing_ok=True)
        except Exception:
            pass
    
    print("\nAll benchmarks complete!")


if __name__ == "__main__":
    main()
