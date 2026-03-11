#!/bin/bash
#
# SLURM batch script for bard speed benchmarks across multiple GPUs.
#
# Benchmarks five methods:
#   1. Pinocchio (C++)        — raw C++ calls, numpy I/O
#   2. Pinocchio (PyTorch)    — C++ calls with PyTorch tensor conversion
#   3. ADAM                   — adam-robotics PyTorch backend
#   4. bard                   — no torch.compile
#   5. bard (compiled)        — with torch.compile
#
# Usage:
#   sbatch benchmarks/slurm_benchmark.sh                  # submit all GPU jobs
#   sbatch --export=GPU_TYPE=a100 benchmarks/slurm_benchmark.sh  # single GPU type
#
# Run from the project root directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# GPU configurations: partition,time_limit,job_suffix
GPU_CONFIGS=(
    "a100,01:00:00,a100"
    "l4,02:00:00,l4"
    "l40,01:30:00,l40"
    "swarm_h100,01:00:00,h100"
    "quad_h200:dual_h200,01:00:00,h200"
)

# If GPU_TYPE is set, only run that one
if [ -n "${GPU_TYPE:-}" ]; then
    FILTERED=()
    for cfg in "${GPU_CONFIGS[@]}"; do
        suffix="${cfg##*,}"
        if [ "$suffix" = "$GPU_TYPE" ]; then
            FILTERED+=("$cfg")
        fi
    done
    if [ ${#FILTERED[@]} -eq 0 ]; then
        echo "ERROR: Unknown GPU_TYPE=$GPU_TYPE"
        echo "Available: a100, l4, l40, h100, h200"
        exit 1
    fi
    GPU_CONFIGS=("${FILTERED[@]}")
fi

echo "Submitting bard benchmark jobs from: $PROJECT_DIR"
echo ""

for cfg in "${GPU_CONFIGS[@]}"; do
    IFS=',' read -r partitions time_limit suffix <<< "$cfg"
    # Convert partition1:partition2 to partition1,partition2 for SLURM
    slurm_partitions="${partitions//:/$','}"

    JOB_NAME="bard_bench_${suffix}"
    LOG_FILE="${PROJECT_DIR}/benchmarks/results/bench_${suffix}_%j.log"

    echo "Submitting: $JOB_NAME (partition=$slurm_partitions, time=$time_limit)"

    sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${slurm_partitions}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=${time_limit}
#SBATCH --gres=gpu:1
#SBATCH --output=${LOG_FILE}

echo "=========================================================="
echo "Job: \$SLURM_JOB_ID on \$(hostname)"
echo "GPU type: ${suffix}"
echo "Started: \$(date)"
echo "=========================================================="

nvidia-smi
echo ""

# Pin to 6 CPU cores for reproducibility across GPU types
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate conda environment
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate bard

cd ${PROJECT_DIR}

# Run speed benchmark (all 5 methods: Pin C++, Pin PyTorch, ADAM, bard, bard compiled)
python benchmarks/speed_benchmark.py \
    --device cuda \
    --dtype float64 \
    --save \
    --n-repeats 100

echo ""
echo "=========================================================="
echo "Finished: \$(date)"
echo "=========================================================="
SBATCH_EOF

done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
