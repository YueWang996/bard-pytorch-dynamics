import os
import sys
import torch
from pathlib import Path


script_dir = Path(__file__).parent
URDF_PATH = script_dir / "../examples/example_robots/go2_description/urdf/go2.urdf"

BATCH_SIZES = [10, 100, 1000, 10000]
NUM_REPEATS = 100
WARMUP_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

print(f"Device: {DEVICE}, Dtype: {DTYPE}")
