# Bard: Batched Articulated Robot Dynamics

Efficient robot kinematics and dynamics in PyTorch, designed for batch processing and differentiability.

`bard` is a lightweight, PyTorch-native library for rigid-body dynamics that leverages tensor operations to perform batched computations on the CPU or GPU. It's an ideal tool for robotics research, especially in areas like reinforcement learning, trajectory optimization, and model-based control, where performing parallel computations over many robot states is critical.

-----

## Key Features ✨

  * **PyTorch Native**: Built entirely on PyTorch for seamless integration with modern machine learning pipelines.
  * **Batch Processing**: All core functions operate on batches of robot states, enabling massive parallelism.
  * **GPU Acceleration**: Run your dynamics computations on NVIDIA GPUs for significant speedups.
  * **Differentiable**: The entire computation graph is differentiable, allowing for gradient-based optimization through the dynamics.
  * **Comprehensive Dynamics**:
      * Forward Kinematics
      * Jacobian Calculation (in world and local frames)
      * Inverse Dynamics (RNEA)
      * Mass Matrix / Inertia Matrix (CRBA)
  * **Floating-Base Support**: Natively handles both fixed-base manipulators and floating-base systems like humanoids or quadrupeds.
  * **URDF Parsing**: Load robots directly from URDF files.

-----

## Installation

You can install `bard` directly from PyPI.

```bash
pip install bard
```

For development, clone this repository and install in editable mode with all development dependencies (including `pytest` for testing and `pinocchio` for baseline comparisons):

```bash
git clone https://github.com/yourusername/bard.git
cd bard
pip install -e .
```

-----

## Quick Start

Here is a simple example of performing batched forward kinematics and Jacobian calculations for a 2-link robot.

```python
import torch
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.forward_kinematics import calc_forward_kinematics
from bard.core.jacobian import calc_jacobian

# A simple 2-link robot URDF
urdf_string = """
<robot name="simple_robot">
    <link name="link1"/>
    <link name="link2"/>
    <link name="link3"/>
    <joint name="joint1" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 1"/>
        <axis xyz="0 0 1"/>
    </joint>
    <joint name="joint2" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 1"/>
        <axis xyz="0 1 0"/>
    </joint>
</robot>
"""

# 1. Build the kinematic chain
chain = build_chain_from_urdf(urdf_string).to(dtype=torch.float64, device="cpu")

# 2. Define a batch of joint configurations (N=100)
N = 100
q = torch.rand(N, chain.n_joints, dtype=torch.float64) * torch.pi

# 3. Get the index of the end-effector frame
ee_frame = "link3"
ee_idx = chain.get_frame_indices(ee_frame).item()

# 4. Perform batched forward kinematics
# Returns a Transform3d object containing the results
transforms = calc_forward_kinematics(chain, q, ee_idx)
ee_positions = transforms.get_matrix()[:, :3, 3] # Extract XYZ positions

# 5. Perform batched Jacobian calculation
J = calc_jacobian(chain, q, ee_idx, reference_frame="world")

# Print the shapes to see the batched output
print(f"Batch size: {N}")
print(f"End-effector position shape: {ee_positions.shape}") # torch.Size([100, 3])
print(f"Jacobian shape: {J.shape}")                       # torch.Size([100, 6, 2])
```

-----

## API Overview

The core API is simple and functional.

  * `build_chain_from_urdf(urdf_data, floating_base=False)`: Parses a URDF file string and returns a `Chain` object.
  * `calc_forward_kinematics(chain, q, frame_id)`: Computes the world pose of a specific frame.
  * `calc_jacobian(chain, q, frame_id, reference_frame="world")`: Computes the Jacobian for a specific frame.
  * `calc_inverse_dynamics(chain, q, qd, qdd, gravity=...)`: Computes joint forces/torques via RNEA.
  * `crba_inertia_matrix(chain, q)`: Computes the joint-space inertia matrix (mass matrix).
  * `end_effector_acceleration(chain, q, qd, qdd, frame_id, ...)`: Computes the spatial acceleration of a frame.

-----

## Running Tests

The library is rigorously tested against the `pinocchio` library. To run the tests, first install the development dependencies, then run `pytest`.

```bash
# From the project root directory
pip install -e .[dev]
pytest
```

-----

## License

This project is licensed under the MIT License.