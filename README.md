# bard: Batched Articulated Robot Dynamics

[![CI](https://github.com/YueWang996/bard/actions/workflows/ci.yml/badge.svg)](https://github.com/YueWang996/bard/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17291122.svg)](https://doi.org/10.5281/zenodo.17291122)


`bard` is a lightweight, PyTorch-native library for rigid-body dynamics that leverages tensor operations to perform efficient, batched computations on either the CPU or GPU. It provides a simple yet powerful API for loading robots from URDF files and analyzing their motion using standard robotics algorithms.

The primary motivation behind `bard` is to provide a dynamics library that integrates seamlessly into modern machine learning workflows. By treating the robot's state and dynamics as a differentiable computation graph, it becomes an ideal tool for robotics research in areas like reinforcement learning, trajectory optimization, physics-informed learning, and system identification.

## Key Features ✨

  * **PyTorch Native**: Built entirely on PyTorch for seamless integration with ML pipelines.
  * **Batch Processing**: All core functions operate on batches of robot states, enabling massive parallelism.
  * **GPU Acceleration**: Run dynamics computations on NVIDIA GPUs for significant speedups.
  * **Differentiable**: The entire computation graph is differentiable, allowing for gradient-based optimization through the robot's dynamics.
  * **Comprehensive Algorithms**:
      * Forward Kinematics
      * Jacobian Calculation (in world and body frames)
      * Inverse Dynamics (using RNEA)
      * Mass Matrix / Inertia Matrix (using CRBA)
  * **Floating-Base Support**: Natively handles both fixed-base manipulators and floating-base systems like humanoids or quadrupeds.
  * **URDF Parsing**: Load robot models directly from URDF files.

## Installation

We strongly recommend using the **Conda** package manager to create an isolated environment. This simplifies the management of complex dependencies like PyTorch and Pinocchio (for testing).

First, clone the repository:

```bash
# create a conda environment
conda create -n yourEnv

# clone the repo
git clone https://github.com/YueWang996/bard.git
cd bard
```

Next, follow the instructions for your desired compute device.

### Option 1: CUDA Installation (Recommended for GPU Acceleration)

1.  Install a CUDA-enabled build of PyTorch by following the official instructions for your specific platform and CUDA version:
    [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    > **Note:** `bard` has been tested against `PyTorch 2.8.0` and `CUDA 12.6`. If you encounter any errors, we recommend switching to these package versions to resolve potential issues.

2.  Once PyTorch is installed, install `bard`:

    ```bash
    pip install -e .
    ```

### Option 2: CPU-Only Installation

If you do not have an NVIDIA GPU or do not require GPU acceleration, you can install the CPU-only version with a single command. This will automatically install a compatible, CPU-only version of PyTorch.

```bash
pip install -e ".[cpu]"
```

### Upgrading from CPU to GPU

If you initially installed the CPU-only version and want to switch to the CUDA version, you must manually reinstall PyTorch.

1.  Uninstall the existing CPU-only PyTorch version:

    ```bash
    pip uninstall -y torch torchvision
    ```

2.  Install the correct CUDA-enabled PyTorch build from the [official site](https://pytorch.org/get-started/locally/).

3.  Reinstall `bard` to ensure all dependencies are correctly linked.

    ```bash
    pip install -e . --force-reinstall
    ```

### Building the Documentation

The documentation is available [here](https://yuewang996.github.io/bard/). If you need to build the documentation locally, follow these steps.

1.  **Install documentation dependencies** from the project's root directory:
    ```bash
    pip install -e .[docs]
    ```
2.  **Navigate to the `docs` folder and run the build command**:
    ```bash
    cd docs
    sphinx-build -b html source build/html
    ```
3.  **View the documentation** by opening the `index.html` file located in the `docs/build/html/` directory in your web browser.

## Running Tests

The library is rigorously tested against `pinocchio` to ensure numerical accuracy. To run the full test suite, you will need to install the development dependencies, including `pinocchio`, which is best installed from `conda-forge`.

```bash
# First, install pinocchio
conda install -c conda-forge pinocchio

# From the project root directory, install dev dependencies
pip install -e .[dev]

# Run the test suite
pytest
```

## Citing bard

If you use `bard` in your research, please consider citing it:

```bibtex
@software{wang_2025_bard,
  author       = {Wang, Yue},
  title        = {{bard: Batched Articulated Robot Dynamics}},
  month        = oct,
  year         = {2025},
  doi          = {10.5281/zenodo.17291122},
  url          = {https://github.com/YueWang996/bard}
}
```


## Acknowledgements

This library builds upon the excellent work of several other open-source projects.

  * The core `bard/transforms` module is adapted from **`pytorch3d`**. This approach was chosen to avoid including the entirety of `pytorch3d` as a dependency. An important difference is that `bard` uses left-multiplied transforms (`T * pt`), which is the standard convention in robotics, as opposed to `pytorch3d`'s right-multiplied convention.
  * The `bard/parsers/urdf_parser_py` module is extracted from **`kinpy`**.
  * This project is heavily inspired by the structure and API of **`pytorch_kinematics`**, from which some of the components were adapted.
  * Numerical results are validated against **`pinocchio`**.

## License

This project is licensed under the MIT License.
