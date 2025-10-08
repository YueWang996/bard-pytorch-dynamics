Installation
============

We strongly recommend using the **Conda** package manager to create an isolated environment. This simplifies the management of complex dependencies like PyTorch and Pinocchio (for testing).

First, clone the repository:

.. code-block:: bash

   git clone https://github.com/YueWang996/bard.git
   cd bard

Next, follow the instructions for your desired compute device.

CUDA Installation (Recommended for GPU Acceleration)
--------------------------------------------------

1. Install a CUDA-enabled build of PyTorch by following the official instructions for your specific platform and CUDA version:
   `https://pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_

2. Once PyTorch is installed, install `bard`:

   .. code-block:: bash

      pip install -e .

CPU-Only Installation
---------------------

If you do not have an NVIDIA GPU or do not require GPU acceleration, you can install the CPU-only version with a single command. This will automatically install a compatible, CPU-only version of PyTorch.

.. code-block:: bash

   pip install -e ".[cpu]"

Upgrading from CPU to GPU
-------------------------

If you initially installed the CPU-only version and want to switch to the CUDA version, you must manually reinstall PyTorch.

1. Uninstall the existing CPU-only PyTorch version:

   .. code-block:: bash

      pip uninstall -y torch torchvision torchaudio

2. Install the correct CUDA-enabled PyTorch build from the `official site <https://pytorch.org/get-started/locally/>`_.

3. Reinstall `bard` to ensure all dependencies are correctly linked.

   .. code-block:: bash

      pip install -e . --force-reinstall

Developer and Test Setup
------------------------

The library is rigorously tested against `pinocchio` to ensure numerical accuracy. To run the full test suite, you will need to install the development dependencies.

.. code-block:: bash

   # First, install pinocchio
   conda install -c conda-forge pinocchio

   # From the project root directory, install dev dependencies
   pip install -e ".[dev]"

   # Run the test suite
   pytest

Building the Documentation
--------------------------

If you need to build the documentation locally, follow these steps.

1. **Install documentation dependencies** from the project's root directory:

   .. code-block:: bash

      pip install -e .[docs]

2. **Navigate to the `docs` folder and run the build command**:

   .. code-block:: bash

      cd docs
      sphinx-build -b html source build/html

3. **View the documentation** by opening the `index.html` file located in the `docs/build/html/` directory in your web browser.