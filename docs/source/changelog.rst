Changelog
=========

v0.4.3
------

**Performance: torch.compile for all algorithms + cleanup**

* Enabled ``torch.compile`` for CRBA and ABA on CUDA (previously skipped when
  Triton kernels were available). Compilation now optimizes surrounding PyTorch
  operations while Triton handles the bottleneck 6x6 double-matmul. Result:
  CRBA 2.1x faster, ABA 1.7x faster at B=1 with compile enabled.

* Moved Triton kernel import to module level, eliminating repeated import
  overhead in hot loops. ``_use_triton_kernels`` now properly gates on
  ``HAS_TRITON`` instead of being hardcoded to ``True``.

* Rewrote ``benchmarks/speed_benchmark.py`` for fair comparison: each bard
  algorithm timing now includes ``update_kinematics`` (end-to-end), matching
  ADAM's self-contained calls. Speedup tables use ADAM as the primary baseline.
  Pinocchio C++ is included as a serial CPU reference.

* Added autograd documentation to Quick Start guide with examples for
  ``d(M)/d(q)`` through CRBA, ``d(qdd)/d(tau)`` through ABA.

* Removed obsolete files: ``quick_bench.py``, ``basic_test.py``,
  ``bard_basic_test.py``, and root-level profiling/debug scripts.

v0.4.2
------

**Performance: Inline spatial cross products, eliminate per-node allocations**

* Inline spatial cross products to eliminate per-node GPU allocations.
* Revert JIT functions to ``torch.zeros`` pattern for ``torch.compile``
  compatibility.
* Eliminate hidden tensor allocations in core transform functions.

v0.3
----

**Unified Model/Data API**

* Introduced ``Model`` and ``Data`` classes following the Pinocchio/MuJoCo
  pattern. All computations are now accessed through top-level free functions
  (``bard.forward_kinematics()``, ``bard.jacobian()``, ``bard.rnea()``, etc.)
  operating on a ``model`` + ``data`` pair.

* Added ``bard.build_model_from_urdf()`` as the primary entry point for loading
  robots. Returns a ``Model`` object that holds the robot's topology, inertias,
  and joint parameters.

* Added ``bard.create_data()`` for creating pre-allocated computation workspaces.
  One ``Model`` can be used with multiple ``Data`` instances.

* ``bard.update_kinematics(model, data, q, qd)`` performs a single tree traversal,
  caching transforms, spatial adjoints, and velocities. All subsequent algorithm
  calls reuse the cached data with zero redundant computation.

* **Breaking change:** Removed ``RobotDynamics``, ``KinematicsState``,
  ``ForwardKinematics``, ``SpatialAcceleration``, ``Jacobian``, ``RNEA``,
  and ``CRBA`` classes. The ``build_chain_from_urdf()`` function is no longer
  part of the public API. See the Quick Start guide for migration examples.

v0.2
----

* Benchmarks and performance documentation.
* Fixed ``lxml`` dependency version.
* Removed Python 3.14 support from CI.

v0.1
----

* Initial release with batched Forward Kinematics, Jacobian, RNEA, CRBA,
  and Spatial Acceleration.
* URDF parsing support.
* ``torch.compile`` compatibility.
* Fixed-base and floating-base robot support.
