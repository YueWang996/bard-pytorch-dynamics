Changelog
=========

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
